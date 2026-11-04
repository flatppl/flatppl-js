// @flatppl/web — a playground panel in its own window (pane.html).
//
// The playground window (pane-owner.ts) opened this window for one panel:
// the source editor, graph, plots or math. It renders that panel from the
// state the owner sends, and turns its own gestures into requests to the
// owner, which keeps the model, the router, the drafts and the stores.
// Messages travel through pane-session.ts.
//
// The source window is where the user types while the editor is away: each
// edit goes back to the owner, which runs it through the same change handler
// as its own editor. If the owner reloads, the text typed here is kept and
// sent back as a draft; if the owner closes, it can be downloaded.
//
// Loaded as a plain <script> after the viewer bundle, the editor and
// pane-session.js.

'use strict';

(function (globalScope: any) {
  const PANES: Record<string, string> = { source: 'Source', graph: 'Graph', plot: 'Plots', math: 'Math' };
  // An edit is sent at most this often; the last one always goes.
  const EDIT_FLUSH_MS = 16;
  // The lint waits this long for the owner to resolve the bundle.
  const RPC_TIMEOUT_MS = 10000;
  const HELLO_RETRY_MS = 1000;
  const HELLO_GIVE_UP_MS = 10000;
  const OWNER_RETURN_MS = 3000;
  // After the closed banner shows, a main window that comes back late (a
  // slow reload) can still find this window on the postMessage fallback.
  const SLOW_HELLO_MS = 5000;
  // The main window's goodbye can be lost when it closes (Chromium may tear
  // the page down before pagehide's message leaves), so also watch its window.
  const OPENER_POLL_MS = 1000;
  const PING_MS = 5000;
  const LOST_MS = 15000;

  const doc = globalScope.document;
  const $ = (id: string) => doc.getElementById(id);

  function boot(): void {
    const pane = new URLSearchParams(globalScope.location.search).get('pane') || '';
    const sid = new URLSearchParams(globalScope.location.hash.slice(1)).get('sid') || '';
    const label = PANES[pane];
    const status = $('pane-status');
    const banner = $('pane-banner');
    const bannerText = $('pane-banner-text');
    const heading = $('pane-title');
    const root = $('surface-root');
    function say(text: string): void { status.textContent = text; }
    function showBanner(text: string, action?: { label: string; run: () => void }, download?: () => void): void {
      bannerText.textContent = text;
      const button = $('pane-banner-action');
      button.hidden = !action;
      if (action) { button.textContent = action.label; button.onclick = action.run; }
      const save = $('pane-banner-download');
      save.hidden = !download;
      save.onclick = download || null;
      banner.hidden = false;
    }

    if (!label || !/^[0-9a-f]{32}$/.test(sid)) {
      heading.textContent = 'FlatPPL panel';
      showBanner('This window was not opened from the playground. Open the playground and use Pop out.',
        { label: 'Open playground here', run: () => { globalScope.location.href = 'index.html'; } });
      return;
    }
    heading.textContent = label;
    doc.title = label + ' — FlatPPL';

    const S = globalScope.FlatPPLPaneSession;
    const tracker = S.createRevisionTracker();
    let welcomed = false;
    let leaving = false;
    let applying = false;
    let lastModel: any = null;
    let lastFocus: string | null = null;
    let lastSelect: string | null = null;
    let settings: any = { editEnabled: false };
    let parkedMeasures: any = null;
    let helloTimer: any = null;
    let slowHelloTimer: any = null;
    let ownerTimer: any = null;
    let adoptedEpoch: string | null = null;

    const host = {
      setTarget(name: string | null): void {
        if (applying || (name || null) === lastFocus) return;
        lastFocus = name || null;
        session.send('intent.focus', { target: lastFocus }, 'owner');
      },
      revealSourceLine(line: number, _name: string, o?: { preserveFocus?: boolean }): void {
        session.send('intent.reveal', { line, silent: !!(o && o.preserveFocus) }, 'owner');
      },
      openModule(path: string, member?: string): void {
        session.send('intent.navigate', { model: path, target: member || null }, 'owner');
      },
      onSelect(name: string | null): void {
        lastSelect = name;
        session.send('intent.select', { binding: name }, 'owner');
      },
      setTitle(name: string): void { retitle(name); },
      canPersist(): boolean { return !!settings.editEnabled; },
      editSource(args: any): Promise<boolean> {
        session.send('intent.replaceRange', { range: args.range || null, newText: args.newText }, 'owner');
        return Promise.resolve(true);
      },
      promptForName(args: any): Promise<string | null> {
        const FE = globalScope.FlatPPLEngine;
        const existing = new Set(args.existingNames || []);
        let current = args.suggested || 'preset';
        for (;;) {
          const raw = globalScope.prompt('Save current values as a new preset. Name:', current);
          if (raw == null) return Promise.resolve(null);
          const name = raw.trim();
          if (!name || (FE && FE.isValidBindingName && !FE.isValidBindingName(name)) || existing.has(name)) {
            current = raw;
            continue;
          }
          return Promise.resolve(name);
        }
      },
    };

    const isSource = pane === 'source';
    const view = isSource ? null : globalScope.FlatPPLViewer.mount(root, { host, panels: [pane] });

    // ---- the source editor (pane=source) ----
    let editor: any = null;
    let sourcePath: string | null = null;
    let sourceLanguage: string | null = null;
    let pendingEdit: { text: string; path: string | null } | null = null;
    let editTimer: any = null;
    // Text typed here when the owner unloads: kept until the reloaded owner
    // shows the same file in edit mode, then sent back as a draft.
    let pendingDraft: { path: string | null; text: string } | null = null;
    // Revisions (see remote-editor.ts): the owner's epoch, the last revision
    // sent, and each change made here since the owner's last ack, tagged with
    // the revision that sends it.
    let srcEpoch: string | null = null;
    let srcRev = 0;
    let changeLog: { rev: number; changes: any }[] = [];
    let rpcId = 0;
    const rpcs = new Map<number, { resolve: (v: any) => void; reject: (e: any) => void; timer: any }>();

    function flushEdit(): void {
      if (editTimer) { globalScope.clearTimeout(editTimer); editTimer = null; }
      if (!pendingEdit) return;
      const edit = pendingEdit;
      pendingEdit = null;
      srcRev++;
      session.send('intent.edit', { ...edit, epoch: srcEpoch, textRev: srcRev }, 'owner');
    }

    function dropPendingEdit(): void {
      pendingEdit = null;
      if (editTimer) { globalScope.clearTimeout(editTimer); editTimer = null; }
    }

    /** Take the owner's text as the base of its epoch; what was typed
     *  against an older epoch is gone. */
    function adoptEpoch(st: any): void {
      srcEpoch = st.epoch;
      srcRev = st.textRev | 0;
      changeLog = [];
      dropPendingEdit();
    }

    /** Apply an owner edit whose range is relative to revision `textRev`. */
    function applyReplace(body: any): void {
      if (body.epoch !== srcEpoch) { console.warn('FlatPPL: source edit for another text dropped'); return; }
      // Text typed at either end of the range stays outside it.
      const insert = body.from === body.to;
      let from = body.from, to = body.to;
      for (const entry of changeLog) {
        if (entry.rev <= body.textRev) continue;
        from = entry.changes.mapPos(from, 1);
        to = entry.changes.mapPos(to, insert ? 1 : -1);
      }
      editor.replaceRange(from, Math.max(from, to), body.text);
    }

    function rpcResolve(text: string): Promise<any> {
      const id = ++rpcId;
      return new Promise((resolve, reject) => {
        // Without an answer the lint falls back to the standalone parse.
        const timer = globalScope.setTimeout(() => { rpcs.delete(id); resolve(null); }, RPC_TIMEOUT_MS);
        rpcs.set(id, { resolve, reject, timer });
        session.send('rpc.resolve', { id, text }, 'owner');
      });
    }

    if (isSource) {
      // The same id as the playground's editor, so its stylesheet applies.
      const editorHost = doc.createElement('div');
      editorHost.id = 'source-editor';
      root.appendChild(editorHost);
      editor = globalScope.FlatPPLWebEditor.mountEditor(editorHost, {
        initialSource: '',
        initialReadOnly: true,
        onChange(text: string, changes: any) {
          changeLog.push({ rev: srcRev + 1, changes });
          pendingEdit = { text, path: sourcePath };
          if (!editTimer) editTimer = globalScope.setTimeout(flushEdit, EDIT_FLUSH_MS);
        },
        onSave() { flushEdit(); session.send('intent.save', {}, 'owner'); },
        lintBundle: rpcResolve,
        onNavigate(name: string) {
          lastFocus = name || null;
          session.send('intent.focus', { target: lastFocus }, 'owner');
        },
      });
      $('pane-source-actions').hidden = false;
      $('pane-edit').addEventListener('click', () => {
        session.send('intent.setEdit', { on: !settings.editEnabled }, 'owner');
      });
      $('pane-save').addEventListener('click', () => { flushEdit(); session.send('intent.save', {}, 'owner'); });
    }

    function updateSourceActions(): void {
      if (!isSource) return;
      $('pane-edit').setAttribute('aria-pressed', settings.editEnabled ? 'true' : 'false');
      $('pane-save').hidden = !settings.editEnabled;
    }

    /** Show what the owner's editor holds, unless this window keeps typed text
     *  across an owner reload (see pendingDraft). */
    function applySourceState(st: any): void {
      if (!editor || !st) return;
      if (pendingDraft && st.path === pendingDraft.path) {
        // Still loading, or in view mode: keep showing what was typed here.
        if (st.readOnly) { editor.setReadOnly(true); return; }
        const own = pendingDraft.text;
        pendingDraft = null;
        setLanguage(st.language);
        editor.setReadOnly(false);
        if (editor.getSource() !== own) editor.setSource(own);
        sourcePath = st.path;
        adoptEpoch(st);
        if (st.text !== own) {
          // The draft is one edit on top of the owner's text.
          const base = globalScope.FlatPPLEditorBundle.EditorState.create({ doc: st.text });
          changeLog.push({ rev: srcRev + 1, changes: base.changes({ from: 0, to: st.text.length, insert: own }) });
          pendingEdit = { text: own, path: st.path };
          flushEdit();
        }
        return;
      }
      if (pendingDraft && st.path && st.path !== pendingDraft.path) pendingDraft = null;
      sourcePath = st.path ?? sourcePath;
      // Within an epoch this window's text is newer than the owner's mirror.
      if (st.epoch !== srcEpoch) {
        adoptEpoch(st);
        if (st.text !== editor.getSource()) editor.setSource(st.text);
      }
      setLanguage(st.language);
      editor.setReadOnly(!!st.readOnly);
      retitle(null);
    }

    function setLanguage(language: string | null | undefined): void {
      if (!language || language === sourceLanguage) return;
      sourceLanguage = language;
      editor.setLanguage(language);
    }

    function sourceSnapshot(): any {
      return { source: { text: editor.getSource(), view: editor.getViewState(), path: sourcePath, epoch: srcEpoch, textRev: srcRev } };
    }

    function downloadSource(): void {
      const name = (sourcePath || 'model.flatppl').split('/').pop() || 'model.flatppl';
      const url = globalScope.URL.createObjectURL(new globalScope.Blob([editor.getSource()], { type: 'text/plain' }));
      const a = doc.createElement('a');
      a.href = url;
      a.download = name;
      doc.body.appendChild(a);
      a.click();
      a.remove();
      globalScope.setTimeout(() => globalScope.URL.revokeObjectURL(url), 1000);
    }

    function retitle(binding?: string | null): void {
      const path = isSource ? sourcePath : lastModel && lastModel.path;
      const model = path ? path.split('/').pop() : null;
      doc.title = [label, model && (model + (binding ? ' / ' + binding : ''))].filter(Boolean).join(' — ') + ' — FlatPPL';
    }

    function bundleOptions(model: any, extra?: any): any {
      const b = model.bundle || {};
      return { ...(extra || {}), variant: model.variant || undefined, path: b.path || undefined,
        bundleSources: b.sources || undefined, bundleFiles: b.files || undefined,
        bundleResolutions: b.resolutions || undefined };
    }

    /** Apply a model and focus from the owner without echoing them back. */
    function applyModel(model: any, focus: string | null, pushHistory = false): void {
      lastModel = model;
      lastFocus = focus;
      // The source window follows the owner's editor (state.source), not the model.
      if (isSource) return;
      if (!model || model.type !== 'flatppl' || model.preview) {
        say(model && model.preview
          ? 'Run the shared preview in the main window to see it here.'
          : 'Open a FlatPPL model in the main window to see it here.');
        return;
      }
      applying = true;
      try { view.update(model.source, focus, bundleOptions(model, { pushHistory })); } finally { applying = false; }
      retitle(focus);
    }

    function applySettings(next: any): void {
      settings = next || settings;
      if (settings.theme) doc.documentElement.dataset.theme = settings.theme;
      updateSourceActions();
    }

    function welcome(env: any): void {
      // Two hellos in flight bring two welcomes from one owner page load.
      // Only the first applies: a second would re-import (and so drop) the
      // samples parked for the first.
      if (welcomed && env.epoch === adoptedEpoch) return;
      adoptedEpoch = env.epoch;
      const body = env.body || {};
      const snap = body.snapshot || {};
      tracker.reset();
      tracker.accept({ epoch: env.epoch, body: { rev: snap.rev || 0 } });
      applySettings(snap.settings);
      if (isSource) {
        lastModel = snap.model || lastModel;
        lastFocus = snap.focus ?? null;
        const st = body.panelState && body.panelState.source;
        applySourceState(st);
        if (st && st.view) editor.setViewState(st.view);
      }
      // Plot settings apply before the model renders, so parked samples match.
      if (pane === 'plot') {
        view.importMeasures(parkedMeasures);
        parkedMeasures = null;
        if (body.panelState) view.restorePanel('plot', body.panelState);
      }
      if (snap.model && !isSource) applyModel(snap.model, snap.focus ?? null);
      if (!isSource && pane !== 'plot' && body.panelState && snap.model) {
        try { view.restorePanel(pane, body.panelState); } catch (err) { console.warn('FlatPPL: panel state not restored', err); }
      }
      if (snap.select && view) { lastSelect = snap.select; view.select(snap.select); }
      welcomed = true;
      banner.hidden = true;
      say('Connected to the main window.');
      if (helloTimer) { globalScope.clearInterval(helloTimer); helloTimer = null; }
      stopSlowHello();
      session.watch('owner', PING_MS, LOST_MS);
    }

    function onMessage(env: any): void {
      const body = env.body || {};
      if (env.type === 'owner.hello') { ownerReturned(); sayHello(); return; }
      if (env.type === 'pane.welcome') { ownerReturned(); welcome(env); return; }
      if (env.type === 'pane.measures') {
        if (!view) return;
        if (welcomed) view.importMeasures(body); else parkedMeasures = body;
        return;
      }
      if (env.type === 'rpc.result') {
        const call = rpcs.get(body.id);
        if (!call) return;
        rpcs.delete(body.id);
        globalScope.clearTimeout(call.timer);
        if (body.ok) call.resolve(body.value); else call.reject(new Error(body.error || 'bundle resolution failed'));
        return;
      }
      if (env.type === 'pane.redock') { leave(); return; }
      if (env.type === 'pane.bye') { ownerLeft(); return; }
      if (!env.type.startsWith('state.') || !welcomed) return;
      const verdict = tracker.accept(env);
      if (verdict === 'stale') return;
      if (verdict === 'new-epoch') { sayHello(); return; }
      switch (env.type) {
        case 'state.model': applyModel(body.model, body.focus ?? null); lastSelect = null; return;
        case 'state.focus':
          // The owner echoes a focus this window set itself (a double-click
          // here). Re-rendering would re-fit the graph over the user's zoom
          // and pan, and nothing changed.
          if ((body.target ?? null) === lastFocus) return;
          if (lastModel) applyModel(lastModel, body.target ?? null, !!body.pushHistory);
          lastSelect = null;
          return;
        case 'state.select':
          if (view && (body.binding ?? null) !== lastSelect) { lastSelect = body.binding ?? null; view.select(lastSelect); }
          return;
        case 'state.settings': applySettings(body); return;
        case 'state.source': applySourceState(body); return;
        case 'state.sourceReveal': if (editor) editor.revealLine(body.line, { silent: !!body.silent }); return;
        case 'state.sourceReplace': if (editor) applyReplace(body); return;
        case 'state.sourceAck':
          if (body.epoch === srcEpoch) changeLog = changeLog.filter((entry) => entry.rev > body.textRev);
          return;
        default: return;
      }
    }

    const session = S.createSession({
      sid, id: 'pane:' + pane,
      transport: S.defaultTransport(sid, () => (globalScope.opener ? [globalScope.opener] : [])),
      onMessage,
      onVersion() {
        showBanner('This window is from a different playground version. Reload it.',
          { label: 'Reload', run: () => globalScope.location.reload() });
      },
      onLost() { ownerGone(); },
    });

    function sayHello(): void {
      session.send('pane.hello', { pane }, 'owner');
    }

    /** Send this panel back: samples first, then the panel state. */
    function leave(): void {
      if (leaving) return;
      leaving = true;
      stopSlowHello();
      if (isSource) flushEdit();
      if (pane === 'plot') session.send('pane.measures', view.exportMeasures(null), 'owner');
      session.send('pane.panelState', { pane, state: isSource ? sourceSnapshot() : view.snapshotPanel(pane) }, 'owner');
      globalScope.close();
      // A window the user opened by hand cannot close itself.
      globalScope.setTimeout(() => showBanner(label + ' is back in the main window. You can close this window.'), 500);
    }

    /** The main window unloaded: it may be reloading. Say hello until it
     *  welcomes this window again; without BroadcastChannel the reloaded
     *  owner learns of this window only from such a hello. */
    function ownerLeft(): void {
      say('Reconnecting to the main window…');
      welcomed = false;
      // Keep what was typed here; the reloaded owner gets it back as a draft.
      if (editor && settings.editEnabled && !pendingDraft) {
        flushEdit();
        pendingDraft = { path: sourcePath, text: editor.getSource() };
      }
      if (ownerTimer) globalScope.clearTimeout(ownerTimer);
      ownerTimer = globalScope.setTimeout(ownerGone, OWNER_RETURN_MS);
      // The first hello waits one retry, so it reaches the next page load,
      // not the one that just said goodbye.
      startHello(OWNER_RETURN_MS, HELLO_RETRY_MS);
    }

    function ownerReturned(): void {
      if (ownerTimer) { globalScope.clearTimeout(ownerTimer); ownerTimer = null; }
    }

    /** Say hello every second until welcomed; give up after `limit` ms.
     *  `delay` holds back the first hello. */
    function startHello(limit: number, delay = 0): void {
      if (helloTimer) globalScope.clearInterval(helloTimer);
      const started = Date.now();
      if (!delay) sayHello();
      helloTimer = globalScope.setInterval(() => {
        if (welcomed) { globalScope.clearInterval(helloTimer); helloTimer = null; return; }
        if (Date.now() - started > limit) {
          globalScope.clearInterval(helloTimer);
          helloTimer = null;
          ownerGone();
          return;
        }
        sayHello();
      }, HELLO_RETRY_MS);
    }

    function ownerGone(): void {
      if (leaving) return;
      if (helloTimer) { globalScope.clearInterval(helloTimer); helloTimer = null; }
      if (ownerTimer) { globalScope.clearTimeout(ownerTimer); ownerTimer = null; }
      session.unwatch('owner');
      const path = lastModel && lastModel.path;
      const target = lastFocus ? '?target=' + encodeURIComponent(lastFocus) : '';
      showBanner('The main playground window is closed.', {
        label: 'Open playground here',
        run: () => { globalScope.location.href = 'index.html' + (path ? '#/files/' + encodeURI(path) + target : ''); },
      }, isSource ? downloadSource : undefined);
      say('The main playground window is closed.');
      stopSlowHello();
      slowHelloTimer = globalScope.setInterval(sayHello, SLOW_HELLO_MS);
    }

    function stopSlowHello(): void {
      if (slowHelloTimer) { globalScope.clearInterval(slowHelloTimer); slowHelloTimer = null; }
    }

    $('pane-dock').addEventListener('click', () => session.send('intent.redock', {}, 'owner'));
    $('pane-main').addEventListener('click', () => {
      try { if (globalScope.opener) globalScope.opener.focus(); } catch (_) { /* the browser decides */ }
    });
    globalScope.addEventListener('pagehide', () => {
      stopSlowHello();
      if (leaving) return;
      if (isSource) flushEdit();
      if (pane === 'plot') session.send('pane.measures', view.exportMeasures(null), 'owner');
      session.send('pane.bye', { reason: 'close', state: isSource ? sourceSnapshot() : view.snapshotPanel(pane) }, 'owner');
    });

    // Chromium sets window.opener to null once the opener closes, so keep the
    // reference taken at boot: its `closed` stays readable.
    const openerAtBoot = globalScope.opener;
    if (openerAtBoot) {
      globalScope.setInterval(() => {
        let closed = false;
        try { closed = !!openerAtBoot.closed || !globalScope.opener; } catch (_) { closed = false; }
        if (closed && !leaving && banner.hidden) ownerGone();
      }, OPENER_POLL_MS);
    }

    say('Connecting to the main window…');
    startHello(HELLO_GIVE_UP_MS);
    heading.focus();

    globalScope.FlatPPLPane = { pane, view, editor, session };
  }

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', boot);
  else boot();
})(typeof window !== 'undefined' ? window : globalThis);
