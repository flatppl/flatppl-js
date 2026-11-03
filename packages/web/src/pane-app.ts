// @flatppl/web — a playground panel in its own window (pane.html).
//
// The playground window (pane-owner.ts) opened this window for one panel:
// graph, plots or math. It renders that panel from the state the owner
// sends, and turns its own gestures into requests to the owner, which keeps
// the model, the router and the editor. Messages travel through
// pane-session.ts.
//
// Loaded as a plain <script> after the viewer bundle and pane-session.js.

'use strict';

(function (globalScope: any) {
  const PANES: Record<string, string> = { graph: 'Graph', plot: 'Plots', math: 'Math' };
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
    function showBanner(text: string, action?: { label: string; run: () => void }): void {
      bannerText.textContent = text;
      const button = $('pane-banner-action');
      button.hidden = !action;
      if (action) { button.textContent = action.label; button.onclick = action.run; }
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

    const view = globalScope.FlatPPLViewer.mount(root, { host, panels: [pane] });

    function retitle(binding?: string | null): void {
      const model = lastModel && lastModel.path ? lastModel.path.split('/').pop() : null;
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
      // Plot settings apply before the model renders, so parked samples match.
      if (pane === 'plot') {
        view.importMeasures(parkedMeasures);
        parkedMeasures = null;
        if (body.panelState) view.restorePanel('plot', body.panelState);
      }
      if (snap.model) applyModel(snap.model, snap.focus ?? null);
      if (pane !== 'plot' && body.panelState && snap.model) {
        try { view.restorePanel(pane, body.panelState); } catch (err) { console.warn('FlatPPL: panel state not restored', err); }
      }
      if (snap.select) { lastSelect = snap.select; view.select(snap.select); }
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
        if (welcomed) view.importMeasures(body); else parkedMeasures = body;
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
          if ((body.binding ?? null) !== lastSelect) { lastSelect = body.binding ?? null; view.select(lastSelect); }
          return;
        case 'state.settings': applySettings(body); return;
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
      if (pane === 'plot') session.send('pane.measures', view.exportMeasures(null), 'owner');
      session.send('pane.panelState', { pane, state: view.snapshotPanel(pane) }, 'owner');
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
      });
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
      if (pane === 'plot') session.send('pane.measures', view.exportMeasures(null), 'owner');
      session.send('pane.bye', { reason: 'close', state: view.snapshotPanel(pane) }, 'owner');
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

    globalScope.FlatPPLPane = { pane, view, session };
  }

  if (doc.readyState === 'loading') doc.addEventListener('DOMContentLoaded', boot);
  else boot();
})(typeof window !== 'undefined' ? window : globalThis);
