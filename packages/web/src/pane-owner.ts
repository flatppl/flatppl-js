// @flatppl/web — the playground window's side of the pane breakout.
//
// The user can move the source editor, or the graph, plots or math panel,
// into its own browser window and back. This window stays the owner: it holds the model, the
// router and the editor, and every other window is a replica that renders
// one panel from the state sent here (pane-app.ts). Messages travel through
// pane-session.ts.
//
// Detach: window.open runs synchronously in the click, because a later call
// has lost the user activation and popup blockers refuse it. The panel's
// state and, for plots, its cached samples go to the new window once it says
// hello; the panel is hidden here. No hello within 10 s re-docks it.
//
// Re-dock: the Dock button in the replica, the panel's toggle here, or the
// replica closing. The replica sends its state (and samples) back first.
// pagehide may not run when a window dies, so a poll of window.closed and
// the session's liveness check back it up.
//
// The session id and the detached panels live in sessionStorage, so a reload
// of this window finds its replicas again; replicas that do not answer within
// 10 s are re-docked.
//
// Loaded as a plain <script> installing window.FlatPPLPaneOwner.

'use strict';

(function (globalScope: any) {
  const STORAGE_KEY = 'flatppl-pane-session-v1';
  const PANES = ['source', 'graph', 'plot', 'math'] as const;
  type Pane = typeof PANES[number];
  const LABEL: Record<Pane, string> = { source: 'Source', graph: 'Graph', plot: 'Plots', math: 'Math' };
  const PANEL_ID: Record<Pane, string> = { source: 'source-pane', graph: 'graph-panel', plot: 'plot-panel', math: 'math-panel' };
  const START_TIMEOUT_MS = 10000;
  // After a reload, how long recorded panel windows have to say hello. Longer
  // than their slow hello (5 s), which is the only way they reach this window
  // on the postMessage fallback after their closed banner shows.
  const RECONNECT_TIMEOUT_MS = 10000;
  const BYE_GRACE_MS = 1500;
  const REDOCK_TIMEOUT_MS = 2000;
  const PING_MS = 5000;
  const LOST_MS = 15000;

  interface Entry {
    pane: Pane;
    state: 'starting' | 'live' | 'reconnecting' | 'leaving' | 'docking';
    win: any;
    timer: any;
    panelState: any;
    handover: any;
    incoming: any;
    leavingState: any;
  }

  /**
   * opts:
   *   controller    — the right-pane controller (surfaces.ts); its update is wrapped
   *   activePath()  — the model path the router shows
   *   navigate(model, target) — the router's navigateTo
   *   host          — the web host adapter (revealSourceLine, openModule, editSource, canPersist)
   *   announce(msg, visible?) — status for screen readers, visible when asked
   *   source        — the source editor's side (app.ts): detach(send) swaps
   *                   in a stand-in and returns the handover; redock(state),
   *                   state(), receiveEdit(text, path, at), save(), setEdit(on),
   *                   resolve(text) for the lint
   */
  function install(opts: any) {
    const S = globalScope.FlatPPLPaneSession;
    const doc = globalScope.document;
    const entries = new Map<Pane, Entry>();
    let sid: string | null = null;
    let session: any = null;
    let rev = 0;
    let placedHandle: any = null;
    let closedPoll: any = null;
    // Set from pagehide: a page that is going away must not answer a panel
    // window's hello, or that window takes the dying page for its owner.
    let unloading = false;
    const last: any = { model: null, focus: null, select: null };

    function view(): any { return opts.controller.viewerHandle ? opts.controller.viewerHandle() : null; }

    function persist(): void {
      try {
        const detached = Array.from(entries.keys());
        if (detached.length) globalScope.sessionStorage.setItem(STORAGE_KEY, JSON.stringify({ sid, detached }));
        else globalScope.sessionStorage.removeItem(STORAGE_KEY);
      } catch (_) { /* Without storage a reload re-docks every panel. */ }
    }

    function placement(): Record<string, string> {
      const out: Record<string, string> = {};
      for (const p of PANES) out[p] = entries.has(p) ? 'detached' : 'docked';
      return out;
    }

    function settings() {
      return { theme: doc.documentElement.dataset.theme || null,
        editEnabled: !!(opts.host.canPersist && opts.host.canPersist()) };
    }

    function snapshot() {
      return { rev, model: last.model, focus: last.focus, select: last.select, settings: settings(), placement: placement() };
    }

    function sendState(type: string, body: any): void {
      if (session && !unloading) session.send(type, { ...body, rev: ++rev });
    }

    /** The source stand-in's channel to the source window. */
    function sendSource(type: string, body: any): void {
      if (entries.has('source')) sendState(type, body);
    }

    function ensureSession(): void {
      if (session) return;
      if (!sid) sid = S.randomId();
      session = S.createSession({
        sid, id: 'owner',
        transport: S.defaultTransport(sid, () => Array.from(entries.values()).map((e) => e.win).filter(Boolean)),
        onMessage,
        onVersion() { opts.announce('A detached panel runs another playground version. Reload that window.', true); },
        onLost(from: string) { finishRedock(from.slice(5) as Pane, null); },
      });
      globalScope.addEventListener('pagehide', () => {
        if (session) session.send('pane.bye', { reason: 'owner-unload' });
        unloading = true;
      });
      // Back from the back/forward cache: the page lives again, so its panel
      // windows may say hello to it.
      globalScope.addEventListener('pageshow', (event: any) => {
        if (!event.persisted || !session) return;
        unloading = false;
        // The last-seen times froze with the page; counted from now, they
        // would declare every live window lost on the next ping.
        for (const e of entries.values()) {
          if (e.state === 'live') session.watch('pane:' + e.pane, PING_MS, LOST_MS);
        }
        session.send('owner.hello', {});
      });
      new MutationObserver(() => sendState('state.settings', settings()))
        .observe(doc.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
      session.send('owner.hello', {});
    }

    function startClosedPoll(): void {
      if (closedPoll) return;
      closedPoll = globalScope.setInterval(function () {
        for (const e of Array.from(entries.values())) {
          if (e.win && e.win.closed) finishRedock(e.pane, e.leavingState);
        }
        if (!entries.size) { globalScope.clearInterval(closedPoll); closedPoll = null; }
      }, 1000);
    }

    function setTimer(e: Entry, ms: number, fn: () => void): void {
      if (e.timer) globalScope.clearTimeout(e.timer);
      e.timer = globalScope.setTimeout(() => { e.timer = null; fn(); }, ms);
    }

    /** Open `pane` in a new window. Runs inside the user's click. */
    function detach(pane: Pane): void {
      const v = view();
      if (entries.has(pane)) return;
      if (pane === 'source' ? !opts.source : !v) return;
      if (!sid) sid = S.randomId();
      const panel = doc.getElementById(PANEL_ID[pane]);
      const rect = panel ? panel.getBoundingClientRect() : { width: 640, height: 480 };
      const width = Math.max(420, Math.round(rect.width));
      const height = Math.max(320, Math.round(rect.height));
      const left = Math.round((globalScope.screenX || 0) + 40);
      const top = Math.round((globalScope.screenY || 0) + 40);
      const url = 'pane.html?pane=' + pane + '#sid=' + sid;
      const win = globalScope.open(url, 'flatppl-pane-' + sid + '-' + pane,
        'popup=1,width=' + width + ',height=' + height + ',left=' + left + ',top=' + top);
      if (!win) {
        opts.announce('The browser blocked the new window. Allow pop-ups for this site, then try again.', true);
        return;
      }
      ensureSession();
      // The source editor hands over its text and view; the playground keeps
      // calling a stand-in that forwards to the new window.
      const panelState = pane === 'source' ? { source: opts.source.detach(sendSource) } : v.snapshotPanel(pane);
      const entry: Entry = { pane, state: 'starting', win, timer: null, panelState,
        handover: pane === 'plot' ? v.exportMeasures(null) : null, incoming: null, leavingState: null };
      entries.set(pane, entry);
      if (pane !== 'source') v.setPanelPlacement(pane, 'detached');
      persist();
      sendState('state.placement', { placement: placement() });
      setTimer(entry, START_TIMEOUT_MS, () => {
        finishRedock(pane, null);
        opts.announce('The new window did not start. ' + LABEL[pane] + ' stays here.', true);
      });
      startClosedPoll();
      opts.announce(LABEL[pane] + ' opened in a new window.');
    }

    /** Ask a detached pane to come back (toggle, Dock, menu). */
    function redock(pane: Pane): void {
      const e = entries.get(pane);
      if (!e) return;
      if (e.state === 'live' && session) {
        e.state = 'docking';
        session.send('pane.redock', {}, 'pane:' + pane);
        setTimer(e, REDOCK_TIMEOUT_MS, () => finishRedock(pane, null));
      } else {
        // A reloading window ('leaving') already sent its state with its goodbye.
        finishRedock(pane, e.leavingState);
      }
    }

    /** Show `pane` here again, with the replica's state when it sent one. */
    function finishRedock(pane: Pane, state: any): void {
      const e = entries.get(pane);
      if (!e) return;
      if (e.timer) globalScope.clearTimeout(e.timer);
      entries.delete(pane);
      if (session) session.unwatch('pane:' + pane);
      const v = view();
      if (pane === 'source') {
        opts.source.redock(state && state.source ? state.source : null);
      } else if (v) {
        // The window's samples if it sent them back, else the ones that
        // never left (a window that did not start).
        if (pane === 'plot') v.importMeasures(e.incoming || e.handover || null);
        if (state && state[pane]) {
          try { v.restorePanel(pane, state); } catch (err) { console.warn('FlatPPL: could not restore ' + pane, err); }
        }
        v.setPanelPlacement(pane, 'docked');
      }
      if (e.win && !e.win.closed) { try { e.win.close(); } catch (_) { /* not ours to close */ } }
      persist();
      sendState('state.placement', { placement: placement() });
      opts.announce(LABEL[pane] + ' returned to the main window.');
      const panel = doc.getElementById(PANEL_ID[pane]);
      if (panel && doc.hasFocus && doc.hasFocus()) {
        panel.setAttribute('tabindex', '-1');
        panel.focus();
      }
    }

    function paneOf(from: string): Pane | null {
      const p = from.startsWith('pane:') ? from.slice(5) : '';
      return (PANES as readonly string[]).includes(p) ? p as Pane : null;
    }

    function onMessage(env: any, source: any): void {
      if (unloading) return;
      const pane = paneOf(env.from);
      if (!pane) return;
      const e = entries.get(pane);
      const body = env.body || {};
      if (env.type === 'pane.hello') {
        if (!e) { session.send('pane.redock', { orphan: true }, env.from); return; }
        if (source) e.win = source;
        if (e.timer) { globalScope.clearTimeout(e.timer); e.timer = null; }
        e.state = 'live';
        // A first hello gets the state taken at detach; a hello after a
        // reload gets what the window sent with its goodbye.
        const handover = e.handover || e.incoming;
        // The source window gets the stand-in's current text and revision,
        // which may have moved since detach, and the last view it knows.
        const panelState = pane === 'source'
          ? { source: { ...opts.source.state(), view: (e.panelState || e.leavingState)?.source?.view ?? null } }
          : e.panelState || e.leavingState;
        // Samples first: the replica parks them before the welcome renders.
        if (handover) session.send('pane.measures', handover, env.from);
        session.send('pane.welcome', { snapshot: snapshot(), panelState }, env.from);
        e.handover = null;
        e.incoming = null;
        e.panelState = null;
        e.leavingState = null;
        session.watch(env.from, PING_MS, LOST_MS);
        return;
      }
      if (!e) return;
      switch (env.type) {
        case 'pane.measures': e.incoming = body; return;
        case 'pane.panelState': finishRedock(pane, body.state); return;
        case 'pane.bye':
          // A reload also sends bye; wait for its hello before re-docking.
          e.state = 'leaving';
          e.leavingState = body.state || null;
          setTimer(e, BYE_GRACE_MS, () => {
            // Still open: the window is reloading slowly, not closing. It gets
            // the start timeout; the closed poll docks it if it does close.
            if (e.win && !e.win.closed) {
              setTimer(e, START_TIMEOUT_MS, () => finishRedock(pane, e.leavingState));
              return;
            }
            finishRedock(pane, e.leavingState);
          });
          return;
        case 'intent.redock': redock(pane); return;
        case 'intent.focus': {
          const model = opts.activePath();
          if (model) opts.navigate(model, body.target ?? null);
          return;
        }
        case 'intent.select': {
          const v = view();
          last.select = body.binding ?? null;
          if (v) v.select(last.select);
          sendState('state.select', { binding: last.select });
          return;
        }
        case 'intent.navigate': if (body.model) opts.host.openModule(body.model, body.target || undefined); return;
        case 'intent.reveal':
          opts.host.revealSourceLine(body.line, null, { preserveFocus: !!body.silent });
          if (!body.silent) { try { globalScope.focus(); } catch (_) { /* the browser decides */ } }
          return;
        case 'intent.replaceRange': opts.host.editSource({ range: body.range || null, newText: body.newText }); return;
        // The source window: typing, Save, the edit toggle, and the lint's
        // bundle resolution all run here, where the drafts and stores live.
        case 'intent.edit':
          if (pane === 'source' && typeof body.text === 'string') opts.source.receiveEdit(body.text, body.path ?? null, body);
          return;
        case 'intent.save': if (pane === 'source') opts.source.save(); return;
        case 'intent.setEdit':
          if (pane !== 'source') return;
          opts.source.setEdit(!!body.on);
          sendState('state.settings', settings());
          return;
        case 'rpc.resolve': {
          if (pane !== 'source') return;
          const reply = (result: any) => { if (session && !unloading) session.send('rpc.result', { id: body.id, ...result }, env.from); };
          Promise.resolve().then(() => opts.source.resolve(String(body.text ?? ''))).then(
            (value: any) => {
              try { reply({ ok: true, value: structuredClone(value) }); } catch (err: any) { reply({ ok: false, error: String(err && err.message || err) }); }
            },
            (err: any) => reply({ ok: false, error: String(err && err.message || err) }));
          return;
        }
        default: return;
      }
    }

    /** Wrap the controller so every model and focus change reaches replicas. */
    const update = opts.controller.update;
    opts.controller.update = function (source: any, target: any, o: any) {
      update.call(opts.controller, source, target, o);
      const v = view();
      if (v && v !== placedHandle) {
        // A new viewer mount (first model, or back from another file type).
        placedHandle = v;
        for (const p of entries.keys()) if (p !== 'source') v.setPanelPlacement(p, 'detached');
      }
      o = o || {};
      const model = {
        path: opts.activePath(), type: opts.controller.currentType(), source: source == null ? '' : String(source),
        preview: !!o.preview, variant: o.variant || null,
        bundle: o.preview ? null : { sources: o.bundleSources || null, files: o.bundleFiles || null,
          resolutions: o.bundleResolutions || null, path: o.path || null },
      };
      const prev = last.model;
      const same = prev && prev.path === model.path && prev.type === model.type && prev.source === model.source
        && prev.preview === model.preview && JSON.stringify(prev.bundle) === JSON.stringify(model.bundle);
      last.focus = target || null;
      last.select = null;
      if (!same) {
        last.model = model;
        sendState('state.model', { model, focus: last.focus });
      } else {
        sendState('state.focus', { target: last.focus, pushHistory: !!o.pushHistory });
      }
    };

    /** Restore a session this tab had before a reload. */
    function restore(): void {
      let saved: any = null;
      try { saved = JSON.parse(globalScope.sessionStorage.getItem(STORAGE_KEY) || 'null'); } catch (_) { saved = null; }
      if (!saved || typeof saved.sid !== 'string' || !/^[0-9a-f]{32}$/.test(saved.sid) || !Array.isArray(saved.detached)) return;
      sid = saved.sid;
      for (const p of saved.detached) {
        if (!(PANES as readonly string[]).includes(p)) continue;
        const entry: Entry = { pane: p, state: 'reconnecting', win: null, timer: null, panelState: null,
          handover: null, incoming: null, leavingState: null };
        entries.set(p, entry);
        setTimer(entry, RECONNECT_TIMEOUT_MS, () => finishRedock(p, null));
      }
      if (entries.size) ensureSession();
      if (entries.has('source') && opts.source) opts.source.detach(sendSource);
    }

    restore();

    return {
      detach, redock,
      /** The owner's own viewer reported a graph tap or a math click. */
      selected(binding: string | null): void {
        last.select = binding;
        sendState('state.select', { binding });
      },
      detached: () => Array.from(entries.keys()),
    };
  }

  globalScope.FlatPPLPaneOwner = { install };
})(typeof window !== 'undefined' ? window : globalThis);
