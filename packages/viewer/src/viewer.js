  // =====================================================================
  // FlatPPL viewer — browser-side DAG + plot rendering for FlatPPL source.
  //
  // Public API (window.FlatPPLViewer.mount):
  //
  //   FlatPPLViewer.mount(container, opts)
  //
  //     container  (Element, optional)  the DOM element this viewer
  //                                      attaches to. The viewer expects
  //                                      to find its layout (#cy,
  //                                      #plot-panel, #info, …) somewhere
  //                                      inside container or its
  //                                      ancestor document. Defaults to
  //                                      document.body. Phase 2b will
  //                                      flip this so mount injects the
  //                                      DOM into container itself.
  //
  //     opts       (object, optional)
  //       host       — host adapter. See below. Defaults to a no-op
  //                    shim layered on top of acquireVsCodeApi() when
  //                    available, so the existing VS Code webview wiring
  //                    keeps working without per-call branching.
  //       (other opts will land in 2c — initial source/target, etc.)
  //
  //   Returns nothing currently. Phase 2c adds an update() / dispose()
  //   control surface.
  //
  // Host adapter (opts.host) — defines the IDE-only concerns the viewer
  // delegates outward (cross-pane navigation, panel-title updates,
  // persistent UI state). All methods are optional; missing methods are
  // treated as no-ops. Standalone embeds typically pass {} or omit
  // opts.host entirely.
  //   revealSourceLine?(line)  — reveal/scroll-to a source line
  //   setTitle?(name)          — update the host-managed panel title
  //   saveState?(state)        — persist webview state across reloads
  //   loadState?()             — load previously-persisted state
  //
  // Standalone (online) embedding will land in 2c+: the same mount
  // entry point, no host adapter required.
  // =====================================================================
  //
  // ── DECOMPOSITION MODULE MAP (in progress) ───────────────────────────
  // This file is being decomposed from one ~7k-line IIFE into layered
  // ES modules (engine-concepts §10; mirrors the orchestrator.js facade
  // split). The seams below are one-way (bottom → top); the ONLY
  // structural obstacle is that almost everything from `mount` onward is
  // a closure capturing per-mount state by lexical scope. The phased
  // plan: (1) document seams [here]; (2) thread an explicit per-mount
  // `ctx` object so functions stop relying on lexical capture
  // (behaviour-neutral); (3) hoist the parameterized groups out of
  // `mount` to IIFE scope; (4) split into ES modules + bundle via
  // esbuild (IIFE, globalName FlatPPLViewer) in BOTH host build
  // pipelines, preserving the global-merge + DOMContentLoaded auto-mount
  // + single-acquireVsCodeApi contract.
  //
  // Layers (leaf → top), with their de-facto module boundaries:
  //   L0  static templates           VIEWER_CSS, VIEWER_BODY_HTML
  //   L1  DOM/host shim              ensureCssInjected, getVscodeApi,
  //                                  defaultVscodeHost, auto-mount
  //         (L0+L1 are capture-free — already module-shaped)
  //   --- everything below is currently nested inside mount() ---
  //   L2  palette/format (leaf)      PALETTE/TYPE_STYLE/resolveNodeColor,
  //                                  esc, format* scalar/array/IR/value
  //   L3  engine facade              fixedValueToMeasure, getMeasure,
  //                                  collectRefArrays, resolveMeasureAlias
  //   L4  state cores                worker (ensure/wire/sendWorker),
  //                                  derivations (rebuildDerivations),
  //                                  override/domain stores, plot-frame
  //   L5  renderers                  record/sample-stats, density/corner,
  //                                  samples/array/empirical, profile,
  //                                  buildPlotPlan
  //   L6  DAG                        initCy, renderDAG,
  //                                  drawReificationLassos, focusNode,
  //                                  enterModuleView
  //   L7  orchestration (top)        updatePlotForBinding,
  //                                  applySourceUpdate, message listener,
  //                                  resize observers, mount prologue
  //
  // ── PHASE 2 EXECUTION MAP (ctx-threading) ───────────────────────────
  // Key safety finding (authoritative inventory): there is NO
  // mount-scope variable shadowing by any nested function, and a
  // single `var ctx` declared at the top of mount() shares the EXACT
  // lexical-capture scope as the per-mount state vars it replaces
  // (including async/RAF/event-listener/returned-closure captures).
  // Therefore a mechanical `X → ctx.X` rewrite is behaviour-neutral
  // and can proceed group-by-group (each group's names are disjoint,
  // so each commit is self-contained). Captured mutable-state groups:
  //   G1 DAG/state     cy, bb, history, currentState
  //   G2 plot control  plotEnabled, currentPlotPlan,
  //                    currentPlotBindingName, plotEchart
  //   G3 worker        samplerWorker(+Promise/Error), samplerReqId,
  //                    pendingRequests
  //   G4 derivations   derivationsState, measureCache, histogramCache,
  //                    profileRangeCache, presetOverrides,
  //                    domainOverrides
  //   G5 sampling cfg  rootSeed, SAMPLE_COUNT, REJECTION_BUDGET
  //   G6 source        currentSource, currentVariantId,
  //                    currentBindings, currentLoweredModule
  //   G7 plan memory   pendingPresetName, pendingDomainName,
  //                    planMemoryByName
  //   G8 mutable cfg   HISTORY_CAP (reassigned in configUpdate)
  // Per-mount constants (PALETTE/PHASE_COLORS/DRAW_EDGE_COLOR/
  // TYPE_STYLE/CODICON_PATHS/MODULE_TARGET/host/CONFIG/HINT/
  // SAMPLER_WORKER_URL) also move onto ctx in Phase 2 so Phase 3 can
  // hoist functions out of mount. KNOWN CLEANUP: `fixedValueToMeasure`
  // is defined twice inside mount — the first (full recursive impl,
  // ~L1330) is dead (the later thin delegator wins via fn-decl
  // hoisting); delete the dead first def during the G4 step, not as a
  // drive-by.
  // ─────────────────────────────────────────────────────────────────────
  (function(global) {
    var FlatPPLViewer = (global.FlatPPLViewer = global.FlatPPLViewer || {});


    // Layout markup + stylesheet for the viewer. Phase 2b moves
    // them out of the host page into here so any container can
    // host the viewer without the host having to know the
    // internal DOM shape. The CSS goes once into <head>; the
    // markup goes into the supplied container.
    var VIEWER_CSS = `
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--vscode-editor-background);
      color: var(--vscode-editor-foreground);
      font-family: var(--vscode-font-family, sans-serif);
      font-size: var(--vscode-font-size, 13px);
      overflow: hidden;
    }
    #header {
      padding: 5px 14px;
      border-bottom: 1px solid var(--vscode-panel-border, #444);
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: var(--vscode-editor-font-size, 14px);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      opacity: 0.8;
      min-height: 26px;
      display: flex;
      align-items: center;
    }
    #header .target-name { font-weight: 600; }
    #header .target-eq { opacity: 0.5; margin: 0 4px; }
    /* Vertical split layout: graph on top, plot on bottom. The header
       hosts a Plot on/off toggle (default off). When the toggle is off
       the plot panel collapses to zero height and the graph fills the
       content area. When on, the area is split 60/40 between the two
       panels. The plot panel is always rendered when enabled — even
       for non-plottable bindings it shows a "Not plottable" message,
       so users navigating the graph see a stable layout instead of
       the panel appearing/disappearing.

       Heights subtract header(~32px) + info(60px). */
    #plot-toggle {
      margin-left: auto;
      background: var(--vscode-button-secondaryBackground, #3a3d41);
      color: var(--vscode-button-secondaryForeground, #ccc);
      border: 1px solid var(--vscode-button-border, transparent);
      border-radius: 3px;
      padding: 2px 10px;
      font-size: 12px;
      cursor: pointer;
      font-family: var(--vscode-font-family, sans-serif);
      flex-shrink: 0;
    }
    #plot-toggle:hover { background: var(--vscode-button-secondaryHoverBackground, #505355); }
    #plot-toggle.on {
      background: var(--vscode-button-background, #0e639c);
      color: var(--vscode-button-foreground, #fff);
      border-color: var(--vscode-button-border, transparent);
    }
    #plot-toggle.on:hover { background: var(--vscode-button-hoverBackground, #1177bb); }
    #main {
      display: flex; flex-direction: column;
      width: 100vw; height: calc(100vh - 86px);
      overflow: hidden;
    }
    #graph-panel {
      flex: 1 1 60%; min-height: 80px;
      position: relative; overflow: hidden;
    }
    #graph-panel.full { flex: 1 1 100%; }
    #plot-panel {
      flex: 1 1 40%; min-height: 80px;
      border-top: 1px solid var(--vscode-panel-border, #444);
      display: flex; align-items: center; justify-content: center;
      overflow: hidden;
      position: relative;
    }
    #plot-panel.hidden {
      flex: 0 0 0; min-height: 0; border-top: none;
    }
    #plot-content { width: 100%; height: 100%; }
    /* Drag handle between #graph-panel and #plot-panel. Lets the user
       redistribute vertical space between the DAG view and the plot
       pane. Hidden when the plot panel itself is hidden; the existing
       border-top on #plot-panel doubles as the handle's visible band
       while in resting state, so the divider only adds the
       interactive hover affordance. */
    #plot-divider {
      flex: 0 0 5px;
      cursor: row-resize;
      user-select: none;
      position: relative;
      background: transparent;
    }
    #plot-divider::before {
      content: '';
      position: absolute;
      left: 0; right: 0; top: 2px; bottom: 2px;
      background: transparent;
      transition: background 0.15s ease;
    }
    #plot-divider:hover::before {
      background: var(--vscode-button-background, #0e639c);
    }
    #plot-divider.hidden { display: none; }
    /* Plot pane layout, controls, and chart ctx.host are styled inline by
       renderPlotFrame — no CSS rules needed here for the per-renderer
       layout. The constant-value / message blocks below still rely on
       global rules. */
    #plot-empty {
      opacity: 0.7; padding: 1.6em; text-align: center;
      font-size: 1.08em; line-height: 1.5;
      max-width: 45em; margin: 0 auto;
    }
    /* Italics for the placeholder hints ("Click a binding…", "Not
       plottable…") but NOT for type-error messages — those need to
       read clearly. */
    #plot-empty.hint { font-style: italic; opacity: 0.5; }
    #plot-empty ul { text-align: left; display: inline-block; }
    /* Stop button shown alongside "Sampling…" while a request is in
       flight. Clicking it terminates the worker (which aborts any
       running tight loop) and rejects in-flight promises; the cache
       on the main thread is preserved, so any binding that finished
       before the cancel stays available. */
    #plot-content .plot-stop-btn {
      margin-top: 14px;
      background: var(--vscode-button-secondaryBackground, #3a3d41);
      color: var(--vscode-button-secondaryForeground, #ccc);
      border: 1px solid var(--vscode-button-border, transparent);
      border-radius: 3px;
      padding: 4px 14px;
      font-size: 12px;
      cursor: pointer;
      font-style: normal; opacity: 0.9;
      font-family: var(--vscode-font-family, sans-serif);
    }
    #plot-content .plot-stop-btn:hover {
      background: var(--vscode-button-secondaryHoverBackground, #505355);
      opacity: 1;
    }
    /* Constant-value display: shown when every sample is the same
       value (literal binding, deterministic arithmetic of literals,
       or a degenerate distribution). A histogram of identical values
       is uninformative, so we render the value as readable text. */
    #plot-content .scalar-display {
      width: 100%; height: 100%;
      display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      gap: 6px;
      font-family: var(--vscode-editor-font-family, monospace);
    }
    #plot-content .scalar-display .name {
      font-size: 13px; opacity: 0.6;
    }
    #plot-content .scalar-display .value {
      font-size: 36px; font-weight: 300;
      max-width: 100%; box-sizing: border-box; padding: 0 16px;
      overflow-wrap: anywhere; text-align: center;
    }
    /* Composite values (records, arrays, Dirac wrappers around
       non-trivial bodies, …) drop to a comfortable monospace size
       so long surface forms don't overflow the pane. The class is
       applied by renderTextValue when the text contains structural
       punctuation. */
    #plot-content .scalar-display .value.composite {
      font-size: 16px; font-weight: normal; line-height: 1.4;
    }
    /* Importance-sampling quality readout in the toolbar.
       The base layout is set inline by renderSampleStats; this
       block only carries the colour by quality band. The same
       palette as the phase tags (with green added) so the visual
       vocabulary is consistent across phase / type / quality. */
    .is-quality.is-good     { color: #66BB6A; }   /* green     */
    .is-quality.is-ok       { color: #FFD54F; }   /* yellow    */
    .is-quality.is-bad      { color: #FFB300; }   /* orange    */
    .is-quality.is-unusable { color: #E57373; }   /* red       */
    /* Graph internals fill graph-panel — switched from full-viewport
       sizing to 100% of the parent so the split-flex layout governs. */
    #cy { width: 100%; height: 100%; }
    /* All font-sizes in this stylesheet are relative units (em),
       so the panel scales with VS Code's zoom factor (which adjusts
       --vscode-font-size at the root). The body sets the base font
       size from --vscode-font-size; everything else here is a multiple
       of that. */
    #info {
      min-height: 5.5em;
      padding: 0.6em 1em;
      border-top: 1px solid var(--vscode-panel-border, #444);
      font-size: 1em;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 0.3em;
    }
    #info .row { display: flex; gap: 0.75em; align-items: baseline; flex-wrap: wrap; }
    #info .name { font-weight: 600; font-size: 1.15em; }
    /* The inferred FlatPIR type/shape — sits to the right of the
       name and phase, monospaced so types like "array of real
       (length 10)" align consistently. */
    #info .infer {
      font-size: 0.92em; opacity: 0.8;
      font-family: var(--vscode-editor-font-family, monospace);
      padding: 0.05em 0.45em; border-radius: 3px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
    }
    #info .phase {
      font-size: 0.92em;
      padding: 0.05em 0.45em; border-radius: 3px;
      color: #fff;
    }
    /* Phase tag colors. CSS custom properties are set at startup from
       the JS ctx.PALETTE so the in-bar tag and the node fill share one
       source of truth. */
    #info .phase-fixed         { background: var(--phase-fixed);         color: #222; }
    #info .phase-parameterized { background: var(--phase-parameterized); color: #222; }
    #info .phase-stochastic    { background: var(--phase-stochastic);    color: #222; }
    #info .expr {
      opacity: 0.6; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis;
      font-size: 1em;
    }
    #info .hint { opacity: 0.5; font-style: italic; font-size: 1em; }
    #tooltip {
      position: absolute;
      display: none;
      pointer-events: none;
      background: var(--vscode-editorHoverWidget-background, #2d2d30);
      color: var(--vscode-editorHoverWidget-foreground, #ccc);
      border: 1px solid var(--vscode-editorHoverWidget-border, #454545);
      border-radius: 3px;
      padding: 4px 8px;
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: var(--vscode-editor-font-size, 14px);
      white-space: pre;
      max-width: 400px;
      overflow: hidden;
      text-overflow: ellipsis;
      z-index: 100;
    }
    #back-btn {
      display: none;
      background: var(--vscode-button-secondaryBackground, #3a3d41);
      color: var(--vscode-button-secondaryForeground, #ccc);
      border: 1px solid var(--vscode-button-border, transparent);
      border-radius: 3px;
      padding: 2px 8px;
      font-size: 12px;
      cursor: pointer;
      font-family: var(--vscode-font-family, sans-serif);
      flex-shrink: 0;
      margin-right: 10px;
    }
    #back-btn:hover {
      background: var(--vscode-button-secondaryHoverBackground, #505355);
    }
`;

    var VIEWER_BODY_HTML = `
  <div id="header">
    <button id="back-btn">&larr; Back</button>
    <span id="header-expr"></span>
    <button id="plot-toggle" title="Toggle the plot panel">Plot: off</button>
  </div>
  <div id="main">
    <div id="graph-panel" class="full">
      <div id="cy"></div>
    </div>
    <div id="plot-divider" class="hidden" title="Drag to resize"></div>
    <div id="plot-panel" class="hidden">
      <div id="plot-content"></div>
    </div>
  </div>
  <div id="tooltip"></div>
  <div id="info">
    <span class="hint">Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source</span>
  </div>
`;

    var cssInjected = false;
    function ensureCssInjected() {
      if (cssInjected) return;
      var styleEl = document.createElement('style');
      styleEl.setAttribute('data-flatppl-viewer-css', '');
      styleEl.textContent = VIEWER_CSS;
      document.head.appendChild(styleEl);
      cssInjected = true;
    }

    // Cache acquireVsCodeApi()'s return value: VS Code permits calling
    // it at most once per webview. If the default host adapter is built
    // more than once (e.g. by re-mount), we hand out the same underlying
    // api object instead of throwing.
    var cachedVscodeApi = null;
    function getVscodeApi() {
      if (cachedVscodeApi) return cachedVscodeApi;
      if (typeof acquireVsCodeApi !== 'function') return null;
      try { cachedVscodeApi = acquireVsCodeApi(); } catch (_) { cachedVscodeApi = null; }
      return cachedVscodeApi;
    }

    // Default host adapter for VS Code webviews. Bridges the four
    // host-adapter methods to the corresponding postMessage / setState
    // / getState calls. When NOT inside a VS Code webview
    // (acquireVsCodeApi missing), returns an empty object — the
    // viewer's call sites guard each method with `if (host.foo)`, so
    // missing methods become no-ops cleanly.
    function defaultVscodeHost() {
      var api = getVscodeApi();
      if (!api) return {};
      return {
        revealSourceLine: function(line) { api.postMessage({ type: 'navigateTo', line: line }); },
        setTitle:         function(name) { api.postMessage({ type: 'updateTitle', name: name }); },
        saveState:        function(state) { api.setState(state); },
        loadState:        function() { return api.getState(); },
        signalReady:      function() { api.postMessage({ type: 'webviewReady' }); },
        // VS Code can always edit the source file. The extension
        // host applies the actual WorkspaceEdit on receipt of the
        // persistPreset message.
        canPersist: function() { return true; },
        // Both persist paths use a two-primitive host contract:
        //   promptForName(args) → Promise<string|null>   (UI prompt)
        //   editSource(args)    → Promise<boolean>       (apply edit)
        // The viewer composes them; the host adapter just bridges
        // the primitives to the host's native APIs. For VS Code
        // both round-trip through the extension via postMessage —
        // window.prompt is unavailable in webviews, and edits go
        // through vscode.workspace.applyEdit.
        promptForName: function(args) {
          var nonce = 'pn_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8);
          return new Promise(function(resolve) {
            function listener(event) {
              var m = event.data;
              if (!m || m.type !== 'promptForNameResponse' || m.nonce !== nonce) return;
              window.removeEventListener('message', listener);
              resolve(m.name || null);
            }
            window.addEventListener('message', listener);
            api.postMessage({
              type: 'promptForName',
              suggested:     args.suggested,
              existingNames: args.existingNames || [],
              nonce: nonce,
            });
          });
        },
        editSource: function(args) {
          api.postMessage({
            type: 'editSource',
            range:   args.range || null,
            newText: args.newText,
          });
          // Fire-and-forget; the extension applies the edit and
          // pushes a fresh sourceUpdate. Resolving truthy lets the
          // viewer's promise chain proceed without awaiting an
          // explicit ack across the boundary.
          return Promise.resolve(true);
        },
      };
    }

  // -------------------------------------------------------------
  // Hoisted pure utilities (decomposition Phase 3 / leaf L2).
  // Each of these is a pure function: no references to per-mount
  // ctx state and no calls to in-mount-only helpers. JS function-
  // declaration hoisting keeps in-mount callers working unchanged
  // (they reach these by the normal scope chain). When Phase 4
  // splits viewer.js into modules, this block becomes format.js /
  // util.js, exported as ES modules.
  // -------------------------------------------------------------
  function esc(s) {
    return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }

  function truncateExpr(expr) {
    if (!expr) return '';
    // Truncate array literals: [1, 2, 3, ..., 8, 9, 10]
    var arrMatch = expr.match(/^\\[(.+)\\]$/);
    if (arrMatch) {
      var items = arrMatch[1].split(/\\s*,\\s*/);
      if (items.length > 6) {
        var head = items.slice(0, 3).join(', ');
        var tail = items.slice(-3).join(', ');
        return '[' + head + ', \u2026, ' + tail + ']';
      }
    }
    // Truncate other long expressions in the middle
    if (expr.length > 80) {
      return expr.slice(0, 38) + ' \u2026 ' + expr.slice(-38);
    }
    return expr;
  }

  /**
   * True when every sample equals the first one. Catches:
   *   - literal bindings (c = 5.0)
   *   - derived constants (d = c + 1) — same value at every i
   *   - aliases to constants
   *   - degenerate distributions (e.g. Normal(0, 0))
   * Plotting a histogram for these is just a single tall bar; render
   * the scalar value as text instead. The check is O(n) but n=5000
   * floats is sub-millisecond.
   */
  function samplesAreConstant(samples) {
    if (!samples || samples.length === 0) return false;
    var v = samples[0];
    for (var i = 1; i < samples.length; i++) if (samples[i] !== v) return false;
    return true;
  }

  function formatScalar(v) {
    if (!Number.isFinite(v)) return String(v);
    if (Number.isInteger(v)) return String(v);
    return String(parseFloat(v.toPrecision(4)));
  }

  function formatComplexScalar(re, im) {
    var imv = im === 0 ? 0 : im;            // normalise -0 → 0
    var sign = imv < 0 ? ' - ' : ' + ';
    return formatScalar(re) + sign + formatScalar(Math.abs(imv)) + ' i';
  }

  function complexReBadge() {
    var b = document.createElement('span');
    b.textContent = 'complex — showing Re(z)';
    b.title = 'This binding is complex-valued; the histogram is its '
      + 'real part. Modulus / Im / Argand views are planned.';
    b.style.padding = '0.15em 0.6em';
    b.style.borderRadius = '3px';
    b.style.fontSize = '0.92em';
    b.style.background = 'var(--vscode-badge-background, #4d4d4d)';
    b.style.color = 'var(--vscode-badge-foreground, #fff)';
    return b;
  }

  function formatArrayParts(parts, fullLength, maxShown) {
    var max = maxShown == null ? 8 : maxShown;
    var n = parts.length;
    if (fullLength == null) fullLength = n;
    if (n <= max) return '[' + parts.join(', ') + ']';
    var headN = 3, tailN = 1;
    var head = parts.slice(0, headN);
    var tail = parts.slice(n - tailN, n);
    // No "(length N)" suffix — see formatValue array branch for
    // rationale. Keeping both array-formatters in sync.
    return '[' + head.join(', ') + ', …, ' + tail.join(', ') + ']';
  }

  function formatArrayWithEllipsis(values, maxShown) {
    var parts = new Array(values.length);
    for (var i = 0; i < values.length; i++) parts[i] = formatScalar(values[i]);
    return formatArrayParts(parts, values.length, maxShown);
  }

  /**
   * Pretty-print a FlatPIR IR node as canonical FlatPPL surface
   * syntax. Used by the fixed-Dirac viewer path to render the
   * value argument of `Dirac(value = ...)` without evaluating it
   * (since the value may be non-scalar — record, array — that
   * the engine's main-thread evaluator can't materialise without
   * running the worker).
   *
   * Handles the IR shapes the value-position of a fixed binding
   * can plausibly take: literals, named constants (pi, inf),
   * binding refs, unary neg of literals, vector / record literals.
   * Anything more exotic (calls into transcendental ops, etc.)
   * gets a placeholder "<op>(…)" so the surface form stays
   * legible without claiming false precision.
   */
  function formatIRValue(ir) {
    if (!ir) return '?';
    if (ir.kind === 'lit')   return formatScalar(ir.value);
    if (ir.kind === 'const') return ir.name; // pi / e / inf / true / false
    if (ir.kind === 'ref') {
      return (ir.ns && ir.ns !== 'self' ? ir.ns + '.' : '') + ir.name;
    }
    if (ir.kind === 'call' && ir.op === 'neg' && ir.args && ir.args.length === 1) {
      return '-' + formatIRValue(ir.args[0]);
    }
    if (ir.kind === 'call' && ir.op === 'vector' && Array.isArray(ir.args)) {
      return '[' + ir.args.map(formatIRValue).join(', ') + ']';
    }
    if (ir.kind === 'call' && ir.op === 'record') {
      var entries = [];
      var kwargs = ir.kwargs || {};
      for (var k in kwargs) {
        entries.push(k + ' = ' + formatIRValue(kwargs[k]));
      }
      return 'record(' + entries.join(', ') + ')';
    }
    if (ir.kind === 'call' && ir.op === 'tuple' && Array.isArray(ir.args)) {
      return '(' + ir.args.map(formatIRValue).join(', ') + ')';
    }
    if (ir.kind === 'call' && ir.op) {
      // Generic call: op(arg1, arg2, k=v) — useful for
      // fchain-style nestings without committing to a precise
      // pretty-print of unknown ops.
      var parts = [];
      if (Array.isArray(ir.args)) {
        for (var i = 0; i < ir.args.length; i++) parts.push(formatIRValue(ir.args[i]));
      }
      var kw = ir.kwargs || {};
      for (var k2 in kw) parts.push(k2 + ' = ' + formatIRValue(kw[k2]));
      return ir.op + '(' + parts.join(', ') + ')';
    }
    return '?';
  }

  function formatValue(v, opts) {
    if (typeof v === 'number')  return formatScalar(v);
    if (typeof v === 'boolean') return String(v);
    if (typeof v === 'string')  return JSON.stringify(v);
    if (v == null)              return 'null';
    var isArrayLike = Array.isArray(v) || ArrayBuffer.isView(v);
    if (isArrayLike) {
      var len = v.length;
      var max = (opts && opts.maxArray) || 8;
      if (len <= max) {
        var parts = new Array(len);
        for (var i = 0; i < len; i++) parts[i] = formatValue(v[i], opts);
        return '[' + parts.join(', ') + ']';
      }
      var headN = 3, tailN = 1;
      var head = new Array(headN);
      for (var hi = 0; hi < headN; hi++) head[hi] = formatValue(v[hi], opts);
      var tail = new Array(tailN);
      for (var ti = 0; ti < tailN; ti++) {
        tail[ti] = formatValue(v[len - tailN + ti], opts);
      }
      // Ellipsis form drops the explicit "(length N)" suffix —
      // panes are often narrow and the elided "…" already signals
      // the array continues. Callers that need the count can
      // surface it separately (the corner-plot axis labels and
      // the info panel both already do).
      return '[' + head.join(', ') + ', …, ' + tail.join(', ') + ']';
    }
    if (typeof v === 'object') {
      var keys = Object.keys(v);
      var entries = new Array(keys.length);
      for (var k = 0; k < keys.length; k++) {
        entries[k] = keys[k] + ' = ' + formatValue(v[keys[k]], opts);
      }
      return 'record(' + entries.join(', ') + ')';
    }
    return String(v);
  }

  function formatLogTotalmass(logTotalmass) {
    if (!Number.isFinite(logTotalmass)) {
      return logTotalmass === -Infinity ? '0 (zero mass)' : null;
    }
    if (Math.abs(logTotalmass) < 1e-9) return null;   // ≈ normalized
    // Float64 representable: ~ log(Number.MAX_VALUE) ≈ 709.78. Use a
    // tighter range so the linear form stays human-readable; outside
    // that, render in exp(...) form which stays meaningful at any
    // scale.
    if (Math.abs(logTotalmass) <= 12) {
      var linear = Math.exp(logTotalmass);
      if (linear >= 0.01 && linear < 10000) return linear.toPrecision(4);
      return linear.toExponential(3);
    }
    return 'exp(' + (logTotalmass >= 0 ? '+' : '') + logTotalmass.toPrecision(5) + ')';
  }

  /**
   * Tooltip text for the quality-readout span. Spells out the
   * diagnostic ingredients so a hover gives the full picture.
   */
  function qualityTooltip(q) {
    var parts = [
      'Importance-sampling quality: ' + q.label,
      '',
      'Kish ESS: ' + Math.round(q.ess).toLocaleString('en-US')
        + ' / ' + q.N.toLocaleString('en-US')
        + ' (' + (q.ratio * 100).toFixed(1) + '%)',
    ];
    if (Number.isFinite(q.kHat)) {
      parts.push('PSIS k̂: ' + q.kHat.toFixed(3)
        + '  (≤0.5 finite variance · ≤0.7 usable · >1 untrustworthy)');
    } else {
      parts.push('PSIS k̂: not applicable (unweighted measure)');
    }
    parts.push('Max single-atom weight: ' + (q.wmax * 100).toFixed(2) + '%');
    parts.push('Effective DOF (estimate): ' + q.dof);
    return parts.join('\n');
  }

  function measureAtomCount(measure) {
    // Record / tuple measures have no top-level .samples; pull
    // length from any sub-measure's samples — all components share
    // the same atom count by construction.
    if (measure.fields) {
      var anyKey = Object.keys(measure.fields)[0];
      return anyKey ? measureAtomCount(measure.fields[anyKey]) : 0;
    }
    if (Array.isArray(measure.elems) && measure.elems.length > 0) {
      return measureAtomCount(measure.elems[0]);
    }
    // Array-shape measure: samples is a flat atom-major buffer of
    // length N × stride (e.g. iid(Normal, 10) at N=100k → buffer
    // length 1M, dims=[10]). Divide out the stride so N reads as
    // 100,000, not 1,000,000. Scalar measures have no dims (or
    // dims=[]) and pass through unchanged.
    if (measure.samples) {
      if (measure.dims && measure.dims.length > 0) {
        var stride = measure.dims.reduce(function(p, n) { return p * n; }, 1);
        return stride > 0 ? measure.samples.length / stride : 0;
      }
      return measure.samples.length;
    }
    return 0;
  }

  function formatCount(n) {
    // Integer-formatted count with thousands separators.
    return Math.round(n).toLocaleString('en-US');
  }

  function formatSampleCount(n) {
    if (n > 0 && Math.floor(n) === n) {
      var lg = Math.log10(n);
      if (lg >= 2 && Number.isInteger(lg)) {
        var sup = '';
        var s = String(lg);
        for (var i = 0; i < s.length; i++) {
          sup += '⁰¹²³⁴⁵⁶⁷⁸⁹'[+s[i]];
        }
        return '10' + sup;
      }
    }
    return formatCount(n);
  }

  function hexToRgba(hex, alpha) {
    var m = /^#([0-9a-f]{6})$/i.exec(hex);
    if (!m) return hex;
    var n = parseInt(m[1], 16);
    var r = (n >> 16) & 0xff, g = (n >> 8) & 0xff, b = n & 0xff;
    return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
  }



  /**
   * Wrap a Philox state in a closure that returns U(0,1) uniforms,
   * matching the "() => number" callback shape that
   * empirical.systematicResample / multinomialResample expect.
   * Used for deterministic main-thread resampling under a
   * per-binding seed (currently: superpose).
   *
   * Same-engine RNG as the worker's sampleN, but instantiated locally
   * so empirical.js stays dep-free of rng.js — visualPanel does
   * the wiring at the call site.
   */
  function makeMainThreadPrng(seed) {
    var state = FlatPPLEngine.rng.stateFromKey(seed);
    return function() {
      var pair = FlatPPLEngine.rng.nextUniform(state);
      state = pair[1];
      return pair[0];
    };
  }

  /**
   * Common echarts zoom config — mouse-wheel + drag zoom on x via
   * the inside-type dataZoom, plus a top-left toolbox button for
   * rectangle-select zoom and a reset button. y-axis stays fixed:
   * zooming probability values alone is rarely useful and the
   * rectangle-select would otherwise need a second click to reset
   * each axis.
   *
   * filterMode 'none' keeps out-of-window data rendered (just
   * clipped) so panning is smooth.
   *
   * Returned fresh each call so the caller can pass to setOption.
   */
  function plotZoomOptions(fg) {
    return {
      dataZoom: [
        { type: 'inside', xAxisIndex: 0, filterMode: 'none' },
      ],
      toolbox: {
        show: true,
        left: 12, top: 4,
        itemSize: 14,
        iconStyle: { borderColor: fg, opacity: 0.55 },
        emphasis: { iconStyle: { borderColor: fg, opacity: 1 } },
        feature: {
          dataZoom: {
            yAxisIndex: 'none',
            title: { zoom: 'Zoom rectangle', back: 'Reset zoom' },
          },
          restore: { title: 'Reset' },
        },
      },
    };
  }

  /**
   * Enumerate the plottable scalar leaves of a multivariate
   * EmpiricalMeasure, with display labels and synthetic per-axis
   * sample arrays.
   *
   *   - record fields with scalar samples → axis with the field name.
   *   - record fields with array samples → one axis per array slot,
   *     labelled "field[i]" (1-indexed per spec §03 line 148);
   *     the per-axis samples array is a strided extract from the
   *     atom-major buffer.
   *   - top-level array measures → one axis per slot, labelled
   *     "[i]" (the binding name itself shows up in the bubble).
   *
   * Strided extracts allocate fresh Float64Arrays — small (~N
   * elements) so cheap; gives downstream code the same scalar-shape
   * Float64Array regardless of whether the source was a flat record
   * field or a column of an iid array.
   */
  function listScalarAxes(measure) {
    var out = [];
    function walk(m, prefix) {
      if (m.fields) {
        var ks = Object.keys(m.fields);
        for (var i = 0; i < ks.length; i++) {
          walk(m.fields[ks[i]], prefix ? (prefix + '.' + ks[i]) : ks[i]);
        }
        return;
      }
      if (m.shape === 'tuple' && Array.isArray(m.elems)) {
        // Positional analogue of record. 1-indexed component
        // labels per FlatPPL convention (xs[1], xs[2], ...).
        for (var ti = 0; ti < m.elems.length; ti++) {
          var label = (prefix ? prefix : '') + '[' + (ti + 1) + ']';
          walk(m.elems[ti], label);
        }
        return;
      }
      if (m.shape === 'array' && m.samples instanceof Float64Array && m.dims) {
        // Total inner stride per atom = prod(dims).
        var k = m.dims.reduce(function(p, n) { return p * n; }, 1);
        var N = m.samples.length / k;
        for (var slot = 0; slot < k; slot++) {
          // Stride-extract slot j across atoms.
          var col = new Float64Array(N);
          for (var i = 0; i < N; i++) col[i] = m.samples[i * k + slot];
          // 1-indexed labels per FlatPPL convention.
          var label = (prefix ? prefix : '') + '[' + (slot + 1) + ']';
          out.push({ key: label, label: label, samples: col });
        }
        return;
      }
      if (m.samples instanceof Float64Array) {
        // Plain scalar leaf.
        out.push({ key: prefix, label: prefix, samples: m.samples });
      }
    }
    walk(measure, '');
    return out;
  }

  // -------------------------------------------------------------
  // Hoisted ctx-taking utilities (decomposition Phase 3b.2).
  // Each takes `ctx` (the per-mount state container) as its first
  // parameter; every call site in this file has been updated to
  // pass ctx. In-mount callers reach ctx via lexical capture (the
  // `var ctx = {}` at the top of mount()); IIFE-scope callers
  // pass their own ctx parameter through. Phase 4 will turn this
  // block into ES modules (palette.js / engine-facade.js / etc.).
  // -------------------------------------------------------------
  /**
   * Single source of truth for "what colour does this node get?".
   * Used by the DAG renderer, the plot-view colorForBinding lookup,
   * and the reification-bubble fill so all three views stay coherent.
   *
   * Decision tree:
   *   kind === 'kernel'         → kernelof teal (overrides type)
   *   kind === 'measure'        → lawof blue   (overrides type)
   *   type ∈ {'draw', 'call'}   → ctx.PHASE_COLORS[phase]   (value node)
   *   else                      → ctx.TYPE_STYLE[type].color (structural)
   *
   * Inside a reification bubble, node.phase has already been
   * overridden to the scope-local phase by dag.js's
   * applyScopeLocalPhases — so the same theta1 reads stochastic in
   * the main view and parameterized inside a kernel bubble.
   */
  function resolveNodeColor(ctx, node) {
    if (node.kind === 'kernel')  return ctx.TYPE_STYLE.kernelof.color;
    if (node.kind === 'measure') return ctx.TYPE_STYLE.lawof.color;
    var ts = ctx.TYPE_STYLE[node.type] || ctx.TYPE_STYLE.unknown;
    if (node.type === 'draw' || node.type === 'call') {
      return ctx.PHASE_COLORS[node.phase] || ts.color;
    }
    return ts.color;
  }

  /**
   * FNV-1a 32-bit string hash, then XOR the root seed. Used to give
   * each binding its own RNG stream for sampleN(). Independent of
   * arrival order — two independent variables stay independent
   * regardless of which one the user clicked first.
   */
  function nameSeed(ctx, name) {
    var h = 2166136261;
    for (var i = 0; i < name.length; i++) {
      h = h ^ name.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h ^ ctx.rootSeed) >>> 0;
  }

  function measureIsConstant(ctx, m) {
    if (!m) return false;
    if (m.fields) {
      for (var k in m.fields) {
        if (!measureIsConstant(ctx, m.fields[k])) return false;
      }
      return true;
    }
    if (Array.isArray(m.elems)) {
      for (var i = 0; i < m.elems.length; i++) {
        if (!measureIsConstant(ctx, m.elems[i])) return false;
      }
      return true;
    }
    if (m.shape === 'array' && m.samples instanceof Float64Array && m.dims) {
      // Atom-major SoA: stride k = prod(dims) per atom. The whole
      // array is constant iff slot s has the same value at every
      // atom, for every s in [0, k).
      var stride = m.dims.reduce(function(p, n) { return p * n; }, 1);
      if (stride === 0) return true;
      var N = m.samples.length / stride;
      for (var s = 0; s < stride; s++) {
        var v = m.samples[s];
        for (var ai = 1; ai < N; ai++) {
          if (m.samples[ai * stride + s] !== v) return false;
        }
      }
      return true;
    }
    if (m.samples instanceof Float64Array) {
      // Length-mismatch with SAMPLE_COUNT identifies a literal-array
      // measure (kind: 'array' derivation): per-atom these are
      // deterministic, even though the array's own elements differ.
      if (m.samples.length !== ctx.SAMPLE_COUNT) return true;
      return samplesAreConstant(m.samples);
    }
    return false;
  }

  function formatConstantMeasure(ctx, m) {
    if (!m) return '?';
    if (m.fields) {
      var ks = Object.keys(m.fields);
      var fparts = new Array(ks.length);
      for (var i = 0; i < ks.length; i++) {
        fparts[i] = ks[i] + ' = ' + formatConstantMeasure(ctx, m.fields[ks[i]]);
      }
      return 'record(' + fparts.join(', ') + ')';
    }
    if (Array.isArray(m.elems)) {
      var eparts = new Array(m.elems.length);
      for (var ei = 0; ei < m.elems.length; ei++) {
        // Tuple element may be null when fixedValueToMeasure
        // couldn't represent it (an rngstate, typically). Surface
        // a placeholder so the rest of the tuple's structure stays
        // visible — e.g. `(record(obs = […]), <rngstate>)` for a
        // single-LHS `rand(rs, m)` result.
        eparts[ei] = m.elems[ei] ? formatConstantMeasure(ctx, m.elems[ei]) : '<rngstate>';
      }
      return '(' + eparts.join(', ') + ')';
    }
    if (m.shape === 'array' && m.samples instanceof Float64Array && m.dims) {
      var stride = m.dims.reduce(function(p, n) { return p * n; }, 1);
      return formatValue(m.samples.subarray(0, stride));
    }
    if (m.samples instanceof Float64Array && m.samples.length > 0) {
      // Two cases distinguished by sample length:
      //   - length === SAMPLE_COUNT: a per-atom scalar measure
      //     (caller verified samples are constant across atoms);
      //     surface a single number.
      //   - length !== SAMPLE_COUNT: a literal-data array
      //     (kind:'array' derivation surfaced as a record field);
      //     surface every element with array ellipsis.
      if (m.samples.length === ctx.SAMPLE_COUNT) return formatScalar(m.samples[0]);
      return formatValue(m.samples);
    }
    return '?';
  }

  /**
   * Return the analyzer-level error diagnostics that landed on a
   * binding (typeinfer mismatches, undefined refs, etc.), or null
   * if there are none. Source for both the plot pane's
   * "semantically invalid" message and the DAG's red error border.
   */
  function errorsForBinding(ctx, bindingName) {
    if (!bindingName || !ctx.currentState || !ctx.currentState.data
        || !ctx.currentState.data.nodes) return null;
    var nodes = ctx.currentState.data.nodes;
    for (var i = 0; i < nodes.length; i++) {
      if (nodes[i].id === bindingName) return nodes[i].errors || null;
    }
    return null;
  }

  function colorForBinding(ctx, bindingName) {
    if (ctx.currentState && ctx.currentState.data && ctx.currentState.data.nodes) {
      var nodes = ctx.currentState.data.nodes;
      for (var i = 0; i < nodes.length; i++) {
        if (nodes[i].id === bindingName) return resolveNodeColor(ctx, nodes[i]);
      }
    }
    // Fallback when the plot is updating ahead of the DAG (rare, but
    // possible during config-update reflows). currentBindings has
    // .type but not .kind/.phase, so resolveNodeColor naturally
    // degrades to the type colour.
    var binding = ctx.currentBindings && ctx.currentBindings.get(bindingName);
    return resolveNodeColor(ctx, { type: (binding && binding.type) || 'draw' });
  }



  /**
   * Walk the derivation chain for a measure binding to find the
   * value-typed binding it's mathematically equivalent to (if any).
   * Two equivalence forms after engine canonicalisation:
   *
   *   m = lawof(observed_data)   → derivation 'alias' to observed_data
   *   m = Dirac(observed_data)   → derivation 'sample' on Dirac IR
   *                                whose kwargs.value is a ref to
   *                                observed_data
   *
   * Both routes resolve to 'observed_data' here. Composes through
   * multiple alias hops; cycle-guarded.
   *
   * Returns null when the chain doesn't bottom out on a single
   * named value-typed source binding (e.g. Dirac(value = literal),
   * Dirac(value = inline-call), or a non-degenerate sample step).
   * The caller then falls through to the existing dispatch.
   */
  function resolveMeasureAlias(name, derivations, bindings) {
    if (!derivations || !bindings) return null;
    var seen = new Set();
    var cur = name;
    while (cur && !seen.has(cur)) {
      seen.add(cur);
      var d = derivations[cur];
      if (!d) return null;
      if (d.kind === 'alias') { cur = d.from; continue; }
      // Engine-canonicalised Dirac with a ref to a known binding
      // → follow the ref. Anything else (literal value, inline
      // expression, non-Dirac sample) terminates the resolution.
      if (d.kind === 'sample' && d.distIR
          && d.distIR.kind === 'call' && d.distIR.op === 'Dirac'
          && d.distIR.kwargs && d.distIR.kwargs.value) {
        var v = d.distIR.kwargs.value;
        if (v.kind === 'ref' && v.ns === 'self' && bindings.has(v.name)) {
          cur = v.name;
          continue;
        }
      }
      break;
    }
    return cur === name ? null : cur;
  }


  // -------------------------------------------------------------
  // Hoisted L3 engine facade + L4 worker subgroup (Phase 3d).
  // Each takes ctx as first param; in-file call sites have been
  // updated to pass ctx. Two callback sites (getMeasure and
  // sendWorker passed to FlatPPLEngine.materialiser as 1-arg
  // callbacks) are wrapped at the call point to bind ctx — the
  // engine's signature contract stays unchanged.
  // -------------------------------------------------------------
  function tryGetMeasure(ctx, name) {
    return getMeasure(ctx, name).then(
      function(m) { return m; },
      function(_err) { return null; });
  }

  function getMeasure(ctx, name) {
    if (ctx.measureCache.has(name)) return Promise.resolve(ctx.measureCache.get(name));
    if (!ctx.derivationsState) return Promise.reject(new Error('no model loaded'));
    // All per-kind materialisation lives in the engine — the viewer's
    // job here is just to memoise the result against the cache. The
    // engine-side materialiser dispatches by derivation kind, computes
    // samples + logWeights + logTotalmass + n_eff, and returns the
    // Measure record. Recursion is handled by passing getMeasure
    // itself back in so child materialisations hit the same cache.
    var promise = FlatPPLEngine.materialiser.materialiseMeasure(name, {
      derivations: ctx.derivationsState.derivations,
      bindings:    ctx.derivationsState.bindings,
      fixedValues: ctx.derivationsState.fixedValues,
      // Bind ctx into 1-arg callbacks: the engine's
      // materialiseMeasure expects callbacks with the original
      // signatures (`getMeasure(name)`, `sendWorker(msg)`); our
      // hoisted versions added `ctx` as a first parameter, so we
      // close over `ctx` here to keep the engine ABI unchanged.
      getMeasure:  function (n) { return getMeasure(ctx, n); },
      sendWorker:  function (m) { return sendWorker(ctx, m); },
      sampleCount: ctx.SAMPLE_COUNT,
      rootSeed:    ctx.rootSeed,
      rejectionBudget: ctx.REJECTION_BUDGET,
    });
    promise.then(function(m) { ctx.measureCache.set(name, m); });
    return promise;
  }

  function fixedValueToMeasure(ctx, v) {
    return FlatPPLEngine.materialiser.fixedValueToMeasure(v, ctx.SAMPLE_COUNT);
  }

  function collectRefArrays(ctx, ir) {
    var fv = ctx.derivationsState && ctx.derivationsState.fixedValues;
    return FlatPPLEngine.materialiser.collectRefArrays(
      ir, fv, function (n) { return getMeasure(ctx, n); });
  }

  /**
   * Asynchronously spawn the sampler worker, caching the result.
   * Returns a Promise<Worker> so callers can await spawn completion.
   *
   * Why blob-URL? VS Code webviews are sandboxed iframes whose CSP and
   * cross-origin posture refuses 'new Worker(webview-uri)' in some
   * VS Code versions. The reliable workaround is to fetch the bundle
   * text via the webview URI (which IS allowed by connect-src) and
   * spawn the worker from a same-origin blob: URL. This pattern is
   * also documented in the official VS Code webview samples.
   */
  function ensureSamplerWorker(ctx) {
    if (ctx.samplerWorker) return Promise.resolve(ctx.samplerWorker);
    if (ctx.samplerWorkerPromise) return ctx.samplerWorkerPromise;

    ctx.samplerWorkerPromise = (async function() {
      // Try direct construction first — cheapest path on hosts where
      // it works. Fall back to blob: on any failure (security error,
      // cross-origin block, etc.).
      var w = null;
      try {
        w = new Worker(ctx.SAMPLER_WORKER_URL);
      } catch (e) {
        // continue to blob fallback
        console.warn('FlatPPL: direct worker spawn failed, retrying via blob URL:', e && e.message);
      }
      if (!w) {
        var resp = await fetch(ctx.SAMPLER_WORKER_URL);
        if (!resp.ok) throw new Error('failed to fetch worker bundle: ' + resp.status + ' ' + resp.statusText);
        var src = await resp.text();
        var blob = new Blob([src], { type: 'application/javascript' });
        var url = URL.createObjectURL(blob);
        w = new Worker(url);
        // The blob URL only needs to live until the Worker has parsed
        // its source — revoke after a short delay so the URL isn't
        // leaked (the worker keeps running independently).
        setTimeout(function() { try { URL.revokeObjectURL(url); } catch (_) {} }, 5000);
      }
      wireWorker(ctx, w);
      ctx.samplerWorker = w;
      // Initialize with a fixed seed for deterministic output. Future:
      // plumb a "Resample" button that re-seeds (e.g. from Date.now()).
      sendWorkerNow(ctx, w, { type: 'init', seed: 1 });
      return w;
    })();

    ctx.samplerWorkerPromise.catch(function(err) {
      ctx.samplerWorkerError = err;
      ctx.samplerWorkerPromise = null;
      console.error('FlatPPL: sampler worker unavailable:', err);
    });

    return ctx.samplerWorkerPromise;
  }

  function wireWorker(ctx, w) {
    w.addEventListener('message', function(ev) {
      var reply = ev.data;
      if (!reply || reply.id == null) return;
      var p = ctx.pendingRequests.get(reply.id);
      if (!p) return;
      ctx.pendingRequests.delete(reply.id);
      if (reply.type === 'error') p.reject(new Error(reply.message || 'worker error'));
      else p.resolve(reply);
    });
    w.addEventListener('error', function(e) {
      // A top-level worker error fails every outstanding request — there's
      // no way to know which request the error pertains to, and the worker
      // may be dead. Reject all and reset so a future request can retry
      // the spawn.
      console.error('FlatPPL sampler worker error:', e.message || e);
      for (var entry of ctx.pendingRequests.values()) entry.reject(new Error(e.message || 'worker crashed'));
      ctx.pendingRequests.clear();
      try { w.terminate(); } catch (_) {}
      if (ctx.samplerWorker === w) {
        ctx.samplerWorker = null;
        ctx.samplerWorkerPromise = null;
      }
    });
  }

  function sendWorkerNow(ctx, w, msg) {
    var id = ++ctx.samplerReqId;
    w.postMessage(Object.assign({ id: id }, msg));
  }

  function sendWorker(ctx, msg) {
    return ensureSamplerWorker(ctx).then(function(w) {
      var id = ++ctx.samplerReqId;
      var wrapped = Object.assign({ id: id }, msg);
      return new Promise(function(resolve, reject) {
        ctx.pendingRequests.set(id, { resolve: resolve, reject: reject });
        w.postMessage(wrapped);
      });
    });
  }



  // ---- Hoisted L4 override store + plot-frame helpers (Phase 3e) ----

  /**
   * Reset plot-content's inline style. The marginals view sets
   * display:grid with several layout properties; subsequent
   * single-chart views need a clean slate so their content fills
   * the pane without inheriting a stale grid.
   */
  function resetPlotContentStyle(ctx) {
    var el = document.getElementById('plot-content');
    el.style.display = '';
    el.style.gridTemplateColumns = '';
    el.style.gridTemplateRows = '';
    el.style.gap = '';
    el.style.padding = '';
    el.style.boxSizing = '';
    el.style.flexDirection = '';
  }

  function showPlotMessage(ctx, html, options) {
    if (ctx.plotEchart) { ctx.plotEchart.dispose(); ctx.plotEchart = null; }
    resetPlotContentStyle(ctx);
    var el = document.getElementById('plot-content');
    var cancellable = options && options.cancellable;
    var hint       = options && options.hint;
    var stopHtml = cancellable
      ? '<div><button class="plot-stop-btn" id="plot-stop-btn">Stop</button></div>'
      : '';
    var cls = hint ? ' class="hint"' : '';
    el.innerHTML = '<div id="plot-empty"' + cls + '>' + html + stopHtml + '</div>';
    if (cancellable) {
      var btn = document.getElementById('plot-stop-btn');
      if (btn) btn.addEventListener('click', cancelAllSampling);
    }
  }

  /**
   * Cancel any in-flight sample requests by terminating the worker
   * and rejecting every pending promise. The main-thread sample
   * cache is preserved, so bindings that finished before the cancel
   * stay available — only the in-flight request is dropped.
   *
   * The next sendWorker() call will lazily re-spawn the worker via
   * ensureSamplerWorker(). Cheap enough that we don't bother
   * keeping a "warm" worker around.
   */
  function cancelAllSampling(ctx) {
    if (ctx.samplerWorker) {
      try { ctx.samplerWorker.terminate(); } catch (_) {}
      ctx.samplerWorker = null;
      ctx.samplerWorkerPromise = null;
    }
    var entries = ctx.pendingRequests.values();
    ctx.pendingRequests = new Map();
    for (var entry of entries) {
      try { entry.reject(new Error('cancelled')); } catch (_) {}
    }
  }

  function overrideEntryFor(ctx, plan) {
    if (plan.presetName == null) return plan.autoOverride;
    return ctx.presetOverrides.get(plan.presetName) || null;
  }

  function hasOverrides(ctx, plan) {
    var e = overrideEntryFor(ctx, plan);
    if (!e) return false;
    var v = e.values || {};
    for (var k in v) {
      if (Object.prototype.hasOwnProperty.call(v, k)) return true;
    }
    return false;
  }

  function setOverrideFor(ctx, plan, entry) {
    if (plan.presetName == null) {
      plan.autoOverride = entry;
      return;
    }
    if (entry) {
      ctx.presetOverrides.set(plan.presetName, entry);
    } else {
      ctx.presetOverrides.delete(plan.presetName);
    }
  }

  function ensureOverrideFor(ctx, plan) {
    var existing = overrideEntryFor(ctx, plan);
    if (existing) {
      existing.values = Object.assign({}, existing.values || {});
      return existing;
    }
    return { values: {} };
  }

  function activePresetFor(ctx, plan) {
    var baseValues = baseValuesFor(ctx, plan);
    var entry = overrideEntryFor(ctx, plan);
    if (!entry) return { values: baseValues };
    return {
      values: Object.assign({}, baseValues, entry.values || {}),
    };
  }

  function baseValuesFor(ctx, plan) {
    if (plan.presetName != null && plan.matchedPresets) {
      for (var i = 0; i < plan.matchedPresets.length; i++) {
        if (plan.matchedPresets[i].name === plan.presetName) {
          return plan.matchedPresets[i].values || {};
        }
      }
    }
    return {};
  }

  function domainOverrideEntryFor(ctx, plan) {
    if (plan.domainName == null) return plan.domainAutoOverride || null;
    return ctx.domainOverrides.get(plan.domainName) || null;
  }

  function ensureDomainOverrideFor(ctx, plan) {
    var existing = domainOverrideEntryFor(ctx, plan);
    if (existing) {
      existing.ranges = Object.assign({}, existing.ranges || {});
      return existing;
    }
    return { ranges: {} };
  }

  function setDomainOverrideFor(ctx, plan, entry) {
    if (plan.domainName == null) {
      plan.domainAutoOverride = entry;
      return;
    }
    if (entry) {
      ctx.domainOverrides.set(plan.domainName, entry);
    } else {
      ctx.domainOverrides.delete(plan.domainName);
    }
  }

  function hasDomainOverrides(ctx, plan) {
    var e = domainOverrideEntryFor(ctx, plan);
    if (!e || !e.ranges) return false;
    return Object.keys(e.ranges).length > 0;
  }

  function baseRangesFor(ctx, plan) {
    if (plan.domainName != null && plan.matchedDomains) {
      for (var i = 0; i < plan.matchedDomains.length; i++) {
        if (plan.matchedDomains[i].name === plan.domainName) {
          return plan.matchedDomains[i].ranges || {};
        }
      }
    }
    return {};
  }

  function activeDomainRangesFor(ctx, plan) {
    var base = baseRangesFor(ctx, plan);
    var entry = domainOverrideEntryFor(ctx, plan);
    if (!entry || !entry.ranges) return Object.assign({}, base);
    return Object.assign({}, base, entry.ranges);
  }

  function activeFixedNamesFor(ctx, plan) {
    if (plan.presetName != null && plan.matchedPresets) {
      for (var i = 0; i < plan.matchedPresets.length; i++) {
        if (plan.matchedPresets[i].name === plan.presetName) {
          return plan.matchedPresets[i].fixedNames || new Set();
        }
      }
    }
    return new Set();
  }



  // ---- Hoisted leaf batch (Phase 3f) — header/info, leaf defaults,
  //      persist-helpers, plan-memory, bubble teardown ----

  function showNodeInfo(d) {
    var phase = d.phase || 'unknown';
    var phaseTag = '<span class="phase phase-' + esc(phase) + '">' + esc(phase) + ' phase</span>';
    var unsupportedRow = '';
    if (d.unsupported) {
      var msg = 'disintegration unresolved: ' + esc(d.unsupportedReason || '');
      if (d.unsupportedDetail) msg += ' — ' + esc(d.unsupportedDetail);
      unsupportedRow = '<div class="expr" style="color:#FF8A65;">' + msg + '</div>';
    }
    // Type-error row(s). Drawn in the same red as the node border so
    // the visual link reads at a glance. Each diagnostic gets its own
    // line — a single binding can pick up several mismatches if its
    // RHS has multiple bad arg positions.
    var errorRow = '';
    var errors = errorsForBinding(ctx, d.id);
    if (errors && errors.length > 0) {
      for (var i = 0; i < errors.length; i++) {
        errorRow += '<div class="expr" style="color:#E57373;">' + esc(errors[i].message) + '</div>';
      }
    }
    // Construction kind (binding.type — draw, lawof, call, …) is
    // intentionally omitted: the expression always starts with the
    // operator, and the DAG node's shape + color already encodes
    // the same axis. The inferred FlatPIR type/shape carries
    // strictly richer information (structural result type) and
    // takes that pill's slot.
    var inferTag = d.inferredType
      ? '<span class="infer">' + esc(d.inferredType) + '</span>'
      : '';
    document.getElementById('info').innerHTML =
      '<div class="row"><span class="name">' + esc(d.label) + '</span>'
      + phaseTag
      + inferTag + '</div>'
      + '<div class="expr">' + esc(d.expr) + '</div>'
      + unsupportedRow
      + errorRow;
  }

  function updateHeader(ctx, data) {
    var el = document.getElementById('header-expr');
    // Module view: no per-node target; just label the view.
    if (ctx.currentState && ctx.currentState.targetName === ctx.MODULE_TARGET) {
      el.innerHTML = '<span class="target-name">module</span>';
      return;
    }
    var target = null;
    for (var i = 0; i < data.nodes.length; i++) {
      if (data.nodes[i].isTarget) { target = data.nodes[i]; break; }
    }
    if (!target) { el.innerHTML = ''; return; }
    var name = target.label || target.id;
    var expr = truncateExpr(target.expr);
    el.innerHTML = '<span class="target-name">' + esc(name) + '</span>'
      + (expr ? '<span class="target-eq">=</span>' + esc(expr) : '');
  }

  function updateBackBtn(ctx) {
    document.getElementById('back-btn').style.display = ctx.history.length > 0 ? 'block' : 'none';
  }

  function makeActionButton(ctx, iconKey, title) {
    var b = document.createElement('button');
    b.type = 'button';
    b.title = title;
    b.setAttribute('aria-label', title);
    b.style.background = 'transparent';
    b.style.color = 'var(--vscode-foreground, #cccccc)';
    b.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
    b.style.borderRadius = '3px';
    b.style.padding = '2px 4px';
    b.style.display = 'inline-flex';
    b.style.alignItems = 'center';
    b.style.justifyContent = 'center';
    b.style.cursor = 'pointer';
    b.style.opacity = '0.75';
    b.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" '
      + 'xmlns="http://www.w3.org/2000/svg" fill="currentColor" '
      + 'aria-hidden="true"><path d="' + ctx.CODICON_PATHS[iconKey] + '"/></svg>';
    b.addEventListener('mouseenter', function() { b.style.opacity = '1'; });
    b.addEventListener('mouseleave', function() { b.style.opacity = '0.75'; });
    return b;
  }

  function domainBoundsText(kwargOrder, ranges, setNames) {
    var parts = [];
    for (var i = 0; i < kwargOrder.length; i++) {
      var k = kwargOrder[i];
      var r = ranges && ranges[k];
      if (r) {
        parts.push(k + ' ∈ [' + formatScalar(r.lo) + ', ' + formatScalar(r.hi) + ']');
      } else if (setNames && setNames[k]) {
        parts.push(k + ' ∈ ' + setNames[k]);
      }
    }
    return parts.length ? parts.join(', ') : '(no range)';
  }

  function formatScalarForSource(ctx, v) {
    if (typeof v === 'boolean') {
      // Boolean spelling follows the source-file variant: FlatPPL
      // and FlatPPJ use lowercase `true`/`false`; FlatPPY uses
      // capitalized `True`/`False`.
      if (ctx.currentVariantId === 'flatppy') return v ? 'True' : 'False';
      return v ? 'true' : 'false';
    }
    if (!Number.isFinite(v)) return String(v);
    return String(v);
  }

  function isPersistableSetField(v) {
    if (!v) return false;
    if (v.type === 'CallExpr' && v.callee && v.callee.name === 'interval'
        && Array.isArray(v.args) && v.args.length === 2
        && v.args[0].type === 'NumberLiteral'
        && v.args[1].type === 'NumberLiteral') return true;
    if (v.type === 'Identifier' && KNOWN_NAMED_SETS[v.name]) return true;
    return false;
  }

  function presetValuesText(values) {
    var text = formatValue(values);
    if (text.indexOf('record(') === 0 && text.charAt(text.length - 1) === ')') {
      return text.slice('record('.length, -1);
    }
    return text;
  }

  function defaultValueForLeafType(leafType) {
    if (!leafType) return 0;
    if (leafType.kind === 'scalar') {
      if (leafType.prim === 'integer') return 0;
      if (leafType.prim === 'boolean') return false;
      return 0;
    }
    return 0;
  }

  function defaultRangeForLeafType(leafType) {
    if (leafType && leafType.kind === 'scalar' && leafType.prim === 'integer') {
      return [-10, 10];
    }
    return [-5, 5];
  }

  function rangeFromSetDescriptor(descriptor) {
    if (!descriptor) return null;
    switch (descriptor.kind) {
      case 'interval':       return [descriptor.lo, descriptor.hi];
      case 'reals':          return [-5, 5];
      case 'posreals':       return [0.01, 5];
      case 'nonnegreals':    return [0, 5];
      case 'unitinterval':   return [0, 1];
      case 'integers':       return [-10, 10];
      case 'posintegers':    return [1, 20];
      case 'nonnegintegers': return [0, 20];
      case 'booleans':       return [0, 1];
      default:               return null;
    }
  }

  function resolveSweepRange(ctx, axis) {
    var descriptor = FlatPPLEngine.orchestrator.resolveAxisBaseSet(
      axis.source, ctx.derivationsState && ctx.derivationsState.bindings);
    if (descriptor && descriptor.kind === 'empirical') {
      return getMeasure(ctx, descriptor.name).then(function(m) {
        if (m && m.samples && m.samples.length > 0) {
          var range = FlatPPLEngine.orchestrator.fourSigmaQuantileRange(m.samples);
          if (range && range[0] < range[1]) return range;
        }
        return defaultRangeForLeafType(axis.leafType);
      }, function() {
        return defaultRangeForLeafType(axis.leafType);
      });
    }
    var fromDescriptor = rangeFromSetDescriptor(descriptor);
    if (fromDescriptor) return Promise.resolve(fromDescriptor);
    return Promise.resolve(defaultRangeForLeafType(axis.leafType));
  }

  function filterOverrideToAxes(override, axisKwargs, key) {
    if (!override) return null;
    var src = override[key] || {};
    var dst = {};
    var kept = false;
    for (var k in src) {
      if (!Object.prototype.hasOwnProperty.call(src, k)) continue;
      if (!axisKwargs.has(k)) continue;
      dst[k] = src[k];
      kept = true;
    }
    if (!kept) return null;
    var out = Object.assign({}, override);
    out[key] = dst;
    return out;
  }

  function applyRememberedSelections(ctx, plan) {
    if (!plan) return;
    var mem = ctx.planMemoryByName.get(plan.name);
    if (!mem) return;
    var axisKwargs = new Set();
    if (plan.axes) {
      for (var i = 0; i < plan.axes.length; i++) {
        if (plan.axes[i].kwargName) axisKwargs.add(plan.axes[i].kwargName);
      }
    }
    if (mem.sweepKey
        && plan.axes
        && plan.axes.some(function(a) { return a.key === mem.sweepKey; })) {
      plan.sweepKey = mem.sweepKey;
    }
    if (mem.outputKey
        && plan.outputs
        && plan.outputs.some(function(o) { return o.key === mem.outputKey; })) {
      plan.outputKey = mem.outputKey;
    }
    plan.autoOverride = filterOverrideToAxes(mem.autoOverride, axisKwargs, 'values');
    plan.domainAutoOverride = filterOverrideToAxes(mem.domainAutoOverride, axisKwargs, 'ranges');
    if (mem.presetName != null
        && plan.matchedPresets
        && plan.matchedPresets.some(function(p) { return p.name === mem.presetName; })) {
      plan.presetName = mem.presetName;
    }
    if (mem.domainName != null
        && plan.matchedDomains
        && plan.matchedDomains.some(function(d) { return d.name === mem.domainName; })) {
      plan.domainName = mem.domainName;
    }
  }

  function rememberPlanSelections(ctx, plan) {
    if (!plan || !plan.name) return;
    ctx.planMemoryByName.set(plan.name, {
      sweepKey: plan.sweepKey || null,
      outputKey: plan.outputKey || null,
      presetName: plan.presetName || null,
      domainName: plan.domainName || null,
      autoOverride: plan.autoOverride || null,
      domainAutoOverride: plan.domainAutoOverride || null,
    });
  }

  function teardownBubbles(ctx) {
    if (!ctx.bb) return;
    ctx.bb.getPaths().forEach(function(p) {
      p.update = function() {};
      ctx.bb.removePath(p);
    });
    ctx.cy.elements().forEach(function(el) { el.removeScratch('bubbleSets'); });
  }

  function bubbleMemberIds(r, allReifications) {
    var ids = {};
    for (var i = 0; i < r.kernel.length; i++) ids[r.kernel[i]] = true;
    for (var j = 0; j < allReifications.length; j++) {
      var r2 = allReifications[j];
      if (r2 === r || !ids[r2.name]) continue;
      for (var k = 0; k < r2.kernel.length; k++) ids[r2.kernel[k]] = true;
    }
    return ids;
  }


    FlatPPLViewer.mount = function mount(container, opts) {
      opts = opts || {};

      // Per-mount state container (decomposition Phase 2). Every
      // captured mutable/shared identifier migrates onto `ctx` so the
      // nested functions stop relying on lexical capture and can be
      // hoisted out of mount() (Phase 3) and split into ES modules
      // (Phase 4). `ctx` is declared at the very top of mount so it's
      // initialized BEFORE any `ctx.X = …` assignment in the prologue;
      // var-hoisting alone wouldn't suffice because the assignment
      // `ctx = {}` is what makes ctx an object, and the prologue's
      // first ctx-write (e.g. `ctx.host = …`) must see an object.
      var ctx = {};
      // container: the element the viewer renders inside. Defaults to
      // document.body for backward-compat with the existing VS Code
      // wrapper. The viewer injects its layout markup as innerHTML and
      // ensures its stylesheet is present on the page once.
      container = container || (typeof document !== 'undefined' ? document.body : null);
      if (!container) {
        throw new Error('FlatPPLViewer.mount: no container available (document missing?)');
      }
      ensureCssInjected();
      container.innerHTML = VIEWER_BODY_HTML;
      // Host adapter: IDE-only concerns the viewer delegates outward
      // (cross-pane source navigation, panel-title updates, persistent
      // UI state). Each method is optional; missing methods become
      // no-ops, so a standalone embed can pass {} or omit opts.host
      // entirely and the viewer renders fine — just without the
      // navigate-to-source / restore-state niceties.
      //
      //   revealSourceLine(line)  — host moves its source view's cursor
      //   setTitle(name)          — host sets the surrounding panel title
      //   saveState(state)        — host persists viewer state across reloads
      //   loadState()             — host returns previously-saved state
      //
      // Default: when no host is supplied AND acquireVsCodeApi exists
      // (we're inside a VS Code webview), build a default adapter that
      // bridges to VS Code's postMessage / setState / getState. This
      // keeps the existing extension wrapper working without any
      // host-side changes.
      ctx.host = opts.host || defaultVscodeHost();

      // Host-supplied configuration. The vscode-extension host writes
      // window.__FLATPPL_CONFIG__ via a small inline bootstrap <script>
      // before this file loads. For a standalone embed (no VS Code), an
      // online host can do the same — set the config object before
      // including viewer.js. Currently expected fields:
      //   samplerWorkerUrl: string  — URL of the sampler-worker bundle,
      //                                loaded as a Web Worker.
      ctx.CONFIG = (typeof window !== 'undefined' && window.__FLATPPL_CONFIG__) || {};
      ctx.HINT = 'Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source';
      // Sampler-worker URL. Used lazily — no worker is spawned until the
      // user picks a binding for which the Plot tab is enabled (a 'draw'
      // of a known distribution with literal params).
      ctx.SAMPLER_WORKER_URL = ctx.CONFIG.samplerWorkerUrl || '';


    // ---- Palette ----
    //
    // Single source of truth for every node / edge / bubble colour the
    // visualizer uses. PHASE_COLORS, TYPE_STYLE, DRAW_EDGE_COLOR, and
    // the #info .phase-* CSS rules all reference these names — change a
    // hex here and every consumer follows.
    //
    // Naming reflects what the colour *means*, not where it shows up:
    //   phaseStochastic / parameterized / fixed
    //                          — value-producing nodes (draw, call) and
    //                            their #info phase tags
    //   measure                — lawof bindings + measure-kind reifications
    //   kernel                 — kernelof bindings + kernel-kind reifications
    //   fn                     — functionof / fn bindings
    //   literal/…/unknown      — purely structural type colours
    //   drawEdge               — the "draw" arrow (deterministic →
    //                            stochastic boundary)
    //
    // Hue strategy: lawof/kernelof/functionof form an additive triple
    // (blue + green ≈ teal) so family relationships read visually,
    // viridis-style and colourblind-safe. The phase trio reuses the
    // historical draw/input/call hex values so the visual story stays
    // familiar after the shift to phase-driven colouring.
    ctx.PALETTE = {
      phaseStochastic:    '#B39DDB',  // purple
      phaseParameterized: '#4DD0E1',  // teal
      phaseFixed:         '#90A4AE',  // blue-grey
      measure:            '#42A5F5',  // bright blue
      kernel:             '#26A69A',  // teal-green
      fn:                 '#66BB6A',  // green
      // Note: no `literal` entry — literal bindings are just fixed-
      // phase values, semantically the same kind as `call` bindings,
      // and reuse `phaseFixed` for color. Shape (rectangle vs
      // round-rectangle) carries the surface-form distinction.
      // Using a dedicated red/pink for literals overstated their
      // status and conflicted with red's conventional warning role
      // in dev UIs.
      likelihood:         '#EF9A9A',  // light red
      bayesupdate:        '#FFAB91',  // light orange
      module:             '#80CBC4',  // teal-green (lighter)
      table:              '#A1887F',  // brown
      unknown:            '#BDBDBD',  // grey
      drawEdge:           '#7E57C2',  // darker purple than phaseStochastic
    };

    // Mirror the phase colours into CSS custom properties so the
    // #info .phase-* tag rules pick them up without a duplicate hex
    // literal in the stylesheet.
    (function bindPaletteToCss() {
      var s = document.documentElement.style;
      s.setProperty('--phase-stochastic',    ctx.PALETTE.phaseStochastic);
      s.setProperty('--phase-parameterized', ctx.PALETTE.phaseParameterized);
      s.setProperty('--phase-fixed',         ctx.PALETTE.phaseFixed);
    })();

    // Phase → fill colour for value-producing nodes (draw / call /
    // computed values inside a kernel scope). Used by both the DAG
    // renderer and the legend.
    ctx.PHASE_COLORS = {
      stochastic:    ctx.PALETTE.phaseStochastic,
      parameterized: ctx.PALETTE.phaseParameterized,
      fixed:         ctx.PALETTE.phaseFixed,
    };

    // Stand-alone for the "draw" edge — visually distinct from any
    // node fill so a stochastic boundary reads as an edge, not a fill.
    ctx.DRAW_EDGE_COLOR = ctx.PALETTE.drawEdge;

    // Type → { color, shape, legend label }. The phase trio (input /
    // draw / call) intentionally reuses PALETTE.phase* so a
    // value-producing node falls back to the matching phase colour
    // when phase metadata is missing.
    ctx.TYPE_STYLE = {
      input:       { color: ctx.PALETTE.phaseParameterized, shape: 'diamond',         label: 'input (elementof)' },
      draw:        { color: ctx.PALETTE.phaseStochastic,    shape: 'ellipse',         label: 'draw' },
      call:        { color: ctx.PALETTE.phaseFixed,         shape: 'round-rectangle', label: 'call' },
      lawof:       { color: ctx.PALETTE.measure,            shape: 'round-rectangle', label: 'lawof (measure)' },
      kernelof:    { color: ctx.PALETTE.kernel,             shape: 'round-hexagon',   label: 'kernelof (kernel)' },
      functionof:  { color: ctx.PALETTE.fn,                 shape: 'hexagon',         label: 'functionof' },
      fn:          { color: ctx.PALETTE.fn,                 shape: 'hexagon',         label: 'fn' },
      literal:     { color: ctx.PALETTE.phaseFixed,         shape: 'rectangle',       label: 'literal' },
      likelihood:  { color: ctx.PALETTE.likelihood,         shape: 'octagon',         label: 'likelihood' },
      bayesupdate: { color: ctx.PALETTE.bayesupdate,        shape: 'octagon',         label: 'bayesupdate' },
      module:      { color: ctx.PALETTE.module,             shape: 'round-rectangle', label: 'module' },
      table:       { color: ctx.PALETTE.table,              shape: 'round-rectangle', label: 'table' },
      unknown:     { color: ctx.PALETTE.unknown,            shape: 'rectangle',       label: 'unknown' },
    };


    // G1 DAG/state (decomposition Phase 2 — on ctx).
    ctx.cy = null;
    ctx.bb = null;
    ctx.history = [];
    ctx.currentState = null;
    // Bound on the DAG-navigation history (back-button stack). Cheap
    // insurance against pathological growth (a runaway extension or
    // rapid-fire navigation). Each entry is a sub-DAG's data plus a
    // name string, so a few hundred is plenty without thinking about
    // memory. Owned by the host setting flatppl.visualization.
    // dagNavigationHistoryCap (default 1000); the host pushes its
    // value via configUpdate alongside sampleCount.
    ctx.HISTORY_CAP = 1000;

    // VS Code codicons — `discard`, `save`, `save-as` paths copied
    // verbatim from microsoft/vscode-codicons (src/icons/*.svg) so the
    // viewer's reset/persist buttons match VS Code's own toolbar
    // language without pulling in a font dependency. SVGs use
    // fill="currentColor" so they inherit the button's text color.
    //
    // © Microsoft Corporation. Licensed under CC BY 4.0
    // (https://creativecommons.org/licenses/by/4.0/). See
    // packages/viewer/NOTICE.md for full attribution.
    ctx.CODICON_PATHS = {
      discard: 'M3.00098 2.5C3.00098 2.22386 3.22483 2 3.50098 2C3.77712 2 4.00098 2.22386 4.00098 2.5V6.34262L7.17202 3.17157C8.73412 1.60948 11.2668 1.60948 12.8289 3.17157C14.391 4.73367 14.391 7.26633 12.8289 8.82843L7.80375 13.8536C7.60849 14.0488 7.2919 14.0488 7.09664 13.8536C6.90138 13.6583 6.90138 13.3417 7.09664 13.1464L12.1218 8.12132C13.2933 6.94975 13.2933 5.05025 12.1218 3.87868C10.9502 2.70711 9.0507 2.70711 7.87913 3.87868L4.75781 7H8.50098C8.77712 7 9.00098 7.22386 9.00098 7.5C9.00098 7.77614 8.77712 8 8.50098 8H3.60098C3.26961 8 3.00098 7.73137 3.00098 7.4V2.5Z',
      save: 'M14.414 3.207L12.793 1.586C12.421 1.213 11.905 1 11.379 1H3C1.897 1 1 1.897 1 3V13C1 14.103 1.897 15 3 15H13C14.103 15 15 14.103 15 13V4.621C15 4.095 14.787 3.579 14.414 3.207ZM9 2V3.5C9 3.776 8.776 4 8.5 4H6.5C6.224 4 6 3.776 6 3.5V2H9ZM5 14V9.5C5 9.224 5.224 9 5.5 9H10.5C10.776 9 11 9.224 11 9.5V14H5ZM14 13C14 13.551 13.551 14 13 14H12V9.5C12 8.673 11.327 8 10.5 8H5.5C4.673 8 4 8.673 4 9.5V14H3C2.449 14 2 13.551 2 13V3C2 2.449 2.449 2 3 2H5V3.5C5 4.327 5.673 5 6.5 5H8.5C9.327 5 10 4.327 10 3.5V2H11.379C11.642 2 11.9 2.107 12.086 2.293L13.707 3.914C13.893 4.1 14 4.358 14 4.621V13Z',
      'save-as': 'M5 9.5C5 9.224 5.224 9 5.5 9H10.5C10.738 9 10.929 9.171 10.979 9.394L11.729 8.644C11.458 8.256 11.009 8 10.5 8H5.5C4.673 8 4 8.673 4 9.5V14H3C2.449 14 2 13.551 2 13V3C2 2.449 2.449 2 3 2H5V3.5C5 4.327 5.673 5 6.5 5H8.5C9.327 5 10 4.327 10 3.5V2H11.379C11.642 2 11.9 2.107 12.086 2.293L13.707 3.914C13.893 4.1 14 4.358 14 4.621V7.04C14.143 7.015 14.289 6.997 14.437 6.997C14.629 6.997 14.817 7.023 15 7.064V4.62C15 4.094 14.787 3.578 14.414 3.206L12.793 1.585C12.421 1.212 11.905 0.999001 11.379 0.999001H3C1.897 1 1 1.897 1 3V13C1 14.103 1.897 15 3 15H7.045L7.293 14H5V9.5ZM6 2H9V3.5C9 3.776 8.776 4 8.5 4H6.5C6.224 4 6 3.776 6 3.5V2ZM16 9.559C16 9.764 15.96 9.967 15.882 10.157C15.803 10.346 15.688 10.519 15.543 10.664L11.254 14.951C10.898 15.307 10.452 15.56 9.964 15.682L8.753 15.982C8.651 16.008 8.544 16.006 8.443 15.978C8.342 15.95 8.249 15.896 8.175 15.822C8.101 15.748 8.047 15.655 8.019 15.554C7.991 15.453 7.99 15.346 8.015 15.244L8.315 14.033C8.437 13.544 8.689 13.098 9.045 12.742L13.333 8.455C13.626 8.162 14.023 7.998 14.437 7.998C14.851 7.998 15.248 8.163 15.541 8.455C15.687 8.599 15.802 8.772 15.881 8.961C15.96 9.151 16 9.354 16 9.559Z',
    };




    // Sentinel name for the module-overview state. Distinct from any
    // user binding name (binding identifiers are barewords; the
    // sentinel uses ':' which the analyzer can't produce). Used by
    // updateHeader, updatePlotForBinding, and the back-button to
    // distinguish module view from a single-binding view.
    ctx.MODULE_TARGET = ':module';



    function initCy() {
      ctx.cy = cytoscape({
        container: document.getElementById('cy'),
        style: [
          {
            selector: 'node',
            style: {
              'label': 'data(label)',
              'text-valign': 'center',
              'text-halign': 'center',
              'font-size': '13px',
              'color': '#333',
              'background-color': 'data(color)',
              'shape': 'data(shape)',
              'width': 'data(width)',
              'height': 36,
              'border-width': 2,
              'border-color': '#888',
            }
          },
          {
            // Reification anchor nodes — bindings that head a
            // reification group (lawof / functionof / kernelof / fn
            // with internal kernel members). They sit at the entrance
            // of their bubble; the translucent fill + same-color
            // border read as belonging to the bubble rather than
            // floating inside it.
            //
            // Selecting on the engine-computed isReifAnchor flag
            // (rather than nodeType alone) excludes synthesized
            // measure bindings that happen to have type=lawof but no
            // visible bubble (e.g. prior2 = lawof(disintegrate(…))
            // where disintegrate produces a closed-form rewrite, no
            // new scope to render). Those fall through to the default
            // solid fill — same visual treatment as joint_model and
            // other measure-producing operations without a bubble.
            selector: 'node[?isReifAnchor]',
            style: {
              'background-color': 'data(color)',
              'background-opacity': 0.18,
              'border-color': 'data(color)',
              'border-width': 1.5,
              'color': 'data(color)',
            }
          },
          {
            selector: 'node[?isBoundary]',
            style: {
              'border-color': '#FFD600',
              'border-width': 3,
              'border-style': 'dashed',
            }
          },
          {
            // Disintegration result whose Plan came back Unsupported —
            // the trace through it is the user's literal source, not a
            // structural decomposition. Dotted orange border distinguishes
            // it from boundary inputs (dashed yellow) and target (solid blue).
            selector: 'node[?unsupported]',
            style: {
              'border-color': '#FF8A65',
              'border-width': 3,
              'border-style': 'dotted',
            }
          },
          {
            // Bindings with analyzer-level error diagnostics (typeinfer
            // mismatch, undefined ref, etc.) get a solid red border.
            // Distinct from the dashed yellow boundary and dotted orange
            // unsupported markers so the three semantic signals don't
            // collide visually.
            selector: 'node[?hasError]',
            style: {
              'border-color': '#E57373',
              'border-width': 3,
              'border-style': 'solid',
            }
          },
          {
            selector: 'node[?isTarget]',
            style: {
              'border-color': '#1565C0',
              'border-width': 4,
            }
          },
          {
            selector: 'edge',
            style: {
              'width': 2,
              'line-color': '#999',
              'target-arrow-color': '#999',
              'target-arrow-shape': 'triangle',
              'curve-style': 'bezier',
              'arrow-scale': 1.0,
            }
          },
          {
            selector: 'edge[edgeType = "call"]',
            style: {
              'line-style': 'dashed',
              'line-dash-pattern': [6, 4],
              'line-color': '#bbb',
              'target-arrow-color': '#bbb',
              'width': 1.5,
            }
          },
          {
            // Draw edges: the boundary between deterministic and
            // stochastic. Solid line in a darker purple than the
            // node fill so it reads boldly as a line; thicker than
            // dataflow edges so the eye lands on where stochasticity
            // enters the model.
            selector: 'edge[edgeType = "draw"]',
            style: {
              'line-color': ctx.DRAW_EDGE_COLOR,
              'target-arrow-color': ctx.DRAW_EDGE_COLOR,
              'width': 2.5,
            }
          },
          {
            // Hidden edges — present so dagre uses them for layout, but
            // not rendered (the enclosing bubble conveys the relation).
            selector: 'edge[?hidden]',
            style: {
              'visibility': 'hidden',
            }
          },
          {
            // Tether: faint connection from a reified value to its
            // reification node. Same kernel-internal flow as the hidden
            // edges, but drawn so you can see what is being reified.
            // Labeled with the reification keyword (lawof / functionof /
            // kernelof / fn) so the operation is legible without having
            // to read the target node.
            selector: 'edge[edgeType = "tether"]',
            style: {
              'line-color': function(ele) { return ele.target().data('color') || '#aaa'; },
              'opacity': 0.6,
              'width': 1.5,
              'target-arrow-shape': 'none',
              'curve-style': 'straight',
              'label': 'data(tetherLabel)',
              'font-size': '10px',
              'font-style': 'italic',
              'color': function(ele) { return ele.target().data('color') || '#aaa'; },
              // Full text opacity overrides the edge's 0.6 — the line stays
              // faint, the label reads as bright as a node label.
              'text-opacity': 1,
              // Center the label on the line and let an opaque background
              // pad visually break the line at the label — the tether
              // appears to connect into the lawof/kernelof/… box on both
              // sides, like a labeled link in an electrical schematic.
              // Literal hex (not a CSS var) — cytoscape draws on HTML canvas
              // and cannot resolve "var(--name)" values, so a CSS variable
              // would silently fall back to a transparent background and
              // let the line show through.
              'text-rotation': 'autorotate',
              'text-background-color': '#1e1e1e',
              'text-background-opacity': 1,
              'text-background-padding': '2px',
              'text-background-shape': 'roundrectangle',
              'text-border-width': 1,
              'text-border-color': function(ele) { return ele.target().data('color') || '#aaa'; },
              'text-border-opacity': 0.6,
            }
          },
          {
            selector: 'node:selected',
            style: {
              'border-color': '#2196F3',
              'border-width': 3,
              'overlay-opacity': 0,
            }
          },
        ],
        elements: [],
        layout: { name: 'preset' },
        wheelSensitivity: 2,
      });

      if (typeof ctx.cy.bubbleSets === 'function') {
        // bubblesets uses one scratch key per cytoscape node; when paths
        // share nodes (e.g. theta1 belongs to both prior and forward_kernel),
        // their cached geometry stomps on each other and one path goes empty
        // on update. Workaround: tear down and rebuild all paths on drag
        // release, rAF-batched. Updates skipped during drag for snappiness.
        ctx.bb = ctx.cy.bubbleSets({ interactive: false });
        var bbRedrawScheduled = false;
        ctx.cy.on('free', 'node', function() {
          if (!ctx.bb || bbRedrawScheduled || !ctx.currentState) return;
          bbRedrawScheduled = true;
          requestAnimationFrame(function() {
            bbRedrawScheduled = false;
            if (ctx.currentState) drawReificationLassos(ctx.currentState.data);
          });
        });
      }

      // Ctrl/Cmd+click: jump to source.
      // Plain click: select the node — info bar updates AND the plot
      // panel re-targets to this binding. The plot follows the
      // selection rather than the DAG's terminal target so users can
      // explore the graph node-by-node and read each binding's
      // distribution in place.
      ctx.cy.on('tap', 'node', function(evt) {
        var oe = evt.originalEvent;
        if (oe && (oe.ctrlKey || oe.metaKey)) {
          var line = evt.target.data('line');
          if (line >= 0) {
            if (ctx.host.revealSourceLine) ctx.host.revealSourceLine(line);
          }
          return;
        }
        var d = evt.target.data();
        showNodeInfo(d);
        // Always re-target the plot to whatever the user clicked. For
        // synthetic nodes (anonymous inline expressions, placeholders,
        // holes — recognised by ':' in the id) there's no binding to
        // sample, so updatePlotForBinding ends up rendering a
        // "Not plottable" placeholder. Either way the plot reflects
        // the current selection rather than a stale earlier focus.
        updatePlotForBinding(d.id);
      });

      ctx.cy.on('tap', function(evt) {
        if (evt.target === ctx.cy) {
          document.getElementById('info').innerHTML = '<span class="hint">' + ctx.HINT + '</span>';
        }
      });

      // Double-click: drill into node's sub-DAG. Handled locally — the
      // webview owns the parsed bindings and recomputes the sub-DAG itself
      // (no host round-trip). Title sync to the editor still goes via a
      // postMessage to the host since the title is on the VS Code panel.
      ctx.cy.on('dbltap', 'node', function(evt) {
        var nodeId = evt.target.data('id');
        // Don't drill into synthetic nodes (placeholder/hole inputs).
        if (nodeId.indexOf(':') !== -1) return;
        focusNode(nodeId, /* pushHistory */ true);
        if (ctx.host.setTitle) ctx.host.setTitle(nodeId);
      });

      var tip = document.getElementById('tooltip');
      ctx.cy.on('mouseover', 'node', function(evt) {
        var d = evt.target.data();
        var expr = d.expr || '';
        if (!expr) return;
        tip.textContent = d.label ? (d.label + ' = ' + expr) : expr;
        tip.style.display = 'block';
        var pos = evt.renderedPosition;
        var cRect = document.getElementById('cy').getBoundingClientRect();
        var tx = pos.x + cRect.left + 12;
        var ty = pos.y + cRect.top - 30;
        if (tx + tip.offsetWidth > cRect.right - 8) tx = cRect.right - tip.offsetWidth - 8;
        if (ty < cRect.top + 4) ty = pos.y + cRect.top + 16;
        tip.style.left = tx + 'px';
        tip.style.top = ty + 'px';
      });
      ctx.cy.on('mouseout', 'node', function() {
        tip.style.display = 'none';
      });
      ctx.cy.on('viewport', function() {
        tip.style.display = 'none';
      });
    }

    // ---------------------------------------------------------------
    // Plot panel — density / sample histograms via the sampler-worker.
    //
    // The Plot tab shows the analytical density of the currently focused
    // binding when that binding is a 'draw' of a known distribution with
    // literal parameters (so the worker doesn't need to dependency-walk
    // upstream randoms — that's the orchestrator's job, deferred to a
    // later iteration). When the binding isn't plottable, the Plot tab
    // is shown disabled.
    //
    // The sampler-worker is spawned lazily on first plot request so the
    // ~1 MB worker bundle (stdlib + sampler) doesn't load for users who
    // never open the Plot tab. We keep the worker alive across focus
    // changes; only its 'setSeed' / 'init' is replayed on demand. A
    // request-id counter pairs replies to outstanding promises so we
    // can multiplex multiple in-flight requests cleanly.
    // ---------------------------------------------------------------

    // G3 worker state (decomposition Phase 2 — on ctx).
    ctx.samplerWorker = null;
    ctx.samplerWorkerPromise = null;   // Promise<Worker> while spawn is in-flight
    ctx.samplerWorkerError = null;     // last spawn error, surfaced in the UI
    ctx.samplerReqId = 0;
    ctx.pendingRequests = new Map(); // id → { resolve, reject }
    // G2 plot control (decomposition Phase 2 — on ctx). Decls are
    // scattered across mount() — the other three live further down
    // near their first use; collected onto ctx incrementally.
    ctx.plotEchart = null;

    // ---------------------------------------------------------------
    // Main-thread empirical-measure cache.
    //
    // The cache holds an EmpiricalMeasure per binding:
    //   { samples:    Float64Array,            // the atom values
    //     logWeights: Float64Array | null }    // null = uniform 1/N
    //
    // Why a measure (not just samples)? When we add weighted,
    // bayesupdate, and superpose, we'll need per-atom weights to
    // represent the result correctly. Storing the structure now, even
    // with logWeights always null, lets those operations land later
    // without churning every consumer. For unweighted measures the
    // null-uniform convention costs nothing — logSumExp(null) = 0,
    // so total mass = 1 (probability measure), and histograms take
    // the simple count/N path.
    //
    // Why main-thread cache (not worker-side)?
    //   - Survives worker recycles (the user's Stop button terminates
    //     the worker; the cache stays valid).
    //   - Variates and their underlying measures share the SAME
    //     EmpiricalMeasure object (same samples, same logWeights) —
    //     theta1's measure IS theta1_dist's measure, by reference.
    //   - Click-around the DAG hits the cache → instant re-render.
    //   - Source edits invalidate everything by clearing the map.
    //
    // Per-binding seeding: we derive a deterministic seed from a
    // string hash of the binding name XOR'd with a root seed. Two
    // independent random variables (theta1_dist, theta2_dist) thus
    // get statistically independent streams without coupling to the
    // order of materialisation. A future "Resample" button can bump
    // rootSeed and clear the cache to redraw everything.
    // ---------------------------------------------------------------
    ctx.derivationsState = null;       // { derivations, discrete } from orchestrator
    ctx.measureCache = new Map();      // Map<name, EmpiricalMeasure>
    // Per-binding histogram cache. Histogram computation is O(N) and
    // for N=1M takes a noticeable few ms; caching keeps click-flipping
    // between previously-viewed bindings instant. Invalidated together
    // with measureCache (source change, configUpdate). Key includes
    // the discrete flag so the same name plotted discrete vs. continuous
    // gets distinct cache entries (defensive — discreteness is fixed
    // per binding today but the door's open for future modes).
    ctx.histogramCache = new Map();    // Map<"name|d"|"name|c", histogram>
    // Profile-plot per-axis range cache. Keyed by
    // "binding|sweepKey|presetName" so each (function, axis,
    // preset) combination remembers the user's x-axis edits across
    // navigation. Invalidated alongside measureCache /
    // histogramCache on source / sample-count changes.
    //   Map<key, { lo, hi, fromAuto: boolean }>
    // fromAuto distinguishes ranges initially populated by
    // resolveSweepRange (auto) vs. user-edited (override) — used
    // for tooltip / debug; the renderer treats both the same.
    ctx.profileRangeCache = new Map();
    // Module-wide overrides on named preset (record-point) bindings.
    // Persists across binding navigation, so tuning pars1 on a
    // likelihood plot applies the same overrides when the user visits
    // a forward kernel that shares those kwarg names. Reconciled on
    // every source change via value comparison against the freshly
    // parsed base values (see rebuildDerivations); a kwarg whose
    // source value now matches the override drops from the override
    // automatically.
    //   Map<presetName, { values: { kwargName → number } }>
    ctx.presetOverrides = new Map();
    // Module-wide overrides on named preset-domain bindings (cartprod
    // forms). Same lifetime/reconciliation pattern as presetOverrides:
    // persists across binding navigation, prunes per kwarg against
    // current source values, drops the entry when the source binding
    // is gone.
    //   Map<domainName, { ranges }>
    // ranges: { kwargName → { lo, hi } }   user-set range overrides
    ctx.domainOverrides = new Map();
    ctx.rootSeed = 1;
    // Sample budget for chain-based plots. Higher → smoother histograms,
    // marginal cost grows linearly. Tuned for sub-100ms response.
    // Sample budget per binding when the visualizer renders a histogram.
    // Owned by VS Code's configuration (flatppl.visualization.sampleCount,
    // default 100000, max 10_000_000); the host pushes it via a
    // configUpdate message and updates it on settings changes. Value
    // here is just an in-flight default until the first configUpdate
    // arrives — the panel always boots with a config push from the host.
    ctx.SAMPLE_COUNT = 100000;

    // Per-atom rejection budget for matTruncate's rejection-redraw path
    // (spec §06 truncate). When the parent measure isn't CDF-invertible,
    // the worker redraws from the underlying distribution up to this
    // many times per atom before giving up and emitting NaN. Higher
    // values trade compute for less ESS loss on tightly-truncated
    // measures; lower values keep large-N plots responsive. The host
    // pushes a new value through configUpdate when the user changes
    // the corresponding setting (VS Code: flatppl.truncate.rejectionBudget).
    ctx.REJECTION_BUDGET = 1000;

    function rebuildDerivations() {
      if (!ctx.currentBindings) {
        ctx.derivationsState = null;
        ctx.measureCache = new Map();
        ctx.histogramCache = new Map();
        ctx.profileRangeCache = new Map();
        return;
      }
      try {
        ctx.derivationsState = FlatPPLEngine.orchestrator.buildDerivations(ctx.currentBindings);
        // Surface classification diagnostics instead of letting a
        // silently-dropped binding turn into a confusing plot-time
        // error far from its cause. buildDerivations only emits the
        // unambiguous fixed-phase-dead-end case (an engine gap on a
        // deterministic expression), so any entry here is actionable.
        var bdiags = (ctx.derivationsState && ctx.derivationsState.diagnostics) || [];
        for (var bi = 0; bi < bdiags.length; bi++) {
          console.warn('FlatPPL: ' + bdiags[bi].message);
        }
      } catch (e) {
        console.error('FlatPPL: buildDerivations failed:', e);
        ctx.derivationsState = null;
      }
      // Source change invalidates every cached measure — derivations
      // (or just signatures) may have shifted under any of them. Drop
      // the histogram cache too since histograms are downstream of
      // measures.
      ctx.measureCache = new Map();
      ctx.histogramCache = new Map();
      ctx.profileRangeCache = new Map();

      // Reconcile module-wide preset overrides against the new
      // source. For each existing override:
      //   - If the preset binding is gone, drop the override.
      //   - For each kwarg in override.values, if the new source
      //     value matches the override (or the kwarg is gone),
      //     drop it from the override (matched → redundant,
      //     gone → not applicable). Persist-button writes lean on
      //     this — after writing the override into the source,
      //     the next rebuildDerivations finds equal values and
      //     prunes the override automatically.
      //   - If the override is empty after pruning, retire it.
      ctx.presetOverrides.forEach(function(entry, name) {
        var b = ctx.currentBindings.get(name);
        var curValues = null;
        // analyzer.classifyStatement only returns 'literal' for
        // primitive literal RHS (NumberLiteral etc.); record(...)
        // binds with type='call'. So gate on the AST callee name
        // alone — that's the structural property we actually need.
        if (b && b.node && b.node.value
            && b.node.value.type === 'CallExpr' && b.node.value.callee
            && b.node.value.callee.name === 'record') {
          // Best-effort literal extraction; bail to no-match if
          // anything looks non-trivial (which falls into the
          // "kwarg unknown" branch below). A NumberLiteral wrapped
          // in fixed(...) (spec §03: "held constant during
          // optimization" hint) is identity at runtime, so we look
          // through the wrapper to read the underlying number —
          // otherwise persist-time reconciliation would not see the
          // override's value matching source and would fail to prune.
          curValues = {};
          var args = b.node.value.args || [];
          for (var ai = 0; ai < args.length; ai++) {
            var arg = args[ai];
            if (arg.type !== 'KeywordArg' || !arg.value) continue;
            var v = arg.value;
            if (v.type === 'CallExpr' && v.callee && v.callee.name === 'fixed'
                && Array.isArray(v.args) && v.args.length === 1) {
              v = v.args[0];
            }
            if (v && v.type === 'NumberLiteral') {
              curValues[arg.name] = v.value;
            }
          }
        }
        if (!curValues) {
          ctx.presetOverrides.delete(name);
          return;
        }
        var vs = entry.values || {};
        for (var k in vs) {
          if (!Object.prototype.hasOwnProperty.call(vs, k)) continue;
          if (!Object.prototype.hasOwnProperty.call(curValues, k)) {
            delete vs[k];                          // kwarg gone
          } else if (vs[k] === curValues[k]) {
            delete vs[k];                          // override matches source
          }
        }
        if (Object.keys(vs).length === 0) {
          ctx.presetOverrides.delete(name);
        }
      });

      // Reconcile module-wide domain overrides against the new
      // source. Mirrors the preset-override loop above:
      //   - If the cartprod binding is gone, drop the override.
      //   - sourceKwargs tracks every kwarg the source's cartprod
      //     mentions, regardless of whether its value is an interval
      //     or a bare named set; an override on a kwarg the source
      //     doesn't mention at all is "kwarg gone" → drop it.
      //   - sourceIntervals[k] is set only when the source field is
      //     interval(lit, lit). When set and the override's [lo,hi]
      //     equals it, the override is redundant → drop it. When the
      //     source field is a bare named set (no bounds) we leave
      //     the override in place: the user's range overrides the
      //     unbounded source.
      //   - If the override is empty after pruning, retire it.
      ctx.domainOverrides.forEach(function(entry, name) {
        var b = ctx.currentBindings.get(name);
        var sourceKwargs = null;
        var sourceIntervals = null;
        if (b && b.node && b.node.value
            && b.node.value.type === 'CallExpr' && b.node.value.callee
            && b.node.value.callee.name === 'cartprod') {
          sourceKwargs = new Set();
          sourceIntervals = {};
          var args = b.node.value.args || [];
          for (var ai = 0; ai < args.length; ai++) {
            var arg = args[ai];
            if (arg.type !== 'KeywordArg' || !arg.value) continue;
            sourceKwargs.add(arg.name);
            var ic = arg.value;
            if (ic.type === 'CallExpr' && ic.callee
                && ic.callee.name === 'interval'
                && Array.isArray(ic.args) && ic.args.length === 2
                && ic.args[0].type === 'NumberLiteral'
                && ic.args[1].type === 'NumberLiteral') {
              sourceIntervals[arg.name] = {
                lo: ic.args[0].value, hi: ic.args[1].value,
              };
            }
          }
        }
        if (!sourceKwargs) {
          ctx.domainOverrides.delete(name);
          return;
        }
        var rs = entry.ranges || {};
        for (var k in rs) {
          if (!Object.prototype.hasOwnProperty.call(rs, k)) continue;
          if (!sourceKwargs.has(k)) {
            delete rs[k];                                // kwarg gone
          } else if (Object.prototype.hasOwnProperty.call(sourceIntervals, k)
                     && rs[k].lo === sourceIntervals[k].lo
                     && rs[k].hi === sourceIntervals[k].hi) {
            delete rs[k];                                // matches source interval
          }
          // Otherwise: source uses a bare named set, the override's
          // explicit range still adds information — keep it.
        }
        if (Object.keys(rs).length === 0) {
          ctx.domainOverrides.delete(name);
        }
      });

      // Push fixed-phase pre-evaluated values into the worker's
      // session env. The orchestrator computed these once at module-
      // build time (rnginit / rand results, fixed scalar reductions,
      // etc.); the worker resolves refs to them via env rather than
      // through per-atom refArrays — the only correct semantics for
      // non-scalar fixed values like a length-10 `random_data` array.
      // setEnv with merge=false replaces (so a stale fixedValues map
      // from the previous source can't leak into the new one).
      if (ctx.derivationsState && ctx.derivationsState.fixedValues
          && ctx.derivationsState.fixedValues.size > 0) {
        var envObj = {};
        ctx.derivationsState.fixedValues.forEach(function(v, k) { envObj[k] = v; });
        ensureSamplerWorker(ctx).then(function(w) {
          sendWorkerNow(ctx, w, { type: 'setEnv', env: envObj, merge: false });
        }).catch(function(err) {
          console.error('FlatPPL: setEnv push failed:', err);
        });
      }
    }



    /**
     * Recursively materialise the empirical measure for a binding,
     * reusing cache entries for any deps already computed.
     * Returns Promise<EmpiricalMeasure>.
     *
     * Aliases share the SAME EmpiricalMeasure object (same samples
     * array, same logWeights ref) so click-flipping between a variate
     * and its measure is free. With null-uniform logWeights the cache
     * is purely additive over today's behaviour — no extra allocation.
     */

    /** Soft-fail variant of getMeasure: resolves to null instead of
        rejecting when the binding can't produce a measure (no
        derivation, no fixed value — typically pure inputs like
        `elementof(reals)`). Used by plot paths that want to chase
        sample-derived defaults for every source-binding axis but
        shouldn't blow up on the ones that genuinely have no samples
        to chase. */

    // Plot-plan fallbacks below still need the helper as a local
    // function reference; expose the engine copies under their old
    // names so the rest of viewer.js keeps working without further
    // edits.
    // Current plot plan from buildPlotPlan(). Two shapes:
    //   { mode: 'analytical', ir }
    //   { mode: 'chain', chain, discrete }
    // Used both as the "is plot tab enabled?" flag and as the render
    // input. currentPlotBindingName tracks which binding produced it
    // (for the chart title and stale-reply guards).
    ctx.currentPlotPlan = null;
    ctx.currentPlotBindingName = null;



    // Fire-and-forget message send (used during init when we don't care
    // about the reply). Distinct from sendWorker so we don't allocate a
    // pending-request entry for messages whose reply is just an 'ok'.


    /**
     * Build a plot plan for a binding. The orchestrator decides
     * sample-ability and returns a topo-ordered chain. Here we
     * additionally decide whether the analytical PDF/pmf curve should
     * accompany the histogram.
     *
     * Two semantic rules govern the density overlay:
     *   1. A density belongs to a *measure*, not to a variate. A
     *      stochastic binding like 'theta1 = draw(theta1_dist)' is a
     *      single drawn value — its samples have an empirical
     *      distribution, but the density itself is a property of the
     *      law theta1_dist. So variates (binding.type === 'draw')
     *      get samples-only; their underlying measure is where the
     *      density curve lives.
     *   2. A measure binding shows the analytical density only when
     *      the leaf, after alias resolution, has all-literal kwargs.
     *      A measure with stochastic parents has no closed-form
     *      marginal density; we'd need numerical marginalisation,
     *      which is more honest as a histogram.
     *
     * Returns:
     *   { chain, discrete, analyticalIR? }   — plottable
     *   null                                 — not plottable
     */

    function buildPlotPlan(binding /*, bindingsMap */) {
      if (!binding || !ctx.derivationsState) return null;
      var name = binding.name;

      // Callable bindings (function / kernel / fn / likelihood) don't
      // get a derivation kind — they're functions, not random
      // variables. They take the profile-plot path: sweep one input
      // axis, hold the rest fixed, evaluate the body per point. The
      // engine's signatureOf + distributeAxes (orchestrator.js) shape
      // the input cartprod / cartpow into a flat list of scalar
      // axes; the UI layer here picks the default sweep axis +
      // default range and dispatches to worker.profileN.
      if (binding.type === 'functionof' || binding.type === 'fn'
          || binding.type === 'kernelof' || binding.type === 'likelihood') {
        if (!ctx.derivationsState.bindings) return null;
        var sig = FlatPPLEngine.orchestrator.signatureOf(name, ctx.derivationsState.bindings);
        if (!sig || !sig.body) return null;
        var axes = FlatPPLEngine.orchestrator.distributeAxes(sig);
        if (axes.length === 0) return null;
        var presets = FlatPPLEngine.orchestrator.findMatchingPresets(
          sig, ctx.derivationsState.bindings);
        var domains = FlatPPLEngine.orchestrator.findMatchingDomains(
          sig, ctx.derivationsState.bindings);
        // On-demand specialize the output type at this synthetic call
        // site: scope = {paramName → input type}. typeinfer's
        // inferExprInScope handles polymorphic bodies — module-level
        // inference saw inputs as `any`, but here we have concrete
        // types from sig.inputs[i].type (which signatureOf already
        // resolved through paramSources). For multi-output bodies
        // (record/tuple/array of scalars), enumerateOutputLeaves
        // gives one entry per scalar leaf the user can pick from.
        var outputs = [];
        try {
          var paramTypes = new Map();
          for (var ii = 0; ii < sig.inputs.length; ii++) {
            paramTypes.set(sig.inputs[ii].paramName,
                           sig.inputs[ii].type || { kind: 'any' });
          }
          var specOutType = sig.body && ctx.currentLoweredModule
            ? FlatPPLEngine.typeinfer.inferExprInScope(
                ctx.currentLoweredModule, sig.body, paramTypes)
            : (sig.output && sig.output.type) || null;
          outputs = FlatPPLEngine.orchestrator.enumerateOutputLeaves(specOutType);
        } catch (_) {
          // Fall back to module-level type if specialization fails.
          outputs = FlatPPLEngine.orchestrator.enumerateOutputLeaves(
            sig.output && sig.output.type);
        }
        // Default to the first leaf — single entry with empty path
        // for scalar outputs, so the existing pipeline works
        // unchanged.
        var outputKey = outputs.length > 0 ? outputs[0].key : null;
        // Kernels (sig.kind === 'kernel') don't get a swept-axis
        // profile plot — there's nothing to "sweep" without an
        // observation. Instead we treat them like other measure
        // bindings: pick a preset (or auto-defaults), substitute
        // those into the kernel body, sample N times, and show the
        // resulting empirical measure as a histogram / corner plot.
        if (sig.kind === 'kernel') {
          return {
            name: name,
            mode: 'kernel-sample',
            signature: sig,
            axes: axes,
            matchedPresets: presets,
            presetName: null,            // null = "auto", string = named preset
            // Per-binding override for the auto pseudo-preset.
            // Auto's "values" depend on the binding's signature
            // (type defaults / cached source samples), so they
            // can't be shared module-wide. Reset when the user
            // navigates to a different binding (the plan is
            // rebuilt). Named-preset overrides live in the
            // module-wide presetOverrides map instead.
            //   null | { values: {kwarg: val} }
            autoOverride: null,
            // Domain selector state: same shape as the inputs side,
            // but driving x-axis range per kwarg from cartprod(...)
            // bindings. domainAutoOverride is the per-binding override
            // for the auto pseudo-domain (same lifetime as
            // autoOverride). Named domain overrides live module-wide
            // in domainOverrides.
            matchedDomains: domains,
            domainName: null,
            domainAutoOverride: null,
          };
        }
        return {
          name: name,
          mode: 'profile',
          signature: sig,
          axes: axes,
          sweepKey: axes[0].key,
          matchedPresets: presets,
          presetName: null,
          outputs: outputs,
          outputKey: outputKey,
          autoOverride: null,
          matchedDomains: domains,
          domainName: null,
          domainAutoOverride: null,
        };
      }

      var d = ctx.derivationsState.derivations[name];
      // A binding with no derivation can still be plottable when the
      // orchestrator's pre-eval pass put a value in fixedValues
      // (typically a record / array from rand). The phase-driven
      // dispatch below routes those by inferredType alone.
      var fixedValues = ctx.derivationsState.fixedValues;
      // Or — and this is the implicit-kernelof escape hatch — a
      // stochastic binding can have its derivation pruned because
      // its distIR depends on a parameterized (elementof) ancestor.
      // Per spec §04, clicking on `x` is equivalent to plotting
      // `kernelof(x)` with no boundary kwargs: a kernel whose inputs
      // are x's elementof leaves. We synthesise that signature and
      // route through the kernel-sample plan shape — the user gets
      // the same Inputs dropdown they'd see on an explicit
      // `kernel = kernelof(x, mu = mu)` binding.
      if (!d && !(fixedValues && fixedValues.has(name))) {
        // Pass the LIFTED bindings (derivationsState.bindings, populated
        // by buildDerivations → liftInlineSubexpressions). The unlifted
        // currentBindings don't carry `.ir`, so the structural fallback
        // in expandMeasureIR can't walk them.
        //
        // Dispatch by phase:
        //   stochastic   → implicit kernel (synthesise `kernelof(x)` with
        //                  parametric leaves as inputs; kernel-sample plan).
        //   parameterized → implicit function (synthesise `functionof(x)`
        //                  with parametric leaves as inputs; profile plan).
        // Fixed-phase bindings with no fixedValue entry shouldn't reach
        // here (they'd be in fixedValues or have a derivation); fall
        // through to "Not plottable".
        if (binding.phase === 'stochastic') {
          var implicitSig = FlatPPLEngine.orchestrator.implicitKernelSignature(
            name, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
          if (implicitSig && implicitSig.inputs.length > 0) {
            var iAxes = FlatPPLEngine.orchestrator.distributeAxes(implicitSig);
            if (iAxes.length > 0) {
              var iPresets = FlatPPLEngine.orchestrator.findMatchingPresets(
                implicitSig, ctx.derivationsState.bindings);
              var iDomains = FlatPPLEngine.orchestrator.findMatchingDomains(
                implicitSig, ctx.derivationsState.bindings);
              return {
                name: name,
                mode: 'kernel-sample',
                signature: implicitSig,
                axes: iAxes,
                matchedPresets: iPresets,
                presetName: null,
                autoOverride: null,
                matchedDomains: iDomains,
                domainName: null,
                domainAutoOverride: null,
              };
            }
          }
        } else if (binding.phase === 'parameterized') {
          var implicitFnSig = FlatPPLEngine.orchestrator.implicitFunctionSignature(
            name, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
          if (implicitFnSig && implicitFnSig.inputs.length > 0) {
            var fAxes = FlatPPLEngine.orchestrator.distributeAxes(implicitFnSig);
            if (fAxes.length > 0) {
              var fPresets = FlatPPLEngine.orchestrator.findMatchingPresets(
                implicitFnSig, ctx.derivationsState.bindings);
              var fDomains = FlatPPLEngine.orchestrator.findMatchingDomains(
                implicitFnSig, ctx.derivationsState.bindings);
              var fOutputs = FlatPPLEngine.orchestrator.enumerateOutputLeaves(
                implicitFnSig.output && implicitFnSig.output.type);
              var fOutputKey = fOutputs.length > 0 ? fOutputs[0].key : null;
              return {
                name: name,
                mode: 'profile',
                signature: implicitFnSig,
                axes: fAxes,
                sweepKey: fAxes[0].key,
                matchedPresets: fPresets,
                presetName: null,
                outputs: fOutputs,
                outputKey: fOutputKey,
                autoOverride: null,
                matchedDomains: fDomains,
                domainName: null,
                domainAutoOverride: null,
              };
            }
          }
        }
        return null;
      }
      var discrete = !!ctx.derivationsState.discrete[name];

      // Phase-driven dispatch (per spec §sec:phases):
      //   'stochastic'   → atoms vary across i; histogram / corner plot.
      //   'fixed'        → compile-time-determinate object. Sub-cases by
      //                    inferredType.kind:
      //                      value type (scalar/record/tuple/array)
      //                        → render the value as text. Scalars
      //                          additionally fall through to histogram
      //                          when the per-atom samples differ
      //                          (engine-side broadcast, e.g. lp_obs).
      //                      measure type → atoms come from sampling the
      //                        fixed measure; histogram still applies.
      //   'parameterized' → handled via callable / input bindings above.
      // Records/tuples with phase='fixed' get text directly (no
      // measureIsConstant walk — phase has already classified them as
      // deterministic).
      var phase = binding.phase;
      var inferredType = binding.inferredType;
      var typeKind = inferredType && inferredType.kind;

      // Resolve through measure-equivalence aliases — applies
      // regardless of phase. The principle is "plot by what the
      // binding IS, not how it was constructed":
      //
      //   m = lawof(observed_data)         → alias to observed_data
      //   m = Dirac(observed_data)         → alias to observed_data
      //                                       (engine promotes Dirac-
      //                                        of-ref to alias kind)
      //   y = draw(m)  for any of the above → alias to m → … →
      //                                        observed_data
      //
      // All produce per-atom values identical to observed_data's,
      // so all should render identically. Use the source binding's
      // plan, but tag it with the original name so colorForBinding
      // picks up the alias's own binding-type color (lawof-blue,
      // measure-grey, draw-purple, …) instead of the underlying
      // value's color (literal pink, etc.). For non-aliased
      // bindings (the common case — Normal samples, posterior,
      // function bindings, etc.) resolveMeasureAlias returns null
      // and we fall through to the regular dispatch below.
      var sourceName = resolveMeasureAlias(name, ctx.derivationsState.derivations,
                                           ctx.currentBindings);
      if (sourceName && sourceName !== name) {
        var sourceBinding = ctx.currentBindings.get(sourceName);
        if (sourceBinding) {
          var sourcePlan = buildPlotPlan(sourceBinding);
          if (sourcePlan) {
            var aliased = Object.assign({}, sourcePlan);
            aliased.name = name;
            return aliased;
          }
        }
      }

      if (phase === 'fixed') {
        // Opaque value-typed bindings — rngstate today, future
        // engine-internal types in the same vein — have no useful
        // visual representation. Drop them out of the plot pipeline
        // here (the alternative — falling through to samples mode —
        // produced an empty histogram of NaN values when the
        // per-atom evaluator coerced the opaque object to a Float64
        // entry).
        if (typeKind === 'rngstate') return null;
        if (typeKind === 'record' || typeKind === 'tuple') {
          return { name: name, mode: 'fixed-record' };
        }
        // Static numeric arrays still take the dedicated step-plot
        // path. (kind:'array' derivation also implies phase='fixed'
        // and inferredType=array.)
        //
        // Ground-truth fallback: route by WHAT THE BINDING IS, not by
        // the static type. A fixed-phase binding whose pre-evaluated
        // value (orchestrator fixedValues) is a flat numeric/boolean
        // vector IS an array value — even when inferredType came back
        // 'deferred' because the producing expression isn't covered by
        // typeinfer. The canonical case: `tau = (bkg ./ dbkg) .^ 2`
        // (dotted-broadcast typeinfer is intentionally loose, TODO
        // §07), which materialises byte-identically to the literal
        // `dbkg = [3.0, 7.0]` yet, without this, fell through to the
        // scalar-sample path and rendered as an empty 2-point
        // histogram. Tightening dotted-broadcast typeinfer would also
        // fix it at the source; this fallback hardens the viewer
        // against every present and future loose-typeinfer case.
        var fvMap = ctx.derivationsState.fixedValues;
        var fvVal = fvMap && fvMap.has(name) ? fvMap.get(name) : undefined;
        var isFlatNumericVec = Array.isArray(fvVal) && fvVal.length > 0
          && fvVal.every(function (e) {
            return typeof e === 'number' || typeof e === 'boolean';
          });
        if ((d && d.kind === 'array') || typeKind === 'array'
            || isFlatNumericVec) {
          return { name: name, mode: 'array' };
        }
        if (typeKind === 'scalar') {
          return { name: name, mode: 'fixed-scalar', discrete: discrete };
        }
        // Falls through (typeKind === 'measure' / 'any' / 'deferred'):
        // sample-driven render below.
      } else {
        // phase='stochastic' (or unknown) — keep the sample path.
        if (d && d.kind === 'array') {
          return { name: name, mode: 'array' };
        }
      }

      // Variates never get a density overlay — see rule 1 above.
      // For measures, the overlay is the analytical PDF/PMF when the
      // resolved leaf has all-literal kwargs (closed-form marginal).
      //
      // "Is this a variate?" semantically = stochastic phase. Today
      // only `draw(...)` / `~` produce stochastic-phase value
      // bindings, so binding.type === 'draw' happens to match — but
      // phase is the spec-grounded discriminator and protects against
      // any future syntactic form that also yields a variate. (A
      // measure with stochastic ancestors will still reach this
      // branch and be filtered by the all-literal-kwargs gate below,
      // not the phase check; that's intentional.)
      var analyticalIR = null;
      if (binding.phase !== 'stochastic') {
        var leafIR = FlatPPLEngine.orchestrator.leafSampleIR(name, ctx.derivationsState.derivations);
        if (leafIR && leafIR.kind === 'call' && leafIR.op
            && (!leafIR.args || leafIR.args.length === 0)) {
          var allLit = true;
          var kw = leafIR.kwargs || {};
          for (var k in kw) {
            if (kw[k].kind !== 'lit') { allLit = false; break; }
          }
          if (allLit) analyticalIR = leafIR;
        }
      }
      return { name: name, mode: 'samples', discrete: discrete, analyticalIR: analyticalIR };
    }

    // Plot panel visibility — separate from "is the current binding
    // plottable?". When plotEnabled is true, the plot pane occupies
    // the bottom 40% of the panel; when false, it's collapsed and the
    // graph pane takes the full content area. The pane content always
    // reflects the current focused binding, so flipping plotEnabled
    // back on never shows stale data.
    ctx.plotEnabled = false;

    function setPlotEnabled(enabled) {
      ctx.plotEnabled = !!enabled;
      var plot    = document.getElementById('plot-panel');
      var graph   = document.getElementById('graph-panel');
      var divider = document.getElementById('plot-divider');
      var btn     = document.getElementById('plot-toggle');
      plot.classList.toggle('hidden', !ctx.plotEnabled);
      graph.classList.toggle('full',  !ctx.plotEnabled);
      divider.classList.toggle('hidden', !ctx.plotEnabled);
      btn.classList.toggle('on', ctx.plotEnabled);
      btn.textContent = 'Plot: ' + (ctx.plotEnabled ? 'on' : 'off');
      // Drop any user-dragged inline flex so the class-based defaults
      // (flex: 1 1 100% on graph-full, flex: 0 0 0 on plot-hidden, or
      // the regular 60/40 split when both are showing) take effect.
      // Inline-style takes precedence over our class rules; clearing
      // it here means a toggle-off-then-on resets the split rather
      // than holding the previous drag position into the hidden state.
      graph.style.flex = '';
      plot.style.flex = '';
      // Persist across panel reopens. VS Code restores webview state
      // automatically when the panel is shown again.
      if (ctx.host.saveState) { try { ctx.host.saveState({ plotEnabled: ctx.plotEnabled }); } catch (_) {} }
      if (ctx.plotEnabled) {
        // Render whatever the current plan says — including the
        // "not plottable" message if the focused binding isn't
        // chainable. Echarts also needs resize after becoming visible
        // (it measures 0×0 while collapsed).
        renderPlotForCurrent();
        if (ctx.plotEchart) ctx.plotEchart.resize();
      } else if (ctx.plotEchart) {
        // Tear down the echart instance to avoid keeping its canvas /
        // event listeners alive while the panel is collapsed. It'll
        // be reconstructed on the next renderDensity call.
        try { ctx.plotEchart.dispose(); } catch (_) {}
        ctx.plotEchart = null;
      }
      // Cytoscape skipped resize while the graph pane was at a
      // different height — kick it now so the layout fills correctly.
      if (ctx.cy) {
        // requestAnimationFrame so the flex re-layout has settled
        // before we ask cytoscape for the new size.
        requestAnimationFrame(function() { ctx.cy.resize(); ctx.cy.fit(undefined, 40); });
      }
    }




    /**
     * Single entry-point for laying out a plot. Owns:
     *   - the flex-column structure of #plot-content
     *   - an optional toolbar row (controls on the left, sample-stats
     *     readout pinned right when `measure` is supplied)
     *   - the chart ctx.host that fills the remaining vertical space
     *   - disposal of any prior `ctx.plotEchart` and reset of inline styles
     *
     * Every measure-backed renderer (samples / corner / strips / kernel-
     * sample / profile / array-step) goes through here so the visual
     * framing is consistent across binding kinds. Plain text views
     * (constant scalars / records) use `renderTextValue` instead.
     *
     * opts:
     *   measure          — optional EmpiricalMeasure; drives N+ESS
     *                      readout (always shown when given, including
     *                      for unweighted measures where ESS = N).
     *   toolbarControls  — optional Element (or DocumentFragment)
     *                      appended to the LEFT of the toolbar. The
     *                      sample-stats readout (if `measure`) sits to
     *                      the RIGHT via `margin-left: auto`.
     *   chartCallback    — function(chartHost) called once the layout
     *                      is in place. The ctx.host is a div that fills
     *                      the remaining vertical space; the callback
     *                      writes its chart DOM (echarts.init,
     *                      grid layout, etc.) directly into it.
     */
    function renderPlotFrame(opts) {
      resetPlotContentStyle(ctx);
      if (ctx.plotEchart) { try { ctx.plotEchart.dispose(); } catch (_) {} ctx.plotEchart = null; }
      var el = document.getElementById('plot-content');
      el.innerHTML = '';
      el.style.display = 'flex';
      el.style.flexDirection = 'column';
      el.style.padding = '10px';
      el.style.boxSizing = 'border-box';
      el.style.gap = '8px';

      var hasToolbarLeft = opts.toolbarControls != null;
      var hasMeasureStats = opts.measure != null;
      if (hasToolbarLeft || hasMeasureStats) {
        var bar = document.createElement('div');
        bar.className = 'plot-frame-toolbar';
        bar.style.display = 'flex';
        bar.style.flexWrap = 'wrap';
        bar.style.gap = '0.75em';
        bar.style.alignItems = 'center';
        bar.style.padding = '0.4em 0.6em';
        bar.style.background = 'rgba(255,255,255,0.02)';
        bar.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.08))';
        bar.style.borderRadius = '3px';
        bar.style.fontSize = '0.92em';
        bar.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
        bar.style.flexShrink = '0';
        if (hasToolbarLeft) bar.appendChild(opts.toolbarControls);
        if (hasMeasureStats) {
          // margin-left:auto on the spacer pushes the stats readout
          // to the right edge regardless of how many controls are
          // on the left.
          var spacer = document.createElement('div');
          spacer.style.marginLeft = 'auto';
          bar.appendChild(spacer);
          bar.appendChild(renderSampleStats(opts.measure));
        }
        el.appendChild(bar);
      }

      var chartHost = document.createElement('div');
      chartHost.style.flex = '1 1 auto';
      chartHost.style.minHeight = '0';
      chartHost.style.minWidth = '0';
      chartHost.style.position = 'relative';
      el.appendChild(chartHost);

      // Optional row beneath the chart. Used by the profile-plot path
      // to surface the lo/hi x-axis limit inputs alongside the axis
      // name, vertically aligned with where echarts' axis-name would
      // otherwise sit. Other plot types pass nothing and get the
      // previous layout (chart fills remaining height).
      if (opts.bottomRow) {
        var bottom = document.createElement('div');
        bottom.style.flexShrink = '0';
        bottom.appendChild(opts.bottomRow);
        el.appendChild(bottom);
      }

      opts.chartCallback(chartHost);
    }

    /**
     * Render a constant value (literal, deterministic arithmetic of
     * literals, or a degenerate distribution) as plain text in the
     * scalar-display block. Used by:
     *   - constant scalar bindings (samplesAreConstant short-circuit)
     *   - phase=fixed records / tuples (renderConstantRecord)
     *   - kernel-sample bindings whose substituted body collapses to
     *     a single value
     * The font-size auto-shrinks for long renderings (record(...) with
     * many fields) so the value still fits within the pane.
     */
    function renderTextValue(bindingName, text) {
      resetPlotContentStyle(ctx);
      if (ctx.plotEchart) { try { ctx.plotEchart.dispose(); } catch (_) {} ctx.plotEchart = null; }
      var el = document.getElementById('plot-content');
      var name = bindingName ? esc(bindingName) : '';
      // Atomic values (e.g. "5", "Dirac(5)", "true") get the hero
      // 36px treatment so the value pops as the answer. Composite
      // values (records, multi-element arrays, Dirac wrappers around
      // structured bodies) fall back to a comfortable monospace
      // size — the .composite class flip is enough; the threshold
      // is "contains structural punctuation AND non-trivial length",
      // which catches both "record(a = 1.5, …)" (long) and
      // "[1.2, 3.4, 5.1, …, 3.9]" (medium). Short Dirac wraps like
      // "Dirac(5)" stay big.
      var composite = text.length > 16 && /[(\[]/.test(text);
      var valueClass = composite ? 'value composite' : 'value';
      el.innerHTML =
        '<div class="scalar-display">'
        + (name ? '<div class="name">' + name + '</div>' : '')
        + '<div class="' + valueClass + '">' + esc(text) + '</div>'
        + '</div>';
    }

    function renderPlotForCurrent() {
      // The plot panel stays mounted whenever plotEnabled is true. When
      // the focused binding isn't plottable (lawof, modules, etc.) we
      // still show *something* — a "Not plottable" message — so the
      // panel doesn't appear/disappear under the user as they click
      // around the DAG.
      //
      // Type errors take priority over EVERYTHING. The orchestrator
      // is structural and may produce a derivation for a binding
      // whose body has a type error — e.g. weighted(exp(pow(M),1), N)
      // where pow(measure) is invalid. The plot pane must short-
      // circuit on errors before sampling, otherwise we'd render a
      // valid-looking empty histogram with NaN samples instead of
      // the actionable diagnostic.
      var name = ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'this binding';
      var typeErrors = errorsForBinding(ctx, ctx.currentPlotBindingName);
      if (typeErrors && typeErrors.length > 0) {
        var msg = '<strong>' + name + '</strong> is semantically invalid:'
          + '<ul>';
        for (var i = 0; i < typeErrors.length; i++) {
          msg += '<li style="color: #E57373;">' + esc(typeErrors[i].message) + '</li>';
        }
        msg += '</ul>';
        showPlotMessage(ctx, msg);
        return;
      }
      if (!ctx.currentPlotPlan) {
        if (ctx.currentState && ctx.currentState.targetName === ctx.MODULE_TARGET) {
          showPlotMessage(ctx, 'Click a binding in the graph to plot it.', { hint: true });
          return;
        }
        // Synthetic / internal nodes (anonymous lifted subexpressions,
        // placeholders, holes, lawof / kernelof / draw bridge nodes,
        // disintegration outputs that don't carry a user binding name)
        // fail the binding lookup in updatePlotForBinding, which sets
        // currentPlotBindingName=null. Surface a generic message here
        // — there's nothing user-meaningful to plot, and pointing at
        // a different binding would be guesswork.
        if (ctx.currentPlotBindingName == null) {
          showPlotMessage(ctx, 'Internal nodes are not plottable.', { hint: true });
          return;
        }
        showPlotMessage(ctx, 'Not plottable for <strong>' + name + '</strong>.', { hint: true });
        return;
      }
      // Profile mode (function / likelihood bindings) dispatches to
      // its own worker primitive (profileN) and renderer; the rest
      // of this function handles the sample-mode pipeline.
      if (ctx.currentPlotPlan.mode === 'profile') {
        renderProfilePlotForCurrent();
        return;
      }
      // Kernel-sample mode: kernel binding rendered like any
      // sampled measure, with a preset dropdown selecting the
      // kernel's input parameters before sampling.
      if (ctx.currentPlotPlan.mode === 'kernel-sample') {
        renderKernelSampleForCurrent();
        return;
      }
      // Phase=fixed value-typed bindings: render the surface form
      // directly (text for scalars/records/tuples; existing step plot
      // for arrays). Scalars whose per-atom samples differ (engine
      // broadcast) fall through to the sample histogram path.
      if (ctx.currentPlotPlan.mode === 'fixed-record') {
        renderFixedRecord(ctx.currentPlotPlan);
        return;
      }
      // mode='fixed-scalar' falls through to the sample pipeline
      // below. renderSamplesAndDensity already short-circuits to
      // scalar-text when samplesAreConstant — phase=fixed bindings
      // whose samples are uniform get the text rendering, while
      // engine-broadcast cases (lp_obs, where phase says fixed but
      // each atom's logp differs) keep the histogram.
      // Array-mode loads the cached array synchronously (no worker
      // round-trip), so a Stop button is pointless for it. Sampling
      // mode shows the Stop button so the user can abort long
      // operations (per-i ref chains under huge sample counts).
      var arrayMode = ctx.currentPlotPlan.mode === 'array';
      showPlotMessage(ctx, arrayMode ? 'Loading…' : 'Sampling…', { cancellable: !arrayMode, hint: true });
      var planForCall = ctx.currentPlotPlan;

      // Cache hit avoids the worker entirely. We still defer through
      // a microtask so the UI flush is uniform and the stale-reply
      // guard pattern stays the same.
      Promise.resolve()
        .then(function() { return getMeasure(ctx, planForCall.name); })
        .then(function(measure) {
          if (ctx.currentPlotPlan !== planForCall) return null;
          return renderEmpiricalMeasure(measure, {
            name: planForCall.name,
            mode: planForCall.mode,
            discrete: planForCall.discrete,
            analyticalIR: planForCall.analyticalIR,
            toolbarControls: null,
            staleGuard: function() { return ctx.currentPlotPlan === planForCall; },
          });
        })
        .catch(function(err) {
          if (ctx.currentPlotPlan !== planForCall) return;
          var msg = err && err.message ? err.message : String(err);
          if (msg === 'cancelled') {
            // User clicked Stop. Make the message actionable rather
            // than dead-end so they know how to retry.
            var name = ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'this binding';
            showPlotMessage(ctx, 'Sampling cancelled. Click <strong>' + name + '</strong> in the graph to retry.', { hint: true });
          } else {
            // Real errors are actionable; not italic/dimmed.
            showPlotMessage(ctx, 'Could not compute plot: ' + esc(msg));
          }
        });
    }

    /**
     * Render the Plot panel from a samplesPlot worker reply.
     *
     * The reply has three parts:
     *   reply.samples   — raw Float64Array (kept for future use; not
     *                     directly drawn here, but available if we want
     *                     to add e.g. a sample trace later)
     *   reply.histogram — equal-width bars (FD for continuous, integer
     *                     for discrete), area-normalised so they read
     *                     directly against a PDF/PMF curve
     *   reply.density   — smooth analytical curve (when leaf has all-
     *                     literal kwargs) OR KDE estimate; null for
     *                     discrete-with-no-analytical (the histogram
     *                     itself is already the empirical pmf)
     *
     * Both layers (bars + curve) use the focused binding's ctx.TYPE_STYLE
     * color from the DAG view, so a stochastic 'draw' node plots
     * purple, a measure-alias 'call' node plots grey-blue, etc. Bars
     * sit at low alpha; the line/dots are opaque on top.
     */

    /**
     * Format a fixed scalar for display. JavaScript's default String()
     * gives "5" for 5.0 and full precision for things like 0.1+0.2;
     * we strip trailing zeros via toPrecision(12) → parseFloat → String
     * so 5.0 reads "5", 3.14159 stays "3.14159", and noisy
     * float-arithmetic results like 0.30000000000000004 become "0.3".
     */
    // Compact UI rendering of a numeric value. Truncates to 4
    // significant digits — enough to distinguish typical
    // posterior-style values (e.g. -0.1930 vs 0.2998) without the
    // false-precision look of floats printed at full Float64 width.
    // Used by inline labels (preset dropdowns, x-range inputs),
    // value-as-text displays, and as the echarts axisLabel formatter
    // so chart ticks match the same convention. Integers pass through
    // unchanged (Number.isInteger short-circuit) so axis ticks at
    // whole numbers stay readable as "1", "2", … rather than "1.000".

    // "a + b i" / "a - b i" for a complex scalar constant. Both parts
    // go through formatScalar so precision/integer handling matches
    // the real path. The sign is folded into the connector so we never
    // print "a + -b i"; -0 imaginary reads as "+ 0 i".

    // Toolbar badge for a complex binding rendered as its real part.
    // Static (no interaction) in v1 — the |z| / Im / Argand mode
    // toggle is a tracked follow-up and will replace this with a
    // button group in the same toolbar slot.

    // Compose pre-formatted element strings into "[a, b, c]" or
    // "[a, b, c, …, z] (length N)" for long arrays. The threshold
    // balances readability against verbosity: 8 fits on typical
    // screen widths even with ~5-digit values.

    // Back-compat shim: takes a numeric array, formats each element
    // via formatScalar, then composes with formatArrayParts.

    // Composable value-to-string for plain JS values — numbers,
    // booleans, strings, arrays, plain objects. Mirrors the
    // FlatPPL surface form (record(k = v, …) for objects, [v, …]
    // for arrays, ellipsised when long). The kind of light-weight
    // pretty-printer that Julia's Base.show pairs with each value
    // type. Used for preset value display in the toolbar dropdown
    // and as the leaf-formatter for constant-measure rendering.


    // True iff every scalar leaf of a record/tuple/array measure has
    // identical samples across all N atoms. The deterministic
    // detection drives the constant-as-text rendering for
    // record-shaped bindings whose value is the same at every atom
    // (literal records, deterministic arithmetic over literals, etc.)
    // — same idea as the scalar samplesAreConstant short-circuit, but
    // walks the SoA tree.
    //
    // Special case for literal-array fields: a `kind: 'array'`
    // derivation materialises as { samples: Float64Array(K), ... }
    // where K is the array length (NOT SAMPLE_COUNT). Per-atom these
    // are deterministic — all atoms see the same array — so we treat
    // them as "constant" even though the array's values differ from
    // each other. We detect this via samples.length !== SAMPLE_COUNT;
    // a per-atom scalar measure has length === SAMPLE_COUNT.

    // Render a constant measure as the FlatPPL surface form. Used by
    // the plot-pane dispatch when measureIsConstant returns true:
    // record-shaped bindings show "record(a = …, b = …)" text rather
    // than a corner plot of N copies of the same point. Array leaves
    // ellipsize past length 8 so a 10-observation literal stays
    // readable. Walks the SoA tree top-down — same shape conventions
    // as listScalarAxes.

    /**
     * Resolve a binding's plot color to match the DAG renderer's
     * choice exactly. The DAG picks color from ctx.TYPE_STYLE[node.type]
     * but then overrides it when node.kind says "measure" (lawof
     * blue) or "kernel" (kernelof teal). Without those overrides the
     * plot for a measure-typed binding (theta1_dist, type='call')
     * would draw in grey instead of the blue used in the DAG bubble,
     * breaking the visual link between the two views.
     *
     * Fall back to ctx.TYPE_STYLE[binding.type] when the binding isn't in
     * the current DAG — paths that update the plot independent of
     * the DAG (rare, but possible during config-update reflows).
     */



    /**
     * Render a fixed-length array as an index→value step plot. Used
     * for literal-array bindings (observed_data = [1.2, 3.4, …]),
     * which aren't samples of a distribution. The series is drawn as
     * piecewise-constant horizontal segments — same shape as the
     * legacy "data preview" view that used to swap into the graph
     * pane, now living in the plot pane next to the DAG.
     */
    // Persistent per-binding plot state for record-shaped measures:
    // selected axes (which scalar leaves to plot in correlations
    // mode) and the chosen view mode ('correlations' | 'marginals').
    // Reset when the focused binding changes; survives re-renders
    // triggered by checkbox / toggle clicks.
    //
    // Correlations mode (NxN matrix of marginals + joint scatters)
    // becomes unreadable past 4x4, so we cap selection at 4. Marginals
    // mode (one density-shaded column per axis) scales linearly to
    // the full axis count and ignores the selection.
    var recordSelection = null;
    var CORRELATIONS_MAX_AXES = 4;


    /**
     * Render a record-shaped EmpiricalMeasure as a corner plot,
     * with a checkbox row above for axis selection (max 4 axes).
     *
     * Corner plot:
     *   - diagonal:        1D marginal histogram of each selected axis
     *   - below-diagonal:  2D joint scatter for each (axis_j, axis_i)
     *                      pair with i > j
     *   - above-diagonal:  empty (corner-plot convention)
     *
     * SoA pays off here: marginals are sub.samples; joints are just
     * two columns zipped index-wise — no copy, no projection.
     */
    // Render a constant record/tuple as plain text — same scalar-display
    // styling the constant-scalar branch uses, just with the surface
    // form as the value. We cap font-size when the rendered string is
    // long so the corner-plot 36px doesn't overflow on a multi-field
    // record; the simple len-based cutoff is fine here (the value is
    // either short and reads at 36px or long enough to want 16px).
    function renderConstantRecord(measure, bindingName) {
      renderTextValue(bindingName, formatConstantMeasure(ctx, measure));
    }

    function renderRecordMarginals(measure, bindingName, extraToolbarControls) {
      var axes = listScalarAxes(measure);
      if (axes.length === 0) {
        showPlotMessage(ctx, 'No scalar fields to plot for <strong>' + esc(bindingName) + '</strong>.', { hint: true });
        return;
      }

      // Group prefix per axis (drop any trailing "[k]"). Used by
      // marginals view's group-level selector and (separately) by
      // its boundary insets between groups. Same definition both
      // places — kept here so selection state and rendering stay in
      // sync via a single source of truth.
      function axisGroupKey(label) {
        var i = label.lastIndexOf('[');
        return i >= 0 ? label.slice(0, i) : label;
      }
      var allGroups = [];
      var seenGroup = {};
      for (var gi = 0; gi < axes.length; gi++) {
        var g = axisGroupKey(axes[gi].label);
        if (!seenGroup[g]) { seenGroup[g] = true; allGroups.push(g); }
      }

      // Reset selection when the focused binding changes. Defaults:
      //   mode='correlations'; selected = first CORRELATIONS_MAX_AXES
      //                                    axes (per-axis selection)
      //   marginalGroups = all groups (group-level selection used in
      //                                marginals mode)
      if (!recordSelection || recordSelection.bindingName !== bindingName) {
        recordSelection = {
          bindingName: bindingName,
          mode: 'correlations',
          selected: axes.slice(0, CORRELATIONS_MAX_AXES).map(function(a) { return a.key; }),
          marginalGroups: allGroups.slice(),
        };
      } else {
        // Drop any selections that no longer exist (rare — defensive).
        var present = {}; axes.forEach(function(a) { present[a.key] = true; });
        recordSelection.selected = recordSelection.selected.filter(function(k) { return present[k]; });
        if (!recordSelection.marginalGroups) recordSelection.marginalGroups = allGroups.slice();
        else {
          var presentGroups = {}; allGroups.forEach(function(g) { presentGroups[g] = true; });
          recordSelection.marginalGroups = recordSelection.marginalGroups.filter(
            function(g) { return presentGroups[g]; });
          if (recordSelection.marginalGroups.length === 0) recordSelection.marginalGroups = allGroups.slice();
        }
      }

      // chartHostRef captures the chart-area div from the frame, so
      // rerenderChart can clear and repopulate it without rebuilding
      // the toolbar (which would close any open dropdown).
      var chartHostRef = null;

      // Two-tier re-render. rerenderAll rebuilds the entire frame
      // (including the toolbar) — used when mode-button styling
      // changes. rerenderChart only repaints the chart host —
      // used by axis-selection toggles so the open dropdown survives.
      function rerenderChart() {
        if (!chartHostRef) return;
        chartHostRef.innerHTML = '';
        // Reset inline styles the strip / grid renderer may have set
        // on a previous pass (display:grid for cornerGrid; flex for
        // strips). We re-establish from scratch each draw.
        chartHostRef.style.display = '';
        chartHostRef.style.gridTemplateColumns = '';
        chartHostRef.style.gridTemplateRows = '';
        chartHostRef.style.gap = '';
        if (recordSelection.mode === 'marginals') {
          // Marginals mode: filter axes by selected groups (group =
          // axis label's prefix before any "[k]"). Default is all
          // groups → full axis list; users uncheck to narrow.
          var selSet = {};
          (recordSelection.marginalGroups || allGroups).forEach(function(g) {
            selSet[g] = true;
          });
          var picked = axes.filter(function(a) { return selSet[axisGroupKey(a.label)]; });
          renderDensityStrips(chartHostRef, measure, bindingName, picked);
        } else {
          renderCornerGrid(chartHostRef, measure, bindingName);
        }
      }
      function rerenderAll() {
        // extraToolbarControls is a builder thunk (or null) — resolve
        // to a fresh Element/Fragment each rebuild. A static Element
        // captured once gets emptied on the first appendChild (for
        // DocumentFragments) or destroyed by renderPlotFrame's
        // innerHTML='' before the next rebuild can re-use it.
        var extra = typeof extraToolbarControls === 'function'
          ? extraToolbarControls()
          : extraToolbarControls;
        var toolbarControls = renderRecordToolbar(
          axes, allGroups, rerenderAll, rerenderChart, extra);
        renderPlotFrame({
          measure: measure,
          toolbarControls: toolbarControls,
          chartCallback: function(chartHost) {
            chartHostRef = chartHost;
            rerenderChart();
          },
        });
      }

      rerenderAll();
    }

    /**
     * Build the inner controls of the corner-plot toolbar: view-mode
     * toggle on the left, axis (or group) selector to its right, and
     * the kernel-sample preset dropdown (when supplied) further right.
     *
     * Returns a DocumentFragment that the caller hands to
     * renderPlotFrame as `toolbarControls`. The frame owns the
     * outer toolbar styling and pins the N+ESS readout to the right
     * — this builder no longer touches sample-stats.
     *
     * Rebuilt on every full rerender (cheap; <100 elements) so the
     * mode buttons reflect active state and the selector visibility
     * tracks the mode.
     */
    function renderRecordToolbar(axes, groups, onModeChange, onSelectionChange, extraToolbarControls) {
      var bar = document.createDocumentFragment();

      // ---- Mode toggle group ----
      var modeGroup = document.createElement('div');
      modeGroup.style.display = 'flex';
      modeGroup.style.gap = '0.25em';

      function makeModeBtn(modeKey, label, title) {
        var b = document.createElement('button');
        b.textContent = label;
        b.title = title;
        b.style.cursor = 'pointer';
        b.style.fontSize = '1em';
        b.style.padding = '0.2em 0.8em';
        b.style.border = '1px solid var(--vscode-button-border, transparent)';
        b.style.borderRadius = '3px';
        var active = recordSelection.mode === modeKey;
        b.style.background = active
          ? 'var(--vscode-button-background, #0e639c)'
          : 'var(--vscode-button-secondaryBackground, #3a3d41)';
        b.style.color = active
          ? 'var(--vscode-button-foreground, #fff)'
          : 'var(--vscode-button-secondaryForeground, #ccc)';
        b.addEventListener('click', function() {
          if (recordSelection.mode === modeKey) return;
          recordSelection.mode = modeKey;
          // Clip selection to correlations cap when switching back.
          if (modeKey === 'correlations'
              && recordSelection.selected.length > CORRELATIONS_MAX_AXES) {
            recordSelection.selected = recordSelection.selected.slice(0, CORRELATIONS_MAX_AXES);
          }
          // Mode toggle changes button styling → full toolbar rebuild.
          onModeChange();
        });
        return b;
      }
      modeGroup.appendChild(makeModeBtn('correlations', 'Correlations',
        'Pairwise corner plot: marginals on the diagonal, joint scatters below'));
      modeGroup.appendChild(makeModeBtn('marginals', 'Marginals',
        'One column per axis with vertical density shading; plots every axis'));
      bar.appendChild(modeGroup);

      // Axis-level selector in correlations mode (per-leaf
      // checkboxes, capped at CORRELATIONS_MAX_AXES); group-level
      // selector in marginals mode (one entry per name-prefix —
      // obs[1]…obs[10] collapse into a single "obs" toggle).
      if (recordSelection.mode === 'correlations') {
        var sep = document.createElement('div');
        sep.style.width = '1px';
        sep.style.alignSelf = 'stretch';
        sep.style.background = 'rgba(255,255,255,0.1)';
        bar.appendChild(sep);
        // Axis-checkbox toggles only need to redraw the chart (the
        // toolbar's button styling is unaffected) — pass the
        // chart-only callback so the dropdown doesn't get rebuilt
        // out from under its open popup.
        bar.appendChild(renderAxisDropdown(axes, onSelectionChange));
      } else if (recordSelection.mode === 'marginals' && groups && groups.length > 1) {
        var sep2 = document.createElement('div');
        sep2.style.width = '1px';
        sep2.style.alignSelf = 'stretch';
        sep2.style.background = 'rgba(255,255,255,0.1)';
        bar.appendChild(sep2);
        bar.appendChild(renderGroupDropdown(groups, onSelectionChange));
      }

      // Caller-supplied controls (currently: the kernel-sample
      // preset dropdown) sit after the axis selector so the
      // toolbar reads left-to-right as
      //   [plot style] [axes] [preset] [...N + ESS pinned right by frame]
      if (extraToolbarControls) bar.appendChild(extraToolbarControls);
      return bar;
    }

    /**
     * Compact "N: ...  ESS: ..." readout for the toolbar's right
     * edge. Format:
     *   "<N> samples (<label>: ESS <ratio>%, PSIS k̂ <value>)"
     * where <label> ∈ {good, ok, bad, unusable} colours the
     * parenthesised diagnostic span. Quality is computed by
     * FlatPPLEngine.empirical.importanceSamplingQuality, which
     * combines PSIS k̂ (Vehtari et al.; Pareto-tail shape of the
     * upper importance weights) with Kish ESS, max-weight share,
     * and a sample-size-aware k̂ threshold. See empirical.js for
     * the threshold table; the worst trigger across diagnostics
     * sets the label.
     *
     * Unweighted measures (logWeights == null) always read 'good'
     * with ratio 100% and k̂ shown as "—" (not meaningful for
     * uniform weights).
     */
    /** Format a log-total-mass for the stats readout. The engine carries
        mass on the log scale precisely because deep compositions can
        easily overflow Float64; the display layer formats it back.
        Returns null when the mass is essentially 1 (normalized) — the
        caller skips the badge entirely so the readout only surfaces
        info when there's something to say. */

    function renderSampleStats(measure) {
      var wrap = document.createElement('span');
      wrap.style.display = 'inline-flex';
      wrap.style.alignItems = 'center';
      wrap.style.gap = '0.4em';
      wrap.style.opacity = '0.85';
      wrap.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
      wrap.style.fontSize = '0.92em';

      // Totalmass badge — shows when the measure is non-normalized
      // (weighted, superpose, bayesupdate's posterior carrying the
      // marginal likelihood Z, etc.). Normalized measures (every leaf
      // distribution, normalize(...), lawof(...)) show no badge so the
      // readout stays uncluttered.
      if (measure && typeof measure.logTotalmass === 'number') {
        var massText = formatLogTotalmass(measure.logTotalmass);
        if (massText != null) {
          var massSpan = document.createElement('span');
          massSpan.textContent = 'total mass: ' + massText;
          massSpan.title = 'log total mass: ' + measure.logTotalmass.toFixed(4)
            + '\nThe measure is unnormalized — its total mass differs from 1. '
            + 'Wrap in normalize(...) to rescale.';
          massSpan.style.opacity = '0.9';
          wrap.appendChild(massSpan);
          // Visual separator between badges. A middle dot collides
          // with the math context here — `exp(-20.564) · 10⁵` reads as
          // a product. A pipe stays neutral and is the conventional
          // "and now a different stat" separator in technical UIs.
          // We also bump the surrounding gap so the boundary is
          // visually distinct without needing a heavy glyph.
          var sep = document.createElement('span');
          sep.textContent = '│';   // U+2502 BOX DRAWINGS LIGHT VERTICAL
          sep.style.opacity = '0.35';
          sep.style.margin = '0 0.25em';
          wrap.appendChild(sep);
        }
      }

      // Defensive try/catch: a thrown error here would propagate up
      // through renderPlotFrame → renderRecordMarginals' rerenderAll,
      // poisoning the entire plot render. Diagnostic-readout failure
      // is non-fatal — fall back to a count-only display so the chart
      // still draws. console.error surfaces real bugs in the quality
      // classifier without breaking user-facing rendering.
      try {
        var dof = FlatPPLEngine.empirical.estimateDof(measure);
        var q = FlatPPLEngine.empirical.importanceSamplingQuality(measure, dof);

        var nLabel = document.createElement('span');
        nLabel.textContent = formatSampleCount(q.N) + ' samples';
        nLabel.title = 'Total atom count in the empirical measure'
                     + (q.N >= 100 && Math.log10(q.N) === Math.floor(Math.log10(q.N))
                        ? ' (' + formatCount(q.N) + ')'
                        : '');
        wrap.appendChild(nLabel);

        // Effectively-uniform measures (no logWeights, or logWeights
        // all-equal-within-epsilon) carry no IS-quality information:
        // every atom has equal weight by construction, so PSIS k̂
        // doesn't apply and the ESS readout would just say "100%".
        // The engine signals this via kHat NaN — we skip the
        // diagnostic span and show the bare count. The diagnostic
        // only appears when it actually carries information
        // (importance-reweighted measures: bayesupdate / weighted /
        // logweighted / posterior outputs).
        if (!Number.isFinite(q.kHat)) return wrap;

        var diag = document.createElement('span');
        diag.className = 'is-quality is-' + q.label;
        var ratioPct = (q.ratio * 100);
        var ratioStr = ratioPct >= 10 ? ratioPct.toFixed(0)
                                      : ratioPct.toFixed(1);
        diag.textContent = '(' + q.label + ': ESS ' + ratioStr + '%, PSIS k̂ ' + q.kHat.toFixed(2) + ')';
        diag.title = qualityTooltip(q);
        wrap.appendChild(diag);
      } catch (err) {
        try { console.error('IS-quality classifier failed:', err); } catch (_) {}
        wrap.textContent = '— samples';
      }
      return wrap;
    }




    // Compact sample-count rendering: powers of 10 collapse to
    // superscript form ("10⁵" instead of "100,000") to save toolbar
    // width — typical default sample sizes (10⁴, 10⁵, 10⁶) all win.
    // Anything else falls back to the comma-grouped count. Only
    // exact powers ≥ 10² qualify; "10" itself stays "10" and small
    // counts read better verbatim.

    /**
     * Compact dropdown axis selector for correlations mode. Button
     * shows the count ("Plot axes (3 / 12) ▾"); click opens a
     * popup-anchored panel with a scrollable checkbox list. Outside
     * clicks close it. Cap enforcement (max 4) shows an inline red
     * note in the panel when the user tries to exceed.
     */
    function renderAxisDropdown(axes, onChange) {
      var wrap = document.createElement('div');
      wrap.style.position = 'relative';
      wrap.style.display = 'inline-flex';
      wrap.style.alignItems = 'center';
      wrap.style.gap = '0.4em';

      var hint = document.createElement('span');
      hint.textContent = 'Variates:';
      hint.style.opacity = '0.6';
      wrap.appendChild(hint);

      var btn = document.createElement('button');
      btn.style.cursor = 'pointer';
      btn.style.fontSize = '1em';
      btn.style.padding = '0.2em 0.6em';
      btn.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
      btn.style.borderRadius = '3px';
      btn.style.background = 'var(--vscode-button-secondaryBackground, #3a3d41)';
      btn.style.color = 'var(--vscode-button-secondaryForeground, #ccc)';
      btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
      btn.textContent = recordSelection.selected.length
        + ' / ' + axes.length + '  ▾';
      wrap.appendChild(btn);

      // Popup panel — absolutely positioned beneath the button.
      var panel = document.createElement('div');
      panel.style.position = 'absolute';
      panel.style.top = 'calc(100% + 4px)';
      panel.style.left = '0';
      panel.style.zIndex = '50';
      panel.style.minWidth = '14em';
      panel.style.maxHeight = '20em';
      panel.style.overflowY = 'auto';
      panel.style.padding = '0.4em';
      panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
      panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
      panel.style.borderRadius = '3px';
      panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
      panel.style.display = 'none';
      wrap.appendChild(panel);

      // Cap-error slot inside the panel (red note shown briefly when
      // the user tries to add a 5th).
      var capErr = document.createElement('div');
      capErr.style.color = '#E57373';
      capErr.style.fontSize = '0.92em';
      capErr.style.padding = '0.3em 0.4em';
      capErr.style.opacity = '0';
      capErr.style.transition = 'opacity 0.2s';
      panel.appendChild(capErr);

      axes.forEach(function(axis) {
        var label = document.createElement('label');
        label.style.display = 'flex';
        label.style.alignItems = 'center';
        label.style.gap = '0.4em';
        label.style.padding = '0.2em 0.4em';
        label.style.cursor = 'pointer';
        label.style.userSelect = 'none';
        label.style.borderRadius = '2px';
        label.addEventListener('mouseenter', function() { label.style.background = 'rgba(255,255,255,0.05)'; });
        label.addEventListener('mouseleave', function() { label.style.background = ''; });

        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = recordSelection.selected.indexOf(axis.key) >= 0;
        cb.addEventListener('change', function(ev) {
          // Don't bubble up to the wrap's outside-click closer.
          ev.stopPropagation();
          var idx = recordSelection.selected.indexOf(axis.key);
          if (cb.checked) {
            if (idx >= 0) return;
            if (recordSelection.selected.length >= CORRELATIONS_MAX_AXES) {
              cb.checked = false;
              capErr.textContent = 'At most ' + CORRELATIONS_MAX_AXES
                + ' axes — uncheck one first.';
              capErr.style.opacity = '1';
              return;
            }
            recordSelection.selected.push(axis.key);
          } else {
            if (idx >= 0) recordSelection.selected.splice(idx, 1);
          }
          capErr.style.opacity = '0';
          // Update the count on the button without rebuilding the
          // toolbar (which would tear down this dropdown's open
          // panel). The axis-dropdown stays open until the user
          // clicks outside.
          btn.textContent = recordSelection.selected.length
            + ' / ' + axes.length + '  ▾';
          onChange();
        });
        label.appendChild(cb);

        var name = document.createElement('span');
        name.textContent = axis.label;
        name.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
        label.appendChild(name);
        panel.appendChild(label);
      });

      // Toggle on button click; close on outside click.
      btn.addEventListener('click', function(ev) {
        ev.stopPropagation();
        var open = panel.style.display !== 'none';
        panel.style.display = open ? 'none' : 'block';
        if (!open) {
          // One-shot outside-click handler — registers on this open,
          // tears itself down on close so we don't accumulate handlers.
          var off = function(ev2) {
            if (panel.contains(ev2.target) || btn.contains(ev2.target)) return;
            panel.style.display = 'none';
            document.removeEventListener('click', off, true);
          };
          // capture phase so we close before any inner click is processed
          setTimeout(function() {
            document.addEventListener('click', off, true);
          }, 0);
        }
      });

      return wrap;
    }

    /**
     * Group-level checkbox dropdown for marginals view. Same shape
     * as renderAxisDropdown but operates on group prefixes (obs[1]
     * …obs[10] collapse to a single "obs" entry) and has no
     * selection cap. State lives in recordSelection.marginalGroups.
     */
    function renderGroupDropdown(groups, onChange) {
      var wrap = document.createElement('div');
      wrap.style.position = 'relative';
      wrap.style.display = 'inline-flex';
      wrap.style.alignItems = 'center';
      wrap.style.gap = '0.4em';

      var hint = document.createElement('span');
      hint.textContent = 'Variates:';
      hint.style.opacity = '0.6';
      wrap.appendChild(hint);

      var btn = document.createElement('button');
      btn.style.cursor = 'pointer';
      btn.style.fontSize = '1em';
      btn.style.padding = '0.2em 0.6em';
      btn.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
      btn.style.borderRadius = '3px';
      btn.style.background = 'var(--vscode-button-secondaryBackground, #3a3d41)';
      btn.style.color = 'var(--vscode-button-secondaryForeground, #ccc)';
      btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
      function updateBtn() {
        btn.textContent = recordSelection.marginalGroups.length
          + ' / ' + groups.length + '  ▾';
      }
      updateBtn();
      wrap.appendChild(btn);

      var panel = document.createElement('div');
      panel.style.position = 'absolute';
      panel.style.top = 'calc(100% + 4px)';
      panel.style.left = '0';
      panel.style.zIndex = '50';
      panel.style.minWidth = '12em';
      panel.style.maxHeight = '20em';
      panel.style.overflowY = 'auto';
      panel.style.padding = '0.4em';
      panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
      panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
      panel.style.borderRadius = '3px';
      panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
      panel.style.display = 'none';
      wrap.appendChild(panel);

      groups.forEach(function(g) {
        var label = document.createElement('label');
        label.style.display = 'flex';
        label.style.alignItems = 'center';
        label.style.gap = '0.4em';
        label.style.padding = '0.2em 0.4em';
        label.style.cursor = 'pointer';
        label.style.userSelect = 'none';
        label.style.borderRadius = '2px';
        label.addEventListener('mouseenter', function() { label.style.background = 'rgba(255,255,255,0.05)'; });
        label.addEventListener('mouseleave', function() { label.style.background = ''; });

        var cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = recordSelection.marginalGroups.indexOf(g) >= 0;
        cb.addEventListener('change', function(ev) {
          ev.stopPropagation();
          var idx = recordSelection.marginalGroups.indexOf(g);
          if (cb.checked) {
            if (idx < 0) recordSelection.marginalGroups.push(g);
          } else {
            if (idx >= 0) recordSelection.marginalGroups.splice(idx, 1);
          }
          updateBtn();
          onChange();
        });
        label.appendChild(cb);

        var name = document.createElement('span');
        name.textContent = g;
        name.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
        label.appendChild(name);
        panel.appendChild(label);
      });

      btn.addEventListener('click', function(ev) {
        ev.stopPropagation();
        var open = panel.style.display !== 'none';
        panel.style.display = open ? 'none' : 'block';
        if (!open) {
          var off = function(ev2) {
            if (panel.contains(ev2.target) || btn.contains(ev2.target)) return;
            panel.style.display = 'none';
            document.removeEventListener('click', off, true);
          };
          setTimeout(function() {
            document.addEventListener('click', off, true);
          }, 0);
        }
      });

      return wrap;
    }

    /**
     * Render the selected axes as a 2D density-strip view: one
     * column per axis, where each column shades by the per-axis
     * marginal density along y. Useful for array-shaped data where
     * each index slot has its own marginal — corner plots scale
     * O(N²) cells, this scales O(N).
     *
     * Implementation: per axis, compute an FD histogram. For each
     * bin in that axis, draw a horizontal rect spanning the column
     * width with opacity proportional to bin density (relative to
     * the axis's max). The y-axis is shared across all columns
     * (global value range) so columns are visually comparable.
     *
     * ECharts has a heatmap series that could do this for free,
     * but the value-axis variant restricts to a regular grid; we
     * want each axis's bin grid to align to its own FD-derived
     * edges. Custom render gives that flexibility cheaply.
     */
    function renderDensityStrips(hostEl, measure, bindingName, axesArg) {
      hostEl.innerHTML = '';
      // Marginals mode passes the full axis list (no selection cap); we
      // fall back to listScalarAxes for legacy callers.
      var axes = axesArg || listScalarAxes(measure);
      var n = axes.length;
      if (n === 0) {
        var empty = document.createElement('div');
        empty.textContent = 'No scalar axes to plot.';
        empty.style.opacity = '0.5';
        empty.style.padding = '24px';
        empty.style.textAlign = 'center';
        hostEl.appendChild(empty);
        return;
      }

      var fg = getComputedStyle(document.body).color || '#ccc';
      var color = colorForBinding(ctx, bindingName);
      var logWeights = measure.logWeights;
      var histOptsBase = logWeights ? { logWeights: logWeights } : {};

      // Per-axis FD histograms + global y range.
      var hists = axes.map(function(a) {
        return FlatPPLEngine.histogram.freedmanDiaconisHistogram(a.samples, histOptsBase);
      });
      // Per-axis peak densities: a tightly-concentrated marginal has
      // a much higher per-bin density than a broad one (Σ density ×
      // binwidth ≈ 1 in either case), so a single global peak makes
      // broad columns look near-empty. Normalising each column to
      // its own peak keeps every column's shape readable.
      var yMin = Infinity, yMax = -Infinity;
      var peakDensities = new Array(hists.length);
      for (var i = 0; i < hists.length; i++) {
        var h = hists[i];
        peakDensities[i] = 0;
        if (!h.binEdges || h.binEdges.length === 0) continue;
        if (h.binEdges[0] < yMin) yMin = h.binEdges[0];
        if (h.binEdges[h.binEdges.length - 1] > yMax) yMax = h.binEdges[h.binEdges.length - 1];
        for (var j = 0; j < h.ys.length; j++) {
          if (h.ys[j] > peakDensities[i]) peakDensities[i] = h.ys[j];
        }
      }
      if (!isFinite(yMin) || !isFinite(yMax) || yMin === yMax) {
        var info = document.createElement('div');
        info.textContent = 'No variation across selected axes.';
        info.style.opacity = '0.5';
        info.style.padding = '24px';
        info.style.textAlign = 'center';
        hostEl.appendChild(info);
        return;
      }

      // One echarts instance hosts the whole strip view. Categories on
      // x (one per axis); value y (continuous, shared range). Bins
      // rendered as semi-transparent rects via custom series.
      hostEl.style.display = '';
      hostEl.style.gridTemplateColumns = '';
      hostEl.style.gridTemplateRows = '';
      hostEl.innerHTML = '';
      var chartDiv = document.createElement('div');
      chartDiv.style.width = '100%';
      chartDiv.style.height = '100%';
      hostEl.appendChild(chartDiv);

      // Group adjacent axes that belong to the same parent (an iid
      // array's slots all share the prefix before "[k]"). Within a
      // group, columns render edge-to-edge so obs[1]..obs[10] reads
      // as a continuous shaded sequence. Between groups, the
      // boundary cells get an inset on the gap side so a small
      // visible gap separates groups (a full empty slot was too
      // wide). axisGroup extracts the prefix before any trailing
      // "[…]" — same-group axes share that prefix.
      function axisGroup(label) {
        var i = label.lastIndexOf('[');
        return i >= 0 ? label.slice(0, i) : label;
      }
      var groups = axes.map(function(a) { return axisGroup(a.label); });
      // Per-axis gap flags. Boundary cells (group differs from
      // neighbour) shrink on the gap side; the renderer reads these
      // off the rect data. GAP_FRACTION is the inset depth as a
      // fraction of bandSize — 0.18 gives a ~36% combined visible
      // gap between groups (much tighter than a full empty slot)
      // while still leaving the bulk of the column for the data.
      var GAP_FRACTION = 0.18;
      // Build the rect data: one entry per (axis_idx, bin) pair.
      // Each entry carries [axis_idx, bin_y_center, density] plus
      // the bin's [lo, hi] and per-side gap insets.
      var data = [];
      for (var ai2 = 0; ai2 < hists.length; ai2++) {
        var hh = hists[ai2];
        var gapLeft  = (ai2 > 0)             && groups[ai2] !== groups[ai2 - 1];
        var gapRight = (ai2 < axes.length-1) && groups[ai2] !== groups[ai2 + 1];
        for (var bi = 0; bi < hh.ys.length; bi++) {
          data.push({
            value: [ai2, hh.xs[bi], hh.ys[bi]],
            edges: [hh.binEdges[bi], hh.binEdges[bi + 1]],
            gapLeft: gapLeft, gapRight: gapRight,
          });
        }
      }
      var catLabels = axes.map(function(a) { return a.label; });
      var seriesColor = color;
      var ec = echarts.init(chartDiv);
      ec.setOption({
        backgroundColor: 'transparent',
        animation: false,
        grid: { left: 60, right: 25, top: 10, bottom: 60, containLabel: false },
        xAxis: {
          type: 'category',
          data: catLabels,
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: {
            color: fg, opacity: 0.7, fontSize: 11,
            interval: 0,
            rotate: axes.length > 8 ? 60 : 0,
            fontFamily: 'var(--vscode-editor-font-family, monospace)',
          },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value', scale: true,
          min: yMin, max: yMax,
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, fontSize: 11, formatter: formatScalar },
          splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
        },
        series: [{
          type: 'custom',
          data: data,
          renderItem: function(_p, api) {
            var d = data[_p.dataIndex];
            var density = d.value[2];
            // Column-centred horizontal extent: ~70% of slot width.
            var cx = api.coord([d.value[0], (d.edges[0] + d.edges[1]) / 2]);
            var top = api.coord([d.value[0], d.edges[1]]);
            var bot = api.coord([d.value[0], d.edges[0]]);
            // ECharts category axes give bandWidth via api.size.
            // Use the full band width so adjacent columns share
            // their edges — the marginal strips read as a continuous
            // shaded sequence within a group. At group boundaries
            // (gapLeft / gapRight set on the rect data) we inset the
            // boundary cell by GAP_FRACTION of bandSize on the gap
            // side, so different-prefix axes get a small visible
            // separator without a full empty column.
            var bandSize = api.size([1, 0])[0];
            var leftEdge  = cx[0] - bandSize * 0.5
                          + (d.gapLeft  ? bandSize * GAP_FRACTION : 0);
            var rightEdge = cx[0] + bandSize * 0.5
                          - (d.gapRight ? bandSize * GAP_FRACTION : 0);
            // Opacity scales linearly with density relative to the
            // PER-AXIS peak — concentrated and broad marginals both
            // read at full intensity at their mode rather than the
            // broad ones fading under a tightly-concentrated peer.
            var axisPeak = peakDensities[d.value[0]];
            var opacity = axisPeak > 0
              ? Math.max(0.04, 0.85 * density / axisPeak)
              : 0;
            return {
              type: 'rect',
              shape: {
                x: leftEdge,
                y: top[1],
                width: rightEdge - leftEdge,
                height: bot[1] - top[1],
              },
              style: api.style({ fill: seriesColor, opacity: opacity, stroke: 'none' }),
            };
          },
          encode: { x: 0, y: 1 },
        }],
      });
    }

    /**
     * Build the corner-plot grid (diagonal marginals + below-diagonal
     * scatters) for the currently-selected axes. ctx.host is the parent
     * div whose contents we replace; it must be a flex/block child
     * with a fixed height so the inner grid expands correctly.
     */
    function renderCornerGrid(hostEl, measure, bindingName) {
      hostEl.innerHTML = '';
      var axes = listScalarAxes(measure)
        .filter(function(a) { return recordSelection.selected.indexOf(a.key) >= 0; });
      var n = axes.length;
      if (n === 0) {
        var empty = document.createElement('div');
        empty.textContent = 'Select at least one axis to plot.';
        empty.style.opacity = '0.5';
        empty.style.padding = '24px';
        empty.style.textAlign = 'center';
        hostEl.appendChild(empty);
        return;
      }

      var fg = getComputedStyle(document.body).color || '#ccc';
      var color = colorForBinding(ctx, bindingName);
      var logWeights = measure.logWeights;
      var histOptsBase = logWeights ? { logWeights: logWeights } : {};

      // Grid layout with two extra tracks: a leftmost column for
      // vertical y-axis labels (one per plot row), and a bottom row
      // for horizontal x-axis labels (one per plot column). The y
      // labels' track is auto-sized to the label width; the x
      // labels' track to the label height.
      //
      //   col:    [auto] [1fr] [1fr] ... [1fr]       n+1 columns
      //   row:    [1fr]                              n rows of plots
      //           [1fr]
      //           ...
      //           [auto]                             1 row of x labels
      hostEl.style.display = 'grid';
      hostEl.style.gridTemplateColumns = 'auto repeat(' + n + ', 1fr)';
      hostEl.style.gridTemplateRows    = 'repeat(' + n + ', 1fr) auto';
      hostEl.style.gap = '6px';
      hostEl.style.minHeight = '0';

      // ---- y-axis labels (left column, vertical) -----------------
      // Each label sits in column 1 (the auto-sized leftmost
      // track) at row r+1, naming the variable on the y-axis of
      // every cell in that row. Rotated 90deg counterclockwise so
      // it reads bottom-up.
      for (var yi = 0; yi < n; yi++) {
        var ylab = document.createElement('div');
        ylab.textContent = axes[yi].label;
        ylab.style.gridColumn = '1 / span 1';
        ylab.style.gridRow = (yi + 1) + ' / span 1';
        ylab.style.writingMode = 'vertical-rl';
        ylab.style.transform = 'rotate(180deg)';
        ylab.style.display = 'flex';
        ylab.style.alignItems = 'center';
        ylab.style.justifyContent = 'center';
        ylab.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
        ylab.style.fontSize = '0.92em';
        ylab.style.opacity = '0.85';
        ylab.style.padding = '0 0.3em';
        hostEl.appendChild(ylab);
      }

      // ---- x-axis labels (bottom row, horizontal) ----------------
      // Each label sits in row n+1 (the auto-sized bottom track)
      // at column c+2 (skipping the leftmost y-label track),
      // naming the variable on the x-axis of every cell in that
      // column.
      for (var xi = 0; xi < n; xi++) {
        var xlab = document.createElement('div');
        xlab.textContent = axes[xi].label;
        xlab.style.gridColumn = (xi + 2) + ' / span 1';
        xlab.style.gridRow = (n + 1) + ' / span 1';
        xlab.style.textAlign = 'center';
        xlab.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
        xlab.style.fontSize = '0.92em';
        xlab.style.opacity = '0.85';
        xlab.style.padding = '0.2em 0';
        hostEl.appendChild(xlab);
      }

      // Per-cell builder: chart container only — no internal label,
      // axis names live on the grid edges. (Plot row r, plot col c
      // → grid row r+1, grid col c+2 because of the two label tracks.)
      function makeCell(row, col) {
        var cell = document.createElement('div');
        cell.style.gridRow    = (row + 1) + ' / span 1';
        cell.style.gridColumn = (col + 2) + ' / span 1';
        cell.style.minHeight = '0';
        cell.style.minWidth  = '0';
        cell.style.background = 'rgba(255,255,255,0.02)';
        cell.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.08))';
        cell.style.borderRadius = '3px';
        hostEl.appendChild(cell);
        return cell;
      }

      // ---- Diagonals: 1D marginals --------------------------------
      // Each cell delegates to a helper so the renderItem closure
      // captures *that call's* `rects` / `color` parameters by name
      // — not a `var`-hoisted loop variable that gets overwritten on
      // the next iteration. (The bug: var seriesRects/seriesColor
      // are function-scoped, so all closures end up reading the LAST
      // iteration's values; the first paint looks correct because
      // setOption renders synchronously, but resize-triggered
      // re-renders pick up the wrong data and the upper cells go
      // blank.)
      function renderDiagonalCell(inner, rects, color) {
        var ec1 = echarts.init(inner);
        ec1.setOption({
          backgroundColor: 'transparent',
          animation: false,
          grid: { left: 50, right: 12, top: 6, bottom: 24, containLabel: false },
          xAxis: {
            type: 'value', scale: true,
            axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
            axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
            axisLabel: { color: fg, opacity: 0.6, fontSize: 10, formatter: formatScalar },
            splitLine: { show: false },
          },
          yAxis: {
            type: 'value', scale: true,
            axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
            axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
            axisLabel: { color: fg, opacity: 0.5, fontSize: 10, formatter: formatScalar },
            splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
          },
          series: [{
            type: 'custom',
            data: rects,
            renderItem: function(_p, api) {
              var d = rects[_p.dataIndex];
              var lt = api.coord([d.x0, d.value[1]]);
              var rb = api.coord([d.x1, 0]);
              return {
                type: 'rect',
                shape: { x: lt[0], y: lt[1], width: rb[0] - lt[0], height: rb[1] - lt[1] },
                style: api.style({ fill: color, opacity: 0.55, stroke: color, lineWidth: 0.5 }),
              };
            },
            encode: { x: 0, y: 1 },
          }],
        });
      }
      for (var i = 0; i < n; i++) {
        var samples = axes[i].samples;
        var inner = makeCell(i, i);
        var hist = FlatPPLEngine.histogram.freedmanDiaconisHistogram(samples, histOptsBase);
        var rects = [];
        for (var k = 0; k < hist.xs.length; k++) {
          rects.push({
            value: [hist.xs[k], hist.ys[k]],
            x0: hist.binEdges[k],
            x1: hist.binEdges[k + 1],
          });
        }
        renderDiagonalCell(inner, rects, color);
      }

      if (n < 2) return;   // single-field record: only the diagonal

      // ---- Below-diagonal: 2D joint scatters ----------------------
      // Subsample if N is large enough that overplotting kills
      // readability. ECharts handles 50k points fine; 100k starts to
      // chug.
      //
      // Two paths:
      //   * Unweighted measure: take an even slice (deterministic
      //     stride) so the visual is stable across re-renders.
      //   * Weighted measure (e.g. an importance-weighted posterior
      //     from bayesupdate): plain stride would render the
      //     *prior's* atom positions with no regard for weights —
      //     the resulting scatter looks like the prior, even though
      //     the diagonal histograms (which use logWeights) correctly
      //     show the posterior. Fix: importance-resample atom
      //     indices via systematicResample, producing a uniform-
      //     weight subset that visualises the actual posterior.
      //     Per-binding seeded PRNG keeps the resample deterministic
      //     across re-renders.
      var anyN = axes[0].samples.length;
      var maxPoints = 20000;
      var indices;
      if (measure.logWeights) {
        var nOut = Math.min(anyN, maxPoints);
        var rsPrng = makeMainThreadPrng(nameSeed(ctx, bindingName + ':scatter'));
        indices = FlatPPLEngine.empirical.systematicResample(
          measure.logWeights, nOut, rsPrng);
      } else {
        var stride = anyN > maxPoints ? Math.ceil(anyN / maxPoints) : 1;
        var len = Math.ceil(anyN / stride);
        indices = new Int32Array(len);
        for (var ii = 0, jj = 0; ii < anyN; ii += stride, jj++) indices[jj] = ii;
      }
      // Pre-build one positional list per axis to avoid repeated
      // .samples lookups in the inner loop below.
      var cols = axes.map(function(a) { return a.samples; });

      for (var row = 1; row < n; row++) {
        for (var col = 0; col < row; col++) {
          var xCol = cols[col], yCol = cols[row];
          var inner2 = makeCell(row, col);
          var pts = new Array(indices.length);
          for (var p = 0; p < indices.length; p++) {
            var idx = indices[p];
            pts[p] = [xCol[idx], yCol[idx]];
          }
          // Point opacity scales with point count — denser data
          // gets more transparency so clouds don't saturate.
          var alpha = Math.max(0.05, Math.min(0.6, 800 / pts.length));
          var ec2 = echarts.init(inner2);
          ec2.setOption({
            backgroundColor: 'transparent',
            animation: false,
            grid: { left: 50, right: 12, top: 6, bottom: 24, containLabel: false },
            xAxis: {
              type: 'value', scale: true,
              axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.6, fontSize: 10, formatter: formatScalar },
              splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
            },
            yAxis: {
              type: 'value', scale: true,
              axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.5, fontSize: 10, formatter: formatScalar },
              splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
            },
            series: [{
              type: 'scatter',
              data: pts,
              symbolSize: 3,
              large: true,
              largeThreshold: 2000,
              itemStyle: { color: color, opacity: alpha },
            }],
          });
        }
      }
      // Above-diagonal cells are intentionally left empty (corner-
      // plot convention).
    }

    // ---- Fixed-value plot --------------------------------------------
    //
    // Phase-driven dispatch for compile-time-determinate bindings.
    // Records / tuples render the FlatPPL surface form as text;
    // scalars render the value as a single number when the per-atom
    // samples are constant, otherwise fall through to the histogram
    // path (engine-broadcast cases like lp_obs).
    function renderFixedRecord(plan) {
      showPlotMessage(ctx, 'Loading…', { hint: true });
      var planForCall = plan;
      getMeasure(ctx, plan.name).then(function(measure) {
        if (ctx.currentPlotPlan !== planForCall) return;
        renderConstantRecord(measure, plan.name);
      }).catch(function(err) {
        if (ctx.currentPlotPlan !== planForCall) return;
        showPlotMessage(ctx, 'Failed to load <strong>' + esc(plan.name) + '</strong>: '
          + esc(err && err.message || String(err)));
      });
    }

    // No separate renderFixedScalar — the existing
    // renderSamplesAndDensity already short-circuits to scalar-text
    // when samplesAreConstant. We keep mode='fixed-scalar' as the
    // plan label (so the source intent is visible in plan dumps /
    // logs) but route it through the same sample pipeline.

    // ---- Kernel sample plot ------------------------------------------
    //
    // Kernels (kernelof / functionof returning a measure) are plotted
    // by picking concrete values for their inputs (a preset, or
    // type-aware / source-empirical defaults), substituting those
    // into the kernel body, and sampling N atoms from the resulting
    // self-contained measure. The samples render via the existing
    // histogram / corner-plot pipeline in renderSamplesAndDensity.
    //
    // Cache: kernel-sample measures are stored in measureCache under
    // a synthetic key "<kernelName>|kernel-sample|<presetName>" so
    // switching presets doesn't re-sample, and switching back to a
    // previously-rendered kernel is instant.
    function renderKernelSampleForCurrent() {
      var plan = ctx.currentPlotPlan;
      if (!plan || plan.mode !== 'kernel-sample') return;
      var sig = plan.signature;
      var inputByKwarg = {};
      for (var k = 0; k < sig.inputs.length; k++) {
        inputByKwarg[sig.inputs[k].kwargName] = sig.inputs[k];
      }
      // Restrict (for now) to top-level scalar inputs — same limit
      // as the function/likelihood profile path.
      for (var ai = 0; ai < plan.axes.length; ai++) {
        if (plan.axes[ai].path && plan.axes[ai].path.length > 0) {
          showPlotMessage(ctx, 'Kernel plot: record / array inputs not yet supported '
            + '— try a kernel with scalar inputs only.',
            { hint: true });
          return;
        }
      }
      var active = activePresetFor(ctx, plan);
      // Cache key embeds the active preset's values directly so two
      // states of the same preset (with vs. without overrides, or
      // two different override sets) don't collide on cached
      // samples. Stable JSON suffices for our short kwarg lists.
      var cacheKey = plan.name + '|kernel-sample|' + (plan.presetName || '')
        + '|' + JSON.stringify(active.values || {});
      // Build the input env (paramName → number). Auto values for
      // axes not covered by the active preset (incl. modified
      // overrides) come from source-binding samples[0] (or type-
      // aware defaults for placeholder sources).
      var env = {};
      var bindingSourceLookups = [];   // [{paramName, sourceName}, ...]
      for (var a = 0; a < plan.axes.length; a++) {
        var ax = plan.axes[a];
        var inp = inputByKwarg[ax.kwargName];
        if (!inp) continue;
        if (active.values && Object.prototype.hasOwnProperty.call(active.values, ax.kwargName)) {
          env[inp.paramName] = active.values[ax.kwargName];
          continue;
        }
        env[inp.paramName] = defaultValueForLeafType(ax.leafType);
        if (ax.source && ax.source.kind === 'binding') {
          // Queue an empirical-sample lookup unconditionally —
          // tryGetMeasure soft-fails to null for sources that can't
          // produce samples (pure inputs like elementof). The
          // leaf-type default stays in env in that case.
          bindingSourceLookups.push({
            paramName: inp.paramName,
            sourceName: ax.source.name,
          });
        }
      }
      // Build the substituted measure IR. expandMeasureRefsInIR peels
      // any outer lawof and inlines measure-typed self-refs;
      // inlineForProfile (with all params named) inlines value-position
      // deterministic deps and rewrites self.<param> → %local.<param>;
      // substituteLocals replaces %local refs with their concrete env
      // values. Result: a self-contained measure IR with no refs.
      var paramNames = sig.inputs.map(function(inp) { return inp.paramName; });
      // We can do most of the IR work synchronously, but we need
      // the binding-source samples first to fill env entries.
      showPlotMessage(ctx, 'Sampling…', { cancellable: true, hint: true });
      var planForCall = plan;
      // Cache hit: use previously-sampled measure directly.
      if (ctx.measureCache.has(cacheKey)) {
        return Promise.resolve(ctx.measureCache.get(cacheKey)).then(function(m) {
          if (ctx.currentPlotPlan !== planForCall) return;
          renderKernelSampleMeasure(m, plan);
        });
      }
      // Two-phase pre-materialise:
      //   (1) binding-typed input sources we already know about (for
      //       a kernel with input `mu` whose source is `mu` in scope,
      //       we want samples[0] of that binding to seed the env);
      //   (2) self-refs captured from the outer scope by the kernel
      //       body (e.g. `sigma` referenced inside `iid(Normal(mu=0,
      //       sigma=sigma), 3)` even though `sigma` isn't a kernel
      //       input). These appear as (ref self sigma) after
      //       inlineForProfile because sigma is stochastic and so
      //       isn't inlined as a deterministic dep. substituteLocals
      //       only touches %local refs, so the materialise would
      //       otherwise fail with "unbound self reference".
      //
      // The actual captured self-refs are collected *after*
      // inlineForProfile because that pass inlines deterministic
      // deps. Anything still self-ref'd is genuinely a captured
      // stochastic/fixed dep from the outer scope.
      Promise.all(bindingSourceLookups.map(function(s) {
        return tryGetMeasure(ctx, s.sourceName);
      })).then(function(srcMeasures) {
        for (var i = 0; i < bindingSourceLookups.length; i++) {
          var sm = srcMeasures[i];
          if (sm && sm.samples && sm.samples.length > 0) {
            env[bindingSourceLookups[i].paramName] = sm.samples[0];
          }
        }
        var ir = sig.body;
        ir = FlatPPLEngine.orchestrator.expandMeasureRefsInIR(
          ir, ctx.derivationsState.derivations);
        // expandMeasureRefsInIR fails closed for refs whose derivation
        // was pruned by buildDerivations (e.g. `x` here, because its
        // distIR depends on the parameterized `mu`). The kernel-sample
        // path substitutes that parameter via env at materialise time,
        // so it still needs the structural shape. Re-run with the
        // bindings fallback to recover from binding.ir directly.
        if (ir && ir.kind === 'ref' && ir.ns === 'self') {
          var expanded = FlatPPLEngine.orchestrator.expandMeasureIR(
            ir.name, ctx.derivationsState.derivations,
            undefined, ctx.derivationsState.bindings);
          if (expanded) ir = expanded;
        }
        ir = FlatPPLEngine.orchestrator.inlineForProfile(
          ir, paramNames, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
        ir = FlatPPLEngine.orchestrator.substituteLocals(ir, env);

        // Captured self-refs (outer-scope stochastic / fixed bindings
        // that aren't kernel inputs) are no longer collapsed to
        // samples[0] here. materialiseConcreteMeasure threads
        // refArrays through to the worker's sampleN — atom i of the
        // kernel sample uses atom i of every captured ref, matching
        // the per-atom semantics of the closed-measure getMeasure
        // path. Per spec §04, stochastic ancestors that aren't
        // boundary inputs participate in the kernel's randomness.
        return materialiseConcreteMeasure(ir, ctx.SAMPLE_COUNT, nameSeed(ctx, plan.name));
      }).then(function(measure) {
        if (ctx.currentPlotPlan !== planForCall) return;
        ctx.measureCache.set(cacheKey, measure);
        renderKernelSampleMeasure(measure, plan);
      }).catch(function(err) {
        if (ctx.currentPlotPlan !== planForCall) return;
        showPlotMessage(ctx, 'Kernel plot failed: ' + esc(err && err.message || String(err)));
      });
    }

    /** The override entry for the active selection — auto's lives
        on the plan (per-binding), named presets live in the
        module-wide ctx.presetOverrides map. Returns null if none. */

    /** Whether the active preset selection currently has any value
        overrides. Drives the "(modified)" tag and the reset/persist
        button visibility on the Inputs control. Axis-range overrides
        moved to the Domain control's hasDomainOverrides(ctx, plan). */

    /** Write back an override entry for the active selection.
        Routes auto entries to the plan (per-binding), named
        entries to the module-wide store. Pass null to clear. */

    /** Get-or-create a fresh override entry for the active
        selection (caller will mutate it and call setOverrideFor
        to commit). */

    /** Effective {values} for a plan, merging base preset values
        with any override on top. Base values for named presets come
        from matchedPresets[i].values; for auto, base is an empty
        object (the dropdown "auto: …" label uses computeAutoValues
        separately, but env-substitution falls through to type
        defaults + source-sample materialisation when no explicit
        value is present). */

    /** Source-declared base values for the active preset (no
        overrides applied). For named presets this is
        matchedPresets[i].values; for auto, an empty object. */

    // ===================================================================
    // Domain (cartprod) override plumbing — mirrors the preset path
    // above, but stores per-kwarg [lo, hi] ranges instead of per-kwarg
    // values. Domains drive the x-axis range; presets drive the
    // non-swept input values.
    // ===================================================================

    /** Override entry for the active domain, or null when none. Auto
        domain's entry lives on plan.domainAutoOverride; named ones in
        ctx.domainOverrides keyed by name. */

    /** Get-or-create a domain override entry for the active selection.
        Caller mutates entry.ranges and commits via setDomainOverrideFor. */

    /** Commit (or clear, with null) a domain override entry. */

    /** True when the active domain has at least one ranged kwarg in
        its override entry. Used to gate visibility of the reset /
        save buttons. */

    /** Source-declared base ranges for the active domain (no overrides
        applied). For named domains this is matchedDomains[i].ranges;
        for auto, an empty object (the auto domain has no source-side
        ranges — the per-axis auto-fit code computes them on demand). */

    /** Effective ranges for the active domain — base merged with any
        override entry's ranges on top. Returns { kwarg: {lo, hi} }. */

    /** Held-constant kwargs for the active preset. Drawn from the
        engine's findMatchingPresets `fixedNames` (kwargs whose source
        value was wrapped in `fixed(...)` — spec §03's "hold constant
        during optimization" hint). For the auto preset no fixed-hint
        exists, so the set is always empty. Returns a Set so callers
        can do .has(name) directly. */

    // Shared icon-button helper used by the Inputs and Domain
    // toolbars' reset / save / save-as buttons. iconKey picks a
    // codicon (see CODICON_PATHS); title is the hover tooltip and the
    // accessible name. Buttons are icon-only — the toolbar already
    // shows the action verb implicitly via context, and dropping the
    // text labels frees the horizontal space the dropdown needs to
    // breathe.

    // Build a "Inputs: [auto / pars1 / …]" control fragment for the
    // profile / kernel-sample plot toolbar.
    //
    // We use a custom button-plus-popup instead of <select> so the
    // collapsed control can show just the short label
    // ("auto (modified)") while the open dropdown shows the longer
    // "name: theta1 = X, theta2 = Y" form. <select> doesn't support
    // different text in collapsed vs. open states across browsers
    // (the `label` attribute is spec but Chromium ignores it).
    function buildPresetControl(plan, onChange) {
      var frag = document.createDocumentFragment();
      if (!plan.axes || plan.axes.length === 0) return frag;

      // One row per preset name (auto plus each named preset). The
      // "(modified)" tag is appended to the label when the active
      // override entry has values — there's no separate
      // "<name> (modified)" row anymore. Switching presets just
      // changes plan.presetName; the override store decides whether
      // the row reads as modified.
      var entries = [];
      var presets = plan.matchedPresets || [];
      var autoValues = computeAutoValues(plan);

      function buildEntry(name, baseValues, isAuto) {
        var entryOverride = (name == null) ? plan.autoOverride : ctx.presetOverrides.get(name);
        var modified = !!(entryOverride && entryOverride.values
          && Object.keys(entryOverride.values).length > 0);
        var combined = Object.assign({}, baseValues, (entryOverride && entryOverride.values) || {});
        var displayName = isAuto ? 'auto' : name;
        var tag = modified ? ' (modified)' : '';
        return {
          name: name,
          modified: modified,
          shortLabel: displayName + tag,
          longLabel: displayName + tag + ': ' + presetValuesText(combined),
        };
      }

      entries.push(buildEntry(null, autoValues, true));
      for (var pi = 0; pi < presets.length; pi++) {
        entries.push(buildEntry(presets[pi].name, presets[pi].values || {}, false));
      }

      function isActive(entry) { return entry.name === plan.presetName; }
      var activeEntry = null;
      for (var k = 0; k < entries.length; k++) {
        if (isActive(entries[k])) { activeEntry = entries[k]; break; }
      }
      if (!activeEntry) activeEntry = entries[0];

      var wrap = document.createElement('span');
      wrap.style.position = 'relative';
      wrap.style.display = 'inline-flex';
      wrap.style.alignItems = 'center';
      wrap.style.gap = '0.3em';

      var lbl = document.createElement('label');
      lbl.textContent = 'Inputs:';
      lbl.style.opacity = '0.6';

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
      btn.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
      btn.style.border = '1px solid var(--vscode-dropdown-border, #555)';
      btn.style.padding = '2px 6px';
      btn.style.fontSize = '1em';
      btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
      btn.style.cursor = 'pointer';
      btn.style.borderRadius = '2px';
      btn.textContent = activeEntry.shortLabel + '  ▾';
      btn.title = activeEntry.longLabel;

      var panel = document.createElement('div');
      panel.style.position = 'absolute';
      panel.style.top = 'calc(100% + 4px)';
      panel.style.left = '0';
      panel.style.zIndex = '50';
      panel.style.minWidth = '100%';
      panel.style.maxHeight = '20em';
      panel.style.overflowY = 'auto';
      panel.style.padding = '0.2em';
      panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
      panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
      panel.style.borderRadius = '3px';
      panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
      panel.style.display = 'none';
      panel.style.whiteSpace = 'nowrap';

      var outsideClickHandler = null;
      function closePanel() {
        panel.style.display = 'none';
        if (outsideClickHandler) {
          document.removeEventListener('mousedown', outsideClickHandler);
          outsideClickHandler = null;
        }
      }
      function openPanel() {
        panel.style.display = 'block';
        // Defer the outside-click attach so the same click that
        // opened the panel doesn't immediately close it.
        setTimeout(function() {
          outsideClickHandler = function(ev) {
            if (!wrap.contains(ev.target)) closePanel();
          };
          document.addEventListener('mousedown', outsideClickHandler);
        }, 0);
      }

      btn.addEventListener('click', function(ev) {
        ev.stopPropagation();
        if (panel.style.display === 'none') openPanel(); else closePanel();
      });

      entries.forEach(function(entry) {
        var row = document.createElement('div');
        row.textContent = entry.longLabel;
        row.style.padding = '0.25em 0.6em';
        row.style.cursor = 'pointer';
        row.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
        row.style.borderRadius = '2px';
        if (isActive(entry)) {
          row.style.background = 'rgba(13, 113, 199, 0.45)';
          row.style.color = '#fff';
        }
        row.addEventListener('mouseenter', function() {
          if (!isActive(entry)) row.style.background = 'rgba(255,255,255,0.06)';
        });
        row.addEventListener('mouseleave', function() {
          if (!isActive(entry)) row.style.background = '';
        });
        row.addEventListener('click', function(ev) {
          ev.stopPropagation();
          plan.presetName = entry.name;
          closePanel();
          onChange();
        });
        panel.appendChild(row);
      });

      wrap.appendChild(lbl);
      wrap.appendChild(btn);
      wrap.appendChild(panel);

      frag.appendChild(wrap);

      // Reset / save action buttons live in a tight inline-flex group
      // so the two icons read as a single control rather than each
      // inheriting the toolbar's wider gap.
      // Reset button — visible only when the active selection has
      // overrides. Clears the override entry (auto → plan.autoOverride
      // = null; named → presetOverrides.delete(name)) and re-renders
      // through onChange. The dropdown row's "(modified)" tag then
      // disappears with no further user action.
      if (hasOverrides(ctx, plan)) {
        var actionGroup = document.createElement('span');
        actionGroup.style.display = 'inline-flex';
        actionGroup.style.gap = '2px';
        var resetBtn = makeActionButton(ctx, 'discard', 'Reset preset to source values');
        resetBtn.addEventListener('click', function(ev) {
          ev.stopPropagation();
          setOverrideFor(ctx, plan, null);
          onChange();
        });
        actionGroup.appendChild(resetBtn);

        // Persist button — visible when the active selection is a
        // named preset with overrides AND the host supports
        // writing (web edit-mode on, or VS Code) AND the source RHS
        // is preset(<kwarg>=<literal>, …) with no non-literal
        // values. Hidden otherwise so the user never sees a
        // disabled-looking button.
        //   - named preset + overrides → 'save'    (overwrite RHS)
        //   - auto + overrides         → 'save-as' (append new binding)
        // canPersistActive enforces the host-capability split: 'save'
        // needs host.editSource; 'save-as' additionally needs
        // host.promptForName.
        if (canPersistActive(plan)) {
          var isSaveAs = (plan.presetName == null);
          var persistBtn = makeActionButton(ctx, 
            isSaveAs ? 'save-as' : 'save',
            isSaveAs
              ? 'Save as new preset binding'
              : 'Save overrides into preset'
          );
          persistBtn.addEventListener('click', function(ev) {
            ev.stopPropagation();
            persistActive(plan);
          });
          actionGroup.appendChild(persistBtn);
        }
        frag.appendChild(actionGroup);
      }
      return frag;
    }

    // ----- Domain control: parallel of buildPresetControl, but for
    // cartprod(...) preset domains. Drives x-axis range per kwarg
    // rather than non-swept input values.

    /** Compose a human-readable summary of a domain's effective
        bounds, one entry per kwarg in `kwargOrder`. Reads bounded
        kwargs from `ranges` (lo/hi pairs from interval(...) fields
        or user overrides) and unbounded kwargs from `setNames`
        (bare `reals` / `posreals` / … fields). */

    function buildDomainControl(plan, onChange) {
      var frag = document.createDocumentFragment();
      if (!plan.axes || plan.axes.length === 0) return frag;

      var domains = plan.matchedDomains || [];
      // kwarg display order: take it from plan.signature.inputs so
      // every entry — including modifications — reads in the same
      // order as the source signature, regardless of which kwargs
      // got user overrides.
      var inputs = (plan.signature && plan.signature.inputs) || [];
      var kwargOrder = [];
      for (var ki = 0; ki < inputs.length; ki++) {
        if (inputs[ki].kwargName) kwargOrder.push(inputs[ki].kwargName);
      }

      function buildEntry(name, baseRanges, baseSetNames, isAuto) {
        var entryOverride = (name == null)
          ? plan.domainAutoOverride
          : ctx.domainOverrides.get(name);
        var modified = !!(entryOverride && entryOverride.ranges
          && Object.keys(entryOverride.ranges).length > 0);
        var combinedRanges = Object.assign({}, baseRanges,
          (entryOverride && entryOverride.ranges) || {});
        // User overrides shadow source named-set fields: drop those
        // entries from setNames so the kwarg renders with the
        // bounded interval rather than both.
        var combinedSetNames = Object.assign({}, baseSetNames);
        for (var k in combinedRanges) {
          if (Object.prototype.hasOwnProperty.call(combinedRanges, k)) {
            delete combinedSetNames[k];
          }
        }
        var displayName = isAuto ? 'auto' : name;
        var tag = modified ? ' (modified)' : '';
        return {
          name: name,
          modified: modified,
          shortLabel: displayName + tag,
          longLabel: displayName + tag + ': '
            + domainBoundsText(kwargOrder, combinedRanges, combinedSetNames),
        };
      }

      var entries = [];
      entries.push(buildEntry(null, {}, {}, true));
      for (var di = 0; di < domains.length; di++) {
        entries.push(buildEntry(
          domains[di].name,
          domains[di].ranges || {},
          domains[di].setNames || {},
          false));
      }

      function isActive(entry) { return entry.name === plan.domainName; }
      var activeEntry = null;
      for (var k = 0; k < entries.length; k++) {
        if (isActive(entries[k])) { activeEntry = entries[k]; break; }
      }
      if (!activeEntry) activeEntry = entries[0];

      var wrap = document.createElement('span');
      wrap.style.position = 'relative';
      wrap.style.display = 'inline-flex';
      wrap.style.alignItems = 'center';
      wrap.style.gap = '0.3em';

      var lbl = document.createElement('label');
      lbl.textContent = 'Domain:';
      lbl.style.opacity = '0.6';

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
      btn.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
      btn.style.border = '1px solid var(--vscode-dropdown-border, #555)';
      btn.style.padding = '2px 6px';
      btn.style.fontSize = '1em';
      btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
      btn.style.cursor = 'pointer';
      btn.style.borderRadius = '2px';
      btn.textContent = activeEntry.shortLabel + '  ▾';
      btn.title = activeEntry.longLabel;

      var panel = document.createElement('div');
      panel.style.position = 'absolute';
      panel.style.top = 'calc(100% + 4px)';
      panel.style.left = '0';
      panel.style.zIndex = '50';
      panel.style.minWidth = '100%';
      panel.style.maxHeight = '20em';
      panel.style.overflowY = 'auto';
      panel.style.padding = '0.2em';
      panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
      panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
      panel.style.borderRadius = '3px';
      panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
      panel.style.display = 'none';
      panel.style.whiteSpace = 'nowrap';

      var outsideClickHandler = null;
      function closePanel() {
        panel.style.display = 'none';
        if (outsideClickHandler) {
          document.removeEventListener('mousedown', outsideClickHandler);
          outsideClickHandler = null;
        }
      }
      function openPanel() {
        panel.style.display = 'block';
        setTimeout(function() {
          outsideClickHandler = function(ev) {
            if (!wrap.contains(ev.target)) closePanel();
          };
          document.addEventListener('mousedown', outsideClickHandler);
        }, 0);
      }

      btn.addEventListener('click', function(ev) {
        ev.stopPropagation();
        if (panel.style.display === 'none') openPanel(); else closePanel();
      });

      entries.forEach(function(entry) {
        var row = document.createElement('div');
        row.textContent = entry.longLabel;
        row.style.padding = '0.25em 0.6em';
        row.style.cursor = 'pointer';
        row.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
        row.style.borderRadius = '2px';
        if (isActive(entry)) {
          row.style.background = 'rgba(13, 113, 199, 0.45)';
          row.style.color = '#fff';
        }
        row.addEventListener('mouseenter', function() {
          if (!isActive(entry)) row.style.background = 'rgba(255,255,255,0.06)';
        });
        row.addEventListener('mouseleave', function() {
          if (!isActive(entry)) row.style.background = '';
        });
        row.addEventListener('click', function(ev) {
          ev.stopPropagation();
          plan.domainName = entry.name;
          closePanel();
          onChange();
        });
        panel.appendChild(row);
      });

      wrap.appendChild(lbl);
      wrap.appendChild(btn);
      wrap.appendChild(panel);
      frag.appendChild(wrap);

      // Reset / save / save-as icons mirror the Inputs control,
      // grouped in a tight inline-flex span so they read as one
      // pair rather than picking up the toolbar's wider gap.
      if (hasDomainOverrides(ctx, plan)) {
        var actionGroup = document.createElement('span');
        actionGroup.style.display = 'inline-flex';
        actionGroup.style.gap = '2px';
        var resetBtn = makeActionButton(ctx, 'discard', 'Reset domain to source ranges');
        resetBtn.addEventListener('click', function(ev) {
          ev.stopPropagation();
          setDomainOverrideFor(ctx, plan, null);
          onChange();
        });
        actionGroup.appendChild(resetBtn);

        if (canPersistDomain(plan)) {
          var isSaveAs = (plan.domainName == null);
          var persistBtn = makeActionButton(ctx, 
            isSaveAs ? 'save-as' : 'save',
            isSaveAs
              ? 'Save as new cartprod domain binding'
              : 'Save range overrides into cartprod'
          );
          persistBtn.addEventListener('click', function(ev) {
            ev.stopPropagation();
            persistDomain(plan);
          });
          actionGroup.appendChild(persistBtn);
        }
        frag.appendChild(actionGroup);
      }

      return frag;
    }

    /** Persist is supported when there's an override AND the ctx.host
        adapter can write (ctx.host.editSource defined and ctx.host.canPersist
        returns true). For named presets the source RHS also has to
        be literal-friendly; for auto we additionally need
        ctx.host.promptForName for the new-binding name. Hidden
        otherwise so the user never sees a disabled-looking button. */
    function canPersistActive(plan) {
      if (!hasOverrides(ctx, plan)) return false;
      if (!ctx.host || typeof ctx.host.editSource !== 'function') return false;
      if (typeof ctx.host.canPersist === 'function' && !ctx.host.canPersist()) return false;
      if (ctx.host.canPersist === false) return false;
      if (plan.presetName == null) {
        return typeof ctx.host.promptForName === 'function';
      }
      if (!ctx.currentBindings) return false;
      var b = ctx.currentBindings.get(plan.presetName);
      if (!b || !b.node || !b.node.value
          || b.node.value.type !== 'CallExpr'
          || !b.node.value.callee
          || b.node.value.callee.name !== 'record') return false;
      // Persist only when every field is a literal — possibly wrapped
      // in fixed(...) which is identity at runtime (spec §03) and just
      // a "hold constant" hint we preserve when rewriting. Anything
      // more structural (refs, nested calls) means the source isn't
      // edit-in-place writable; canPersist returns false so the
      // toolbar hides the button rather than offering a broken
      // write-back.
      var args = b.node.value.args || [];
      for (var i = 0; i < args.length; i++) {
        var a = args[i];
        if (a.type !== 'KeywordArg' || !a.value) return false;
        var v = a.value;
        if (v.type === 'CallExpr' && v.callee && v.callee.name === 'fixed'
            && Array.isArray(v.args) && v.args.length === 1) {
          v = v.args[0];
        }
        if (v.type !== 'NumberLiteral' && v.type !== 'BoolLiteral') return false;
      }
      return true;
    }

    /** Format a JS number for source emission. We use String(v)
        rather than formatScalar because formatScalar rounds to 4
        significant figures for display; source needs full
        precision. */

    /** Build the replacement source text for a named preset-point
        record binding, merging the current source RHS kwargs with the
        active override values. Preserves source kwarg order and
        re-wraps overridden values in `fixed(...)` when the original
        source did so — the spec's "held constant" hint must survive
        the round-trip, otherwise persisting a tweak to an
        optimization starting point would silently strip the
        hold-constant annotation. */
    function buildPersistedPresetLine(plan) {
      var active = activePresetFor(ctx, plan);
      var b = ctx.currentBindings.get(plan.presetName);
      var srcArgs = b.node.value.args || [];
      var parts = [];
      for (var i = 0; i < srcArgs.length; i++) {
        var sa = srcArgs[i];
        var kwarg = sa.name;
        var srcVal = sa.value;
        var wasFixed = srcVal && srcVal.type === 'CallExpr'
                     && srcVal.callee && srcVal.callee.name === 'fixed';
        var innerSrc = wasFixed ? srcVal.args[0] : srcVal;
        var override = active.values
                    && Object.prototype.hasOwnProperty.call(active.values, kwarg);
        var v = override ? active.values[kwarg]
                         : (innerSrc && innerSrc.value);
        var text = formatScalarForSource(ctx, v);
        if (wasFixed) text = 'fixed(' + text + ')';
        parts.push(kwarg + ' = ' + text);
      }
      return plan.presetName + ' = record(' + parts.join(', ') + ')';
    }

    /** Invoke ctx.host.persistPreset for the active selection. Routes
        to "replace existing binding" or "append new binding"
        depending on whether the active selection is a named
        preset or auto. Host applies the edit; the next source-
        update cycle reconciles the override away because the
        source values now match. */
    function persistActive(plan) {
      if (!canPersistActive(plan)) return;
      if (plan.presetName == null) {
        persistAutoAsNewBinding(plan);
      } else {
        persistNamedPreset(plan);
      }
    }

    function persistNamedPreset(plan) {
      var b = ctx.currentBindings.get(plan.presetName);
      var newText = buildPersistedPresetLine(plan);
      try {
        ctx.host.editSource({
          range: {
            start: { line: b.node.loc.start.line, col: b.node.loc.start.col },
            end:   { line: b.node.loc.end.line,   col: b.node.loc.end.col },
          },
          newText: newText,
        });
      } catch (err) {
        console.error('[viewer] editSource (named persist) failed:', err);
      }
    }

    /** Auto persist: ask the ctx.host to prompt for a binding name,
        then ask it to append the new preset binding at end-of-
        source. The two-step contract keeps line/text construction
        and queuing the next-active-preset hint in the viewer
        (single source of truth); each ctx.host implements only the
        primitives (UI prompt + edit application). */
    function persistAutoAsNewBinding(plan) {
      if (typeof ctx.host.promptForName !== 'function'
          || typeof ctx.host.editSource !== 'function') {
        console.warn('[viewer] persist auto: ctx.host missing promptForName / editSource');
        return;
      }
      var autoValues = computeAutoValues(plan);
      var override = plan.autoOverride;
      var combined = Object.assign({}, autoValues, (override && override.values) || {});
      var parts = [];
      for (var k in combined) {
        if (!Object.prototype.hasOwnProperty.call(combined, k)) continue;
        var v = combined[k];
        if (!Number.isFinite(v)) continue;
        parts.push(k + ' = ' + formatScalarForSource(ctx, v));
      }
      if (parts.length === 0) return;
      var existingNames = [];
      if (ctx.currentBindings) ctx.currentBindings.forEach(function(_b, n) { existingNames.push(n); });
      var pairsText = parts.join(', ');
      var suggested = (plan.name || 'inputs') + '_default';
      Promise.resolve(ctx.host.promptForName({
        suggested: suggested,
        existingNames: existingNames,
      })).then(function(name) {
        if (!name) return;
        ctx.pendingPresetName = name;
        ctx.host.editSource({
          range: null,
          newText: name + ' = record(' + pairsText + ')',
        });
      }).catch(function(err) {
        console.error('[viewer] persistAutoAsNewBinding failed:', err);
      });
    }

    // ===================================================================
    // Domain persist — parallel of canPersistActive / persistActive
    // ===================================================================

    var KNOWN_NAMED_SETS = {
      reals: 1, posreals: 1, nonnegreals: 1, unitinterval: 1,
      integers: 1, posintegers: 1, nonnegintegers: 1, booleans: 1,
    };

    /** Whether an AST node is a recognized cartprod field value:
        either `interval(NumberLiteral, NumberLiteral)` or a bare
        named-set reference. Used by canPersistDomain to gate the
        save button. */

    /** Serialize a recognized cartprod field value back to source
        text. Mirrors isPersistableSetField — caller has already
        gated. */
    function setFieldToSource(v) {
      if (v.type === 'Identifier') return v.name;
      // interval(NumLit, NumLit)
      return 'interval('
        + formatScalarForSource(ctx, v.args[0].value) + ', '
        + formatScalarForSource(ctx, v.args[1].value) + ')';
    }

    /** Pick a "natural" set-source-text for one of a plan's input
        kwargs — used when the user persists a partial domain and we
        want to fill the unset kwargs with something matchable rather
        than dropping them (which would make the resulting cartprod
        fail findMatchingDomains' shape check). Strategy: ask the
        engine for the axis's base set descriptor; map known kinds to
        their source names; fall back to 'reals' for empirical /
        unresolved descriptors.

        For multi-axis kwargs (vector / record-typed inputs) the
        simple per-axis mapping is wrong (we'd need cartpow /
        cartprod), so we surface 'reals' there too — the user can
        edit the source by hand if they want a tighter set. */
    function defaultSetSourceForKwarg(plan, kwargName) {
      if (!plan.axes) return 'reals';
      var matching = [];
      for (var i = 0; i < plan.axes.length; i++) {
        if (plan.axes[i].kwargName === kwargName) matching.push(plan.axes[i]);
      }
      if (matching.length !== 1) return 'reals';  // non-scalar — defer
      var bindings = ctx.derivationsState && ctx.derivationsState.bindings;
      var d = null;
      try {
        d = FlatPPLEngine.orchestrator.resolveAxisBaseSet(matching[0].source, bindings);
      } catch (_) { d = null; }
      if (!d) return 'reals';
      switch (d.kind) {
        case 'reals':           return 'reals';
        case 'posreals':        return 'posreals';
        case 'nonnegreals':     return 'nonnegreals';
        case 'integers':        return 'integers';
        case 'posintegers':     return 'posintegers';
        case 'nonnegintegers':  return 'nonnegintegers';
        case 'booleans':        return 'booleans';
        case 'interval':
          if (d.lo === 0 && d.hi === 1) return 'unitinterval';
          return 'interval('
            + formatScalarForSource(ctx, d.lo) + ', '
            + formatScalarForSource(ctx, d.hi) + ')';
        default:                return 'reals';  // empirical / unknown
      }
    }

    /** Persist is supported when there's an override AND the ctx.host
        adapter can write. For named domains every source field has
        to be a recognized set form (interval-with-literal-bounds OR
        a named set like `reals` / `posreals` / …). For auto we
        additionally need ctx.host.promptForName for the new-binding name. */
    function canPersistDomain(plan) {
      if (!hasDomainOverrides(ctx, plan)) return false;
      if (!ctx.host || typeof ctx.host.editSource !== 'function') return false;
      if (typeof ctx.host.canPersist === 'function' && !ctx.host.canPersist()) return false;
      if (ctx.host.canPersist === false) return false;
      if (plan.domainName == null) {
        return typeof ctx.host.promptForName === 'function';
      }
      if (!ctx.currentBindings) return false;
      var b = ctx.currentBindings.get(plan.domainName);
      if (!b || !b.node || !b.node.value
          || b.node.value.type !== 'CallExpr'
          || !b.node.value.callee
          || b.node.value.callee.name !== 'cartprod') return false;
      var args = b.node.value.args || [];
      for (var i = 0; i < args.length; i++) {
        var a = args[i];
        if (a.type !== 'KeywordArg' || !a.value) return false;
        if (!isPersistableSetField(a.value)) return false;
      }
      return true;
    }

    function persistDomain(plan) {
      if (!canPersistDomain(plan)) return;
      if (plan.domainName == null) {
        persistAutoDomainAsNewBinding(plan);
      } else {
        persistNamedDomain(plan);
      }
    }

    /** Build the replacement source text for a named cartprod domain,
        merging source-declared field values with overridden ranges.
        For each kwarg:
          - if the override has a range → emit interval(lo, hi)
          - else → preserve the source field as-is (interval(...)
            with original bounds, or the bare named-set reference)
        Preserves source kwarg order. */
    function buildPersistedDomainLine(plan) {
      var b = ctx.currentBindings.get(plan.domainName);
      var srcArgs = b.node.value.args || [];
      var override = domainOverrideEntryFor(ctx, plan);
      var or = (override && override.ranges) || {};
      var parts = [];
      for (var i = 0; i < srcArgs.length; i++) {
        var sa = srcArgs[i];
        var kwarg = sa.name;
        if (Object.prototype.hasOwnProperty.call(or, kwarg)) {
          parts.push(kwarg + ' = interval('
            + formatScalarForSource(ctx, or[kwarg].lo) + ', '
            + formatScalarForSource(ctx, or[kwarg].hi) + ')');
        } else {
          parts.push(kwarg + ' = ' + setFieldToSource(sa.value));
        }
      }
      return plan.domainName + ' = cartprod(' + parts.join(', ') + ')';
    }

    function persistNamedDomain(plan) {
      var b = ctx.currentBindings.get(plan.domainName);
      var newText = buildPersistedDomainLine(plan);
      try {
        ctx.host.editSource({
          range: {
            start: { line: b.node.loc.start.line - 1, col: 0 },
            end:   { line: b.node.loc.end.line   - 1, col: 1000000 },
          },
          newText: newText,
        });
      } catch (err) {
        console.error('[viewer] persistNamedDomain failed:', err);
      }
    }

    /** Append a fresh cartprod(...) binding capturing the current
        domain override. Asks the ctx.host for a name via promptForName.
        Fills *every* input kwarg in the signature: overridden ones
        get `interval(lo, hi)`, the rest get their natural base set
        (`reals` / `posreals` / …) via defaultSetSourceForKwarg.
        Filling in the unset kwargs keeps the new cartprod matchable
        in findMatchingDomains' shape check — otherwise a partial
        domain like `cartprod(theta1 = interval(-4, 4))` wouldn't
        appear in the Domain dropdown after persist. */
    function persistAutoDomainAsNewBinding(plan) {
      if (typeof ctx.host.promptForName !== 'function'
          || typeof ctx.host.editSource !== 'function') {
        console.warn('[viewer] persist domain auto: ctx.host missing promptForName / editSource');
        return;
      }
      var override = plan.domainAutoOverride;
      var ranges = (override && override.ranges) || {};
      // Enumerate every signature input so the resulting cartprod has
      // full shape coverage. Per-kwarg precedence:
      //   1. user override range          → interval(lo, hi)
      //   2. auto-fit cached for this kwarg in profileRangeCache
      //      (the plot engine populated it when the user previously
      //      had this kwarg selected as sweep axis) → interval(lo, hi)
      //   3. natural base set from the input's source descriptor
      //      → bare named set (reals / posreals / …)
      // Step 2 means an axis the user looked at but never edited
      // still persists with its observed bounds rather than being
      // weakened to the natural set.
      var inputs = (plan.signature && plan.signature.inputs) || [];
      var parts = [];
      for (var i = 0; i < inputs.length; i++) {
        var kw = inputs[i].kwargName;
        if (!kw) continue;
        var r = Object.prototype.hasOwnProperty.call(ranges, kw) ? ranges[kw] : null;
        if (r && Number.isFinite(r.lo) && Number.isFinite(r.hi)) {
          parts.push(kw + ' = interval('
            + formatScalarForSource(ctx, r.lo) + ', '
            + formatScalarForSource(ctx, r.hi) + ')');
          continue;
        }
        var cached = ctx.profileRangeCache.get(
          plan.name + '|' + kw + '|D=' + (plan.domainName || ''));
        if (cached && Number.isFinite(cached.lo) && Number.isFinite(cached.hi)) {
          parts.push(kw + ' = interval('
            + formatScalarForSource(ctx, cached.lo) + ', '
            + formatScalarForSource(ctx, cached.hi) + ')');
          continue;
        }
        parts.push(kw + ' = ' + defaultSetSourceForKwarg(plan, kw));
      }
      if (parts.length === 0) return;
      var existingNames = [];
      if (ctx.currentBindings) ctx.currentBindings.forEach(function(_b, n) { existingNames.push(n); });
      var pairsText = parts.join(', ');
      var suggested = (plan.name || 'domain') + '_domain';
      Promise.resolve(ctx.host.promptForName({
        suggested: suggested,
        existingNames: existingNames,
      })).then(function(name) {
        if (!name) return;
        ctx.pendingDomainName = name;
        ctx.host.editSource({
          range: null,
          newText: name + ' = cartprod(' + pairsText + ')',
        });
      }).catch(function(err) {
        console.error('[viewer] persistAutoDomainAsNewBinding failed:', err);
      });
    }

    // Strip the outer "record(...)" wrapper from formatValue's
    // output so the dropdown reads cleanly:
    //   record(theta1 = 1.4, theta2 = 1.0)  →  theta1 = 1.4, theta2 = 1.0

    // Synthesise the auto-mode fixed-input values for a profile /
    // kernel-sample plan, matching the renderer's fallback
    // behaviour: source-binding axes use the cached samples[0] (or
    // type default if not yet cached); placeholder/other axes use
    // the type default. Returned as { kwargName: value }.
    function computeAutoValues(plan) {
      var out = {};
      var axes = plan.axes || [];
      for (var i = 0; i < axes.length; i++) {
        var ax = axes[i];
        var def = defaultValueForLeafType(ax.leafType);
        if (ax.source && ax.source.kind === 'binding'
            && ctx.measureCache && ctx.measureCache.has(ax.source.name)) {
          var m = ctx.measureCache.get(ax.source.name);
          if (m && m.samples && m.samples.length > 0) def = m.samples[0];
        }
        out[ax.kwargName] = def;
      }
      return out;
    }

    // Render a kernel-sampled empirical measure. Record / tuple /
    // array measures route through renderRecordMarginals with the
    // preset dropdown injected into its toolbar (no extra row).
    // Scalar sampled measures use a simple histogram via
    // renderSamplesAndDensity; constant scalars / records get the
    // text-render path. This avoids wrapping the existing renderers
    // in a flex-column container that compressed the corner-plot
    // cells under a layout race.
    // Kernel sampling produces an empirical measure exactly like any
    // other measure binding — semantically a kernel IS a nullary
    // kernel once its inputs are bound. So the entire shape dispatch
    // (constant text / record-marginals / scalar histogram) is shared
    // with the standard measure path via renderEmpiricalMeasure; the
    // only kernel-specific bit is the preset selector in the toolbar.
    // Discreteness defaults to false: kernel bodies don't go through
    // orchestrator typing, so we don't have a discrete flag to plumb.
    // (No analytical density either — kernel bodies are empirical.)
    //
    // toolbarControls is passed as a *builder thunk*, not a static
    // Element / DocumentFragment. The corner-plot rerender path
    // (mode toggle, axis selection) blows away and rebuilds the
    // toolbar; appendChild on a DocumentFragment moves its children
    // out and leaves it empty, so a static fragment would render
    // once and disappear on every subsequent rebuild. The thunk
    // produces fresh DOM each call.
    function renderKernelSampleMeasure(measure, plan) {
      // Always wire the input-selection toolbar when the plan has
      // axes — the "auto" option still carries useful information
      // even without user-declared presets, and the control stays
      // visible across bindings for consistency.
      var hasAxes = plan.axes && plan.axes.length > 0;
      // Kernel-sample renders a histogram of empirical draws with an
      // auto-fit x-axis range — the Domain selector (which drives a
      // swept x-axis range) doesn't apply here, so we only mount the
      // Inputs control.
      var toolbarBuilder = hasAxes
        ? function() {
            return buildPresetControl(plan, function() {
              renderKernelSampleForCurrent();
            });
          }
        : null;
      renderEmpiricalMeasure(measure, {
        name: plan.name,
        mode: 'samples',
        discrete: false,
        analyticalIR: null,
        toolbarControls: toolbarBuilder,
      });
    }

    // Recursively materialise a self-contained measure IR (no
    // measure-position self-refs; %local refs already substituted to
    // literals) into an EmpiricalMeasure. Used by the kernel-sample
    // path to draw N atoms from a kernel body at fixed parameter
    // values.
    //
    // Cases:
    //   leaf distribution (Normal, Exp, …)  → worker.sampleN with
    //                                         refArrays for captured
    //                                         self-refs (per-atom
    //                                         semantics matching the
    //                                         closed-measure getMeasure
    //                                         path)
    //   joint(field=M, …) / record(…)       → recordMeasure(materialise(field), …)
    //   iid(M, dim, …)                      → arrayMeasure(materialise(M, count×∏dims))
    //   lawof(M)                            → recurse into M (lawof is a no-op on measures)
    //
    // weighted / normalize / superpose / kernels-applied-to-iid
    // surface as a clear error rather than a silent broken plot.
    function materialiseConcreteMeasure(ir, count, seed) {
      if (!ir) return Promise.reject(new Error('materialiseConcreteMeasure: null IR'));
      if (ir.kind !== 'call') {
        return Promise.reject(new Error(
          "materialiseConcreteMeasure: non-call IR (kind '" + ir.kind + "')"));
      }
      if (ir.op === 'lawof' && Array.isArray(ir.args) && ir.args.length === 1) {
        return materialiseConcreteMeasure(ir.args[0], count, seed);
      }
      if (ir.op === 'iid' && Array.isArray(ir.args) && ir.args.length >= 2) {
        var inner = ir.args[0];
        var dims = [];
        for (var di = 1; di < ir.args.length; di++) {
          var d = ir.args[di];
          if (!d || d.kind !== 'lit' || !Number.isInteger(d.value)) {
            return Promise.reject(new Error('materialiseConcreteMeasure: iid dim must be integer literal'));
          }
          dims.push(d.value);
        }
        var k = dims.reduce(function(p, n) { return p * n; }, 1);
        // Leaf-distribution inner: use sampleN's `repeat` so the per-
        // atom refArrays line up — atom i gets refArrays[i], then k
        // independent draws share it. Mirrors getMeasure's iid path.
        // Naive recursion with count*k would mis-index refArrays
        // (only `count` entries available, repeated k times by the
        // atom index — out-of-bounds for i >= count).
        var SAMPLEABLE = FlatPPLEngine.orchestrator.SAMPLEABLE_DISTRIBUTIONS;
        if (inner.kind === 'call' && SAMPLEABLE && SAMPLEABLE.has(inner.op)) {
          return collectRefArrays(ctx, inner).then(function(refArrays) {
            return sendWorker(ctx, {
              type: 'sampleN', ir: inner, count: count, repeat: k,
              refArrays: refArrays, seed: seed,
            });
          }).then(function(reply) {
            return FlatPPLEngine.empirical.arrayMeasure(reply.samples, dims, null);
          });
        }
        // Non-leaf inner (nested iid / record / joint inside iid).
        // The recursive form keeps the structure but doesn't handle
        // captured refs correctly under expansion — flag if we ever
        // hit it in practice. Today's kernel-sample bodies all
        // bottom out at leaf distributions after the IR pipeline.
        return materialiseConcreteMeasure(inner, count * k, seed).then(function(innerM) {
          return FlatPPLEngine.empirical.arrayMeasure(innerM.samples, dims, null);
        });
      }
      if ((ir.op === 'joint' || ir.op === 'record') && Array.isArray(ir.fields)) {
        var fieldNames = ir.fields.map(function(f) { return f.name; });
        var fieldIRs = ir.fields.map(function(f) { return f.value; });
        return Promise.all(fieldIRs.map(function(v, i) {
          return materialiseConcreteMeasure(v, count,
            seed != null ? (seed ^ (i + 1) * 0x9e3779b1) : null);
        })).then(function(subs) {
          var fields = {};
          for (var i = 0; i < fieldNames.length; i++) fields[fieldNames[i]] = subs[i];
          return FlatPPLEngine.empirical.recordMeasure(fields, null);
        });
      }
      // Leaf distribution (or unrecognised op — sampleN throws if
      // it's not in the registry). Captured self-refs in the dist's
      // kwargs (e.g. `Normal(mu = lit, sigma = pow(ref self sqrt_sigma, 2))`
      // after substituteLocals) are resolved per-atom via refArrays
      // — same mechanism getMeasure uses for closed-measure
      // sampling. Fixed-phase refs flow through the worker's session
      // env, so collectRefArrays drops them.
      return collectRefArrays(ctx, ir).then(function(refArrays) {
        return sendWorker(ctx, {
          type: 'sampleN', ir: ir, count: count, seed: seed,
          refArrays: refArrays,
        });
      }).then(function(reply) {
        // Phase 8: hand-built Measures populate `.value` for
        // consistency with materialiser-produced ones.
        var data = reply.samples;
        return {
          samples: data,
          value: { shape: [data.length], data: data },
          logWeights: null,
        };
      });
    }

    // ---- Profile plot ------------------------------------------------
    //
    // Type-aware default value for an axis leafType. Used to populate
    // fixedEnv for non-swept inputs at first plot. Posreals defaults
    // to 1.0 (avoids degenerate cases like sigma=0); intervals
    // default to the midpoint; integers default to 0; etc. F4b will
    // let the user override these via the fixed-values panel.

    // Default sweep range for an axis from leaf-type alone. Used as
    // the final fallback after the axis-set descriptor and empirical
    // backref both fail to give a range.

    // Map a structural set descriptor (from
    // orchestrator.resolveAxisBaseSet) to a concrete sweep range.
    // Empirical sets defer to the caller (the viewer materialises
    // the source binding and computes a 4-σ quantile range).
    //   reals          → [-5, 5]            (Gaussian-like default)
    //   posreals       → [eps, 5]           (avoid 0 boundary for log etc.)
    //   nonnegreals    → [0, 5]
    //   integers       → [-10, 10]
    //   posintegers    → [1, 20]
    //   nonnegintegers → [0, 20]
    //   booleans       → [0, 1]
    //   interval(a, b) → [a, b]

    // Resolve the auto-range for a swept axis. Three-tier fallback:
    //   1. Set descriptor (interval / reals / posreals / …) from
    //      resolveAxisBaseSet — covers identifier-bound elementof
    //      bindings.
    //   2. Empirical 4-σ quantile from the source binding's samples
    //      — covers identifier boundaries pointing at stochastic /
    //      derived bindings.
    //   3. Leaf-type default — placeholders, anything unresolved.
    // Returns a Promise<[lo, hi]> since step 2 may need to await
    // getMeasure(...).

    // Render the profile plot for a callable binding. Builds env with
    // default values for non-swept inputs, picks a default range for
    // the swept axis, fires worker.profileN, then draws a line plot.
    //
    // Limitations (F4a):
    //   - Top-level scalar inputs only — record / array inputs
    //     classify a path on each axis, but populating a fixedEnv with
    //     a record literal is F4b work.
    //   - Plain kernelof bindings (not wrapped in likelihoodof) need
    //     an obs value the user has to provide; defer to F4b.
    function renderProfilePlotForCurrent() {
      var plan = ctx.currentPlotPlan;
      if (!plan || plan.mode !== 'profile') return;
      var sig = plan.signature;
      var axes = plan.axes;
      var sweepAxis = null;
      for (var i = 0; i < axes.length; i++) {
        if (axes[i].key === plan.sweepKey) { sweepAxis = axes[i]; break; }
      }
      if (!sweepAxis) {
        showPlotMessage(ctx, 'Profile plot: no axis selected for <strong>'
          + esc(plan.name) + '</strong>.', { hint: true });
        return;
      }
      // Mode dispatch. Kernels are routed to renderKernelSampleForCurrent
      // by buildPlotPlan; profile mode here only sees function /
      // likelihood bindings.
      var mode = sig.kind === 'function' ? 'function' : 'logdensity';
      // Build paramName ↔ kwargName index for env construction.
      var inputByKwarg = {};
      for (var k = 0; k < sig.inputs.length; k++) {
        inputByKwarg[sig.inputs[k].kwargName] = sig.inputs[k];
      }
      // Top-level-scalar-only restriction for F4a.
      for (var a = 0; a < axes.length; a++) {
        if (axes[a].path && axes[a].path.length > 0) {
          showPlotMessage(ctx, 'Profile plot: record / array inputs not yet supported — '
            + 'try a binding with scalar inputs only.',
            { hint: true });
          return;
        }
      }
      // Build fixedEnv with three-tier precedence:
      //   1. Active preset's value for the kwarg (highest) — base
      //      preset values overridden by (modified) overrides.
      //   2. Source binding's samples[0] for binding-source axes
      //      (resolved later, after materialisation).
      //   3. Type-aware default for the leaf type (lowest).
      // The activePreset lookup short-circuits the materialise path
      // for any axis whose kwarg the preset (incl. modified
      // overrides) covers — so when the user picks a preset, we
      // don't even fetch the source binding.
      var active = activePresetFor(ctx, plan);
      var fixedEnv = {};
      var nonSweptBindingSources = [];   // [{paramName, sourceName}, ...]
      for (var a2 = 0; a2 < axes.length; a2++) {
        if (axes[a2].key === plan.sweepKey) continue;
        var inp = inputByKwarg[axes[a2].kwargName];
        if (!inp) continue;
        if (active.values && Object.prototype.hasOwnProperty.call(active.values, axes[a2].kwargName)) {
          fixedEnv[inp.paramName] = active.values[axes[a2].kwargName];
          continue;
        }
        fixedEnv[inp.paramName] = defaultValueForLeafType(axes[a2].leafType);
        if (axes[a2].source && axes[a2].source.kind === 'binding') {
          // tryGetMeasure (below) soft-fails to null for inputs and
          // any other source that has no samples — the leaf-type
          // default then stays in fixedEnv.
          nonSweptBindingSources.push({
            paramName: inp.paramName,
            sourceName: axes[a2].source.name,
          });
        }
      }
      var sweepInput = inputByKwarg[sweepAxis.kwargName];
      var sweepParamName = sweepInput && sweepInput.paramName;
      if (!sweepParamName) {
        showPlotMessage(ctx, 'Profile plot: cannot resolve sweep parameter.', { hint: true });
        return;
      }
      // For kernels / likelihoods we walk the kernel body via
      // traceeval — peel any outer lawof and substitute self-refs to
      // other measure bindings via expandMeasureRefsInIR (the same
      // helper bayesupdate uses). Functions evaluate the body
      // verbatim through evaluateExpr.
      var ir = sig.body;
      // Multi-output: extract the sub-IR for the currently-selected
      // output leaf. For scalar outputs (single leaf, empty path)
      // this is a no-op pass-through.
      if (plan.outputs && plan.outputs.length > 1 && plan.outputKey) {
        var selectedOut = null;
        for (var oj = 0; oj < plan.outputs.length; oj++) {
          if (plan.outputs[oj].key === plan.outputKey) {
            selectedOut = plan.outputs[oj];
            break;
          }
        }
        if (selectedOut) {
          var extracted = FlatPPLEngine.orchestrator.extractOutputIR(
            ir, selectedOut.path);
          if (extracted) ir = extracted;
        }
      }
      if (mode === 'logdensity') {
        ir = FlatPPLEngine.orchestrator.expandMeasureRefsInIR(
          ir, ctx.derivationsState.derivations);
      }
      // Propagate the swept axis (and other params) through transitive
      // deterministic deps. Without this, e.g. sweeping `theta1` in a
      // kernel whose body references `a = c * theta1` leaves `a` at a
      // single fixed value (collected once by the pre-materialise
      // step) — the plot is flat because the swept axis never reaches
      // the leaf. inlineForProfile inlines `a`'s IR into the body and
      // rewrites self.theta1 → %local.theta1.
      var paramNames = sig.inputs.map(function(inp) { return inp.paramName; });
      ir = FlatPPLEngine.orchestrator.inlineForProfile(
        ir, paramNames, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
      var POINT_COUNT = 200;
      showPlotMessage(ctx, 'Profiling…', { cancellable: true, hint: true });
      var planForCall = plan;
      // The body may reference other bindings via (ref self <name>) —
      // e.g. `f_a = functionof(c * _par_, ...)` where `c` is an outer
      // literal. Pre-materialise those, take their samples[0] as a
      // single fixed value, and merge into fixedEnv. F4b will let
      // the user pick a different atom or override these. For
      // stochastic self refs this picks the first atom, which is
      // arbitrary but deterministic — a "good enough" first cut.
      var selfRefs = [];
      FlatPPLEngine.orchestrator.collectSelfRefs(ir).forEach(function(n) {
        selfRefs.push(n);
      });
      // Range resolution per (binding, axis, domain):
      //   1. Active domain has a range for the sweep kwarg (named
      //      source binding, override, or both) → use it.
      //   2. profileRangeCache hit for the auto-fit of this
      //      (binding, axis, domainName) → reuse.
      //   3. Otherwise compute via resolveSweepRange and cache the
      //      auto-fit. The cache stores auto-fits only.
      // Note: presetOverrides are orthogonal — they drive non-swept
      // input values, not x-axis ranges.
      var domainRanges = activeDomainRangesFor(ctx, plan);
      var domainRangeForSweep = domainRanges[plan.sweepKey];
      var cacheKey = plan.name + '|' + plan.sweepKey + '|D=' + (plan.domainName || '');
      var rangePromise;
      if (domainRangeForSweep) {
        rangePromise = Promise.resolve(
          [domainRangeForSweep.lo, domainRangeForSweep.hi]);
      } else {
        var cached = ctx.profileRangeCache.get(cacheKey);
        rangePromise = cached
          ? Promise.resolve([cached.lo, cached.hi])
          : resolveSweepRange(ctx, sweepAxis).then(function(r) {
              ctx.profileRangeCache.set(cacheKey, { lo: r[0], hi: r[1], fromAuto: true });
              return r;
            });
      }
      var rangeRef = [defaultRangeForLeafType(sweepAxis.leafType)];
      Promise.all([
        rangePromise,
        Promise.all(selfRefs.map(function(n) { return tryGetMeasure(ctx, n); })),
        Promise.all(nonSweptBindingSources.map(function(s) {
          return tryGetMeasure(ctx, s.sourceName);
        })),
      ]).then(function(arr) {
        rangeRef[0] = arr[0];
        var measures = arr[1];
        for (var i = 0; i < selfRefs.length; i++) {
          var m = measures[i];
          if (m && m.samples && m.samples.length > 0) {
            fixedEnv[selfRefs[i]] = m.samples[0];
          }
        }
        var srcMeasures = arr[2];
        for (var k = 0; k < nonSweptBindingSources.length; k++) {
          var sm = srcMeasures[k];
          if (sm && sm.samples && sm.samples.length > 0) {
            fixedEnv[nonSweptBindingSources[k].paramName] = sm.samples[0];
          }
        }
        // Integer-typed sweep axis (e.g. a count parameter, a
        // Bernoulli k, a Poisson observation): only integer x values
        // are mathematically meaningful. Snap the range to integer
        // bounds and pick a sweep count so profileN's evenly-spaced
        // grid x_i = lo + (hi−lo)·i/(n−1) lands on integers exactly
        // (n = hi−lo+1 with integer lo, hi). Cap at POINT_COUNT for
        // very wide ranges; renderProfileLine then draws a step
        // plot rather than smoothing between integer values.
        var pointCount = POINT_COUNT;
        var isIntegerAxis = sweepAxis.leafType
          && sweepAxis.leafType.kind === 'scalar'
          && sweepAxis.leafType.prim === 'integer';
        if (isIntegerAxis) {
          var ilo = Math.ceil(rangeRef[0][0]);
          var ihi = Math.floor(rangeRef[0][1]);
          if (ihi >= ilo) {
            rangeRef[0] = [ilo, ihi];
            pointCount = Math.min(ihi - ilo + 1, POINT_COUNT);
          }
        }
        // For likelihood profile plots, resolve the obs IR to a JS
        // value at sample time. sig.obsIR is set by
        // signatureOfLikelihood; non-likelihood signatures don't
        // carry one. Resolution failures (e.g. an observation we
        // can't materialise) propagate as a clean plot-time error,
        // same as the bayesupdate path.
        var observed;
        if (sig.obsIR != null) {
          observed = FlatPPLEngine.orchestrator.resolveIRToValue(
            sig.obsIR, ctx.derivationsState.bindings, ctx.derivationsState.fixedValues);
        }
        return sendWorker(ctx, {
          type: 'profileN',
          ir: ir,
          sweepName: sweepParamName,
          range: rangeRef[0],
          count: pointCount,
          mode: mode,
          fixedEnv: fixedEnv,
          observed: observed,
          tally: 'clamped',
        });
      }).then(function(reply) {
        if (!reply) return;
        if (ctx.currentPlotPlan !== planForCall) return;
        renderProfileLine(reply.samples, rangeRef[0], plan, sweepAxis);
      }).catch(function(err) {
        if (ctx.currentPlotPlan !== planForCall) return;
        showPlotMessage(ctx, 'Profile plot failed: ' + esc(err && err.message || String(err)));
      });
    }

    /**
     * Build the profile-plot toolbar controls (axis dropdown, preset
     * dropdown, y-cutoff selector, x-range inputs). Returns a
     * DocumentFragment that the caller hands to renderPlotFrame as
     * `toolbarControls`. Logic mirrors the original inline build; only
     * the styling ctx.host moved.
     */
    function buildProfileControls(plan, range) {
      var frag = document.createDocumentFragment();
      var isLogDensity = plan.signature.kind === 'kernel'
                      || plan.signature.kind === 'likelihood';
      var hasAxes = plan.axes && plan.axes.length > 1;
      // "hasInputs" tells whether the plan has any input axes at all
      // (single or multiple). The input/preset dropdown is shown
      // whenever the callable has inputs, even with no user-declared
      // presets — the "auto" option in buildPresetControl still
      // surfaces the values being used.
      var hasInputs = plan.axes && plan.axes.length > 0;
      var hasMultiOutput = plan.outputs && plan.outputs.length > 1;
      // Output selector — appears for callables whose specialized
      // output is multi-leaf (record / tuple / array). Single-leaf
      // outputs (scalar functions) skip this control. Picking a
      // leaf rewrites the IR sent to profileN to the matching
      // sub-expression of the body, so the sweep evaluates that
      // specific scalar component along the chosen input axis.
      if (hasMultiOutput) {
        var outLabel = document.createElement('label');
        outLabel.textContent = 'Output:';
        outLabel.style.opacity = '0.6';
        outLabel.style.marginRight = '0.25em';
        var outSelect = document.createElement('select');
        outSelect.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
        outSelect.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
        outSelect.style.border = '1px solid var(--vscode-dropdown-border, #555)';
        outSelect.style.padding = '2px 4px';
        outSelect.style.fontSize = '1em';
        for (var oi = 0; oi < plan.outputs.length; oi++) {
          var oOpt = document.createElement('option');
          oOpt.value = plan.outputs[oi].key;
          oOpt.textContent = plan.outputs[oi].label || '<scalar>';
          if (plan.outputs[oi].key === plan.outputKey) oOpt.selected = true;
          outSelect.appendChild(oOpt);
        }
        outSelect.addEventListener('change', function(e) {
          plan.outputKey = e.target.value;
          renderProfilePlotForCurrent();
        });
        frag.appendChild(outLabel);
        frag.appendChild(outSelect);
      }
      // Output-side controls go LEFT of Inputs in the toolbar.
      // Rel. cut-off is a y-axis (output) display filter for
      // log-density / log-likelihood plots, so it belongs with the
      // Output: selector rather than between Inputs: and the
      // x-Axis block. The order ends up reading:
      //   [Output:] [Rel. cut-off:] [Inputs:] [x-Axis: lo ≤ axis ≤ hi]
      // — output-related ←left, input-related→right.
      if (isLogDensity) {
        if (plan.yCutoff == null) plan.yCutoff = 100;
        var cutLabel = document.createElement('label');
        cutLabel.textContent = 'Rel. cut-off:';
        cutLabel.style.opacity = '0.6';
        cutLabel.style.marginRight = '0.25em';
        var cutSel = document.createElement('select');
        cutSel.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
        cutSel.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
        cutSel.style.border = '1px solid var(--vscode-dropdown-border, #555)';
        cutSel.style.padding = '2px 4px';
        cutSel.style.fontSize = '1em';
        var cutoffs = [10, 100, 1000, 10000];
        for (var ci = 0; ci < cutoffs.length; ci++) {
          var copt = document.createElement('option');
          copt.value = cutoffs[ci];
          copt.textContent = '−' + cutoffs[ci];
          if (cutoffs[ci] === plan.yCutoff) copt.selected = true;
          cutSel.appendChild(copt);
        }
        cutSel.addEventListener('change', function(e) {
          plan.yCutoff = parseInt(e.target.value, 10);
          renderProfilePlotForCurrent();
        });
        frag.appendChild(cutLabel);
        frag.appendChild(cutSel);
      }
      if (hasInputs) {
        // Reuse buildPresetControl so the option text (name + value
        // record) and styling stay consistent with the kernel-sample
        // path. Always shown when there are input axes — the "auto"
        // option carries the default values even without user-
        // declared presets.
        frag.appendChild(buildPresetControl(plan, function() {
          renderProfilePlotForCurrent();
        }));
        // Domain selector — same row, drives x-axis ranges from
        // cartprod(...) bindings. Falls back to a no-op fragment when
        // the binding has no axes; we already returned early for that.
        frag.appendChild(buildDomainControl(plan, function() {
          renderProfilePlotForCurrent();
        }));
      }
      // The lo/hi limit inputs live under the plot now (see
      // buildProfileBottomRow). The toolbar carries only the axis
      // selector (or static label for single-axis), as
      //   x-Axis: <axis selector | static name>
      var xBlock = document.createElement('span');
      xBlock.style.display = 'inline-flex';
      xBlock.style.alignItems = 'center';
      xBlock.style.gap = '0.35em';

      var xLabel = document.createElement('label');
      xLabel.textContent = 'x-Axis:';
      xLabel.style.opacity = '0.6';

      // Hide kwargs the active preset wrapped in `fixed(...)` —
      // spec §03 marks those as "held constant during optimization",
      // so offering them as a sweep axis contradicts the annotation.
      // Edge case: if filtering removes every option (every kwarg was
      // marked fixed), fall back to the unfiltered list so the user
      // can still pick something; the absence of any sweepable axis
      // is a model-authoring decision, not a viewer constraint.
      var fixedNames = activeFixedNamesFor(ctx, plan);
      var visibleAxes = plan.axes;
      if (fixedNames && fixedNames.size > 0) {
        var kept = plan.axes.filter(function(a) {
          return !fixedNames.has(a.key) || a.key === plan.sweepKey;
        });
        if (kept.length > 0) visibleAxes = kept;
      }

      var axisEl;
      if (hasAxes) {
        axisEl = document.createElement('select');
        axisEl.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
        axisEl.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
        axisEl.style.border = '1px solid var(--vscode-dropdown-border, #555)';
        axisEl.style.padding = '2px 4px';
        axisEl.style.fontSize = '1em';
        axisEl.title = 'Axis to sweep';
        for (var ai = 0; ai < visibleAxes.length; ai++) {
          var opt = document.createElement('option');
          opt.value = visibleAxes[ai].key;
          opt.textContent = visibleAxes[ai].label;
          if (visibleAxes[ai].key === plan.sweepKey) opt.selected = true;
          axisEl.appendChild(opt);
        }
        axisEl.addEventListener('change', function(e) {
          plan.sweepKey = e.target.value;
          renderProfilePlotForCurrent();
        });
      } else if (plan.axes && plan.axes.length === 1) {
        axisEl = document.createElement('span');
        axisEl.textContent = plan.axes[0].label;
        axisEl.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
        axisEl.style.padding = '2px 4px';
        axisEl.style.opacity = '0.85';
      }

      xBlock.appendChild(xLabel);
      if (axisEl) xBlock.appendChild(axisEl);
      frag.appendChild(xBlock);
      return frag;
    }

    /**
     * Row that sits under the echarts plot in profile mode:
     *
     *   [lo input]    <axis name>    [hi input]
     *
     * Wrapper-edge-aligned (lo pinned left, hi pinned right; axis
     * name centred). The axis name comes from plan.axes[…].label;
     * the (default = V) decoration is added in a later step.
     * Range edits commit to ctx.profileRangeCache keyed by
     * (binding, sweepKey, preset) — same store as before, same
     * effect on the re-render path.
     */
    function buildProfileBottomRow(plan, range) {
      var fg = getComputedStyle(document.body).color || '#ccc';
      var row = document.createElement('div');
      row.style.display = 'flex';
      row.style.alignItems = 'baseline';
      row.style.gap = '0.6em';
      // Horizontal padding matches the echarts grid insets used by
      // renderProfileLine (grid.left:60, grid.right:25). With this
      // padding the lo input's left edge aligns with the chart's
      // leftmost data extent and the hi input's right edge with the
      // rightmost — so the values read as labels of the extents
      // rather than floating off in the pane margins.
      row.style.padding = '0.2em 25px 0.4em 60px';
      row.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
      row.style.fontSize = '0.92em';

      var xLoInput = document.createElement('input');
      xLoInput.type = 'number'; xLoInput.step = 'any';
      xLoInput.value = formatScalar(range[0]);
      xLoInput.title = 'x-axis lower limit';
      var xHiInput = document.createElement('input');
      xHiInput.type = 'number'; xHiInput.step = 'any';
      xHiInput.value = formatScalar(range[1]);
      xHiInput.title = 'x-axis upper limit';
      [xLoInput, xHiInput].forEach(function(inp) {
        inp.style.background = 'var(--vscode-input-background, #3c3c3c)';
        inp.style.color = 'var(--vscode-input-foreground, #cccccc)';
        inp.style.border = '1px solid var(--vscode-input-border, #555)';
        inp.style.padding = '2px 4px';
        inp.style.fontSize = '1em';
        inp.style.width = '6.5em';
        inp.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
      });

      // Limit edits commit into the active *domain* override (auto
      // pseudo-domain on plan.domainAutoOverride, named domains in
      // module-wide domainOverrides). Per-kwarg ranges live here
      // because a cartprod(...) source binding spans every input
      // axis; rebuildDerivations reconciles each kwarg independently
      // against the source intervals on the next refresh.
      var commitRange = function() {
        var newLo = parseFloat(xLoInput.value);
        var newHi = parseFloat(xHiInput.value);
        if (!Number.isFinite(newLo) || !Number.isFinite(newHi) || newLo >= newHi) {
          xLoInput.value = formatScalar(range[0]);
          xHiInput.value = formatScalar(range[1]);
          return;
        }
        var key = plan.sweepKey;
        if (!key) return;
        var entry = ensureDomainOverrideFor(ctx, plan);
        entry.ranges = entry.ranges || {};
        entry.ranges[key] = { lo: newLo, hi: newHi };
        setDomainOverrideFor(ctx, plan, entry);
        renderProfilePlotForCurrent();
      };
      xLoInput.addEventListener('change', commitRange);
      xHiInput.addEventListener('change', commitRange);

      // Centred axis name, with a "(default = V)" suffix showing
      // the value this axis is pinned at when another axis is the
      // sweep. V comes from the active preset's user-set values
      // when present (named preset value, or click-set override
      // on a modified entry); otherwise from the same auto
      // computation the dropdown's "auto: …" label uses (type
      // default, or samples[0] from a source binding when cached).
      // Lets the user read the current slice and "navigate" through
      // axes by clicking + switching the sweep direction.
      var axisName = plan.sweepKey;
      var sweepKwarg = plan.sweepKey;
      if (plan.axes) {
        for (var i = 0; i < plan.axes.length; i++) {
          if (plan.axes[i].key === plan.sweepKey) {
            axisName = plan.axes[i].label;
            sweepKwarg = plan.axes[i].kwargName;
            break;
          }
        }
      }
      var defaultText = '';
      if (sweepKwarg) {
        var activeForLabel = activePresetFor(ctx, plan);
        var v;
        if (activeForLabel.values
            && Object.prototype.hasOwnProperty.call(activeForLabel.values, sweepKwarg)) {
          v = activeForLabel.values[sweepKwarg];
        } else {
          var av = computeAutoValues(plan);
          v = av[sweepKwarg];
        }
        if (v !== undefined && v !== null) {
          defaultText = '  (default = ' + formatScalar(v) + ')';
        }
      }
      var nameSpan = document.createElement('span');
      nameSpan.textContent = axisName + defaultText;
      nameSpan.style.flex = '1';
      nameSpan.style.textAlign = 'center';
      nameSpan.style.color = fg;
      nameSpan.style.opacity = '0.75';
      nameSpan.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';

      row.appendChild(xLoInput);
      row.appendChild(nameSpan);
      row.appendChild(xHiInput);
      return row;
    }

    function renderProfileLine(values, range, plan, sweepAxis) {
      var fg = getComputedStyle(document.body).color || '#ccc';
      var color = colorForBinding(ctx, ctx.currentPlotBindingName);
      var n = values.length;
      var lo = range[0], hi = range[1];
      // Integer-typed sweep axis: only integer x values are
      // mathematically meaningful, so render as a step plot
      // (piecewise-constant between adjacent integers) plus a dot at
      // each evaluated point. Echarts' step:'middle' on a line series
      // gives the right shape — the value at integer k extends to
      // (k − 0.5, k + 0.5).
      var isIntegerAxis = sweepAxis.leafType
        && sweepAxis.leafType.kind === 'scalar'
        && sweepAxis.leafType.prim === 'integer';
      // For log-density / log-likelihood, find the maximum finite
      // value across the sweep and clamp the y-axis to
      // [yMax − cutoff, yMax]. Below that we show no-man's-land —
      // values get clamped to the cutoff line so the curve stays
      // visible rather than disappearing under a -∞ singularity.
      var yMax = -Infinity, yMin = Infinity;
      for (var yi = 0; yi < n; yi++) {
        var v = values[yi];
        if (Number.isFinite(v)) {
          if (v > yMax) yMax = v;
          if (v < yMin) yMin = v;
        }
      }
      var yClipMin = null, yClipMax = null;
      if ((plan.signature.kind === 'kernel' || plan.signature.kind === 'likelihood')
          && Number.isFinite(yMax)) {
        var cut = (plan.yCutoff != null) ? plan.yCutoff : 100;
        yClipMin = yMax - cut;
        // Add a small upper headroom (~5% of the cutoff) so the peak
        // doesn't sit on the chart's top edge.
        yClipMax = yMax + 0.05 * cut;
      }
      // Build (x, y) pairs. Three cases produce gaps (null) so the
      // line renders as broken segments rather than misleading
      // connections:
      //   1. NaN / ±∞                        — domain-of-definition holes.
      //   2. y < yClipMin (below the cutoff) — same conceptual gap:
      //      "no useful information here". Drawing them at the floor
      //      would suggest the curve sits at the cutoff value there,
      //      which it doesn't. echarts honours { connectNulls: false }
      //      and stops the line at each null.
      var data = new Array(n);
      for (var i = 0; i < n; i++) {
        var t = n === 1 ? 0 : i / (n - 1);
        var x = lo + t * (hi - lo);
        var y = values[i];
        if (!Number.isFinite(y) || (yClipMin != null && y < yClipMin)) {
          data[i] = [x, null];
        } else {
          data[i] = [x, y];
        }
      }
      var titleText = (ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'profile')
        + ' — ' + esc(sweepAxis.label);
      // Per spec / convention: a kernel with obs fixed (likelihoodof)
      // computes the log-LIKELIHOOD; a bare kernel (or any other
      // measure-bodied callable) computes the log-DENSITY at obs.
      // Functions evaluate to a value.
      var legendLabel = plan.signature.kind === 'function'   ? 'value'
                       : plan.signature.kind === 'likelihood' ? 'log-likelihood'
                       :                                        'log-density';
      // Profile plots evaluate a function/kernel at a grid of points;
      // they aren't sampled empirical measures, so no measure / N+ESS
      // readout. The toolbar carries the controls only.
      renderPlotFrame({
        toolbarControls: buildProfileControls(plan, range),
        bottomRow:       buildProfileBottomRow(plan, range),
        chartCallback: function(chartHost) {
          ctx.plotEchart = echarts.init(chartHost);
          var zoomOpts = plotZoomOptions(fg);
          ctx.plotEchart.setOption({
            animation: false,
            dataZoom: zoomOpts.dataZoom,
            toolbox: zoomOpts.toolbox,
            // The axis-name is rendered by the bottom-row instead of
            // echarts (so the lo/hi limit inputs can sit beside it,
            // wrapper-edge aligned). bottom can shrink because
            // there's no axis-name eating vertical space inside the
            // grid.
            grid: { left: 60, right: 25, top: 30, bottom: 25, containLabel: false },
            title: {
              text: titleText,
              left: 'center', top: 4,
              textStyle: { color: fg, fontSize: 13, fontWeight: 'normal' },
            },
            legend: {
              data: [legendLabel],
              top: 4, right: 12,
              textStyle: { color: fg, fontSize: 11 },
              itemWidth: 14, itemHeight: 8,
            },
            // Vertical-only crosshair on hover. Communicates "click to
            // slice on this x-axis value" — y is irrelevant, so no
            // horizontal indicator. Keep the tooltip text minimal:
            // just the rounded x value, so users can read off the
            // value they're about to commit before clicking.
            tooltip: {
              show: true,
              trigger: 'axis',
              triggerOn: 'mousemove',
              axisPointer: { type: 'line', snap: false, animation: false,
                             label: { show: false } },
              backgroundColor: 'rgba(40, 40, 40, 0.92)',
              borderColor: 'rgba(120, 120, 120, 0.6)',
              borderWidth: 1,
              padding: [3, 6],
              textStyle: { color: fg, fontSize: 11 },
              formatter: function(params) {
                if (!params || !params.length) return '';
                var x = params[0].axisValue;
                return sweepAxis.label + ' = ' + formatScalar(x);
              },
            },
            xAxis: {
              type: 'value',
              name: '',
              min: lo, max: hi,
              axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
              splitLine: { show: false },
            },
            yAxis: Object.assign({
              type: 'value',
              axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
              splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
              // No y-axis pointer — only the x-axis crosshair is
              // meaningful for click-to-slice.
              axisPointer: { show: false },
            }, yClipMin != null ? { min: yClipMin, max: yClipMax } : {}),
            series: [{
              name: legendLabel,
              type: 'line', data: data,
              // Integer-axis profile: piecewise-constant 'middle' step
              // (value at integer k extends to k±0.5) with dots at
              // each evaluated point so the discrete grid is visible.
              // Continuous axis: smooth line, no markers.
              step: isIntegerAxis ? 'middle' : false,
              symbol: isIntegerAxis ? 'circle' : 'none',
              symbolSize: 5,
              lineStyle: { color: color, width: 2 },
              itemStyle: isIntegerAxis ? { color: color } : undefined,
              connectNulls: false,
            }],
          });

          // Click-to-slice. The click lands at some x in the chart's
          // grid; we convert pixel → data, then commit that x as the
          // sweep-axis value in a modified preset (creating one if
          // we were on the base). The plot itself doesn't change
          // (the sweep axis still spans lo..hi) but the user has
          // pinned this axis's value for when they switch the
          // sweep direction to another axis. The under-plot axis
          // name picks up `(default = V)` immediately.
          ctx.plotEchart.getZr().on('click', function(ev) {
            var pt = [ev.offsetX, ev.offsetY];
            if (!ctx.plotEchart.containPixel('grid', pt)) return;
            // xAxisIndex finder takes a scalar pixel and returns a
            // scalar data value (echarts API quirk — only the grid/
            // series finders take arrays). Passing the [x,y] array
            // here returns NaN.
            var clickedX = ctx.plotEchart.convertFromPixel({ xAxisIndex: 0 }, ev.offsetX);
            if (!Number.isFinite(clickedX)) return;
            commitSliceX(plan, clickedX);
            renderProfilePlotForCurrent();
          });
        },
      });
    }

    /** Commit a clicked x value as the sweep-axis value of the
        active preset. Writes through the unified override store
        (autoOverride for auto, ctx.presetOverrides for named). */
    function commitSliceX(plan, x) {
      if (!plan || !plan.axes) return;
      var kwarg = null;
      for (var i = 0; i < plan.axes.length; i++) {
        if (plan.axes[i].key === plan.sweepKey) {
          kwarg = plan.axes[i].kwargName;
          break;
        }
      }
      if (!kwarg) return;
      var entry = ensureOverrideFor(ctx, plan);
      entry.values[kwarg] = x;
      setOverrideFor(ctx, plan, entry);
    }

    function renderArrayStepPlot(arr) {
      var fg = getComputedStyle(document.body).color || '#ccc';
      var n = arr.length;
      // Build piecewise-constant step data: each value v at index i
      // contributes two points (i, v) and (i+1, v). echarts then
      // draws segments connecting consecutive entries.
      var stepData = new Array(n * 2);
      for (var i = 0; i < n; i++) {
        stepData[2 * i]     = [i, arr[i]];
        stepData[2 * i + 1] = [i + 1, arr[i]];
      }
      // Same DAG-aligned color resolution the histogram path uses —
      // colorForBinding maps the binding's type/kind to the same
      // palette the DAG view paints. For literal arrays that's the
      // shared phaseFixed grey (post the literal-color unification);
      // for other shapes it picks up the node.kind overrides.
      var color = colorForBinding(ctx, ctx.currentPlotBindingName);
      var distLabel = ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'array';
      var arrayLegendLabel = n + ' values';
      // No measure passed — fixed array data isn't a sampled empirical
      // measure, so the frame skips the N+ESS readout. (A future
      // refinement could surface "length: n" in the toolbar instead.)
      renderPlotFrame({
        chartCallback: function(chartHost) {
          ctx.plotEchart = echarts.init(chartHost);
          var zoomOpts = plotZoomOptions(fg);
          ctx.plotEchart.setOption({
            animation: false,
            dataZoom: zoomOpts.dataZoom,
            toolbox: zoomOpts.toolbox,
            grid: { left: 60, right: 25, top: 30, bottom: 50, containLabel: false },
            title: {
              text: distLabel,
              left: 'center', top: 4,
              textStyle: { color: fg, fontSize: 13, fontWeight: 'normal' },
            },
            legend: {
              data: [arrayLegendLabel],
              top: 4, right: 12,
              textStyle: { color: fg, fontSize: 11 },
              itemWidth: 14, itemHeight: 8,
            },
            tooltip: { show: false },
            xAxis: {
              type: 'value',
              name: 'index', nameLocation: 'center', nameGap: 28,
              min: 0, max: n,
              minInterval: 1,
              axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
              splitLine: { show: false },
            },
            yAxis: {
              type: 'value',
              axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
              splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
            },
            series: [{
              name: arrayLegendLabel,
              type: 'line', data: stepData, symbol: 'none',
              lineStyle: { color: color, width: 2 },
            }],
          });
        },
      });
    }

    /**
     * Single dispatch entry point for all empirical-measure plots.
     * A measure is just a nullary kernel — so kernel-sample bindings
     * (with substituted inputs) and ordinary measure bindings render
     * through exactly the same path; the only difference is whether
     * the toolbar carries a preset selector. Higher up, this is
     * called by:
     *
     *   - renderPlotForCurrent's measure path → opts.toolbarControls
     *     is null; opts.analyticalIR drives the optional density
     *     overlay.
     *   - renderKernelSampleMeasure → opts.toolbarControls is the
     *     preset dropdown; opts.analyticalIR is null (kernel bodies
     *     aren't closed-form).
     *
     * Dispatch:
     *   - record / tuple / array shape  → corner / marginals view
     *     (constant short-circuit → renderConstantRecord)
     *   - mode === 'array' (fixed array) → step plot
     *   - constant scalar samples         → renderTextValue
     *   - otherwise scalar              → bar histogram (+ optional
     *                                       analytical density curve)
     *
     * opts:
     *   name             — binding name (display + cache key seed)
     *   mode             — plan.mode (only 'array' is read here)
     *   discrete         — drives integerHistogram vs FD
     *   analyticalIR     — optional IR to send to the density worker
     *   toolbarControls  — optional builder THUNK () → Element|Fragment
     *                      that produces fresh toolbar DOM each call.
     *                      Must be a thunk (not a static Element /
     *                      Fragment) because the corner-plot rerender
     *                      path rebuilds the toolbar repeatedly, and
     *                      appendChild on a DocumentFragment empties
     *                      it after the first append. Static Elements
     *                      get destroyed by renderPlotFrame's
     *                      innerHTML='' on the next frame rebuild.
     *                      Pass null when no extra controls are needed.
     *   staleGuard       — optional () → bool; when supplied and false,
     *                      the async density round-trip's reply is
     *                      ignored. Lets the caller cancel via plan
     *                      identity comparison without leaking a
     *                      stale plot across binding navigation.
     */
    function renderEmpiricalMeasure(measure, opts) {
      var name = opts.name;
      // Multivariate measure (record / tuple / array shapes): route
      // to the corner / 2D-strip renderer.
      if (measure.shape === 'record' || measure.shape === 'tuple' || measure.shape === 'array') {
        // Constant short-circuit: a record / tuple measure whose
        // every scalar leaf is the same across atoms is a literal
        // value masquerading as a measure. N copies of the same
        // point histogram-mash into N tall bars per axis —
        // uninformative — so render the surface form as text instead.
        // Top-level array measures stay on the corner-plot path:
        // even when "constant" they're more useful as per-slot
        // histograms.
        if ((measure.shape === 'record' || measure.shape === 'tuple')
            && measureIsConstant(ctx, measure)) {
          renderConstantRecord(measure, name);
          return;
        }
        renderRecordMarginals(measure, name, opts.toolbarControls);
        return;
      }
      var samples = measure.samples;
      // Complex-valued binding (engine sets dtype:'complex' + .imag,
      // planar with .samples = Re). v1 renders the real part — honest
      // about it via a toolbar badge — rather than silently showing
      // Re with no indication it's one projection of a complex value.
      // |z| / Im / Argand modes + a mode toggle are a tracked
      // follow-up (TODO-flatppl-js.md §03); the badge is the seam
      // they'll grow from. Array/record-shaped complex (per-atom
      // vectors) keep their existing corner rendering of Re for now.
      var isComplex = measure.dtype === 'complex'
        && measure.imag instanceof Float64Array;
      // Array-mode: skip histogram + density entirely; the data
      // is a fixed-length sequence to plot as index→value, not
      // a sample of a distribution.
      if (opts.mode === 'array') {
        renderArrayStepPlot(samples);
        return;
      }
      // Constant scalar samples: render as text (same path as
      // phase=fixed scalars and degenerate distributions). A constant
      // complex value shows both parts ("a + b i") — showing only Re
      // would be actively misleading for a fixed complex constant.
      if (samplesAreConstant(samples)) {
        if (isComplex) {
          renderTextValue(name, formatComplexScalar(samples[0], measure.imag[0]));
        } else {
          renderTextValue(name, formatScalar(samples[0]));
        }
        return;
      }
      // Histogram lives on the main thread now — no round-trip.
      // Cache by (name, discrete) so click-flipping a binding is
      // instant. Cache lives only as long as the underlying measure:
      // rebuildDerivations and configUpdate (sampleCount change)
      // clear it.
      var histKey = name + '|' + (opts.discrete ? 'd' : 'c');
      var hist = ctx.histogramCache.get(histKey);
      if (!hist) {
        // Pass logWeights through so weighted measures (post
        // weighted/bayesupdate/normalize) render their bars
        // correctly. For unweighted measures this is null and the
        // histogram takes its fast count/N path.
        var histOpts = measure.logWeights ? { logWeights: measure.logWeights } : {};
        hist = opts.discrete
          ? FlatPPLEngine.histogram.integerHistogram(samples, histOpts)
          : FlatPPLEngine.histogram.freedmanDiaconisHistogram(samples, histOpts);
        ctx.histogramCache.set(histKey, hist);
      }
      var staleGuard = opts.staleGuard || function() { return true; };
      // Scalar histogram path renders once (no internal rerenders), so
      // we resolve the toolbar thunk to a static Element here. The
      // record-marginals path above keeps the thunk so each rebuild
      // produces fresh DOM.
      var resolvedToolbar = typeof opts.toolbarControls === 'function'
        ? opts.toolbarControls()
        : opts.toolbarControls;
      // Complex scalar bindings carry no toolbar of their own — surface
      // the "showing Re(z)" badge so the histogram isn't mistaken for
      // the whole value.
      if (isComplex && resolvedToolbar == null) {
        resolvedToolbar = complexReBadge();
      }
      // Only fetch analytical density when applicable. This is the
      // only worker round-trip per plot for measure bindings, and
      // it's skipped entirely for variates and chain-mode (stochastic-
      // parent) measures.
      if (opts.analyticalIR) {
        // Anchor the density curve's x-range to the histogram's
        // first/last bin edges. Otherwise the curve uses its own
        // quantile-derived grid which can extend past the bars
        // (and into impossible regions, e.g. x<0 for Exponential).
        var range;
        if (hist.binEdges && hist.binEdges.length > 1) {
          range = [hist.binEdges[0], hist.binEdges[hist.binEdges.length - 1]];
        } else if (hist.support) {
          range = [hist.support[0], hist.support[1]];
        }
        var densOpts = { gridPoints: 256 };
        if (range) densOpts.range = range;
        return sendWorker(ctx, { type: 'density', ir: opts.analyticalIR, opts: densOpts })
          .then(function(densReply) {
            if (!staleGuard()) return;
            renderSamplesAndDensity(
              { samples: samples, histogram: hist, density: densReply, measure: measure },
              { mode: opts.mode, toolbarControls: resolvedToolbar });
          });
      }
      renderSamplesAndDensity(
        { samples: samples, histogram: hist, density: null, measure: measure },
        { mode: opts.mode, toolbarControls: resolvedToolbar });
    }

    function renderSamplesAndDensity(reply, plan) {
      // Array-data short-circuit: render an index→value step plot.
      // Skips the constant check below — a five-element array of all
      // 1s is a legitimate data sequence, not a scalar to be displayed
      // as text. (Reachable only when callers bypass
      // renderEmpiricalMeasure, which already handles the array
      // short-circuit.)
      if (plan && plan.mode === 'array') {
        renderArrayStepPlot(reply.samples);
        return;
      }

      // Constant-value short-circuit: render the value as text.
      // (Same defensive duplicate of the renderEmpiricalMeasure
      // short-circuit — keeps direct callers safe.)
      if (samplesAreConstant(reply.samples)) {
        renderTextValue(ctx.currentPlotBindingName, formatScalar(reply.samples[0]));
        return;
      }

      var fg = getComputedStyle(document.body).color || '#ccc';

      // Look up the binding's DAG-view color so the plot reads as
      // belonging to the same node the user is hovering on the graph.
      // Match the DAG renderer's color choice exactly — including its
      // node.kind override that maps a measure-typed binding to the
      // lawof blue rather than the generic 'call' grey. See
      // colorForBinding above.
      var color = colorForBinding(ctx, ctx.currentPlotBindingName);

      var hist = reply.histogram;
      var dens = reply.density;
      var discrete = (hist && hist.reference === 'counting');

      // Empirical histogram bars. For continuous, echarts' bar series
      // doesn't naturally do equal-width bars on a value xAxis; we use
      // a custom-render with the precomputed bin edges so the bars sit
      // at their actual x positions and widths regardless of zoom.
      // For discrete, the simpler bar series with categoryGap=40% gives
      // the spaced "lollipop"-ish look standard for pmfs.
      var samplesSeries;
      if (discrete) {
        var pairs = new Array(hist.xs.length);
        for (var i = 0; i < hist.xs.length; i++) pairs[i] = [hist.xs[i], hist.ys[i]];
        samplesSeries = {
          name: 'samples',
          type: 'bar',
          data: pairs,
          itemStyle: { color: color, opacity: 0.5 },
          barCategoryGap: '40%',
          z: 1,
        };
      } else {
        // Continuous: render bars via custom shape so widths track the
        // actual bin edges (not echarts' auto category spacing).
        var rects = [];
        for (var i = 0; i < hist.xs.length; i++) {
          rects.push({
            value: [hist.xs[i], hist.ys[i]],
            x0: hist.binEdges[i],
            x1: hist.binEdges[i + 1],
          });
        }
        samplesSeries = {
          name: 'samples',
          type: 'custom',
          data: rects,
          renderItem: function(_params, api) {
            var pt = api.value(0); // unused — we use the explicit edges below
            var rec = api.value(2); // also unused; we close over rects instead
            // Use the data point reference rather than pt/rec so this
            // closure stays compatible with echarts' value indexing across
            // versions. api.coord maps [x, y] data → pixel coordinates.
            var idx = _params.dataIndex;
            var d = rects[idx];
            var lt = api.coord([d.x0, d.value[1]]);  // left-top corner
            var rb = api.coord([d.x1, 0]);           // right-bottom corner
            return {
              type: 'rect',
              shape: { x: lt[0], y: lt[1], width: rb[0] - lt[0], height: rb[1] - lt[1] },
              style: api.style({ fill: color, opacity: 0.5, stroke: color, lineWidth: 0.5 }),
            };
          },
          encode: { x: 0, y: 1 },
          z: 1,
        };
      }

      // Density curve overlay (when available). Discrete + analytical
      // renders as scatter dots at integer atoms; continuous as a line.
      var densitySeries = null;
      if (dens && dens.xs && dens.xs.length > 0) {
        var dPairs = new Array(dens.xs.length);
        for (var j = 0; j < dens.xs.length; j++) dPairs[j] = [dens.xs[j], dens.ys[j]];
        if (discrete) {
          densitySeries = {
            name: 'density',
            type: 'scatter',
            data: dPairs,
            symbol: 'circle', symbolSize: 8,
            itemStyle: { color: color, borderColor: fg, borderWidth: 1, opacity: 1 },
            z: 2,
          };
        } else {
          densitySeries = {
            name: 'density',
            type: 'line',
            data: dPairs,
            symbol: 'none', smooth: false,
            lineStyle: { color: color, width: 2, opacity: 1 },
            z: 2,
          };
        }
      }

      // Density overlay is only ever the analytical PDF/pmf — there's
      // no smoothed-from-samples curve. (KDE was tried but dropped: it
      // smears mass past the support, which mis-suggests density where
      // there is none. See worker.js.)
      samplesSeries.name = 'samples';
      var series = densitySeries ? [samplesSeries, densitySeries] : [samplesSeries];
      var legendData = densitySeries ? ['samples', 'density'] : ['samples'];

      var distLabel = ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'distribution';

      // Frame owns the N + ESS readout (in the toolbar above the
      // chart). Pass reply.measure so the frame can compute it; the
      // chart itself only carries the binding-name title.
      // toolbarControls (e.g. kernel-sample preset selector) are
      // appended to the LEFT of the toolbar; N+ESS sits on the right.
      renderPlotFrame({
        measure: reply.measure,
        toolbarControls: plan && plan.toolbarControls ? plan.toolbarControls : null,
        chartCallback: function(chartHost) {
          ctx.plotEchart = echarts.init(chartHost);
          var zoomOpts2 = plotZoomOptions(fg);
          ctx.plotEchart.setOption({
            animation: false,
            dataZoom: zoomOpts2.dataZoom,
            toolbox: zoomOpts2.toolbox,
            grid: { left: 60, right: 25, top: 30, bottom: 50, containLabel: false },
            title: {
              text: distLabel,
              left: 'center', top: 4,
              textStyle: { color: fg, fontSize: 13, fontWeight: 'normal' },
            },
            legend: {
              data: legendData,
              top: 4, right: 12,
              textStyle: { color: fg, fontSize: 11 },
              itemWidth: 14, itemHeight: 8,
            },
            // No tooltip / axisPointer — the user doesn't need to read off
            // exact values from a hover crosshair, and the moving lines
            // are visually noisy. Re-enable here if a future plot view
            // (e.g. trace diagnostics) actually needs precise readouts.
            tooltip: { show: false },
            xAxis: {
              type: 'value',
              name: 'x', nameLocation: 'center', nameGap: 28,
              axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
              splitLine: { show: false },
              minInterval: discrete ? 1 : null,
              // Anchor the visible range to the histogram support — the
              // density curve, computed on a wider quantile-padded grid,
              // can extend a bit beyond; let echarts auto-fit so both fit.
            },
            yAxis: {
              type: 'value',
              name: discrete ? 'P(X=x)' : 'p(x)',
              nameLocation: 'center', nameGap: 45,
              axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
              axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
              axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
              splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
              min: 0,
            },
            series: series,
          });
        },
      });
    }

    // Set by persistAutoAsNewBinding / persistAutoDomainAsNewBinding
    // — the freshly-coined preset / domain name the rebuilt plan
    // should land on as its initial selection. Consumed once by the
    // next updatePlotForBinding call (then cleared). Two separate
    // slots so a domain save-as can't accidentally knock the user
    // off a selected preset and vice-versa.
    ctx.pendingPresetName = null;
    ctx.pendingDomainName = null;

    // Per-binding memory of the user's plan-level selections (sweep
    // axis, output leaf, preset / domain name, auto-overrides for
    // both inputs and domain). Keyed by binding name. Repopulated
    // after every plan build, consulted at the start of the next
    // build so navigating away and back restores the prior view.
    // No clear-on-source-change: stale entries are filtered by the
    // matchedPresets / matchedDomains / axes existence checks in
    // applyRememberedSelections, so a renamed-then-restored binding
    // re-applies its memory rather than discarding it.
    ctx.planMemoryByName = new Map();

    // Drop override values for kwargs the rebuilt plan no longer has
    // (e.g. the source was edited so an input went away). Without
    // this, navigating back later would re-apply a value to a kwarg
    // that no longer exists, leaking stale state into the override
    // map.



    // Call after every focusNode() to update the Plot tab's enabled
    // state and (if visible) re-render its content.
    function updatePlotForBinding(bindingName) {
      // Snapshot the outgoing plan first — the user may have
      // mutated it since it was first built (selected a different
      // preset, edited an override value, picked a sweep axis).
      // rememberPlanSelections re-keys on plan.name, so this also
      // captures same-binding edits in time for applyRemembered…
      // to restore them onto the rebuilt plan below.
      rememberPlanSelections(ctx, ctx.currentPlotPlan);
      var binding = ctx.currentBindings ? ctx.currentBindings.get(bindingName) : null;
      var plan = buildPlotPlan(binding, ctx.currentBindings);
      // Restore user-driven plan state across rebuilds — both same-
      // binding rebuilds (source edit) and cross-binding navigation
      // (click away and back). pendingPresetName / pendingDomainName
      // (set by auto-save-as) take precedence over remembered
      // selection so a freshly-coined name lands selected.
      applyRememberedSelections(ctx, plan);
      if (plan) {
        if (ctx.pendingPresetName != null) {
          var pn = ctx.pendingPresetName;
          ctx.pendingPresetName = null;
          if (plan.matchedPresets
              && plan.matchedPresets.some(function(p) { return p.name === pn; })) {
            plan.presetName = pn;
          }
        }
        if (ctx.pendingDomainName != null) {
          var dn = ctx.pendingDomainName;
          ctx.pendingDomainName = null;
          if (plan.matchedDomains
              && plan.matchedDomains.some(function(d) { return d.name === dn; })) {
            plan.domainName = dn;
          }
        }
      }
      ctx.currentPlotPlan = plan;
      // Save the freshly-hydrated plan too so a save-as pending name
      // or applyRemembered's filter decisions are reflected in memory
      // before the next mutation. The matching outgoing snapshot at
      // the top of this function captures user edits between calls.
      rememberPlanSelections(ctx, plan);
      // Only surface the clicked name in the plot UI when it actually
      // names a binding. Synthetic nodes (anonymous inline expressions,
      // placeholders, holes) carry IDs like 'prior:target' that aren't
      // useful to the user — fall back to a generic message.
      ctx.currentPlotBindingName = binding ? bindingName : null;
      // Plot pane stays visible whenever plotEnabled is true. When the
      // current binding isn't plottable, renderPlotForCurrent() shows
      // a "Not plottable" message in place of a chart.
      if (ctx.plotEnabled) renderPlotForCurrent();
    }

    // Plot toggle click handler. Restores from VS Code webview state on
    // first paint (see initial setPlotEnabled call below) so the user's
    // preference survives reloads.
    document.getElementById('plot-toggle').addEventListener('click', function() {
      setPlotEnabled(!ctx.plotEnabled);
    });

    // Drag handle between the DAG and plot panes. Lets the user
    // redistribute vertical space; both panes have a min-height clamp
    // so neither can be dragged into invisibility. The DAG and plot
    // ResizeObservers (set up further below) pick up the resulting
    // size change and refit cytoscape / echarts automatically — no
    // explicit resize / fit calls needed here.
    document.getElementById('plot-divider').addEventListener('mousedown', function (ev) {
      if (!ctx.plotEnabled) return;
      ev.preventDefault();
      var graph = document.getElementById('graph-panel');
      var plot  = document.getElementById('plot-panel');
      var startY = ev.clientY;
      var startGraphPx = graph.getBoundingClientRect().height;
      var startPlotPx  = plot.getBoundingClientRect().height;
      var combinedPx = startGraphPx + startPlotPx;
      var MIN_PX = 80;
      function onMove(mv) {
        var dy = mv.clientY - startY;
        var newGraph = startGraphPx + dy;
        var newPlot  = startPlotPx  - dy;
        if (newGraph < MIN_PX) { newGraph = MIN_PX; newPlot = combinedPx - MIN_PX; }
        if (newPlot  < MIN_PX) { newPlot  = MIN_PX; newGraph = combinedPx - MIN_PX; }
        // Use flex-basis in px so the two panes' relative split is
        // exactly what the user dragged to. flex-grow stays 1 on
        // both so subsequent host-pane resizes redistribute the
        // delta proportionally rather than parking it on one side.
        graph.style.flex = '1 1 ' + newGraph + 'px';
        plot.style.flex  = '1 1 ' + newPlot  + 'px';
      }
      function onUp() {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
      }
      document.body.style.cursor = 'row-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    });

    // --- DAG rendering ---

    // Tear down all bubble paths and clear leftover scratch. Two bubblesets-js
    // bugs we work around here:
    //   1) path.remove() sets scratch.bubbleSets to {} on each element. The
    //      next addPath's update sees the truthy empty object and crashes on
    //      linesEquals(undefined, lines). Fix: removeScratch fully.
    //   2) path.remove() detaches listeners but does NOT cancel callbacks
    //      already queued in the throttle. Those queued callbacks fire after
    //      tear-down and call this.update() on the dead path. Fix: stomp
    //      path.update to a no-op before removing.

    // Member-id set for one reification's bubble: its own kernel PLUS the
    // full kernel of any nested reification whose name appears in this
    // kernel. Nested-reification synthetic nodes need positive potential —
    // not just "avoid exemption" — for the outer contour to wrap around
    // them rather than pinching past.

    function drawReificationLassos(data) {
      if (!ctx.bb || !data.reifications) return;
      teardownBubbles(ctx);

      for (var k = 0; k < data.reifications.length; k++) {
        var r = data.reifications[k];
        if (r.kernel.length < 2) continue;
        if (!ctx.TYPE_STYLE[r.type]) continue;
        // Same colour the bubble's reification node would get — keeps
        // bubble fill, bubble stroke, and node fill in lockstep.
        var bubbleColor = resolveNodeColor(ctx, r);

        var memberIds = bubbleMemberIds(r, data.reifications);
        var nodes = ctx.cy.collection();
        for (var memId in memberIds) {
          nodes = nodes.union(ctx.cy.getElementById(memId));
        }
        // Hidden edges (visibility:hidden) can return undefined endpoints,
        // which silently corrupts bubblesets' potential field — exclude.
        var edges = ctx.cy.edges().filter(function(e) {
          return nodes.contains(e.source())
            && nodes.contains(e.target())
            && !e.data('hidden');
        });
        var avoid = ctx.cy.nodes().difference(nodes);

        ctx.bb.addPath(nodes, edges, avoid, {
          // virtualEdges: connect spatially-disconnected member groups via
          // routed connectors. Required for kernels spread across the
          // canvas — marching squares only traces one component per call.
          virtualEdges: true,
          style: {
            fill: hexToRgba(bubbleColor, 0.12),
            stroke: bubbleColor,
            strokeWidth: '1.5px',
            strokeOpacity: '0.7',
          },
        });
      }
    }


    function renderDAG(data) {
      if (!ctx.cy) initCy();
      updateHeader(ctx, data);

      var elements = [];

      // Reification anchor names — bindings that head a reification
      // group (i.e. spawn a bubble with internal kernel members).
      // Used to gate the "hollow fill" cytoscape style: only nodes
      // that actually anchor a visible bubble get the translucent
      // treatment, so synthesized bindings like prior2 =
      // lawof(disintegrate(...)) (no internal scope, no bubble
      // drawn) render with the default solid measure style.
      var reifAnchorNames = {};
      if (data.reifications) {
        for (var ra = 0; ra < data.reifications.length; ra++) {
          reifAnchorNames[data.reifications[ra].name] = true;
        }
      }

      for (var i = 0; i < data.nodes.length; i++) {
        var node = data.nodes[i];
        var ts = ctx.TYPE_STYLE[node.type] || ctx.TYPE_STYLE.unknown;

        // Shape: type-driven (carries the structural info — what *kind*
        // of binding this is). The engine-computed reification kind
        // overrides for "functionof acting on a measure → render as a
        // kernel" so the user sees a kernel regardless of which
        // keyword they wrote.
        var shape = ts.shape;
        if (node.kind === 'kernel')      shape = 'round-hexagon';
        else if (node.kind === 'measure') shape = 'round-rectangle';

        var color = resolveNodeColor(ctx, node);
        // Anonymous nodes (inline-expression targets) have label === ''
        // deliberately and show their expression on hover only. Others
        // fall back to their id.
        var displayLabel = node.label === '' ? '' : (node.label || node.id);
        var width = displayLabel === ''
          ? 60
          : Math.max(displayLabel.length * 9 + 24, 60);
        elements.push({
          group: 'nodes',
          data: {
            id: node.id,
            label: displayLabel,
            color: color,
            shape: shape,
            nodeType: node.type,
            phase: node.phase || '',
            expr: node.expr || '',
            line: node.line != null ? node.line : -1,
            isBoundary: node.isBoundary || false,
            isTarget: node.isTarget || false,
            unsupported: !!node.unsupported,
            unsupportedReason: node.unsupportedReason || '',
            unsupportedDetail: node.unsupportedDetail || '',
            inferredType: node.inferredType || '',
            hasError: !!(node.errors && node.errors.length > 0),
            isReifAnchor: !!reifAnchorNames[node.id],
            width: width,
          },
        });
      }

      // For edges entering a reification node from inside its bubble:
      //   - if source is one of the reification's targets (the value being
      //     reified): keep visible but render as a faint "tether"
      //   - else (boundary arg or other kernel member): fully hide; the
      //     bubble already conveys that flow. Edge is kept in cy so dagre
      //     uses it for layout.
      var reifMembers = {}; // reifName -> {memberId: true}
      var reifTargets = {}; // reifName -> {targetId: true}
      if (data.reifications) {
        for (var ri = 0; ri < data.reifications.length; ri++) {
          var rf = data.reifications[ri];
          reifMembers[rf.name] = {};
          for (var mi = 0; mi < rf.kernel.length; mi++) reifMembers[rf.name][rf.kernel[mi]] = true;
          reifTargets[rf.name] = {};
          var ts2 = rf.targets || [];
          for (var ti = 0; ti < ts2.length; ti++) reifTargets[rf.name][ts2[ti]] = true;
        }
      }

      // Map binding name -> binding type, used to label tether edges with
      // the reification keyword (lawof / functionof / kernelof / fn).
      var typeByName = {};
      for (var ni = 0; ni < data.nodes.length; ni++) {
        typeByName[data.nodes[ni].id] = data.nodes[ni].type;
      }

      for (var j = 0; j < data.edges.length; j++) {
        var edge = data.edges[j];
        var edgeType = edge.edgeType || 'data';
        var hidden = false;
        var membersForTarget = reifMembers[edge.target];
        if (membersForTarget && membersForTarget[edge.source] && edge.source !== edge.target) {
          if (reifTargets[edge.target] && reifTargets[edge.target][edge.source]) {
            edgeType = 'tether';
          } else {
            hidden = true;
          }
        }
        var tetherLabel = '';
        if (edgeType === 'tether') {
          var t = typeByName[edge.target];
          if (t === 'lawof' || t === 'functionof' || t === 'kernelof' || t === 'fn') {
            tetherLabel = t;
          }
        }
        elements.push({
          group: 'edges',
          data: {
            source: edge.source,
            target: edge.target,
            edgeType: edgeType,
            hidden: hidden,
            tetherLabel: tetherLabel,
          },
        });
      }

      // Tear down old bubble paths BEFORE detaching elements so we can
      // clear scratch on still-attached cytoscape elements.
      teardownBubbles(ctx);
      ctx.cy.elements().remove();
      ctx.cy.add(elements);

      ctx.cy.layout({
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 40,
        rankSep: 55,
        padding: 30,
        animate: false,
      }).run();

      ctx.cy.fit(undefined, 40);
      drawReificationLassos(data);

      // Show details for the target node automatically (the cursor is already
      // on it in the source). Falls back to the hint if no target is present.
      var target = data.nodes.find(function(n) { return n.isTarget; });
      if (target) {
        showNodeInfo({
          label: target.label || target.id,
          nodeType: target.type,
          phase: target.phase || '',
          expr: target.expr || '',
        });
      } else {
        document.getElementById('info').innerHTML = '<span class="hint">' + ctx.HINT + '</span>';
      }
    }

    // ---------------------------------------------------------------
    // Local model state
    //
    // The webview parses the .flatppl source itself (via the bundled
    // FlatPPLEngine) instead of receiving pre-rendered DAG data from the
    // extension host. This keeps the visualizer self-contained and lets
    // the same code run in a future standalone web preview.
    //
    // Two caches:
    //   currentSource  — last parsed source text (string)
    //   currentBindings — engine.processSource(currentSource).bindings
    // We re-parse only when source changes; clicking through nodes (zoom-
    // into) reuses currentBindings and just recomputes the sub-DAG.
    // ---------------------------------------------------------------
    ctx.currentSource = null;
    // Active surface-syntax variant id for the in-memory source —
    // drives both processSource grammar selection and persist write-
    // back syntax. Updated whenever a sourceUpdate carries a
    // variant; defaults to 'flatppl'.
    ctx.currentVariantId = 'flatppl';
    ctx.currentBindings = null;
    // The lowered module forwarded by processSource — used by
    // typeinfer.inferExprInScope for on-demand call-site
    // specialization (multi-output Output: selector, etc.).
    ctx.currentLoweredModule = null;

    /**
     * Re-render the DAG focused on targetName using the cached bindings.
     * If pushHistory is true, the current view is pushed onto the back-
     * button stack first. If targetName is null, falls back to the last
     * binding in document order (the same default the extension ctx.host used
     * before this refactor).
     */
    function focusNode(targetName, pushHistory) {
      if (!ctx.currentBindings) return;
      // No targetName supplied → prefer keeping the current focus.
      // This is the path used by source-only updates from the host
      // (the user is editing the RHS of the already-shown binding —
      // they don't want their place reset to "last binding"). Falls
      // through to the last binding when there's no prior focus or
      // the focused binding was deleted by the edit.
      if (!targetName) {
        if (ctx.currentState && ctx.currentBindings.has(ctx.currentState.targetName)) {
          targetName = ctx.currentState.targetName;
        } else {
          var allNames = [];
          ctx.currentBindings.forEach(function(_b, name) { allNames.push(name); });
          if (allNames.length === 0) return;
          targetName = allNames[allNames.length - 1];
        }
      }
      var dagData = FlatPPLEngine.computeSubDAG(ctx.currentBindings, targetName);
      if (!dagData || dagData.nodes.length === 0) return;

      // History grows only when (a) the caller asked us to push, and
      // (b) the target actually changed from what's currently shown.
      //   - cursor moves / ctrl-click / drill-down → push (target moved)
      //   - source-only updates (RHS edits) → no-op (target preserved)
      //   - same-target refocus → no-op
      // Capped at HISTORY_CAP entries to bound memory: each entry holds
      // a sub-DAG's nodes + edges (~few KB), so a few hundred entries
      // is plenty for navigation but well below any pressure point. On
      // overflow we drop the oldest entry (FIFO trim) — going way back
      // is rare enough that this is the right trade-off.
      if (pushHistory && ctx.currentState && ctx.currentState.targetName !== targetName) {
        ctx.history.push(ctx.currentState);
        if (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
      }

      ctx.currentState = { data: dagData, targetName: targetName };
      renderDAG(dagData);
      updateBackBtn(ctx);
      updatePlotForBinding(targetName);
      // Notify the host so any URL / panel state stays in sync with
      // the viewer's actual focus. Internal navigations (DAG node
      // clicks, double-clicks, "show whole module" toolbar) used to
      // diverge from the host's recorded target, which then leaked
      // back into the viewer when the host pushed a fresh
      // sourceUpdate carrying its (stale) target — e.g. typing in
      // an editor triggered a debounced update that yanked focus
      // back to a previous binding. With this call, host and viewer
      // share one target.
      if (ctx.host && typeof ctx.host.setTarget === 'function') {
        try { ctx.host.setTarget(targetName); } catch (_) {}
      }
    }

    /**
     * Render the module-level (multi-root) DAG. Plot pane shows a
     * "click a binding to plot it" message because there's no single
     * focused binding here. Pushes onto ctx.history when requested and
     * the previous view wasn't already the module view.
     */
    function enterModuleView(pushHistory) {
      if (!ctx.currentBindings) return;
      var dagData = FlatPPLEngine.computeFullDAG(ctx.currentBindings);
      if (!dagData || dagData.nodes.length === 0) return;

      if (pushHistory && ctx.currentState && ctx.currentState.targetName !== ctx.MODULE_TARGET) {
        ctx.history.push(ctx.currentState);
        if (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
      }

      ctx.currentState = { data: dagData, targetName: ctx.MODULE_TARGET };
      renderDAG(dagData);
      updateBackBtn(ctx);
      // Mirror module-view focus to the host (null = whole module).
      if (ctx.host && typeof ctx.host.setTarget === 'function') {
        try { ctx.host.setTarget(null); } catch (_) {}
      }
      // No specific binding to plot in module view. Pass null so the
      // Plot panel renders its placeholder; renderPlotForCurrent
      // recognizes module mode and tailors the message.
      updatePlotForBinding(null);
    }

    // Back button: pop the previous view; bindings are unchanged, only
    // re-render with the saved sub-DAG data. (We push state objects that
    // hold both the data and the target name, so we don't have to recompute
    // when going back.)
    document.getElementById('back-btn').addEventListener('click', function() {
      if (ctx.history.length === 0) return;
      ctx.currentState = ctx.history.pop();
      renderDAG(ctx.currentState.data);
      updateBackBtn(ctx);
      // Module view has no per-binding plot target and no per-binding
      // title — call updatePlotForBinding(null) so the plot pane shows
      // its module-mode placeholder, and tell the host to set a
      // generic title rather than the sentinel string.
      if (ctx.currentState.targetName === ctx.MODULE_TARGET) {
        updatePlotForBinding(null);
        if (ctx.host.setTitle) ctx.host.setTitle('module');
      } else {
        updatePlotForBinding(ctx.currentState.targetName);
        if (ctx.host.setTitle) ctx.host.setTitle(ctx.currentState.targetName);
      }
    });

    // Source-update handler shared by:
    //   - the postMessage listener below (VS Code extension host pushes
    //     fresh source on cursor moves and edits)
    //   - the public view.update(source, target?) method (programmatic
    //     re-render from any host)
    //   - the initial-source bootstrap (opts.source / opts.target on
    //     mount)
    function applySourceUpdate(msg) {
      var sourceChanged = (msg.source !== ctx.currentSource);
      // Track the surface-syntax variant of the in-memory source so
      // (a) processSource picks the right grammar and (b) persist
      // write-back chooses matching syntax (e.g. `True` vs `true`).
      // Variant comes from the host as an id string ('flatppl' /
      // 'flatppy' / 'flatppj') in msg.variant; if absent, falls back
      // to canonical FlatPPL.
      if (msg.variant) ctx.currentVariantId = msg.variant;
      if (sourceChanged) {
        ctx.currentSource = msg.source;
        try {
          var result = FlatPPLEngine.processSource(msg.source,
            { variant: ctx.currentVariantId });
          ctx.currentBindings = result.bindings;
          ctx.currentLoweredModule = result.loweredModule;
          // Source change → rebuild derivations and clear sample cache.
          // The orchestrator's derivations key the cache, so any change
          // (renamed bindings, edited dist params, new dependencies)
          // requires a full reset.
          rebuildDerivations();
        } catch (e) {
          // Parse error: keep the previous bindings so the visualizer
          // stays usable while the user fixes their syntax. The host's
          // own diagnostics (VS Code editor squiggles, embed page
          // markers, …) surface the error to the user.
          console.error('FlatPPL parse error:', e);
          return;
        }
      }
      if (msg.type === 'showModule') {
        enterModuleView(msg.pushHistory);
        return;
      }
      // The DAG view tracks two distinct foci:
      //   currentState.targetName   — sub-DAG root (set by initial
      //                                nav and DAG dbltap; mirrored
      //                                to URL via host.setTarget).
      //   currentPlotBindingName    — node whose plot is rendered in
      //                                the right pane (set additionally
      //                                by single-tap on a DAG node).
      // On a source-only refresh (persist's debounced re-render) we
      // want to KEEP whatever the user was looking at. focusNode
      // already preserves currentState.targetName when msg.targetName
      // is null, and re-renders the same sub-DAG. But its
      // updatePlotForBinding call resets the plot pane back to the
      // sub-DAG root, losing any divergent single-tap selection.
      // Capture currentPlotBindingName here and restore it after
      // focusNode finishes.
      var preservedPlotBinding = null;
      if (sourceChanged && ctx.currentPlotBindingName
          && ctx.currentState
          && ctx.currentPlotBindingName !== ctx.currentState.targetName) {
        preservedPlotBinding = ctx.currentPlotBindingName;
      }
      focusNode(msg.targetName, msg.pushHistory);
      if (preservedPlotBinding
          && ctx.currentBindings && ctx.currentBindings.has(preservedPlotBinding)
          && ctx.currentPlotBindingName !== preservedPlotBinding) {
        updatePlotForBinding(preservedPlotBinding);
      }
    }

    window.addEventListener('message', function(event) {
      var msg = event.data;
      if (!msg) return;

      if (msg.type === 'configUpdate') {
        // The host pushed updated visualization settings.
        var cfg = msg.config || {};

        // sampleCount: drop every cached EmpiricalMeasure on change
        // (each was sized to the old SAMPLE_COUNT and can't be reused)
        // and re-render the current plot at the new count. The
        // histogram cache must go too — it's keyed by binding name
        // but the underlying samples will be different.
        if (typeof cfg.sampleCount === 'number'
            && cfg.sampleCount > 0
            && cfg.sampleCount !== ctx.SAMPLE_COUNT) {
          ctx.SAMPLE_COUNT = cfg.sampleCount | 0;
          ctx.measureCache = new Map();
          ctx.histogramCache = new Map();
          if (ctx.plotEnabled) renderPlotForCurrent();
        }

        // dagNavigationHistoryCap: re-bind the limit and trim oldest
        // entries that exceed the new cap. Doesn't affect currentState
        // or the back button beyond the trim.
        if (typeof cfg.dagNavigationHistoryCap === 'number'
            && cfg.dagNavigationHistoryCap >= 0) {
          ctx.HISTORY_CAP = cfg.dagNavigationHistoryCap | 0;
          while (ctx.history.length > ctx.HISTORY_CAP) ctx.history.shift();
          updateBackBtn(ctx);
        }

        // truncateRejectionBudget: re-bind. Drop the cache because any
        // cached truncate(...)-derived measure was sized at the prior
        // budget — its n_eff and NaN slots would no longer reflect the
        // current setting. Re-render to recompute against the new value.
        if (typeof cfg.truncateRejectionBudget === 'number'
            && cfg.truncateRejectionBudget >= 1
            && cfg.truncateRejectionBudget !== ctx.REJECTION_BUDGET) {
          ctx.REJECTION_BUDGET = cfg.truncateRejectionBudget | 0;
          ctx.measureCache = new Map();
          ctx.histogramCache = new Map();
          if (ctx.plotEnabled) renderPlotForCurrent();
        }
        return;
      }

      if (msg.type !== 'sourceUpdate' && msg.type !== 'showModule') return;
      applySourceUpdate(msg);
    });

    initCy();

    // Tell the host the message listener is attached. VS Code's
    // webview.postMessage doesn't reliably buffer pre-load — messages
    // sent before this point can be lost, which produced the
    // "empty panel on first Visualize" issue. The host buffers
    // sourceUpdate / showModule / configUpdate until it sees this
    // 'webviewReady' and then flushes in order.
    if (ctx.host && ctx.host.signalReady) {
      try { ctx.host.signalReady(); } catch (_) {}
    }

    // Resize every echart instance inside #plot-content whenever the
    // plot pane changes size. Multi-chart layouts (corner plot,
    // density strips) hold many echart instances we don't track in
    // a single global; a ResizeObserver on plot-content lets us
    // resize them uniformly without needing each renderer to wire
    // up its own listener. Falls back to window.resize where
    // ResizeObserver isn't available (older webview hosts).
    function resizeAllEchartsInPlot() {
      var root = document.getElementById('plot-content');
      if (!root) return;
      var nodes = root.querySelectorAll('div');
      for (var i = 0; i < nodes.length; i++) {
        var inst = echarts.getInstanceByDom(nodes[i]);
        if (inst) try { inst.resize(); } catch (_) {}
      }
      // The root itself may host a single chart (samples / array /
      // profile single-line modes) — resize that too.
      var rootInst = echarts.getInstanceByDom(root);
      if (rootInst) try { rootInst.resize(); } catch (_) {}
    }
    if (typeof ResizeObserver === 'function') {
      var plotResizeObserver = new ResizeObserver(resizeAllEchartsInPlot);
      var plotRoot = document.getElementById('plot-content');
      if (plotRoot) plotResizeObserver.observe(plotRoot);
    } else {
      window.addEventListener('resize', resizeAllEchartsInPlot);
    }

    // Resize the cytoscape DAG when its container changes size. Without
    // this, hosts that mount the viewer inside a flex/grid layout that
    // hasn't fully settled at mount time end up with a DAG sized to
    // whatever the container was at first paint — typically too small,
    // with the layout fit-zoomed and panned for the wrong dimensions, so
    // the visible nodes appear off-center against the post-settle pane.
    // The VS Code webview avoids this because the panel resizes through
    // window.resize, which cytoscape already handles internally; the
    // standalone web host (CSS Grid + flex) needs the explicit observer.
    function resizeAndFitCy() {
      if (!ctx.cy) return;
      // requestAnimationFrame so the layout pass that triggered the
      // resize has settled before we ask cytoscape for the new size.
      requestAnimationFrame(function () {
        try { ctx.cy.resize(); ctx.cy.fit(undefined, 40); }
        catch (_) {}
      });
    }
    if (typeof ResizeObserver === 'function') {
      var cyResizeObserver = new ResizeObserver(resizeAndFitCy);
      var cyRoot = document.getElementById('cy');
      if (cyRoot) cyResizeObserver.observe(cyRoot);
    } else {
      window.addEventListener('resize', resizeAndFitCy);
    }

    // Restore Plot toggle state from the host's persistent state so the
    // user's preference survives panel close/reopen and reloads. Default
    // is OFF for first-time use — the plot panel is opt-in to keep the
    // initial DAG-only experience clean.
    var prevState = null;
    if (ctx.host.loadState) { try { prevState = ctx.host.loadState(); } catch (_) {} }
    setPlotEnabled(prevState && prevState.plotEnabled === true);

    // Initial source bootstrap. When opts.source is supplied, render
    // immediately. Otherwise the viewer waits for a postMessage
    // sourceUpdate (the existing VS Code flow) — the message listener
    // above feeds applySourceUpdate when the host sends one.
    if (typeof opts.source === 'string') {
      applySourceUpdate({
        source: opts.source,
        targetName: opts.target,
        type: opts.target ? 'sourceUpdate' : 'showModule',
        pushHistory: false,
        variant: opts.variant,
      });
    }

    // Public control surface. update(source, target, opts?) re-parses
    // and re-renders. opts.pushHistory: when true (default false),
    // treat the update as a user-initiated navigation and grow the
    // viewer's internal back-button stack (matching how DAG dbltap
    // pushes). Hosts that route through the browser's URL history
    // (e.g. the gallery's hash-based router) set this on user-driven
    // navigations so the in-viewer back button stays usable for
    // target-only steps within one model.
    // dispose() is a placeholder for now.
    return {
      update: function(source, target, opts) {
        applySourceUpdate({
          source: source,
          targetName: target,
          type: target ? 'sourceUpdate' : 'showModule',
          pushHistory: !!(opts && opts.pushHistory),
          variant: opts && opts.variant,
        });
      },
      dispose: function() {},
    };
    };

    // Auto-mount when the host provides a marker container in the DOM
    // (id="flatppl-viewer-root"). Hosts that want explicit control over
    // mount timing or args (e.g. standalone embed pages that wait for
    // user input) can omit the marker and call FlatPPLViewer.mount(...)
    // themselves. The vscode-extension's _getHtml() includes the marker,
    // so existing webview behaviour is preserved.
    function autoMountIfMarkerPresent() {
      var marker = (typeof document !== 'undefined')
        ? document.getElementById('flatppl-viewer-root')
        : null;
      if (marker) FlatPPLViewer.mount(marker);
    }
    if (typeof document !== 'undefined') {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', autoMountIfMarkerPresent);
      } else {
        autoMountIfMarkerPresent();
      }
    }
  })(typeof window !== 'undefined' ? window : globalThis);
