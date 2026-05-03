'use strict';
const vscode = require('vscode');

class FlatPPLPanel {
  static currentPanel = undefined;
  static viewType = 'flatpplPanel';

  static createOrShow(context) {
    const column = vscode.ViewColumn.Beside;
    if (FlatPPLPanel.currentPanel) {
      // Don't steal focus from the editor when the panel updates.
      FlatPPLPanel.currentPanel._panel.reveal(column, /* preserveFocus */ true);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      FlatPPLPanel.viewType,
      'FlatPPL',
      { viewColumn: column, preserveFocus: true },
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(context.extensionUri, 'lib'),
        ],
      }
    );
    FlatPPLPanel.currentPanel = new FlatPPLPanel(panel, context);
  }

  constructor(panel, context) {
    this._panel = panel;
    this._context = context;
    this._sourceUri = null;
    this._panel.webview.html = this._getHtml();
    this._panel.onDidDispose(() => {
      FlatPPLPanel.currentPanel = undefined;
    });
    this._panel.webview.onDidReceiveMessage(msg => {
      // Editor-navigation request from a webview node click. The webview
      // owns its own DAG state now (parses source locally, handles zoom-
      // into events without a host round-trip), so the only remaining
      // host-bound messages are ones that need VS Code API access:
      // moving the editor cursor and updating the panel title.
      if (msg.type === 'navigateTo' && this._sourceUri != null) {
        const line = msg.line;
        const uri = this._sourceUri;
        vscode.window.showTextDocument(uri, {
          viewColumn: vscode.ViewColumn.One,
          preserveFocus: false,
        }).then(editor => {
          const pos = new vscode.Position(line, 0);
          editor.selection = new vscode.Selection(pos, pos);
          editor.revealRange(
            new vscode.Range(pos, pos),
            vscode.TextEditorRevealType.InCenterIfOutsideViewport
          );
        });
      }
      if (msg.type === 'updateTitle') {
        this._panel.title = `FlatPPL: ${msg.name}`;
      }
    });
  }

  /**
   * Push a fresh source text to the webview, optionally with a target
   * binding name to focus on. The webview parses the source via its own
   * `FlatPPLEngine` instance, computes the sub-DAG for `targetName` (or
   * picks a sensible default if null), and renders.
   *
   * Replaces the older `update(dagData, ...)` API: the extension host no
   * longer pre-computes DAGs. This keeps a single source of truth in the
   * webview and means one engine codebase serves both the VS Code panel
   * and the future standalone web preview.
   */
  updateSource(source, targetName, sourceUri, pushHistory) {
    if (sourceUri) this._sourceUri = sourceUri;
    if (targetName) this._panel.title = `FlatPPL: ${targetName}`;
    this._panel.webview.postMessage({
      type: 'sourceUpdate',
      source,
      targetName: targetName || null,
      pushHistory: !!pushHistory,
    });
  }

  /**
   * Push the current visualization config to the webview. Called once
   * on panel creation and again whenever a relevant `flatppl.*`
   * configuration value changes. The webview clears its sample cache
   * on receipt if numbers like sampleCount have changed — cached
   * arrays were sized to the old count and can't be reused.
   */
  updateConfig(config) {
    this._panel.webview.postMessage({ type: 'configUpdate', config });
  }

  /**
   * Render the module-level (multi-root) DAG. Distinct from
   * updateSource which centers a single-target sub-DAG; module mode
   * shows every binding linked by its dependencies. Title is
   * normalized to "FlatPPL: module" so the editor tab reflects the
   * mode. The webview pushes a history entry when pushHistory is
   * true so the back-button can return to a prior single-binding view.
   */
  showModule(source, sourceUri, pushHistory) {
    if (sourceUri) this._sourceUri = sourceUri;
    this._panel.title = 'FlatPPL: module';
    this._panel.webview.postMessage({
      type: 'showModule',
      source,
      pushHistory: !!pushHistory,
    });
  }

  _getHtml() {
    const webview = this._panel.webview;
    const nonce = getNonce();

    const cytoscapeUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'cytoscape.min.js')
    );
    const dagreUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'dagre.min.js')
    );
    const cytoscapeDagreUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'cytoscape-dagre.js')
    );
    const cytoscapeLayersUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'cytoscape-layers.min.js')
    );
    const cytoscapeBubblesetsUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'cytoscape-bubblesets.min.js')
    );
    const echartsUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'echarts.min.js')
    );
    // Engine bundle: same FlatPPL parser/analyzer/DAG-builder used by the
    // extension host, packaged as a browser-loadable IIFE that exposes
    // `globalThis.FlatPPLEngine`. The webview uses it to parse incoming
    // source text, build bindings, and compute sub-DAGs locally — replacing
    // the older flow where the extension host did this work and shipped
    // pre-rendered DAG data over the postMessage wire. Running the engine
    // in the webview makes the visualizer self-contained and lays the
    // foundation for the future standalone web preview (no extension host).
    const engineUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'engine.min.js')
    );
    // Sampler-worker bundle: loaded by the webview as a Web Worker (not a
    // top-level script). The worker owns Philox RNG state + stdlib's
    // distribution code, so the main webview thread never sees stdlib. The
    // CSP below adds `worker-src` for this URI.
    const samplerWorkerUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'sampler-worker.min.js')
    );

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src 'nonce-${nonce}' ${webview.cspSource} blob:; style-src 'unsafe-inline'; worker-src ${webview.cspSource} blob:; connect-src ${webview.cspSource};">
  <title>FlatPPL</title>
  <style>
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
    #plot-empty {
      opacity: 0.5; font-style: italic; padding: 20px; text-align: center;
    }
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
    }
    /* Graph internals fill graph-panel — switched from full-viewport
       sizing to 100% of the parent so the split-flex layout governs. */
    #cy { width: 100%; height: 100%; }
    #dataview {
      display: none; width: 100%; height: 100%;
      align-items: center; justify-content: center;
    }
    #dataview canvas { display: block; }
    #dataview .scalar-value {
      font-family: var(--vscode-editor-font-family, monospace);
      font-size: 36px; font-weight: 300; opacity: 0.9;
    }
    #info {
      height: 60px;
      padding: 6px 14px;
      border-top: 1px solid var(--vscode-panel-border, #444);
      font-size: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 2px;
    }
    #info .row { display: flex; gap: 8px; align-items: baseline; }
    #info .name { font-weight: 600; font-size: 13px; }
    #info .type {
      font-size: 11px; opacity: 0.6;
      padding: 1px 6px; border-radius: 3px;
      background: var(--vscode-badge-background, #444);
      color: var(--vscode-badge-foreground, #fff);
    }
    #info .phase {
      font-size: 11px;
      padding: 1px 6px; border-radius: 3px;
      color: #fff;
    }
    #info .phase-fixed         { background: #607D8B; }
    #info .phase-parameterized { background: #4DD0E1; color: #222; }
    #info .phase-stochastic    { background: #B39DDB; color: #222; }
    #info .expr {
      opacity: 0.5; white-space: nowrap;
      overflow: hidden; text-overflow: ellipsis;
    }
    #info .hint { opacity: 0.4; font-style: italic; }
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
    #legend {
      position: absolute; top: 8px; right: 8px;
      background: var(--vscode-editor-background);
      border: 1px solid var(--vscode-panel-border, #444);
      border-radius: 4px; padding: 6px 10px;
      font-size: 11px; opacity: 0.85;
      display: flex; flex-direction: column; gap: 3px;
    }
    #legend .item { display: flex; align-items: center; gap: 6px; }
    #legend .swatch {
      width: 14px; height: 14px; border-radius: 3px;
      border: 1px solid #888; flex-shrink: 0;
    }
  </style>
</head>
<body>
  <div id="header">
    <button id="back-btn">&larr; Back</button>
    <span id="header-expr"></span>
    <button id="plot-toggle" title="Toggle the plot panel">Plot: off</button>
  </div>
  <div id="main">
    <div id="graph-panel" class="full">
      <div id="cy"></div>
      <div id="dataview"></div>
      <div id="legend"></div>
    </div>
    <div id="plot-panel" class="hidden">
      <div id="plot-content"></div>
    </div>
  </div>
  <div id="tooltip"></div>
  <div id="info">
    <span class="hint">Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source</span>
  </div>

  <script nonce="${nonce}" src="${cytoscapeUri}"></script>
  <script nonce="${nonce}" src="${dagreUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeDagreUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeLayersUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeBubblesetsUri}"></script>
  <script nonce="${nonce}" src="${echartsUri}"></script>
  <script nonce="${nonce}" src="${engineUri}"></script>
  <script nonce="${nonce}">
  (function() {
    var vscodeApi = acquireVsCodeApi();
    var HINT = 'Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source';
    // Sampler-worker URL injected by the host. Used lazily — no worker is
    // spawned until the user picks a binding for which the Plot tab is
    // enabled (a 'draw' of a known distribution with literal params).
    var SAMPLER_WORKER_URL = ${JSON.stringify(samplerWorkerUri.toString())};

    // Color choices form an additive triple (blue + green ≈ teal), so the
    // family relationships read visually: lawof (measure) and functionof
    // (function) sit at the two "primary" hues, kernelof (a function of a
    // measure) sits at their additive mix. Viridis-style green/blue/teal
    // palette — perceptually ordered, colorblind-safe (the differences
    // live in distinct RGB channels), and quiet enough to read as bubble
    // fills at low alpha against a dark editor background.
    var TYPE_STYLE = {
      input:         { color: '#4DD0E1', shape: 'diamond',          label: 'input (elementof)' },
      draw:          { color: '#B39DDB', shape: 'ellipse',          label: 'draw' },
      call:          { color: '#90A4AE', shape: 'round-rectangle',  label: 'call' },
      // lawof always produces a measure; rendered as a round-rectangle.
      lawof:         { color: '#42A5F5', shape: 'round-rectangle',  label: 'lawof (measure)' },
      // kernelof always produces a Markov kernel — round-hexagon. Color
      // is also applied to functionof-of-measure below (same kind).
      kernelof:      { color: '#26A69A', shape: 'round-hexagon',    label: 'kernelof (kernel)' },
      // functionof produces a function by default (hexagon). When its
      // first arg is a measure the engine reports kind='kernel' and the
      // node picks up kernelof's shape and color.
      functionof:    { color: '#66BB6A', shape: 'hexagon',          label: 'functionof' },
      fn:            { color: '#66BB6A', shape: 'hexagon',          label: 'fn' },
      literal:       { color: '#F48FB1', shape: 'rectangle',        label: 'literal' },
      likelihood:    { color: '#EF9A9A', shape: 'octagon',          label: 'likelihood' },
      bayesupdate:   { color: '#FFAB91', shape: 'octagon',          label: 'bayesupdate' },
      module:        { color: '#80CBC4', shape: 'round-rectangle',  label: 'module' },
      table:         { color: '#A1887F', shape: 'round-rectangle',  label: 'table' },
      unknown:       { color: '#BDBDBD', shape: 'rectangle',        label: 'unknown' },
    };

    var cy = null;
    var bb = null;
    var shownTypes = new Set();
    var history = [];
    var currentState = null;
    // Bound on the DAG-navigation history (back-button stack). Cheap
    // insurance against pathological growth (a runaway extension or
    // rapid-fire navigation). Each entry is a sub-DAG's data plus a
    // name string, so a few hundred is plenty without thinking about
    // memory. Owned by the host setting flatppl.visualization.
    // dagNavigationHistoryCap (default 1000); the host pushes its
    // value via configUpdate alongside sampleCount.
    var HISTORY_CAP = 1000;

    function esc(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function showNodeInfo(d) {
      var phase = d.phase || 'unknown';
      var phaseTag = '<span class="phase phase-' + esc(phase) + '">' + esc(phase) + ' phase</span>';
      var unsupportedRow = '';
      if (d.unsupported) {
        var msg = 'disintegration unresolved: ' + esc(d.unsupportedReason || '');
        if (d.unsupportedDetail) msg += ' — ' + esc(d.unsupportedDetail);
        unsupportedRow = '<div class="expr" style="color:#FF8A65;">' + msg + '</div>';
      }
      document.getElementById('info').innerHTML =
        '<div class="row"><span class="name">' + esc(d.label)
        + '</span><span class="type">' + esc(d.nodeType) + '</span>'
        + phaseTag + '</div>'
        + '<div class="expr">' + esc(d.expr) + '</div>'
        + unsupportedRow;
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

    // Sentinel name for the module-overview state. Distinct from any
    // user binding name (binding identifiers are barewords; the
    // sentinel uses ':' which the analyzer can't produce). Used by
    // updateHeader, updatePlotForBinding, and the back-button to
    // distinguish module view from a single-binding view.
    var MODULE_TARGET = ':module';

    function updateHeader(data) {
      var el = document.getElementById('header-expr');
      // Module view: no per-node target; just label the view.
      if (currentState && currentState.targetName === MODULE_TARGET) {
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

    function updateBackBtn() {
      document.getElementById('back-btn').style.display = history.length > 0 ? 'block' : 'none';
    }

    function buildLegend() {
      var el = document.getElementById('legend');
      el.innerHTML = '';
      for (var t of shownTypes) {
        var s = TYPE_STYLE[t] || TYPE_STYLE.unknown;
        el.innerHTML += '<div class="item">'
          + '<span class="swatch" style="background:' + s.color + '"></span>'
          + '<span>' + esc(s.label) + '</span></div>';
      }
    }

    function initCy() {
      cy = cytoscape({
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
            // Reification nodes (lawof, functionof, fn): match the surrounding
            // bubble — translucent fill in the type color, solid border in
            // the same color at the bubble's stroke width — so they read
            // as belonging to the bubble rather than floating inside.
            selector: 'node[nodeType = "lawof"], node[nodeType = "functionof"], node[nodeType = "kernelof"], node[nodeType = "fn"]',
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

      if (typeof cy.bubbleSets === 'function') {
        // bubblesets uses one scratch key per cytoscape node; when paths
        // share nodes (e.g. theta1 belongs to both prior and forward_kernel),
        // their cached geometry stomps on each other and one path goes empty
        // on update. Workaround: tear down and rebuild all paths on drag
        // release, rAF-batched. Updates skipped during drag for snappiness.
        bb = cy.bubbleSets({ interactive: false });
        var bbRedrawScheduled = false;
        cy.on('free', 'node', function() {
          if (!bb || bbRedrawScheduled || !currentState) return;
          bbRedrawScheduled = true;
          requestAnimationFrame(function() {
            bbRedrawScheduled = false;
            if (currentState) drawReificationLassos(currentState.data);
          });
        });
      }

      // Ctrl/Cmd+click: jump to source.
      // Plain click: select the node — info bar updates AND the plot
      // panel re-targets to this binding. The plot follows the
      // selection rather than the DAG's terminal target so users can
      // explore the graph node-by-node and read each binding's
      // distribution in place.
      cy.on('tap', 'node', function(evt) {
        var oe = evt.originalEvent;
        if (oe && (oe.ctrlKey || oe.metaKey)) {
          var line = evt.target.data('line');
          if (line >= 0) {
            vscodeApi.postMessage({ type: 'navigateTo', line: line });
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

      cy.on('tap', function(evt) {
        if (evt.target === cy) {
          document.getElementById('info').innerHTML = '<span class="hint">' + HINT + '</span>';
        }
      });

      // Double-click: drill into node's sub-DAG. Handled locally — the
      // webview owns the parsed bindings and recomputes the sub-DAG itself
      // (no host round-trip). Title sync to the editor still goes via a
      // postMessage to the host since the title is on the VS Code panel.
      cy.on('dbltap', 'node', function(evt) {
        var nodeId = evt.target.data('id');
        // Don't drill into synthetic nodes (placeholder/hole inputs).
        if (nodeId.indexOf(':') !== -1) return;
        focusNode(nodeId, /* pushHistory */ true);
        vscodeApi.postMessage({ type: 'updateTitle', name: nodeId });
      });

      var tip = document.getElementById('tooltip');
      cy.on('mouseover', 'node', function(evt) {
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
      cy.on('mouseout', 'node', function() {
        tip.style.display = 'none';
      });
      cy.on('viewport', function() {
        tip.style.display = 'none';
      });
    }

    // --- Data visualization ---

    var echart = null;

    function parseValues(expr) {
      if (!expr) return null;
      if (/^[+\\-]?[0-9]+\\.?[0-9]*(?:[eE][+\\-]?[0-9]+)?$/.test(expr))
        return { type: 'scalar', value: parseFloat(expr) };
      var m = expr.match(/^\\[(.+)\\]$/);
      if (m) {
        var parts = m[1].split(/\\s*,\\s*/), nums = [];
        for (var i = 0; i < parts.length; i++) {
          var v = parseFloat(parts[i]);
          if (isNaN(v)) return null;
          nums.push(v);
        }
        return { type: 'array', values: nums };
      }
      return null;
    }

    function showDataView(data) {
      var target = null;
      for (var i = 0; i < data.nodes.length; i++)
        if (data.nodes[i].isTarget) { target = data.nodes[i]; break; }
      if (!target || target.type !== 'literal') return false;
      var parsed = parseValues(target.expr);
      if (!parsed) return false;

      var dv = document.getElementById('dataview');
      document.getElementById('cy').style.display = 'none';
      document.getElementById('legend').style.display = 'none';
      if (echart) { echart.dispose(); echart = null; }
      dv.innerHTML = '';

      if (parsed.type === 'scalar') {
        dv.style.display = 'flex';
        dv.innerHTML = '<span class="scalar-value">' + esc(String(parsed.value)) + '</span>';
      } else if (parsed.type === 'array' && parsed.values.length > 0) {
        dv.style.display = 'block';
        var fg = getComputedStyle(document.body).color || '#ccc';
        var vals = parsed.values;
        var n = vals.length;
        var stepData = [];
        for (var si = 0; si < n; si++) {
          stepData.push([si, vals[si]]);
          stepData.push([si + 1, vals[si]]);
        }
        echart = echarts.init(dv);
        echart.setOption({
          animation: false,
          grid: { left: 55, right: 20, top: 15, bottom: 40, containLabel: false },
          xAxis: {
            type: 'value', name: 'index', nameLocation: 'center', nameGap: 25,
            min: 0, max: n,
            axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
            axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
            axisLabel: { color: fg, opacity: 0.6 },
            splitLine: { show: false },
            minInterval: 1,
          },
          yAxis: {
            type: 'value',
            axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
            axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
            axisLabel: { color: fg, opacity: 0.6 },
            splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
          },
          series: [{
            type: 'line', data: stepData, symbol: 'none',
            lineStyle: { color: TYPE_STYLE.literal.color, width: 2 },
          }],
        });
      } else {
        dv.style.display = 'none';
        document.getElementById('cy').style.display = 'block';
        document.getElementById('legend').style.display = '';
        return false;
      }
      return true;
    }

    function hideDataView() {
      if (echart) { echart.dispose(); echart = null; }
      document.getElementById('dataview').style.display = 'none';
      document.getElementById('dataview').innerHTML = '';
      document.getElementById('cy').style.display = 'block';
      document.getElementById('legend').style.display = '';
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

    var samplerWorker = null;
    var samplerWorkerPromise = null;   // Promise<Worker> while spawn is in-flight
    var samplerWorkerError = null;     // last spawn error, surfaced in the UI
    var samplerReqId = 0;
    var pendingRequests = new Map(); // id → { resolve, reject }
    var plotEchart = null;

    // ---------------------------------------------------------------
    // Main-thread sample cache.
    //
    // The cache lives here, not on the worker, so:
    //   - Samples persist across worker recycles.
    //   - Variates and their underlying measures share Float64Arrays
    //     (theta1's "samples" ARE theta1_dist's samples — same reference).
    //   - Click-around the DAG hits the cache → instant re-render.
    //   - Source edits invalidate everything by clearing the map.
    //
    // Each binding has at most one entry. Aliases share the array
    // reference (no copy). Deterministic transforms reuse parents'
    // arrays element-wise via the worker's evaluateN primitive.
    //
    // Per-binding seeding: we derive a deterministic seed from a
    // string hash of the binding name XOR'd with a root seed. Two
    // independent random variables (theta1_dist, theta2_dist) thus
    // get statistically independent streams without coupling to the
    // order of materialisation. A future "Resample" button can bump
    // rootSeed and clear the cache to redraw everything.
    // ---------------------------------------------------------------
    var derivationsState = null;       // { derivations, discrete } from orchestrator
    var sampleCache = new Map();       // Map<name, Float64Array>
    var rootSeed = 1;
    // Sample budget for chain-based plots. Higher → smoother histograms,
    // marginal cost grows linearly. Tuned for sub-100ms response.
    // Sample budget per binding when the visualizer renders a histogram.
    // Owned by VS Code's configuration (flatppl.visualization.sampleCount,
    // default 100000, max 10_000_000); the host pushes it via a
    // configUpdate message and updates it on settings changes. Value
    // here is just an in-flight default until the first configUpdate
    // arrives — the panel always boots with a config push from the host.
    var SAMPLE_COUNT = 100000;

    function rebuildDerivations() {
      if (!currentBindings) { derivationsState = null; sampleCache = new Map(); return; }
      try {
        derivationsState = FlatPPLEngine.orchestrator.buildDerivations(currentBindings);
      } catch (e) {
        console.error('FlatPPL: buildDerivations failed:', e);
        derivationsState = null;
      }
      // Source change invalidates every cached array — derivations
      // (or just signatures) may have shifted under any of them.
      sampleCache = new Map();
    }

    /**
     * FNV-1a 32-bit string hash, then XOR the root seed. Used to give
     * each binding its own RNG stream for drawN(). Independent of
     * arrival order — two independent variables stay independent
     * regardless of which one the user clicked first.
     */
    function nameSeed(name) {
      var h = 2166136261;
      for (var i = 0; i < name.length; i++) {
        h = h ^ name.charCodeAt(i);
        h = Math.imul(h, 16777619);
      }
      return (h ^ rootSeed) >>> 0;
    }

    /**
     * Recursively materialise samples for a binding, reusing cache
     * entries for any deps already computed. Returns Promise<Float64Array>
     * because dep materialisation involves worker round-trips. Callers
     * must await.
     *
     * Aliases share the SAME array reference as their target (no copy)
     * so click-flipping between a variate and its measure is free.
     */
    function getSamples(name) {
      if (sampleCache.has(name)) return Promise.resolve(sampleCache.get(name));
      if (!derivationsState) return Promise.reject(new Error('no model loaded'));
      var d = derivationsState.derivations[name];
      if (!d) return Promise.reject(new Error("no derivation for '" + name + "'"));

      var promise;
      if (d.kind === 'alias') {
        promise = getSamples(d.from).then(function(arr) {
          // Cache the alias under its own name pointing at the same
          // Float64Array. They are now genuinely identical references.
          sampleCache.set(name, arr);
          return arr;
        });
      } else if (d.kind === 'sample') {
        promise = collectRefArrays(d.distIR).then(function(refArrays) {
          return sendWorker({
            type: 'drawN',
            ir: d.distIR,
            count: SAMPLE_COUNT,
            refArrays: refArrays,
            seed: nameSeed(name),
          });
        }).then(function(reply) {
          sampleCache.set(name, reply.samples);
          return reply.samples;
        });
      } else if (d.kind === 'evaluate') {
        promise = collectRefArrays(d.ir).then(function(refArrays) {
          return sendWorker({
            type: 'evaluateN',
            ir: d.ir,
            count: SAMPLE_COUNT,
            refArrays: refArrays,
          });
        }).then(function(reply) {
          sampleCache.set(name, reply.samples);
          return reply.samples;
        });
      } else if (d.kind === 'array') {
        // Static array literal — values verbatim, no sampling, no
        // worker round-trip. Cached length equals the array length,
        // NOT SAMPLE_COUNT; the plot path detects this via the
        // derivation kind on the plan.
        var arr = Float64Array.from(d.values);
        sampleCache.set(name, arr);
        promise = Promise.resolve(arr);
      } else {
        return Promise.reject(new Error('unknown derivation kind: ' + d.kind));
      }
      return promise;
    }

    /** Walk an IR for all (ref self <name>) and return a refName→Float64Array map. */
    function collectRefArrays(ir) {
      var refs = FlatPPLEngine.orchestrator.collectSelfRefs(ir);
      var names = [];
      refs.forEach(function(n) { names.push(n); });
      return Promise.all(names.map(function(n) { return getSamples(n); }))
        .then(function(arrays) {
          var out = {};
          for (var i = 0; i < names.length; i++) out[names[i]] = arrays[i];
          return out;
        });
    }
    // Current plot plan from buildPlotPlan(). Two shapes:
    //   { mode: 'analytical', ir }
    //   { mode: 'chain', chain, discrete }
    // Used both as the "is plot tab enabled?" flag and as the render
    // input. currentPlotBindingName tracks which binding produced it
    // (for the chart title and stale-reply guards).
    var currentPlotPlan = null;
    var currentPlotBindingName = null;

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
    function ensureSamplerWorker() {
      if (samplerWorker) return Promise.resolve(samplerWorker);
      if (samplerWorkerPromise) return samplerWorkerPromise;

      samplerWorkerPromise = (async function() {
        // Try direct construction first — cheapest path on hosts where
        // it works. Fall back to blob: on any failure (security error,
        // cross-origin block, etc.).
        var w = null;
        try {
          w = new Worker(SAMPLER_WORKER_URL);
        } catch (e) {
          // continue to blob fallback
          console.warn('FlatPPL: direct worker spawn failed, retrying via blob URL:', e && e.message);
        }
        if (!w) {
          var resp = await fetch(SAMPLER_WORKER_URL);
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
        wireWorker(w);
        samplerWorker = w;
        // Initialize with a fixed seed for deterministic output. Future:
        // plumb a "Resample" button that re-seeds (e.g. from Date.now()).
        sendWorkerNow(w, { type: 'init', seed: 1 });
        return w;
      })();

      samplerWorkerPromise.catch(function(err) {
        samplerWorkerError = err;
        samplerWorkerPromise = null;
        console.error('FlatPPL: sampler worker unavailable:', err);
      });

      return samplerWorkerPromise;
    }

    function wireWorker(w) {
      w.addEventListener('message', function(ev) {
        var reply = ev.data;
        if (!reply || reply.id == null) return;
        var p = pendingRequests.get(reply.id);
        if (!p) return;
        pendingRequests.delete(reply.id);
        if (reply.type === 'error') p.reject(new Error(reply.message || 'worker error'));
        else p.resolve(reply);
      });
      w.addEventListener('error', function(e) {
        // A top-level worker error fails every outstanding request — there's
        // no way to know which request the error pertains to, and the worker
        // may be dead. Reject all and reset so a future request can retry
        // the spawn.
        console.error('FlatPPL sampler worker error:', e.message || e);
        for (var entry of pendingRequests.values()) entry.reject(new Error(e.message || 'worker crashed'));
        pendingRequests.clear();
        try { w.terminate(); } catch (_) {}
        if (samplerWorker === w) {
          samplerWorker = null;
          samplerWorkerPromise = null;
        }
      });
    }

    // Fire-and-forget message send (used during init when we don't care
    // about the reply). Distinct from sendWorker so we don't allocate a
    // pending-request entry for messages whose reply is just an 'ok'.
    function sendWorkerNow(w, msg) {
      var id = ++samplerReqId;
      w.postMessage(Object.assign({ id: id }, msg));
    }

    function sendWorker(msg) {
      return ensureSamplerWorker().then(function(w) {
        var id = ++samplerReqId;
        var wrapped = Object.assign({ id: id }, msg);
        return new Promise(function(resolve, reject) {
          pendingRequests.set(id, { resolve: resolve, reject: reject });
          w.postMessage(wrapped);
        });
      });
    }

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
      if (!binding || !derivationsState) return null;
      var name = binding.name;
      var d = derivationsState.derivations[name];
      if (!d) return null;
      var discrete = !!derivationsState.discrete[name];

      // Render mode dispatches the plot path:
      //   'array'   — static fixed-length sequence, plotted as
      //               index/value step line (legacy data preview).
      //   'samples' — Monte-Carlo samples, plotted as histogram + an
      //               optional analytical density overlay.
      if (d.kind === 'array') {
        return { name: name, mode: 'array' };
      }

      // Variates never get a density overlay — see rule 1 above.
      // For measures, the overlay is the analytical PDF/PMF when the
      // resolved leaf has all-literal kwargs (closed-form marginal).
      var analyticalIR = null;
      if (binding.type !== 'draw') {
        var leafIR = FlatPPLEngine.orchestrator.leafSampleIR(name, derivationsState.derivations);
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
    var plotEnabled = false;

    function setPlotEnabled(enabled) {
      plotEnabled = !!enabled;
      var plot = document.getElementById('plot-panel');
      var graph = document.getElementById('graph-panel');
      var btn  = document.getElementById('plot-toggle');
      plot.classList.toggle('hidden', !plotEnabled);
      graph.classList.toggle('full',  !plotEnabled);
      btn.classList.toggle('on', plotEnabled);
      btn.textContent = 'Plot: ' + (plotEnabled ? 'on' : 'off');
      // Persist across panel reopens. VS Code restores webview state
      // automatically when the panel is shown again.
      try { vscodeApi.setState({ plotEnabled: plotEnabled }); } catch (_) {}
      if (plotEnabled) {
        // Render whatever the current plan says — including the
        // "not plottable" message if the focused binding isn't
        // chainable. Echarts also needs resize after becoming visible
        // (it measures 0×0 while collapsed).
        renderPlotForCurrent();
        if (plotEchart) plotEchart.resize();
      } else if (plotEchart) {
        // Tear down the echart instance to avoid keeping its canvas /
        // event listeners alive while the panel is collapsed. It'll
        // be reconstructed on the next renderDensity call.
        try { plotEchart.dispose(); } catch (_) {}
        plotEchart = null;
      }
      // Cytoscape skipped resize while the graph pane was at a
      // different height — kick it now so the layout fills correctly.
      if (cy) {
        // requestAnimationFrame so the flex re-layout has settled
        // before we ask cytoscape for the new size.
        requestAnimationFrame(function() { cy.resize(); cy.fit(undefined, 40); });
      }
    }

    function showPlotMessage(html, options) {
      if (plotEchart) { plotEchart.dispose(); plotEchart = null; }
      var el = document.getElementById('plot-content');
      var cancellable = options && options.cancellable;
      var stopHtml = cancellable
        ? '<div><button class="plot-stop-btn" id="plot-stop-btn">Stop</button></div>'
        : '';
      el.innerHTML = '<div id="plot-empty">' + html + stopHtml + '</div>';
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
    function cancelAllSampling() {
      if (samplerWorker) {
        try { samplerWorker.terminate(); } catch (_) {}
        samplerWorker = null;
        samplerWorkerPromise = null;
      }
      var entries = pendingRequests.values();
      pendingRequests = new Map();
      for (var entry of entries) {
        try { entry.reject(new Error('cancelled')); } catch (_) {}
      }
    }

    function renderPlotForCurrent() {
      // The plot panel stays mounted whenever plotEnabled is true. When
      // the focused binding isn't plottable (lawof, modules, etc.) we
      // still show *something* — a "Not plottable" message — so the
      // panel doesn't appear/disappear under the user as they click
      // around the DAG.
      if (!currentPlotPlan) {
        if (currentState && currentState.targetName === MODULE_TARGET) {
          showPlotMessage('Click a binding in the graph to plot it.');
        } else {
          var name = currentPlotBindingName ? esc(currentPlotBindingName) : 'this binding';
          showPlotMessage('Not plottable for <strong>' + name + '</strong>.');
        }
        return;
      }
      // Array-mode loads the cached array synchronously (no worker
      // round-trip), so a Stop button is pointless for it. Sampling
      // mode shows the Stop button so the user can abort long
      // operations (per-i ref chains under huge sample counts).
      var arrayMode = currentPlotPlan.mode === 'array';
      showPlotMessage(arrayMode ? 'Loading…' : 'Sampling…', { cancellable: !arrayMode });
      var planForCall = currentPlotPlan;

      // Cache hit avoids the worker entirely. We still defer through
      // a microtask so the UI flush is uniform and the stale-reply
      // guard pattern stays the same.
      Promise.resolve()
        .then(function() { return getSamples(planForCall.name); })
        .then(function(samples) {
          if (currentPlotPlan !== planForCall) return null;
          // Array-mode: skip histogram + density entirely; the data
          // is a fixed-length sequence to plot as index→value, not
          // a sample of a distribution.
          if (planForCall.mode === 'array') {
            return { samples: samples, mode: 'array' };
          }
          // Histogram lives on the main thread now — no round-trip.
          var hist = planForCall.discrete
            ? FlatPPLEngine.histogram.integerHistogram(samples)
            : FlatPPLEngine.histogram.freedmanDiaconisHistogram(samples);
          // Only fetch analytical density when applicable. This is
          // the only worker round-trip per plot for measure bindings,
          // and it's skipped entirely for variates and chain-mode
          // (stochastic-parent) measures.
          if (planForCall.analyticalIR) {
            // Anchor the density curve's x-range to the histogram's
            // first/last bin edges. Otherwise the curve uses its own
            // quantile-derived grid which can extend past the bars
            // (and into impossible regions, e.g. x<0 for Exponential).
            // Discrete histograms expose [lo, hi] integer atoms in
            // their support field; FD histograms expose binEdges[0]
            // through binEdges[N].
            var range;
            if (hist.binEdges && hist.binEdges.length > 1) {
              range = [hist.binEdges[0], hist.binEdges[hist.binEdges.length - 1]];
            } else if (hist.support) {
              range = [hist.support[0], hist.support[1]];
            }
            var densOpts = { gridPoints: 256 };
            if (range) densOpts.range = range;
            return sendWorker({ type: 'density', ir: planForCall.analyticalIR, opts: densOpts })
              .then(function(densReply) { return { samples: samples, histogram: hist, density: densReply }; });
          }
          return { samples: samples, histogram: hist, density: null };
        })
        .then(function(reply) {
          if (!reply || currentPlotPlan !== planForCall) return;
          renderSamplesAndDensity(reply, planForCall);
        })
        .catch(function(err) {
          if (currentPlotPlan !== planForCall) return;
          var msg = err && err.message ? err.message : String(err);
          if (msg === 'cancelled') {
            // User clicked Stop. Make the message actionable rather
            // than dead-end so they know how to retry.
            var name = currentPlotBindingName ? esc(currentPlotBindingName) : 'this binding';
            showPlotMessage('Sampling cancelled. Click <strong>' + name + '</strong> in the graph to retry.');
          } else {
            showPlotMessage('Could not compute plot: ' + esc(msg));
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
     * Both layers (bars + curve) use the focused binding's TYPE_STYLE
     * color from the DAG view, so a stochastic 'draw' node plots
     * purple, a measure-alias 'call' node plots grey-blue, etc. Bars
     * sit at low alpha; the line/dots are opaque on top.
     */
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

    /**
     * Format a fixed scalar for display. JavaScript's default String()
     * gives "5" for 5.0 and full precision for things like 0.1+0.2;
     * we strip trailing zeros via toPrecision(12) → parseFloat → String
     * so 5.0 reads "5", 3.14159 stays "3.14159", and noisy
     * float-arithmetic results like 0.30000000000000004 become "0.3".
     */
    function formatScalar(v) {
      if (!Number.isFinite(v)) return String(v);
      if (Number.isInteger(v)) return String(v);
      return String(parseFloat(v.toPrecision(12)));
    }

    /**
     * Resolve a binding's plot color to match the DAG renderer's
     * choice exactly. The DAG picks color from TYPE_STYLE[node.type]
     * but then overrides it when node.kind says "measure" (lawof
     * blue) or "kernel" (kernelof teal). Without those overrides the
     * plot for a measure-typed binding (theta1_dist, type='call')
     * would draw in grey instead of the blue used in the DAG bubble,
     * breaking the visual link between the two views.
     *
     * Fall back to TYPE_STYLE[binding.type] when the binding isn't in
     * the current DAG — paths that update the plot independent of
     * the DAG (rare, but possible during config-update reflows).
     */
    function colorForBinding(bindingName) {
      if (currentState && currentState.data && currentState.data.nodes) {
        var nodes = currentState.data.nodes;
        for (var i = 0; i < nodes.length; i++) {
          var n = nodes[i];
          if (n.id !== bindingName) continue;
          if (n.kind === 'kernel')  return TYPE_STYLE.kernelof.color;
          if (n.kind === 'measure') return TYPE_STYLE.lawof.color;
          var ts = TYPE_STYLE[n.type] || TYPE_STYLE.unknown;
          return ts.color;
        }
      }
      var binding = currentBindings && currentBindings.get(bindingName);
      var bindingType = (binding && binding.type) || 'draw';
      return (TYPE_STYLE[bindingType] || TYPE_STYLE.draw).color;
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
     * Render a fixed-length array as an index→value step plot. Used
     * for literal-array bindings (observed_data = [1.2, 3.4, …]),
     * which aren't samples of a distribution. The series is drawn as
     * piecewise-constant horizontal segments — same shape as the
     * legacy "data preview" view that used to swap into the graph
     * pane, now living in the plot pane next to the DAG.
     */
    function renderArrayStepPlot(arr) {
      var el = document.getElementById('plot-content');
      el.innerHTML = '';
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
      // Same DAG-aligned color resolution the histogram path uses.
      // For a literal-array node this normally lands on TYPE_STYLE.literal
      // (pink), but going via colorForBinding picks up any future
      // node.kind overrides automatically.
      var color = colorForBinding(currentPlotBindingName);
      var distLabel = currentPlotBindingName ? esc(currentPlotBindingName) : 'array';

      if (plotEchart) { try { plotEchart.dispose(); } catch (_) {} plotEchart = null; }
      plotEchart = echarts.init(el);
      var zoomOpts = plotZoomOptions(fg);
      var arrayLegendLabel = n + ' values';
      plotEchart.setOption({
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
        // No tooltip / axisPointer — the user doesn't need to read off
        // exact values from a hover crosshair, and the moving lines
        // are visually noisy. Re-enable here if a future plot view
        // (e.g. trace diagnostics) actually needs precise readouts.
        tooltip: { show: false },
        xAxis: {
          type: 'value',
          name: 'index', nameLocation: 'center', nameGap: 28,
          min: 0, max: n,
          minInterval: 1,
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6 },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value',
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6 },
          splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
        },
        series: [{
          name: arrayLegendLabel,
          type: 'line', data: stepData, symbol: 'none',
          lineStyle: { color: color, width: 2 },
        }],
      });
    }

    function renderSamplesAndDensity(reply, plan) {
      var el = document.getElementById('plot-content');
      el.innerHTML = '';
      var fg = getComputedStyle(document.body).color || '#ccc';

      // Array-data short-circuit: render an index→value step plot.
      // Skips the constant check below — a five-element array of all
      // 1s is a legitimate data sequence, not a scalar to be displayed
      // as text.
      if (plan && plan.mode === 'array') {
        renderArrayStepPlot(reply.samples);
        return;
      }

      // Constant-value short-circuit: dispose any stale echart and
      // render a simple "name = value" block. Doing this *before*
      // looking up colors / building series so we don't waste any
      // setup work that the bars/density branches would do.
      if (samplesAreConstant(reply.samples)) {
        if (plotEchart) { try { plotEchart.dispose(); } catch (_) {} plotEchart = null; }
        var name = currentPlotBindingName ? esc(currentPlotBindingName) : '';
        el.innerHTML =
          '<div class="scalar-display">'
          + (name ? '<div class="name">' + name + '</div>' : '')
          + '<div class="value">' + esc(formatScalar(reply.samples[0])) + '</div>'
          + '</div>';
        return;
      }

      // Look up the binding's DAG-view color so the plot reads as
      // belonging to the same node the user is hovering on the graph.
      // Match the DAG renderer's color choice exactly — including its
      // node.kind override that maps a measure-typed binding to the
      // lawof blue rather than the generic 'call' grey. See
      // colorForBinding above.
      var color = colorForBinding(currentPlotBindingName);

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
      // there is none. See worker.js.) The samples-series legend label
      // doubles as the sample count display so we don't burn vertical
      // space on a subtitle.
      var samplesLabel = SAMPLE_COUNT + ' samples';
      samplesSeries.name = samplesLabel;
      var series = densitySeries ? [samplesSeries, densitySeries] : [samplesSeries];
      var legendData = densitySeries ? [samplesLabel, 'density'] : [samplesLabel];

      var distLabel = currentPlotBindingName ? esc(currentPlotBindingName) : 'distribution';

      plotEchart = echarts.init(el);
      var zoomOpts2 = plotZoomOptions(fg);
      plotEchart.setOption({
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
          axisLabel: { color: fg, opacity: 0.6 },
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
          axisLabel: { color: fg, opacity: 0.6 },
          splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
          min: 0,
        },
        series: series,
      });
    }

    // Call after every focusNode() to update the Plot tab's enabled
    // state and (if visible) re-render its content.
    function updatePlotForBinding(bindingName) {
      var binding = currentBindings ? currentBindings.get(bindingName) : null;
      var plan = buildPlotPlan(binding, currentBindings);
      currentPlotPlan = plan;
      // Only surface the clicked name in the plot UI when it actually
      // names a binding. Synthetic nodes (anonymous inline expressions,
      // placeholders, holes) carry IDs like 'prior:target' that aren't
      // useful to the user — fall back to a generic message.
      currentPlotBindingName = binding ? bindingName : null;
      // Plot pane stays visible whenever plotEnabled is true. When the
      // current binding isn't plottable, renderPlotForCurrent() shows
      // a "Not plottable" message in place of a chart.
      if (plotEnabled) renderPlotForCurrent();
    }

    // Plot toggle click handler. Restores from VS Code webview state on
    // first paint (see initial setPlotEnabled call below) so the user's
    // preference survives reloads.
    document.getElementById('plot-toggle').addEventListener('click', function() {
      setPlotEnabled(!plotEnabled);
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
    function teardownBubbles() {
      if (!bb) return;
      bb.getPaths().forEach(function(p) {
        p.update = function() {};
        bb.removePath(p);
      });
      cy.elements().forEach(function(el) { el.removeScratch('bubbleSets'); });
    }

    // Member-id set for one reification's bubble: its own kernel PLUS the
    // full kernel of any nested reification whose name appears in this
    // kernel. Nested-reification synthetic nodes need positive potential —
    // not just "avoid exemption" — for the outer contour to wrap around
    // them rather than pinching past.
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

    function drawReificationLassos(data) {
      if (!bb || !data.reifications) return;
      teardownBubbles();

      for (var k = 0; k < data.reifications.length; k++) {
        var r = data.reifications[k];
        if (r.kernel.length < 2) continue;
        var ts = TYPE_STYLE[r.type];
        if (!ts) continue;
        // Mirror the kind-based color override applied to nodes: a
        // functionof of a measure is semantically a kernel, so its bubble
        // takes the kernelof color.
        var bubbleColor = ts.color;
        if (r.kind === 'kernel')      bubbleColor = TYPE_STYLE.kernelof.color;
        else if (r.kind === 'measure') bubbleColor = TYPE_STYLE.lawof.color;

        var memberIds = bubbleMemberIds(r, data.reifications);
        var nodes = cy.collection();
        for (var memId in memberIds) {
          nodes = nodes.union(cy.getElementById(memId));
        }
        // Hidden edges (visibility:hidden) can return undefined endpoints,
        // which silently corrupts bubblesets' potential field — exclude.
        var edges = cy.edges().filter(function(e) {
          return nodes.contains(e.source())
            && nodes.contains(e.target())
            && !e.data('hidden');
        });
        var avoid = cy.nodes().difference(nodes);

        bb.addPath(nodes, edges, avoid, {
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

    function hexToRgba(hex, alpha) {
      var m = /^#([0-9a-f]{6})$/i.exec(hex);
      if (!m) return hex;
      var n = parseInt(m[1], 16);
      var r = (n >> 16) & 0xff, g = (n >> 8) & 0xff, b = n & 0xff;
      return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
    }

    function renderDAG(data) {
      if (!cy) initCy();
      updateHeader(data);

      // Note: the legacy dataview-swap (which replaced cytoscape with a
      // floating "5" / step-line array view when the DAG had a single
      // literal node) used to fire here. We removed it because the Plot
      // panel now owns scalar / value rendering — keeping the graph
      // pane always graphical means clicking around the DAG never
      // surprises the user with a different layout in the same area.
      // showDataView / hideDataView remain defined above in case we
      // want to repurpose them for array-data preview later.
      shownTypes.clear();
      var elements = [];

      for (var i = 0; i < data.nodes.length; i++) {
        var node = data.nodes[i];
        var ts = TYPE_STYLE[node.type] || TYPE_STYLE.unknown;
        shownTypes.add(node.type);
        // Override shape and color based on the engine-computed reification
        // kind. functionof on a measure is semantically a Markov kernel —
        // it should read as kernelof (purple round-hexagon), regardless
        // of which keyword the user wrote.
        var shape = ts.shape;
        var color = ts.color;
        if (node.kind === 'kernel') {
          shape = 'round-hexagon';
          color = TYPE_STYLE.kernelof.color;
        } else if (node.kind === 'measure') {
          shape = 'round-rectangle';
          color = TYPE_STYLE.lawof.color;
        }
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
      teardownBubbles();
      cy.elements().remove();
      cy.add(elements);

      cy.layout({
        name: 'dagre',
        rankDir: 'TB',
        nodeSep: 40,
        rankSep: 55,
        padding: 30,
        animate: false,
      }).run();

      cy.fit(undefined, 40);
      drawReificationLassos(data);
      buildLegend();

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
        document.getElementById('info').innerHTML = '<span class="hint">' + HINT + '</span>';
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
    var currentSource = null;
    var currentBindings = null;

    /**
     * Re-render the DAG focused on targetName using the cached bindings.
     * If pushHistory is true, the current view is pushed onto the back-
     * button stack first. If targetName is null, falls back to the last
     * binding in document order (the same default the extension host used
     * before this refactor).
     */
    function focusNode(targetName, pushHistory) {
      if (!currentBindings) return;
      // No targetName supplied → prefer keeping the current focus.
      // This is the path used by source-only updates from the host
      // (the user is editing the RHS of the already-shown binding —
      // they don't want their place reset to "last binding"). Falls
      // through to the last binding when there's no prior focus or
      // the focused binding was deleted by the edit.
      if (!targetName) {
        if (currentState && currentBindings.has(currentState.targetName)) {
          targetName = currentState.targetName;
        } else {
          var allNames = [];
          currentBindings.forEach(function(_b, name) { allNames.push(name); });
          if (allNames.length === 0) return;
          targetName = allNames[allNames.length - 1];
        }
      }
      var dagData = FlatPPLEngine.computeSubDAG(currentBindings, targetName);
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
      if (pushHistory && currentState && currentState.targetName !== targetName) {
        history.push(currentState);
        if (history.length > HISTORY_CAP) history.shift();
      }

      currentState = { data: dagData, targetName: targetName };
      renderDAG(dagData);
      updateBackBtn();
      updatePlotForBinding(targetName);
    }

    /**
     * Render the module-level (multi-root) DAG. Plot pane shows a
     * "click a binding to plot it" message because there's no single
     * focused binding here. Pushes onto history when requested and
     * the previous view wasn't already the module view.
     */
    function enterModuleView(pushHistory) {
      if (!currentBindings) return;
      var dagData = FlatPPLEngine.computeFullDAG(currentBindings);
      if (!dagData || dagData.nodes.length === 0) return;

      if (pushHistory && currentState && currentState.targetName !== MODULE_TARGET) {
        history.push(currentState);
        if (history.length > HISTORY_CAP) history.shift();
      }

      currentState = { data: dagData, targetName: MODULE_TARGET };
      renderDAG(dagData);
      updateBackBtn();
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
      if (history.length === 0) return;
      currentState = history.pop();
      renderDAG(currentState.data);
      updateBackBtn();
      // Module view has no per-binding plot target and no per-binding
      // title — call updatePlotForBinding(null) so the plot pane shows
      // its module-mode placeholder, and tell the host to set a
      // generic title rather than the sentinel string.
      if (currentState.targetName === MODULE_TARGET) {
        updatePlotForBinding(null);
        vscodeApi.postMessage({ type: 'updateTitle', name: 'module' });
      } else {
        updatePlotForBinding(currentState.targetName);
        vscodeApi.postMessage({ type: 'updateTitle', name: currentState.targetName });
      }
    });

    window.addEventListener('message', function(event) {
      var msg = event.data;
      if (!msg) return;

      if (msg.type === 'configUpdate') {
        // The host pushed updated visualization settings.
        var cfg = msg.config || {};

        // sampleCount: drop every cached Float64Array on change (each
        // was sized to the old SAMPLE_COUNT and can't be reused) and
        // re-render the current plot at the new count.
        if (typeof cfg.sampleCount === 'number'
            && cfg.sampleCount > 0
            && cfg.sampleCount !== SAMPLE_COUNT) {
          SAMPLE_COUNT = cfg.sampleCount | 0;
          sampleCache = new Map();
          if (plotEnabled) renderPlotForCurrent();
        }

        // dagNavigationHistoryCap: re-bind the limit and trim oldest
        // entries that exceed the new cap. Doesn't affect currentState
        // or the back button beyond the trim.
        if (typeof cfg.dagNavigationHistoryCap === 'number'
            && cfg.dagNavigationHistoryCap >= 0) {
          HISTORY_CAP = cfg.dagNavigationHistoryCap | 0;
          while (history.length > HISTORY_CAP) history.shift();
          updateBackBtn();
        }
        return;
      }

      if (msg.type !== 'sourceUpdate' && msg.type !== 'showModule') return;

      // Only re-parse when source actually changed. Cursor-driven retargets
      // re-use the cached bindings (saves the parse for typical "click
      // around" interactions where the user navigates without editing).
      if (msg.source !== currentSource) {
        currentSource = msg.source;
        try {
          var result = FlatPPLEngine.processSource(msg.source);
          currentBindings = result.bindings;
          // Source change → rebuild derivations and clear sample cache.
          // The orchestrator's derivations key the cache, so any change
          // (renamed bindings, edited dist params, new dependencies)
          // requires a full reset.
          rebuildDerivations();
        } catch (e) {
          // Parse error: keep the previous bindings so the visualizer
          // stays usable while the user fixes their syntax. Errors flow
          // through VS Code diagnostics in the editor anyway.
          console.error('FlatPPL parse error:', e);
          return;
        }
      }
      if (msg.type === 'showModule') {
        enterModuleView(msg.pushHistory);
      } else {
        focusNode(msg.targetName, msg.pushHistory);
      }
    });

    initCy();

    // Restore Plot toggle state from the webview's persistent state so
    // the user's preference survives panel close/reopen and VS Code
    // window reloads. Default is OFF for first-time use — the plot
    // panel is opt-in to keep the initial DAG-only experience clean.
    var prevState = null;
    try { prevState = vscodeApi.getState(); } catch (_) {}
    setPlotEnabled(prevState && prevState.plotEnabled === true);
  })();
  </script>
</body>
</html>`;
  }
}

function getNonce() {
  let text = '';
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return text;
}

module.exports = { FlatPPLPanel };
