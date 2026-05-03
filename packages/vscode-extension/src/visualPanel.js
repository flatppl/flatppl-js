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

    function updateHeader(data) {
      var el = document.getElementById('header-expr');
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

      // Ctrl/Cmd+click: jump to source
      cy.on('tap', 'node', function(evt) {
        var oe = evt.originalEvent;
        if (oe && (oe.ctrlKey || oe.metaKey)) {
          var line = evt.target.data('line');
          if (line >= 0) {
            vscodeApi.postMessage({ type: 'navigateTo', line: line });
          }
          return;
        }
        showNodeInfo(evt.target.data());
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
    // Current plot plan from buildPlotPlan(). Two shapes:
    //   { mode: 'analytical', ir }
    //   { mode: 'chain', chain, discrete }
    // Used both as the "is plot tab enabled?" flag and as the render
    // input. currentPlotBindingName tracks which binding produced it
    // (for the chart title and stale-reply guards).
    var currentPlotPlan = null;
    var currentPlotBindingName = null;
    // Sample budget for chain-based plots. Higher → smoother KDE / more
    // accurate histograms, but quadratic in KDE cost (O(n*gridPoints)).
    // Tuned for sub-100ms response on a single Normal chain.
    var CHAIN_SAMPLE_COUNT = 5000;

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
     * Decide *how* to plot a binding and return a plan the renderer can
     * dispatch on. Three outcomes:
     *
     *   { mode: 'analytical', ir }  — exact PDF/pmf from stdlib via the
     *     worker's 'density' message. Used when the binding is a 'draw'
     *     of a registered distribution with all-literal kwargs (so the
     *     worker doesn't need any upstream env).
     *   { mode: 'chain', chain, discrete }  — sample-based density via
     *     'densityFromChain'. Used when the binding (or one of its
     *     dependencies) is stochastic; the orchestrator topologically
     *     orders the steps.
     *   null  — not plottable; Plot tab gets disabled.
     *
     * Analytical wins when both are applicable: it's exact and free of
     * KDE bandwidth artifacts. Chains are the fallback for everything
     * the orchestrator marks as supported but the analytical path
     * doesn't.
     */
    function buildPlotPlan(binding, bindingsMap) {
      if (!binding || !binding.node || !binding.node.value) return null;

      // Path 1: try analytical. Lower the (inner) distribution call and
      // require all-literal kwargs.
      var v = binding.node.value;
      if (v.callee && v.callee.type === 'Identifier') {
        var inner = v;
        if (v.callee.name === 'draw') {
          if (v.args && v.args.length === 1) inner = v.args[0];
          else inner = null;
        }
        if (inner && inner.type === 'CallExpr' && inner.callee
            && inner.callee.type === 'Identifier') {
          var ir = null;
          try { ir = FlatPPLEngine.lower.lowerExpr(inner); }
          catch (_) { ir = null; }
          if (ir && ir.kind === 'call' && ir.op
              && (!ir.args || ir.args.length === 0)) {
            var allLit = true;
            var kw = ir.kwargs || {};
            for (var k in kw) {
              if (kw[k].kind !== 'lit') { allLit = false; break; }
            }
            if (allLit) return { mode: 'analytical', ir: ir };
          }
        }
      }

      // Path 2: chain-based. Ask the orchestrator if it can sample the
      // target. Returns unsupported for reified scopes, modules,
      // unsupported distributions, etc. — in which case we end up with
      // the Plot tab disabled.
      try {
        var plan = FlatPPLEngine.orchestrator.buildSampleChain(binding.name, bindingsMap);
        if (plan && plan.chain && plan.chain.length > 0) {
          return { mode: 'chain', chain: plan.chain, discrete: !!plan.discrete };
        }
      } catch (_) { /* fall through */ }

      return null;
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

    function showPlotMessage(html) {
      if (plotEchart) { plotEchart.dispose(); plotEchart = null; }
      var el = document.getElementById('plot-content');
      el.innerHTML = '<div id="plot-empty">' + html + '</div>';
    }

    function renderPlotForCurrent() {
      // The plot panel stays mounted whenever plotEnabled is true. When
      // the focused binding isn't plottable (lawof, modules, etc.) we
      // still show *something* — a "Not plottable" message — so the
      // panel doesn't appear/disappear under the user as they click
      // around the DAG.
      if (!currentPlotPlan) {
        var name = currentPlotBindingName ? esc(currentPlotBindingName) : 'this binding';
        showPlotMessage('Not plottable for <strong>' + name + '</strong>.');
        return;
      }
      showPlotMessage('Computing density…');
      // Snapshot the plan reference so a focus change during the round-
      // trip can be detected and the stale reply discarded.
      var planForCall = currentPlotPlan;
      var request;
      if (planForCall.mode === 'analytical') {
        request = sendWorker({ type: 'density', ir: planForCall.ir, opts: { gridPoints: 256 } });
      } else if (planForCall.mode === 'chain') {
        request = sendWorker({
          type: 'densityFromChain',
          chain: planForCall.chain,
          count: CHAIN_SAMPLE_COUNT,
          discrete: planForCall.discrete,
          opts: { gridPoints: 256 },
        });
      } else {
        showPlotMessage('No distribution to plot.');
        return;
      }
      request
        .then(function(reply) {
          if (currentPlotPlan !== planForCall) return;
          renderDensity(reply, planForCall);
        })
        .catch(function(err) {
          if (currentPlotPlan !== planForCall) return;
          showPlotMessage('Could not compute density: ' + esc(err.message || String(err)));
        });
    }

    function renderDensity(d, plan) {
      var el = document.getElementById('plot-content');
      el.innerHTML = '';
      var fg = getComputedStyle(document.body).color || '#ccc';
      // Convert the typed-array transferred from the worker to plain
      // arrays for echarts. Float64Array works as a series source in
      // recent echarts but plain pairs are simpler and let us combine
      // x/y trivially.
      var xs = d.xs, ys = d.ys;
      var pairs = new Array(xs.length);
      for (var i = 0; i < xs.length; i++) pairs[i] = [xs[i], ys[i]];

      // Discrete (counting reference) distributions plot as bars at the
      // integer atoms; continuous (Lebesgue) plot as a filled area curve.
      var discrete = d.reference === 'counting';
      var distLabel = currentPlotBindingName ? esc(currentPlotBindingName) : 'distribution';
      // Density-source tag: analytical paths show "(pdf)" / "(pmf)";
      // sampled paths show the estimator method ("kde" / "histogram")
      // plus the sample count, so the user can tell at a glance whether
      // they're looking at an exact curve or a Monte-Carlo estimate.
      var methodLabel;
      if (plan && plan.mode === 'chain') {
        var m = d.method || (discrete ? 'histogram' : 'kde');
        methodLabel = m + ' from ' + CHAIN_SAMPLE_COUNT + ' samples';
      } else {
        methodLabel = discrete ? 'pmf' : 'pdf';
      }
      var seriesColor = TYPE_STYLE.draw.color;

      plotEchart = echarts.init(el);
      var series;
      if (discrete) {
        series = [{
          type: 'bar',
          data: pairs,
          itemStyle: { color: seriesColor },
          barCategoryGap: '40%',
        }];
      } else {
        series = [{
          type: 'line',
          data: pairs,
          symbol: 'none',
          smooth: false,
          lineStyle: { color: seriesColor, width: 2 },
          areaStyle: { color: seriesColor, opacity: 0.18 },
        }];
      }
      plotEchart.setOption({
        animation: false,
        grid: { left: 60, right: 25, top: 30, bottom: 50, containLabel: false },
        title: {
          text: distLabel + ' — ' + methodLabel,
          left: 'center', top: 4,
          textStyle: { color: fg, fontSize: 12, fontWeight: 'normal', opacity: 0.8 },
        },
        tooltip: { trigger: 'axis' },
        xAxis: {
          type: 'value',
          name: 'x', nameLocation: 'center', nameGap: 28,
          axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6 },
          splitLine: { show: false },
          minInterval: discrete ? 1 : null,
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
      currentPlotBindingName = bindingName;
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

      if (data.nodes.length === 1 && showDataView(data)) {
        // Data view has no clickable graph nodes — show the target's details
        // directly in the info bar.
        var t = data.nodes[0];
        showNodeInfo({
          label: t.label || t.id,
          nodeType: t.type,
          phase: t.phase || '',
          expr: t.expr || '',
        });
        return;
      }
      hideDataView();

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
      // Pick a default target when none is supplied: the last user-defined
      // binding. Mirrors the extension host's previous fall-back logic.
      if (!targetName) {
        var allNames = [];
        currentBindings.forEach(function(_b, name) { allNames.push(name); });
        if (allNames.length === 0) return;
        targetName = allNames[allNames.length - 1];
      }
      var dagData = FlatPPLEngine.computeSubDAG(currentBindings, targetName);
      if (!dagData || dagData.nodes.length === 0) return;

      if (pushHistory && currentState) {
        history.push(currentState);
      } else if (!pushHistory) {
        history = [];
      }

      currentState = { data: dagData, targetName: targetName };
      renderDAG(dagData);
      updateBackBtn();
      updatePlotForBinding(targetName);
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
      updatePlotForBinding(currentState.targetName);
      vscodeApi.postMessage({ type: 'updateTitle', name: currentState.targetName });
    });

    window.addEventListener('message', function(event) {
      var msg = event.data;
      if (!msg || msg.type !== 'sourceUpdate') return;

      // Only re-parse when source actually changed. Cursor-driven retargets
      // re-use the cached bindings (saves the parse for typical "click
      // around" interactions where the user navigates without editing).
      if (msg.source !== currentSource) {
        currentSource = msg.source;
        try {
          var result = FlatPPLEngine.processSource(msg.source);
          currentBindings = result.bindings;
        } catch (e) {
          // Parse error: keep the previous bindings so the visualizer
          // stays usable while the user fixes their syntax. Errors flow
          // through VS Code diagnostics in the editor anyway.
          console.error('FlatPPL parse error:', e);
          return;
        }
      }
      focusNode(msg.targetName, msg.pushHistory);
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
