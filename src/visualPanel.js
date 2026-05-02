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
    this._onZoomInto = null;
    this._panel.webview.html = this._getHtml();
    this._panel.onDidDispose(() => {
      FlatPPLPanel.currentPanel = undefined;
    });
    this._panel.webview.onDidReceiveMessage(msg => {
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
      if (msg.type === 'zoomInto' && this._onZoomInto) {
        this._onZoomInto(msg.nodeId);
      }
      if (msg.type === 'updateTitle') {
        this._panel.title = `FlatPPL: ${msg.name}`;
      }
    });
  }

  set onZoomInto(callback) {
    this._onZoomInto = callback;
  }

  update(dagData, targetName, sourceUri, pushHistory) {
    if (sourceUri) this._sourceUri = sourceUri;
    this._panel.title = `FlatPPL: ${targetName}`;
    this._panel.webview.postMessage({
      type: 'updateDAG',
      data: dagData,
      targetName,
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

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src 'nonce-${nonce}'; style-src 'unsafe-inline';">
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
    #cy { width: 100vw; height: calc(100vh - 86px); }
    #dataview {
      display: none; width: 100vw; height: calc(100vh - 86px);
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
  <div id="header"><button id="back-btn">&larr; Back</button><span id="header-expr"></span></div>
  <div id="cy"></div>
  <div id="dataview"></div>
  <div id="tooltip"></div>
  <div id="legend"></div>
  <div id="info">
    <span class="hint">Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source</span>
  </div>

  <script nonce="${nonce}" src="${cytoscapeUri}"></script>
  <script nonce="${nonce}" src="${dagreUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeDagreUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeLayersUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeBubblesetsUri}"></script>
  <script nonce="${nonce}" src="${echartsUri}"></script>
  <script nonce="${nonce}">
  (function() {
    var vscodeApi = acquireVsCodeApi();
    var HINT = 'Click a node to see details &middot; double-click to drill down &middot; Ctrl+click to jump to source';

    // Color choices form an additive triple (blue + pink ≈ purple), so the
    // family relationships read visually: lawof (measure) and functionof
    // (function) sit at the two "primary" hues, kernelof (a function of a
    // measure) sits at their additive mix. Material 300 palette — pastel
    // enough to read as bubble fills at low alpha against a dark editor
    // background, and the blue+pink+purple triple is robust under
    // protanopia/deuteranopia (red-green colorblind).
    var TYPE_STYLE = {
      input:         { color: '#4DD0E1', shape: 'diamond',          label: 'input (elementof)' },
      draw:          { color: '#B39DDB', shape: 'ellipse',          label: 'draw' },
      call:          { color: '#90A4AE', shape: 'round-rectangle',  label: 'call' },
      // lawof always produces a measure; rendered as a round-rectangle.
      lawof:         { color: '#64B5F6', shape: 'round-rectangle',  label: 'lawof (measure)' },
      // kernelof always produces a Markov kernel — round-hexagon. Color
      // is also applied to functionof-of-measure below (same kind).
      kernelof:      { color: '#BA68C8', shape: 'round-hexagon',    label: 'kernelof (kernel)' },
      // functionof produces a function by default (hexagon). When its
      // first arg is a measure the engine reports kind='kernel' and the
      // node picks up kernelof's shape and color.
      functionof:    { color: '#F06292', shape: 'hexagon',          label: 'functionof' },
      fn:            { color: '#F06292', shape: 'hexagon',          label: 'fn' },
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
            selector: 'edge[edgeType = "tether"]',
            style: {
              'line-color': function(ele) { return ele.target().data('color') || '#aaa'; },
              'opacity': 0.6,
              'width': 1.5,
              'target-arrow-shape': 'none',
              'curve-style': 'straight',
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
        wheelSensitivity: 0.3,
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

      // Double-click: drill into node's sub-DAG
      cy.on('dbltap', 'node', function(evt) {
        var nodeId = evt.target.data('id');
        // Don't drill into synthetic nodes (placeholder/hole inputs)
        if (nodeId.indexOf(':') !== -1) return;
        vscodeApi.postMessage({ type: 'zoomInto', nodeId: nodeId });
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
        elements.push({
          group: 'edges',
          data: {
            source: edge.source,
            target: edge.target,
            edgeType: edgeType,
            hidden: hidden,
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

    // Back button
    document.getElementById('back-btn').addEventListener('click', function() {
      if (history.length === 0) return;
      currentState = history.pop();
      renderDAG(currentState.data);
      updateBackBtn();
      vscodeApi.postMessage({ type: 'updateTitle', name: currentState.targetName });
    });

    window.addEventListener('message', function(event) {
      var msg = event.data;
      if (!msg || msg.type !== 'updateDAG') return;

      if (msg.pushHistory && currentState) {
        history.push(currentState);
      } else if (!msg.pushHistory) {
        history = [];
      }

      currentState = { data: msg.data, targetName: msg.targetName };
      renderDAG(msg.data);
      updateBackBtn();
    });

    initCy();
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
