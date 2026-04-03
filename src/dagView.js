'use strict';
const vscode = require('vscode');

class DAGPanel {
  static currentPanel = undefined;
  static viewType = 'flatpplDAG';

  static createOrShow(context) {
    const column = vscode.ViewColumn.Beside;
    if (DAGPanel.currentPanel) {
      DAGPanel.currentPanel._panel.reveal(column);
      return;
    }
    const panel = vscode.window.createWebviewPanel(
      DAGPanel.viewType,
      'FlatPPL DAG',
      column,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.joinPath(context.extensionUri, 'lib'),
        ],
      }
    );
    DAGPanel.currentPanel = new DAGPanel(panel, context);
  }

  constructor(panel, context) {
    this._panel = panel;
    this._context = context;
    this._sourceUri = null;
    this._panel.webview.html = this._getHtml();
    this._panel.onDidDispose(() => {
      DAGPanel.currentPanel = undefined;
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
    });
  }

  update(dagData, targetName, sourceUri) {
    if (sourceUri) this._sourceUri = sourceUri;
    this._panel.title = `FlatPPL DAG: ${targetName}`;
    this._panel.webview.postMessage({ type: 'updateDAG', data: dagData });
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

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src 'nonce-${nonce}'; style-src 'unsafe-inline';">
  <title>FlatPPL DAG</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--vscode-editor-background);
      color: var(--vscode-editor-foreground);
      font-family: var(--vscode-font-family, sans-serif);
      font-size: var(--vscode-font-size, 13px);
      overflow: hidden;
    }
    #cy { width: 100vw; height: calc(100vh - 60px); }
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
      font-size: 12px;
      white-space: pre;
      max-width: 400px;
      overflow: hidden;
      text-overflow: ellipsis;
      z-index: 100;
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
  <div id="cy"></div>
  <div id="tooltip"></div>
  <div id="legend"></div>
  <div id="info">
    <span class="hint">Click a node to see details &middot; double-click to jump to source</span>
  </div>

  <script nonce="${nonce}" src="${cytoscapeUri}"></script>
  <script nonce="${nonce}" src="${dagreUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeDagreUri}"></script>
  <script nonce="${nonce}">
  (function() {
    const vscodeApi = acquireVsCodeApi();

    const TYPE_STYLE = {
      input:         { color: '#4DD0E1', shape: 'diamond',          label: 'input (elementof)' },
      stochastic:    { color: '#B39DDB', shape: 'ellipse',          label: 'stochastic (draw)' },
      deterministic: { color: '#90A4AE', shape: 'round-rectangle',  label: 'deterministic' },
      lawof:         { color: '#81C784', shape: 'hexagon',          label: 'lawof' },
      functionof:    { color: '#FFB74D', shape: 'hexagon',          label: 'functionof' },
      fn:            { color: '#FFF176', shape: 'tag',              label: 'fn' },
      literal:       { color: '#F48FB1', shape: 'rectangle',        label: 'literal' },
      likelihood:    { color: '#EF9A9A', shape: 'octagon',          label: 'likelihood' },
      bayesupdate:   { color: '#FFAB91', shape: 'octagon',          label: 'bayesupdate' },
      module:        { color: '#80CBC4', shape: 'round-rectangle',  label: 'module' },
      table:         { color: '#A1887F', shape: 'round-rectangle',  label: 'table' },
      unknown:       { color: '#BDBDBD', shape: 'rectangle',        label: 'unknown' },
    };

    let cy = null;
    let shownTypes = new Set();

    function esc(s) {
      return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function buildLegend() {
      const el = document.getElementById('legend');
      el.innerHTML = '';
      for (const t of shownTypes) {
        const s = TYPE_STYLE[t] || TYPE_STYLE.unknown;
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
            selector: 'node[?isBoundary]',
            style: {
              'border-color': '#FFD600',
              'border-width': 3,
              'border-style': 'dashed',
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

      cy.on('tap', 'node', function(evt) {
        var d = evt.target.data();
        document.getElementById('info').innerHTML =
          '<div class="row"><span class="name">' + esc(d.label)
          + '</span><span class="type">' + esc(d.nodeType) + '</span></div>'
          + '<div class="expr">' + esc(d.expr) + '</div>';
      });

      cy.on('tap', function(evt) {
        if (evt.target === cy) {
          document.getElementById('info').innerHTML =
            '<span class="hint">Click a node to see details &middot; double-click to jump to source</span>';
        }
      });

      cy.on('dbltap', 'node', function(evt) {
        var line = evt.target.data('line');
        if (line >= 0) {
          vscodeApi.postMessage({ type: 'navigateTo', line: line });
        }
      });

      var tip = document.getElementById('tooltip');
      cy.on('mouseover', 'node', function(evt) {
        var d = evt.target.data();
        var expr = d.expr || '';
        if (!expr) return;
        tip.textContent = d.label + ' = ' + expr;
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

    function updateDAG(data) {
      if (!cy) initCy();

      shownTypes.clear();
      var elements = [];

      for (var i = 0; i < data.nodes.length; i++) {
        var node = data.nodes[i];
        var ts = TYPE_STYLE[node.type] || TYPE_STYLE.unknown;
        shownTypes.add(node.type);
        elements.push({
          group: 'nodes',
          data: {
            id: node.id,
            label: node.label || node.id,
            color: ts.color,
            shape: ts.shape,
            nodeType: node.type,
            expr: node.expr || '',
            line: node.line != null ? node.line : -1,
            isBoundary: node.isBoundary || false,
            isTarget: node.isTarget || false,
            width: Math.max((node.label || node.id).length * 9 + 24, 60),
          },
        });
      }

      for (var j = 0; j < data.edges.length; j++) {
        var edge = data.edges[j];
        elements.push({
          group: 'edges',
          data: { source: edge.source, target: edge.target, edgeType: edge.edgeType || 'data' },
        });
      }

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
      buildLegend();

      document.getElementById('info').innerHTML =
        '<span class="hint">Click a node to see details &middot; double-click to jump to source</span>';
    }

    window.addEventListener('message', function(event) {
      if (event.data && event.data.type === 'updateDAG') {
        updateDAG(event.data.data);
      }
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

module.exports = { DAGPanel };
