'use strict';
const vscode = require('vscode');
// Same vendored-bundle require pattern as extension.js — the
// installed VSIX doesn't ship node_modules/, so the path goes
// through the build-vendor output instead of the workspace
// `@flatppl/engine` symlink. isValidBindingName and variants are
// re-exported by engine/index.js, the IIFE wraps that, and the
// bundle's footer exposes the same shape to CommonJS require.
const { isValidBindingName, variants } = require('../lib/engine.min.js');

/** There is one canonical FlatPPL surface syntax (flatppl-design
    cc81e4b removed FlatPPY/FlatPPJ). Retained as a function so the
    single call sites stay stable. */
function variantIdFromUri(_uri) {
  return 'flatppl';
}

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
    // Webview-ready handshake: messages posted before the webview's
    // script attaches its `message` listener are dropped silently
    // (VS Code's webview.postMessage doesn't buffer reliably). The
    // viewer signals 'webviewReady' once its listener is in place;
    // until then we queue. After ready, we flush in FIFO order and
    // every subsequent post bypasses the queue.
    this._webviewReady = false;
    this._pendingMessages = [];
    this._panel.webview.html = this._getHtml();
    this._panel.onDidDispose(() => {
      FlatPPLPanel.currentPanel = undefined;
    });
    this._panel.webview.onDidReceiveMessage(msg => {
      if (msg.type === 'webviewReady') {
        this._webviewReady = true;
        for (const m of this._pendingMessages) this._panel.webview.postMessage(m);
        this._pendingMessages = [];
        return;
      }
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
      // Two persist primitives matching the viewer's host-adapter
      // contract:
      //   editSource(args)    — apply a WorkspaceEdit; replace
      //                         args.range or append at end when
      //                         args.range == null.
      //   promptForName(args) — collect a binding name via
      //                         vscode.window.showInputBox (the
      //                         webview's window.prompt is blocked).
      // After applyEdit we push a fresh sourceUpdate directly. The
      // workspace changeListener gates on activeTextEditor.document
      // and can miss the change when the webview has focus, so
      // relying on it would leave the viewer with stale source.
      if (msg.type === 'editSource' && this._readOnly) {
        vscode.window.showInformationMessage(
          'This FlatPPL DAG is a read-only snapshot of an embedded model — '
          + 'edit the FlatPPL inside the host Python/Julia file directly, '
          + 'then re-run "FlatPPL: Visualize Embedded Model".');
        return;
      }
      if (msg.type === 'editSource' && this._sourceUri != null) {
        const uri = this._sourceUri;
        vscode.workspace.openTextDocument(uri).then(doc => {
          const edit = new vscode.WorkspaceEdit();
          if (msg.range) {
            const range = new vscode.Range(
              new vscode.Position(msg.range.start.line, msg.range.start.col),
              new vscode.Position(msg.range.end.line,   msg.range.end.col)
            );
            edit.replace(uri, range, msg.newText);
          } else {
            const text = doc.getText();
            const endPos = doc.positionAt(text.length);
            const sep = text.length === 0 || text.endsWith('\n') ? '' : '\n';
            edit.insert(uri, endPos, sep + msg.newText + '\n');
          }
          return vscode.workspace.applyEdit(edit).then(success => {
            if (!success) return null;
            return vscode.workspace.openTextDocument(uri);
          });
        }).then(doc => {
          if (!doc) return;
          this.updateSource(doc.getText(), null, this._sourceUri, false);
        });
      }
      if (msg.type === 'promptForName') {
        const existing = new Set(msg.existingNames || []);
        const validate = (raw) => {
          const trimmed = (raw || '').trim();
          if (!trimmed) return 'Name required';
          if (!isValidBindingName(trimmed)) {
            return 'Use letters, digits, underscores; start with a letter or underscore';
          }
          if (existing.has(trimmed)) return `"${trimmed}" already exists in this module`;
          return null;
        };
        vscode.window.showInputBox({
          prompt: 'Save current values as a new preset',
          value: msg.suggested || 'preset',
          validateInput: validate,
        }).then(raw => {
          const name = raw ? raw.trim() : null;
          this._post({ type: 'promptForNameResponse', nonce: msg.nonce, name: name || null });
        });
      }
    });
  }

  _post(msg) {
    if (this._webviewReady) {
      this._panel.webview.postMessage(msg);
    } else {
      this._pendingMessages.push(msg);
    }
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
    this._readOnly = false;             // writable host-document path
    if (sourceUri) this._sourceUri = sourceUri;
    if (targetName) this._panel.title = `FlatPPL: ${targetName}`;
    this._post({
      type: 'sourceUpdate',
      source,
      targetName: targetName || null,
      pushHistory: !!pushHistory,
      variant: variantIdFromUri(this._sourceUri),
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
    this._post({ type: 'configUpdate', config });
  }



  /**
   * Render the module-level (multi-root) DAG. Distinct from
   * updateSource which centers a single-target sub-DAG; module mode
   * shows every binding linked by its dependencies. Title is
   * normalized to "FlatPPL: module" so the editor tab reflects the
   * mode. The webview pushes a history entry when pushHistory is
   * true so the back-button can return to a prior single-binding view.
   */
  showModule(source, sourceUri, pushHistory, readOnly) {
    if (readOnly) {
      // Embedded FlatPPL (extracted from a Python/Julia host string):
      // a read-only snapshot. Clear _sourceUri so the DAG-rename
      // write-back path (which targets _sourceUri at FlatPPL-relative
      // ranges) can never corrupt the host file, and don't leave a
      // stale prior .flatppl uri behind.
      this._readOnly = true;
      this._sourceUri = null;
    } else {
      this._readOnly = false;
      if (sourceUri) this._sourceUri = sourceUri;
    }
    this._panel.title = 'FlatPPL: module';
    this._post({
      type: 'showModule',
      source,
      pushHistory: !!pushHistory,
      variant: variantIdFromUri(this._sourceUri),
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
    // Webview viewer JS, sourced from the sibling @flatppl/viewer
    // workspace package and copied into lib/ by build-vendor.mjs.
    // Loading it as a regular external script — rather than as text
    // spliced into a host template literal — means the viewer source
    // can use backticks and \n freely without the host's outer
    // template literal eating them. (See the "webview escape traps"
    // memory for context.)
    const viewerUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'viewer.js')
    );

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src 'nonce-${nonce}' ${webview.cspSource} blob:; style-src 'unsafe-inline'; worker-src ${webview.cspSource} blob:; connect-src ${webview.cspSource};">
  <title>FlatPPL</title>
</head>
<body>
  <div id="flatppl-viewer-root"></div>

  <script nonce="${nonce}" src="${cytoscapeUri}"></script>
  <script nonce="${nonce}" src="${dagreUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeDagreUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeLayersUri}"></script>
  <script nonce="${nonce}" src="${cytoscapeBubblesetsUri}"></script>
  <script nonce="${nonce}" src="${echartsUri}"></script>
  <script nonce="${nonce}" src="${engineUri}"></script>
  <script nonce="${nonce}">
    // Host-supplied configuration for the viewer. The viewer reads
    // window.__FLATPPL_CONFIG__ at startup; setting it here before
    // viewer.js loads is enough. A standalone (non-VS-Code) host can
    // do the same — no other host-specific wiring is needed.
    window.__FLATPPL_CONFIG__ = {
      samplerWorkerUrl: ${JSON.stringify(samplerWorkerUri.toString())},
    };
  </script>
  <script nonce="${nonce}" src="${viewerUri}"></script>
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
