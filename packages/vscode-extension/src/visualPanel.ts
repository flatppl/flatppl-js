'use strict';
const vscode = require('vscode');
// Same vendored-bundle require pattern as extension.js — the
// installed VSIX doesn't ship node_modules/, so the path goes
// through the build-vendor output instead of the workspace
// `@flatppl/engine` symlink. isValidBindingName and variants are
// re-exported by engine/index.js, the IIFE wraps that, and the
// bundle's footer exposes the same shape to CommonJS require.
const { isValidBindingName, variants, moduleDeps, resolveBundle } = require('../lib/engine.min.js');
// The Node-side URL cache (spec §04 Remote file caching): fetches http/https
// load_module sources into the shared on-disk cache. Vendored separately — it
// is Node-only and deliberately NOT part of the browser engine bundle.
const urlCache = require('../lib/url-cache.cjs');

/** There is one canonical FlatPPL surface syntax (flatppl-design
    cc81e4b removed FlatPPY/FlatPPJ). Retained as a function so the
    single call sites stay stable. */
function variantIdFromUri(_uri: any) {
  return 'flatppl';
}

class FlatPPLPanel {
  static currentPanel: any = undefined;
  static viewType = 'flatpplPanel';

  // Class fields used as bag-of-state by the legacy constructor /
  // updateSource path. Declared `any` so the migration stays a pure
  // type-stripping pass; tighten incrementally if/when the panel
  // surface is reworked.
  _panel: any;
  _context: any;
  _sourceUri: any;
  _navUri: any;
  _navBaseLine: any;
  _readOnly: any;
  _webviewReady: any;
  _pendingMessages: any;

  static createOrShow(context: any) {
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

  constructor(panel: any, context: any) {
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
    this._panel.webview.onDidReceiveMessage((msg: any) => {
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
      if (msg.type === 'navigateTo' && (this._sourceUri != null || this._navUri != null)) {
        // Native .flatppl: jump to the reported line in _sourceUri.
        // Embedded (read-only): _sourceUri is null (write-back is
        // disabled), but a reveal-only jump into the host .py/.jl is
        // safe — shift the FlatPPL-relative line by the host line
        // where the embedded block's content starts.
        const embedded = this._sourceUri == null;
        const uri = embedded ? this._navUri : this._sourceUri;
        const line = (embedded ? this._navBaseLine : 0) + msg.line;
        vscode.window.showTextDocument(uri, {
          viewColumn: vscode.ViewColumn.One,
          preserveFocus: false,
        }).then((editor: any) => {
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
      // Cross-module navigation (spec §04 load_module): a double-click on a
      // load_module node (whole module) or a member / member-alias node
      // (focused on that member) in the DAG. Open the loaded module's file
      // in the editor (source-sync) and re-point the visualizer at it,
      // focusing `msg.member` when present. `msg.path` is the resolved
      // module path (URI path component); resolve it against the current
      // document's URI to keep the scheme/authority.
      if (msg.type === 'openModule' && msg.path && this._sourceUri) {
        // `msg.path` is the engine's RESOLVED module path (spec §04) — an
        // absolute URI-path under the always-send-path contract, so it rides
        // the current source's scheme + authority (works under remote / WSL).
        // Defensive: a non-absolute path (a contract gap) resolves against the
        // current source's directory instead of collapsing to the FS root
        // (the old `file:///bayesian_inference_common.flatppl` failure).
        const cur = this._sourceUri;
        const absPath = String(msg.path).charAt(0) === '/'
          ? msg.path
          : cur.path.replace(/\/[^/]*$/, '') + '/' + msg.path;
        const depUri = cur.with({ path: absPath });
        vscode.window.showTextDocument(depUri, {
          viewColumn: vscode.ViewColumn.One,
          preserveFocus: false,
        }).then((editor: any) => {
          // pushHistory=false: the viewer's dbltap already pushed the loader's
          // state before openModule (so the back-button can return to it).
          // Pushing again here would double-stack the loader / oscillate.
          this.updateSource(editor.document.getText(), msg.member || null, depUri, false);
        }, (err: any) => {
          vscode.window.showErrorMessage(
            'FlatPPL: could not open loaded module \'' + msg.path + '\': '
            + (err && err.message || err));
        });
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
        vscode.workspace.openTextDocument(uri).then((doc: any) => {
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
          return vscode.workspace.applyEdit(edit).then((success: any) => {
            if (!success) return null;
            return vscode.workspace.openTextDocument(uri);
          });
        }).then((doc: any) => {
          if (!doc) return;
          this.updateSource(doc.getText(), null, this._sourceUri, false);
        });
      }
      if (msg.type === 'promptForName') {
        const existing = new Set(msg.existingNames || []);
        const validate = (raw: any) => {
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
        }).then((raw: any) => {
          const name = raw ? raw.trim() : null;
          this._post({ type: 'promptForNameResponse', nonce: msg.nonce, name: name || null });
        });
      }
    });
  }

  _post(msg: any) {
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
  updateSource(source: any, targetName: any, sourceUri: any, pushHistory: any, opts?: any) {
    if (opts && opts.readOnly) {
      // Embedded cursor-follow: re-center on a binding while keeping
      // the snapshot read-only (write-back stays refused) and the
      // reveal-only host nav origin.
      this._readOnly = true;
      this._sourceUri = null;
      this._navUri = opts.navOrigin ? opts.navOrigin.uri : null;
      this._navBaseLine = opts.navOrigin ? opts.navOrigin.baseLine : 0;
      if (targetName) this._panel.title = `FlatPPL: ${targetName}`;
      this._post({
        type: 'sourceUpdate',
        source,
        targetName: targetName || null,
        pushHistory: !!pushHistory,
        variant: 'flatppl',
        // Explicit null: an embedded block has no module path. The viewer's
        // module context is sticky (a same-model update preserves it), so a
        // path-less surface must CLEAR it rather than omit it (else it would
        // inherit the previously-viewed module's path).
        path: null,
      });
      return;
    }
    this._readOnly = false;             // writable host-document path
    this._navUri = null;                // not an embedded snapshot
    if (sourceUri) this._sourceUri = sourceUri;
    if (targetName) this._panel.title = `FlatPPL: ${targetName}`;
    const base: any = {
      type: 'sourceUpdate',
      source,
      targetName: targetName || null,
      pushHistory: !!pushHistory,
      variant: variantIdFromUri(this._sourceUri),
      // ALWAYS carry the primary's resolved path (spec §04). The viewer keys
      // its cross-module navigation on it — `currentPath` tracking and the
      // back-button's source re-sync — so even a LEAF module (no further
      // load_module deps, no bundle) must report its path; otherwise the
      // back-button can't tell the DAG left the loader. It also makes the
      // module registry resolve deps against an absolute importer path, so a
      // drill's `openModule` URI stays absolute (never collapses to root).
      // Explicit (never undefined): the viewer's module context is sticky, so
      // a leaf/path-less post must CLEAR it. A native model always has a uri.
      path: this._sourceUri ? this._sourceUri.path : null,
    };
    // Multi-file (spec §04 load_module): if the source loads other modules,
    // pre-fetch their `.flatppl` sources via the workspace file system and
    // ship them as `bundleSources` (keyed by resolved path). Single-file
    // sources skip this and post synchronously (the common case, no I/O).
    let rels: string[] = [];
    try { rels = moduleDeps(source); } catch (_) { rels = []; }
    if (rels.length === 0 || !this._sourceUri) {
      this._post(base);
      return;
    }
    this._resolveBundle(source, this._sourceUri).then((bundle: any) => {
      base.bundleSources = bundle.sources;   // base.path already == bundle.primaryPath
      this._post(base);
    }, () => {
      // Resolution failed entirely — still render the primary so the
      // panel is never blank; the engine reports unresolved deps.
      this._post(base);
    });
  }

  /**
   * Pre-fetch the transitive `.flatppl` dependencies a source loads via
   * `load_module(...)` (spec §04), reading each through the workspace file
   * system. Returns `{ sources, primaryPath }` where `sources` maps each
   * dependency's RESOLVED path to its text (the engine's bundle key) and
   * `primaryPath` is the importing file's own path. Paths use the URI
   * path component (always `/`-separated, matching spec path resolution);
   * a missing dependency is left out (the engine reports it).
   */
  async _resolveBundle(source: any, sourceUri: any) {
    // Host primitive only: read one resolved dependency through the
    // workspace file system (a resolved path is a URI path component, so it
    // rides `sourceUri.with({ path })` to keep scheme + authority — works
    // under remote / WSL too). The shared engine walk (`resolveBundle`)
    // owns the moduleDeps + resolveModulePath + dedupe + tolerate-missing
    // structure, identical to the web host's.
    const readSource = async (resolved: string) => {
      // A URL dependency (spec §04 #sec:url-cache) goes through the shared
      // on-disk cache, prompting for trust on a first, unknown URL; a local
      // dependency is read through the workspace file system.
      if (urlCache.isUrl(resolved)) {
        try {
          return await urlCache.readText(resolved, { approve: (u: string) => this._approveUrl(u) });
        } catch (e: any) {
          vscode.window.showWarningMessage(
            'FlatPPL: could not load ' + resolved + ' — ' + (e && e.message || e));
          return null;   // left out of the bundle; the engine reports the dep at its call
        }
      }
      const bytes = await vscode.workspace.fs.readFile(sourceUri.with({ path: resolved }));
      return Buffer.from(bytes).toString('utf8');
    };
    return resolveBundle(sourceUri.path, source, readSource);
  }

  /**
   * Trust prompt for a not-yet-cached remote module URL (spec §04
   * #sec:url-cache: interactive tooling must obtain the user's approval before
   * fetching). Returns true to fetch and persist a `trust/<kk>/<key>` marker.
   */
  async _approveUrl(url: string) {
    const pick = await vscode.window.showWarningMessage(
      'FlatPPL: fetch and trust the remote module?\n' + url,
      { modal: false }, 'Trust', 'Cancel');
    return pick === 'Trust';
  }

  /**
   * Push the current visualization config to the webview. Called once
   * on panel creation and again whenever a relevant `flatppl.*`
   * configuration value changes. The webview clears its sample cache
   * on receipt if numbers like sampleCount have changed — cached
   * arrays were sized to the old count and can't be reused.
   */
  updateConfig(config: any) {
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
  showModule(source: any, sourceUri: any, pushHistory: any, readOnly: any, navOrigin: any) {
    if (readOnly) {
      // Embedded FlatPPL (extracted from a Python/Julia host string):
      // a read-only snapshot. Clear _sourceUri so the DAG-rename
      // write-back path (which targets _sourceUri at FlatPPL-relative
      // ranges) can never corrupt the host file, and don't leave a
      // stale prior .flatppl uri behind. `navOrigin` ({uri, baseLine})
      // enables reveal-only DAG→source jumps into the host file
      // without re-enabling write-back.
      this._readOnly = true;
      this._sourceUri = null;
      this._navUri = navOrigin ? navOrigin.uri : null;
      this._navBaseLine = navOrigin ? navOrigin.baseLine : 0;
    } else {
      this._readOnly = false;
      this._navUri = null;
      this._navBaseLine = 0;
      if (sourceUri) this._sourceUri = sourceUri;
    }
    this._panel.title = 'FlatPPL: module';
    this._post({
      type: 'showModule',
      source,
      pushHistory: !!pushHistory,
      variant: variantIdFromUri(this._sourceUri),
      // Carry the module's path (null when embedded) so the engine resolves the
      // module's load_module deps against it — the registry powers drill-down
      // from the whole-module view. Sticky viewer context: never omit.
      path: this._sourceUri ? this._sourceUri.path : null,
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
    // Temml MathML companion stylesheet — sets a font-family chain
    // over locally-installed math fonts. The accompanying
    // Temml.woff2 (script-capital font) lives next to it in lib/
    // and is referenced by a relative URL inside temml.css; the
    // webview's CSP must permit the same origin for `style-src` and
    // `font-src` so the link tag loads and the font fetch succeeds.
    const temmlCssUri = webview.asWebviewUri(
      vscode.Uri.joinPath(this._context.extensionUri, 'lib', 'temml.css')
    );

    return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src 'nonce-${nonce}' ${webview.cspSource} blob:; style-src ${webview.cspSource} 'unsafe-inline'; font-src ${webview.cspSource}; worker-src ${webview.cspSource} blob:; connect-src ${webview.cspSource};">
  <title>FlatPPL</title>
  <link rel="stylesheet" href="${temmlCssUri}">
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
