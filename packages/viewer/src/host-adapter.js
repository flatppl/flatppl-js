// @ts-check
// @flatppl/viewer — default VS Code webview host adapter (Phase 4g).
//
// VS Code permits acquireVsCodeApi() at most once per webview, so
// cachedVscodeApi gates it across multiple mount() calls. The
// defaultVscodeHost factory wires postMessage / setState / getState
// through to the four host-adapter methods the viewer's call sites
// expect. Standalone (non-VS-Code) embeds receive {} and the call
// sites no-op gracefully.

/** @typedef {import('./types').HostAdapter} HostAdapter */
/** @typedef {import('./types').VsCodeApi} VsCodeApi */

/** @type {VsCodeApi | null} */
var cachedVscodeApi = null;

/** @returns {VsCodeApi | null} */
export function getVscodeApi() {
  if (cachedVscodeApi) return cachedVscodeApi;
  if (typeof acquireVsCodeApi !== 'function') return null;
  try { cachedVscodeApi = acquireVsCodeApi(); } catch (_) { cachedVscodeApi = null; }
  return cachedVscodeApi;
}

/** @returns {HostAdapter} */
export function defaultVscodeHost() {
  // `const` (vs. the previous `var`) lets TS narrow `api` to non-null
  // inside the closures below — `var` reads as "possibly reassigned" and
  // TS conservatively keeps the wider type. Same runtime semantics.
  const api = getVscodeApi();
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
