// @flatppl/viewer entry — Phase 4.
//
// Bundles main.js into a browser global (window.FlatPPLViewer.mount)
// and wires DOMContentLoaded auto-mount, preserving the pre-Phase-4
// contract exactly: `global.FlatPPLViewer = global.FlatPPLViewer || {}`
// merge semantics (does not clobber a pre-existing object the host
// may have populated), single mount property assignment, and the
// readyState==='loading' branch for the auto-mount.
//
// esbuild bundles this entry with format: 'iife' (no globalName —
// the global merge below is explicit). Both packages/viewer/build.mjs
// and packages/vscode-extension/build-vendor.mjs use this entry.

// @ts-check
import { mount } from './main.js';

/** @type {FlatPPLViewerGlobal} */
var FlatPPLViewer: FlatPPLViewerGlobal;
if (typeof window !== 'undefined') {
  FlatPPLViewer = (window.FlatPPLViewer = window.FlatPPLViewer || {});
} else if (typeof globalThis !== 'undefined') {
  // @ts-ignore — globalThis is loosely typed without DOM lib here.
  FlatPPLViewer = (globalThis.FlatPPLViewer = globalThis.FlatPPLViewer || {});
} else {
  FlatPPLViewer = {};
}
FlatPPLViewer.mount = mount;

// Auto-mount when the host provides a marker container in the DOM
// (id="flatppl-viewer-root"). Hosts that want explicit control over
// mount timing or args (e.g. standalone embed pages that wait for
// user input) can omit the marker and call FlatPPLViewer.mount(...)
// themselves. The vscode-extension's _getHtml() includes the marker,
// so existing webview behaviour is preserved.
function autoMountIfMarkerPresent() {
  const marker = (typeof document !== 'undefined')
    ? document.getElementById('flatppl-viewer-root')
    : null;
  // FlatPPLViewer.mount is typed as optional (a host could in principle
  // pre-populate the global with their own surface); the assignment above
  // guarantees it's set here, but TS doesn't track the flow across the
  // function-call boundary. The runtime guard is just for paranoia.
  if (marker && FlatPPLViewer.mount) FlatPPLViewer.mount(marker);
}
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', autoMountIfMarkerPresent);
  } else {
    autoMountIfMarkerPresent();
  }
}
