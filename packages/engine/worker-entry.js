'use strict';

// Browser/Node-worker entry shim for the FlatPPL sampler worker.
//
// This file is intentionally tiny — all interesting logic lives in
// worker.js (createWorkerHandler). The shim's only job is to wire the
// transport-agnostic handler to the host's actual worker API, in two
// flavours so the *same source file* can serve both:
//
//   * Web Worker         — `self.onmessage` / `self.postMessage`. This is
//                          what the VS Code webview uses (and what Browser
//                          / VS Code Web will use when we ship there).
//   * Node worker_threads — `parentPort.on('message')`. Used by tests and
//                          by the extension host if we ever sample server-
//                          side. We sniff it via `require` so the file is
//                          still browser-loadable when bundled (esbuild
//                          tree-shakes the unreachable branch).
//
// The shim is bundled by build-vendor.mjs as `lib/sampler-worker.min.js`
// for the webview. Tests don't need this shim — they import worker.js
// directly and drive `handle()` synchronously.

const { createWorkerHandler, transferablesOf } = require('./worker');

const handler = createWorkerHandler();

// Detect environment. `self` is defined in Web Workers (and browser main
// threads, but this file should only be loaded as a worker entry).
// Node's worker_threads exposes `parentPort`. Falling through to a no-op
// keeps `require()`-ing this file from a script harmless.
//
// IMPORTANT: keep both branches lazy. If we eagerly `require('worker_threads')`
// at the top, esbuild's browser build would inline a polyfill we don't want.
const isBrowserWorker =
  typeof self !== 'undefined' && typeof self.postMessage === 'function' &&
  typeof self.addEventListener === 'function';

if (isBrowserWorker) {
  // Web Worker path. `self` is the global scope inside the worker.
  // We attach via addEventListener (rather than self.onmessage =) so a
  // host that already attached a listener for handshake purposes isn't
  // overwritten.
  self.addEventListener('message', (e) => {
    const reply = handler.handle(e.data);
    if (reply) {
      const transfer = transferablesOf(reply);
      // postMessage's second argument is the transferList; passing an
      // empty array is harmless but explicit. Float64Array buffers are
      // moved zero-copy out of the worker.
      self.postMessage(reply, transfer);
    }
    // `dispose` returns null; honour it by closing the worker so the
    // host-side `worker.terminate()` isn't strictly required.
    if (e.data && e.data.type === 'dispose' && typeof self.close === 'function') {
      self.close();
    }
  });
} else {
  // Node worker_threads path. The `try` guards against this file being
  // required from a context that isn't a worker_threads parent (in which
  // case `parentPort` is null and we silently no-op). The `eval('require')`
  // dodge prevents esbuild from following the require at bundle time —
  // we never want `worker_threads` pulled into the browser bundle.
  try {
    const nodeRequire = (typeof require === 'function')
      ? require
      // eslint-disable-next-line no-eval
      : eval('require');
    const { parentPort } = nodeRequire('worker_threads');
    if (parentPort) {
      parentPort.on('message', (msg) => {
        const reply = handler.handle(msg);
        if (reply) {
          // Node's parentPort.postMessage takes a transferList in the
          // second argument too, but only ArrayBuffers (not typed arrays)
          // are valid entries. transferablesOf returns buffers already.
          parentPort.postMessage(reply, transferablesOf(reply));
        }
        if (msg && msg.type === 'dispose') {
          parentPort.close();
        }
      });
    }
  } catch (_e) {
    // Not in a worker context. That's fine — the file is still useful
    // when required for its side-effect-free top-level (creating the
    // handler) and tests that import handler directly via worker.js.
  }
}
