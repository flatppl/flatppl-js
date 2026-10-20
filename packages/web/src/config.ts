// Keep the bootstrap external so the page CSP can forbid inline scripts.
window.__FLATPPL_CONFIG__ = {
  samplerWorkerUrl: new URL('vendor/sampler-worker.min.js', document.baseURI).href,
  // Editing starts disabled. The gallery remembers the user's chosen mode.
  // Embedding hosts can set false to keep the gallery read-only.
  allowEdit: true,
};
