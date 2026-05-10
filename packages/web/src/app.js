// @flatppl/web — gallery shell entry point.
//
// Step 1.3: hash-driven model selection. On boot the router parses
// `location.hash`; if `model=...` is present, the resolver fetches
// that .flatppl source and the viewer renders it. Otherwise the
// shell falls back to the inline smoke-test source from step 1.2 so
// `index.html` still renders on its own. On hashchange (file-tree
// click in step 1.4, URL edit, back/forward) the resolver re-fetches
// and the viewer re-renders without a page reload.
//
// Subsequent steps add the manifest-driven file tree (1.4), syntax
// highlighting (1.5), source ↔ DAG cross-pane navigation (1.6, 1.7),
// and a real web host adapter that bridges revealSourceLine into a
// source-pane scroll.

'use strict';

(function () {
  // Inline fallback used when no `#model=...` is present in the URL.
  // Kept tiny and only exercising a handful of constructs — its job
  // is to confirm the viewer renders when the user opens index.html
  // raw. The manifest-driven tree (step 1.4) will pick a real demo
  // model as the default selection and this fallback retires.
  var FALLBACK_SOURCE = [
    '# @flatppl/web — open with #model=path/to/file.flatppl to load a real model.',
    'mu = elementof(reals)',
    'sigma = elementof(interval(0.0, inf))',
    'x = draw(Normal(mu = mu, sigma = sigma))',
    'y = 2 * x + 1',
    '',
  ].join('\n');

  var sourceView   = null;
  var sourceHeader = null;
  var viewer       = null;

  function showSource(text, label) {
    if (sourceView)   sourceView.textContent = text;
    if (sourceHeader) sourceHeader.textContent = label || 'Source';
  }

  function showError(label, err) {
    var msg = '# Error\n# ' + (err && err.message || String(err));
    showSource(msg, label || 'error');
    if (viewer && typeof viewer.update === 'function') {
      // Surface the error in the viewer too so the right pane isn't
      // stuck on stale content. The viewer will report a parse error
      // for the leading '#' line followed by free text, which is the
      // closest "render an error message" we have without a dedicated
      // error path. Acceptable for now; refine later if it confuses.
      viewer.update(msg);
    }
  }

  async function applyState(state) {
    if (!state.model) {
      // No `model=` in hash — render the inline fallback so the page
      // still shows something useful.
      showSource(FALLBACK_SOURCE, 'inline-smoke-test.flatppl');
      if (viewer) viewer.update(FALLBACK_SOURCE, state.target || null);
      document.title = 'FlatPPL';
      return;
    }
    showSource('# Loading ' + state.model + ' …', state.model);
    try {
      var bundle = await window.FlatPPLWebResolver.resolveBundle(state.model);
      showSource(bundle.primarySource, state.model);
      if (viewer) viewer.update(bundle.primarySource, state.target || null);
      document.title = 'FlatPPL: ' + state.model;
    } catch (err) {
      console.error('[@flatppl/web] resolveBundle failed:', err);
      showError(state.model, err);
      document.title = 'FlatPPL: ' + state.model + ' (error)';
    }
  }

  function boot() {
    sourceView   = document.getElementById('source-view');
    sourceHeader = document.getElementById('source-header');

    if (!window.FlatPPLViewer || typeof window.FlatPPLViewer.mount !== 'function') {
      console.error('[@flatppl/web] FlatPPLViewer.mount is not available');
      return;
    }
    if (!window.FlatPPLWebResolver || !window.FlatPPLWebRouter) {
      console.error('[@flatppl/web] resolver or router missing — check script tags in index.html');
      return;
    }

    // Mount the viewer with no initial source. applyState below
    // pushes the first source through viewer.update either from the
    // hash or from the fallback.
    var viewerRoot = document.getElementById('flatppl-viewer-root');
    viewer = window.FlatPPLViewer.mount(viewerRoot, { host: {} });

    window.FlatPPLWebRouter.onChange(applyState);
    window.FlatPPLWebRouter.emitInitial();

    window.FlatPPLWeb = { viewer: viewer };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
