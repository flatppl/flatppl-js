// @flatppl/web — gallery shell entry point.
//
// Step 1.2: three-pane layout up, viewer mounted in the right pane
// against an inline source so the integration works end-to-end before
// fetch and the manifest land. Subsequent steps replace the inline
// source with content fetched via an async resolver (1.3), wire the
// manifest-driven file tree (1.4), add syntax highlighting and
// cross-pane navigation (1.5–1.7).

'use strict';

(function () {
  // Inline smoke-test source. Small and intentionally diverse so the
  // DAG view exercises elementof / draw / arithmetic / Normal — enough
  // that "viewer renders" is a meaningful check. Replaced when the
  // fetch resolver lands.
  var SAMPLE_SOURCE = [
    '# @flatppl/web inline smoke test',
    'mu = elementof(reals)',
    'sigma = elementof(interval(0.0, inf))',
    'x = draw(Normal(mu = mu, sigma = sigma))',
    'y = 2 * x + 1',
    '',
  ].join('\n');

  function boot() {
    document.getElementById('source-view').textContent = SAMPLE_SOURCE;
    document.getElementById('source-header').textContent = 'inline-smoke-test.flatppl';

    if (!window.FlatPPLViewer || typeof window.FlatPPLViewer.mount !== 'function') {
      console.error('[@flatppl/web] FlatPPLViewer.mount is not available');
      return;
    }

    // Mount the viewer with an empty host adapter — VS-Code-only
    // callbacks (revealSourceLine, setTitle, saveState, loadState,
    // signalReady) become no-ops. The full web host adapter that
    // bridges revealSourceLine into a source-pane scroll lands in
    // step 1.7. Returns { update(source, target), dispose() }; we
    // hold the handle so later steps (manifest selection, hash
    // routing) can swap source without re-mounting.
    var viewerRoot = document.getElementById('flatppl-viewer-root');
    var viewer = window.FlatPPLViewer.mount(viewerRoot, {
      source: SAMPLE_SOURCE,
      host: {},
    });

    // Stash the viewer handle on a namespace for later steps and for
    // ad-hoc inspection from the dev console.
    window.FlatPPLWeb = { viewer: viewer };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
