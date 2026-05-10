// @flatppl/web — gallery shell entry point.
//
// Skeleton placeholder. Subsequent steps replace this with the
// three-pane orchestration: async resolver (fetch-based), hash router
// (#model=path/to/foo.flatppl), manifest loader (models.json), file
// tree, syntax-highlighted source pane, viewer mount, and the
// source-view ↔ DAG cross-pane navigation glue.
//
// The viewer bundle and engine bundle are loaded by index.html as
// regular <script> tags, so by the time this file runs they're
// available on window (FlatPPLViewer / FlatPPLEngine).

'use strict';

(function () {
  function boot() {
    // Sanity check: confirm sibling bundles loaded. Surfaced as a
    // console message rather than UI text — the boot DIV in index.html
    // is the user-visible "skeleton placeholder" message and stays
    // there for now.
    var haveEngine = (typeof window.FlatPPLEngine === 'object');
    var haveViewer = (typeof window.FlatPPLViewer === 'object');
    console.log('[@flatppl/web] skeleton booted',
      { engine: haveEngine, viewer: haveViewer });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
