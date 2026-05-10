// @flatppl/web — gallery shell entry point.
//
// Step 1.4: manifest-driven file tree. On boot the manifest loader
// fetches `models.json`; entries populate the left-pane file tree.
// Clicking an entry sets the URL hash via the router, which fires
// the existing applyState path — selection, source pane, viewer all
// stay in sync with `#model=...`. The current selection is
// highlighted in the tree.
//
// If `models.json` is absent (404), the manifest loader fails
// gracefully: the file tree shows a "no manifest" placeholder, and
// the inline-fallback source from earlier steps still works for
// hash-less / arbitrary paths. This keeps the gallery functional
// before step 1.8 ships demo content + a real manifest.
//
// Subsequent steps add syntax highlighting (1.5), source ↔ DAG
// cross-pane navigation (1.6, 1.7), demo content (1.8), and the
// Pages deploy workflow (1.9).

'use strict';

(function () {
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
  var fileTree     = null;
  var headerEl     = null;
  var viewer       = null;
  var manifest     = null;

  // The source string currently rendered in the source pane. Used by
  // showSourceIfChanged to skip the full innerHTML rewrite (and the
  // CSS / repaint flash that comes with it) when the user is only
  // navigating between bindings inside the same file.
  var lastRenderedSource = null;

  /**
   * Push text into the source pane. When the engine is available
   * we run it through the FlatPPL-aware syntax highlighter, which
   * also stamps `data-binding` attributes on identifier spans that
   * match defined binding names (used by the cross-pane click flows
   * landing in subsequent steps). If the engine isn't ready yet, we
   * fall back to plain text so the pane stays useful.
   */
  function showSource(text, label) {
    if (sourceHeader) sourceHeader.textContent = label || 'Source';
    if (!sourceView) return;
    var FE = window.FlatPPLEngine;
    var bindings = null;
    if (FE && typeof FE.processSource === 'function') {
      try {
        var processed = FE.processSource(text);
        if (processed && processed.bindings) {
          bindings = new Set(processed.bindings.keys());
        }
      } catch (e) {
        // Parse / analyzer error — fall through to highlighter
        // without binding info; the source still renders.
        bindings = null;
      }
    }
    if (window.FlatPPLWebSyntax && typeof window.FlatPPLWebSyntax.highlight === 'function') {
      sourceView.innerHTML = window.FlatPPLWebSyntax.highlight(text, bindings);
    } else {
      sourceView.textContent = text;
    }
    lastRenderedSource = text;
  }

  /** Cheap variant of showSource: skip the full re-highlight when the
      content didn't change. Lets binding-click navigation (which only
      changes the focused target, not the source) avoid an innerHTML
      rewrite and the brief flicker that would come with it. */
  function showSourceIfChanged(text, label) {
    if (text === lastRenderedSource) {
      if (sourceHeader) sourceHeader.textContent = label || 'Source';
      return;
    }
    showSource(text, label);
  }

  function showError(label, err) {
    var msg = '# Error\n# ' + (err && err.message || String(err));
    showSource(msg, label || 'error');
    if (viewer && typeof viewer.update === 'function') viewer.update(msg);
  }

  /** Render the manifest entries as a clickable list in the left pane. */
  function renderTree(currentModel) {
    if (!fileTree) return;
    fileTree.innerHTML = '';
    if (!manifest || manifest.entries.length === 0) {
      var p = document.createElement('div');
      p.className = 'pane-placeholder';
      p.textContent = 'No models.json — open a model with #model=path/to/file.flatppl.';
      fileTree.appendChild(p);
      return;
    }
    var ul = document.createElement('ul');
    ul.className = 'file-list';
    for (var i = 0; i < manifest.entries.length; i++) {
      var entry = manifest.entries[i];
      var li = document.createElement('li');
      li.className = 'file-list-item';
      if (entry.path === currentModel) li.classList.add('selected');
      li.textContent = entry.title;
      li.title = entry.path;
      li.dataset.path = entry.path;
      li.addEventListener('click', onTreeClick);
      ul.appendChild(li);
    }
    fileTree.appendChild(ul);
  }

  function onTreeClick(ev) {
    var path = ev.currentTarget.dataset.path;
    if (path) window.FlatPPLWebRouter.navigateTo({ model: path });
  }

  async function applyState(state) {
    // Repaint tree highlight even before the fetch completes.
    renderTree(state.model);

    if (!state.model) {
      showSourceIfChanged(FALLBACK_SOURCE, 'inline-smoke-test.flatppl');
      if (viewer) viewer.update(FALLBACK_SOURCE, state.target || null);
      document.title = 'FlatPPL';
      return;
    }
    showSourceIfChanged('# Loading ' + state.model + ' …', state.model);
    try {
      var bundle = await window.FlatPPLWebResolver.resolveBundle(state.model);
      showSourceIfChanged(bundle.primarySource, state.model);
      if (viewer) viewer.update(bundle.primarySource, state.target || null);
      document.title = 'FlatPPL: ' + state.model + (state.target ? ' / ' + state.target : '');
    } catch (err) {
      console.error('[@flatppl/web] resolveBundle failed:', err);
      showError(state.model, err);
      document.title = 'FlatPPL: ' + state.model + ' (error)';
    }
  }

  /** Source-pane click handler: when the user clicks on an identifier
      span carrying `data-binding`, focus that binding in the DAG view
      via a router navigation. Routing through the hash means browser
      back/forward and bookmarkable URLs stay coherent (the hash now
      encodes both the model and the focused binding). Walks up from
      ev.target so clicks on the text inside a span hit the same
      handler as clicks on the span itself. */
  function onSourceClick(ev) {
    var el = ev.target;
    while (el && el !== sourceView) {
      if (el.dataset && el.dataset.binding) {
        var name = el.dataset.binding;
        var cur = window.FlatPPLWebRouter.parseHash();
        window.FlatPPLWebRouter.navigateTo({
          model: cur.model,
          target: name,
        });
        ev.preventDefault();
        return;
      }
      el = el.parentNode;
    }
  }

  async function boot() {
    sourceView   = document.getElementById('source-view');
    sourceHeader = document.getElementById('source-header');
    fileTree     = document.getElementById('file-tree');
    headerEl     = document.getElementById('app-header');

    if (!window.FlatPPLViewer || typeof window.FlatPPLViewer.mount !== 'function') {
      console.error('[@flatppl/web] FlatPPLViewer.mount is not available');
      return;
    }
    var missing = [];
    if (!window.FlatPPLWebResolver) missing.push('resolver');
    if (!window.FlatPPLWebRouter)   missing.push('router');
    if (!window.FlatPPLWebManifest) missing.push('manifest');
    if (missing.length) {
      console.error('[@flatppl/web] missing modules:', missing.join(', '));
      return;
    }

    var viewerRoot = document.getElementById('flatppl-viewer-root');
    viewer = window.FlatPPLViewer.mount(viewerRoot, { host: {} });

    // Delegated click handler: turns a click on a binding identifier
    // in the source pane into a router navigation that focuses the
    // corresponding DAG node. Lives on the source-view root so it
    // survives every innerHTML rewrite the highlighter does.
    sourceView.addEventListener('click', onSourceClick);

    // Manifest is non-fatal: a missing/broken models.json leaves the
    // tree empty and the gallery still works for hash-driven navigation.
    try {
      manifest = await window.FlatPPLWebManifest.load();
      if (headerEl && manifest.title) {
        headerEl.textContent = manifest.title;
      }
    } catch (err) {
      console.warn('[@flatppl/web] manifest load failed (non-fatal):', err.message);
      manifest = null;
    }

    window.FlatPPLWebRouter.onChange(applyState);

    // First render: if the URL specifies a model, that wins. Otherwise
    // pick the first manifest entry as the default selection so the
    // gallery shows something real on a bare visit.
    var initial = window.FlatPPLWebRouter.parseHash();
    if (!initial.model && manifest && manifest.entries.length > 0) {
      window.FlatPPLWebRouter.navigateTo({ model: manifest.entries[0].path });
    } else {
      window.FlatPPLWebRouter.emitInitial();
    }

    window.FlatPPLWeb = { viewer: viewer, manifest: manifest };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
