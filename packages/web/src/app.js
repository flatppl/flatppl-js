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
  var titleEl      = null;
  var viewer       = null;
  var manifest     = null;

  // Playground state. `playgroundEditor` is the EditorHandle returned
  // by FlatPPLWebEditor.mountEditor, present only in playground mode.
  // showSource / showSourceIfChanged switch their write path through
  // this when set.
  var playgroundEditor = null;

  // The source string currently rendered in the source pane. Used by
  // showSourceIfChanged to skip the full innerHTML rewrite (and the
  // CSS / repaint flash that comes with it) when the user is only
  // navigating between bindings inside the same file.
  var lastRenderedSource = null;
  // The currently loaded model path, so we can short-circuit when a
  // navigation event only changes the focus target. Critical in
  // playground mode: rewriting the editor's content on every target-
  // change would wipe user edits and reset the cursor to the start
  // of the file.
  var lastModel = null;

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
    // Playground mode: write through the CodeMirror editor instead
    // of the read-only <pre>. The editor's setSource is silent (no
    // onChange fires), so we don't trigger a redundant viewer.update —
    // applyState already calls viewer.update directly with the right
    // target after this returns.
    if (playgroundEditor) {
      playgroundEditor.setSource(text);
      lastRenderedSource = text;
      return;
    }
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

  /** Tiny debounce: collapse rapid calls into one delayed call.
      Used by the playground's onChange path so the viewer re-renders
      after the user pauses typing rather than on every keystroke. */
  function debounce(fn, ms) {
    var t = null;
    return function () {
      var args = arguments;
      if (t) clearTimeout(t);
      t = setTimeout(function () { t = null; fn.apply(null, args); }, ms);
    };
  }

  /** When playground mode is enabled, load CodeMirror and swap the
      source pane's read-only <pre> for an editable view. Bails
      silently (read-only stays) if the bundle fails to load. */
  async function maybeInitPlayground() {
    var cfg = window.__FLATPPL_CONFIG__ || {};
    if (!cfg.playground) return;
    if (!window.FlatPPLWebEditor) {
      console.warn('[@flatppl/web] playground enabled but FlatPPLWebEditor missing — staying read-only');
      return;
    }
    try {
      await window.FlatPPLWebEditor.loadBundle();
    } catch (e) {
      console.warn('[@flatppl/web] CodeMirror load failed — staying read-only:', e && e.message);
      return;
    }
    if (!sourceView) return;
    var paneBody = sourceView.parentNode;
    var sourcePane = document.getElementById('source-pane');
    var editorContainer = document.createElement('div');
    editorContainer.id = 'source-editor';
    paneBody.appendChild(editorContainer);
    sourceView.style.display = 'none';
    if (sourcePane) sourcePane.classList.add('playground');

    playgroundEditor = window.FlatPPLWebEditor.mountEditor(editorContainer, {
      initialSource: '',
      onChange: debounce(function (text) {
        // User typed: re-render the visualization. Keep the current
        // focused target (if any) by reading the router's view of
        // the URL hash. No router-navigate here — typing isn't a
        // navigation event; it just refreshes the existing target.
        if (!viewer) return;
        var cur = window.FlatPPLWebRouter.parseHash();
        viewer.update(text, cur.target || null);
      }, 250),
      onNavigate: function (name) {
        // Ctrl/Cmd+click on a binding identifier in the editor —
        // route through the same hash-navigation flow the
        // read-only pane uses. Keeps URL state coherent and
        // browser back/forward working.
        var cur = window.FlatPPLWebRouter.parseHash();
        window.FlatPPLWebRouter.navigateTo({ model: cur.model, target: name });
      },
    });
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

    // Same model, target-only change: skip the fetch and the source-
    // pane rewrite entirely. In playground mode the editor IS the
    // source of truth — rewriting it with the cached original would
    // wipe user edits and reset the cursor. Read the live text out
    // of the editor when one is mounted; fall back to the last
    // rendered text otherwise.
    if (state.model === lastModel) {
      if (viewer) {
        var liveText = playgroundEditor
          ? playgroundEditor.getSource()
          : (lastRenderedSource || '');
        viewer.update(liveText, state.target || null);
      }
      document.title = state.model
        ? ('FlatPPL: ' + state.model + (state.target ? ' / ' + state.target : ''))
        : 'FlatPPL';
      return;
    }

    if (!state.model) {
      showSourceIfChanged(FALLBACK_SOURCE, 'inline-smoke-test.flatppl');
      if (viewer) viewer.update(FALLBACK_SOURCE, state.target || null);
      document.title = 'FlatPPL';
      lastModel = null;
      return;
    }
    showSourceIfChanged('# Loading ' + state.model + ' …', state.model);
    try {
      var bundle = await window.FlatPPLWebResolver.resolveBundle(state.model);
      showSourceIfChanged(bundle.primarySource, state.model);
      if (viewer) viewer.update(bundle.primarySource, state.target || null);
      document.title = 'FlatPPL: ' + state.model + (state.target ? ' / ' + state.target : '');
      lastModel = state.model;
    } catch (err) {
      console.error('[@flatppl/web] resolveBundle failed:', err);
      showError(state.model, err);
      document.title = 'FlatPPL: ' + state.model + ' (error)';
      // Leave lastModel unchanged so a successful retry triggers the
      // full fetch + source-rewrite path.
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
    titleEl      = document.getElementById('app-title');

    // Back / forward navigate through the browser's URL-hash
    // history. Hash navigation pushes entries automatically on every
    // router.navigateTo, so binding clicks, cursor moves onto a
    // binding, file-tree clicks, and DAG node selections all
    // become history entries the user can walk through. The
    // browser controls the history cap (~50 entries on most
    // browsers); we don't try to override it.
    var backBtn = document.getElementById('nav-back');
    var fwdBtn  = document.getElementById('nav-forward');
    if (backBtn) backBtn.addEventListener('click', function () { window.history.back(); });
    if (fwdBtn)  fwdBtn.addEventListener('click',  function () { window.history.forward(); });

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

    // Install the layout manager (collapsible file pane + drag-to-
    // resize handles between every pair of adjacent panes). The
    // manager owns #app's grid-template-columns and persists the
    // user's chosen widths + collapse state in localStorage.
    if (window.FlatPPLWebLayout) {
      window.FlatPPLWebLayout.install({
        toggleButton: document.getElementById('toggle-files'),
      });
    }

    // Initialize playground mode (lazy-load CodeMirror, swap source
    // pane to editor) before manifest+router so the first applyState
    // writes into the editor rather than the now-hidden <pre>.
    await maybeInitPlayground();

    var viewerRoot = document.getElementById('flatppl-viewer-root');

    // Web host adapter for the viewer. The viewer calls these
    // host-side hooks for IDE-only concerns it can't perform itself
    // (cross-pane source navigation, panel-title updates, persisted
    // UI state). Each method is optional from the viewer's
    // standpoint; we implement only what makes sense for a web
    // gallery.
    //
    //   revealSourceLine(line) — Ctrl+click on a DAG node fires this
    //                            with the binding's source line. We
    //                            scroll the source pane to that line
    //                            and flash-highlight it. The line is
    //                            zero-indexed, matching the tokenizer
    //                            (and the data-line attributes the
    //                            highlighter writes).
    //   setTitle(name)         — viewer requests a panel-title
    //                            update; we mirror it into the
    //                            document.title so the browser-tab
    //                            label reflects the focused binding.
    //
    // saveState / loadState are intentionally omitted — the URL hash
    // is the source of truth for navigation state, no per-tab storage
    // needed yet.
    var webHost = {
      revealSourceLine: function (line) {
        // Playground mode: scroll the CodeMirror editor and place
        // the cursor at the start of the target line.
        if (playgroundEditor && typeof playgroundEditor.revealLine === 'function') {
          playgroundEditor.revealLine(line);
          return;
        }
        // Read-only mode: scroll + flash the pre-rendered .src-line.
        if (!sourceView) return;
        var sel = '.src-line[data-line="' + line + '"]';
        var el = sourceView.querySelector(sel);
        if (!el) return;
        el.scrollIntoView({ block: 'center', behavior: 'smooth' });
        // Re-trigger the CSS animation by toggling the class with a
        // forced reflow in between, so two consecutive Ctrl-clicks
        // on the same line both flash.
        el.classList.remove('src-line-flash');
        // Force reflow.
        // eslint-disable-next-line no-unused-expressions
        el.offsetWidth;
        el.classList.add('src-line-flash');
      },
      setTitle: function (name) {
        // Reflect the focused binding in the browser tab; the model
        // name still leads the title.
        var cur = window.FlatPPLWebRouter.parseHash();
        var modelLabel = cur.model ? ('FlatPPL: ' + cur.model) : 'FlatPPL';
        document.title = name ? (modelLabel + ' / ' + name) : modelLabel;
      },
    };

    viewer = window.FlatPPLViewer.mount(viewerRoot, { host: webHost });

    // Delegated click handler: turns a click on a binding identifier
    // in the source pane into a router navigation that focuses the
    // corresponding DAG node. Lives on the source-view root so it
    // survives every innerHTML rewrite the highlighter does.
    sourceView.addEventListener('click', onSourceClick);

    // Manifest is non-fatal: a missing/broken models.json leaves the
    // tree empty and the gallery still works for hash-driven navigation.
    try {
      manifest = await window.FlatPPLWebManifest.load();
      if (titleEl && manifest.title) {
        titleEl.textContent = manifest.title;
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
