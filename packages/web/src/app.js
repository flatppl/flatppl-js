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

  /** Map a model path's extension to a surface-syntax variant id
      ('flatppl' / 'flatppy' / 'flatppj'). Returns undefined for an
      unrecognized extension or missing path; the viewer treats
      undefined as "fall back to FlatPPL". Wraps the engine's
      variantForPath helper so any future detection rules land in
      one place. */
  function variantIdForPath(path) {
    if (!window.FlatPPLEngine || !window.FlatPPLEngine.variants) return undefined;
    var v = window.FlatPPLEngine.variants.variantForPath(path);
    return v ? v.id : undefined;
  }
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
    var cur = window.FlatPPLWebRouter
      ? window.FlatPPLWebRouter.parseHash() : { model: null };
    var variantId = variantIdForPath(cur.model);
    var bindings = null;
    if (FE && typeof FE.processSource === 'function') {
      try {
        var processed = FE.processSource(text, { variant: variantId });
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
      sourceView.innerHTML = window.FlatPPLWebSyntax.highlight(text, bindings,
        { variant: variantId });
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

  // Edit-mode state, persisted across reloads (only when the deploy
  // allows editing in the first place — see EDIT_STORAGE_KEY).
  var EDIT_STORAGE_KEY = 'flatppl-web-edit-on';
  function readEditPref() {
    try { return window.localStorage && window.localStorage.getItem(EDIT_STORAGE_KEY) === '1'; }
    catch (_) { return false; }
  }
  function writeEditPref(on) {
    try { if (window.localStorage) window.localStorage.setItem(EDIT_STORAGE_KEY, on ? '1' : '0'); }
    catch (_) {}
  }

  /** Wire the edit toggle. Called once at boot. When the deploy
      config has allowEdit=true, the toggle button becomes visible
      and clickable; otherwise it stays hidden and edit mode is
      unreachable. Restoring the user's last choice from
      localStorage on boot keeps their preference across reloads.
      Returns a promise the caller can await so the first
      applyState writes into the editor (when restored to "on")
      rather than briefly flashing the read-only <pre>. */
  function setupEditToggle() {
    var cfg = window.__FLATPPL_CONFIG__ || {};
    var toggleBtn = document.getElementById('edit-toggle');
    if (!toggleBtn) return Promise.resolve();
    if (!cfg.allowEdit) {
      // Stays `hidden`; never appears in the toolbar. The CodeMirror
      // bundle is never fetched.
      return Promise.resolve();
    }
    toggleBtn.hidden = false;
    toggleBtn.addEventListener('click', function () {
      setEditMode(!playgroundEditor);
    });
    if (readEditPref()) {
      return setEditMode(true);
    }
    return Promise.resolve();
  }

  /** Switch the source pane between read-only and editor mode.
      Lazy-loads the CodeMirror bundle on first enable, mounts the
      editor with the current pane content; disable disposes the
      editor and restores the read-only <pre> in place with whatever
      text was in the editor (so user edits survive a toggle off+on
      within the same session). */
  async function setEditMode(on) {
    var toggleBtn = document.getElementById('edit-toggle');
    var sourcePane = document.getElementById('source-pane');
    if (on === !!playgroundEditor) {
      // Already in the requested state; sync the button visuals
      // just in case (e.g. on first boot when persisted state is
      // already off).
      if (toggleBtn) toggleBtn.setAttribute('aria-pressed', playgroundEditor ? 'true' : 'false');
      return;
    }
    if (on) {
      if (!window.FlatPPLWebEditor) {
        console.warn('[@flatppl/web] edit requested but FlatPPLWebEditor missing — staying read-only');
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
      var editorContainer = document.getElementById('source-editor');
      if (!editorContainer) {
        editorContainer = document.createElement('div');
        editorContainer.id = 'source-editor';
        paneBody.appendChild(editorContainer);
      }
      sourceView.style.display = 'none';
      if (sourcePane) sourcePane.classList.add('playground');
      var initial = lastRenderedSource != null ? lastRenderedSource : '';
      playgroundEditor = window.FlatPPLWebEditor.mountEditor(editorContainer, {
        initialSource: initial,
        onChange: debounce(function (text) {
          if (!viewer) return;
          var cur = window.FlatPPLWebRouter.parseHash();
          // Push edits back into the ephemeral store so they survive
          // navigation away and back to the same path. update() is a
          // no-op for paths not in the store, so this is safe for
          // manifest entries the user is editing in playground mode.
          if (window.FlatPPLWebEphemeral) {
            window.FlatPPLWebEphemeral.update(cur.model, text);
          }
          viewer.update(text, cur.target || null,
            { variant: variantIdForPath(cur.model) });
        }, 250),
        onNavigate: function (name) {
          var cur = window.FlatPPLWebRouter.parseHash();
          window.FlatPPLWebRouter.navigateTo({ model: cur.model, target: name });
        },
      });
      lastRenderedSource = initial;
      if (toggleBtn) toggleBtn.setAttribute('aria-pressed', 'true');
      writeEditPref(true);
    } else {
      // Disable: pull current text out of the editor, dispose, show
      // the <pre> again. The <pre> picks up whatever the user typed
      // — read-only mode preserves the last state, no edits are
      // discarded by the toggle alone.
      var currentText = playgroundEditor.getSource();
      try { playgroundEditor.destroy(); } catch (_) {}
      playgroundEditor = null;
      var ed = document.getElementById('source-editor');
      if (ed && ed.parentNode) ed.parentNode.removeChild(ed);
      if (sourcePane) sourcePane.classList.remove('playground');
      if (sourceView) sourceView.style.display = '';
      lastRenderedSource = null;  // force a fresh highlight pass
      showSource(currentText, (sourceHeader && sourceHeader.textContent) || 'Source');
      if (toggleBtn) toggleBtn.setAttribute('aria-pressed', 'false');
      writeEditPref(false);
    }
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

  /** Render the manifest + ephemeral entries as clickable lists in
      the left pane. Ephemeral entries (created via "+ New file") get
      their own section above the manifest so the unsaved files are
      easy to find regardless of how many manifest entries there are. */
  function renderTree(currentModel) {
    if (!fileTree) return;
    fileTree.innerHTML = '';

    var eph = window.FlatPPLWebEphemeral;
    var ephEntries = (eph && eph.list) ? eph.list() : [];
    if (ephEntries.length > 0) {
      var ephHeader = document.createElement('div');
      ephHeader.className = 'file-list-header';
      ephHeader.textContent = 'Unsaved';
      fileTree.appendChild(ephHeader);
      var ephUl = document.createElement('ul');
      ephUl.className = 'file-list';
      for (var i = 0; i < ephEntries.length; i++) {
        var ent = ephEntries[i];
        var li = document.createElement('li');
        li.className = 'file-list-item file-list-item--ephemeral';
        if (ent.path === currentModel) li.classList.add('selected');
        li.textContent = ent.path.replace(/^new\//, '');
        li.title = ent.path + ' (session-only)';
        li.dataset.path = ent.path;
        li.addEventListener('click', onTreeClick);
        ephUl.appendChild(li);
      }
      fileTree.appendChild(ephUl);
    }

    if (!manifest || manifest.entries.length === 0) {
      if (ephEntries.length === 0) {
        var p = document.createElement('div');
        p.className = 'pane-placeholder';
        p.textContent = 'No models.json — open a model with #model=path/to/file.flatppl.';
        fileTree.appendChild(p);
      }
      return;
    }
    var ul = document.createElement('ul');
    ul.className = 'file-list';
    for (var j = 0; j < manifest.entries.length; j++) {
      var entry = manifest.entries[j];
      var li2 = document.createElement('li');
      li2.className = 'file-list-item';
      if (entry.path === currentModel) li2.classList.add('selected');
      li2.textContent = entry.title;
      li2.title = entry.path;
      li2.dataset.path = entry.path;
      li2.addEventListener('click', onTreeClick);
      ul.appendChild(li2);
    }
    fileTree.appendChild(ul);
  }

  /** Build the default starter source for a new ephemeral file.
      A one-line comment is friendlier than an empty buffer — the
      user can delete it or build from it. The variant comes from
      the file extension; the engine handles the rest. */
  function starterSourceFor(path) {
    return '# ' + path + ' — new ephemeral file (session-only, not saved to disk).\n';
  }

  /** Sanitize a user-supplied filename into a path the rest of the
      gallery accepts: strip surrounding whitespace, prepend `new/`
      if missing, append a default extension when none of the three
      known surface-variant extensions are present. */
  function normalizeEphemeralPath(raw) {
    var s = String(raw || '').trim();
    if (!s) return null;
    if (s.indexOf('/') === -1) s = 'new/' + s;
    if (!/\.(flatppl|flatppy|flatppj)$/i.test(s)) s = s + '.flatppl';
    return s;
  }

  /** "+ New file" handler. Prompt for a name (default = next
      auto-numerated untitled), stash a starter source in the
      ephemeral store, and navigate to the new path. Auto-enables
      edit mode so the user can start typing immediately. */
  async function onNewFileClick() {
    var eph = window.FlatPPLWebEphemeral;
    if (!eph) return;
    var defaultPath = eph.nextUntitled('.flatppl');
    var raw = window.prompt(
      'New ephemeral file (session-only). Path:', defaultPath);
    if (raw == null) return;
    var path = normalizeEphemeralPath(raw);
    if (!path) return;
    // Collision check across both the manifest and the ephemeral store.
    var collidesWithManifest = manifest && manifest.entries
      && manifest.entries.some(function (e) { return e.path === path; });
    if (eph.has(path) || collidesWithManifest) {
      window.alert('Path already exists: ' + path);
      return;
    }
    eph.add(path, starterSourceFor(path));
    if (!playgroundEditor && (window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      await setEditMode(true);
    }
    window.FlatPPLWebRouter.navigateTo({ model: path });
  }

  function onTreeClick(ev) {
    var path = ev.currentTarget.dataset.path;
    if (path) window.FlatPPLWebRouter.navigateTo({ model: path });
  }

  async function applyState(state) {
    // Repaint tree highlight even before the fetch completes.
    renderTree(state.model);

    // Ephemeral files have no on-disk source — they're meaningless
    // outside edit mode. Auto-enable edit mode when navigating to
    // one (gated by allowEdit so a strictly read-only deploy still
    // can't unlock editing through a hash-link).
    var ephRef = window.FlatPPLWebEphemeral;
    if (ephRef && state.model && ephRef.has(state.model)
        && !playgroundEditor
        && (window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      await setEditMode(true);
    }

    // Same model, target-only change: skip the fetch and the source-
    // pane rewrite entirely. In playground mode the editor IS the
    // source of truth — rewriting it with the cached original would
    // wipe user edits and reset the cursor. Read the live text out
    // of the editor when one is mounted; fall back to the last
    // rendered text otherwise.
    // The surface-syntax variant follows the model path's extension
    // (.flatppl / .flatppy / .flatppj). The viewer reads this on
    // every update so persist write-back emits matching syntax.
    var variantId = variantIdForPath(state.model);

    if (state.model === lastModel) {
      if (viewer) {
        var liveText = playgroundEditor
          ? playgroundEditor.getSource()
          : (lastRenderedSource || '');
        // pushHistory: true — target-only navigation within one model
        // (source-pane click, cursor-driven onNavigate from the
        // editor, file-tree re-click on the current entry). Treat it
        // as a user-initiated step so the viewer's in-pane back button
        // becomes useful for "where was I before I jumped to X". Has
        // the side-effect that browser back/forward through the same
        // model will also grow the viewer history; that's acceptable
        // (history is capped, and the viewer's back-btn just walks
        // whatever's on the stack).
        viewer.update(liveText, state.target || null,
          { pushHistory: true, variant: variantId });
      }
      document.title = state.model
        ? ('FlatPPL: ' + state.model + (state.target ? ' / ' + state.target : ''))
        : 'FlatPPL';
      return;
    }

    if (!state.model) {
      showSourceIfChanged(FALLBACK_SOURCE, 'inline-smoke-test.flatppl');
      if (viewer) viewer.update(FALLBACK_SOURCE, state.target || null,
        { variant: 'flatppl' });
      document.title = 'FlatPPL';
      lastModel = null;
      return;
    }
    showSourceIfChanged('# Loading ' + state.model + ' …', state.model);
    try {
      var bundle = await window.FlatPPLWebResolver.resolveBundle(state.model);
      showSourceIfChanged(bundle.primarySource, state.model);
      if (viewer) viewer.update(bundle.primarySource, state.target || null,
        { variant: variantId });
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

    // "Visualize whole module" button in the source-pane header.
    // Drops the focused target so the DAG renders every binding.
    // Routes purely through the hash router — the viewer's
    // host.setTarget hook keeps URL = viewer focus in sync, so
    // there's no longer a divergent-state case that needed an
    // explicit viewer.update fallback.
    var showModuleBtn = document.getElementById('show-module-btn');
    if (showModuleBtn) {
      showModuleBtn.addEventListener('click', function () {
        var cur = window.FlatPPLWebRouter.parseHash();
        window.FlatPPLWebRouter.navigateTo({ model: cur.model, target: null });
      });
    }

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

    // Wire the edit-toggle button (visible only when allowEdit is
    // true) and restore the user's last edit-mode preference. We
    // await it so the editor (if restored to "on") is already
    // mounted by the time the first applyState fires — that way
    // the initial render lands in the editor, not in a briefly-
    // visible <pre>.
    await setupEditToggle();

    // "+ New file" button: only visible when allowEdit is on. Same
    // gate as edit mode, since creating a file is meaningless when
    // editing is disabled. Subscribing to ephemeral changes here
    // keeps the file tree in sync as entries get added.
    var newFileBtn = document.getElementById('new-file-btn');
    if (newFileBtn && (window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      newFileBtn.hidden = false;
      newFileBtn.addEventListener('click', onNewFileClick);
    }
    if (window.FlatPPLWebEphemeral && window.FlatPPLWebEphemeral.subscribe) {
      window.FlatPPLWebEphemeral.subscribe(function () {
        var cur = window.FlatPPLWebRouter.parseHash();
        renderTree(cur.model);
      });
    }

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
      /** Mirror the viewer's internal focus into the URL so the
          hash always reflects the actual current target. DAG-side
          navigations (node clicks, double-clicks, the whole-module
          toolbar) call this; without it those internal moves
          diverged from the URL and the next source-update read a
          stale target out of the hash. The router de-dupes
          identical hashes, so the call is a no-op when no state
          change is needed. */
      setTarget: function (name) {
        var cur = window.FlatPPLWebRouter.parseHash();
        window.FlatPPLWebRouter.navigateTo({
          model: cur.model,
          target: name || null,
        });
      },
      /** Whether the gallery can write to source right now. True
          when an editor is mounted (playground edit mode); false
          when the source pane is read-only or no source is loaded. */
      canPersist: function () { return !!playgroundEditor; },
      /** Prompt the user for a new binding name; loop until valid
          or cancelled. Returns Promise<string|null>. Validation
          uses the engine's isValidBindingName (shared with the
          extension host) so the rules don't drift. */
      promptForName: function (args) {
        var FE = window.FlatPPLEngine;
        var existing = new Set(args.existingNames || []);
        var current = args.suggested || 'preset';
        for (;;) {
          var raw = window.prompt('Save current values as a new preset. Name:', current);
          if (raw == null) return Promise.resolve(null);
          var name = raw.trim();
          if (!name) { current = raw; continue; }
          if (FE && typeof FE.isValidBindingName === 'function'
              && !FE.isValidBindingName(name)) {
            window.alert('Invalid binding name: ' + name);
            current = raw;
            continue;
          }
          if (existing.has(name)) {
            window.alert('"' + name + '" is already a binding in this module.');
            current = raw;
            continue;
          }
          return Promise.resolve(name);
        }
      },
      /** Apply an edit to the source. range = { start, end } in
          (line, col) → replace; range = null → append at end-of-
          source. The editor's docChanged fires onChange, which
          re-renders the viewer via the debounced refresh; the
          rebuildDerivations reconciliation drops any matching
          override on the next pass. */
      editSource: function (args) {
        if (!playgroundEditor) return Promise.resolve(false);
        var src = playgroundEditor.getSource();
        if (args.range) {
          var lineStarts = [0];
          for (var i = 0; i < src.length; i++) {
            if (src.charCodeAt(i) === 10) lineStarts.push(i + 1);
          }
          function offsetOf(loc) {
            var ls = lineStarts[loc.line];
            return (typeof ls === 'number' ? ls : 0) + (loc.col || 0);
          }
          var from = offsetOf(args.range.start);
          var to   = offsetOf(args.range.end);
          playgroundEditor.replaceRange(from, to, args.newText);
        } else {
          var sep = src.length === 0 || src.charAt(src.length - 1) === '\n' ? '' : '\n';
          playgroundEditor.replaceRange(src.length, src.length,
            sep + args.newText + '\n');
        }
        return Promise.resolve(true);
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
