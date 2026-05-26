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
  const FALLBACK_SOURCE = [
    '# @flatppl/web — open with #model=path/to/file.flatppl to load a real model.',
    'mu = elementof(reals)',
    'sigma = elementof(interval(0.0, inf))',
    'x = draw(Normal(mu = mu, sigma = sigma))',
    'y = 2 * x + 1',
    '',
  ].join('\n');

  let sourceHeader: any = null;
  let fileTree:     any = null;
  let titleEl:      any = null;
  let viewer:       any = null;
  let manifest:     any = null;

  // Unified source surface: one CodeMirror instance, eagerly mounted
  // at boot, toggled between view-only (readOnly + non-editable) and
  // edit (writable) via the source-pane edit button when allowEdit
  // is true. Replaces the old dual surface (`<pre>` + lazy
  // CodeMirror) so syntax highlighting, hover, find, decorations,
  // and virtual scrolling work uniformly at any file size.
  let sourceEditor: any = null;
  // Whether the editor is currently in writable mode. Mirrors the
  // toggle button's aria-pressed state and is the gate for the
  // viewer host's `canPersist` / `editSource` hooks (persist is
  // legal only while the user is in edit mode).
  let editEnabled = false;

  // The source string currently rendered in the source pane. Used by
  // showSourceIfChanged to skip the full innerHTML rewrite (and the
  // CSS / repaint flash that comes with it) when the user is only
  // navigating between bindings inside the same file.
  let lastRenderedSource: string | null = null;

  /** Map a model path's extension to a surface-syntax variant id
      ('flatppl' / 'flatppy' / 'flatppj'). Returns undefined for an
      unrecognized extension or missing path; the viewer treats
      undefined as "fall back to FlatPPL". Wraps the engine's
      variantForPath helper so any future detection rules land in
      one place. */
  function variantIdForPath(path: any) {
    if (!window.FlatPPLEngine || !window.FlatPPLEngine.variants) return undefined;
    const v = window.FlatPPLEngine.variants.variantForPath(path);
    return v ? v.id : undefined;
  }
  // The currently loaded model path, so we can short-circuit when a
  // navigation event only changes the focus target. Critical in
  // playground mode: rewriting the editor's content on every target-
  // change would wipe user edits and reset the cursor to the start
  // of the file.
  let lastModel: string | null = null;

  /** Push text into the source editor. The editor's own
   *  highlight plugin walks the engine's tokenizer + binding table
   *  on every doc change and applies decorations — so we don't need
   *  to pre-classify here. setSource is silent (suppresses onChange)
   *  so callers don't have to worry about a redundant viewer
   *  re-render. */
  function showSource(text: any, label?: any) {
    if (sourceHeader) sourceHeader.textContent = label || 'Source';
    if (!sourceEditor) return;
    sourceEditor.setSource(text);
    lastRenderedSource = text;
  }

  /** Tiny debounce: collapse rapid calls into one delayed call.
      Used by the playground's onChange path so the viewer re-renders
      after the user pauses typing rather than on every keystroke. */
  function debounce(fn: any, ms: any) {
    let t: any = null;
    return function () {
      const args = arguments;
      if (t) clearTimeout(t);
      t = setTimeout(function () { t = null; fn.apply(null, args); }, ms);
    };
  }

  // Edit-mode state, persisted across reloads (only when the deploy
  // allows editing in the first place — see EDIT_STORAGE_KEY).
  const EDIT_STORAGE_KEY = 'flatppl-web-edit-on';
  function readEditPref() {
    try { return window.localStorage && window.localStorage.getItem(EDIT_STORAGE_KEY) === '1'; }
    catch (_) { return false; }
  }
  function writeEditPref(on: any) {
    try { if (window.localStorage) window.localStorage.setItem(EDIT_STORAGE_KEY, on ? '1' : '0'); }
    catch (_) {}
  }

  /** Wire the edit-toggle button. Called once at boot, after the
   *  editor has been mounted. The button stays hidden when the
   *  deploy config has allowEdit=false (read-only-only deploys).
   *  Restores the user's last edit-mode preference from
   *  localStorage so it survives reloads. */
  function setupEditToggle() {
    const cfg = window.__FLATPPL_CONFIG__ || {};
    const toggleBtn = document.getElementById('edit-toggle');
    if (!toggleBtn) return;
    if (!cfg.allowEdit) return;
    toggleBtn.hidden = false;
    toggleBtn.addEventListener('click', function () {
      setEditMode(!editEnabled);
    });
    if (readEditPref()) setEditMode(true);
  }

  /** Switch the source editor between view (readOnly + non-editable)
   *  and edit (writable + editable). Same instance throughout — the
   *  doc, selection, history, and decorations all persist across the
   *  toggle. */
  function setEditMode(on: any) {
    if (!sourceEditor) return;
    const toggleBtn = document.getElementById('edit-toggle');
    editEnabled = !!on;
    sourceEditor.setReadOnly(!editEnabled);
    if (toggleBtn) toggleBtn.setAttribute('aria-pressed', editEnabled ? 'true' : 'false');
    writeEditPref(editEnabled);
  }

  /** Cheap variant of showSource: skip the full re-highlight when the
      content didn't change. Lets binding-click navigation (which only
      changes the focused target, not the source) avoid an innerHTML
      rewrite and the brief flicker that would come with it. */
  function showSourceIfChanged(text: any, label?: any) {
    if (text === lastRenderedSource) {
      if (sourceHeader) sourceHeader.textContent = label || 'Source';
      return;
    }
    showSource(text, label);
  }

  function showError(label: any, err: any) {
    const msg = '# Error\n# ' + (err && err.message || String(err));
    showSource(msg, label || 'error');
    if (viewer && typeof viewer.update === 'function') viewer.update(msg);
  }

  /** Render the manifest + ephemeral entries as clickable lists in
      the left pane. Ephemeral entries (created via "+ New file") get
      their own section above the manifest so the unsaved files are
      easy to find regardless of how many manifest entries there are. */
  function renderTree(currentModel: any) {
    if (!fileTree) return;
    fileTree.innerHTML = '';

    const eph = window.FlatPPLWebEphemeral;
    const ephEntries = (eph && eph.list) ? eph.list() : [];
    if (ephEntries.length > 0) {
      const ephHeader = document.createElement('div');
      ephHeader.className = 'file-list-header';
      ephHeader.textContent = 'Unsaved';
      fileTree.appendChild(ephHeader);
      const ephUl = document.createElement('ul');
      ephUl.className = 'file-list';
      for (let i = 0; i < ephEntries.length; i++) {
        const ent = ephEntries[i];
        const li = document.createElement('li');
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
        const p = document.createElement('div');
        p.className = 'pane-placeholder';
        p.textContent = 'No models.json — open a model with #model=path/to/file.flatppl.';
        fileTree.appendChild(p);
      }
      return;
    }
    const ul = document.createElement('ul');
    ul.className = 'file-list';
    for (let j = 0; j < manifest.entries.length; j++) {
      const entry = manifest.entries[j];
      const li2 = document.createElement('li');
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
  function starterSourceFor(path: any) {
    return '# ' + path + ' — new ephemeral file (session-only, not saved to disk).\n';
  }

  /** Sanitize a user-supplied filename into a path the rest of the
      gallery accepts: strip surrounding whitespace, prepend `new/`
      if missing, append a default extension when none of the three
      known surface-variant extensions are present. */
  function normalizeEphemeralPath(raw: any) {
    let s = String(raw || '').trim();
    if (!s) return null;
    if (s.indexOf('/') === -1) s = 'new/' + s;
    if (!/\.flatppl$/i.test(s)) s = s + '.flatppl';
    return s;
  }

  /** "+ New file" handler. Prompt for a name (default = next
      auto-numerated untitled), stash a starter source in the
      ephemeral store, and navigate to the new path. Auto-enables
      edit mode so the user can start typing immediately. */
  async function onNewFileClick() {
    const eph = window.FlatPPLWebEphemeral;
    if (!eph) return;
    const defaultPath = eph.nextUntitled('.flatppl');
    const raw = window.prompt(
      'New ephemeral file (session-only). Path:', defaultPath);
    if (raw == null) return;
    const path = normalizeEphemeralPath(raw);
    if (!path) return;
    // Collision check across both the manifest and the ephemeral store.
    const collidesWithManifest = manifest && manifest.entries
      && manifest.entries.some(function (e: any) { return e.path === path; });
    if (eph.has(path) || collidesWithManifest) {
      window.alert('Path already exists: ' + path);
      return;
    }
    eph.add(path, starterSourceFor(path));
    if (!editEnabled && (window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      setEditMode(true);
    }
    window.FlatPPLWebRouter.navigateTo({ model: path });
  }

  function onTreeClick(ev: any) {
    const path = ev.currentTarget.dataset.path;
    if (path) window.FlatPPLWebRouter.navigateTo({ model: path });
  }

  async function applyState(state: any) {
    // Repaint tree highlight even before the fetch completes.
    renderTree(state.model);

    // Ephemeral files have no on-disk source — they're meaningless
    // outside edit mode. Auto-enable edit mode when navigating to
    // one (gated by allowEdit so a strictly read-only deploy still
    // can't unlock editing through a hash-link).
    const ephRef = window.FlatPPLWebEphemeral;
    if (ephRef && state.model && ephRef.has(state.model)
        && !editEnabled
        && (window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      setEditMode(true);
    }

    // Same model, target-only change: skip the fetch and the source-
    // pane rewrite entirely. The editor IS the source of truth now —
    // rewriting it with the cached original would wipe any user edits
    // and reset the cursor. Read the live text directly out of the
    // editor; fall back to the last rendered text only if the editor
    // hasn't mounted yet (boot race).
    // The surface-syntax variant follows the model path's extension
    // (.flatppl / .flatppy / .flatppj). The viewer reads this on
    // every update so persist write-back emits matching syntax.
    const variantId = variantIdForPath(state.model);

    if (state.model === lastModel) {
      if (viewer) {
        const liveText = sourceEditor
          ? sourceEditor.getSource()
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
      const bundle = await window.FlatPPLWebResolver.resolveBundle(state.model);
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

  async function boot() {
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
    const backBtn = document.getElementById('nav-back');
    const fwdBtn  = document.getElementById('nav-forward');
    if (backBtn) backBtn.addEventListener('click', function () { window.history.back(); });
    if (fwdBtn)  fwdBtn.addEventListener('click',  function () { window.history.forward(); });

    // "Visualize whole module" button in the source-pane header.
    // Drops the focused target so the DAG renders every binding.
    // Routes purely through the hash router — the viewer's
    // host.setTarget hook keeps URL = viewer focus in sync, so
    // there's no longer a divergent-state case that needed an
    // explicit viewer.update fallback.
    const showModuleBtn = document.getElementById('show-module-btn');
    if (showModuleBtn) {
      showModuleBtn.addEventListener('click', function () {
        const cur = window.FlatPPLWebRouter.parseHash();
        window.FlatPPLWebRouter.navigateTo({ model: cur.model, target: null });
      });
    }

    if (!window.FlatPPLViewer || typeof window.FlatPPLViewer.mount !== 'function') {
      console.error('[@flatppl/web] FlatPPLViewer.mount is not available');
      return;
    }
    const missing: string[] = [];
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

    // Mount the source editor eagerly. The editor is the only
    // source surface now (no read-only <pre>) — it lives in view-
    // only mode by default and the edit toggle flips it to writable.
    const editorContainer = document.getElementById('source-editor');
    if (editorContainer && window.FlatPPLWebEditor) {
      sourceEditor = window.FlatPPLWebEditor.mountEditor(editorContainer, {
        initialSource: '',
        initialReadOnly: true,
        onChange: debounce(function (text: any) {
          if (!viewer) return;
          if (!editEnabled) return;  // pure-view dispatches don't re-render
          const cur = window.FlatPPLWebRouter.parseHash();
          if (window.FlatPPLWebEphemeral) {
            window.FlatPPLWebEphemeral.update(cur.model, text);
          }
          viewer.update(text, cur.target || null,
            { variant: variantIdForPath(cur.model) });
        }, 250),
        onNavigate: function (name: any) {
          const cur = window.FlatPPLWebRouter.parseHash();
          window.FlatPPLWebRouter.navigateTo({ model: cur.model, target: name });
        },
      });
    }

    // Wire the edit-toggle button (visible only when allowEdit is
    // true) and restore the user's last edit-mode preference. The
    // editor is already mounted above so there's nothing async to
    // wait for — toggling is a synchronous Compartment reconfigure.
    setupEditToggle();

    // "+ New file" button: only visible when allowEdit is on. Same
    // gate as edit mode, since creating a file is meaningless when
    // editing is disabled. Subscribing to ephemeral changes here
    // keeps the file tree in sync as entries get added.
    const newFileBtn = document.getElementById('new-file-btn');
    if (newFileBtn && (window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      newFileBtn.hidden = false;
      newFileBtn.addEventListener('click', onNewFileClick);
    }
    if (window.FlatPPLWebEphemeral && window.FlatPPLWebEphemeral.subscribe) {
      window.FlatPPLWebEphemeral.subscribe(function () {
        const cur = window.FlatPPLWebRouter.parseHash();
        renderTree(cur.model);
      });
    }

    const viewerRoot = document.getElementById('flatppl-viewer-root');

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
    const webHost = {
      revealSourceLine: function (line: any) {
        if (sourceEditor && typeof sourceEditor.revealLine === 'function') {
          sourceEditor.revealLine(line);
        }
      },
      setTitle: function (name: any) {
        // Reflect the focused binding in the browser tab; the model
        // name still leads the title.
        const cur = window.FlatPPLWebRouter.parseHash();
        const modelLabel = cur.model ? ('FlatPPL: ' + cur.model) : 'FlatPPL';
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
      setTarget: function (name: any) {
        const cur = window.FlatPPLWebRouter.parseHash();
        window.FlatPPLWebRouter.navigateTo({
          model: cur.model,
          target: name || null,
        });
      },
      /** Whether the gallery can write to source right now. True
       *  when the source editor is in edit mode; false in view
       *  mode. The unified CodeMirror surface is always present,
       *  but persist is only legal when the user has explicitly
       *  flipped the toggle to edit. */
      canPersist: function () { return editEnabled; },
      /** Prompt the user for a new binding name; loop until valid
          or cancelled. Returns Promise<string|null>. Validation
          uses the engine's isValidBindingName (shared with the
          extension host) so the rules don't drift. */
      promptForName: function (args: any) {
        const FE = window.FlatPPLEngine;
        const existing = new Set(args.existingNames || []);
        let current = args.suggested || 'preset';
        for (;;) {
          const raw = window.prompt('Save current values as a new preset. Name:', current);
          if (raw == null) return Promise.resolve(null);
          const name = raw.trim();
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
      editSource: function (args: any) {
        if (!sourceEditor || !editEnabled) return Promise.resolve(false);
        const src = sourceEditor.getSource();
        if (args.range) {
          const lineStarts = [0];
          for (let i = 0; i < src.length; i++) {
            if (src.charCodeAt(i) === 10) lineStarts.push(i + 1);
          }
          function offsetOf(loc: any) {
            const ls = lineStarts[loc.line];
            return (typeof ls === 'number' ? ls : 0) + (loc.col || 0);
          }
          const from = offsetOf(args.range.start);
          const to   = offsetOf(args.range.end);
          sourceEditor.replaceRange(from, to, args.newText);
        } else {
          const sep = src.length === 0 || src.charAt(src.length - 1) === '\n' ? '' : '\n';
          sourceEditor.replaceRange(src.length, src.length,
            sep + args.newText + '\n');
        }
        return Promise.resolve(true);
      },
    };

    viewer = window.FlatPPLViewer.mount(viewerRoot, { host: webHost });

    // The source editor handles its own click navigation:
    //   - Plain click: caret placement → cursor-driven onNavigate
    //     fires onNavigate(name) if the new caret lands on a
    //     defined-binding identifier.
    //   - Ctrl/Cmd-click: jump-to-definition (also via onNavigate).
    // No delegated `<pre>` click handler needed.

    // Manifest is non-fatal: a missing/broken models.json leaves the
    // tree empty and the gallery still works for hash-driven navigation.
    try {
      manifest = await window.FlatPPLWebManifest.load();
      if (titleEl && manifest.title) {
        titleEl.textContent = manifest.title;
      }
    } catch (err: any) {
      console.warn('[@flatppl/web] manifest load failed (non-fatal):', err.message);
      manifest = null;
    }

    window.FlatPPLWebRouter.onChange(applyState);

    // First render: if the URL specifies a model, that wins. Otherwise
    // pick the first manifest entry as the default selection so the
    // gallery shows something real on a bare visit.
    const initial = window.FlatPPLWebRouter.parseHash();
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
