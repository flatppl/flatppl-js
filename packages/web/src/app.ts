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

  // The source text as it was last persisted (loaded or saved). The
  // dirty marker on the source-pane header reads
  //   dirty = (sourceEditor.getSource() !== lastSavedSource)
  // and the save button flips between active and idle on the same
  // signal. Updated by showSource (which loads a file) and by
  // handleSave (which writes it back). Stays null until the first
  // file loads so the dirty marker doesn't briefly flash on boot
  // while the empty initial buffer compares unequal to nothing.
  let lastSavedSource: string | null = null;
  // The model path whose source `lastSavedSource` belongs to. Reset
  // alongside lastSavedSource on every showSource so navigation
  // between files re-anchors the dirty check correctly.
  let lastSavedModel: string | null = null;

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
   *  re-render. Also resets the dirty-tracking baseline — loading a
   *  fresh file is by definition not dirty. */
  function showSource(text: any, label?: any) {
    if (sourceHeader) sourceHeader.textContent = label || 'Source';
    if (!sourceEditor) return;
    sourceEditor.setSource(text);
    lastRenderedSource = text;
    const cur = window.FlatPPLWebRouter
      ? window.FlatPPLWebRouter.parseHash() : { model: null };
    lastSavedSource = text;
    lastSavedModel  = cur.model;
    updateDirtyMarker();
  }

  /** True when the path has work that would be lost on reload:
   *
   *  - It's the currently-loaded model AND the editor buffer
   *    differs from `lastSavedSource` (we're in edit mode and
   *    the user has typed something since the last persist).
   *  - It's any path in the ephemeral store — those are session-
   *    only buffers (cleared on reload), so they always count as
   *    "would lose work".
   *
   *  Used by the sidebar's per-entry save-icon marker AND by the
   *  beforeunload tab-close warning so the two stay in sync. */
  function isPathDirty(path: string): boolean {
    if (!path) return false;
    const eph = window.FlatPPLWebEphemeral;
    if (eph && eph.has && eph.has(path)) return true;
    if (!sourceEditor) return false;
    if (!editEnabled) return false;
    const cur = window.FlatPPLWebRouter
      ? window.FlatPPLWebRouter.parseHash() : { model: null };
    if (cur.model !== path) return false;
    if (lastSavedModel !== path) return false;
    return sourceEditor.getSource() !== lastSavedSource;
  }

  /** True iff anything in the gallery currently holds unsaved
   *  work. Drives the `beforeunload` warning — the browser only
   *  prompts on tab close / reload / external navigation when
   *  this returns true. */
  function hasUnsavedWork(): boolean {
    const eph = window.FlatPPLWebEphemeral;
    if (eph && eph.list && eph.list().length > 0) return true;
    if (!sourceEditor || !editEnabled) return false;
    const cur = window.FlatPPLWebRouter
      ? window.FlatPPLWebRouter.parseHash() : { model: null };
    if (!cur.model) return false;
    if (lastSavedModel !== cur.model) return false;
    return sourceEditor.getSource() !== lastSavedSource;
  }

  /** Refresh visibility of the source-pane header buttons that
   *  depend on the currently-loaded model. Cheap; called from
   *  applyState (URL changes) and from the user-store subscriber
   *  (so deleting the open file from outside still hides the
   *  delete button). */
  function updateHeaderButtons() {
    const cur = window.FlatPPLWebRouter
      ? window.FlatPPLWebRouter.parseHash() : { model: null };
    const userStore = window.FlatPPLWebUserStore;
    const downloadBtn = document.getElementById('download-btn');
    const deleteBtn   = document.getElementById('delete-btn');
    if (downloadBtn) downloadBtn.hidden = !cur.model;
    if (deleteBtn) {
      const isUserFile = !!(userStore && cur.model
        && cur.model.indexOf(userStore.USER_PREFIX) === 0
        && userStore.has(cur.model));
      deleteBtn.hidden = !isUserFile;
    }
  }

  function onDownloadClick() {
    const cur = window.FlatPPLWebRouter.parseHash();
    if (cur && cur.model) downloadFile(cur.model);
  }

  function onDeleteClick() {
    const cur = window.FlatPPLWebRouter.parseHash();
    if (!cur || !cur.model) return;
    if (!window.confirm('Delete "' + cur.model + '"? This cannot be undone.')) return;
    deleteUserFile(cur.model, /* navigateTo */ null);
  }

  /** Click handler for the inline save (💾) icon next to a dirty
   *  entry in the file tree. For the currently-loaded model this is
   *  just `handleSave`. For an ephemeral entry that ISN'T currently
   *  loaded we promote it directly without changing the active
   *  view — picks the same `user/<basename>` target the regular
   *  promote path would, drops the ephemeral entry, and lets the
   *  user keep working where they are. */
  function onEntrySaveClick(path: string) {
    const cur = window.FlatPPLWebRouter.parseHash();
    if (cur && cur.model === path) {
      handleSave();
      return;
    }
    // Non-active path. Only ephemerals reach here in practice —
    // a user-store or read-only entry can only be "dirty" when
    // it's the active model.
    const eph = window.FlatPPLWebEphemeral;
    const userStore = window.FlatPPLWebUserStore;
    if (eph && userStore && eph.has && eph.has(path)) {
      const text = eph.get(path) || '';
      const target = userStore.pathForUpload(basenameOf(path));
      userStore.save(target, text, { parent: null });
      eph.remove(path);
    }
  }

  /** Click handler for the inline revert (↺) icon. Mirrors the
   *  destructive actions in the context menu but does NOT prompt
   *  for confirmation — the icon is right there, deliberate, and
   *  the user can re-edit by typing again. */
  function onEntryRevertClick(path: string) {
    const cur = window.FlatPPLWebRouter.parseHash();
    const eph = window.FlatPPLWebEphemeral;
    if (eph && eph.has && eph.has(path)) {
      eph.remove(path);
      if (cur && cur.model === path) {
        const fallback = manifest && manifest.entries && manifest.entries[0]
          ? manifest.entries[0].path : null;
        if (fallback) window.FlatPPLWebRouter.navigateTo({ model: fallback });
      }
      return;
    }
    // Non-ephemeral revert: discard the editor buffer and reload
    // the persisted source. Only meaningful for the currently-
    // loaded model with a dirty buffer.
    if (!cur || cur.model !== path || !sourceEditor) return;
    (async function () {
      try {
        const bundle = await window.FlatPPLWebResolver.resolveBundle(path);
        if (bundle && typeof bundle.primarySource === 'string') {
          showSource(bundle.primarySource, path);
          // Refresh the viewer from the restored source.
          if (viewer) viewer.update(bundle.primarySource, cur.target || null);
        }
      } catch (e: any) {
        console.error('[@flatppl/web] revert failed:', e);
      }
    })();
  }

  // Was the current model dirty on the previous keystroke? Used to
  // avoid redrawing the whole file tree on every keystroke — the
  // sidebar's save/revert icons only need to flip when the dirty
  // status changes, not when a single character ticks the
  // already-dirty buffer further.
  let _wasDirty = false;

  /** Update the source-pane header's "* " dirty prefix and the
   *  save button's visual state to reflect the buffer's
   *  agreement (or not) with the last persisted source. Cheap to
   *  call on every keystroke. Also fires a sidebar re-render on
   *  the transition from clean→dirty or dirty→clean so the inline
   *  save/revert icons appear/disappear in step. */
  function updateDirtyMarker() {
    if (!sourceHeader) return;
    const cur = window.FlatPPLWebRouter
      ? window.FlatPPLWebRouter.parseHash() : { model: null };
    const text = sourceEditor ? sourceEditor.getSource() : '';
    const dirty = !!editEnabled
      && lastSavedModel === cur.model
      && text !== lastSavedSource;
    const baseLabel = (sourceHeader.textContent || '').replace(/^\* /, '');
    sourceHeader.textContent = (dirty ? '* ' : '') + baseLabel;
    const saveBtn = document.getElementById('save-btn');
    if (saveBtn) saveBtn.setAttribute('aria-pressed', dirty ? 'true' : 'false');
    if (dirty !== _wasDirty) {
      _wasDirty = dirty;
      renderTree(cur.model);
    }
  }

  /** Tiny debounce: collapse rapid calls into one delayed call.
      Used by the playground's onChange path so the viewer re-renders
      after the user pauses typing rather than on every keystroke. */
  function debounce(fn: (...args: any[]) => void, ms: number): (...args: any[]) => void {
    let t: any = null;
    return function (...args: any[]) {
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
    const saveBtn = document.getElementById('save-btn');
    if (saveBtn) {
      saveBtn.addEventListener('click', handleSave);
    }
    if (readEditPref()) setEditMode(true);
  }

  /** Switch the source editor between view (readOnly + non-editable)
   *  and edit (writable + editable). Same instance throughout — the
   *  doc, selection, history, and decorations all persist across the
   *  toggle. */
  function setEditMode(on: any) {
    if (!sourceEditor) return;
    const toggleBtn = document.getElementById('edit-toggle');
    const saveBtn   = document.getElementById('save-btn');
    editEnabled = !!on;
    sourceEditor.setReadOnly(!editEnabled);
    if (toggleBtn) toggleBtn.setAttribute('aria-pressed', editEnabled ? 'true' : 'false');
    // Save action only makes sense in edit mode. The Ctrl/Cmd-S
    // keymap binding stays installed at all times (it's a single
    // CodeMirror keymap entry) — handleSave's own guard turns
    // out-of-mode invocations into no-ops.
    if (saveBtn) saveBtn.hidden = !editEnabled;
    writeEditPref(editEnabled);
    updateDirtyMarker();
  }

  /** Default user/ filename suggestion for forking a read-only
   *  file: `user/<stem>-modified.flatppl`, with a numeric suffix
   *  bumped (-modified2, -modified3, ...) until the result is
   *  free. The user can edit the suggestion in the save prompt. */
  function suggestForkName(origPath: string): string {
    const userStore = window.FlatPPLWebUserStore;
    if (!userStore) return 'user/' + basenameOf(origPath);
    const base = basenameOf(origPath);
    const dot  = base.lastIndexOf('.');
    const stem = dot > 0 ? base.slice(0, dot) : base;
    const ext  = dot > 0 ? base.slice(dot)    : '.flatppl';
    let candidate = userStore.USER_PREFIX + stem + '-modified' + ext;
    if (!userStore.has(candidate)) return candidate;
    let n = 2;
    for (;;) {
      candidate = userStore.USER_PREFIX + stem + '-modified' + n + ext;
      if (!userStore.has(candidate)) return candidate;
      n += 1;
    }
  }

  /** Prompt the user for a `user/`-prefixed target path for a
   *  read-only fork. Returns the validated path or null when the
   *  user cancelled. Allows either `user/<name>.flatppl` or a
   *  bare `<name>.flatppl` (prepended with `user/` automatically);
   *  rejects names that don't end in `.flatppl`. If the chosen
   *  name already exists in the store, asks for overwrite
   *  confirmation before returning. */
  function promptForUserPath(origPath: string): string | null {
    const userStore = window.FlatPPLWebUserStore;
    if (!userStore) return null;
    const suggested = suggestForkName(origPath);
    const raw = window.prompt(
      'Save your edits to. The original "' + origPath
      + '" stays unchanged:',
      suggested);
    if (raw == null) return null;
    let typed = raw.trim();
    if (!typed) return null;
    if (typed.indexOf(userStore.USER_PREFIX) !== 0) {
      typed = userStore.USER_PREFIX + typed;
    }
    if (!/\.flatppl$/.test(typed)) {
      window.alert('File name must end with .flatppl');
      return null;
    }
    if (userStore.has(typed)) {
      if (!window.confirm('"' + typed + '" already exists. Overwrite?')) {
        return null;
      }
    }
    return typed;
  }

  /** Save-on-action. Called by the save button and the Ctrl/Cmd-S
   *  keymap binding. Behaviour depends on the current model path:
   *
   *  - `user/...`  — already a user file. Persist via
   *    `FlatPPLWebUserStore.updateSource`.
   *  - `new/...`   — ephemeral session-only buffer. Promote into
   *    the user store under `user/<basename>` (with collision
   *    bumping), drop the ephemeral entry, and re-route the URL
   *    hash so the gallery follows the new path.
   *  - read-only  — prompt the user for a `user/`-prefixed
   *    filename (default `user/<stem>-modified.flatppl`, with
   *    `-modified2` / `-modified3` / ... bumping if the default
   *    is taken). The user can edit the suggestion; an existing
   *    name triggers an overwrite confirm. The saved file has
   *    NO linkage back to the original — it's an ordinary user
   *    file from there on. Then re-route the URL hash to it.
   *
   *  In every case `lastSavedSource` is bumped so the dirty marker
   *  clears, and the sidebar redraws via the user-store's own
   *  change emitter. */
  function handleSave() {
    if (!sourceEditor || !editEnabled) return;
    const userStore = window.FlatPPLWebUserStore;
    if (!userStore) return;
    const cur = window.FlatPPLWebRouter.parseHash();
    if (!cur.model) return;
    const text = sourceEditor.getSource();

    let target = cur.model;

    if (cur.model.indexOf(userStore.USER_PREFIX) === 0) {
      // Already a user file. Quiet update — modifiedAt bumps,
      // parent preserved by updateSource.
      userStore.updateSource(cur.model, text);
    } else if (cur.model.indexOf('new/') === 0) {
      // Ephemeral → user/. Promote, then drop the ephemeral
      // entry so the "Unsaved" section empties.
      target = userStore.pathForUpload(basenameOf(cur.model));
      userStore.save(target, text, { parent: null });
      if (window.FlatPPLWebEphemeral) {
        window.FlatPPLWebEphemeral.remove(cur.model);
      }
    } else {
      // Read-only (examples/ or test-cases/). Forks become
      // independent files — no metadata linkage back to the
      // original — so the user picks a name explicitly and the
      // sidebar later treats the fork like any other user file.
      // The default suggestion follows the user's stated
      // convention: `user/<stem>-modified.flatppl`, with the
      // suffix bumped (-modified2, -modified3, ...) until free.
      // The user can edit the suggestion; existing names trigger
      // an overwrite confirm rather than silent collision.
      const typed = promptForUserPath(cur.model);
      if (typed == null) return;
      target = typed;
      userStore.save(target, text, { parent: null });
    }

    lastSavedSource = text;
    lastSavedModel  = target;
    updateDirtyMarker();

    if (target !== cur.model) {
      window.FlatPPLWebRouter.navigateTo({ model: target, target: cur.target });
    }
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

  // localStorage key + helpers for folder open/closed state.
  // Default expansion: examples open, test-cases collapsed, user
  // open iff non-empty. Once the user toggles a folder we remember
  // their choice so reloads don't fight muscle memory.
  const FOLDER_STATE_KEY = 'flatppl-web:folder-open';
  function readFolderState(): Record<string, boolean | undefined> {
    try {
      const raw = window.localStorage && window.localStorage.getItem(FOLDER_STATE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === 'object') return parsed;
      }
    } catch (_) {}
    return {};
  }
  function writeFolderState(state: Record<string, boolean>) {
    try {
      if (window.localStorage) {
        window.localStorage.setItem(FOLDER_STATE_KEY, JSON.stringify(state));
      }
    } catch (_) {}
  }
  function basenameOf(path: string): string {
    const i = path.lastIndexOf('/');
    return i < 0 ? path : path.slice(i + 1);
  }

  /** Append the inline save (💾) + revert (↺) action icons to a
   *  file-tree entry. Right-floated so they sit at the row's
   *  trailing edge regardless of filename length, with
   *  pointer-events restricted so clicking the icons doesn't also
   *  trigger the row's navigation click. */
  function attachEntryActions(li: HTMLElement, path: string) {
    const revert = document.createElement('span');
    revert.className = 'file-list-action file-list-action-revert';
    revert.setAttribute('aria-hidden', 'true');
    revert.textContent = '↺';   // ↺ anticlockwise open circle arrow
    revert.title = 'Revert (discard changes / delete ephemeral)';
    revert.addEventListener('click', function (ev: any) {
      ev.stopPropagation();
      onEntryRevertClick(path);
    });
    const save = document.createElement('span');
    save.className = 'file-list-action file-list-action-save';
    save.setAttribute('aria-hidden', 'true');
    save.textContent = '💾';   // 💾 floppy disk
    save.title = 'Save';
    save.addEventListener('click', function (ev: any) {
      ev.stopPropagation();
      onEntrySaveClick(path);
    });
    // Float order: rightmost span = first floated, so save floats
    // first (right edge), then revert (left of save).
    li.appendChild(save);
    li.appendChild(revert);
    li.classList.add('file-list-item--dirty');
  }

  /** Render the file pane as three top-level folders
   *  (examples / test-cases / user) plus an optional "Unsaved"
   *  section above them for ephemeral session-only buffers.
   *  Folders are collapsible; open/closed state persists in
   *  localStorage. A read-only file with a user fork shows a small
   *  modified-dot indicator and clicking it routes to the fork
   *  (the original is still reachable via "Reset to original"
   *  on the fork). */
  function renderTree(currentModel: any) {
    if (!fileTree) return;
    fileTree.innerHTML = '';

    const eph = window.FlatPPLWebEphemeral;
    const ephEntries = (eph && eph.list) ? eph.list() : [];
    const userStore = window.FlatPPLWebUserStore;
    const userEntries = (userStore && userStore.list) ? userStore.list() : [];

    // Bucket manifest entries by top-level prefix. Anything that
    // doesn't match a known bucket goes into the examples folder by
    // default (the manifest historically held only demo/ and
    // examples/; future categories would land here).
    const buckets: Record<string, any[]> = { examples: [], 'test-cases': [] };
    if (manifest && manifest.entries) {
      for (let i = 0; i < manifest.entries.length; i++) {
        const e = manifest.entries[i];
        if (typeof e.path === 'string' && e.path.indexOf('test-cases/') === 0) {
          buckets['test-cases'].push(e);
        } else if (typeof e.path === 'string' && e.path.indexOf('demo/') === 0) {
          // Legacy bucket name — keep mapping demo/ → test-cases
          // until the manifest catches up with the gallery's
          // user-facing folder labels.
          buckets['test-cases'].push(e);
        } else {
          buckets.examples.push(e);
        }
      }
    }

    // Unsaved (ephemeral) section above the folders.
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
        li.addEventListener('contextmenu', (function (path: string) {
          return function (ev: any) {
            ev.preventDefault();
            showTreeContextMenu(ev, {
              path: path, targetPath: path,
              isEphemeral: true,
            });
          };
        })(ent.path));
        // Every ephemeral is by definition "unsaved" (session-only;
        // lost on reload), so the save + revert action icons live
        // on every ephemeral row.
        attachEntryActions(li, ent.path);
        ephUl.appendChild(li);
      }
      fileTree.appendChild(ephUl);
    }

    const persisted = readFolderState();
    renderFolder('examples',   'Examples',   buckets.examples,
      persisted.examples !== undefined ? !!persisted.examples : true,
      currentModel, /* italic */ false);
    renderFolder('test-cases', 'Test cases', buckets['test-cases'],
      persisted['test-cases'] !== undefined ? !!persisted['test-cases'] : false,
      currentModel, /* italic */ false);
    renderFolder('user',       'User',       userEntries.map(function (u: any) {
        return { path: u.path, title: basenameOf(u.path) };
      }),
      persisted.user !== undefined ? !!persisted.user : (userEntries.length > 0),
      currentModel, /* italic */ false);

    // Empty state — only relevant if literally nothing is in any
    // folder and there are no ephemeral entries. The folders
    // themselves render their own per-folder empty labels.
    if (ephEntries.length === 0
        && buckets.examples.length === 0
        && buckets['test-cases'].length === 0
        && userEntries.length === 0) {
      const p = document.createElement('div');
      p.className = 'pane-placeholder';
      p.textContent = 'No models.json — open a model with #model=path/to/file.flatppl.';
      fileTree.appendChild(p);
    }
  }

  /** Render one folder header + entry list. Open/closed state is
   *  toggled on click and persisted across reloads. Entries are
   *  rendered as plain navigation targets; user files are
   *  independent of any read-only origin (saving a fork prompts
   *  for a fresh `user/...` name rather than tying back to the
   *  original). */
  function renderFolder(key: string, label: string, items: any[],
                        open: boolean, currentModel: any,
                        italic: boolean) {
    const folder = document.createElement('div');
    folder.className = 'file-folder' + (open ? ' file-folder--open' : '');
    folder.dataset.key = key;

    const header = document.createElement('div');
    header.className = 'file-folder-header';
    const chev = document.createElement('span');
    chev.className = 'file-folder-chev';
    chev.textContent = open ? '▾' : '▸';  // ▾ / ▸
    chev.setAttribute('aria-hidden', 'true');
    const name = document.createElement('span');
    name.className = 'file-folder-name';
    name.textContent = label;
    const count = document.createElement('span');
    count.className = 'file-folder-count';
    count.textContent = String(items.length);
    header.appendChild(chev);
    header.appendChild(name);
    header.appendChild(count);
    header.addEventListener('click', function () {
      const state = readFolderState();
      const wasOpen = state[key] !== undefined ? !!state[key]
        : (key === 'examples'
           || (key === 'user' && items.length > 0));
      state[key] = !wasOpen;
      writeFolderState(state as Record<string, boolean>);
      // Cheapest correct redraw — re-render the whole tree, which
      // keeps the per-folder open/closed states + selection in sync.
      renderTree(currentModel);
    });
    folder.appendChild(header);

    if (open) {
      if (items.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'file-folder-empty';
        empty.textContent = '(empty)';
        folder.appendChild(empty);
      } else {
        const ul = document.createElement('ul');
        ul.className = 'file-list';
        for (let i = 0; i < items.length; i++) {
          const entry = items[i];
          const li = document.createElement('li');
          li.className = 'file-list-item';
          if (italic) li.classList.add('file-list-item--ephemeral');
          if (entry.path === currentModel) li.classList.add('selected');
          li.textContent = entry.title || basenameOf(entry.path);
          li.title = entry.path;
          li.dataset.path = entry.path;
          li.addEventListener('click', onTreeClick);
          li.addEventListener('contextmenu', (function (path: string) {
            return function (ev: any) {
              ev.preventDefault();
              showTreeContextMenu(ev, {
                path:   path,
                isUser: key === 'user',
              });
            };
          })(entry.path));
          // Surface the save + revert action icons when the row's
          // path has unsaved work — currently-loaded read-only or
          // user file with a dirty editor buffer.
          if (isPathDirty(entry.path)) attachEntryActions(li, entry.path);
          ul.appendChild(li);
        }
        folder.appendChild(ul);
      }
    }

    fileTree.appendChild(folder);
  }

  // -------------------- Tree context menu --------------------

  // Single floating menu element reused across right-clicks. Built
  // lazily on first show; closed on outside-click, Escape, or
  // pane scroll. Actions are computed per-entry from the same
  // metadata renderFolder already has.
  let _ctxMenu: HTMLElement | null = null;
  function getTreeContextMenu(): HTMLElement {
    if (_ctxMenu) return _ctxMenu;
    const m = document.createElement('div');
    m.id = 'tree-context-menu';
    m.style.display = 'none';
    document.body.appendChild(m);
    document.addEventListener('mousedown', function (ev: any) {
      if (!_ctxMenu || _ctxMenu.style.display === 'none') return;
      if (_ctxMenu.contains(ev.target)) return;
      hideTreeContextMenu();
    });
    document.addEventListener('keydown', function (ev: any) {
      if (ev.key === 'Escape') hideTreeContextMenu();
    });
    _ctxMenu = m;
    return m;
  }
  function hideTreeContextMenu() {
    if (_ctxMenu) _ctxMenu.style.display = 'none';
  }
  function showTreeContextMenu(ev: any, info: any) {
    const menu = getTreeContextMenu();
    menu.innerHTML = '';

    const items: Array<{ label: string; action: () => void }> = [];
    items.push({
      label: 'Download',
      action: function () { downloadFile(info.path); },
    });
    if (info.isUser) {
      items.push({
        label: 'Delete',
        action: function () {
          if (!window.confirm('Delete "' + info.path + '"? This cannot be undone.')) return;
          deleteUserFile(info.path, /* navigateTo */ null);
        },
      });
    }
    if (info.isEphemeral) {
      items.push({
        label: 'Discard',
        action: function () {
          if (!window.confirm('Discard "' + info.path + '"? It will be lost.')) return;
          if (window.FlatPPLWebEphemeral) {
            window.FlatPPLWebEphemeral.remove(info.path);
          }
          const cur = window.FlatPPLWebRouter.parseHash();
          if (cur.model === info.path) {
            const fallback = manifest && manifest.entries && manifest.entries[0]
              ? manifest.entries[0].path : null;
            if (fallback) window.FlatPPLWebRouter.navigateTo({ model: fallback });
          }
        },
      });
    }

    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const row = document.createElement('div');
      row.className = 'tree-ctx-item';
      row.textContent = it.label;
      row.addEventListener('click', function () {
        hideTreeContextMenu();
        try { it.action(); }
        catch (e) { console.error('[@flatppl/web] context-menu action failed:', e); }
      });
      menu.appendChild(row);
    }

    // Position at the mouse, clamped so the menu doesn't fall off
    // the right edge of the viewport.
    menu.style.display = 'block';
    const rect = menu.getBoundingClientRect();
    let x = ev.clientX;
    let y = ev.clientY;
    if (x + rect.width  > window.innerWidth)  x = window.innerWidth  - rect.width  - 4;
    if (y + rect.height > window.innerHeight) y = window.innerHeight - rect.height - 4;
    menu.style.left = x + 'px';
    menu.style.top  = y + 'px';
  }

  /** Trigger a browser download of `path`'s source. Resolves the
   *  text the same way the gallery normally does (ephemeral → user
   *  → network) so any in-memory or persisted override wins over
   *  the on-disk read-only original. */
  async function downloadFile(path: string) {
    if (!path) return;
    let text: string | null = null;
    try {
      const bundle = await window.FlatPPLWebResolver.resolveBundle(path);
      text = bundle && bundle.primarySource;
    } catch (e: any) {
      window.alert('Download failed: ' + (e && e.message ? e.message : String(e)));
      return;
    }
    if (text == null) return;
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = basenameOf(path);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Revoke after a tick so the browser has time to start the
    // download — some browsers (Safari) cancel the download if the
    // blob URL is revoked synchronously.
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  /** Ingest a list of files (from an `<input type="file">` change
   *  or a drag-drop on the file tree) into the user store. Only
   *  `.flatppl` files are accepted; everything else is silently
   *  skipped with a single combined toast at the end. A 1 MB
   *  per-file size cap protects localStorage's ~5 MB total quota.
   *  Collision-avoidance is delegated to userStore.pathForUpload
   *  so an existing `user/<basename>` doesn't get clobbered. The
   *  first accepted file becomes the new active model so the user
   *  sees their upload land. */
  async function ingestUploadedFiles(files: FileList | File[] | null) {
    if (!files) return;
    const userStore = window.FlatPPLWebUserStore;
    if (!userStore) return;
    const list = Array.from(files as any) as File[];
    const accepted: string[] = [];
    let skipped = 0;
    let oversized = 0;
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if (!f || !f.name.endsWith('.flatppl')) { skipped += 1; continue; }
      if (f.size > 1024 * 1024) { oversized += 1; continue; }
      try {
        // File.text() is widely supported (Chrome ≥ 76, Firefox ≥ 69,
        // Safari ≥ 14). For older browsers a FileReader fallback
        // would be needed; declining to add the polyfill until
        // a real user complains.
        const text = await f.text();
        const path = userStore.pathForUpload(f.name);
        userStore.save(path, text, { parent: null });
        accepted.push(path);
      } catch (e: any) {
        console.warn('[@flatppl/web] upload read failed for', f.name, e);
        skipped += 1;
      }
    }
    const notes: string[] = [];
    if (accepted.length > 0) notes.push('Uploaded ' + accepted.length + ' file' + (accepted.length > 1 ? 's' : ''));
    if (skipped > 0)         notes.push('skipped ' + skipped + ' non-.flatppl');
    if (oversized > 0)       notes.push('skipped ' + oversized + ' over 1 MB');
    if (notes.length > 0) showToast(notes.join('; ') + '.');
    if (accepted.length > 0) {
      window.FlatPPLWebRouter.navigateTo({ model: accepted[0] });
    }
  }

  /** Open the browser's file picker via a transient hidden
   *  `<input type="file">`. Click is dispatched synchronously
   *  inside the user-initiated event handler so the picker
   *  actually opens (browsers reject programmatic .click() on a
   *  file input outside a user gesture). */
  function onUploadClick() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.flatppl';
    input.multiple = true;
    input.style.display = 'none';
    document.body.appendChild(input);
    input.addEventListener('change', function () {
      try { ingestUploadedFiles(input.files); }
      finally {
        if (input.parentNode) input.parentNode.removeChild(input);
      }
    });
    input.click();
  }

  /** Wire drag-and-drop on the file tree. dragover sets dropEffect
   *  so the cursor shows a copy affordance; dragleave / drop clear
   *  the drag-target highlight. Multi-file drops are handled
   *  natively by FileList. */
  function installFileTreeDragDrop() {
    if (!fileTree) return;
    fileTree.addEventListener('dragenter', function (ev: any) {
      ev.preventDefault();
      fileTree.classList.add('file-tree--drop-target');
    });
    fileTree.addEventListener('dragover', function (ev: any) {
      ev.preventDefault();
      if (ev.dataTransfer) ev.dataTransfer.dropEffect = 'copy';
    });
    fileTree.addEventListener('dragleave', function (ev: any) {
      // Only clear when leaving the file-tree element itself, not
      // when crossing into a child. `relatedTarget` is the element
      // the pointer entered; if it's null (left the window) or not
      // a descendant, we're really leaving.
      if (!ev.relatedTarget || !fileTree.contains(ev.relatedTarget)) {
        fileTree.classList.remove('file-tree--drop-target');
      }
    });
    fileTree.addEventListener('drop', function (ev: any) {
      ev.preventDefault();
      fileTree.classList.remove('file-tree--drop-target');
      const dt = ev.dataTransfer;
      if (dt && dt.files) ingestUploadedFiles(dt.files);
    });
  }

  /** Minimal transient notification for upload/skip summaries.
   *  Single reusable div appended on first show; auto-fades after
   *  ~2.5 s. No queue — successive calls just overwrite the
   *  current message and reset the timer. */
  let _toastTimer: any = null;
  function showToast(msg: string) {
    let t = document.getElementById('gallery-toast');
    if (!t) {
      t = document.createElement('div');
      t.id = 'gallery-toast';
      document.body.appendChild(t);
    }
    t.textContent = msg;
    t.style.display = 'block';
    t.style.opacity = '1';
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(function () {
      if (!t) return;
      t.style.opacity = '0';
      _toastTimer = setTimeout(function () {
        if (t) t.style.display = 'none';
      }, 300);
    }, 2500);
  }

  /** Delete a user-store entry and optionally navigate the gallery
   *  to a fallback path (e.g. the original after a "Reset to
   *  original"). The store's subscribe fires automatically, which
   *  redraws the sidebar. */
  function deleteUserFile(path: string, navigateTo: string | null) {
    const userStore = window.FlatPPLWebUserStore;
    if (!userStore) return;
    userStore.remove(path);
    if (navigateTo) {
      window.FlatPPLWebRouter.navigateTo({ model: navigateTo });
    } else {
      // No fallback — if the current model was just deleted, route
      // to whatever the next manifest entry is so the gallery
      // shows something. The user-store subscriber redraws the
      // tree separately.
      const cur = window.FlatPPLWebRouter.parseHash();
      if (cur.model === path) {
        const fallback = manifest && manifest.entries && manifest.entries[0]
          ? manifest.entries[0].path : null;
        if (fallback) window.FlatPPLWebRouter.navigateTo({ model: fallback });
      }
    }
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
    updateHeaderButtons();

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
          { pushHistory: true });
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
      const bundle = await window.FlatPPLWebResolver.resolveBundle(state.model);
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

    // Source-pane header buttons for the currently-loaded file:
    //   Download — always visible when a file is loaded; same
    //              action as the tree context-menu's Download.
    //   Delete   — visible only for user/ files; same action as
    //              the tree context-menu's Delete.
    // Both buttons read the current model from the router on
    // click rather than holding a reference, so they stay correct
    // across navigation. Visibility is refreshed by
    // updateHeaderButtons on URL changes and user-store events.
    const downloadBtn = document.getElementById('download-btn');
    const deleteBtn   = document.getElementById('delete-btn');
    if (downloadBtn) downloadBtn.addEventListener('click', onDownloadClick);
    if (deleteBtn)   deleteBtn.addEventListener('click',   onDeleteClick);

    // Tab-close / reload / external-navigation guard. Standard
    // beforeunload pattern: setting returnValue to a non-empty
    // value triggers the browser's native "Leave site? Changes
    // you made may not be saved" prompt. Modern browsers ignore
    // the actual string and show a generic message; the gate is
    // what matters. Only fires when something would actually be
    // lost (per hasUnsavedWork), so unmodified sessions reload
    // silently as before.
    window.addEventListener('beforeunload', function (ev: BeforeUnloadEvent) {
      if (!hasUnsavedWork()) return;
      ev.preventDefault();
      // Required for the prompt to fire in older Chromium-based
      // browsers; ignored by current Chrome / Firefox / Safari
      // but harmless to set.
      ev.returnValue = '';
    });

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
      // Debounced viewer refresh — every keystroke triggering a
      // full DAG rebuild would be wasteful. 250ms collapses bursts
      // into one render after the user pauses typing.
      const debouncedViewerUpdate = debounce(function (text: any, target: any) {
        if (viewer) viewer.update(text, target || null);
      }, 250);
      sourceEditor = window.FlatPPLWebEditor.mountEditor(editorContainer, {
        initialSource: '',
        initialReadOnly: true,
        onChange: function (text: any) {
          // Persistence requires an explicit save (Ctrl/Cmd-S or
          // the save button) — no implicit auto-save into the user
          // store, even for files that already live there. The
          // dirty marker reflects the buffer-vs-persisted gap.
          //
          // The single in-keystroke side-effect we DO keep is the
          // ephemeral-store update for `new/...` paths: ephemeral
          // is the in-memory session buffer, not persistent
          // storage, so keeping it in sync just makes
          // navigate-away-and-back non-destructive within the
          // session.
          if (!editEnabled) return;
          const cur = window.FlatPPLWebRouter.parseHash();
          if (cur.model && cur.model.indexOf('new/') === 0
              && window.FlatPPLWebEphemeral) {
            window.FlatPPLWebEphemeral.update(cur.model, text);
          }
          updateDirtyMarker();
          debouncedViewerUpdate(text, cur.target);
        },
        onSave: handleSave,
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
    const uploadBtn  = document.getElementById('upload-btn');
    if ((window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      if (newFileBtn) {
        newFileBtn.hidden = false;
        newFileBtn.addEventListener('click', onNewFileClick);
      }
      if (uploadBtn) {
        uploadBtn.hidden = false;
        uploadBtn.addEventListener('click', onUploadClick);
      }
      installFileTreeDragDrop();
    }
    if (window.FlatPPLWebEphemeral && window.FlatPPLWebEphemeral.subscribe) {
      window.FlatPPLWebEphemeral.subscribe(function () {
        const cur = window.FlatPPLWebRouter.parseHash();
        renderTree(cur.model);
        updateHeaderButtons();
      });
    }
    if (window.FlatPPLWebUserStore && window.FlatPPLWebUserStore.subscribe) {
      window.FlatPPLWebUserStore.subscribe(function () {
        const cur = window.FlatPPLWebRouter.parseHash();
        renderTree(cur.model);
        updateHeaderButtons();
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
