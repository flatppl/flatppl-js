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
  let layout:       any = null;
  let fileFilter: HTMLInputElement | null = null;
  // Edited files keep their source and baseline across route changes.
  const drafts = new Map<string, { source: string; baseline: string | null }>();

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

  // Fallback for hosts without an editor and the active draft's saved baseline.
  let lastRenderedSource: string | null = null;
  let lastSavedSource: string | null = null;

  // The currently loaded model path, so we can short-circuit when a
  // navigation event only changes the focus target. Critical in
  // playground mode: rewriting the editor's content on every target-
  // change would wipe user edits and reset the cursor to the start
  // of the file.
  let lastModel: string | null = null;
  // Every route change supersedes earlier dependency reads, including a
  // target-only change or a return to the fallback source.
  let navigationSeq = 0;

  /** Set source silently, restoring a retained draft's baseline when supplied. */
  function showSource(text: any, label?: any, baseline: string | null = text) {
    if (sourceHeader) sourceHeader.textContent = label || 'Source';
    if (!sourceEditor) return;
    // Pick the editor language pack from the active model's path so a
    // .md / .json buffer is highlighted (and folded) on its own terms,
    // not as FlatPPL. The router's model path is authoritative for the
    // file type; `label` is the path on real loads but can be a transient
    // status string ("# Loading …", "error"), so prefer the router.
    if (typeof sourceEditor.setLanguage === 'function') {
      const S = window.FlatPPLWebSurfaces;
      const cur = window.FlatPPLWebRouter
        ? window.FlatPPLWebRouter.parseHash() : { model: null };
      const p = (cur && cur.model) || label;
      sourceEditor.setLanguage(S ? S.typeForPath(p) : 'flatppl');
    }
    sourceEditor.setSource(text);
    lastRenderedSource = text;
    lastSavedSource = baseline;
    updateDirtyMarker();
  }

  /** Drafts and failed saves remain dirty even while another file is open. */
  function isPathDirty(path: string): boolean {
    if (!path) return false;
    const eph = window.FlatPPLWebEphemeral;
    if (eph && eph.has && eph.has(path)) return true;
    const store = window.FlatPPLWebUserStore;
    return drafts.has(path) || !!(store && store.has(path) && !store.isPersisted(path));
  }

  /** True iff anything in the gallery currently holds unsaved
   *  work. Drives the `beforeunload` warning — the browser only
   *  prompts on tab close / reload / external navigation when
   *  this returns true. */
  function hasUnsavedWork(): boolean {
    const eph = window.FlatPPLWebEphemeral;
    if (eph && eph.list && eph.list().some((entry: { text: string }) => entry.text.length > 0)) return true;
    const store = window.FlatPPLWebUserStore;
    return drafts.size > 0 || !!(store && store.hasUnpersisted());
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
    const isUserFile = !!(userStore && cur.model
      && cur.model.indexOf(userStore.USER_PREFIX) === 0
      && userStore.has(cur.model));
    const downloadBtn = document.getElementById('download-btn');
    const deleteBtn   = document.getElementById('delete-btn');
    if (downloadBtn) downloadBtn.hidden = !cur.model;
    if (deleteBtn) deleteBtn.hidden = !isUserFile;
    // The whole-module graph view only makes sense for FlatPPL — hide it for
    // Markdown / hs3 / pyhf (and anything without a graph surface) so the
    // button doesn't promise a view those file types can't deliver.
    const showModuleBtn = document.getElementById('show-module-btn');
    if (showModuleBtn) {
      const type = window.FlatPPLWebSurfaces
        ? window.FlatPPLWebSurfaces.typeForPath(cur.model) : 'flatppl';
      showModuleBtn.hidden = !cur.model || type !== 'flatppl';
    }
    // Editor tools button (wrench) — shown when the active file has any
    // applicable tool (Convert to FlatPPL for pyhf / hs3, Export math as …
    // for a model).
    const toolsBtn = document.getElementById('tools-btn');
    if (toolsBtn) {
      toolsBtn.hidden = toolsForFile(cur.model).length === 0;
    }
    // The header filename is click/F2-renamable for user/ files only.
    if (sourceHeader) {
      sourceHeader.classList.toggle('renamable', isUserFile);
      sourceHeader.title = isUserFile ? 'Rename (F2)' : '';
    }
  }

  function onDownloadClick() {
    const cur = window.FlatPPLWebRouter.parseHash();
    if (cur && cur.model) return downloadFile(cur.model);
  }

  function onDeleteClick() {
    const cur = window.FlatPPLWebRouter.parseHash();
    if (!cur || !cur.model) return;
    if (!window.confirm('Delete "' + cur.model + '"? This cannot be undone.')) return;
    deleteUserFile(cur.model, /* navigateTo */ null);
  }

  /** Rename a user/ file to a new basename. The extension/type is kept
   *  (renaming user/x.flatppl → user/x.md would silently swap the surface),
   *  collisions are refused with a toast, and the active view follows the
   *  file if it was the one renamed. Entry point for the tree context
   *  menu, F2, and the inline-editable header. */
  function renameUserFile(oldPath: any, newBase: any) {
    const userStore = window.FlatPPLWebUserStore;
    const S = window.FlatPPLWebSurfaces;
    if (!userStore || !oldPath) return;
    newBase = (newBase == null ? '' : String(newBase)).trim();
    if (!newBase) return;
    const newPath = userStore.USER_PREFIX + newBase;
    if (newPath === oldPath) return;
    if (S && S.typeForPath(newPath) !== S.typeForPath(oldPath)) {
      showToast('Keep the same file extension when renaming.');
      return;
    }
    if (userStore.has(newPath)) {
      showToast('"' + newBase + '" already exists.');
      return;
    }
    if (!userStore.rename(oldPath, newPath)) {
      showToast('Rename failed.');
      return;
    }
    const draft = drafts.get(oldPath);
    if (draft) { drafts.set(newPath, draft); drafts.delete(oldPath); }
    const cur = window.FlatPPLWebRouter.parseHash();
    if (cur && cur.model === oldPath) {
      window.FlatPPLWebRouter.navigateTo({ model: newPath, target: cur.target || null });
    }
  }

  /** Turn the source-pane header's filename into an inline editor for the
   *  active user/ file (triggered by F2 or a click on the header). Enter
   *  commits, Escape / blur cancels. No-op for read-only / non-user files.
   *  The `user/` prefix shows as static text; only the basename is edited. */
  function startHeaderRename() {
    if (!sourceHeader) return;
    const userStore = window.FlatPPLWebUserStore;
    const cur = window.FlatPPLWebRouter.parseHash();
    const model = cur && cur.model;
    if (!userStore || !model
        || model.indexOf(userStore.USER_PREFIX) !== 0 || !userStore.has(model)) return;
    if (sourceHeader.querySelector('input')) return;  // already editing
    const base = basenameOf(model);
    const prevText = sourceHeader.textContent;
    sourceHeader.textContent = '';
    const prefix = document.createElement('span');
    prefix.className = 'rename-prefix';
    prefix.textContent = userStore.USER_PREFIX;       // "user/" shown static
    const input = document.createElement('input');
    input.type = 'text';
    input.setAttribute('aria-label', 'File name');
    input.className = 'rename-input';
    input.value = base;
    input.spellcheck = false;
    sourceHeader.appendChild(prefix);
    sourceHeader.appendChild(input);
    input.focus();
    const dot = base.lastIndexOf('.');
    input.setSelectionRange(0, dot > 0 ? dot : base.length);   // select the stem
    let done = false;
    function finish(commit: boolean) {
      if (done) return;
      done = true;
      const val = input.value;
      // Restore the static header; a successful rename re-navigates and
      // re-labels it anyway, so this only matters on cancel.
      if (sourceHeader) sourceHeader.textContent = prevText;
      if (commit) renameUserFile(model, val);
    }
    input.addEventListener('keydown', function (e: any) {
      if (e.key === 'Enter') { e.preventDefault(); finish(true); }
      else if (e.key === 'Escape') { e.preventDefault(); finish(false); }
    });
    input.addEventListener('blur', function () { finish(false); });
  }

  // --- The flatppl-wasm-api artifact (lazy wasm) --------------------------
  //
  // One artifact, three uses: Convert (pyhf / HS3 → FlatPPL), the viewer's
  // math pane (`render_math`, loaded by the viewer itself through
  // `__FLATPPL_CONFIG__.wasmApiUrl`) and the "Export math as …" tools
  // (`export_math`). It is provisioned into vendor/ by the build ONLY when
  // available (sibling wasm-pack build, or a CI-staged artifact). Absent ⇒
  // convertAvailable stays false and every affordance that needs it stays
  // hidden, so the gallery degrades cleanly with no broken button.
  let convertAvailable = false;
  let _wasmApi: any = null;
  let _wasmApiLoading: Promise<any> | null = null;
  // Glue path in a variable so tsc treats the import as runtime-only (no
  // compile-time resolution of the build-provisioned, possibly-absent glue).
  const _wasmGluePath = './vendor/flatppl_wasm_api.js';

  /** The initialised wasm API module (its exports: convert, render_math,
   *  export_math), imported once on first use. */
  async function loadWasmApi() {
    if (_wasmApi) return _wasmApi;
    // One init at a time: two first calls (Convert and an export, say) must
    // share the instantiation, not each run it.
    if (!_wasmApiLoading) {
      _wasmApiLoading = (async () => {
        // Dynamic ESM import of the wasm-pack `--target web` glue, resolved at
        // runtime against the page (app.js is a classic script; esbuild's
        // transform leaves import() untouched). default() runs the wasm init.
        const mod: any = await import(_wasmGluePath);
        if (mod && typeof mod.default === 'function') await mod.default();
        _wasmApi = mod;
        return mod;
      })().catch((e) => { _wasmApiLoading = null; throw e; });
    }
    return _wasmApiLoading;
  }

  async function loadWasmConvert() {
    return (await loadWasmApi()).convert;
  }

  // --- Export math as a document (via flatppl-wasm-api `export_math`) ------
  //
  // The model's mathematics as a whole document — HTML (MathML), Markdown,
  // LaTeX or Typst — saved into the user area and opened; design:
  // flatppl-dev/math-view-design.md §4 "Whole documents". The catalogue,
  // the request and the file name come from the viewer's contract layer
  // (FlatPPLViewer.mathExport) so this host and VS Code derive them alike.
  // The source is the editor buffer when the file is the one being edited
  // (what the math pane renders), else the stored text; the dependency
  // bundle is the same map the viewer gets.

  /** Text of `path` as the user sees it: the live editor buffer when
   *  `path` is the active file (edits included), null otherwise. */
  function editorBufferFor(path: string): string | null {
    const cur = window.FlatPPLWebRouter.parseHash();
    if (sourceEditor && cur && cur.model === path && lastModel === path) return sourceEditor.getSource();
    return drafts.get(path)?.source ?? null;
  }

  async function exportMathDocument(srcPath: string, format: string) {
    const userStore = window.FlatPPLWebUserStore;
    const mathExport = window.FlatPPLViewer && window.FlatPPLViewer.mathExport;
    if (!userStore || !mathExport) return;
    try {
      // The buffer as edited walks its own load_module deps; the stored text
      // tells whether those edits are unsaved.
      const buffer = editorBufferFor(srcPath);
      const bundle = await window.FlatPPLWebResolver.resolveBundle(srcPath, buffer);
      if (!bundle || typeof bundle.primarySource !== 'string') {
        showToast('Could not read ' + srcPath + '.');
        return;
      }
      const api = await loadWasmApi();
      if (typeof api.export_math !== 'function') {
        showToast('Math export is not available in this build.');
        return;
      }
      const request = mathExport.buildRequest({
        source: bundle.primarySource,
        path: srcPath,
        bundleSources: bundle.sources,
        document: format,
      });
      const out = api.export_math(JSON.stringify(request));   // throws on a malformed module
      // A derived file with a fixed name (`<stem>.<ext>`, as VS Code writes
      // beside the source): a repeat export replaces it after asking, it
      // never piles up numbered copies.
      const dest = 'user/' + mathExport.fileName(srcPath, format);
      if (userStore.has(dest) && !window.confirm('Replace "' + dest + '"?')) return;
      const persisted = userStore.save(dest, out, { parent: null });
      // Opening the result replaces the editor content. With unsaved edits in
      // the buffer that would throw them away, so then the file only appears
      // under User for the user to open.
      const dirty = buffer !== null && buffer !== (await window.FlatPPLWebResolver.resolveBundle(srcPath)).primarySource;
      let note = persisted ? '' : ' (for this session only: browser storage is full)';
      if (dirty) {
        showToast('Exported → ' + dest + note + '. Open it under User; your unsaved edits stay here.');
      } else {
        showToast('Exported → ' + dest + note + '.');
        window.FlatPPLWebRouter.navigateTo({ model: dest });
      }
    } catch (e: any) {
      console.error('[@flatppl/web] math export failed:', e);
      showToast('Export failed: ' + ((e && e.message) || e));
    }
  }

  /** Destination basename for a converted file: strip the import extension
   *  and add `.flatppl` (uniquified into user/ by the caller). */
  function convertedFlatpplName(srcPath: any): string {
    return basenameOf(srcPath).replace(/\.(pyhf|hs3)\.json$/i, '') + '.flatppl';
  }

  /** Convert a pyhf / HS3 file to FlatPPL and drop the result into the user
   *  area (always writable), then open it. The source may be a user upload
   *  or a read-only corpus entry — the output uniformly lands in user/. */
  async function convertToFlatppl(srcPath: any) {
    const S = window.FlatPPLWebSurfaces;
    const userStore = window.FlatPPLWebUserStore;
    const type = S ? S.typeForPath(srcPath) : 'unknown';
    const from = type === 'pyhf' ? 'pyhf' : (type === 'hs3' ? 'hs3' : null);
    if (!from || !userStore) return;
    try {
      const bundle = await window.FlatPPLWebResolver.resolveBundle(srcPath, editorBufferFor(srcPath));
      const input = bundle && bundle.primarySource;
      if (typeof input !== 'string') { showToast('Could not read ' + srcPath + '.'); return; }
      const convert = await loadWasmConvert();
      const out = convert(input, from, 'flatppl');   // throws on importer error
      const dest = userStore.pathForUpload(convertedFlatpplName(srcPath));
      const persisted = userStore.save(dest, out, { parent: null });
      showToast('Converted → ' + dest + (persisted ? '.' : '. Save failed — draft kept in this tab.'));
      window.FlatPPLWebRouter.navigateTo({ model: dest });
    } catch (e: any) {
      console.error('[@flatppl/web] convert failed:', e);
      showToast('Convert failed: ' + ((e && e.message) || e));
    }
  }

  function onEntrySaveClick(path: string) {
    const text = editorBufferFor(path)
      ?? window.FlatPPLWebEphemeral?.get(path)
      ?? window.FlatPPLWebUserStore?.getSource(path);
    if (text != null) saveFile(path, text);
  }

  function onEntryRevertClick(path: string) {
    if (!window.confirm('Discard unsaved changes to "' + basenameOf(path) + '"?')) return;
    const cur = window.FlatPPLWebRouter.parseHash();
    const eph = window.FlatPPLWebEphemeral;
    const store = window.FlatPPLWebUserStore;
    drafts.delete(path);
    if (eph?.has(path)) eph.remove(path);
    else if (store?.has(path)) store.revert(path);
    renderTree(cur.model);
    if (cur.model !== path) return;
    lastModel = null;
    if (path.startsWith('new/') || (path.startsWith('user/') && !store?.has(path))) {
      const fallback = manifest?.entries?.[0]?.path || null;
      window.FlatPPLWebRouter.navigateTo({ model: fallback });
    } else {
      applyState(cur);
    }
  }

  // Was the current model dirty on the previous keystroke? Used to
  // avoid redrawing the whole file tree on every keystroke — the
  // sidebar's save/revert icons only need to flip when the dirty
  // status changes, not when a single character ticks the
  // already-dirty buffer further.
  let _wasDirty = false;

  /** Keep save state visible even after leaving edit mode or changing files. */
  function updateDirtyMarker() {
    const cur = window.FlatPPLWebRouter.parseHash();
    const store = window.FlatPPLWebUserStore;
    const dirty = isPathDirty(cur.model);
    const failed = !!(cur.model && store?.has(cur.model) && !store.isPersisted(cur.model));
    if (sourceHeader) {
      const label = (sourceHeader.textContent || '').replace(/^\* /, '');
      sourceHeader.textContent = (dirty ? '* ' : '') + label;
    }
    const status = document.getElementById('source-status');
    if (status) status.textContent = failed ? 'Save failed — kept in this tab'
      : dirty ? 'Unsaved draft' : store?.has(cur.model) ? 'Saved in this browser' : '';
    const error = document.getElementById('save-error');
    if (error) error.hidden = !failed;
    const message = document.getElementById('save-error-message');
    if (message && failed) message.textContent = 'Browser storage is full or unavailable. Download your draft or retry saving before closing this tab.';
    const saveBtn = document.getElementById('save-btn');
    if (saveBtn) saveBtn.setAttribute('data-dirty', String(dirty));
    const welcome = document.getElementById('welcome-actions');
    if (welcome) {
      welcome.hidden = !sourceEditor || sourceEditor.getSource().trim().length > 0;
      const surface = document.getElementById('surface-root');
      if (surface) surface.hidden = !welcome.hidden;
    }
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
    sourceEditor.setReadOnly(!editEnabled || !lastModel);
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

  /** Name a saved copy, preserving its type and confirming an overwrite. */
  function promptForUserPath(origPath: string): string | null {
    const userStore = window.FlatPPLWebUserStore;
    if (!userStore) return null;
    const suggested = suggestForkName(origPath);
    const raw = window.prompt(
      'Save your edits as (the original "' + origPath
      + '" stays unchanged):',
      basenameOf(suggested));
    if (raw == null) return null;
    let typed = raw.trim();
    if (!typed) return null;
    if (typed.indexOf(userStore.USER_PREFIX) !== 0) {
      typed = userStore.USER_PREFIX + typed;
    }
    const S = window.FlatPPLWebSurfaces;
    if (S && S.typeForPath(typed) !== S.typeForPath(origPath)) {
      window.alert('Keep the same file extension when saving a copy.');
      return null;
    }
    if (userStore.has(typed)) {
      if (!window.confirm('"' + typed + '" already exists. Overwrite?')) {
        return null;
      }
    }
    return typed;
  }

  /** Save explicitly. Failed writes remain in the store's in-memory fallback. */
  function saveFile(path: string, text: string) {
    const store = window.FlatPPLWebUserStore;
    if (!store || !(window.__FLATPPL_CONFIG__ || {}).allowEdit) return;
    const cur = window.FlatPPLWebRouter.parseHash();
    const fromTree = fileTree?.contains(document.activeElement);
    let target = path;
    if (path.startsWith('new/')) target = store.pathForUpload(basenameOf(path));
    else if (!path.startsWith(store.USER_PREFIX)) {
      const chosen = promptForUserPath(path);
      if (chosen == null) return;
      target = chosen;
    }
    const persisted = store.has(target)
      ? store.updateSource(target, text) : store.save(target, text, { parent: null });
    drafts.delete(path);
    drafts.delete(target);
    if (window.FlatPPLWebEphemeral?.has(path)) window.FlatPPLWebEphemeral.remove(path);
    if (cur.model === path) {
      if (target !== path) window.FlatPPLWebRouter.navigateTo({ model: target, target: cur.target });
      else {
        lastSavedSource = text;
        updateDirtyMarker();
      }
    } else if (cur.model === target) {
      lastModel = null;
      applyState(cur);
    }
    renderTree(cur.model === path ? target : cur.model);
    revealFile(target);
    if (fromTree) {
      const link = Array.from(fileTree.querySelectorAll('.file-list-link'))
        .find((el: any) => el.dataset.path === target) as HTMLElement | undefined;
      link?.focus({ preventScroll: true });
    }
    if (!persisted && cur.model !== path) showToast('Save failed for ' + basenameOf(target) + '. Draft kept in this tab.');
  }

  function handleSave() {
    const cur = window.FlatPPLWebRouter.parseHash();
    if (!sourceEditor || !cur.model || lastModel !== cur.model) return;
    saveFile(cur.model, sourceEditor.getSource());
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

  function matchesFilter(entry: any): boolean {
    const query = (fileFilter?.value || '').trim().toLowerCase();
    return !query || [entry.path, entry.title, entry.description]
      .join(' ').toLowerCase().includes(query);
  }

  function fileRow(entry: any, currentModel: string, kind: string): HTMLElement {
    const li = document.createElement('li');
    li.className = 'file-list-item fp-nav-item';
    li.dataset.path = entry.path;
    if (kind === 'new') li.classList.add('file-list-item--ephemeral');
    if (entry.path === currentModel) li.classList.add('selected', 'is-active');
    const link = document.createElement('a');
    link.className = 'file-list-link';
    link.href = '#model=' + encodeURIComponent(entry.path);
    link.dataset.path = entry.path;
    link.dataset.action = 'open';
    link.title = entry.path;
    if (entry.path === currentModel) link.setAttribute('aria-current', 'page');
    const label = document.createElement('span');
    label.textContent = entry.title || basenameOf(entry.path);
    link.appendChild(label);
    if (entry.description) {
      const description = document.createElement('small');
      description.className = 'file-list-description';
      description.textContent = entry.description;
      link.appendChild(description);
    }
    link.addEventListener('click', onTreeClick);
    link.addEventListener('contextmenu', function (ev: MouseEvent) {
      ev.preventDefault();
      showTreeContextMenu(ev, { path: entry.path, isUser: kind === 'user', isEphemeral: kind === 'new' });
    });
    li.appendChild(link);
    if (isPathDirty(entry.path)) {
      li.classList.add('file-list-item--dirty');
      for (const [action, symbol, handler] of [
        ['save', '💾', onEntrySaveClick], ['discard', '↺', onEntryRevertClick],
      ] as const) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'file-list-action file-list-action-' + (action === 'discard' ? 'revert' : action);
        button.dataset.path = entry.path;
        button.dataset.action = action;
        button.textContent = symbol;
        button.setAttribute('aria-label', (action === 'save' ? 'Save ' : 'Discard changes to ') + basenameOf(entry.path));
        button.title = button.getAttribute('aria-label')!;
        button.addEventListener('click', function () { handler(entry.path); });
        li.appendChild(button);
      }
    }
    return li;
  }

  /** Redraw without losing the current file-list focus or scroll position. */
  function renderTree(currentModel: any) {
    if (!fileTree) return;
    const active = document.activeElement as HTMLElement | null;
    const focus = active && fileTree.contains(active) ? { ...active.dataset } : null;
    const focusedFolder = active?.closest('.file-folder')?.getAttribute('data-key');
    const scrollTop = fileTree.scrollTop;
    fileTree.innerHTML = '';
    const ephEntries = window.FlatPPLWebEphemeral?.list() || [];
    const userEntries = window.FlatPPLWebUserStore?.list() || [];
    const buckets: Record<string, any[]> = { examples: [], 'test-cases': [] };
    for (const entry of manifest?.entries || []) {
      if (typeof entry.path !== 'string') continue;
      const key = window.FlatPPLFileTypes?.bucketKeyForPath(entry.path) || 'examples';
      (buckets[key] || buckets.examples).push(entry);
    }
    if (ephEntries.length) renderFolder('new', 'Unsaved', ephEntries, currentModel);
    renderFolder('examples', 'Examples', buckets.examples, currentModel);
    if (!(window.__FLATPPL_CONFIG__ || {}).hideTestExamples) {
      renderFolder('test-cases', 'Test cases', buckets['test-cases'], currentModel);
    }
    renderFolder('user', 'Saved in this browser', userEntries, currentModel);
    if (fileFilter?.value ? !fileTree.querySelector('.file-list-link')
      : !ephEntries.length && !userEntries.length && !manifest?.entries?.length) {
      const empty = document.createElement('p');
      empty.className = 'pane-placeholder';
      empty.textContent = fileFilter?.value ? 'No models match your search.' : 'Open a file or create a model to begin.';
      fileTree.appendChild(empty);
    }
    if (focus) {
      const targets = Array.from(fileTree.querySelectorAll('a, button')) as HTMLElement[];
      const target = targets.find((el) =>
        focus.folder ? el.dataset.folder === focus.folder
          : el.dataset.path === focus.path && el.dataset.action === focus.action)
        || targets.find((el) => el.dataset.path === focus.path && el.dataset.action === 'open')
        || targets.find((el) => el.dataset.folder === focusedFolder);
      target?.focus({ preventScroll: true });
    }
    fileTree.scrollTop = scrollTop;
  }

  function renderFolder(key: string, label: string, entries: any[], currentModel: string) {
    const items = entries.filter(matchesFilter);
    const filtering = !!fileFilter?.value.trim();
    if (filtering && !items.length) return;
    const st = readFolderState();
    const defaultOpen = key === 'new'
      || (window.FlatPPLFileTypes?.defaultFolderOpen(key, items.length) ?? key === 'examples');
    const open = filtering || (st[key] ?? defaultOpen);
    const folder = document.createElement('div');
    folder.className = 'file-folder' + (open ? ' file-folder--open' : '');
    folder.dataset.key = key;
    const header = document.createElement('button');
    header.type = 'button';
    header.className = 'file-folder-header fp-nav-item';
    header.dataset.folder = key;
    header.setAttribute('aria-expanded', String(open));
    header.setAttribute('aria-controls', 'folder-' + key);
    const chev = document.createElement('span');
    chev.className = 'file-folder-chev';
    chev.textContent = open ? '▾' : '▸';
    chev.setAttribute('aria-hidden', 'true');
    const name = document.createElement('span');
    name.className = 'file-folder-name';
    name.textContent = label;
    const count = document.createElement('span');
    count.className = 'file-folder-count';
    count.textContent = String(items.length);
    header.append(chev, name, count);
    header.addEventListener('click', function () {
      const state = readFolderState();
      state[key] = !open;
      writeFolderState(state as Record<string, boolean>);
      renderTree(currentModel);
    });
    folder.appendChild(header);
    const list = document.createElement('ul');
    list.id = 'folder-' + key;
    list.className = 'file-list';
    list.hidden = !open;
    if (open) {
      for (const entry of items) list.appendChild(fileRow(entry, currentModel, key));
    }
    folder.appendChild(list);
    fileTree.appendChild(folder);
  }

  /** Reveal only on navigation/save. Keystrokes leave file-list scroll alone. */
  function revealFile(path: string) {
    if (!fileTree || !path) return;
    const entry = manifest?.entries?.find((entry: any) => entry.path === path) || { path };
    if (fileFilter?.value && !matchesFilter(entry)) fileFilter.value = '';
    const key = path.startsWith('new/') ? 'new'
      : path.startsWith('user/') ? 'user' : (window.FlatPPLFileTypes?.bucketKeyForPath(path) || 'examples');
    const state = readFolderState();
    state[key] = true;
    writeFolderState(state as Record<string, boolean>);
    renderTree(window.FlatPPLWebRouter.parseHash().model);
    const link = Array.from(fileTree.querySelectorAll('.file-list-link'))
      .find((el: any) => el.dataset.path === path) as HTMLElement | undefined;
    link?.scrollIntoView({ block: 'nearest' });
  }

  // -------------------- Tree context menu --------------------

  // Single floating menu element reused across right-clicks. Built
  // lazily on first show; closed on outside-click, Escape, or
  // pane scroll. Actions are computed per-entry from the same
  // metadata renderFolder already has.
  // Element to restore focus to when the floating menu closes (the
  // right-clicked tree row, or the tools button). Set on open.
  let _ctxMenuTrigger: HTMLElement | null = null;

  function menuItemEls(menu: HTMLElement): HTMLElement[] {
    return Array.prototype.slice.call(
      menu.querySelectorAll('.tree-ctx-item')) as HTMLElement[];
  }
  function focusMenuItemAt(menu: HTMLElement, idx: number): void {
    const rows = menuItemEls(menu);
    if (rows.length === 0) return;
    const n = ((idx % rows.length) + rows.length) % rows.length;  // wrap
    rows.forEach(function (r, i) { r.tabIndex = i === n ? 0 : -1; });
    rows[n].focus();
  }
  let _ctxMenu: HTMLElement | null = null;
  function getTreeContextMenu(): HTMLElement {
    if (_ctxMenu) return _ctxMenu;
    const m = document.createElement('div');
    m.id = 'tree-context-menu';
    m.setAttribute('role', 'menu');
    m.style.display = 'none';
    document.body.appendChild(m);
    document.addEventListener('mousedown', function (ev: any) {
      if (!_ctxMenu || _ctxMenu.style.display === 'none') return;
      if (_ctxMenu.contains(ev.target)) return;
      const toolsBtn = document.getElementById('tools-btn');
      if (toolsBtn && toolsBtn.contains(ev.target)) return;  // let the button toggle
      hideTreeContextMenu();
    });
    document.addEventListener('keydown', function (ev: any) {
      if (ev.key === 'Escape') hideTreeContextMenu();
    });
    m.addEventListener('keydown', function (ev: any) {
      const rows = menuItemEls(m);
      if (rows.length === 0) return;
      const active = document.activeElement as HTMLElement;
      const cur = rows.indexOf(active);
      if (ev.key === 'ArrowDown') {
        ev.preventDefault();
        focusMenuItemAt(m, cur < 0 ? 0 : cur + 1);
      } else if (ev.key === 'ArrowUp') {
        ev.preventDefault();
        focusMenuItemAt(m, cur < 0 ? rows.length - 1 : cur - 1);
      } else if (ev.key === 'Home') {
        ev.preventDefault();
        focusMenuItemAt(m, 0);
      } else if (ev.key === 'End') {
        ev.preventDefault();
        focusMenuItemAt(m, rows.length - 1);
      } else if (ev.key === 'Enter' || ev.key === ' ') {
        ev.preventDefault();
        if (active && rows.indexOf(active) >= 0) active.click();
      } else if (ev.key === 'Tab') {
        hideTreeContextMenu();  // let focus proceed naturally
      }
      // Escape is handled by the existing document-level keydown.
    });
    _ctxMenu = m;
    return m;
  }
  function hideTreeContextMenu() {
    if (_ctxMenu) _ctxMenu.style.display = 'none';
    // aria-expanded lives only on the tools button; a closed menu means the
    // tools popup is collapsed regardless of which trigger opened the menu.
    const toolsBtn = document.getElementById('tools-btn');
    if (toolsBtn) toolsBtn.setAttribute('aria-expanded', 'false');
    if (_ctxMenuTrigger) {
      const t = _ctxMenuTrigger;
      _ctxMenuTrigger = null;
      // Only pull focus back if it's still inside the (now-hidden) menu —
      // otherwise an outside-click that moved focus elsewhere keeps it.
      if (!_ctxMenu || _ctxMenu.contains(document.activeElement)
          || document.activeElement === document.body) {
        try { t.focus(); } catch (_) {}
      }
    }
  }
  /** Tools applicable to `path` — appended LAST in menus so the common /
   *  destructive actions stay on top. Grows as more conversions/tools land;
   *  the single source the context menu and the editor tools dropdown share. */
  function toolsForFile(path: any): Array<{ label: string; action: () => void }> {
    const tools: Array<{ label: string; action: () => void }> = [];
    const t = window.FlatPPLWebSurfaces
      ? window.FlatPPLWebSurfaces.typeForPath(path) : 'unknown';
    if (convertAvailable && (t === 'pyhf' || t === 'hs3')) {
      tools.push({
        label: 'Convert to FlatPPL',
        action: function () { convertToFlatppl(path); },
      });
    }
    // "Export math as …" for a model: one entry per document format of the
    // viewer's catalogue (HTML, Markdown, LaTeX, Typst). The same artifact
    // as Convert, so the same build-time availability gate.
    const mathExport = window.FlatPPLViewer && window.FlatPPLViewer.mathExport;
    if (path && convertAvailable && mathExport && t === 'flatppl') {
      mathExport.formats.forEach(function (f: { document: string; label: string }) {
        tools.push({
          label: 'Export math as ' + f.label,
          action: function () { exportMathDocument(path, f.document); },
        });
      });
    }
    return tools;
  }

  /** Render `items` as rows into the floating menu element; each row hides the
   *  menu and runs its action on click. Shared by the tree context menu and
   *  the editor tools dropdown. */
  function renderMenuItems(menu: HTMLElement,
                           items: Array<{ label: string; action: () => void }>) {
    menu.innerHTML = '';
    for (let i = 0; i < items.length; i++) {
      const it = items[i];
      const row = document.createElement('div');
      row.className = 'tree-ctx-item';
      row.setAttribute('role', 'menuitem');
      row.tabIndex = -1;  // roving tabindex; opener focuses the first row
      row.textContent = it.label;
      row.addEventListener('click', function () {
        hideTreeContextMenu();
        try { it.action(); }
        catch (e) { console.error('[@flatppl/web] menu action failed:', e); }
      });
      menu.appendChild(row);
    }
  }

  function showTreeContextMenu(ev: any, info: any) {
    const menu = getTreeContextMenu();
    _ctxMenuTrigger = (ev && ev.currentTarget instanceof HTMLElement)
      ? ev.currentTarget : null;

    const items: Array<{ label: string; action: () => void }> = [];
    items.push({
      label: 'Download',
      action: function () { downloadFile(info.path); },
    });
    if (info.isUser) {
      items.push({
        label: 'Rename',
        action: function () {
          const nb = window.prompt('Rename to:', basenameOf(info.path));
          if (nb != null) renameUserFile(info.path, nb);
        },
      });
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
    // Tools (Convert to FlatPPL, …) come LAST.
    const tools = toolsForFile(info.path);
    for (let i = 0; i < tools.length; i++) items.push(tools[i]);

    renderMenuItems(menu, items);

    // Position at the mouse, clamped so the menu doesn't fall off the edge.
    menu.style.display = 'block';
    const rect = menu.getBoundingClientRect();
    let x = ev.clientX;
    let y = ev.clientY;
    if (x + rect.width  > window.innerWidth)  x = window.innerWidth  - rect.width  - 4;
    if (y + rect.height > window.innerHeight) y = window.innerHeight - rect.height - 4;
    menu.style.left = x + 'px';
    menu.style.top  = y + 'px';
    focusMenuItemAt(menu, 0);
  }

  /** Open the editor's tools dropdown (the wrench button) anchored below
   *  `btn`, listing the tools applicable to the active file. No-op if none. */
  function showToolsMenu(btn: HTMLElement) {
    const cur = window.FlatPPLWebRouter.parseHash();
    const tools = toolsForFile(cur && cur.model);
    if (tools.length === 0) return;
    const menu = getTreeContextMenu();
    renderMenuItems(menu, tools);
    _ctxMenuTrigger = btn;
    menu.style.display = 'block';
    btn.setAttribute('aria-expanded', 'true');
    const br = btn.getBoundingClientRect();
    const mr = menu.getBoundingClientRect();
    let x = br.left;
    let y = br.bottom + 2;
    if (x + mr.width  > window.innerWidth)  x = window.innerWidth - mr.width - 4;
    if (y + mr.height > window.innerHeight) y = br.top - mr.height - 2;  // flip above
    menu.style.left = x + 'px';
    menu.style.top  = y + 'px';
    focusMenuItemAt(menu, 0);
  }

  /** Trigger a browser download of `path`'s source. Resolves the
   *  text the same way the gallery normally does (ephemeral → user
   *  → network) so any in-memory or persisted override wins over
   *  the on-disk read-only original. */
  async function downloadFile(path: string) {
    if (!path) return;
    let text: string | null = editorBufferFor(path);
    try {
      if (text == null) {
        const bundle = await window.FlatPPLWebResolver.resolveBundle(path);
        text = bundle && bundle.primarySource;
      }
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
   *  or a drag-drop on the file tree) into the user store. Files of a
   *  recognised type (`.flatppl`, `.md`, `.hs3.json`, `.pyhf.json` — per
   *  surfaces.typeForPath; the last two render a placeholder for now) are
   *  accepted; everything else is silently skipped with a single combined
   *  toast at the end. A 1 MB
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
    let unsaved = 0;
    for (let i = 0; i < list.length; i++) {
      const f = list[i];
      if (!f) { skipped += 1; continue; }
      // Accept any file the gallery has a surface for. Single source of
      // truth is surfaces.typeForPath ('unknown' = no surface registered),
      // so upload acceptance tracks the surface registry automatically as
      // new types are added — no separate extension list to keep in sync.
      if (window.FlatPPLWebSurfaces.typeForPath(f.name) === 'unknown') { skipped += 1; continue; }
      if (f.size > 1024 * 1024) { oversized += 1; continue; }
      try {
        // File.text() is widely supported (Chrome ≥ 76, Firefox ≥ 69,
        // Safari ≥ 14). For older browsers a FileReader fallback
        // would be needed; declining to add the polyfill until
        // a real user complains.
        const text = await f.text();
        const path = userStore.pathForUpload(f.name);
        if (!userStore.save(path, text, { parent: null })) unsaved += 1;
        accepted.push(path);
      } catch (e: any) {
        console.warn('[@flatppl/web] upload read failed for', f.name, e);
        skipped += 1;
      }
    }
    const notes: string[] = [];
    if (accepted.length > 0) notes.push('Uploaded ' + accepted.length + ' file' + (accepted.length > 1 ? 's' : ''));
    if (skipped > 0)         notes.push('skipped ' + skipped + ' unsupported');
    if (oversized > 0)       notes.push('skipped ' + oversized + ' over 1 MB');
    if (unsaved > 0)         notes.push(unsaved + ' kept only in this tab because saving failed');
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
    // Everything the gallery has a surface for: the model types plus the
    // document types the math export writes (drag-drop accepts the same set
    // through typeForPath; the picker needs the list spelled out).
    const ft = window.FlatPPLFileTypes;
    input.accept = ft
      ? ft.MODEL_EXTENSIONS.concat(ft.DOCUMENT_EXTENSIONS).join(',')
      : '.flatppl,.md,.markdown,.hs3.json,.pyhf.json,.html,.htm,.tex,.typ';
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
      // Announce transient outcomes (upload/convert summaries) to screen
      // readers. polite = wait for a pause; atomic = read the whole message.
      t.setAttribute('role', 'status');
      t.setAttribute('aria-live', 'polite');
      t.setAttribute('aria-atomic', 'true');
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
    drafts.delete(path);
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
    return '# ' + basenameOf(path) + '\n';
  }

  /** Sanitize a user-supplied filename into a path the rest of the
      gallery accepts: strip surrounding whitespace, prepend `new/`
      if missing, append a default extension when none of the known
      model extensions (file-types.mjs MODEL_EXTENSIONS) are present. */
  function normalizeEphemeralPath(raw: any) {
    let s = String(raw || '').trim();
    if (!s) return null;
    if (!s.startsWith('new/')) s = 'new/' + s;
    const FT = window.FlatPPLFileTypes;
    const exts: string[] = (FT && FT.MODEL_EXTENSIONS) || ['.flatppl'];
    const lower = s.toLowerCase();
    const hasKnownExt = exts.some(function (ext: string) { return lower.endsWith(ext); });
    if (!hasKnownExt) s = s + '.flatppl';
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
      'New file name. Save keeps it in this browser:', basenameOf(defaultPath));
    if (raw == null) return;
    const path = normalizeEphemeralPath(raw);
    if (!path) return;
    // Collision check across both the manifest and the ephemeral store.
    const collidesWithManifest = manifest && manifest.entries
      && manifest.entries.some(function (e: any) { return e.path === path; });
    if (eph.has(path) || collidesWithManifest) {
      window.alert('A file with this name already exists: ' + basenameOf(path));
      return;
    }
    eph.add(path, starterSourceFor(path));
    if (!editEnabled && (window.__FLATPPL_CONFIG__ || {}).allowEdit) {
      setEditMode(true);
    }
    window.FlatPPLWebRouter.navigateTo({ model: path });
    layout?.closeFiles();
    layout?.showPane('source');
  }

  function onTreeClick(ev: any) {
    if (ev.metaKey || ev.ctrlKey || ev.shiftKey || ev.altKey) return;
    ev.preventDefault();
    const path = ev.currentTarget.dataset.path;
    if (path) {
      window.FlatPPLWebRouter.navigateTo({ model: path });
      layout?.closeFiles();
    }
  }

  async function applyState(state: any) {
    const seq = ++navigationSeq;
    // Repaint tree highlight even before the fetch completes.
    renderTree(state.model);
    updateHeaderButtons();
    if (state.model !== lastModel) revealFile(state.model);

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

    if (state.model && state.model === lastModel) {
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
      showSource(FALLBACK_SOURCE, 'untitled');
      if (viewer) viewer.update(FALLBACK_SOURCE, state.target || null);
      document.title = 'FlatPPL';
      lastModel = null;
      return;
    }
    // The loading placeholder is no longer the last model's source. A
    // return to that model must reload it rather than reuse this buffer.
    lastModel = null;
    sourceEditor?.setReadOnly?.(true);
    showSource('# Loading ' + state.model + ' …', state.model);
    try {
      const draft = drafts.get(state.model);
      const bundle = await window.FlatPPLWebResolver.resolveBundle(state.model, draft?.source);
      if (seq !== navigationSeq) return;
      showSource(bundle.primarySource, state.model, draft ? draft.baseline : bundle.primarySource);
      sourceEditor?.setReadOnly?.(!editEnabled);
      if (viewer) viewer.update(bundle.primarySource, state.target || null,
        { bundleSources: bundle.sources, path: state.model });
      document.title = 'FlatPPL: ' + state.model + (state.target ? ' / ' + state.target : '');
      lastModel = state.model;
    } catch (err) {
      if (seq !== navigationSeq) return;
      console.error('[@flatppl/web] resolveBundle failed:', err);
      showError(state.model, err);
      sourceEditor?.setReadOnly?.(true);
      document.title = 'FlatPPL: ' + state.model + ' (error)';
      // Leave lastModel null so a successful retry triggers the
      // full fetch + source-rewrite path.
    }
  }

  async function boot() {
    sourceHeader = document.getElementById('source-header');
    fileTree     = document.getElementById('file-tree');
    titleEl      = document.getElementById('app-title');
    fileFilter   = document.getElementById('file-filter') as HTMLInputElement | null;
    fileFilter?.addEventListener('input', function () {
      renderTree(window.FlatPPLWebRouter.parseHash().model);
    });
    fileFilter?.addEventListener('keydown', function (ev: KeyboardEvent) {
      if (ev.key === 'Enter') {
        ev.preventDefault();
        fileTree?.querySelector('.file-list-link')?.click();
      } else if (ev.key === 'Escape' && fileFilter?.value) {
        ev.preventDefault();
        ev.stopPropagation();
        fileFilter.value = '';
        renderTree(window.FlatPPLWebRouter.parseHash().model);
      }
    });
    document.addEventListener('keydown', function (ev: KeyboardEvent) {
      if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === 'k') {
        ev.preventDefault();
        layout?.revealFiles();
        fileFilter?.focus();
        fileFilter?.select();
      }
    });
    document.getElementById('retry-save-btn')?.addEventListener('click', handleSave);
    document.getElementById('save-error-download-btn')?.addEventListener('click', onDownloadClick);
    document.getElementById('try-model-btn')?.addEventListener('click', function () {
      window.FlatPPLWebRouter.navigateTo({ model: 'examples/linear-regression.flatppl' });
    });
    document.getElementById('browse-examples-btn')?.addEventListener('click', function () {
      layout?.revealFiles();
      fileFilter?.focus();
    });
    const welcomeUpload = document.getElementById('welcome-upload-btn');
    if (welcomeUpload) {
      welcomeUpload.hidden = !(window.__FLATPPL_CONFIG__ || {}).allowEdit;
      welcomeUpload.addEventListener('click', onUploadClick);
    }

    // Inline rename of the active user/ file: click the header filename, or
    // press F2 anywhere outside a form field. Both no-op for read-only /
    // non-user files (gated inside startHeaderRename).
    if (sourceHeader) {
      sourceHeader.addEventListener('click', function () { startHeaderRename(); });
    }
    document.addEventListener('keydown', function (ev: any) {
      if (ev.key !== 'F2') return;
      const tag = ev.target && ev.target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;
      ev.preventDefault();
      startHeaderRename();
    });

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

    // Editor tools button (wrench): toggles a dropdown of tools for the active
    // file (currently Convert to FlatPPL). Visibility is set by
    // updateHeaderButtons; the menu is reused from the tree context menu.
    const toolsBtn = document.getElementById('tools-btn');
    if (toolsBtn) {
      toolsBtn.addEventListener('click', function () {
        if (_ctxMenu && _ctxMenu.style.display === 'block') { hideTreeContextMenu(); return; }
        showToolsMenu(toolsBtn);
      });
    }

    // Convert availability is decided at BUILD TIME — build-flags.js sets
    // __FLATPPL_CONFIG__.convert (true when the build provisioned the wasm
    // artifact; false for a FLATPPL_CONVERT=off build). No runtime probe: the
    // flag and the artifact are produced together, so the UI never promises a
    // command the build didn't ship.
    convertAvailable = !!((window.__FLATPPL_CONFIG__ || {}).convert);

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
    if (!window.FlatPPLWebSurfaces) missing.push('surfaces');
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
      layout = window.FlatPPLWebLayout.install({
        toggleButton: document.getElementById('toggle-files'),
      });
    }

    // Retarget matching shared-footer links when this deployment provides a
    // local page through its site overlay.
    if (window.FlatPPLWebFooter) {
      window.FlatPPLWebFooter.install(window.__FLATPPL_CONFIG__ || {});
    }

    // Mount the source editor eagerly. The editor is the only
    // source surface now (no read-only <pre>) — it lives in view-
    // only mode by default and the edit toggle flips it to writable.
    const editorContainer = document.getElementById('source-editor');
    if (editorContainer && window.FlatPPLWebEditor) {
      // Debounced viewer refresh — every keystroke triggering a
      // full DAG rebuild would be wasteful. 250ms collapses bursts
      // into one render after the user pauses typing.
      const debouncedViewerUpdate = debounce(function (text: string, target: string, path: string, seq: number) {
        if (viewer && seq === navigationSeq && path === lastModel) viewer.update(text, target || null);
      }, 250);
      sourceEditor = window.FlatPPLWebEditor.mountEditor(editorContainer, {
        initialSource: '',
        initialReadOnly: true,
        onChange: function (text: any) {
          // Retain edits in memory. Only an explicit Save writes browser storage.
          if (!editEnabled) return;
          const cur = window.FlatPPLWebRouter.parseHash();
          if (!cur.model || lastModel !== cur.model) return;
          if (cur.model && cur.model.indexOf('new/') === 0
              && window.FlatPPLWebEphemeral) {
            window.FlatPPLWebEphemeral.update(cur.model, text);
          } else if (text !== lastSavedSource) {
            drafts.set(cur.model, { source: text, baseline: lastSavedSource });
          } else {
            drafts.delete(cur.model);
          }
          updateDirtyMarker();
          debouncedViewerUpdate(text, cur.target, cur.model, navigationSeq);
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

    const surfaceRoot = document.getElementById('surface-root');

    // Web host adapter for the viewer. The viewer calls these
    // host-side hooks for IDE-only concerns it can't perform itself
    // (cross-pane source navigation, panel-title updates, persisted
    // UI state). Each method is optional from the viewer's
    // standpoint; we implement only what makes sense for a web
    // gallery.
    //
    //   revealSourceLine(line, name, opts)
    //                          — Ctrl+click on a DAG node or a math-pane
    //                            row fires this with the binding's
    //                            source line: we move the caret there,
    //                            scroll and flash-highlight it, and the
    //                            caret move routes through onNavigate
    //                            like a click in the code would. The
    //                            math pane's PLAIN click passes
    //                            preserveFocus: the caret follows
    //                            silently (no onNavigate, no focus), so
    //                            the pane and the code stay in step
    //                            without re-targeting the DAG. The line
    //                            is zero-indexed, matching the tokenizer
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
      revealSourceLine: function (line: any, _name: any, opts: any) {
        if (!opts?.preserveFocus) layout?.showPane('source');
        if (sourceEditor && typeof sourceEditor.revealLine === 'function') {
          sourceEditor.revealLine(line, { silent: !!(opts && opts.preserveFocus) });
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
      /** Open a loaded module's file (spec §04 cross-module navigation).
          Fired by double-clicking a `load_module` node (whole module) or a
          member / member-alias node (focused on that member). `path` is the
          resolved module path (already the gallery-root-relative key the
          router uses); `member`, when present, is the binding inside the
          module to focus the DAG on. The router change also swaps the source
          editor to the module file (source-sync, spec §04). */
      openModule: function (path: any, member?: any) {
        if (!path) return;
        window.FlatPPLWebRouter.navigateTo({ model: path, target: member || null });
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

    // Right-pane surface dispatch (surfaces.ts): the FlatPPL viewer is
    // ONE registered surface, not the default. The controller derives the
    // active file type from the router (the source of truth for "what's
    // shown"), disposes-and-remounts on a cross-type switch (e.g.
    // .flatppl ↔ .md), and keeps a same-type switch a cheap update().
    // Every `viewer.update(source, target, opts)` call site stays
    // unchanged — the controller preserves that signature.
    viewer = window.FlatPPLWebSurfaces.createPaneController(
      surfaceRoot,
      { host: webHost, config: window.__FLATPPL_CONFIG__ || {} },
      function () {
        const cur = window.FlatPPLWebRouter.parseHash();
        return (cur && cur.model) || null;
      }
    );

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
    // open an empty ephemeral buffer, because a bare visit is a place
    // to start typing, and routing to a manifest entry ties the landing
    // page to whichever example happens to ship first.
    const initial = window.FlatPPLWebRouter.parseHash();
    const ephBoot = window.FlatPPLWebEphemeral;
    if (!initial.model && ephBoot) {
      const scratch = ephBoot.nextUntitled('.flatppl');
      ephBoot.add(scratch, '');
      window.FlatPPLWebRouter.navigateTo({ model: scratch });
      layout?.showPane('viewer');
    } else {
      if (initial.model?.startsWith('new/') && ephBoot && !ephBoot.has(initial.model)) {
        ephBoot.add(initial.model, '');
      }
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
