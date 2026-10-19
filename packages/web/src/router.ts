// @flatppl/web — hash-based router.
//
// Encodes the current model selection (and, later, focused binding
// inside that model) in `location.hash`. Hash routing rather than
// query-string routing because static hosts like GitHub Pages don't
// rewrite paths server-side, and hash navigation never triggers a
// page reload.
//
// Hash format:
//   #/files/path/to/foo.flatppl?target=binding_name
// Legacy #model= paths still load. Unsaved bundles retain their content token
// or hosted archive URL while navigating within the same bundle. Share links
// use #share=<token>, QR codes #~<digits> (see share-links.md).
//
// `model` is the path the resolver will fetch (relative to the
// document base, so e.g. `demo/x.flatppl` resolves under the
// gallery's deployed root). `target` is optional and, once wired,
// focuses the viewer on a specific binding.
//
// The router emits a "change" callback when the user navigates
// (clicks a file in the tree, edits the URL bar, or presses
// back/forward). Coalesces consecutive identical states so callers
// don't re-render redundantly.
//
// Lives on globalThis as window.FlatPPLWebRouter.

'use strict';

(function (globalScope: any) {
  /** Parse the current location.hash into { model, target }. */
  function parseHash() {
    let raw = (globalScope.location && globalScope.location.hash) || '';
    if (raw.charAt(0) === '#') raw = raw.slice(1);
    const out: { model: string | null; target: string | null; share?: string; archive?: string; error?: string } = { model: null, target: null };
    // Apply the generation budget on input too, before splitting or decoding.
    if (raw.length > 64 * 1024) return { ...out, error: 'This link is too long. Open the ZIP bundle instead.' };
    if (!raw) return out;
    const seen = new Set<string>();
    // A QR share token starts with its marker `~` and stands alone.
    if (raw.startsWith('~')) {
      const end = raw.indexOf('&');
      out.share = raw.slice(0, end < 0 ? undefined : end);
      seen.add('share');
      raw = end < 0 ? '' : raw.slice(end + 1);
    }
    if (raw.startsWith('/files/')) {
      const end = raw.indexOf('?');
      try { out.model = decodeURIComponent(raw.slice(7, end < 0 ? undefined : end)); }
      catch (_) { return { ...out, error: 'Invalid file path in link.' }; }
      seen.add('model');
      raw = end < 0 ? '' : raw.slice(end + 1);
    }
    const parts = raw.split('&');
    for (let i = 0; i < parts.length; i++) {
      const eq = parts[i].indexOf('=');
      if (eq <= 0) continue;
      const key = parts[i].slice(0, eq);
      if (!['model', 'target', 'share', 'archive'].includes(key)) continue;
      if (seen.has(key)) return { model: null, target: null, error: 'Duplicate parameter in link: ' + key };
      seen.add(key);
      var val;
      try { val = decodeURIComponent(parts[i].slice(eq + 1).replace(/\+/g, ' ')); }
      catch (_) { return { model: null, target: null, error: 'Invalid encoding in link.' }; }
      if (key === 'model')  out.model = val;
      if (key === 'target') out.target = val;
      if (key === 'share') out.share = val;
      if (key === 'archive') out.archive = val;
    }
    if (out.share && out.archive) return { model: null, target: null, error: 'A link cannot contain both share and archive.' };
    return out;
  }

  /** Serialize { model, target } back to a hash string (no leading #). */
  function serialize(state: any) {
    const parts: string[] = [];
    if (state.target) parts.push('target=' + encodeURIComponent(state.target));
    if (state.share) parts.push('share=' + encodeURIComponent(state.share));
    if (state.archive) parts.push('archive=' + encodeURIComponent(state.archive));
    const path = state.model ? '/files/' + state.model.split('/').map(encodeURIComponent).join('/') : '';
    return path + (parts.length ? (path ? '?' : '') + parts.join('&') : '');
  }

  /**
   * Set the URL hash without reloading. If the new hash matches the
   * current one, no event fires (browsers coalesce). Use this from
   * file-tree clicks rather than mutating location.hash directly so
   * future router behaviour (e.g. push-vs-replace history entries)
   * has one place to live.
   */
  function navigateTo(state: any) {
    const bundles = globalScope.FlatPPLWebBundles;
    const folder = bundles?.folder(state.model);
    if (folder) {
      if (bundles.isSaved(state.model)) {
        state = { ...state, share: null, archive: null };
      } else {
        const current = parseHash();
        if (bundles.folder(current.model) === folder) state = { ...state,
          share: state.share === undefined ? current.share : state.share,
          archive: state.archive === undefined ? current.archive : state.archive,
        };
      }
    }
    const next = serialize(state);
    const current = (globalScope.location.hash || '').replace(/^#/, '');
    if (next === current) return;
    globalScope.location.hash = next;
  }

  const changeListeners: Array<(state: any) => void> = [];

  /** Register a callback fired on every hash navigation. */
  function onChange(fn: any) {
    changeListeners.push(fn);
  }

  // De-dupe consecutive identical states so a single in-place
  // hash assignment doesn't trigger a duplicate callback.
  let lastSerialized: string | null = null;
  function emitIfChanged() {
    const state = parseHash();
    const ser = state.error || serialize(state);
    if (ser === lastSerialized) return;
    lastSerialized = ser;
    for (let i = 0; i < changeListeners.length; i++) {
      try { changeListeners[i](state); }
      catch (e) { console.error('[@flatppl/web] router listener error:', e); }
    }
  }

  if (typeof globalScope.addEventListener === 'function') {
    globalScope.addEventListener('hashchange', emitIfChanged);
  }

  globalScope.FlatPPLWebRouter = {
    parseHash: parseHash,
    serialize: serialize,
    navigateTo: navigateTo,
    onChange: onChange,
    /** Fire listeners once with the initial state. Called from app boot
        after listeners are registered, so the initial render happens
        through the same code path as subsequent navigation. */
    emitInitial: function () { lastSerialized = null; emitIfChanged(); },
  };
})(typeof window !== 'undefined' ? window : globalThis);
