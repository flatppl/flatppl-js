// @flatppl/web — hash-based router.
//
// Encodes the current model selection (and, later, focused binding
// inside that model) in `location.hash`. Hash routing rather than
// query-string routing because static hosts like GitHub Pages don't
// rewrite paths server-side, and hash navigation never triggers a
// page reload.
//
// Hash format:
//   #model=path/to/foo.flatppl
//   #model=path/to/foo.flatppl&target=binding_name      (later step)
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
    var raw = (globalScope.location && globalScope.location.hash) || '';
    if (raw.charAt(0) === '#') raw = raw.slice(1);
    var out = { model: null, target: null };
    if (!raw) return out;
    var parts = raw.split('&');
    for (var i = 0; i < parts.length; i++) {
      var eq = parts[i].indexOf('=');
      if (eq <= 0) continue;
      var key = parts[i].slice(0, eq);
      var val;
      try { val = decodeURIComponent(parts[i].slice(eq + 1)); }
      catch (_) { val = parts[i].slice(eq + 1); }
      if (key === 'model')  out.model = val;
      if (key === 'target') out.target = val;
    }
    return out;
  }

  /** Serialize { model, target } back to a hash string (no leading #). */
  function serialize(state) {
    var parts: string[] = [];
    if (state.model)  parts.push('model=' + encodeURIComponent(state.model));
    if (state.target) parts.push('target=' + encodeURIComponent(state.target));
    return parts.join('&');
  }

  /**
   * Set the URL hash without reloading. If the new hash matches the
   * current one, no event fires (browsers coalesce). Use this from
   * file-tree clicks rather than mutating location.hash directly so
   * future router behaviour (e.g. push-vs-replace history entries)
   * has one place to live.
   */
  function navigateTo(state) {
    var next = serialize(state);
    var current = (globalScope.location.hash || '').replace(/^#/, '');
    if (next === current) return;
    globalScope.location.hash = next;
  }

  var changeListeners: Array<(state: any) => void> = [];

  /** Register a callback fired on every hash navigation. */
  function onChange(fn) {
    changeListeners.push(fn);
  }

  // De-dupe consecutive identical states so a single in-place
  // hash assignment doesn't trigger a duplicate callback.
  var lastSerialized: string | null = null;
  function emitIfChanged() {
    var state = parseHash();
    var ser = serialize(state);
    if (ser === lastSerialized) return;
    lastSerialized = ser;
    for (var i = 0; i < changeListeners.length; i++) {
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
