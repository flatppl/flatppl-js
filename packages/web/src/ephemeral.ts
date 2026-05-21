// @flatppl/web — in-memory store for ephemeral (unsaved) source files.
//
// "Ephemeral" here means session-only: a file the user created in the
// gallery via the "+ New file" button, with no backing on disk. The
// store holds the source text in memory; reloading the page drops
// every ephemeral file. The resolver consults this store before
// trying to fetch anything from the network, so an ephemeral path
// (e.g. `new/untitled.flatppl`) resolves to its in-memory text
// without ever 404'ing against the server.
//
// The store is variant-agnostic: paths can end in `.flatppl` /
// `.flatppy` / `.flatppj` and the variant detection downstream
// (variantForPath) picks the right surface syntax. Default file
// names increment a counter so consecutive "+ New file" clicks
// produce distinct paths.
//
// Lives on globalThis as window.FlatPPLWebEphemeral.

'use strict';

(function (globalScope: any) {
  // path → source-text. Map preserves insertion order, which the
  // app uses to render the "Unsaved" section in the order entries
  // were created.
  var store = new Map();

  // Listeners notified when an entry is added / removed / updated.
  // The app subscribes once at boot to re-render the file tree.
  var listeners = new Set<() => void>();

  function emit() {
    listeners.forEach(function (fn) {
      try { fn(); } catch (e) { console.error('[FlatPPLWebEphemeral] listener threw:', e); }
    });
  }

  /** True iff `path` was added to the ephemeral store this session. */
  function has(path: any) { return store.has(path); }

  /** Get the source text for `path`, or undefined when not stored. */
  function get(path: any) { return store.get(path); }

  /** Snapshot of every ephemeral entry, in insertion order. Returns
      an array of `{ path, text }` so the caller can render them
      without mutating the underlying map. */
  function list() {
    var out: any[] = [];
    store.forEach(function (text: any, path: any) { out.push({ path: path, text: text }); });
    return out;
  }

  /** Stash a new ephemeral entry. Overwrites any previous entry at
      the same path silently — the caller is responsible for
      collision-avoidance (use `nextUntitled` for default names). */
  function add(path: any, text: any) {
    store.set(path, text);
    emit();
  }

  /** Update the source text for an existing entry. No-op when the
      path isn't in the store (the editor's onChange fires for every
      keystroke; we don't want it to silently re-introduce a deleted
      file). */
  function update(path: any, text: any) {
    if (!store.has(path)) return;
    store.set(path, text);
    // Don't emit — content changes shouldn't redraw the tree.
  }

  /** Drop an entry. */
  function remove(path: any) {
    if (!store.delete(path)) return;
    emit();
  }

  /** Subscribe to add/remove events. Returns an unsubscribe function. */
  function subscribe(fn: any) {
    listeners.add(fn);
    return function () { listeners.delete(fn); };
  }

  /** Pick a default path for the next "+ New file" click. We use
      the `new/` prefix so ephemeral entries cluster together
      visually in any path-sorted display AND don't collide with
      real `demo/`, `examples/` paths. The numeric suffix
      increments past any path already in the store. */
  function nextUntitled(extension: any) {
    var ext = extension || '.flatppl';
    var n = 1;
    for (;;) {
      var candidate = 'new/untitled-' + n + ext;
      if (!store.has(candidate)) return candidate;
      n += 1;
    }
  }

  globalScope.FlatPPLWebEphemeral = {
    has: has,
    get: get,
    list: list,
    add: add,
    update: update,
    remove: remove,
    subscribe: subscribe,
    nextUntitled: nextUntitled,
  };
})(typeof window !== 'undefined' ? window : globalThis);
