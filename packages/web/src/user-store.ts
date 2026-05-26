// @flatppl/web — persistent store for user files.
//
// Companion to FlatPPLWebEphemeral: where the ephemeral store keeps
// session-only buffers (in-memory, lost on reload), this store
// holds FlatPPL files the user has explicitly saved — either by
// editing a read-only example/test-case under a fresh name, or by
// creating a new file and saving it, or by uploading a file. All
// entries live under the conceptual `user/` folder in the sidebar
// and are independent of any read-only origin. The on-disk schema
// keeps a nullable `parent` field for forward-compatibility but
// no current caller writes a non-null value.
//
// Backing storage is `localStorage`. Each entry is one key/value
// pair; an index key lists every entry path in insertion order so
// the sidebar can render them stably. Schema:
//
//   flatppl-web:user-files         → JSON array of paths
//                                    ['user/foo.flatppl', 'user/bar.flatppl', ...]
//   flatppl-web:user-file:<path>   → JSON {
//                                      source:     string,
//                                      parent:     string | null,  // forked-from path or null
//                                      modifiedAt: ISO string,
//                                      version:    1
//                                    }
//
// The `version` field reserves room for future schema migrations
// without touching every key on read.
//
// localStorage can fail in two ways we care about:
//   1. Private-browsing modes that throw on write (Safari ≤ 11,
//      some embedded WebViews).
//   2. Quota exceeded (5–10 MB depending on browser; a hard cap
//      we won't usually hit since source files are KB-scale).
// Every read and write is wrapped in try/catch so a failure
// degrades gracefully — the store stays usable for the rest of
// the session via an in-memory fallback, and the user just sees
// their changes vanish on reload.
//
// Lives on globalThis as window.FlatPPLWebUserStore.

'use strict';

(function (globalScope: any) {
  const STORAGE_VERSION = 1;
  const INDEX_KEY        = 'flatppl-web:user-files';
  const ENTRY_KEY_PREFIX = 'flatppl-web:user-file:';
  /** Prefix that marks a path as belonging to the user folder.
   *  Paths in this store start with `user/`. The URL hash and the
   *  router round-trip the full path, so callers usually deal with
   *  the prefixed form throughout. */
  const USER_PREFIX = 'user/';

  type Entry = {
    source: string;
    parent: string | null;
    modifiedAt: string;
  };

  // In-memory mirror of localStorage state. Two roles:
  //   - Read fast path (parsing JSON on every render would be wasteful).
  //   - Fallback for environments where localStorage is unavailable
  //     or throws — the store still works for the session.
  const entries  = new Map<string, Entry>();
  let   pathOrder: string[] = [];
  const listeners = new Set<() => void>();

  function emit() {
    listeners.forEach(function (fn) {
      try { fn(); }
      catch (e) { console.error('[FlatPPLWebUserStore] listener threw:', e); }
    });
  }

  function lsAvailable(): boolean {
    try {
      return typeof globalScope.localStorage !== 'undefined'
        && globalScope.localStorage !== null;
    } catch (_) { return false; }
  }

  function lsGetItem(k: string): string | null {
    try { return lsAvailable() ? globalScope.localStorage.getItem(k) : null; }
    catch (_) { return null; }
  }

  function lsSetItem(k: string, v: string): boolean {
    if (!lsAvailable()) return false;
    try { globalScope.localStorage.setItem(k, v); return true; }
    catch (_) { return false; }
  }

  function lsRemoveItem(k: string) {
    if (!lsAvailable()) return;
    try { globalScope.localStorage.removeItem(k); }
    catch (_) {}
  }

  function readIndex(): string[] {
    const raw = lsGetItem(INDEX_KEY);
    if (!raw) return [];
    try {
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter(function (x: any) { return typeof x === 'string'; });
    } catch (_) { return []; }
  }

  function writeIndex(arr: string[]): boolean {
    return lsSetItem(INDEX_KEY, JSON.stringify(arr));
  }

  function readEntry(path: string): Entry | null {
    const raw = lsGetItem(ENTRY_KEY_PREFIX + path);
    if (!raw) return null;
    try {
      const obj = JSON.parse(raw);
      if (!obj || typeof obj.source !== 'string') return null;
      return {
        source:     obj.source,
        parent:     typeof obj.parent === 'string' ? obj.parent : null,
        modifiedAt: typeof obj.modifiedAt === 'string' ? obj.modifiedAt : new Date(0).toISOString(),
      };
    } catch (_) { return null; }
  }

  function writeEntry(path: string, entry: Entry): boolean {
    return lsSetItem(ENTRY_KEY_PREFIX + path, JSON.stringify({
      source:     entry.source,
      parent:     entry.parent,
      modifiedAt: entry.modifiedAt,
      version:    STORAGE_VERSION,
    }));
  }

  // Boot: replay every entry from localStorage into the in-memory
  // mirror. Path order is taken from the index key; entries whose
  // body is missing or unparseable get dropped silently (the index
  // is repaired by the next save).
  function loadFromStorage() {
    entries.clear();
    pathOrder = [];
    const ix = readIndex();
    const repaired: string[] = [];
    for (let i = 0; i < ix.length; i++) {
      const p = ix[i];
      const e = readEntry(p);
      if (e) {
        entries.set(p, e);
        pathOrder.push(p);
        repaired.push(p);
      }
    }
    if (repaired.length !== ix.length) writeIndex(repaired);
  }

  /** True iff `path` is in the user store. */
  function has(path: string): boolean { return entries.has(path); }

  /** Full entry record, or null. Use `getSource` when you only
   *  need the text. */
  function get(path: string): Entry | null {
    return entries.has(path) ? Object.assign({}, entries.get(path)!) : null;
  }

  /** Source text of `path`, or null. Resolver-facing convenience. */
  function getSource(path: string): string | null {
    const e = entries.get(path);
    return e ? e.source : null;
  }

  /** Snapshot of every entry, in insertion (= sidebar render) order. */
  function list() {
    return pathOrder.map(function (p) {
      const e = entries.get(p)!;
      return { path: p, source: e.source, parent: e.parent, modifiedAt: e.modifiedAt };
    });
  }

  /** Persist `path` ← `source`. `opts.parent` records the read-only
   *  path this was forked from, for the sidebar's "from examples/X"
   *  affordance and the "Reset to original" action. Pass `null` for
   *  fresh user-created files. */
  function save(path: string, source: string, opts?: { parent?: string | null }) {
    if (path.indexOf(USER_PREFIX) !== 0) {
      throw new Error('FlatPPLWebUserStore.save: path must start with "user/", got: ' + path);
    }
    const parent = opts && typeof opts.parent === 'string' ? opts.parent : null;
    const entry: Entry = {
      source: source,
      parent: parent,
      modifiedAt: new Date().toISOString(),
    };
    const isNew = !entries.has(path);
    entries.set(path, entry);
    if (isNew) pathOrder.push(path);
    writeEntry(path, entry);
    if (isNew) writeIndex(pathOrder);
    emit();
  }

  /** Update only the source text (modifiedAt bumps; parent
   *  preserved). Cheaper-feeling alias of `save` for auto-save
   *  paths where the parent doesn't change. */
  function updateSource(path: string, source: string) {
    const existing = entries.get(path);
    if (!existing) return;
    if (existing.source === source) return;
    const entry: Entry = {
      source: source,
      parent: existing.parent,
      modifiedAt: new Date().toISOString(),
    };
    entries.set(path, entry);
    writeEntry(path, entry);
    // No emit() — text edits shouldn't redraw the tree. Sidebar
    // listeners only care about add/remove.
  }

  /** Drop `path`. The original read-only file (if any) becomes the
   *  only copy again — the resolver next sees the unmodified text. */
  function remove(path: string) {
    if (!entries.has(path)) return;
    entries.delete(path);
    pathOrder = pathOrder.filter(function (p) { return p !== path; });
    lsRemoveItem(ENTRY_KEY_PREFIX + path);
    writeIndex(pathOrder);
    emit();
  }

  /** Decide a user/-prefixed path for a freshly-uploaded or
   *  freshly-promoted file. Suffix bumps avoid collisions with
   *  unrelated existing entries (e.g. an upload of `foo.flatppl`
   *  while `user/foo.flatppl` already exists lands at
   *  `user/foo-2.flatppl`). */
  function pathForUpload(filename: string): string {
    return uniquePath(USER_PREFIX + filename);
  }

  /** Subscribe to add / remove events. Returns an unsubscribe
   *  function. Text-only changes (updateSource) do NOT fire. */
  function subscribe(fn: () => void) {
    listeners.add(fn);
    return function () { listeners.delete(fn); };
  }

  // ---------- helpers ----------

  function basename(path: string): string {
    const i = path.lastIndexOf('/');
    return i < 0 ? path : path.slice(i + 1);
  }

  function uniquePath(candidate: string): string {
    if (!entries.has(candidate)) return candidate;
    // Split off any existing extension so suffixes land before it:
    //   user/foo.flatppl + collision → user/foo-2.flatppl
    const dot = candidate.lastIndexOf('.');
    const stem = dot > 0 ? candidate.slice(0, dot) : candidate;
    const ext  = dot > 0 ? candidate.slice(dot)    : '';
    let n = 2;
    for (;;) {
      const tryPath = stem + '-' + n + ext;
      if (!entries.has(tryPath)) return tryPath;
      n += 1;
    }
  }

  loadFromStorage();

  globalScope.FlatPPLWebUserStore = {
    USER_PREFIX:   USER_PREFIX,
    has:           has,
    get:           get,
    getSource:     getSource,
    list:          list,
    save:          save,
    updateSource:  updateSource,
    remove:        remove,
    pathForUpload: pathForUpload,
    subscribe:     subscribe,
  };
})(typeof window !== 'undefined' ? window : globalThis);
