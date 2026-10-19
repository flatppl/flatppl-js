// Browser bundle storage. Opening is ephemeral; Save is one IndexedDB transaction.
'use strict';

(function (scope: any) {
  interface Mounted {
    folder: string;
    id: string;
    bytes: Uint8Array;
    archive: any;
    saved: boolean;
    budget: { bytes: number; entries: number };
    limits?: Record<string, number>;
    /** An unlisted preview stays out of the file tree and the unsaved-work warning. */
    listed: boolean;
  }
  interface ArchiveScope {
    id: string;
    archive: any;
    owner: Mounted;
    prefix: string;
    aliases: Map<string, string>;
    depth: number;
  }
  const mounts = new Map<string, Mounted>();
  /** Folders of previews still being mounted. */
  const reserved = new Set<string>();
  const scopes = new Map<string, ArchiveScope>();
  const pendingScopes = new Map<string, Promise<ArchiveScope>>();
  const listeners = new Set<() => void>();
  let database: Promise<IDBDatabase> | undefined;
  const engine = () => scope.FlatPPLEngine;
  const changed = () => listeners.forEach(fn => fn());
  // URL previews have a smaller budget than deliberate local ZIP uploads.
  const LINK_LIMITS = { maxArchiveBytes: 8 * 1024 ** 2, maxMemberBytes: 2 * 1024 ** 2,
    maxTotalBytes: 8 * 1024 ** 2, maxEntries: 256, maxDepth: 4, maxPathBytes: 1024 };

  function db(): Promise<IDBDatabase> {
    if (!database) database = new Promise((resolve, reject) => {
      const request = scope.indexedDB.open('flatppl-bundles', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('bundles', { keyPath: 'folder' });
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      request.onblocked = () => reject(new Error('Close other FlatPPL tabs, then retry saving.'));
    });
    return database;
  }

  /** Nested archives inherit the limits of their mount. */
  async function mount(folder: string, bytes: Uint8Array, saved: boolean,
    limits?: Record<string, number>, listed = true): Promise<Mounted> {
    const id = folder + '.flatppl.zip';
    const budget = { bytes: 0, entries: 0 };
    const archive = await engine().readArchive(bytes, { id, budget, limits });
    const entry = { folder, id, bytes, archive, saved, budget, limits, listed };
    mounts.set(folder, entry);
    try { await addScope(id, archive, entry, folder); }
    catch (error) { mounts.delete(folder); await archive.close(); throw error; }
    return entry;
  }

  function checkLinkPaths(archive: any, prefix: string) {
    // The web tree stores full prefixes and renders folders recursively.
    // Bound URL-supplied names before any tree construction.
    for (const name of Object.keys(archive.entries)) {
      if ((prefix + '/' + name).split('/').length > 32) {
        throw new Error('Shared bundle paths exceed the browser folder depth limit.');
      }
    }
  }

  /** Do not publish a link the receiving playground would refuse to mount. */
  async function validateLink(bytes: Uint8Array) {
    const archive = await engine().readArchive(bytes, { id: 'share.zip', limits: LINK_LIMITS });
    try { checkLinkPaths(archive, 'new/shared-model'); }
    finally { await archive.close(); }
  }

  /** Cache link identities without decoding regular members. */
  async function addScope(id: string, archive: any, owner: Mounted, prefix: string, depth = 1): Promise<ArchiveScope> {
    const result = { id, archive, owner, prefix, depth, aliases: new Map<string, string>() };
    if (owner.limits) checkLinkPaths(archive, prefix);
    for (const [name, meta] of Object.entries(archive.entries) as [string, any][]) {
      if (meta.symlink) result.aliases.set(name, await archive.resolve(name));
    }
    if (mounts.get(owner.folder) !== owner) {
      await archive.close();
      throw new Error('The bundle preview is no longer open.');
    }
    scopes.set(id, result);
    return result;
  }

  /** Restore bytes without reading or decoding every member. */
  async function init() {
    try {
      const connection = await db();
      const records = await new Promise<any[]>((resolve, reject) => {
        const request = connection.transaction('bundles').objectStore('bundles').getAll();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
      });
      for (const record of records) {
        try { await mount(record.folder, record.bytes, true); }
        catch (error) { console.warn('Cannot open saved bundle ' + record.folder, error); }
      }
      changed();
    } catch (error) {
      // Storage failures must not prevent unsaved previews and downloads.
      console.warn('Bundle storage is unavailable', error);
    }
  }

  function owner(path: string | null | undefined): Mounted | undefined {
    if (!path) return undefined;
    let current = path;
    for (;;) {
      const found = Array.from(mounts.values()).find(m => current === m.id || current.startsWith(m.folder + '/'));
      if (found) return found;
      const member = engine().parseArchiveMemberId(current);
      if (!member) return undefined;
      current = member.archiveId;
    }
  }

  function relativeMember(archive: any, path: string): string {
    if (!archive.root) return path;
    if (!path.startsWith(archive.root + '/')) throw new Error('Member lies outside the bundle root.');
    return path.slice(archive.root.length + 1);
  }

  function memberId(s: ArchiveScope, path: string): string {
    return engine().archiveMemberId(s.id, (s.archive.root ? s.archive.root + '/' : '') + path);
  }

  function displayedScope(path: string): ArchiveScope | undefined {
    return Array.from(scopes.values()).filter(s => path.startsWith(s.prefix + '/'))
      .sort((a, b) => b.prefix.length - a.prefix.length)[0];
  }

  /** Nested ZIPs and directory bundles retain their own roots and identities. */
  async function ensureScope(id: string, graph?: any): Promise<ArchiveScope> {
    if (scopes.has(id)) return scopes.get(id)!;
    if (pendingScopes.has(id)) return pendingScopes.get(id)!;
    const pending = (async () => {
      const member = engine().parseArchiveMemberId(id);
      if (!member) throw new Error('Bundle is not open: ' + id);
      const outer = await ensureScope(member.archiveId, graph);
      if (outer.depth >= (outer.owner.limits?.maxDepth ?? engine().DEFAULT_LIMITS.maxDepth)) {
        throw new Error('Bundle exceeds the archive nesting limit');
      }
      const path = await outer.archive.resolve(relativeMember(outer.archive, member.path));
      const directory = (await outer.archive.stat(path))?.directory;
      const archive = directory ? await outer.archive.subdirectory(path)
        : await engine().readArchive(graph?.files?.[id] || await outer.archive.read(path),
          { id, budget: outer.owner.budget, limits: outer.owner.limits });
      try { return await addScope(id, archive, outer.owner, outer.prefix + '/' + path, outer.depth + 1); }
      catch (error) { await archive.close(); throw error; }
    })();
    pendingScopes.set(id, pending);
    try { return await pending; } finally { pendingScopes.delete(id); }
  }

  /** Expose reached nested scopes after the engine resolves its byte graph. */
  async function registerGraph(graph: any) {
    let added = false;
    for (const id of Object.keys(graph.archives || {})) {
      if (owner(id) && !scopes.has(id)) { await ensureScope(id, graph); added = true; }
    }
    if (added) changed();
  }

  /** Restore a selected member before its nested archives have been visited. */
  async function select(folder: string, relativePath: string): Promise<string | null> {
    const m = mounts.get(folder);
    if (!m || !relativePath || /[\\\0]/.test(relativePath)
        || relativePath.split('/').some(part => !part || part === '.' || part === '..')) return null;
    let s = scopes.get(m.id)!;
    let path = relativePath;
    let links = 0;
    for (;;) {
      let boundary: { path: string; symlink: boolean } | undefined;
      // Check actual entries once, not every growing prefix of a deep path.
      for (const [name, meta] of Object.entries(s.archive.entries) as [string, any][]) {
        const candidate = !meta.directory && name.endsWith('/main.flatppl') ? name.slice(0, name.lastIndexOf('/'))
          : (meta.symlink || (!meta.directory && /\.zip$/i.test(name))) ? name : null;
        if (candidate && path.startsWith(candidate + '/')
            && (!boundary || candidate.length < boundary.path.length)) {
          boundary = { path: candidate, symlink: candidate === name && meta.symlink };
        }
      }
      if (!boundary) {
        const meta = await s.archive.stat(path);
        return !meta || meta.directory ? null : s.prefix + '/' + await s.archive.resolve(path);
      }
      const resolved = await s.archive.resolve(boundary.path);
      const suffix = path.slice(boundary.path.length + 1);
      if (boundary.symlink) {
        if (++links > 40) throw new Error('Cyclic or excessive ZIP symbolic links');
        path = resolved + '/' + suffix;
      } else {
        s = await ensureScope(memberId(s, resolved));
        path = suffix;
        changed();
      }
    }
  }

  function canonical(path: string): string {
    const parsed = engine().parseArchiveMemberId(path);
    const s = parsed ? scopes.get(parsed.archiveId) : displayedScope(path);
    if (!s) return path;
    const relative = parsed ? relativeMember(s.archive, parsed.path) : path.slice(s.prefix.length + 1);
    return memberId(s, s.aliases.get(relative) || relative);
  }

  function displayPath(path: string): string {
    const member = engine().parseArchiveMemberId(path);
    const s = member && scopes.get(member.archiveId);
    return s ? s.prefix + '/' + relativeMember(s.archive, member.path) : path;
  }

  async function read(path: string): Promise<Uint8Array | null> {
    const m = owner(path);
    if (!m) return null;
    if (path === m.id) return m.bytes;
    const member = engine().parseArchiveMemberId(path);
    const s = member ? await ensureScope(member.archiveId) : displayedScope(path);
    if (!s) return null;
    const relative = member ? relativeMember(s.archive, member.path) : path.slice(s.prefix.length + 1);
    const resolved = await s.archive.resolve(relative);
    if (relative !== resolved) s.aliases.set(relative, resolved);
    return s.archive.read(resolved);
  }

  function list(saved: boolean) {
    const files = new Map<string, any>();
    for (const s of scopes.values()) if (s.owner.saved === saved && s.owner.listed) {
      for (const [name, meta] of Object.entries(s.archive.entries) as [string, any][]) {
        if (!meta.directory) files.set(s.prefix + '/' + name,
          { path: s.prefix + '/' + name, title: name.split('/').pop(), bundle: true });
      }
    }
    return [...files.values()];
  }

  /** A draft opened through a link belongs to the link's canonical target. */
  function draftFor(path: string, drafts: Record<string, string>): string | undefined {
    const target = displayPath(canonical(path));
    let result: string | undefined;
    for (const [name, text] of Object.entries(drafts)) if (displayPath(canonical(name)) === target) {
      if (result !== undefined && result !== text) throw new Error('Conflicting drafts refer to the same bundle member.');
      result = text;
    }
    return result;
  }

  function nameOf(name: string): string {
    const stem = name.split('/').pop()!.replace(/(?:\.flatppl)?\.zip$/i, '').replace(/\.flatppl$/i, '');
    return stem.replace(/[^\p{L}\p{N}_. -]/gu, '-').replace(/^\.+/, '').trim() || 'model';
  }

  function available(folder: string): boolean {
    const loose = scope.FlatPPLWebUserStore?.list() || [];
    const unsaved = scope.FlatPPLWebEphemeral?.list() || [];
    return !mounts.has(folder) && !reserved.has(folder) && ![...loose, ...unsaved].some(e =>
      e.path === folder || e.path.startsWith(folder + '/') || folder.startsWith(e.path + '/'));
  }

  /** Equality compares every member, independent of ZIP timestamps and compression. */
  async function equal(a: any, b: any): Promise<boolean> {
    if (a.root !== b.root) return false;
    const names = (archive: any) => Object.keys(archive.entries).sort();
    const left = names(a), right = names(b);
    if (left.length !== right.length || left.some((path, i) => path !== right[i])) return false;
    for (const path of left) {
      const am = a.entries[path], bm = b.entries[path];
      if (am.directory !== bm.directory || am.symlink !== bm.symlink || am.size !== bm.size) return false;
      if (am.directory) continue;
      if (am.symlink) {
        if (await a.readLink(path) !== await b.readLink(path)) return false;
        continue;
      }
      const x = await a.read(path), y = await b.read(path);
      if (x.length !== y.length || !x.every((byte: number, i: number) => byte === y[i])) return false;
    }
    return true;
  }

  /** Open a whole bundle without writes. Reuse only an unchanged saved copy. */
  async function open(bytes: Uint8Array, name: string, canReuse: (folder: string) => boolean = () => true,
    options: { limits?: Record<string, number>; listed?: boolean } = {}) {
    let folder = 'new/' + nameOf(name);
    for (let n = 2; !available(folder); n++) folder = 'new/' + nameOf(name) + '-' + n;
    reserved.add(folder);
    let m: Mounted;
    try { m = await mount(folder, bytes, false, options.limits, options.listed); }
    finally { reserved.delete(folder); }
    try {
      for (const existing of mounts.values()) {
        if (existing.saved && canReuse(existing.folder) && await equal(existing.archive, m.archive)) {
          // Readers capture their limits. Replacing the field alone leaves
          // already-open nested archives using the local-upload budget.
          if (options.limits) {
            // Logical equality permits different ZIP compression. Reuse the
            // validated incoming representation, and keep old readers intact
            // until its replacement has mounted.
            try { await mount(existing.folder, m.bytes, true, options.limits); }
            catch (error) { mounts.set(existing.folder, existing); throw error; }
            await existing.archive.close();
            await release(existing);
          }
          mounts.delete(folder);
          scopes.delete(m.id);
          await m.archive.close();
          return { path: existing.folder + '/' + existing.archive.entryPath, reused: true };
        }
      }
    } catch (error) {
      mounts.delete(folder);
      scopes.delete(m.id);
      await m.archive.close();
      throw error;
    }
    changed();
    return { path: folder + '/' + m.archive.entryPath, reused: false };
  }

  /** Capture all members and current drafts before compression or asynchronous Save. */
  async function snapshot(path: string, drafts: Record<string, string>, compression = 'store') {
    const m = owner(path);
    if (!m) throw new Error('No bundle is open for ' + path);
    const frozen = { ...drafts };
    async function encodeScope(s: ArchiveScope): Promise<Uint8Array> {
      const files: Record<string, Uint8Array> = Object.create(null);
      const symlinks: string[] = [];
      for (const [name, meta] of Object.entries(s.archive.entries) as [string, any][]) {
        const output = (s.archive.root ? s.archive.root + '/' : '') + name;
        if (meta.directory) { files[output + '/'] = new Uint8Array(); continue; }
        if (meta.symlink) {
          files[output] = new TextEncoder().encode(await s.archive.readLink(name));
          symlinks.push(output);
          continue;
        }
        const nested = scopes.get(memberId(s, name));
        const hasNestedDraft = nested && Object.keys(frozen)
          .some(path => displayPath(canonical(path)).startsWith(nested.prefix + '/'));
        const draft = draftFor(memberId(s, name), frozen);
        files[output] = draft !== undefined ? new TextEncoder().encode(draft)
          : hasNestedDraft ? await encodeScope(nested!) : await s.archive.read(name);
      }
      return engine().writeArchive(files, { compression, symlinks });
    }
    return encodeScope(scopes.get(m.id)!);
  }

  /** Add or explicitly replace a saved folder in one transaction. */
  async function save(bytes: Uint8Array, name: string, replaceFolder?: string) {
    const folder = 'user/' + nameOf(name);
    const previous = mounts.get(folder);
    const replace = replaceFolder === folder && previous?.saved;
    if (!replace && !available(folder)) throw new Error('Choose a different name. This folder already exists.');
    const budget = { bytes: 0, entries: 0 };
    const archive = await engine().readArchive(bytes, { id: folder + '.flatppl.zip', budget });
    try {
      const connection = await db();
      await new Promise<void>((resolve, reject) => {
        const tx = connection.transaction('bundles', 'readwrite');
        if (replace) tx.objectStore('bundles').put({ folder, bytes });
        else tx.objectStore('bundles').add({ folder, bytes });
        tx.oncomplete = () => resolve();
        tx.onabort = () => reject(tx.error || new Error('The bundle was not saved.'));
        tx.onerror = () => reject(tx.error);
      });
    } catch (error) {
      await archive.close();
      throw error;
    }
    if (previous) await release(previous);
    const mounted = { folder, id: folder + '.flatppl.zip', bytes, archive, saved: true, budget, listed: true };
    mounts.set(folder, mounted);
    await addScope(mounted.id, archive, mounted, folder);
    changed();
    return folder + '/' + archive.entryPath;
  }

  async function discard(path: string) {
    const m = owner(path);
    if (!m || m.saved) return;
    mounts.delete(m.folder);
    await release(m);
    changed();
  }

  /** Delete a saved bundle only after its whole-folder transaction succeeds. */
  async function remove(path: string) {
    const m = owner(path);
    if (!m) return;
    if (!m.saved) return discard(path);
    const connection = await db();
    await new Promise<void>((resolve, reject) => {
      const tx = connection.transaction('bundles', 'readwrite');
      tx.objectStore('bundles').delete(m.folder);
      tx.oncomplete = () => resolve();
      tx.onabort = () => reject(tx.error || new Error('The bundle was not deleted.'));
      tx.onerror = () => reject(tx.error);
    });
    mounts.delete(m.folder);
    await release(m);
    changed();
  }

  async function release(m: Mounted) {
    for (const [id, s] of scopes) if (s.owner === m) {
      scopes.delete(id);
      await s.archive.close();
    }
  }

  scope.FlatPPLWebBundles = { init, open, save, snapshot, read, canonical, displayPath, list, LINK_LIMITS, validateLink,
    registerGraph, select, draftFor,
    has: (path: string | null | undefined) => !!owner(path),
    folder: (path: string | null | undefined) => owner(path)?.folder,
    /** The archive limits a path's bundle was opened with, if any. */
    limits: (path: string | null | undefined) => owner(path)?.limits,
    isSaved: (path: string | null | undefined) => !!owner(path)?.saved,
    hasUnsaved: () => Array.from(mounts.values()).some(m => !m.saved && m.listed), discard, remove,
    subscribe(fn: () => void) { listeners.add(fn); return () => listeners.delete(fn); },
  };
})(typeof window !== 'undefined' ? window : globalThis);
