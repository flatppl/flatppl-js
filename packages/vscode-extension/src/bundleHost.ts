'use strict';
// VS Code's §04 archive boundary. The engine owns ZIP parsing and import walks.
// This host snapshots editor buffers, supplies trusted bytes, and keeps virtual
// members read-only. It never extracts files while opening a model.

const { BUNDLE_SCHEME, REMOTE_SCHEME, enginePathOf, localSourceUri } = require('./remoteModule');

function createBundleHost(vscode: any, api: any, cache: any, approve: (url: string) => Promise<boolean>) {
  const graphs = new Map<string, any>();
  const origins = new Map<string, any>();
  const changed = new vscode.EventEmitter();
  let onMount: ((graph: any) => void) | undefined;

  /** Keep each archive's namespace intact, including nested ZIPs (§04 Bundles). */
  function remember(graph: any, origin: any) {
    const changes = [];
    for (const id of new Set([...Object.keys(graph.files), ...Object.keys(graph.archives)])) {
      const previous = graphs.get(id)?.files[id];
      const next = graph.files[id];
      if (previous && next && api.parseArchiveMemberId(id) && previous !== next
          && (previous.length !== next.length || previous.some((value: number, i: number) => value !== next[i]))) {
        changes.push({ type: vscode.FileChangeType.Changed, uri: uriFor(id) });
      }
      graphs.set(id, graph);
      origins.set(id, origin);
    }
    graphs.set(graph.primaryPath, graph);
    origins.set(graph.primaryPath, origin);
    if (changes.length) changed.fire(changes);
    onMount?.(graph);
    return graph;
  }

  /** Nested directory scopes reuse their containing ZIP, without decoding it as text. */
  async function archiveFor(id: string, graph: any): Promise<any> {
    const info = graph.archives[id];
    if (info?.parent) {
      const parent = await archiveFor(info.parent, graph);
      try { return await parent.subdirectory(info.directory); }
      catch (error) { await parent.close(); throw error; }
    }
    return api.readArchive(graph.files[id], { id });
  }

  /** The query preserves the opaque engine ID. The path only names the tab. */
  function uriFor(id: string) {
    const member = api.parseArchiveMemberId(id);
    if (!member) throw new Error('Not an archive member: ' + id);
    return vscode.Uri.from({ scheme: BUNDLE_SCHEME, path: '/' + member.path, query: id });
  }

  /** Keep wrapper roots and reader lifetime consistent across virtual filesystem calls. */
  async function withMember(id: string, action: (archive: any, path: string) => Promise<any>) {
    const member = api.parseArchiveMemberId(id);
    const owner = member && graphs.get(member.archiveId);
    if (!owner) throw vscode.FileSystemError.FileNotFound(id);
    const archive = await archiveFor(member.archiveId, owner);
    try {
      if (archive.root && member.path !== archive.root && !member.path.startsWith(archive.root + '/')) {
        throw vscode.FileSystemError.FileNotFound(id);
      }
      const path = (archive.root ? member.path.slice(archive.root.length + 1) : member.path) || '.';
      return await action(archive, path);
    } finally { await archive.close(); }
  }

  /** Read an unvisited member without decoding binary data or fetching elsewhere. */
  async function readMember(id: string): Promise<Uint8Array> {
    const graph = graphs.get(id);
    return graph?.files[id] || withMember(id, (archive, path) => archive.read(path));
  }

  /** Capture all open text buffers before I/O, so one export sees one snapshot. */
  function reader(sourceUri: any, options: { offline?: boolean; force?: boolean } = {}) {
    const buffers = new Map<string, Uint8Array>();
    for (const doc of vscode.workspace.textDocuments || []) {
      if (doc.isDirty) buffers.set(doc.uri.toString(), Buffer.from(doc.getText(), 'utf8'));
    }
    const member = sourceUri.scheme === BUNDLE_SCHEME ? api.parseArchiveMemberId(enginePathOf(sourceUri)) : null;
    const snapshot = member ? graphs.get(member.archiveId) : undefined;
    const origin = member ? origins.get(member.archiveId) : sourceUri;
    return async (resolved: string): Promise<Uint8Array | null> => {
      if (snapshot?.files[resolved]) return snapshot.files[resolved];
      if (api.parseArchiveMemberId(resolved)) return snapshot ? readMember(resolved) : null;
      if (cache.isUrl(resolved)) {
        const dirty = buffers.get(vscode.Uri.parse(resolved).with({ scheme: REMOTE_SCHEME, query: resolved }).toString());
        if (dirty) return dirty;
        const result = await cache.fetchToCache(resolved, { approve, ...options });
        return result.content;
      }
      const uri = localSourceUri(origin, resolved, vscode.Uri.parse);
      return buffers.get(uri.toString()) || await vscode.workspace.fs.readFile(uri);
    };
  }

  /** Resolve source, ZIP, or virtual source with the same host snapshot. */
  async function resolve(uri: any, source?: string, strict = false,
    options: { offline?: boolean; force?: boolean } = {}) {
    const id = enginePathOf(uri);
    const read = reader(uri, options);
    const bytes = source === undefined ? await read(id) : Buffer.from(source, 'utf8');
    const graph = await api.resolveBundleBytes(id, bytes, read, { strict });
    const member = api.parseArchiveMemberId(id);
    return remember(graph, origins.get(id) || (member && origins.get(member.archiveId)) || uri);
  }

  /** List a mounted archive independently of which members its main imports. */
  function members(id: string): Array<{ id: string; path: string; directory: boolean }> {
    const member = api.parseArchiveMemberId(id);
    const archiveId = member ? member.archiveId : id;
    const graph = graphs.get(archiveId) || graphs.get(id);
    const archive = graph?.archives[archiveId];
    if (!archive) throw vscode.FileSystemError.FileNotFound(id);
    return Object.entries(archive.entries).map(([path, entry]: [string, any]) => ({
      id: api.archiveMemberId(archiveId, archive.root ? archive.root + '/' + path : path),
      path, directory: !!entry.directory,
    }));
  }

  /** Explicit working-copy creation never overwrites an existing destination. */
  async function extract(id: string, destination: any) {
    const member = api.parseArchiveMemberId(id);
    const archiveId = member ? member.archiveId : id;
    const archive = graphs.get(archiveId)?.archives[archiveId];
    if (Object.values(archive?.entries || {}).some((entry: any) => entry.symlink)) {
      throw new Error('This archive contains symbolic links. Extract it with a ZIP tool that preserves links.');
    }
    try {
      await vscode.workspace.fs.stat(destination);
    } catch (error: any) {
      if (error.code !== 'FileNotFound') throw error;
      await vscode.workspace.fs.createDirectory(destination);
      for (const member of members(id)) {
        if (member.directory) continue;
        const target = vscode.Uri.joinPath(destination, member.path);
        const parent = target.with({ path: target.path.replace(/\/[^/]+$/, '') });
        await vscode.workspace.fs.createDirectory(parent);
        await vscode.workspace.fs.copy(uriFor(member.id), target, { overwrite: false });
      }
      return;
    }
    throw vscode.FileSystemError.FileExists(destination);
  }

  const provider = {
    onDidChangeFile: changed.event,
    watch() { return { dispose() {} }; },
    async stat(uri: any) {
      const bytes = graphs.get(uri.query)?.files[uri.query];
      const entry = bytes ? { directory: false, size: bytes.length }
        : await withMember(uri.query, (archive, path) => archive.stat(path));
      if (!entry) throw vscode.FileSystemError.FileNotFound(uri);
      return { type: entry.directory ? vscode.FileType.Directory : vscode.FileType.File,
        ctime: 0, mtime: 0, size: entry.size,
        permissions: vscode.FilePermission.Readonly };
    },
    readFile(uri: any) { return readMember(uri.query); },
    readDirectory(uri: any) {
      return withMember(uri.query, async (archive, path) => {
        if (!(await archive.stat(path))?.directory) throw vscode.FileSystemError.FileNotADirectory(uri);
        const canonical = await archive.resolve(path);
        const prefix = canonical ? canonical + '/' : '';
        const children = new Map<string, number>();
        for (const [name, entry] of Object.entries(archive.entries) as [string, any][]) {
          if (!name.startsWith(prefix) || name === canonical) continue;
          const relative = name.slice(prefix.length), slash = relative.indexOf('/');
          const child = slash < 0 ? relative : relative.slice(0, slash);
          const directory = slash >= 0 || entry.directory;
          children.set(child, directory ? vscode.FileType.Directory : vscode.FileType.File);
        }
        return Array.from(children);
      });
    },
    writeFile() { throw vscode.FileSystemError.NoPermissions('Extract a working copy to edit this bundle.'); },
    createDirectory() { throw vscode.FileSystemError.NoPermissions(); },
    delete() { throw vscode.FileSystemError.NoPermissions(); },
    rename() { throw vscode.FileSystemError.NoPermissions(); },
  };

  return { resolve, reader, uriFor, readMember, members, extract, provider,
    graphFor: (id: string) => graphs.get(id),
    pack: (graph: any) => api.packBundle(graph, { compression: 'deflate' }),
    onMount(fn: (graph: any) => void) { onMount = fn; },
    dispose() { graphs.clear(); origins.clear(); changed.dispose(); },
  };
}

module.exports = { createBundleHost };
