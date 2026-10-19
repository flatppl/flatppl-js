'use strict';

// §04 Module composition — Bundles. Hosts supply bytes and trust policy.
// Archive reads are lazy, bounded and confined. Compilation only consumes the
// resulting plain graph, so neither ZIP I/O nor decompression enters execution.
import type { LoadDependency, ArchiveLimits, ArchiveEntry, Archive, ArchiveInfo,
  BundleGraph, ReadBytes } from './bundle-types';
const { sourceDependencies } = require('./bundle-deps.ts');
const { resolveModulePath } = require('./module-resolve.ts');

type Budget = { bytes: number; entries: number };
type PathNode = { children: Map<string, PathNode>; bytes: number; entry?: any };

const DEFAULT_LIMITS: ArchiveLimits = Object.freeze({
  maxArchiveBytes: 64 * 1024 * 1024,
  maxMemberBytes: 128 * 1024 * 1024,
  maxTotalBytes: 256 * 1024 * 1024,
  maxEntries: 10_000,
  maxDepth: 8,
  // ZIP's filename field is 16 bits. Apply the same bound to link targets.
  maxPathBytes: 65_535,
});

/** RFC3986 encoding shared with flatppl-bundle in Rust. */
function encodeIdentityPart(text: string): string {
  return encodeURIComponent(text).replace(/[!'()*]/g, c => `%${c.charCodeAt(0).toString(16).toUpperCase()}`);
}

/** Member IDs retain archive identity and physical wrapper paths (spec §04). */
function archiveMemberId(archiveId: string, path: string): string {
  return `flatppl-zip:${encodeIdentityPart(archiveId)}!/${path.split('/').map(encodeIdentityPart).join('/')}`;
}

/** Decode only this virtual scheme. Hosts must not treat it as a fetch URL. */
function parseArchiveMemberId(id: string): { archiveId: string; path: string } | null {
  if (!id.startsWith('flatppl-zip:')) return null;
  const separator = id.indexOf('!/', 12);
  if (separator < 0) throw new Error('Invalid FlatPPL archive member identity');
  const archiveId = decodeURIComponent(id.slice(12, separator));
  const path = id.slice(separator + 2).split('/').map(part => {
    const decoded = decodeURIComponent(part);
    if (decoded.includes('/')) throw new Error('Invalid encoded separator in archive member identity');
    return decoded;
  }).join('/');
  validateEntryName(path);
  return { archiveId, path };
}

/** Check names before normalization, including names from raw ZIP headers. */
function validateEntryName(path: string, maxBytes = DEFAULT_LIMITS.maxPathBytes, raw = false): void {
  if (raw) {
    if (path.length > maxBytes) throw new Error('Bundle path exceeds its byte limit');
  } else checkPathBytes(path, maxBytes);
  const name = path.endsWith('/') ? path.slice(0, -1) : path;
  if (!name || path.startsWith('/') || /^[a-z]:/i.test(path) || /[\\\0]/.test(path)
      || name.split('/').some(part => !part || part === '.' || part === '..')) {
    throw new Error(`Unsafe ZIP entry name: '${path}'`);
  }
}

function checkPathBytes(path: string, maxBytes: number): number {
  if (path.length > maxBytes) throw new Error('Bundle path exceeds its byte limit');
  const size = new TextEncoder().encode(path).length;
  if (size > maxBytes) throw new Error('Bundle path exceeds its byte limit');
  return size;
}

/** Components keep deep implicit directories linear in the input name size. */
function indexPath(root: PathNode, path: string, entry: any): void {
  let node = root;
  for (const part of path.split('/')) {
    if (node.entry && !node.entry.directory) throw new Error(`ZIP file/directory collision: '${path}'`);
    let child = node.children.get(part);
    if (!child) {
      child = { children: new Map(), bytes: new TextEncoder().encode(part).length };
      node.children.set(part, child);
    }
    node = child;
  }
  if (node.entry) throw new Error(`Duplicate ZIP entry: '${path}'`);
  if (node.children.size && !entry.directory) throw new Error(`ZIP file/directory collision: '${path}'`);
  node.entry = entry;
}

function pathNode(root: PathNode, path: string): PathNode | undefined {
  let node: PathNode | undefined = root;
  if (path) for (const part of path.split('/')) node = node?.children.get(part);
  return node;
}

function metadata(node: PathNode): ArchiveEntry {
  return node.entry ? { size: node.entry.uncompressedSize, directory: node.entry.directory, symlink: node.entry.symlink }
    : { size: 0, directory: true, symlink: false };
}

/** Join a relative reference. Resolve links before collapsing parent segments. */
function relativePath(importer: string, source: string): string {
  if (!source || source.startsWith('/') || /^[a-z][a-z0-9+.-]*:/i.test(source)
      || /[\\\0]/.test(source)) {
    throw new Error(`Bundle paths must be relative: '${source}'`);
  }
  return [...importer.split('/').slice(0, -1), source].join('/');
}

function limitsFor(options?: Partial<ArchiveLimits>): ArchiveLimits {
  const result = { ...DEFAULT_LIMITS, ...options };
  for (const [key, value] of Object.entries(result)) {
    if (!Number.isSafeInteger(value) || value < 1) throw new Error(`Invalid archive limit '${key}'`);
  }
  return result;
}

function utf8(bytes: Uint8Array, path: string): string {
  try { return new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
  catch (_) { throw new Error(`FlatPPL source is not valid UTF-8: '${path}'`); }
}

/** Index one ZIP without extracting files. STORE and Deflate are portable. */
async function readArchive(bytes: Uint8Array,
  options: { id: string; limits?: Partial<ArchiveLimits>; budget?: Budget }): Promise<Archive> {
  const limits = limitsFor(options.limits);
  const budget = options.budget || { bytes: 0, entries: 0 };
  if (bytes.length > limits.maxArchiveBytes) throw new Error('ZIP exceeds the compressed archive size limit');
  const zip = await import('@zip.js/zip.js');
  const reader = new zip.ZipReader(new zip.Uint8ArrayReader(bytes), {
    strictness: 'strict', filenameValidation: 'strict', checkCrc32: true,
    checkOverlappingEntry: true, useWebWorkers: false, useCompressionStream: true,
  });
  const records = new Map<string, any>();
  const tree: PathNode = { children: new Map(), bytes: 0 };
  const cache = new Map<string, Promise<Uint8Array>>();
  let declaredBytes = 0;
  try {
    for await (const entry of reader.getEntriesGenerator()) {
      if (++budget.entries > limits.maxEntries) throw new Error('ZIP exceeds the entry count limit');
      validateEntryName(entry.filename, limits.maxPathBytes);
      // Legacy encodings preserve ASCII separators. Validate raw names too,
      // since a Unicode extra field may replace an unsafe original filename.
      validateEntryName(Array.from(entry.rawFilename, byte => String.fromCharCode(byte)).join(''), limits.maxPathBytes, true);
      const path = entry.filename.replace(/\/$/, '');
      if (records.has(path)) throw new Error(`Duplicate ZIP entry: '${path}'`);
      if (entry.encrypted || ![0, 8].includes(entry.compressionMethod)) {
        throw new Error(`Unsupported encrypted or compressed ZIP entry: '${path}'`);
      }
      const kind = (entry.unixMode || 0) & 0o170000;
      if (kind && ![0o100000, 0o040000, 0o120000].includes(kind)) {
        throw new Error(`Unsupported ZIP file type: '${path}'`);
      }
      if (!Number.isSafeInteger(entry.uncompressedSize) || entry.uncompressedSize < 0
          || entry.uncompressedSize > limits.maxMemberBytes) throw new Error(`ZIP member exceeds its size limit: '${path}'`);
      if (entry.symlink && entry.uncompressedSize > limits.maxPathBytes) throw new Error('ZIP symbolic link target exceeds its path byte limit');
      declaredBytes += entry.uncompressedSize;
      if (declaredBytes > limits.maxTotalBytes) throw new Error('ZIP exceeds the total size limit');
      records.set(path, entry);
      indexPath(tree, path, entry);
    }
    let root = '';
    if (!records.has('main.flatppl')) {
      const top = new Set([...records.keys()].map(path => path.split('/')[0]));
      if (top.size === 1) {
        const candidate = [...top][0];
        if (records.has(`${candidate}/main.flatppl`)) root = candidate;
      }
    }
    if (!records.has(root ? `${root}/main.flatppl` : 'main.flatppl')) {
      throw new Error('ZIP bundle has no root main.flatppl or sole enclosing directory with main.flatppl');
    }

    /** The sink bounds actual output before retaining each decoded chunk. */
    function readPhysical(path: string): Promise<Uint8Array> {
      const cached = cache.get(path);
      if (cached) return cached;
      const pending = (async () => {
        const entry = records.get(path);
        if (!entry || entry.directory) throw new Error(`ZIP member is not a file: '${path}'`);
        let size = 0;
        const chunks: Uint8Array[] = [];
        const sink = new WritableStream<Uint8Array>({ write(chunk) {
          size += chunk.length;
          budget.bytes += chunk.length;
          if (size > (entry.symlink ? Math.min(limits.maxMemberBytes, limits.maxPathBytes) : limits.maxMemberBytes)
              || budget.bytes > limits.maxTotalBytes) {
            throw new Error('ZIP decompression exceeds its size limit');
          }
          chunks.push(chunk);
        } });
        await entry.getData(sink);
        if (size !== entry.uncompressedSize) throw new Error(`ZIP member size mismatch: '${path}'`);
        const result = new Uint8Array(size);
        let offset = 0;
        for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.length; }
        return result;
      })();
      cache.set(path, pending);
      return pending;
    }

    /** A directory bundle narrows containment without copying archive bytes. */
    function scope(id: string, prefix: string, identityRoot: string, parent?: string,
      directory?: string): Archive {
      const entries: Record<string, ArchiveEntry> = Object.create(null);
      for (const [path, entry] of records) {
        if (prefix && !path.startsWith(`${prefix}/`)) continue;
        const relative = prefix ? path.slice(prefix.length + 1) : path;
        entries[relative] = { size: entry.uncompressedSize, directory: entry.directory, symlink: entry.symlink };
      }
      const full = (path: string) => prefix ? `${prefix}/${path}` : path;
      const scopeRoot = pathNode(tree, prefix)!;
      async function canonical(path: string, allowMissing = false): Promise<{ path: string; node: PathNode } | null> {
        checkPathBytes(path, limits.maxPathBytes);
        const pending = [{ text: relativePath('main.flatppl', path), offset: 0 }];
        const resolved: string[] = [];
        const nodes: PathNode[] = [scopeRoot];
        let length = 0;
        let links = 0;
        while (pending.length) {
          const frame = pending[pending.length - 1];
          if (frame.offset > frame.text.length) { pending.pop(); continue; }
          const slash = frame.text.indexOf('/', frame.offset);
          const end = slash < 0 ? frame.text.length : slash;
          const part = frame.text.slice(frame.offset, end);
          frame.offset = end + 1;
          if (!part || part === '.') continue;
          if (part === '..') {
            if (!resolved.length) throw new Error(`Bundle path escapes its root: '${path}'`);
            resolved.pop();
            length -= nodes.pop()!.bytes + (resolved.length ? 1 : 0);
            continue;
          }
          const child = nodes[nodes.length - 1].children.get(part);
          if (!child) {
            if (allowMissing) return null;
            throw new Error(`ZIP member not found: '${path}'`);
          }
          const entry = child.entry;
          if (entry?.symlink) {
            if (++links > 40) throw new Error(`Cyclic or excessive ZIP symbolic links: '${path}'`);
            const bytes = await readPhysical(entry.filename.replace(/\/$/, ''));
            const target = utf8(bytes, path);
            pending.push({ text: relativePath('main.flatppl', target), offset: 0 });
          } else {
            if (entry && !entry.directory && pending.some(frame => frame.offset <= frame.text.length)) {
              throw new Error(`ZIP member is not a directory: '${path}'`);
            }
            length += child.bytes + (resolved.length ? 1 : 0);
            if (length > limits.maxPathBytes) throw new Error('Expanded bundle path exceeds its byte limit');
            resolved.push(part);
            nodes.push(child);
          }
        }
        return { path: resolved.join('/'), node: nodes[nodes.length - 1] };
      }
      async function resolve(path: string): Promise<string> { return (await canonical(path))!.path; }
      const archive: Archive = {
        id, root: identityRoot, entries, entryPath: 'main.flatppl', parent, directory,
        resolve,
        async stat(path) { const result = await canonical(path, true); return result ? metadata(result.node) : null; },
        async read(path) { return readPhysical(full(await resolve(path))); },
        async readLink(path) {
          validateEntryName(path);
          if (!records.get(full(path))?.symlink) throw new Error(`ZIP member is not a symbolic link: '${path}'`);
          return utf8(await readPhysical(full(path)), path);
        },
        async subdirectory(path) {
          const resolved = await resolve(path);
          if (!resolved) return archive;
          if (!metadata(pathNode(scopeRoot, resolved)!).directory) throw new Error(`ZIP member is not a directory: '${path}'`);
          if (!records.has(`${full(resolved)}/main.flatppl`)) throw new Error(`Bundle directory has no main.flatppl: '${path}'`);
          const childId = archiveMemberId(id, identityRoot ? `${identityRoot}/${resolved}` : resolved);
          return scope(childId, full(resolved), '', id, resolved);
        },
        async close() { await reader.close(); },
      };
      return archive;
    }
    const archive = scope(options.id, root, root);
    // Links are metadata with executable path semantics. Validate every link,
    // including unused links, before a host exposes this archive as a bundle.
    for (const [path, entry] of Object.entries(archive.entries)) if (entry.symlink) await archive.resolve(path);
    const main = await archive.resolve('main.flatppl');
    if ((await archive.stat(main))?.directory) throw new Error('ZIP main.flatppl must be a file');
    return archive;
  } catch (error) {
    await reader.close();
    throw error;
  }
}

/** Create an ordinary ZIP. A trailing slash denotes an explicit directory. */
async function writeArchive(files: Record<string, Uint8Array>,
  options: { compression?: 'store' | 'deflate'; limits?: Partial<ArchiveLimits>;
    symlinks?: string[] } = {}): Promise<Uint8Array> {
  const limits = limitsFor(options.limits);
  const names = Object.keys(files).sort();
  const symlinks = new Set(options.symlinks || []);
  const tree: PathNode = { children: new Map(), bytes: 0 };
  if (names.length > limits.maxEntries) throw new Error('Bundle exceeds the entry count limit');
  let total = 0;
  for (const name of names) {
    validateEntryName(name, limits.maxPathBytes);
    const bytes = files[name];
    if (!(bytes instanceof Uint8Array)) throw new Error(`Bundle member must contain bytes: '${name}'`);
    total += bytes.length;
    if (bytes.length > limits.maxMemberBytes || total > limits.maxTotalBytes) throw new Error('Bundle exceeds its size limit');
    if (symlinks.has(name) && bytes.length > limits.maxPathBytes) throw new Error('ZIP symbolic link target exceeds its path byte limit');
    indexPath(tree, name.replace(/\/$/, ''), { directory: name.endsWith('/') });
  }
  const zip = await import('@zip.js/zip.js');
  const writer = new zip.ZipWriter(new zip.Uint8ArrayWriter(), {
    useWebWorkers: false, useCompressionStream: true,
    level: options.compression === 'store' ? 0 : 6,
    lastModDate: new Date('1980-01-01T00:00:00Z'), extendedTimestamp: false,
  });
  for (const name of names) {
    await writer.add(name, name.endsWith('/') ? undefined : new zip.Uint8ArrayReader(files[name]),
      { directory: name.endsWith('/'), ...(symlinks.has(name) ? { unixMode: 0o120777 } : {}) });
  }
  const bytes = await writer.close();
  if (bytes.length > limits.maxArchiveBytes) throw new Error('ZIP exceeds the compressed archive size limit');
  return bytes;
}

/** Load a graph once. Reader overrides capture unsaved buffers before ZIP bytes. */
async function resolveBundleBytes(primaryPath: string, primaryBytes: Uint8Array,
  readBytes: ReadBytes, options: { strict?: boolean; limits?: Partial<ArchiveLimits> } = {}): Promise<BundleGraph> {
  const strict = options.strict !== false;
  const limits = limitsFor(options.limits);
  const budget: Budget = { bytes: 0, entries: 0 };
  const graph: BundleGraph = { primaryPath, primarySource: '', sources: Object.create(null),
    files: Object.create(null), resolutions: Object.create(null), archives: Object.create(null) };
  const opened = new Map<string, Promise<Archive>>();
  const origins = new Map<string, { archive: Archive; path: string }>();
  const walked = new Set<string>();
  let capturedBytes = 0;
  let capturedFiles = 0;
  function capture(path: string, bytes: Uint8Array): Uint8Array {
    if (!(bytes instanceof Uint8Array)) throw new Error(`Bundle reader must return bytes: '${path}'`);
    capturedBytes += bytes.length;
    if (bytes.length > limits.maxMemberBytes || capturedBytes > limits.maxTotalBytes
        || ++capturedFiles > limits.maxEntries + 1) throw new Error('Resolved bundle exceeds its size or entry count limit');
    return graph.files[path] = bytes;
  }
  capture(primaryPath, primaryBytes);
  function recordArchive(archive: Archive): Archive {
    graph.archives[archive.id] = { root: archive.root, entries: archive.entries,
      ...(archive.parent ? { parent: archive.parent, directory: archive.directory } : {}) };
    return archive;
  }
  async function bytesAt(path: string): Promise<Uint8Array> {
    if (Object.prototype.hasOwnProperty.call(graph.files, path)) return graph.files[path];
    const override = await readBytes(path);
    if (override != null) return capture(path, override);
    const member = parseArchiveMemberId(path);
    if (member) {
      const archive = await open(member.archiveId, 0);
      return capture(path, await archive.read(withoutRoot(archive, member.path)));
    }
    throw new Error(`Bundle dependency not found: '${path}'`);
  }
  function withoutRoot(archive: Archive, path: string): string {
    if (!archive.root) return path;
    if (!path.startsWith(`${archive.root}/`)) throw new Error(`Archive member lies outside the selected bundle root: '${path}'`);
    return path.slice(archive.root.length + 1);
  }
  async function open(id: string, depth: number): Promise<Archive> {
    let count = 1;
    let ancestor = parseArchiveMemberId(id);
    while (ancestor) { count++; ancestor = parseArchiveMemberId(ancestor.archiveId); }
    if (count > limits.maxDepth || depth >= limits.maxDepth) throw new Error('Bundle exceeds the archive nesting limit');
    const prior = opened.get(id);
    if (prior) return prior;
    const pending = (async () => {
      const member = parseArchiveMemberId(id);
      if (member) {
        const outer = await open(member.archiveId, depth + 1);
        const relative = withoutRoot(outer, member.path);
        const canonical = await outer.resolve(relative);
        if ((await outer.stat(canonical))?.directory) return recordArchive(await outer.subdirectory(canonical));
      }
      const bytes = await bytesAt(id);
      return recordArchive(await readArchive(bytes, { id, limits, budget }));
    })();
    opened.set(id, pending);
    return pending;
  }
  function idFor(archive: Archive, path: string): string {
    return archiveMemberId(archive.id, archive.root ? `${archive.root}/${path}` : path);
  }
  async function moduleAt(path: string): Promise<string> {
    if (isZipPath(path)) {
      const archive = await open(path, 0);
      const canonical = await archive.resolve(archive.entryPath);
      const id = idFor(archive, canonical);
      origins.set(id, { archive, path: canonical });
      await bytesAt(id);
      return id;
    }
    const member = parseArchiveMemberId(path);
    if (member) {
      const archive = await open(member.archiveId, 0);
      const canonical = await archive.resolve(withoutRoot(archive, member.path));
      if ((await archive.stat(canonical))?.directory) {
        const child = await open(idFor(archive, canonical), 0);
        const entry = await child.resolve(child.entryPath);
        const id = idFor(child, entry);
        origins.set(id, { archive: child, path: entry });
        await bytesAt(id);
        return id;
      }
      const id = idFor(archive, canonical);
      origins.set(id, { archive, path: canonical });
      await bytesAt(id);
      return id;
    }
    await bytesAt(path);
    return path;
  }
  async function dependency(importer: string, source: string, kind: 'modules' | 'data'): Promise<string> {
    const origin = origins.get(importer);
    let path: string;
    if (origin) {
      const relative = relativePath(origin.path, source);
      const canonical = await origin.archive.resolve(relative);
      if (!canonical && kind === 'modules') {
        return idFor(origin.archive, await origin.archive.resolve(origin.archive.entryPath));
      }
      path = idFor(origin.archive, canonical);
    } else path = resolveModulePath(importer, source);
    return kind === 'modules' ? moduleAt(path) : (await bytesAt(path), path);
  }
  async function walk(path: string): Promise<void> {
    if (walked.has(path)) return;
    walked.add(path);
    const source = utf8(await bytesAt(path), path);
    graph.sources[path] = source;
    graph.resolutions[path] = { modules: Object.create(null), data: Object.create(null) };
    for (const dep of sourceDependencies(source, strict) as LoadDependency[]) {
      if (dep.source === null) continue;
      try {
        const resolved = await dependency(path, dep.source, dep.kind);
        graph.resolutions[path][dep.kind][dep.source] = resolved;
        if (dep.kind === 'modules') await walk(resolved);
      } catch (error) { if (strict) throw error; }
    }
  }
  try {
    graph.primaryPath = await moduleAt(primaryPath);
    await walk(graph.primaryPath);
    graph.primarySource = graph.sources[graph.primaryPath];
    return graph;
  } finally {
    for (const pending of opened.values()) {
      try { await (await pending).close(); } catch (_) { /* Retain the original load error. */ }
    }
  }
}

function isZipPath(path: string): boolean {
  const member = parseArchiveMemberId(path);
  const filename = member ? member.path : path.split(/[?#]/, 1)[0];
  return filename.toLowerCase().endsWith('.zip');
}

/** Portable export uses the same parsed graph and rewrites path arguments only. */
async function packBundle(graph: BundleGraph,
  options: { compression?: 'store' | 'deflate'; limits?: Partial<ArchiveLimits> } = {}): Promise<Uint8Array> {
  const files: Record<string, Uint8Array> = Object.create(null);
  const names = new Map<string, string>([[graph.primaryPath, 'main.flatppl']]);
  const usedNames = new Set<string>(['main.flatppl']);
  const modules: string[] = [];
  const seen = new Set<string>();
  function allocate(path: string, module: boolean): string {
    const previous = names.get(path);
    if (previous) return previous;
    const member = parseArchiveMemberId(path);
    const basename = (member ? member.path : path.split(/[?#]/, 1)[0]).split('/').pop() || 'data';
    const clean = basename.replace(/[^A-Za-z0-9._-]/g, '_');
    const directory = module ? 'modules' : 'data';
    const dot = clean.lastIndexOf('.');
    const stem = dot > 0 ? clean.slice(0, dot) : clean;
    const extension = dot > 0 ? clean.slice(dot) : '';
    let name = `${directory}/${clean}`;
    for (let suffix = 2; usedNames.has(name); suffix++) name = `${directory}/${stem}-${suffix}${extension}`;
    names.set(path, name);
    usedNames.add(name);
    return name;
  }
  function collect(path: string): void {
    if (seen.has(path)) return;
    seen.add(path);
    const source = path === graph.primaryPath ? graph.primarySource : graph.sources[path];
    if (source === undefined) throw new Error(`Missing source while packing: '${path}'`);
    modules.push(path);
    for (const dep of sourceDependencies(source, true) as LoadDependency[]) {
      const resolved = graph.resolutions[path]?.[dep.kind]?.[dep.source!];
      if (!resolved) throw new Error(`Unresolved dependency while packing: '${dep.source}' from '${path}'`);
      allocate(resolved, dep.kind === 'modules');
      if (dep.kind === 'modules') collect(resolved);
      else if (!Object.prototype.hasOwnProperty.call(graph.files, resolved)) throw new Error(`Missing data while packing: '${resolved}'`);
    }
  }
  collect(graph.primaryPath);
  for (const [path, name] of names) if (!seen.has(path)) files[name] = graph.files[path];
  for (const path of modules) {
    const source = path === graph.primaryPath ? graph.primarySource : graph.sources[path];
    const name = names.get(path)!;
    const starts = [0];
    for (let i = 0; i < source.length; i++) if (source[i] === '\n') starts.push(i + 1);
    const edits = (sourceDependencies(source, true) as LoadDependency[]).map(dep => {
      const resolved = graph.resolutions[path][dep.kind][dep.source!];
      const destination = names.get(resolved)!;
      const parent = name.split('/').slice(0, -1);
      const target = destination.split('/');
      while (parent.length && target.length && parent[0] === target[0]) { parent.shift(); target.shift(); }
      const relative = [...parent.map(() => '..'), ...target].join('/');
      if (relative === dep.source) return null;
      const loc = dep.argumentLoc;
      if (!loc) throw new Error('Missing source span while packing a dependency');
      return { start: starts[loc.start.line] + loc.start.col,
        end: starts[loc.end.line] + loc.end.col, text: JSON.stringify(relative) };
    }).filter(edit => edit !== null).sort((a, b) => b.start - a.start);
    let rewritten = source;
    for (const edit of edits) rewritten = rewritten.slice(0, edit.start) + edit.text + rewritten.slice(edit.end);
    files[name] = new TextEncoder().encode(rewritten);
  }
  return writeArchive(files, options);
}

module.exports = { DEFAULT_LIMITS, archiveMemberId, parseArchiveMemberId,
  readArchive, writeArchive, resolveBundleBytes, packBundle };
