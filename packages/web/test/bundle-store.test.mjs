import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');
const encode = text => new TextEncoder().encode(text);
const decode = bytes => new TextDecoder().decode(bytes);

function store(records, window = { FlatPPLEngine: engine }) {
  if (records) window.indexedDB = {
    open() {
      const request = { result: { transaction: () => ({ objectStore: () => ({
        getAll() {
          const read = { result: records };
          queueMicrotask(() => read.onsuccess());
          return read;
        },
      }) }) } };
      queueMicrotask(() => request.onsuccess());
      return request;
    },
  };
  else Object.defineProperty(window, 'indexedDB', { get() { throw new Error('Preview touched persistent storage'); } });
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(new URL('../src/bundle-store.ts', import.meta.url), 'utf8')),
    { window, console, TextEncoder, Uint8Array });
  return window.FlatPPLWebBundles;
}

/** The bundle store, the resolver reading through it, and the options of each graph resolution. */
function storeWithResolver() {
  const resolutions = [];
  const window = { FlatPPLEngine: { ...engine, resolveBundleBytes(...args) {
    resolutions.push(args[3]);
    return engine.resolveBundleBytes(...args);
  } } };
  const bundles = store(undefined, window);
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(new URL('../src/resolver.ts', import.meta.url), 'utf8')),
    { window, console, document: { baseURI: 'https://gallery.invalid/' }, URL, TextEncoder, TextDecoder, Uint8Array });
  return { bundles, resolver: window.FlatPPLWebResolver, resolutions };
}

test('bundle preview mounts nested scopes without storage and saves every member with its drafts', async () => {
  const inner = await engine.writeArchive({
    'wrapped/main.flatppl': encode('x=3\n'), 'wrapped/data.bin': new Uint8Array([0, 255, 128]),
    'wrapped/empty/': new Uint8Array(),
  });
  const outer = await engine.writeArchive({
    'main.flatppl': encode('m=load_module("inner.zip")\nn=load_module("sub")\n'),
    'inner.zip': inner, 'sub/main.flatppl': encode('x=4\n'),
  });
  const bundles = store();
  const opened = await bundles.open(outer, 'model.zip');
  assert.equal(opened.path, 'new/model/main.flatppl');
  const primary = bundles.canonical(opened.path);
  const graph = await engine.resolveBundleBytes(primary, await bundles.read(primary), path => bundles.read(path));
  await bundles.registerGraph(graph);
  assert.ok(bundles.list(false).some(entry => entry.path === 'new/model/inner.zip/main.flatppl'));
  assert.deepEqual(Array.from(await bundles.read('new/model/inner.zip/data.bin')), [0, 255, 128]);
  const draft = { 'new/model/inner.zip/main.flatppl': 'x=9\n', 'new/model/sub/main.flatppl': 'x=8\n' };
  const zip = await bundles.snapshot(opened.path, draft);
  const saved = await engine.readArchive(zip, { id: 'saved.zip' });
  const nested = await engine.readArchive(await saved.read('inner.zip'), { id: 'inner.zip' });
  assert.equal(nested.root, 'wrapped');
  assert.equal(decode(await nested.read('main.flatppl')), 'x=9\n');
  assert.deepEqual(Array.from(await nested.read('data.bin')), [0, 255, 128]);
  assert.equal(nested.entries.empty.directory, true);
  assert.equal(decode(await saved.read('sub/main.flatppl')), 'x=8\n');
  await nested.close(); await saved.close();
  await bundles.discard(opened.path);
  assert.deepEqual(Array.from(bundles.list(false)), []);
  assert.equal(await bundles.read(primary), null);
});

test('a draft through a symbolic link updates its target and preserves the link and empty directories', async () => {
  const bytes = await engine.writeArchive({
    'main.flatppl': encode('x=1\n'), 'alias.flatppl': encode('main.flatppl'), 'empty/': new Uint8Array(),
  }, { symlinks: ['alias.flatppl'] });
  const bundles = store();
  const opened = await bundles.open(bytes, 'links.zip');
  const drafts = { 'new/links/alias.flatppl': 'x=12\n' };
  assert.equal(bundles.draftFor(bundles.canonical(opened.path), drafts), 'x=12\n');
  const archive = await engine.readArchive(await bundles.snapshot(opened.path, drafts), { id: 'saved.zip' });
  assert.equal(decode(await archive.read('main.flatppl')), 'x=12\n');
  assert.equal(archive.entries['alias.flatppl'].symlink, true);
  assert.equal(await archive.readLink('alias.flatppl'), 'main.flatppl');
  assert.equal(archive.entries.empty.directory, true);
  await archive.close();
});

test('a shared selection restores unvisited nested and directory bundles within their roots', async () => {
  const nested = await engine.writeArchive({
    'wrapped/main.flatppl': encode('x=3\n'), 'wrapped/point name.flatppl': encode('x=4\n'),
  });
  const outer = await engine.writeArchive({
    'main.flatppl': encode('x=1\n'), 'unused.zip': nested,
    'sub/main.flatppl': encode('x=2\n'), 'sub/point.flatppl': encode('x=5\n'),
    ['deep/'.repeat(1_000) + 'point.flatppl']: encode('x=6\n'),
  });
  const bundles = store();
  const opened = await bundles.open(outer, 'selection.zip');
  const folder = bundles.folder(opened.path);
  assert.ok(!bundles.list(false).some(entry => entry.path.includes('unused.zip/')));
  const selected = await bundles.select(folder, 'unused.zip/point name.flatppl');
  assert.equal(selected, folder + '/unused.zip/point name.flatppl');
  assert.equal(decode(await bundles.read(selected)), 'x=4\n');
  assert.equal(engine.parseArchiveMemberId(bundles.canonical(selected)).path, 'wrapped/point name.flatppl');
  const directory = await bundles.select(folder, 'sub/point.flatppl');
  const member = engine.parseArchiveMemberId(bundles.canonical(directory));
  assert.equal(member.path, 'point.flatppl');
  assert.equal(engine.parseArchiveMemberId(member.archiveId).path, 'sub');
  const deep = await bundles.select(folder, 'deep/'.repeat(1_000) + 'point.flatppl');
  assert.equal(decode(await bundles.read(deep)), 'x=6\n');
  assert.equal(await bundles.select(folder, 'unused.zip/missing.flatppl'), null);
  assert.equal(await bundles.select(folder, '../main.flatppl'), null);
  await bundles.discard(opened.path);
});

test('restoring a member cannot bypass the shared archive nesting limit', async () => {
  let bytes = await engine.writeArchive({ 'main.flatppl': encode('x=1\n') });
  for (let i = 0; i < engine.DEFAULT_LIMITS.maxDepth; i++) {
    bytes = await engine.writeArchive({ 'main.flatppl': encode('x=1\n'), 'nested.zip': bytes });
  }
  const bundles = store();
  const opened = await bundles.open(bytes, 'deep.zip');
  const selection = 'nested.zip/'.repeat(engine.DEFAULT_LIMITS.maxDepth) + 'main.flatppl';
  await assert.rejects(bundles.select(bundles.folder(opened.path), selection), /nesting limit/);
  await bundles.discard(opened.path);
});

test('limits given when opening a bundle also bound its nested archives', async () => {
  const inner = await engine.writeArchive({ 'main.flatppl': encode('x=1\n'), 'a.flatppl': encode('a=1\n'),
    'b.flatppl': encode('b=1\n') });
  const outer = await engine.writeArchive({ 'main.flatppl': encode('m=load_module("inner.zip")\n'), 'inner.zip': inner });
  const bundles = store();
  const limited = await bundles.open(outer, 'limited.zip', undefined, { limits: { maxEntries: 4 } });
  await assert.rejects(bundles.select(bundles.folder(limited.path), 'inner.zip/main.flatppl'), /entry count/);
  const unlimited = await bundles.open(outer, 'unlimited.zip');
  assert.equal(await bundles.select(bundles.folder(unlimited.path), 'inner.zip/main.flatppl'),
    'new/unlimited/inner.zip/main.flatppl');
  const shallow = await bundles.open(outer, 'shallow.zip', undefined, { limits: { maxDepth: 1 } });
  await assert.rejects(bundles.select(bundles.folder(shallow.path), 'inner.zip/main.flatppl'), /nesting limit/);
  for (const opened of [limited, unlimited, shallow]) await bundles.discard(opened.path);
});

test('concurrent opens under one name get separate folders', async () => {
  const bundles = store();
  const archives = await Promise.all(['x=1\n', 'x=2\n'].map(source => engine.writeArchive({ 'main.flatppl': encode(source) })));
  const [first, second] = await Promise.all(archives.map(bytes => bundles.open(bytes, 'same.zip')));
  assert.notEqual(first.path, second.path);
  assert.equal(decode(await bundles.read(first.path)), 'x=1\n');
  assert.equal(decode(await bundles.read(second.path)), 'x=2\n');
  for (const opened of [first, second]) await bundles.discard(opened.path);
});

test('an unlisted preview stays out of the file tree and the unsaved-work warning', async () => {
  const bundles = store();
  const bytes = await engine.writeArchive({ 'main.flatppl': encode('x=1\n') });
  const hidden = await bundles.open(bytes, 'hidden.zip', undefined, { listed: false });
  assert.equal(await bundles.read(hidden.path) instanceof Uint8Array, true);
  assert.equal(bundles.list(false).length, 0);
  assert.equal(bundles.hasUnsaved(), false);
  const shown = await bundles.open(bytes, 'shown.zip');
  assert.deepEqual(Array.from(bundles.list(false), entry => entry.path), [shown.path]);
  assert.equal(bundles.hasUnsaved(), true);
  for (const opened of [hidden, shown]) await bundles.discard(opened.path);
});

test('resolving a dependency graph applies the limits of its bundle', async () => {
  const inner = await engine.writeArchive({ 'main.flatppl': encode('x=1\n') });
  const outer = await engine.writeArchive({ 'main.flatppl': encode('m=load_module("inner.zip")\n'), 'inner.zip': inner });
  const { bundles, resolver, resolutions } = storeWithResolver();
  const limits = { maxEntries: 9, maxPathBytes: 64 };
  const limited = await bundles.open(outer, 'limited.zip', undefined, { limits });
  const unlimited = await bundles.open(outer, 'unlimited.zip');
  // Graph resolution reads nested archives itself, so it needs the same bounds as the mount.
  for (const opened of [limited, unlimited]) {
    const graph = await resolver.resolveBundle(opened.path, undefined, { strict: true });
    assert.ok(Object.keys(graph.sources).some(path => path.includes('inner.zip')));
  }
  assert.deepEqual(resolutions.map(options => options.limits), [limits, undefined]);
});

function router(bundles) {
  const window = { FlatPPLWebBundles: bundles, location: { hash: '' } };
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(new URL('../src/router.ts', import.meta.url), 'utf8')),
    { window, console });
  return { window, ...window.FlatPPLWebRouter };
}

test('routes retain content or hosted origins only within their unsaved bundle', async () => {
  const bundles = store();
  const bytes = await engine.writeArchive({ 'main.flatppl': encode('x=1\n'), 'other.flatppl': encode('x=2\n') });
  const first = await bundles.open(bytes, 'first.zip');
  const second = await bundles.open(bytes, 'second.zip');
  const r = router({ ...bundles, isSaved: path => path.startsWith('user/') || bundles.isSaved(path),
    folder: path => path?.startsWith('user/saved/') ? 'user/saved' : bundles.folder(path) });
  for (const origin of [{ share: 'br1.snapshot' }, { archive: 'https://example.org/model.zip?revision=2' }]) {
    r.window.location.hash = r.serialize({ model: first.path, ...origin });
    r.navigateTo({ model: 'new/first/other.flatppl', target: 'x' });
    const same = r.parseHash();
    assert.equal(same.model, 'new/first/other.flatppl');
    assert.equal(same.target, 'x');
    for (const [key, value] of Object.entries(origin)) assert.equal(same[key], value);
    r.navigateTo({ model: second.path });
    assert.equal(r.parseHash().share, undefined);
    assert.equal(r.parseHash().archive, undefined);
    r.navigateTo({ model: 'user/saved/main.flatppl', ...origin });
    assert.equal(r.parseHash().share, undefined);
    assert.equal(r.parseHash().archive, undefined);
  }
  await bundles.discard(first.path); await bundles.discard(second.path);
});

test('share selection parameters decode URLSearchParams spaces and retain literal plus signs', () => {
  const r = router();
  const selection = new URLSearchParams({ share: 'br1.snapshot', model: 'a folder/model+1.flatppl' });
  r.window.location.hash = '#' + selection;
  assert.equal(r.parseHash().model, 'a folder/model+1.flatppl');
});

test('URL inputs enforce the link budget and a single value per routing field', () => {
  const r = router();
  for (const hash of [
    '#share=' + 'A'.repeat(64 * 1024),
    '#share=first&share=second',
    '#/files/a.flatppl?model=b.flatppl',
    '#share=first&archive=https%3A%2F%2Fexample.org%2Fa.zip',
    '#model=%XX',
  ]) {
    r.window.location.hash = hash;
    assert.ok(r.parseHash().error);
  }
  r.window.location.hash = '#/files/caf%C3%A9.flatppl?target=x';
  assert.equal(r.parseHash().model, 'café.flatppl');
  assert.equal(r.parseHash().target, 'x');
});

test('incoming and outgoing link limits preserve local ZIP support', async () => {
  const bundles = store();
  const long = 'a/'.repeat(33) + 'x.txt';
  const deep = await engine.writeArchive({ 'main.flatppl': encode('x=1'), [long]: encode('data') });
  await assert.rejects(bundles.validateLink(deep));
  await assert.rejects(bundles.open(deep, 'deep.zip', undefined, { limits: bundles.LINK_LIMITS }));
  assert.deepEqual(Array.from(bundles.list(false)), []);
  const local = await bundles.open(deep, 'deep.zip');
  assert.equal(decode(await bundles.read(bundles.folder(local.path) + '/' + long)), 'data');
  await bundles.discard(local.path);
  const large = await engine.writeArchive({
    'main.flatppl': encode('x=1'), 'data.txt': new Uint8Array(2 * 1024 ** 2 + 1),
  }, { compression: 'deflate' });
  await assert.rejects(bundles.validateLink(large));
  await assert.rejects(bundles.open(large, 'large.zip', undefined, { limits: bundles.LINK_LIMITS }));
  assert.deepEqual(Array.from(bundles.list(false)), []);
});

test('nested archives retain the URL preview budgets and accept ordinary Deflate', async () => {
  const bundles = store();
  const nested = await engine.writeArchive({
    'main.flatppl': encode('x=2'), 'data.txt': new Uint8Array(2 * 1024 ** 2 + 1),
  }, { compression: 'deflate' });
  const outer = await engine.writeArchive({ 'main.flatppl': encode('x=1'), 'nested.zip': nested },
    { compression: 'deflate' });
  await bundles.validateLink(outer);
  assert.equal(bundles.list(false).length, 0);
  const opened = await bundles.open(outer, 'outer.zip', undefined, { limits: bundles.LINK_LIMITS });
  assert.equal(decode(await bundles.read(opened.path)), 'x=1');
  await assert.rejects(bundles.select(bundles.folder(opened.path), 'nested.zip/main.flatppl'));
  assert.equal(bundles.list(false).length, 2);
  await bundles.discard(opened.path);
});

test('reusing a saved bundle replaces nested readers that had local upload limits', async () => {
  const nested = await engine.writeArchive({
    'main.flatppl': encode('x=2'), 'data.bin': new Uint8Array(2 * 1024 ** 2 + 1),
  }, { compression: 'deflate' });
  const bytes = await engine.writeArchive({ 'main.flatppl': encode('x=1'), 'nested.zip': nested });
  const bundles = store([{ folder: 'user/saved', bytes }]);
  await bundles.init();
  assert.equal(await bundles.select('user/saved', 'nested.zip/main.flatppl'), 'user/saved/nested.zip/main.flatppl');
  const reopened = await bundles.open(bytes, 'shared', undefined, { limits: bundles.LINK_LIMITS });
  assert.equal(reopened.reused, true);
  assert.equal(reopened.path, 'user/saved/main.flatppl');
  await assert.rejects(bundles.select('user/saved', 'nested.zip/main.flatppl'));
});

test('saved reuse accepts a smaller ZIP representation of the same files', async () => {
  const files = { 'main.flatppl': encode('x=1'), 'data.bin': new Uint8Array(8192) };
  const stored = await engine.writeArchive(files, { compression: 'store' });
  const compressed = await engine.writeArchive(files, { compression: 'deflate' });
  assert.ok(stored.length > compressed.length);
  const bundles = store([{ folder: 'user/saved', bytes: stored }]);
  await bundles.init();
  const opened = await bundles.open(compressed, 'shared', undefined,
    { limits: { ...bundles.LINK_LIMITS, maxArchiveBytes: compressed.length } });
  assert.equal(opened.reused, true);
  assert.equal(decode(await bundles.read(opened.path)), 'x=1');
  assert.equal((await bundles.read('user/saved/data.bin')).length, files['data.bin'].length);
});
