import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire, stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

const require = createRequire(import.meta.url);
const engine = require('../../engine/index.ts');
const encode = text => new TextEncoder().encode(text);
const decode = bytes => new TextDecoder().decode(bytes);

function store() {
  const window = { FlatPPLEngine: engine };
  Object.defineProperty(window, 'indexedDB', { get() { throw new Error('Preview touched persistent storage'); } });
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(new URL('../src/bundle-store.ts', import.meta.url), 'utf8')),
    { window, console, TextEncoder, Uint8Array });
  return window.FlatPPLWebBundles;
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
