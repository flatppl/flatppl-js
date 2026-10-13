import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';
import { syncExampleCollections } from '../build-files.mjs';

const collections = [{ path: 'examples', title: 'Examples' }, { path: 'posteriordb', title: 'PosteriorDB' }];

async function put(path, text) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, text);
}

async function corpus(t) {
  const root = await mkdtemp(join(tmpdir(), 'flatppl-collections-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const siblingDir = join(root, 'source');
  const distDir = join(root, 'dist');
  await put(join(siblingDir, 'collections.json'), JSON.stringify(collections));
  await put(join(siblingDir, 'examples/model.flatppl'), 'x = 1');
  await put(join(siblingDir, 'posteriordb/group/model.flatppl'), 'm = load_module("shared.flatppl")\nx = m.x');
  await put(join(siblingDir, 'posteriordb/group/shared.flatppl'), 'x = 2');
  await put(join(siblingDir, 'posteriordb/group/README.md'), '# Group');
  await put(join(siblingDir, 'posteriordb/group/data.json'), '[1, 2, 3]');
  await put(join(siblingDir, 'private/model.flatppl'), 'x = 99');
  await put(join(distDir, 'vendor/engine.js'), 'web asset');
  return { root, siblingDir, distDir, archiveURL: 'https://example.invalid/examples.tar.gz', log() {} };
}

test('sibling and archive builds publish the same listed models and support files', async t => {
  const options = await corpus(t);
  const archive = join(options.root, 'source.tar.gz');
  execFileSync('tar', ['-czf', archive, '-C', options.root, 'source']);
  const requests = [];
  const bytes = await readFile(archive);
  t.mock.method(globalThis, 'fetch', async url => {
    requests.push(url);
    return new Response(bytes);
  });
  const local = await syncExampleCollections(options);
  assert.deepEqual(requests, []);
  assert.deepEqual(local.collections, collections);
  assert.deepEqual(local.entries.map(entry => entry.path), [
    'examples/model.flatppl', 'posteriordb/group/README.md',
    'posteriordb/group/data.json',
    'posteriordb/group/model.flatppl', 'posteriordb/group/shared.flatppl',
  ]);
  const remoteDir = join(options.root, 'remote-dist');
  const remote = await syncExampleCollections({ ...options, siblingDir: join(options.root, 'absent'), distDir: remoteDir });
  assert.deepEqual(remote, local);
  assert.deepEqual(requests, [options.archiveURL]);
  for (const root of [options.distDir, remoteDir]) {
    assert.equal(await readFile(join(root, 'posteriordb/group/data.json'), 'utf8'), '[1, 2, 3]');
    assert.equal(await readFile(join(root, 'posteriordb/group/model.flatppl'), 'utf8'), 'm = load_module("shared.flatppl")\nx = m.x');
    assert.equal(existsSync(join(root, 'private')), false);
  }
});

test('a full rebuild removes deleted models and collections but keeps web assets', async t => {
  const options = await corpus(t);
  await put(join(options.distDir, 'models.json'), JSON.stringify(await syncExampleCollections(options)));
  await rm(join(options.siblingDir, 'posteriordb/group/shared.flatppl'));
  const edited = await syncExampleCollections(options);
  assert.equal(edited.entries.some(entry => entry.path.endsWith('shared.flatppl')), false);
  assert.equal(existsSync(join(options.distDir, 'posteriordb/group/shared.flatppl')), false);
  await put(join(options.distDir, 'models.json'), JSON.stringify(edited));
  await put(join(options.siblingDir, 'collections.json'), JSON.stringify(collections.slice(0, 1)));
  // Watch startup changes the visible manifest but retains watched directories.
  await put(join(options.distDir, 'models.json'), JSON.stringify(await syncExampleCollections({ ...options, clean: false })));
  const reduced = await syncExampleCollections(options);
  assert.deepEqual(reduced.entries.map(entry => entry.path), ['examples/model.flatppl']);
  assert.equal(existsSync(join(options.distDir, 'posteriordb')), false);
  assert.equal(await readFile(join(options.distDir, 'vendor/engine.js'), 'utf8'), 'web asset');
});

test('invalid catalogs cannot overwrite assets or partially publish a collection', async t => {
  const options = await corpus(t);
  await put(join(options.distDir, 'examples/model.flatppl'), 'previous example');
  // Each input exercises a different publication boundary, before any copy or prune.
  for (const invalid of [
    {},
    [collections[0], { path: '../escape', title: 'Escape' }],
    [collections[0], { path: 'vendor', title: 'Assets' }],
    [collections[0], { path: 'EXAMPLES', title: 'Duplicate' }],
    [collections[0], { path: 'posteriordb', title: ' ' }],
    [collections[1]],
    [collections[0], { path: 'absent', title: 'Missing directory' }],
  ]) {
    await put(join(options.siblingDir, 'collections.json'), JSON.stringify(invalid));
    await assert.rejects(syncExampleCollections(options));
    assert.equal(await readFile(join(options.distDir, 'examples/model.flatppl'), 'utf8'), 'previous example');
    assert.equal(await readFile(join(options.distDir, 'vendor/engine.js'), 'utf8'), 'web asset');
    assert.equal(existsSync(join(options.distDir, 'posteriordb')), false);
  }
});
