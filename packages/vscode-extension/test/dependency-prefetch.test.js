'use strict';

// Unit tests for prefetchDependencies — the host-agnostic walk/counter behind
// the "FlatPPL: Download Dependencies" / "Update Dependencies" commands. Lives
// in src/dependencyPrefetch.ts (no vscode import; every host dependency is
// injected), so it is required directly with no VS Code host stubbing — same
// pattern as the lspHelpers / inferenceLens helper tests.

const test = require('node:test');
const assert = require('node:assert/strict');
const { prefetchDependencies } = require('../src/dependencyPrefetch.js');

const isUrl = (s) => /^https?:\/\//i.test(s);

// A fake engine walk: invokes readSource once per resolved dep in `deps`, in
// order — exactly what the real resolveBundle does as it walks the graph.
function fakeWalk(deps) {
  return async (_primaryPath, _primarySource, readSource) => {
    for (const d of deps) await readSource(d);
    return { sources: {} };
  };
}

// A fake fetchToCache keyed by url → { fromCache, body } | { throw }.
function fakeFetch(table) {
  return async (url) => {
    const e = table[url] || {};
    if (e.throw) throw new Error(e.throw);
    return { content: Buffer.from(e.body != null ? e.body : ''), fromCache: !!e.fromCache };
  };
}

test('counts fresh fetches and cache hits separately', async () => {
  const r = await prefetchDependencies({
    primaryPath: '/m.flatppl', primarySource: 'x = 1',
    resolveBundle: fakeWalk(['https://h/a.flatppl', 'https://h/b.flatppl']),
    isUrl,
    fetchToCache: fakeFetch({
      'https://h/a.flatppl': { fromCache: false },
      'https://h/b.flatppl': { fromCache: true },
    }),
    approve: () => true, readLocal: async () => '', force: false,
  });
  assert.deepEqual(r, { fetched: 1, cached: 1, failed: [] });
});

test('a failed URL is collected and the walk continues to the rest', async () => {
  const r = await prefetchDependencies({
    primaryPath: '/m.flatppl', primarySource: '',
    resolveBundle: fakeWalk([
      'https://h/ok.flatppl', 'https://h/bad.flatppl', 'https://h/ok2.flatppl']),
    isUrl,
    fetchToCache: fakeFetch({
      'https://h/ok.flatppl': { fromCache: false },
      'https://h/bad.flatppl': { throw: 'HTTP status 404' },
      'https://h/ok2.flatppl': { fromCache: false },
    }),
    approve: () => true, readLocal: async () => '', force: false,
  });
  assert.equal(r.fetched, 2, 'both good URLs still fetched');
  assert.equal(r.failed.length, 1);
  assert.equal(r.failed[0].url, 'https://h/bad.flatppl');
  assert.match(r.failed[0].error, /404/);
});

test('force is passed through to fetchToCache (the explicit update)', async () => {
  let seen;
  await prefetchDependencies({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['https://h/a.flatppl']),
    isUrl,
    fetchToCache: async (_u, opts) => {
      seen = opts; return { content: Buffer.from(''), fromCache: false };
    },
    approve: () => true, readLocal: async () => '', force: true,
  });
  assert.equal(seen.force, true);
});

test('local deps are read (not fetched) and are not counted', async () => {
  let fetchCalls = 0, localReads = 0;
  const r = await prefetchDependencies({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['./local.flatppl', 'https://h/a.flatppl']),
    isUrl,
    fetchToCache: async () => { fetchCalls++; return { content: Buffer.from(''), fromCache: false }; },
    approve: () => true,
    readLocal: async () => { localReads++; return 'y = 2'; },
    force: false,
  });
  assert.equal(fetchCalls, 1, 'only the URL dep is fetched');
  assert.equal(localReads, 1, 'the local dep is read');
  assert.equal(r.fetched, 1);
  assert.deepEqual(r.failed, []);
});

test('an unreadable local dep is tolerated and is NOT a remote failure', async () => {
  const r = await prefetchDependencies({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['./missing.flatppl']),
    isUrl,
    fetchToCache: async () => { throw new Error('a local path must never be fetched'); },
    approve: () => true,
    readLocal: async () => { throw new Error('ENOENT'); },
    force: false,
  });
  // Local read failures are the engine's to report at the load_module call
  // (mirrors the visualizer); `failed` is remote-fetch failures only.
  assert.deepEqual(r, { fetched: 0, cached: 0, failed: [] });
});

test('the trust-prompt approver is forwarded to fetchToCache', async () => {
  let seenApprove;
  const approve = () => true;
  await prefetchDependencies({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['https://h/a.flatppl']),
    isUrl,
    fetchToCache: async (_u, opts) => {
      seenApprove = opts.approve; return { content: Buffer.from(''), fromCache: false };
    },
    approve, readLocal: async () => '', force: false,
  });
  assert.equal(seenApprove, approve);
});
