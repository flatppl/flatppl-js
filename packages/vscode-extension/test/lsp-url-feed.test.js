'use strict';

// Unit tests for collectUrlSources — the host-agnostic walk that gathers the
// active model's transitive URL load_module closure from the on-disk cache,
// as the `{ sources: [{uri, text}] }` payload for the flatppl-rust LSP's
// `flatppl/urlSources` notification. Lives in src/lspUrlFeed.ts (no vscode
// import; every host dependency injected), so it is required directly with no
// VS Code host stubbing — same pattern as dependency-prefetch / lspHelpers.
//
// The transitive CLOSURE is resolveBundle's job (engine-tested): the fake walk
// here just replays the resolved paths the real walk visits, exactly as the
// dependency-prefetch tests do.

const test = require('node:test');
const assert = require('node:assert/strict');
const { collectUrlSources } = require('../src/lspUrlFeed.js');

const isUrl = (s) => /^https?:\/\//i.test(s);

// A fake engine walk: invokes readSource once per resolved dep in `deps`, in
// order — exactly what the real resolveBundle does as it walks the graph.
function fakeWalk(deps) {
  return async (_primaryPath, _primarySource, readSource) => {
    for (const d of deps) await readSource(d);
    return { sources: {} };
  };
}

// A cache-only reader keyed by url → text (absent ⇒ not cached ⇒ null).
function fakeCache(table) {
  return async (url) => (url in table ? table[url] : null);
}

test('collects cached URL deps as { uri, text } keyed by the resolved URL', async () => {
  const { sources } = await collectUrlSources({
    primaryPath: '/m.flatppl', primarySource: 'c = load_module("https://h/common.flatppl")',
    resolveBundle: fakeWalk(['https://h/common.flatppl', 'https://h/priors.flatppl']),
    isUrl,
    readUrlCached: fakeCache({
      'https://h/common.flatppl': 'pr = load_module("priors.flatppl")\nm = add(pr.theta, 1.0)',
      'https://h/priors.flatppl': 'theta = elementof(reals)',
    }),
    readLocal: async () => '',
  });
  assert.deepEqual(sources, [
    { uri: 'https://h/common.flatppl', text: 'pr = load_module("priors.flatppl")\nm = add(pr.theta, 1.0)' },
    { uri: 'https://h/priors.flatppl', text: 'theta = elementof(reals)' },
  ]);
});

test('an uncached URL is skipped and the walk continues to the rest', async () => {
  const { sources } = await collectUrlSources({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk([
      'https://h/cached.flatppl', 'https://h/uncached.flatppl', 'https://h/cached2.flatppl']),
    isUrl,
    readUrlCached: fakeCache({
      'https://h/cached.flatppl': 'a = 1',
      // uncached.flatppl absent ⇒ null ⇒ the LSP reports "source not available"
      'https://h/cached2.flatppl': 'b = 2',
    }),
    readLocal: async () => '',
  });
  assert.deepEqual(sources.map((s) => s.uri),
    ['https://h/cached.flatppl', 'https://h/cached2.flatppl']);
});

test('local deps are read (so the walk recurses) but never collected', async () => {
  let localReads = 0;
  const { sources } = await collectUrlSources({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['/abs/local.flatppl', 'https://h/a.flatppl']),
    isUrl,
    readUrlCached: fakeCache({ 'https://h/a.flatppl': 'a = 1' }),
    readLocal: async () => { localReads++; return 'y = 2'; },
  });
  assert.equal(localReads, 1, 'the local dep is read so deeper URL deps are reachable');
  assert.deepEqual(sources, [{ uri: 'https://h/a.flatppl', text: 'a = 1' }],
    'only the URL dep is fed; the LSP reads local files itself');
});

test('a URL reached more than once yields a single entry (idempotent payload)', async () => {
  const { sources } = await collectUrlSources({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['https://h/a.flatppl', 'https://h/a.flatppl']),
    isUrl,
    readUrlCached: fakeCache({ 'https://h/a.flatppl': 'a = 1' }),
    readLocal: async () => '',
  });
  assert.deepEqual(sources, [{ uri: 'https://h/a.flatppl', text: 'a = 1' }]);
});

test('a model with no URL deps produces an empty payload', async () => {
  const { sources } = await collectUrlSources({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['/abs/local.flatppl']),
    isUrl,
    readUrlCached: async () => { throw new Error('a local path must never be cache-read'); },
    readLocal: async () => 'y = 2',
  });
  assert.deepEqual(sources, []);
});

test('an unreadable local dep is tolerated and the walk continues', async () => {
  const { sources } = await collectUrlSources({
    primaryPath: '/m', primarySource: '',
    resolveBundle: fakeWalk(['/abs/missing.flatppl', 'https://h/a.flatppl']),
    isUrl,
    readUrlCached: fakeCache({ 'https://h/a.flatppl': 'a = 1' }),
    readLocal: async () => { throw new Error('ENOENT'); },
  });
  // The engine reports a bad local load_module at the call site (mirrors the
  // visualizer); it must not abort the URL feed.
  assert.deepEqual(sources, [{ uri: 'https://h/a.flatppl', text: 'a = 1' }]);
});
