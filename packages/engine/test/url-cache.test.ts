'use strict';

// §04 "Remote file caching" — the Node host cache primitive (url-cache.ts).
// Spec: flatppl-design/docs/04-design.md #sec:url-cache (committed b53d013).
// One shared local http server (the spec's recommended approach) with a path
// per case, plus a fresh tmp cache dir per test and injected now/cacheDir, so
// nothing touches the real network, the real cache, or process.env.

const { test, before, after } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');

const { isUrl, resolveCacheDir, urlKey, cachePaths, isTrusted, fetchToCache } = require('../url-cache.ts');

function sha256hex(s: string) { return crypto.createHash('sha256').update(s, 'utf8').digest('hex'); }
async function tmpCacheDir() { return await fsp.mkdtemp(path.join(os.tmpdir(), 'flatppl-cache-')); }

// One http server for the whole file; routes keyed by path, hit-counted.
const ROUTES: any = {
  '/miss.flatppl': { body: 'x = 1', contentType: 'text/plain', etag: '"abc"' },
  '/hit.flatppl': { body: 'x = 1' },
  '/r-orig.flatppl': { redirect: '/r-real.flatppl' },
  '/r-real.flatppl': { body: 'x = 2' },
  '/e404.flatppl': { status: 404, body: 'nope' },
  '/off.flatppl': { body: 'x = 1' },
  '/untrusted.flatppl': { body: 'x = 1' },
  '/approve.flatppl': { body: 'x = 1' },
  '/trustall.flatppl': { body: 'x = 1' },
  '/force.flatppl': { body: 'x = 1' },
};
let SRV: any;
before(async () => {
  SRV = await new Promise((resolve: any) => {
    const hits: any = {};
    const server = http.createServer((req: any, res: any) => {
      hits[req.url] = (hits[req.url] || 0) + 1;
      const r = ROUTES[req.url];
      if (!r) { res.statusCode = 404; res.end('not found'); return; }
      if (r.redirect) { res.statusCode = 302; res.setHeader('location', r.redirect); res.end(); return; }
      res.statusCode = r.status || 200;
      if (r.contentType) res.setHeader('content-type', r.contentType);
      if (r.etag) res.setHeader('etag', r.etag);
      res.end(r.body == null ? '' : r.body);
    });
    server.listen(0, '127.0.0.1', () => {
      resolve({ base: 'http://127.0.0.1:' + server.address().port, hits, close: () => new Promise((c: any) => server.close(c)) });
    });
  });
});
after(async () => { if (SRV) await SRV.close(); });

// ---------------------------------------------------------------- cache dir
test('resolveCacheDir: FLATPPL_CACHEDIR overrides everything', () => {
  assert.equal(resolveCacheDir({ env: { FLATPPL_CACHEDIR: '/custom/cache' }, platform: 'linux', homedir: '/home/u' }),
    '/custom/cache');
});
test('resolveCacheDir: linux uses $XDG_CACHE_HOME/flatppl', () => {
  assert.equal(resolveCacheDir({ env: { XDG_CACHE_HOME: '/x/cache' }, platform: 'linux', homedir: '/home/u' }),
    path.join('/x/cache', 'flatppl'));
});
test('resolveCacheDir: linux default is $HOME/.cache/flatppl', () => {
  assert.equal(resolveCacheDir({ env: {}, platform: 'linux', homedir: '/home/u' }),
    path.join('/home/u', '.cache', 'flatppl'));
});
test('resolveCacheDir: macOS uses ~/Library/Caches/flatppl', () => {
  assert.equal(resolveCacheDir({ env: {}, platform: 'darwin', homedir: '/Users/u' }),
    path.join('/Users/u', 'Library', 'Caches', 'flatppl'));
});
test('resolveCacheDir: windows uses %LOCALAPPDATA%\\flatppl', () => {
  assert.equal(resolveCacheDir({ env: { LOCALAPPDATA: 'C:\\Users\\u\\AppData\\Local' }, platform: 'win32', homedir: 'C:\\Users\\u' }),
    path.join('C:\\Users\\u\\AppData\\Local', 'flatppl'));
});

// ------------------------------------------------------------------ keys
test('urlKey: key is the lowercase-hex SHA-256 of the fragment-stripped URL', () => {
  const url = 'https://h.example/a/model.flatppl';
  const { key, kk } = urlKey(url);
  assert.equal(key, sha256hex(url));
  assert.equal(kk, key.slice(0, 2));
});
test('urlKey: the #fragment is removed before hashing; the query is kept', () => {
  const base = 'https://h.example/a/model.flatppl?v=2';
  assert.equal(urlKey(base + '#frag').key, sha256hex(base));
  assert.notEqual(urlKey(base).key, urlKey('https://h.example/a/model.flatppl').key);
});
test('urlKey: ext is the full trailing extension (after the first dot of the final segment), query ignored', () => {
  assert.equal(urlKey('https://h/a/model.flatppl.zip').ext, 'flatppl.zip');
  assert.equal(urlKey('https://h/a/model.flatppl').ext, 'flatppl');
  assert.equal(urlKey('https://h/a/model.flatppl.zip?x=1').ext, 'flatppl.zip');
});
test('urlKey: a final segment with no dot yields an empty ext', () => {
  assert.equal(urlKey('https://h/a/model').ext, '');
});
test('isUrl recognizes http/https and rejects local paths / file://', () => {
  assert.ok(isUrl('http://h/x') && isUrl('https://h/x'));
  assert.ok(!isUrl('./local.flatppl') && !isUrl('/abs/local.flatppl') && !isUrl('file:///x'));
});

// ----------------------------------------------------------- fetch & cache
test('fetchToCache: a miss fetches and writes the object + metadata', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/miss.flatppl';
  const r = await fetchToCache(url, { cacheDir, trustAll: true, now: () => '2026-06-24T00:00:00Z' });
  assert.equal(r.content.toString('utf8'), 'x = 1');
  assert.equal(r.fromCache, false);
  const p = cachePaths(cacheDir, url);
  assert.ok(fs.existsSync(p.object) && fs.existsSync(p.meta));
  const meta = JSON.parse(fs.readFileSync(p.meta, 'utf8'));
  assert.equal(meta.url, url);
  assert.equal(meta.content_type, 'text/plain');
  assert.equal(meta.etag, '"abc"');
  assert.equal(meta.retrieved, '2026-06-24T00:00:00Z');
});

test('fetchToCache: a cache hit is never revalidated against the network', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/hit.flatppl';
  await fetchToCache(url, { cacheDir, trustAll: true });
  const r2 = await fetchToCache(url, { cacheDir, trustAll: true });
  assert.equal(r2.fromCache, true);
  assert.equal(r2.content.toString('utf8'), 'x = 1');
  assert.equal(SRV.hits['/hit.flatppl'], 1, 'only one network hit');
});

test('fetchToCache: redirects are followed; resolved_url differs but the key is the original URL', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/r-orig.flatppl';
  const r = await fetchToCache(url, { cacheDir, trustAll: true });
  assert.equal(r.content.toString('utf8'), 'x = 2');
  assert.equal(r.meta.url, url);
  assert.equal(r.meta.resolved_url, SRV.base + '/r-real.flatppl');
  assert.ok(fs.existsSync(cachePaths(cacheDir, url).object), 'object keyed by the original URL');
});

test('fetchToCache: a non-2xx final response errors and writes nothing', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/e404.flatppl';
  await assert.rejects(() => fetchToCache(url, { cacheDir, trustAll: true }), /404|status/i);
  const p = cachePaths(cacheDir, url);
  assert.ok(!fs.existsSync(p.object) && !fs.existsSync(p.meta), 'no partial object or metadata');
});

test('fetchToCache: offline + miss is an error, no fetch attempted', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/off.flatppl';
  await assert.rejects(() => fetchToCache(url, { cacheDir, offline: true, trustAll: true }), /offline|cache miss/i);
  assert.equal(SRV.hits['/off.flatppl'], undefined, 'no network hit');
});

test('fetchToCache: an untrusted URL with no approver errors and fetches nothing', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/untrusted.flatppl';
  await assert.rejects(() => fetchToCache(url, { cacheDir }), /trust|untrusted|approv/i);
  assert.equal(SRV.hits['/untrusted.flatppl'], undefined, 'no fetch before trust');
});

test('fetchToCache: approval creates a per-URL trust marker (trusted next time)', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/approve.flatppl';
  let prompts = 0;
  await fetchToCache(url, { cacheDir, approve: () => { prompts++; return true; } });
  assert.equal(prompts, 1);
  assert.ok(await isTrusted(cacheDir, url));
  assert.ok(fs.existsSync(cachePaths(cacheDir, url).trust));
});

test('fetchToCache: FLATPPL_TRUST (trustAll) fetches but creates NO trust marker', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/trustall.flatppl';
  await fetchToCache(url, { cacheDir, trustAll: true });
  assert.ok(!fs.existsSync(cachePaths(cacheDir, url).trust), 'no marker under trustAll');
});

test('fetchToCache: opts.force re-fetches and replaces even on a cache hit (update)', async () => {
  const cacheDir = await tmpCacheDir();
  const url = SRV.base + '/force.flatppl';
  await fetchToCache(url, { cacheDir, trustAll: true });
  assert.equal(SRV.hits['/force.flatppl'], 1);
  const r2 = await fetchToCache(url, { cacheDir, trustAll: true, force: true });
  assert.equal(r2.fromCache, false);
  assert.equal(SRV.hits['/force.flatppl'], 2, 'force triggers a fresh fetch');
});
