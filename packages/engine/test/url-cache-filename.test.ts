'use strict';

// Spec §04 Remote file caching, Layout and keys: suffix fallback is part
// of the shared on-disk format. Exercise fetch and offline reuse, not only
// the key helper. The injected fetch never opens a network connection.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { fetchToCache, cachePaths, urlKey } = require('../url-cache.ts');

test('URL cache retains valid suffixes and falls back for long or invalid filenames', async () => {
  const cacheDir = await fs.mkdtemp(path.join(os.tmpdir(), 'flatppl-cache-filenames-'));
  try {
    for (const [suffix, retained] of [
      ['x'.repeat(300), ''], ['bad%2Fext', ''], ['x'.repeat(32), 'x'.repeat(32)],
    ]) {
      const url = 'https://example.invalid/model.' + suffix;
      const body = 'source for ' + suffix;
      await fetchToCache(url, { cacheDir, trustAll: true, env: {},
        fetchImpl: async () => new Response(body) });
      const expectedName = urlKey(url).key + (retained ? '.' + retained : '');
      assert.equal(path.basename(cachePaths(cacheDir, url).object), expectedName);
      assert.equal((await fetchToCache(url, { cacheDir, offline: true, env: {} })).content.toString(), body);
    }
  } finally {
    await fs.rm(cacheDir, { recursive: true, force: true });
  }
});
