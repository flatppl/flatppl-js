'use strict';

// §04 "Remote file caching": the cache is shared with flatppl-rust, which
// refuses a cache directory another user could change. These tests pin the
// same rules here: private modes whatever the umask, directories checked from
// the root down, and no URL userinfo on disk or in a message.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const fsp = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');

const { fetchToCache, cachePaths, urlKey, redactUrlUserinfo } = require('../url-cache.ts');

const POSIX = process.platform !== 'win32';
const URL_ = 'https://example.test/models/m.flatppl';

async function tmpDir() { return await fsp.mkdtemp(path.join(os.tmpdir(), 'flatppl-cache-privacy-')); }
function mode(p: string) { return fs.statSync(p).mode & 0o777; }

// A network-free fetch that counts its calls.
function counting(body = 'x = 1') {
  const calls: string[] = [];
  const fetchImpl = async (url: string) => { calls.push(url); return new Response(body); };
  return { calls, fetchImpl };
}

test('new cache directories are 0700 and files 0600 under umask 002', { skip: !POSIX }, async () => {
  const parent = await tmpDir();
  const previous = process.umask(0o002);
  try {
    fs.chmodSync(parent, 0o777);
    const root = path.join(parent, 'cache');
    await fetchToCache(URL_, { cacheDir: root, env: {}, approve: () => true, fetchImpl: counting().fetchImpl });
    const p = cachePaths(root, URL_);
    for (const dir of [root, path.join(root, 'v1'), path.join(root, 'v1', 'tmp'), path.dirname(p.object),
      path.join(root, 'v1', 'objects'), path.join(root, 'v1', 'trust'), path.dirname(p.trust)]) {
      assert.equal(mode(dir), 0o700, dir);
    }
    for (const file of [p.object, p.meta, p.trust]) assert.equal(mode(file), 0o600, file);

    // A pre-existing root that only its owner can write is used as it is.
    const existing = path.join(parent, 'existing');
    fs.mkdirSync(existing);
    fs.chmodSync(existing, 0o755);
    await fetchToCache(URL_, { cacheDir: existing, env: {}, trustAll: true, fetchImpl: counting().fetchImpl });
    assert.equal(mode(existing), 0o755);
    assert.equal(mode(path.join(existing, 'v1')), 0o700);
  } finally {
    process.umask(previous);
    await fsp.rm(parent, { recursive: true, force: true });
  }
});

test('a root other users can write is refused and left unchanged', { skip: !POSIX }, async () => {
  const root = await tmpDir();
  try {
    fs.chmodSync(root, 0o777);
    const { calls, fetchImpl } = counting();
    await assert.rejects(fetchToCache(URL_, { cacheDir: root, env: {}, trustAll: true, fetchImpl }),
      /cache directory .* is writable by other users/);
    assert.equal(calls.length, 0);
    assert.deepEqual(fs.readdirSync(root), []);
    assert.equal(mode(root), 0o777);
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
  }
});

test('a planted shard symlink is neither a hit nor a publish target', { skip: !POSIX }, async () => {
  const root = await tmpDir();
  const outside = await tmpDir();
  try {
    const p = cachePaths(root, URL_);
    fs.writeFileSync(path.join(outside, path.basename(p.object)), 'planted');
    fs.mkdirSync(path.dirname(path.dirname(p.object)), { recursive: true });
    fs.symlinkSync(outside, path.dirname(p.object));
    const { fetchImpl } = counting('network bytes');
    for (const force of [false, true]) {
      await assert.rejects(fetchToCache(URL_, { cacheDir: root, env: {}, trustAll: true, force, fetchImpl }),
        /cache directory .* is a symbolic link/);
    }
    assert.deepEqual(fs.readdirSync(outside), [path.basename(p.object)]);
    assert.equal(fs.readFileSync(path.join(outside, path.basename(p.object)), 'utf8'), 'planted');
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
    await fsp.rm(outside, { recursive: true, force: true });
  }
});

test('a cache object symlink is not a hit and does not bypass trust', { skip: !POSIX }, async () => {
  const root = await tmpDir();
  try {
    const p = cachePaths(root, URL_);
    fs.mkdirSync(path.dirname(p.object), { recursive: true, mode: 0o700 });
    for (const dir of [root, path.join(root, 'v1'), path.join(root, 'v1', 'objects'), path.dirname(p.object)]) {
      fs.chmodSync(dir, 0o700);
    }
    const planted = path.join(root, 'planted.flatppl');
    fs.writeFileSync(planted, 'attacker bytes');
    fs.symlinkSync(planted, p.object);
    const { calls, fetchImpl } = counting();
    await assert.rejects(fetchToCache(URL_, { cacheDir: root, env: {}, fetchImpl }), /not a trusted URL/);
    assert.equal(calls.length, 0);
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
  }
});

test('a root reached through the user\'s own symlink stays usable', { skip: !POSIX }, async () => {
  const parent = await tmpDir();
  try {
    const target = path.join(parent, 'target');
    fs.mkdirSync(target, { mode: 0o700 });
    const link = path.join(parent, 'link');
    fs.symlinkSync(target, link);
    await fetchToCache(URL_, { cacheDir: link, env: {}, trustAll: true, fetchImpl: counting().fetchImpl });
    assert.ok(fs.statSync(path.join(target, 'v1', 'objects')).isDirectory());
  } finally {
    await fsp.rm(parent, { recursive: true, force: true });
  }
});

test('userinfo URLs are refused before trust or fetch, and no message prints the password', async () => {
  const root = await tmpDir();
  try {
    const { calls, fetchImpl } = counting();
    const approved: string[] = [];
    for (const url of [
      'https://user:secret@example.test/m.flatppl',
      'https://us er:secret@example.test/m.flatppl',
      'https://user:se\tcret@example.test/m.flatppl',
      'https://user:p%40secret@example.test/m.flatppl',
      'https://a@b:secret@example.test/m.flatppl',
      'https://user:secret@x@example.test/m.flatppl',
      'https://evil.test\\@example.test/m.flatppl',
    ]) {
      const err: any = await fetchToCache(url, { cacheDir: root, env: {},
        approve: (u: string) => { approved.push(u); return true; }, fetchImpl })
        .then(() => null, (e: any) => e);
      assert.ok(err, url);
      assert.match(err.message, /userinfo/);
      assert.ok(!err.message.includes('secret') && !err.message.includes('se\tcret'), err.message);
    }
    assert.deepEqual(calls, []);
    assert.deepEqual(approved, []);
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
  }
});

test('a redirect to a userinfo URL is refused and leaves no credentials on disk', async () => {
  const root = await tmpDir();
  try {
    const fetchImpl = async () => new Response(null, { status: 302,
      headers: { location: 'https://user:secret@other.test/m.flatppl' } });
    const err: any = await fetchToCache(URL_, { cacheDir: root, env: {}, approve: () => true, fetchImpl })
      .then(() => null, (e: any) => e);
    assert.match(err.message, /userinfo/);
    assert.ok(!err.message.includes('secret'), err.message);
    assert.equal(fs.existsSync(path.join(root, 'v1', 'objects')), false);
    const files: string[] = fs.readdirSync(root, { recursive: true });
    for (const f of files) {
      const full = path.join(root, f);
      if (fs.statSync(full).isFile()) assert.ok(!fs.readFileSync(full, 'utf8').includes('secret'), full);
    }
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
  }
});

test('an @ after the authority is path or query text, not userinfo', async () => {
  const root = await tmpDir();
  try {
    const url = 'https://example.test/a@b.flatppl?k=@v';
    await fetchToCache(url, { cacheDir: root, env: {}, trustAll: true, fetchImpl: counting().fetchImpl });
    assert.ok(fs.existsSync(cachePaths(root, url).object));
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
  }
});

test('a failed publish removes its temporary file', async () => {
  const root = await tmpDir();
  try {
    const p = cachePaths(root, URL_);
    await fetchToCache(URL_, { cacheDir: root, env: {}, trustAll: true, fetchImpl: counting('old').fetchImpl });
    fs.rmSync(p.object);
    fs.mkdirSync(path.join(p.object, 'occupied'), { recursive: true });
    await assert.rejects(fetchToCache(URL_, { cacheDir: root, env: {}, trustAll: true, force: true,
      fetchImpl: counting('new').fetchImpl }));
    assert.deepEqual(fs.readdirSync(path.join(root, 'v1', 'tmp')), []);
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
  }
});

test('a network error message hides userinfo it quotes', async () => {
  const root = await tmpDir();
  try {
    const fetchImpl = async () => { throw new Error('proxy failed at http://proxy-user:secret@proxy.test/'); };
    const err: any = await fetchToCache(URL_, { cacheDir: root, env: {}, trustAll: true, fetchImpl })
      .then(() => null, (e: any) => e);
    assert.match(err.message, /network error .*<redacted>@proxy\.test/);
    assert.ok(!err.message.includes('secret'), err.message);
  } finally {
    await fsp.rm(root, { recursive: true, force: true });
  }
});

test('redaction hides every authority userinfo and keeps the rest', () => {
  assert.equal(redactUrlUserinfo('at http://io-user:io-secret@host.test/a@b then https://safe.test/x'),
    'at http://<redacted>@host.test/a@b then https://safe.test/x');
  assert.equal(redactUrlUserinfo('https://us er:secret@host.test/m'), 'https://<redacted>@host.test/m');
  assert.equal(redactUrlUserinfo('https://host.test/a@b?keep=@v'), 'https://host.test/a@b?keep=@v');
});

test('unsafe extensions store the object under the bare key', () => {
  for (const url of ['https://h/x.a%5C..%5Coutside.flatppl', 'https://h/x.' + 'a'.repeat(33)]) {
    assert.equal(urlKey(url).ext, '', url);
    assert.equal(path.basename(cachePaths('/c', url).object), urlKey(url).key, url);
  }
});
