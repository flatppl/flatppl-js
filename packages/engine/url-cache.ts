'use strict';

// url-cache.ts — Node host primitive implementing the spec §04 "Remote file
// caching" convention (flatppl-design/docs/04-design.md #sec:url-cache).
//
// NODE-ONLY. Uses node:fs/promises, node:crypto and global fetch, so it is NOT
// part of the browser engine bundle — it must never be imported by
// engine/index.ts. Node hosts (the VS Code extension; a future CLI) use it to
// fetch `http`/`https` `load_module` / `load_data` sources into a shared
// on-disk cache. The browser host cannot use an on-disk cache and fetches
// directly (its readSource lives in packages/web), limited by CORS.
//
// The cache is keyed by the SHA-256 of the request URL (fragment stripped);
// there is no index. It is lock-free: every write goes through a temp file +
// atomic rename, the metadata is renamed before its content object (so a
// present object always has metadata), and trust markers are created with an
// exclusive open (O_CREAT | O_EXCL).
//
// The cache is shared with flatppl-rust (crates/fileaccess), which refuses a
// cache directory that another user could change. The same rules hold here:
// new directories are 0700 and files 0600 whatever the umask, every cache
// directory is checked from the root down before use, and a URL with userinfo
// is refused, so no credentials reach the disk or a message.

const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
// isUrl is owned by module-resolve (the pure resolution module); re-exported
// from here so Node hosts can get it alongside the cache helpers.
const { isUrl } = require('./module-resolve.ts');

const LAYOUT_VERSION = 'v1';

// The cache directory (spec "Cache directory"): FLATPPL_CACHEDIR wins; else the
// per-platform default. env / platform / homedir are injectable for testing.
function resolveCacheDir(opts: any): string {
  opts = opts || {};
  const env = opts.env || process.env;
  const platform = opts.platform || process.platform;
  const home = opts.homedir || os.homedir();
  if (env.FLATPPL_CACHEDIR) return env.FLATPPL_CACHEDIR;
  if (platform === 'win32') {
    const localApp = env.LOCALAPPDATA || path.join(env.USERPROFILE || home, 'AppData', 'Local');
    return path.join(localApp, 'flatppl');
  }
  if (platform === 'darwin') return path.join(home, 'Library', 'Caches', 'flatppl');
  // Linux / BSD: $XDG_CACHE_HOME/flatppl (when absolute), else $HOME/.cache/flatppl.
  const xdg = env.XDG_CACHE_HOME;
  const base = (xdg && path.isAbsolute(xdg)) ? xdg : path.join(home, '.cache');
  return path.join(base, 'flatppl');
}

// The length of the authority at the start of `rest`, the text after a
// `scheme://`. It ends at the first `/`, `?` or `#`. Whitespace and a
// backslash do not end it, so the refusal and the redaction see one span.
function authorityLen(rest: string): number {
  const m = /[/?#]/.exec(rest);
  return m ? m.index : rest.length;
}

function hasUserinfo(url: string): boolean {
  const at = url.indexOf('://');
  if (at < 0) return false;
  const rest = url.slice(at + 3);
  return rest.slice(0, authorityLen(rest)).includes('@');
}

// Redact the authority userinfo of every URL in `text`, a display-only copy.
// The last `@` in an authority ends its userinfo. Text that only looks like an
// authority may be over-redacted, never under-redacted.
function redactUrlUserinfo(text: string): string {
  let out = '';
  let copied = 0;
  let from = 0;
  for (;;) {
    const scheme = text.indexOf('://', from);
    if (scheme < 0) break;
    const start = scheme + 3;
    const end = start + authorityLen(text.slice(start));
    const at = text.lastIndexOf('@', end - 1);
    if (at >= start) {
      out += text.slice(copied, start) + '<redacted>@';
      copied = at + 1;
    }
    from = end;
  }
  return out + text.slice(copied);
}

// Refuse a URL the cache must not fetch. RFC 9110 §4.2.4 forbids userinfo in
// http(s) URIs and WHATWG fetch rejects it, so such a URL is not portable, and
// `_meta.json` and the trust marker would keep its credentials on disk.
function requireFetchableUrl(url: string): void {
  if (!isUrl(url)) throw new Error('FlatPPL cache: only http and https URLs can be cached');
  if (hasUserinfo(url)) {
    throw new Error('FlatPPL cache: "' + redactUrlUserinfo(url)
      + '" has userinfo (`user:password@`) and cannot be fetched');
  }
}

// ---- verified directories
//
// Every directory from the root down must belong to the effective user and
// must not be group- or other-writable, and none below the root may be a
// symlink. No other user can then change a verified parent, so the chain
// cannot change between the check and the publish. A permissive directory is
// refused, not repaired: content planted before a chmod would survive it.

function unsafeDir(dir: string, problem: string): Error {
  return new Error('FlatPPL cache: cache directory "' + dir + '" ' + problem
    + '; only you may write the cache, so fix its ownership and mode or set '
    + 'FLATPPL_CACHEDIR to a private directory');
}

function checkOwnedDir(dir: string, st: any): void {
  if (!st.isDirectory()) throw unsafeDir(dir, 'is not a directory');
  if (typeof process.geteuid !== 'function') return; // no POSIX ownership (Windows)
  if (st.uid !== process.geteuid()) throw unsafeDir(dir, 'is owned by another user');
  if (st.mode & 0o022) throw unsafeDir(dir, 'is writable by other users');
}

// The root's ancestors are the user's choice. A symlink root must itself
// belong to this user or the superuser, so a link planted in a shared
// directory such as /tmp cannot be swapped after the check.
async function checkRoot(root: string): Promise<void> {
  const link = await fs.lstat(root);
  if (link.isSymbolicLink()) {
    if (typeof process.geteuid === 'function' && link.uid !== process.geteuid() && link.uid !== 0) {
      /* c8 ignore start -- a test cannot create a link owned by another user without root */
      throw unsafeDir(root, 'is a symbolic link owned by another user');
    }
    /* c8 ignore stop */
    checkOwnedDir(root, await fs.stat(root));
    return;
  }
  checkOwnedDir(root, link);
}

function checkCacheDir(dir: string, st: any): void {
  if (st.isSymbolicLink()) throw unsafeDir(dir, 'is a symbolic link');
  checkOwnedDir(dir, st);
}

// mkdir's mode is masked by the umask, so set it explicitly on what we create.
async function makePrivateDir(dir: string, recursive: boolean): Promise<void> {
  let first: string | undefined;
  try { first = await fs.mkdir(dir, { recursive, mode: 0o700 }); }
  catch (e: any) { if (e && e.code === 'EEXIST') return; throw e; }
  if (!recursive) { await fs.chmod(dir, 0o700); return; }
  if (first === undefined) return;
  for (let d = dir; ; d = path.dirname(d)) {
    await fs.chmod(d, 0o700);
    if (d === first || path.dirname(d) === d) break;
  }
}

// `<root>/<components>`, created where missing.
async function createdDir(root: string, components: string[]): Promise<string> {
  await makePrivateDir(root, true);
  await checkRoot(root);
  let dir = root;
  for (const c of components) {
    dir = path.join(dir, c);
    await makePrivateDir(dir, false);
    checkCacheDir(dir, await fs.lstat(dir));
  }
  return dir;
}

// `<root>/<components>`, or null while any part of it is missing.
async function existingDir(root: string, components: string[]): Promise<string | null> {
  try { await checkRoot(root); }
  catch (e: any) { if (e && e.code === 'ENOENT') return null; throw e; }
  let dir = root;
  for (const c of components) {
    dir = path.join(dir, c);
    let st: any;
    try { st = await fs.lstat(dir); }
    catch (e: any) { if (e && e.code === 'ENOENT') return null; throw e; }
    checkCacheDir(dir, st);
  }
  return dir;
}

// A regular file, without following a final symlink.
async function isRegularFile(p: string): Promise<boolean> {
  try { return (await fs.lstat(p)).isFile(); } catch (_e) { return false; }
}

// Exclusively create a 0600 file whatever the umask.
async function openPrivateFile(p: string): Promise<any> {
  const fh = await fs.open(p, 'wx', 0o600);
  try { await fh.chmod(0o600); } catch (e) { await fh.close(); throw e; }
  return fh;
}

// Key derivation (spec "Layout and keys"): `key` = lowercase-hex SHA-256 of the
// URL with any `#`-fragment removed (no other normalization); `<ext>` is the
// full trailing extension — everything after the first `.` in the URL's final
// path segment. The spec's filename/length guard falls back to the bare
// hash for suffixes that cannot be shared safely across cache filesystems.
function urlKey(url: string): { key: string, kk: string, ext: string } {
  const stripped = String(url).split('#')[0];
  const key = crypto.createHash('sha256').update(stripped, 'utf8').digest('hex');
  let seg = '';
  try { seg = new URL(stripped).pathname.split('/').pop() || ''; } catch (_e) { seg = ''; }
  const dot = seg.indexOf('.');
  const suffix = dot >= 0 ? seg.slice(dot + 1) : '';
  const ext = suffix.length <= 32 && /^[A-Za-z0-9._-]*$/.test(suffix) ? suffix : '';
  return { key, kk: key.slice(0, 2), ext };
}

// Absolute paths of the three per-URL cache entries under <flatppl-cachedir>/v1/.
function cachePaths(cacheDir: string, url: string): { object: string, meta: string, trust: string } {
  const { key, kk, ext } = urlKey(url);
  const objDir = path.join(cacheDir, LAYOUT_VERSION, 'objects', kk);
  return {
    object: path.join(objDir, ext ? key + '.' + ext : key),
    meta: path.join(objDir, key + '_meta.json'),
    trust: path.join(cacheDir, LAYOUT_VERSION, 'trust', kk, key),
  };
}

function trustDir(url: string): string[] { return [LAYOUT_VERSION, 'trust', urlKey(url).kk]; }
function objectDir(url: string): string[] { return [LAYOUT_VERSION, 'objects', urlKey(url).kk]; }

async function isTrusted(cacheDir: string, url: string): Promise<boolean> {
  const dir = await existingDir(cacheDir, trustDir(url));
  if (dir === null) return false;
  try { await fs.access(path.join(dir, urlKey(url).key)); return true; }
  catch (_e) { return false; }
}

// Create the per-URL trust marker with an exclusive open (atomic; concurrent
// creators race harmlessly). The marker's content is the URL, for inspection;
// the caller refused userinfo before approval.
async function markTrusted(cacheDir: string, url: string): Promise<void> {
  requireFetchableUrl(url);
  const dir = await createdDir(cacheDir, trustDir(url));
  let fh: any;
  try { fh = await openPrivateFile(path.join(dir, urlKey(url).key)); } // 'wx' = O_CREAT | O_EXCL
  catch (e: any) { if (e && e.code === 'EEXIST') return; throw e; }
  try { await fh.writeFile(String(url)); } finally { await fh.close(); }
}

// Atomic write (spec "Atomicity"): write a temp file on the same filesystem,
// fsync it, then rename it into place. The destination is never written in
// place, and its directory comes from createdDir.
async function atomicWrite(cacheDir: string, destPath: string, data: any): Promise<void> {
  const tmpDir = await createdDir(cacheDir, [LAYOUT_VERSION, 'tmp']);
  const tmp = path.join(tmpDir, crypto.randomBytes(16).toString('hex'));
  const fh = await openPrivateFile(tmp);
  try {
    try { await fh.writeFile(data); await fh.sync(); } finally { await fh.close(); }
    await fs.rename(tmp, destPath);
  } catch (e) {
    await fs.rm(tmp, { force: true });
    throw e;
  }
}

// Resolve a URL to its cached bytes, fetching and caching on a miss (spec
// "Resolve and fetch" + "Trust"). Returns { content: Buffer, meta, fromCache }.
//
// opts: { cacheDir?, env?, offline?, trustAll?, force?, approve?(url)=>bool|Promise<bool>,
//         fetchImpl?, now?()=>string }. `offline` / `trustAll` default to the
// FLATPPL_CACHE_OFFLINE / FLATPPL_TRUST environment variables; `force` performs
// the spec's explicit update (re-fetch + atomic replace, ignoring a cache hit).
async function fetchToCache(url: string, opts: any): Promise<any> {
  requireFetchableUrl(url);
  opts = opts || {};
  const env = opts.env || process.env;
  const cacheDir = opts.cacheDir || resolveCacheDir(opts);
  const p = cachePaths(cacheDir, url);

  // Cache hit — never revalidated against the network, unless opts.force
  // requests an explicit update (re-fetch + atomic replace; spec "update").
  // A symlink is not a hit.
  if (!opts.force && await existingDir(cacheDir, objectDir(url)) !== null && await isRegularFile(p.object)) {
    const content = await fs.readFile(p.object);
    let meta: any = null;
    try { meta = JSON.parse(await fs.readFile(p.meta, 'utf8')); } catch (_e) { /* meta optional on read */ }
    return { content, meta, fromCache: true };
  }

  const offline = opts.offline != null ? opts.offline : !!env.FLATPPL_CACHE_OFFLINE;
  if (offline) throw new Error('FlatPPL cache: offline and "' + redactUrlUserinfo(url) + '" is not cached (cache miss)');

  // Trust gate (per-URL). FLATPPL_TRUST trusts every URL without creating markers.
  const trustAll = opts.trustAll != null ? opts.trustAll : !!env.FLATPPL_TRUST;
  const fetchImpl = opts.fetchImpl || (globalThis as any).fetch;
  let res: any;
  let current = url;
  for (let hop = 0; ; hop++) {
    if (!isUrl(current)) throw new Error('FlatPPL cache: redirect requires an http or https URL');
    requireFetchableUrl(current);
    if (hop > 10) throw new Error('FlatPPL cache: too many redirects');
    if (!trustAll && !(await isTrusted(cacheDir, current))) {
      const ok = typeof opts.approve === 'function' && await opts.approve(current);
      if (!ok) throw new Error('FlatPPL cache: "' + current + '" is not a trusted URL (approval required before fetching)');
      await markTrusted(cacheDir, current);
    }
    try { res = await fetchImpl(current, { redirect: 'manual' }); }
    catch (e: any) {
      throw new Error('FlatPPL cache: network error fetching "' + current + '": '
        + redactUrlUserinfo(String(e && e.message || e)));
    }
    if (![301, 302, 303, 307, 308].includes(res.status)) break;
    const location = res.headers.get('location');
    await res.body?.cancel();
    if (!location) throw new Error('FlatPPL cache: redirect has no Location header');
    current = new URL(location, current).href;
  }
  if (!res.ok) throw new Error('FlatPPL cache: fetching "' + current + '" failed with HTTP status ' + res.status);

  const content = Buffer.from(await res.arrayBuffer());
  const now = opts.now || (() => new Date().toISOString());
  const h = res.headers;
  const meta = {
    url: url,
    resolved_url: current,
    retrieved: now(),
    content_type: (h && h.get) ? h.get('content-type') : null,
    etag: (h && h.get) ? h.get('etag') : null,
    last_modified: (h && h.get) ? h.get('last-modified') : null,
  };

  // Metadata first, then the content object (so a present object always has meta).
  await createdDir(cacheDir, objectDir(url));
  await atomicWrite(cacheDir, p.meta, JSON.stringify(meta, null, 2) + '\n');
  await atomicWrite(cacheDir, p.object, content);
  return { content, meta, fromCache: false };
}

// Convenience for source resolvers: resolve a URL and decode its bytes as UTF-8.
async function readText(url: string, opts: any): Promise<string> {
  return (await fetchToCache(url, opts)).content.toString('utf8');
}

module.exports = {
  isUrl, resolveCacheDir, urlKey, cachePaths, isTrusted, markTrusted, fetchToCache, readText,
  redactUrlUserinfo,
};
