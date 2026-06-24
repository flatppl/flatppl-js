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

const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const fssync = require('node:fs');
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

// Key derivation (spec "Layout and keys"): `key` = lowercase-hex SHA-256 of the
// URL with any `#`-fragment removed (no other normalization); `<ext>` is the
// full trailing extension — everything after the first `.` in the URL's final
// path segment — or empty when the final segment has no `.`.
function urlKey(url: string): { key: string, kk: string, ext: string } {
  const stripped = String(url).split('#')[0];
  const key = crypto.createHash('sha256').update(stripped, 'utf8').digest('hex');
  let seg = '';
  try { seg = new URL(stripped).pathname.split('/').pop() || ''; } catch (_e) { seg = ''; }
  const dot = seg.indexOf('.');
  const ext = dot >= 0 ? seg.slice(dot + 1) : '';
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

async function isTrusted(cacheDir: string, url: string): Promise<boolean> {
  try { await fs.access(cachePaths(cacheDir, url).trust); return true; }
  catch (_e) { return false; }
}

// Create the per-URL trust marker with an exclusive open (atomic; concurrent
// creators race harmlessly). The marker's content is the URL, for inspection.
async function markTrusted(cacheDir: string, url: string): Promise<void> {
  const p = cachePaths(cacheDir, url).trust;
  await fs.mkdir(path.dirname(p), { recursive: true });
  let fh: any;
  try { fh = await fs.open(p, 'wx'); }      // 'wx' = O_CREAT | O_EXCL
  catch (e: any) { if (e && e.code === 'EEXIST') return; throw e; }
  try { await fh.writeFile(String(url)); } finally { await fh.close(); }
}

// Atomic write (spec "Atomicity"): write a temp file on the same filesystem,
// fsync it, then rename it into place. The destination is never written in place.
async function atomicWrite(cacheDir: string, destPath: string, data: any): Promise<void> {
  const tmpDir = path.join(cacheDir, LAYOUT_VERSION, 'tmp');
  await fs.mkdir(tmpDir, { recursive: true });
  await fs.mkdir(path.dirname(destPath), { recursive: true });
  const tmp = path.join(tmpDir, crypto.randomBytes(16).toString('hex'));
  const fh = await fs.open(tmp, 'w');
  try { await fh.writeFile(data); await fh.sync(); } finally { await fh.close(); }
  await fs.rename(tmp, destPath);
}

// Resolve a URL to its cached bytes, fetching and caching on a miss (spec
// "Resolve and fetch" + "Trust"). Returns { content: Buffer, meta, fromCache }.
//
// opts: { cacheDir?, env?, offline?, trustAll?, approve?(url)=>bool|Promise<bool>,
//         fetchImpl?, now?()=>string }. `offline` / `trustAll` default to the
// FLATPPL_CACHE_OFFLINE / FLATPPL_TRUST environment variables.
async function fetchToCache(url: string, opts: any): Promise<any> {
  opts = opts || {};
  const env = opts.env || process.env;
  const cacheDir = opts.cacheDir || resolveCacheDir(opts);
  const p = cachePaths(cacheDir, url);

  // Cache hit — never revalidated against the network.
  if (fssync.existsSync(p.object)) {
    const content = await fs.readFile(p.object);
    let meta: any = null;
    try { meta = JSON.parse(await fs.readFile(p.meta, 'utf8')); } catch (_e) { /* meta optional on read */ }
    return { content, meta, fromCache: true };
  }

  const offline = opts.offline != null ? opts.offline : !!env.FLATPPL_CACHE_OFFLINE;
  if (offline) throw new Error('FlatPPL cache: offline and "' + url + '" is not cached (cache miss)');

  // Trust gate (per-URL). FLATPPL_TRUST trusts every URL without creating markers.
  const trustAll = opts.trustAll != null ? opts.trustAll : !!env.FLATPPL_TRUST;
  if (!trustAll && !(await isTrusted(cacheDir, url))) {
    const approve = opts.approve;
    const ok = typeof approve === 'function' ? await approve(url) : false;
    if (!ok) throw new Error('FlatPPL cache: "' + url + '" is not a trusted URL (approval required before fetching)');
    await markTrusted(cacheDir, url);
  }

  // Fetch, following redirects. Network error / non-2xx / unresolvable redirect → error.
  const fetchImpl = opts.fetchImpl || (globalThis as any).fetch;
  let res: any;
  try { res = await fetchImpl(url, { redirect: 'follow' }); }
  catch (e: any) { throw new Error('FlatPPL cache: network error fetching "' + url + '": ' + (e && e.message || e)); }
  if (!res.ok) throw new Error('FlatPPL cache: fetching "' + url + '" failed with HTTP status ' + res.status);

  const content = Buffer.from(await res.arrayBuffer());
  const now = opts.now || (() => new Date().toISOString());
  const h = res.headers;
  const meta = {
    url: url,
    resolved_url: res.url || url,
    retrieved: now(),
    content_type: (h && h.get) ? h.get('content-type') : null,
    etag: (h && h.get) ? h.get('etag') : null,
    last_modified: (h && h.get) ? h.get('last-modified') : null,
  };

  // Metadata first, then the content object (so a present object always has meta).
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
};
