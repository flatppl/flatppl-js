// Downloads from github.com for the builds (theme release asset, examples
// and grammars archives), with a REST API fallback.
//
// The github.com download endpoints (`/releases/download/…`,
// `/archive/refs/…`) fail on their own now and then (502/504 for every
// repository) while api.github.com keeps serving the same bytes. Each
// download therefore tries the github.com URL first and, when that fails,
// the API:
//   - a release asset: GET /repos/<o>/<r>/releases/tags/<tag>, find the
//     asset by name, GET its `url` with `Accept: application/octet-stream`;
//   - an archive: GET /repos/<o>/<r>/tarball/<ref>. Its top directory is
//     `<owner>-<repo>-<sha>` rather than `<repo>-<ref>`; extractTarball
//     returns the single top directory whatever its name.
// GITHUB_TOKEN, when set, is sent to the API as a bearer token (higher rate
// limit in CI). FLATPPL_GITHUB_WEB_BASE replaces https://github.com in the
// first URL; tests point it at an unreachable host to force the fallback.

import { mkdir, mkdtemp, readdir, rm, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const API = 'https://api.github.com';
const TIMEOUT_MS = 30_000;
const MAX_BYTES = 256 * 1024 * 1024;

function webBase(env) {
  return (env.FLATPPL_GITHUB_WEB_BASE || 'https://github.com').replace(/\/+$/, '');
}

export function releaseAssetUrl({ owner, repo, tag, asset, env = process.env }) {
  return `${webBase(env)}/${owner}/${repo}/releases/download/${tag}/${asset}`;
}

export function archiveUrl({ owner, repo, ref, env = process.env }) {
  return `${webBase(env)}/${owner}/${repo}/archive/refs/heads/${ref}.tar.gz`;
}

function apiHeaders(env, accept) {
  const h = { Accept: accept, 'User-Agent': 'flatppl-build', 'X-GitHub-Api-Version': '2022-11-28' };
  if (env.GITHUB_TOKEN) h.Authorization = `Bearer ${env.GITHUB_TOKEN}`;
  return h;
}

async function fetchBytes(url, { fetchImpl, headers, attempts, maxBytes }) {
  let lastErr;
  for (let i = 1; i <= attempts; i++) {
    try {
      // The timeout bounds a stalled connection; the retries cover hard errors.
      const res = await fetchImpl(url, { redirect: 'follow', headers, signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length > maxBytes) throw new Error(`${buf.length} bytes, more than the ${maxBytes} allowed`);
      return buf;
    } catch (err) {
      lastErr = err;
      if (i < attempts) await new Promise((r) => setTimeout(r, 500 * 2 ** (i - 1)));
    }
  }
  throw new Error(`${url}: ${lastErr?.message || lastErr}`);
}

async function fetchJson(url, opts) {
  return JSON.parse((await fetchBytes(url, { ...opts, maxBytes: 16 * 1024 * 1024 })).toString('utf8'));
}

async function withFallback(what, primary, fallback, log) {
  try {
    return await primary();
  } catch (first) {
    log(`  ${what}: ${first.message}; trying the GitHub REST API`);
    try {
      return await fallback();
    } catch (second) {
      throw new Error(`could not download ${what}: ${first.message}; REST API: ${second.message}`);
    }
  }
}

/** The bytes of release asset `asset` of `<owner>/<repo>` tag `tag`. */
export async function downloadReleaseAsset({
  owner, repo, tag, asset, fetchImpl = globalThis.fetch, env = process.env,
  attempts = 3, maxBytes = MAX_BYTES, log = console.log,
}) {
  const opts = { fetchImpl, attempts, maxBytes };
  return withFallback(`${owner}/${repo} ${tag} ${asset}`,
    () => fetchBytes(releaseAssetUrl({ owner, repo, tag, asset, env }), opts),
    async () => {
      const rel = await fetchJson(`${API}/repos/${owner}/${repo}/releases/tags/${encodeURIComponent(tag)}`,
        { ...opts, headers: apiHeaders(env, 'application/vnd.github+json') });
      const a = (rel.assets || []).find((x) => x.name === asset);
      if (!a) throw new Error(`release ${tag} has no asset ${asset}`);
      return fetchBytes(a.url, { ...opts, headers: apiHeaders(env, 'application/octet-stream') });
    }, log);
}

/** The .tar.gz bytes of `<owner>/<repo>` at branch `ref`. */
export async function downloadArchive({
  owner, repo, ref, fetchImpl = globalThis.fetch, env = process.env,
  attempts = 3, maxBytes = MAX_BYTES, log = console.log,
}) {
  const opts = { fetchImpl, attempts, maxBytes };
  return withFallback(`${owner}/${repo}@${ref}`,
    () => fetchBytes(archiveUrl({ owner, repo, ref, env }), opts),
    () => fetchBytes(`${API}/repos/${owner}/${repo}/tarball/${encodeURIComponent(ref)}`,
      { ...opts, headers: apiHeaders(env, 'application/vnd.github+json') }),
    log);
}

/** Extract `.tar.gz` bytes into a fresh temp dir. Returns the archive's
 *  single top directory (any name) and a cleanup(); the caller copies what
 *  it needs out of `topLevel` and awaits cleanup() in a finally. Extract
 *  then copy (not a tar wildcard) works alike on GNU tar, bsdtar and MSYS. */
export async function extractTarball(bytes, prefix) {
  const tmpRoot = await mkdtemp(join(tmpdir(), prefix));
  const tmpFile = join(tmpRoot, 'archive.tar.gz');
  const tmpExtract = join(tmpRoot, 'extract');
  const cleanup = () => rm(tmpRoot, { recursive: true, force: true }).catch(() => {});
  try {
    await writeFile(tmpFile, bytes);
    await mkdir(tmpExtract, { recursive: true });
    await new Promise((resolve, reject) => {
      const tar = spawn('tar', ['-xzf', tmpFile, '-C', tmpExtract], { stdio: ['ignore', 'inherit', 'inherit'] });
      tar.on('error', (err) => reject(err.code === 'ENOENT'
        ? new Error('`tar` is not on PATH — it is needed to unpack a GitHub archive')
        : err));
      tar.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`tar exit ${code}`))));
    });
    const dirs = (await readdir(tmpExtract, { withFileTypes: true })).filter((e) => e.isDirectory());
    if (dirs.length !== 1) throw new Error(`archive has ${dirs.length} top-level directories, expected 1`);
    return { topLevel: join(tmpExtract, dirs[0].name), cleanup };
  } catch (err) {
    await cleanup();
    throw err;
  }
}

/** Download an archive and extract it: `{ topLevel, cleanup }`. */
export async function fetchArchiveTree({ owner, repo, ref, prefix, ...rest }) {
  return extractTarball(await downloadArchive({ owner, repo, ref, ...rest }), prefix);
}
