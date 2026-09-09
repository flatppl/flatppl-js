// flatppl-theme provisioning for the web build — sibling-first, pinned
// GitHub release otherwise — the same resolution the build applies to
// flatppl-examples and flatppl-grammars.
//
//   1. FLATPPL_THEME_DIR, else a sibling checkout `../flatppl-theme` next
//      to this repo: the theme's SOURCE files are copied as they are (the
//      release bundle is exactly those files plus a manifest, so no bun
//      build is needed). Nothing is verified — that is the point of the
//      dev loop: edit tokens.css in the sibling, rebuild, see it. The
//      build says so loudly. FLATPPL_THEME_NO_SIBLING=1 skips this step
//      (to exercise the release path with a sibling present).
//   2. Otherwise the pinned release: the tarball flatppl-theme publishes
//      for the tag (THEME_PIN, overridable through FLATPPL_THEME_REF for a
//      candidate) is downloaded and checked against ITS OWN manifest —
//      name, release tag, every file's size and SHA-256, no stray files.
//      The tag is the only pin; there is no hand-maintained manifest hash
//      to bump when a release is adopted. A bundle already in the drop
//      dir that passes that check for the pinned tag is reused (no
//      network on a rebuild).
//
// The drop dir (vendor/flatppl-theme at the repo root) is build output,
// gitignored. No CDN at runtime, no submodule, no global installs.

import { copyFile, cp, mkdir, mkdtemp, readFile, rename, rm, writeFile } from 'node:fs/promises';
import { existsSync, realpathSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifyThemeBundle } from './verify-theme.mjs';

export const THEME_REPOSITORY = 'https://github.com/flatppl/flatppl-theme';

/** The adopted theme release. Bumping this is the whole adoption. */
export const THEME_PIN = 'v0.1.8';

/** The bundle's file list (flatppl-theme's scripts/bundle.ts
 *  REQUIRED_FILES): what a sibling copy must find in the checkout, and
 *  what a release must declare. The theme bundles `assets/` recursively,
 *  so a sibling copy takes that whole directory, not just the entries
 *  listed here. */
export const THEME_FILES = [
  'assets/android-chrome-192x192.png',
  'assets/android-chrome-512x512.png',
  'assets/apple-touch-icon.png',
  'assets/favicon-16x16.png',
  'assets/favicon-32x32.png',
  'assets/favicon.ico',
  'assets/logo-original.png',
  'assets/logo.svg',
  'assets/site.webmanifest',
  'assets/wordmark.svg',
  'components.css',
  'footer.html',
  'header.html',
  'LICENSE.md',
  'LICENSES/CC-BY-4.0.txt',
  'LICENSES/MIT.txt',
  'shell.css',
  'shell.js',
  'syntax-map.json',
  'tokens.css',
];

const here = dirname(fileURLToPath(import.meta.url));            // packages/web/scripts/
const defaultRepoRoot = dirname(dirname(dirname(here)));         // flatppl-js/

/** What to do when neither source is reachable — attached to every
 *  provisioning error so any caller (the build, not only the CLI) shows it. */
const HINT = 'Set FLATPPL_THEME_DIR to a flatppl-theme checkout, check one out as a sibling of this repo, or restore network access to GitHub.';

function fail(message) {
  return new Error(`${message}\n${HINT}`);
}

export function themeReleaseUrl(tag) {
  return `${THEME_REPOSITORY}/releases/download/${tag}/flatppl-theme-${tag}.tar.gz`;
}

/** Where the theme comes from, by precedence: an explicit directory, the
 *  sibling checkout, the pinned release. Pure (no I/O beyond existsSync). */
export function resolveThemeSource({ repoRoot = defaultRepoRoot, env = process.env } = {}) {
  if (env.FLATPPL_THEME_DIR) return { kind: 'dir', path: resolve(env.FLATPPL_THEME_DIR), label: 'FLATPPL_THEME_DIR' };
  const sibling = join(dirname(repoRoot), 'flatppl-theme');
  if (!env.FLATPPL_THEME_NO_SIBLING && existsSync(sibling)) return { kind: 'dir', path: sibling, label: 'sibling' };
  return { kind: 'release', tag: env.FLATPPL_THEME_REF || THEME_PIN };
}

/** Copy the theme's source files from a checkout into the drop dir:
 *  the listed files, plus everything under assets/ (bundled recursively
 *  by the theme's own builder, so an asset added upstream is not lost). */
async function copyThemeSource(srcDir, themeDir) {
  const missing = THEME_FILES.filter((f) => !existsSync(join(srcDir, f)));
  if (missing.length) {
    throw fail(`theme: ${srcDir} is not a flatppl-theme checkout — missing ${missing.join(', ')}`);
  }
  await rm(themeDir, { recursive: true, force: true });
  for (const f of THEME_FILES) {
    if (f.startsWith('assets/')) continue;
    await mkdir(dirname(join(themeDir, f)), { recursive: true });
    await copyFile(join(srcDir, f), join(themeDir, f));
  }
  await cp(join(srcDir, 'assets'), join(themeDir, 'assets'), { recursive: true, dereference: true });
}

async function readManifest(themeDir) {
  try {
    return JSON.parse(await readFile(join(themeDir, 'manifest.json'), 'utf8'));
  } catch (_) {
    return null;
  }
}

/** A drop-dir bundle that is the pinned release and passes its own
 *  manifest check — the build cache. */
async function cachedRelease(themeDir, tag) {
  const manifest = await readManifest(themeDir);
  if (!manifest || manifest.source?.release !== tag) return false;
  return (await verifyThemeBundle(themeDir)).length === 0;
}

/** A bundle is well under a few MB; anything larger is not the theme. */
const MAX_ARCHIVE_BYTES = 32 * 1024 * 1024;
const DOWNLOAD_TIMEOUT_MS = 30_000;

async function download(url, fetchImpl, attempts = 4) {
  let lastErr;
  for (let i = 1; i <= attempts; i++) {
    try {
      // The timeout bounds a stalled connection; the retry loop only helps
      // with hard errors.
      const res = await fetchImpl(url, { redirect: 'follow', signal: AbortSignal.timeout(DOWNLOAD_TIMEOUT_MS) });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const buf = Buffer.from(await res.arrayBuffer());
      if (buf.length > MAX_ARCHIVE_BYTES) throw new Error(`archive is ${buf.length} bytes, more than a theme bundle can be`);
      return buf;
    } catch (err) {
      lastErr = err;
      if (i < attempts) await new Promise((r) => setTimeout(r, 500 * 2 ** (i - 1)));
    }
  }
  throw fail(`theme: could not download ${url}: ${lastErr?.message || lastErr}`);
}

function untar(archive, dir) {
  return new Promise((resolveP, reject) => {
    const tar = spawn('tar', ['-xzf', archive, '-C', dir], { stdio: ['ignore', 'inherit', 'inherit'] });
    tar.on('error', (err) => reject(err.code === 'ENOENT'
      ? new Error('theme: `tar` is not on PATH — it is needed to unpack the release archive')
      : err));
    tar.on('exit', (code) => (code === 0 ? resolveP() : reject(new Error(`theme: tar could not unpack the release archive (exit ${code})`))));
  });
}

/** Fetch the release tarball into a temp dir, check it, then move it
 *  into place. The tarball holds the bundle's files at its top level. */
async function fetchRelease(tag, themeDir, fetchImpl) {
  const url = themeReleaseUrl(tag);
  const tmpRoot = await mkdtemp(join(tmpdir(), 'flatppl-theme-'));
  try {
    const archive = join(tmpRoot, 'bundle.tar.gz');
    const extract = join(tmpRoot, 'bundle');
    await writeFile(archive, await download(url, fetchImpl));
    await mkdir(extract, { recursive: true });
    await untar(archive, extract);
    const manifest = await readManifest(extract);
    if (!manifest) throw new Error(`theme: ${url} holds no manifest.json`);
    if (manifest.name !== 'flatppl-theme' || manifest.source?.release !== tag) {
      throw new Error(`theme: ${url} is not flatppl-theme ${tag} (manifest says ${manifest.name} ${manifest.source?.release})`);
    }
    const errors = await verifyThemeBundle(extract);
    if (errors.length) throw new Error(`theme: ${url} fails its own manifest:\n${errors.join('\n')}`);
    // A release that dropped a file the consumers rely on is a contract
    // break, not something to discover at page load.
    const declared = new Set(manifest.files.map((x) => x.path));
    const absent = THEME_FILES.filter((f) => !declared.has(f));
    if (absent.length) throw new Error(`theme: ${url} does not bundle ${absent.join(', ')}`);
    await rm(themeDir, { recursive: true, force: true });
    await mkdir(dirname(themeDir), { recursive: true });
    try {
      await rename(extract, themeDir);
    } catch (err) {
      if (err.code !== 'EXDEV') throw err;
      // Temp dir on another filesystem: copy the verified files (the
      // manifest last, so an interrupted copy never looks like a cache).
      for (const f of [...manifest.files.map((x) => x.path), 'manifest.json']) {
        await mkdir(dirname(join(themeDir, f)), { recursive: true });
        await copyFile(join(extract, f), join(themeDir, f));
      }
    }
  } finally {
    await rm(tmpRoot, { recursive: true, force: true }).catch(() => {});
  }
}

/**
 * Provision the theme into `themeDir`. Returns `{ source, verified,
 * version }` — `verified` is false for a checkout copy. Throws with an
 * actionable message when neither a checkout nor the release is
 * reachable.
 */
export async function syncTheme({
  repoRoot = defaultRepoRoot,
  themeDir = join(repoRoot, 'vendor', 'flatppl-theme'),
  env = process.env,
  log = console.log,
  fetchImpl = globalThis.fetch,
} = {}) {
  const source = resolveThemeSource({ repoRoot, env });
  if (source.kind === 'dir') {
    await copyThemeSource(source.path, themeDir);
    log(`theme: using UNVERIFIED ${source.label} checkout at ${source.path}`);
    return { source, verified: false, version: null };
  }
  const { tag } = source;
  if (await cachedRelease(themeDir, tag)) {
    log(`theme: using cached flatppl-theme ${tag} (verified)`);
  } else {
    await fetchRelease(tag, themeDir, fetchImpl);
    log(`theme: fetched flatppl-theme ${tag} from GitHub (verified)`);
  }
  const manifest = await readManifest(themeDir);
  return { source, verified: true, version: manifest?.version ?? null };
}

// Node realpaths the main module, so compare realpaths (a checkout reached
// through a symlinked path would otherwise make this a silent no-op).
if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    await syncTheme();
  } catch (err) {
    console.error(err.message);
    process.exitCode = 1;
  }
}
