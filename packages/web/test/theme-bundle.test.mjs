import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { spawn } from 'node:child_process';
import { join, dirname } from 'node:path';
import { tmpdir } from 'node:os';
import { verifyThemeBundle, describeThemeBundle } from '../scripts/verify-theme.mjs';
import { THEME_FILES, THEME_PIN, resolveThemeSource, syncTheme, themeReleaseUrl } from '../scripts/fetch-theme.mjs';

// The theme is provisioned at build time (scripts/fetch-theme.mjs): a
// sibling checkout is copied unverified, the pinned release is downloaded
// and held to its own manifest. No network here — the release path is
// exercised with a tarball built from a fixture bundle.

async function tmp(t, prefix) {
  const dir = await mkdtemp(join(tmpdir(), prefix));
  t.after(() => rm(dir, { recursive: true, force: true }));
  return dir;
}

/** A fake theme checkout: every required file, small distinct contents. */
async function fakeCheckout(dir) {
  for (const f of THEME_FILES) {
    await mkdir(dirname(join(dir, f)), { recursive: true });
    await writeFile(join(dir, f), `fixture ${f}\n`);
  }
}

/** A fake release bundle: a checkout plus a manifest listing it. */
async function fakeBundle(dir, { release = THEME_PIN, name = 'flatppl-theme' } = {}) {
  await fakeCheckout(dir);
  const files = [];
  for (const path of THEME_FILES) {
    const buf = await readFile(join(dir, path));
    files.push({ path, sha256: createHash('sha256').update(buf).digest('hex'), size: (await stat(join(dir, path))).size });
  }
  const manifest = { schema: 1, name, version: release.replace(/^v/, ''), source: { repository: 'https://github.com/flatppl/flatppl-theme', commit: 'f'.repeat(40), release }, files };
  await writeFile(join(dir, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  return manifest;
}

function tarGz(dir, archive) {
  return new Promise((resolve, reject) => {
    const p = spawn('tar', ['-czf', archive, '-C', dir, '.'], { stdio: 'ignore' });
    p.on('error', reject);
    p.on('exit', (code) => (code === 0 ? resolve() : reject(new Error('tar exit ' + code))));
  });
}

/** A fetch() stand-in serving one archive for one URL. */
function fakeFetch(expectedUrl, archivePath) {
  const calls = [];
  const impl = async (url) => {
    calls.push(url);
    if (url !== expectedUrl) return { ok: false, status: 404, statusText: 'Not Found' };
    const buf = await readFile(archivePath);
    return { ok: true, status: 200, statusText: 'OK', arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) };
  };
  return { impl, calls };
}

test('a bundle passes its own manifest; tampering, stray files and a missing file are reported', async (t) => {
  const dir = await tmp(t, 'theme-bundle-');
  await fakeBundle(dir);
  assert.deepEqual(await verifyThemeBundle(dir), []);
  assert.deepEqual(await describeThemeBundle(dir), { status: 'verified', version: THEME_PIN.slice(1), errors: [] });

  await writeFile(join(dir, 'tokens.css'), 'tampered\n');
  await writeFile(join(dir, 'extra.css'), 'stray\n');
  await rm(join(dir, 'shell.js'));
  const errors = await verifyThemeBundle(dir);
  assert.ok(errors.includes('tokens.css: size mismatch'));
  assert.ok(errors.includes('tokens.css: SHA-256 mismatch'));
  assert.ok(errors.includes('extra.css: not declared in manifest'));
  assert.ok(errors.includes('shell.js: missing'));
  assert.equal((await describeThemeBundle(dir)).status, 'invalid');
});

test('a checkout copy (no manifest) is unverified, an empty drop dir is missing', async (t) => {
  const dir = await tmp(t, 'theme-copy-');
  assert.equal((await describeThemeBundle(dir)).status, 'missing');
  await fakeCheckout(dir);
  assert.deepEqual(await describeThemeBundle(dir), { status: 'unverified', version: null, errors: [] });
  assert.deepEqual(await verifyThemeBundle(dir), ['manifest.json: missing']);
});

test('resolution precedence: FLATPPL_THEME_DIR, then the sibling checkout, then the pinned release', async (t) => {
  const root = await tmp(t, 'theme-root-');
  const repoRoot = join(root, 'flatppl-js');
  await mkdir(repoRoot);
  assert.deepEqual(resolveThemeSource({ repoRoot, env: {} }), { kind: 'release', tag: THEME_PIN });
  assert.deepEqual(resolveThemeSource({ repoRoot, env: { FLATPPL_THEME_REF: 'v9.9.9' } }), { kind: 'release', tag: 'v9.9.9' });

  const sibling = join(root, 'flatppl-theme');
  await mkdir(sibling);
  assert.deepEqual(resolveThemeSource({ repoRoot, env: {} }), { kind: 'dir', path: sibling, label: 'sibling' });
  assert.equal(resolveThemeSource({ repoRoot, env: { FLATPPL_THEME_NO_SIBLING: '1' } }).kind, 'release');
  const explicit = join(root, 'elsewhere');
  assert.deepEqual(resolveThemeSource({ repoRoot, env: { FLATPPL_THEME_DIR: explicit } }), { kind: 'dir', path: explicit, label: 'FLATPPL_THEME_DIR' });
});

test('themeReleaseUrl names the release asset flatppl-theme publishes', () => {
  assert.equal(themeReleaseUrl('v0.1.8'), 'https://github.com/flatppl/flatppl-theme/releases/download/v0.1.8/flatppl-theme-v0.1.8.tar.gz');
});

test('a checkout is copied file by file, unverified, and re-copied on the next sync', async (t) => {
  const root = await tmp(t, 'theme-sync-dir-');
  const checkout = join(root, 'checkout');
  await fakeCheckout(checkout);
  const themeDir = join(root, 'drop');
  const logs = [];
  const r = await syncTheme({ repoRoot: join(root, 'repo'), themeDir, env: { FLATPPL_THEME_DIR: checkout }, log: (m) => logs.push(m) });
  assert.equal(r.verified, false);
  assert.equal(r.source.kind, 'dir');
  assert.equal(await readFile(join(themeDir, 'tokens.css'), 'utf8'), 'fixture tokens.css\n');
  assert.ok(logs[0].startsWith('theme: using UNVERIFIED FLATPPL_THEME_DIR checkout at '));
  assert.equal((await describeThemeBundle(themeDir)).status, 'unverified');

  await writeFile(join(checkout, 'tokens.css'), 'edited\n');
  await syncTheme({ repoRoot: join(root, 'repo'), themeDir, env: { FLATPPL_THEME_DIR: checkout }, log: () => {} });
  assert.equal(await readFile(join(themeDir, 'tokens.css'), 'utf8'), 'edited\n');
});

test('a directory that is not a theme checkout is refused with the missing files named', async (t) => {
  const root = await tmp(t, 'theme-sync-bad-');
  await mkdir(join(root, 'notatheme'));
  await assert.rejects(
    syncTheme({ repoRoot: join(root, 'repo'), themeDir: join(root, 'drop'), env: { FLATPPL_THEME_DIR: join(root, 'notatheme') }, log: () => {} }),
    /not a flatppl-theme checkout — missing .*tokens\.css/,
  );
});

test('the release path downloads the pinned tarball, verifies it, and then reuses the cached copy', async (t) => {
  const root = await tmp(t, 'theme-sync-rel-');
  const src = join(root, 'src');
  await fakeBundle(src);
  const archive = join(root, 'bundle.tar.gz');
  await tarGz(src, archive);
  const { impl, calls } = fakeFetch(themeReleaseUrl(THEME_PIN), archive);
  const themeDir = join(root, 'drop');
  const env = { FLATPPL_THEME_NO_SIBLING: '1' };
  const logs = [];

  const first = await syncTheme({ repoRoot: join(root, 'repo'), themeDir, env, log: (m) => logs.push(m), fetchImpl: impl });
  assert.deepEqual(first, { source: { kind: 'release', tag: THEME_PIN }, verified: true, version: THEME_PIN.slice(1) });
  assert.equal(calls.length, 1);
  assert.equal(logs.at(-1), `theme: fetched flatppl-theme ${THEME_PIN} from GitHub (verified)`);
  assert.deepEqual(await verifyThemeBundle(themeDir), []);

  const second = await syncTheme({ repoRoot: join(root, 'repo'), themeDir, env, log: (m) => logs.push(m), fetchImpl: impl });
  assert.equal(second.verified, true);
  assert.equal(calls.length, 1, 'no second download');
  assert.equal(logs.at(-1), `theme: using cached flatppl-theme ${THEME_PIN} (verified)`);

  // A stale cache (another tag) is replaced, not reused.
  await syncTheme({ repoRoot: join(root, 'repo'), themeDir, env: { ...env, FLATPPL_THEME_REF: 'v0.0.1' }, log: () => {}, fetchImpl: impl })
    .catch((err) => assert.match(err.message, /could not download .*v0\.0\.1/));
});

test('a tarball whose contents fail their manifest is rejected and nothing is installed', async (t) => {
  const root = await tmp(t, 'theme-sync-tamper-');
  const src = join(root, 'src');
  await fakeBundle(src);
  await writeFile(join(src, 'shell.js'), 'tampered after the manifest was written\n');
  const archive = join(root, 'bundle.tar.gz');
  await tarGz(src, archive);
  const { impl } = fakeFetch(themeReleaseUrl(THEME_PIN), archive);
  const themeDir = join(root, 'drop');
  await assert.rejects(
    syncTheme({ repoRoot: join(root, 'repo'), themeDir, env: { FLATPPL_THEME_NO_SIBLING: '1' }, log: () => {}, fetchImpl: impl }),
    /fails its own manifest[\s\S]*shell\.js: SHA-256 mismatch/,
  );
  assert.equal((await describeThemeBundle(themeDir)).status, 'missing');
});

test('a tarball for a different release than the one asked for is rejected', async (t) => {
  const root = await tmp(t, 'theme-sync-wrongtag-');
  const src = join(root, 'src');
  await fakeBundle(src, { release: 'v0.0.9' });
  const archive = join(root, 'bundle.tar.gz');
  await tarGz(src, archive);
  const { impl } = fakeFetch(themeReleaseUrl(THEME_PIN), archive);
  await assert.rejects(
    syncTheme({ repoRoot: join(root, 'repo'), themeDir: join(root, 'drop'), env: { FLATPPL_THEME_NO_SIBLING: '1' }, log: () => {}, fetchImpl: impl }),
    /is not flatppl-theme v0\.1\.8/,
  );
});
