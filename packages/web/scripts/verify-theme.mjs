// The theme bundle's self-check: every file `manifest.json` declares is
// present with the declared size and SHA-256, and nothing else is in the
// bundle. This is what a fetched release is held to (fetch-theme.mjs). A
// drop dir without a manifest is a copy of a checkout — the development
// path — and is reported as unverified, not as an error.

import { createHash } from 'node:crypto';
import { readFile, readdir, stat } from 'node:fs/promises';
import { existsSync, realpathSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const defaultBundle = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../vendor/flatppl-theme',
);

function portable(path) {
  return path.split(sep).join('/');
}

/** Every entry under `root`: regular files as paths, anything else
 *  (symlinks above all — a link in a bundle would be dereferenced by the
 *  copy into the site and could publish files off the build machine) as
 *  a problem. */
async function filesUnder(root) {
  const paths = [];
  const problems = [];
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile()) paths.push(portable(relative(root, path)));
      else problems.push(`${portable(relative(root, path))}: not a regular file`);
    }
  }
  await visit(root);
  return { paths: paths.sort(), problems };
}

async function digest(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

/** Verify a bundle against its own manifest. Returns the list of
 *  problems, empty when it checks out. A missing manifest is a problem
 *  here; callers that accept checkout copies ask describeThemeBundle. */
export async function verifyThemeBundle(bundle = defaultBundle) {
  const root = resolve(bundle);
  const manifestPath = join(root, 'manifest.json');
  const errors = [];
  if (!existsSync(manifestPath)) return ['manifest.json: missing'];
  let manifest;
  try {
    manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  } catch (err) {
    return [`manifest.json: unreadable (${err.message})`];
  }
  if (manifest.name !== 'flatppl-theme') errors.push(`manifest.json: unexpected name ${JSON.stringify(manifest.name)}`);
  if (!Array.isArray(manifest.files)) return [...errors, 'manifest.json: no file list'];

  const declared = new Map(manifest.files.map((file) => [file.path, file]));
  const { paths, problems } = await filesUnder(root);
  errors.push(...problems);
  const actual = paths.filter((path) => path !== 'manifest.json');
  const actualSet = new Set(actual);

  for (const path of actual) {
    if (!declared.has(path)) errors.push(`${path}: not declared in manifest`);
  }
  for (const [path, expected] of declared) {
    if (!actualSet.has(path)) {
      errors.push(`${path}: missing`);
      continue;
    }
    const file = join(root, path);
    if ((await stat(file)).size !== expected.size) errors.push(`${path}: size mismatch`);
    if (await digest(file) !== expected.sha256) errors.push(`${path}: SHA-256 mismatch`);
  }
  return errors;
}

/** What the drop dir holds: `verified` (a release bundle passing its
 *  manifest), `unverified` (a checkout copy: no manifest), or `missing`
 *  (nothing provisioned yet). `errors` is non-empty only for a bundle
 *  that has a manifest and fails it. */
export async function describeThemeBundle(bundle = defaultBundle) {
  const root = resolve(bundle);
  if (!existsSync(join(root, 'tokens.css'))) return { status: 'missing', version: null, errors: [] };
  if (!existsSync(join(root, 'manifest.json'))) return { status: 'unverified', version: null, errors: [] };
  const errors = await verifyThemeBundle(root);
  let version = null;
  try { version = JSON.parse(await readFile(join(root, 'manifest.json'), 'utf8')).version ?? null; } catch (_) { /* reported above */ }
  return { status: errors.length ? 'invalid' : 'verified', version, errors };
}

// Node realpaths the main module, so compare realpaths (a checkout reached
// through a symlinked path would otherwise make this a silent no-op).
if (process.argv[1] && realpathSync(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const info = await describeThemeBundle(process.argv[2]);
  if (info.status === 'missing') {
    console.error('theme: nothing at vendor/flatppl-theme — run `npm run fetch:theme` (or the build) first');
    process.exitCode = 1;
  } else if (info.status === 'unverified') {
    console.warn('theme: UNVERIFIED checkout copy (no manifest) — fine for development, not for a release build');
  } else if (info.status === 'invalid') {
    for (const error of info.errors) console.error(error);
    process.exitCode = 1;
  } else {
    console.log(`theme: flatppl-theme v${info.version} verified against its manifest`);
  }
}
