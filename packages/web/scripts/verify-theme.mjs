import { createHash } from 'node:crypto';
import { readFile, readdir, stat } from 'node:fs/promises';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const EXPECTED_MANIFEST_SHA256 = '7ab3caefa5f8736077945dd4d6f3571fdc17c4f53004c61f6395758e3a3675cc';
const defaultBundle = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '../../../vendor/flatppl-theme',
);

function portable(path) {
  return path.split(sep).join('/');
}

async function filesUnder(root) {
  const paths = [];
  async function visit(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile()) paths.push(portable(relative(root, path)));
    }
  }
  await visit(root);
  return paths.sort();
}

async function digest(path) {
  return createHash('sha256').update(await readFile(path)).digest('hex');
}

/** Verify the exact pinned release manifest and every file it declares. */
export async function verifyThemeBundle(bundle = defaultBundle) {
  const root = resolve(bundle);
  const manifestPath = join(root, 'manifest.json');
  const errors = [];
  if (await digest(manifestPath) !== EXPECTED_MANIFEST_SHA256) {
    errors.push('manifest.json: does not match pinned flatppl-theme v0.1.4');
    return errors;
  }

  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  const declared = new Map(manifest.files.map((file) => [file.path, file]));
  const actual = (await filesUnder(root)).filter((path) => path !== 'manifest.json');
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

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const errors = await verifyThemeBundle(process.argv[2]);
  for (const error of errors) console.error(error);
  if (errors.length > 0) process.exitCode = 1;
}
