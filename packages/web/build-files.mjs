// @flatppl/web — static corpus publication. The sibling checkout and archive
// share one catalog, copy, and manifest path. File helpers also serve demo,
// grammar, and theme assets in build.mjs.

import { copyFile, mkdir, mkdtemp, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { DATA_EXTENSIONS, MODEL_EXTENSIONS, isCollectionPath, normalizeCollections, typeForPath } from './src/file-types.mjs';

/** Include the source path in malformed JSON diagnostics as well as I/O errors. */
async function readJson(path) {
  try { return JSON.parse(await readFile(path, 'utf8')); }
  catch (error) { throw new Error(`${path}: ${error.message}`); }
}

/** Publish only catalog roots, after validating the whole catalog. Full builds
 * prune previously published roots too, so a removed collection cannot survive
 * in a reused dist. Watch startup preserves directories held by serve.mjs. */
export async function syncExampleCollections({ siblingDir, archiveURL, distDir, clean = true, log = console.log }) {
  let sourceDir = siblingDir;
  let cleanup = async () => {};
  if (existsSync(siblingDir)) {
    log(`  examples: using sibling clone at ${siblingDir}`);
  } else {
    log(`  examples: fetching ${archiveURL}`);
    ({ topLevel: sourceDir, cleanup } = await fetchAndExtractTarball(archiveURL, 'flatppl-examples-'));
  }
  try {
    const catalog = join(sourceDir, 'collections.json');
    const collections = normalizeCollections(await readJson(catalog), catalog);
    for (const { path } of collections) {
      const directory = join(sourceDir, path);
      if (!existsSync(directory) || !(await stat(directory)).isDirectory()) {
        throw new Error(`${catalog}: collection ${path}: directory not found at ${directory}`);
      }
    }
    const ownershipPath = join(distDir, '.example-roots.json');
    let oldRoots;
    if (existsSync(ownershipPath)) {
      oldRoots = await readJson(ownershipPath);
      if (!Array.isArray(oldRoots) || !oldRoots.every(isCollectionPath)) {
        throw new Error(`${ownershipPath}: invalid collection output paths`);
      }
    } else {
      const manifestPath = join(distDir, 'models.json');
      const previous = existsSync(manifestPath) ? await readJson(manifestPath) : {};
      // Validate old paths before deleting anything. Minimal manifests predate
      // collection metadata and only owned the examples/ output directory.
      oldRoots = previous.collections === undefined ? ['examples']
        : normalizeCollections(previous.collections, manifestPath).map(entry => entry.path);
    }
    const roots = collections.map(entry => entry.path);
    const owned = new Set([...oldRoots, ...roots]);
    await mkdir(distDir, { recursive: true });
    // Keep output ownership separate from the visible manifest: watch startup
    // rewrites models.json without pruning directories. Record intended roots
    // before copying too, so a failed copy cannot leave an untracked output.
    await writeFile(ownershipPath, JSON.stringify([...owned]));
    if (clean) {
      for (const path of owned) await rm(join(distDir, path), { recursive: true, force: true });
    }
    const entries = [];
    for (const { path: root } of collections) {
      const destination = join(distDir, root);
      await copyDirRecursive(join(sourceDir, root), destination);
      for (const file of await listModelFiles(destination)) {
        const path = root + '/' + file;
        entries.push({ path, title: file.slice(file.lastIndexOf('/') + 1), type: typeForPath(path) });
      }
    }
    if (clean) await writeFile(ownershipPath, JSON.stringify(roots));
    return { collections, entries };
  } finally {
    await cleanup();
  }
}

/** Keep support assets at their source-relative paths alongside model files. */
export async function copyDirRecursive(srcDir, dstDir) {
  if (!existsSync(srcDir)) return;
  await mkdir(dstDir, { recursive: true });
  const entries = await readdir(srcDir, { withFileTypes: true });
  for (const ent of entries) {
    const src = join(srcDir, ent.name);
    const dst = join(dstDir, ent.name);
    if (ent.isDirectory()) {
      await copyDirRecursive(src, dst);
    } else {
      await copyFile(src, dst);
    }
  }
}

// Fetch a .tar.gz, extract the whole archive into a fresh temp dir, and
// return its single top-level directory (GitHub archives unpack into one
// `<repo>-<ref>/` dir) plus a cleanup() to remove the temp tree. The
// caller copies whatever subdir it needs out of `topLevel`, then awaits
// cleanup() in a finally. Two-step "extract then copy" (vs a tar wildcard)
// avoids portability issues across GNU tar / bsdtar / macOS / Windows
// MSYS / CI ubuntu-latest. Shared by collection and grammar publication.
export async function fetchAndExtractTarball(url, prefix) {
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const tmpRoot = await mkdtemp(join(tmpdir(), prefix));
  const tmpFile = join(tmpRoot, 'archive.tar.gz');
  const tmpExtract = join(tmpRoot, 'extract');
  const cleanup = () => rm(tmpRoot, { recursive: true, force: true }).catch(() => {});
  try {
    await writeFile(tmpFile, Buffer.from(await res.arrayBuffer()));
    await mkdir(tmpExtract, { recursive: true });
    await new Promise((resolve, reject) => {
      const tar = spawn('tar', ['-xzf', tmpFile, '-C', tmpExtract], {
        stdio: ['ignore', 'inherit', 'inherit'],
      });
      tar.on('error', reject);
      tar.on('exit', code =>
        code === 0 ? resolve() : reject(new Error(`tar exit ${code}`))
      );
    });
    const entries = await readdir(tmpExtract, { withFileTypes: true });
    const topDir = entries.find(e => e.isDirectory());
    if (!topDir) {
      throw new Error(`archive extracted no top-level directory: ${url}`);
    }
    return { topLevel: join(tmpExtract, topDir.name), cleanup };
  } catch (err) {
    await cleanup();
    throw err;
  }
}

/** Recursively list models, documents and data under rootDir, sorted
    alphabetically. Paths are relative to rootDir (forward slashes).
    Other support assets are copied but have no inspectable surface. */
export async function listModelFiles(rootDir) {
  const out = [];
  const extensions = [...MODEL_EXTENSIONS, ...DATA_EXTENSIONS];
  async function walk(dir, prefix) {
    if (!existsSync(dir)) return;
    const entries = await readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
      const childRelPath = prefix ? prefix + '/' + ent.name : ent.name;
      if (ent.isDirectory()) {
        await walk(join(dir, ent.name), childRelPath);
      } else if (ent.isFile()
          && extensions.some((e) => ent.name.toLowerCase().endsWith(e))) {
        out.push(childRelPath);
      }
    }
  }
  await walk(rootDir, '');
  out.sort();
  return out;
}
