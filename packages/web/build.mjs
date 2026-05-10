// @flatppl/web — build.
//
// Produces a self-contained dist/ directory suitable for static
// hosting (GitHub Pages, Netlify, S3). Layout:
//
//   dist/
//     index.html              — copied from src/
//     app.js                  — copied from src/
//     vendor/                 — third-party + sibling-package bundles
//       cytoscape.min.js, dagre.min.js, …, echarts.min.js
//       engine.min.js         — bundled from @flatppl/engine
//       sampler-worker.min.js — bundled from @flatppl/engine (worker)
//       viewer.js             — copied from @flatppl/viewer source
//     demo/                   — hand-curated, gallery-owned .flatppl
//                               programs (e.g. feature tests)
//     examples/               — synced from sibling repo flatppl-examples
//                               on every build (sibling-first, GitHub
//                               tarball fallback) so the canonical public
//                               examples don't have to be duplicated
//                               into this repo
//     models.json             — auto-generated manifest covering both
//                               demo/ and examples/ (the gallery shell
//                               fetches it on boot)
//
// Mirrors packages/viewer/build.mjs structurally for the engine
// bundling step; the flatppl-examples sync mirrors the
// flatppl-grammars sync in packages/vscode-extension/build-vendor.mjs.
//
// Usage:
//   npm run build         # one-shot
//   npm run watch         # rebuild engine bundles on source changes

import { copyFile, mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import * as esbuild from 'esbuild';

const here     = dirname(fileURLToPath(import.meta.url));   // packages/web/
const repoRoot = dirname(dirname(here));                    // flatppl-js/
const enginePkg = join(repoRoot, 'packages', 'engine');
const viewerPkg = join(repoRoot, 'packages', 'viewer');
const distDir   = join(here, 'dist');
const vendorDir = join(distDir, 'vendor');
const nm        = join(repoRoot, 'node_modules');

// flatppl-examples sibling: the natural sibling layout where developers
// clone all FlatPPL repos next to each other. When this exists we copy
// from it directly; otherwise we fetch the pinned ref from GitHub. Same
// pattern as packages/vscode-extension/build-vendor.mjs uses for
// flatppl-grammars.
const examplesSibling = join(dirname(repoRoot), 'flatppl-examples');
// Pin to a branch (`main`) during dev; switch to a commit SHA or release
// tag for stable nightly deploys.
const examplesPin = 'main';
const examplesRemote = `https://github.com/flatppl/flatppl-examples/archive/refs/heads/${examplesPin}.tar.gz`;

const WATCH = process.argv.includes('--watch');

await mkdir(vendorDir, { recursive: true });

// ---------------------------------------------------------------------
// 1. Copy ready-made UMD/min bundles from node_modules into dist/vendor/.

const COPY_LIBS = [
  { pkg: 'cytoscape',            src: 'cytoscape/dist/cytoscape.min.js',             dst: 'cytoscape.min.js' },
  { pkg: '@dagrejs/dagre',       src: '@dagrejs/dagre/dist/dagre.min.js',            dst: 'dagre.min.js' },
  { pkg: 'cytoscape-dagre',      src: 'cytoscape-dagre/cytoscape-dagre.js',          dst: 'cytoscape-dagre.js' },
  { pkg: 'cytoscape-bubblesets', src: 'cytoscape-bubblesets/build/index.umd.min.js', dst: 'cytoscape-bubblesets.min.js' },
  { pkg: 'cytoscape-layers',     src: 'cytoscape-layers/build/index.umd.min.js',     dst: 'cytoscape-layers.min.js' },
  { pkg: 'echarts',              src: 'echarts/dist/echarts.min.js',                 dst: 'echarts.min.js' },
];

for (const { pkg, src, dst } of COPY_LIBS) {
  const from = join(nm, src);
  const to = join(vendorDir, dst);
  if (!existsSync(from)) {
    console.error(`  ! missing: ${pkg} (looked for ${src} under ${nm})`);
    process.exit(1);
  }
  await copyFile(from, to);
  console.log(`  copied ${pkg} -> dist/vendor/${dst}`);
}

// ---------------------------------------------------------------------
// 2. Copy the viewer source.

await copyFile(join(viewerPkg, 'src', 'viewer.js'), join(vendorDir, 'viewer.js'));
console.log('  copied @flatppl/viewer source -> dist/vendor/viewer.js');

// ---------------------------------------------------------------------
// 3. Copy the page entry-point and app sources from src/ to dist/.

// Files in src/ that are esbuild entry points (bundled into
// dist/vendor/...) rather than served as-is. Skip them in the
// flat copy to keep dist/ tidy and avoid serving ES-module
// source the page would never actually load.
const SRC_BUNDLE_ENTRIES = new Set(['codemirror-bundle-entry.js']);

const SRC_FILES = await readdir(join(here, 'src'));
for (const name of SRC_FILES) {
  if (SRC_BUNDLE_ENTRIES.has(name)) continue;
  await copyFile(join(here, 'src', name), join(distDir, name));
  console.log(`  copied src/${name} -> dist/${name}`);
}

// ---------------------------------------------------------------------
// 4. Hand-curated demo content (gallery-specific .flatppl programs,
//    feature tests, etc.) Recursive copy so subdirectories under demo/
//    are preserved.

if (existsSync(join(here, 'demo'))) {
  await rm(join(distDir, 'demo'), { recursive: true, force: true });
  await copyDirRecursive(join(here, 'demo'), join(distDir, 'demo'));
  console.log('  copied demo/ -> dist/demo/');
}

// ---------------------------------------------------------------------
// 5. Sync flatppl-examples (canonical public examples) into dist/examples/.
//    Sibling-clone first (fastest dev loop, no network), GitHub tarball
//    fallback otherwise (the CI workflow path — actions/checkout only
//    fetches flatppl-js, so the sibling is absent there).

await syncExamples();

// ---------------------------------------------------------------------
// 6. Generate dist/models.json from the assembled dist/{demo,examples}.
//    Demo entries first, then examples; each group sorted alphabetically.
//    Title falls back to the file basename without the .flatppl
//    extension. Future enhancement: allow a sidecar metadata file
//    (e.g. demo-meta.json) to override titles for hand-curated content.

await generateManifest();

// ---------------------------------------------------------------------
// 7. Bundle the FlatPPL engine and sampler-worker for browser loading.

const engineBuildOpts = {
  entryPoints: [join(enginePkg, 'index.js')],
  outfile: join(vendorDir, 'engine.min.js'),
  bundle: true,
  minify: true,
  format: 'iife',
  globalName: 'FlatPPLEngine',
  platform: 'browser',
  target: ['es2020'],
  legalComments: 'inline',
  footer: {
    js: 'if(typeof module!=="undefined"&&module.exports){module.exports=FlatPPLEngine;}'
       + 'if(typeof globalThis!=="undefined"){globalThis.FlatPPLEngine=FlatPPLEngine;}',
  },
};

const samplerWorkerBuildOpts = {
  entryPoints: [join(enginePkg, 'worker-entry.js')],
  outfile: join(vendorDir, 'sampler-worker.min.js'),
  bundle: true,
  minify: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2020'],
  legalComments: 'inline',
};

// CodeMirror 6 bundle for the playground editor. Loaded LAZILY by
// the gallery (only when __FLATPPL_CONFIG__.playground is true), so
// non-playground deploys never request it. We still always emit
// it — the flag flips at runtime, and a static-host deploy can't
// know in advance which deploys want the playground.
const codemirrorBuildOpts = {
  entryPoints: [join(here, 'src', 'codemirror-bundle-entry.js')],
  outfile: join(vendorDir, 'codemirror.min.js'),
  bundle: true,
  minify: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2020'],
  legalComments: 'inline',
};

if (WATCH) {
  const engineCtx     = await esbuild.context(engineBuildOpts);
  const workerCtx     = await esbuild.context(samplerWorkerBuildOpts);
  const codemirrorCtx = await esbuild.context(codemirrorBuildOpts);
  await Promise.all([engineCtx.rebuild(), workerCtx.rebuild(), codemirrorCtx.rebuild()]);
  console.log('  bundled engine        -> dist/vendor/engine.min.js');
  console.log('  bundled sampler-worker -> dist/vendor/sampler-worker.min.js');
  console.log('  bundled codemirror     -> dist/vendor/codemirror.min.js');
  await Promise.all([engineCtx.watch(), workerCtx.watch(), codemirrorCtx.watch()]);
  console.log('  watching for changes (Ctrl+C to exit)…');
} else {
  await Promise.all([
    esbuild.build(engineBuildOpts),
    esbuild.build(samplerWorkerBuildOpts),
    esbuild.build(codemirrorBuildOpts),
  ]);
  console.log('  bundled engine        -> dist/vendor/engine.min.js');
  console.log('  bundled sampler-worker -> dist/vendor/sampler-worker.min.js');
  console.log('  bundled codemirror     -> dist/vendor/codemirror.min.js');
}

// ---------------------------------------------------------------------
// Helpers

async function copyDirRecursive(srcDir, dstDir) {
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

async function syncExamples() {
  const dst = join(distDir, 'examples');
  await rm(dst, { recursive: true, force: true });
  await mkdir(dst, { recursive: true });

  if (existsSync(examplesSibling)) {
    console.log(`  examples: using sibling clone at ${examplesSibling}`);
    const src = join(examplesSibling, 'examples');
    if (!existsSync(src)) {
      console.error(`  ! sibling exists but ${src} not found — flatppl-examples may have moved its content`);
      process.exit(1);
    }
    await copyDirRecursive(src, dst);
  } else {
    console.log(`  examples: fetching pinned ref '${examplesPin}' from GitHub`);
    await fetchAndExtractExamples(examplesRemote, dst);
  }
}

async function fetchAndExtractExamples(url, dstDir) {
  // Pull the tarball, write to a temp file, extract the whole archive
  // to a temp directory, then copy the `examples/` subdir into dstDir.
  // Same two-step "extract then copy" pattern the grammars sync uses
  // — avoids tar wildcard portability issues across GNU tar / bsdtar /
  // macOS / Windows MSYS / CI ubuntu-latest.
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`failed to fetch examples: ${res.status} ${res.statusText} ${url}`);
  }

  const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const tmpFile = join(tmpdir(), `flatppl-examples-${stamp}.tar.gz`);
  const tmpExtract = join(tmpdir(), `flatppl-examples-${stamp}`);

  try {
    const buf = Buffer.from(await res.arrayBuffer());
    await writeFile(tmpFile, buf);
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

    // GitHub archives unpack into a single top-level directory named
    // `<repo>-<ref>/`. Find it (the only entry) and copy its
    // examples/ subdir over.
    const entries = await readdir(tmpExtract);
    if (entries.length === 0) {
      throw new Error(`examples archive extracted empty: ${tmpExtract}`);
    }
    const topLevel = join(tmpExtract, entries[0]);
    const src = join(topLevel, 'examples');
    if (!existsSync(src)) {
      throw new Error(`expected examples/ subdir under ${topLevel}, not found`);
    }
    await copyDirRecursive(src, dstDir);
  } finally {
    await rm(tmpFile,    { force: true }).catch(() => {});
    await rm(tmpExtract, { recursive: true, force: true }).catch(() => {});
  }
}

/** Recursively list .flatppl files under rootDir, sorted alphabetically.
    Paths are relative to rootDir (forward slashes). */
async function listFlatpplFiles(rootDir) {
  const out = [];
  async function walk(dir, prefix) {
    if (!existsSync(dir)) return;
    const entries = await readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
      const childRelPath = prefix ? prefix + '/' + ent.name : ent.name;
      if (ent.isDirectory()) {
        await walk(join(dir, ent.name), childRelPath);
      } else if (ent.isFile() && ent.name.endsWith('.flatppl')) {
        out.push(childRelPath);
      }
    }
  }
  await walk(rootDir, '');
  out.sort();
  return out;
}

function basenameTitle(path) {
  const i = path.lastIndexOf('/');
  const name = i < 0 ? path : path.slice(i + 1);
  return name.replace(/\.flatppl$/, '');
}

async function generateManifest() {
  const demoFiles     = await listFlatpplFiles(join(distDir, 'demo'));
  const exampleFiles  = await listFlatpplFiles(join(distDir, 'examples'));

  const entries = [];
  for (const path of demoFiles) {
    entries.push({ path: 'demo/' + path,     title: basenameTitle(path) });
  }
  for (const path of exampleFiles) {
    entries.push({ path: 'examples/' + path, title: basenameTitle(path) });
  }

  const manifest = {
    title: 'FlatPPL examples',
    entries: entries,
  };
  await writeFile(
    join(distDir, 'models.json'),
    JSON.stringify(manifest, null, 2) + '\n'
  );
  console.log(`  wrote dist/models.json (${entries.length} entries: ${demoFiles.length} demo, ${exampleFiles.length} examples)`);
}
