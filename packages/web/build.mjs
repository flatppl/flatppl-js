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
//     demo/                   — example .flatppl files (added later)
//     models.json             — manifest the gallery fetches at startup
//                               (added later)
//
// Mirrors packages/viewer/build.mjs structurally — the two scripts
// stay independent for now (each owns its own dist/vendor/) so either
// can change without touching the other. If the duplication grows,
// a workspace-shared bundling helper makes sense.
//
// Usage:
//   npm run build         # one-shot
//   npm run watch         # rebuild engine bundles on source changes

import { copyFile, mkdir, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const here     = dirname(fileURLToPath(import.meta.url));   // packages/web/
const repoRoot = dirname(dirname(here));                    // flatppl-js/
const enginePkg = join(repoRoot, 'packages', 'engine');
const viewerPkg = join(repoRoot, 'packages', 'viewer');
const distDir   = join(here, 'dist');
const vendorDir = join(distDir, 'vendor');
const nm        = join(repoRoot, 'node_modules');

const WATCH = process.argv.includes('--watch');

await mkdir(vendorDir, { recursive: true });

// ---------------------------------------------------------------------
// 1. Copy ready-made UMD/min bundles from node_modules into dist/vendor/.
//    Same set the viewer's build copies — the two builds are independent
//    but trade nothing by sharing the source list.

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
// 2. Copy the viewer source. Loaded as <script src="vendor/viewer.js">
//    next to the rest of the bundle.

await copyFile(join(viewerPkg, 'src', 'viewer.js'), join(vendorDir, 'viewer.js'));
console.log('  copied @flatppl/viewer source -> dist/vendor/viewer.js');

// ---------------------------------------------------------------------
// 3. Copy the page entry-point and app sources from src/ to dist/.
//    Plain copy for now; later steps may add a CSS file and helper
//    modules (syntax.js, resolver.js, …) that join the same loop.

const SRC_FILES = await readdir(join(here, 'src'));
for (const name of SRC_FILES) {
  await copyFile(join(here, 'src', name), join(distDir, name));
  console.log(`  copied src/${name} -> dist/${name}`);
}

// ---------------------------------------------------------------------
// 4. Bundle the FlatPPL engine and sampler-worker for browser loading.
//    Same esbuild config the viewer uses — IIFE, minified, browser
//    target. Engine exposes globalThis.FlatPPLEngine; the worker
//    registers self.onmessage side-effects.

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

if (WATCH) {
  const engineCtx = await esbuild.context(engineBuildOpts);
  const workerCtx = await esbuild.context(samplerWorkerBuildOpts);
  await Promise.all([engineCtx.rebuild(), workerCtx.rebuild()]);
  console.log('  bundled engine        -> dist/vendor/engine.min.js');
  console.log('  bundled sampler-worker -> dist/vendor/sampler-worker.min.js');
  await Promise.all([engineCtx.watch(), workerCtx.watch()]);
  console.log('  watching packages/engine/ for changes (Ctrl+C to exit)…');
} else {
  await Promise.all([
    esbuild.build(engineBuildOpts),
    esbuild.build(samplerWorkerBuildOpts),
  ]);
  console.log('  bundled engine        -> dist/vendor/engine.min.js');
  console.log('  bundled sampler-worker -> dist/vendor/sampler-worker.min.js');
}
