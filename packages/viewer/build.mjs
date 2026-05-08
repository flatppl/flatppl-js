// @flatppl/viewer — vendor build.
//
// Populates packages/viewer/vendor/ with the JS the embed page (and any
// online host) needs alongside viewer.js:
//
//   - cytoscape / dagre / cytoscape-dagre / cytoscape-bubblesets /
//     cytoscape-layers / echarts        — copied from hoisted node_modules
//   - engine.min.js                     — bundled from packages/engine
//   - sampler-worker.min.js             — bundled from packages/engine
//   - viewer.js                          — copied from src/viewer.js
//
// This mirrors what packages/vscode-extension/build-vendor.mjs already does
// (and we deliberately keep the two scripts independent for now —
// neither owns the other's lib/vendor dir, and either can change its
// pipeline without touching the other). Later we can factor the shared
// bundling logic into a workspace-shared script if it pays for itself.
//
// Usage:
//   npm run build         # one-shot
//   npm run watch         # rebuild on engine source changes
//
// After the build, the embed-test.html page can be served from
// packages/viewer/ (e.g. via `npm run serve`) with no extra setup —
// no symlinks, no hand-copying.

import { copyFile, mkdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const here     = dirname(fileURLToPath(import.meta.url));      // packages/viewer/
const repoRoot = dirname(dirname(here));                       // flatppl-js/
const enginePkg = join(repoRoot, 'packages', 'engine');        // packages/engine/
const vendorDir = join(here, 'vendor');
const nm        = join(repoRoot, 'node_modules');              // hoisted by npm workspaces

const WATCH = process.argv.includes('--watch');

await mkdir(vendorDir, { recursive: true });

// ---------------------------------------------------------------------
// 1. Copy ready-made UMD/min bundles from node_modules into vendor/.
//    Same list as the vscode-extension build; the embed page references
//    each by its canonical filename via <script src="vendor/...">.

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
  console.log(`  copied ${pkg} -> vendor/${dst}`);
}

// ---------------------------------------------------------------------
// 2. Copy the viewer source. The embed page loads it as
//    <script src="vendor/viewer.js"> alongside the other vendored
//    bundles. Source-of-truth lives at src/viewer.js; the copy in
//    vendor/ is a build artifact.

await copyFile(join(here, 'src', 'viewer.js'), join(vendorDir, 'viewer.js'));
console.log('  copied viewer source -> vendor/viewer.js');

// ---------------------------------------------------------------------
// 3. Bundle the FlatPPL engine and sampler-worker for browser loading.
//    Same esbuild config the vscode-extension uses — IIFE format,
//    minified, browser target. The engine bundle exports
//    `globalThis.FlatPPLEngine`; the worker registers self.onmessage
//    side-effects.

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
  console.log('  bundled engine        -> vendor/engine.min.js');
  console.log('  bundled sampler-worker -> vendor/sampler-worker.min.js');
  await Promise.all([engineCtx.watch(), workerCtx.watch()]);
  console.log('  watching packages/engine/ for changes (Ctrl+C to exit)…');
} else {
  await Promise.all([
    esbuild.build(engineBuildOpts),
    esbuild.build(samplerWorkerBuildOpts),
  ]);
  console.log('  bundled engine        -> vendor/engine.min.js');
  console.log('  bundled sampler-worker -> vendor/sampler-worker.min.js');
}
