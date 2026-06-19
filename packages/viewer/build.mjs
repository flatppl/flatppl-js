// @flatppl/viewer — vendor build.
//
// Populates packages/viewer/vendor/ with the JS the embed page (and any
// online host) needs alongside viewer.js:
//
//   - cytoscape / dagre / cytoscape-dagre / cytoscape-bubblesets /
//     cytoscape-layers / echarts        — copied from hoisted node_modules
//   - engine.min.js                     — bundled from packages/engine
//   - sampler-worker.min.js             — bundled from packages/engine
//   - viewer.js                          — bundled from src/index.js (ES modules → IIFE)
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
  { pkg: 'cytoscape-dagre',      src: 'cytoscape-dagre/dist/cytoscape-dagre.js',     dst: 'cytoscape-dagre.js' },
  { pkg: 'cytoscape-bubblesets', src: 'cytoscape-bubblesets/build/index.umd.min.js', dst: 'cytoscape-bubblesets.min.js' },
  { pkg: 'cytoscape-layers',     src: 'cytoscape-layers/build/index.umd.min.js',     dst: 'cytoscape-layers.min.js' },
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
// 2. Bundle the FlatPPL engine, sampler-worker, and viewer entry for
//    browser loading. The viewer is bundled (not copied) from Phase 4
//    onwards: src/index.js imports src/main.js + ~20 sibling ES modules,
//    and esbuild collapses them into a single IIFE the host can load
//    via <script src="vendor/viewer.js"> (no module-loader needed). The
//    embed page loads it alongside the other vendored bundles. Engine
//    and sampler-worker mirror the vscode-extension's bundling config.

// Custom echarts build (replaces the full-UMD copy).
// Single-sources the entry from packages/vscode-extension/echarts-entry.mjs
// so the chart/component registration list is maintained in one place.
const echartsCustomBuildOpts = {
  entryPoints: [join(repoRoot, 'packages', 'vscode-extension', 'echarts-entry.mjs')],
  outfile: join(vendorDir, 'echarts.min.js'),
  bundle: true,
  minify: true,
  format: 'iife',
  globalName: 'echarts',
  platform: 'browser',
  target: ['es2020'],
};

const engineBuildOpts = {
  entryPoints: [join(enginePkg, 'index.ts')],
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
  entryPoints: [join(enginePkg, 'worker-entry.ts')],
  outfile: join(vendorDir, 'sampler-worker.min.js'),
  bundle: true,
  minify: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2020'],
  legalComments: 'inline',
};

const viewerBuildOpts = {
  entryPoints: [join(here, 'src', 'index.ts')],
  outfile: join(vendorDir, 'viewer.js'),
  bundle: true,
  // minify:false — see ARCHITECTURE.md; viewer bundle stays diffable
  // and the webview's CSP forbids eval (some minify transforms emit).
  // Matches vscode-extension/build-vendor.mjs's choice exactly.
  minify: false,
  format: 'iife',
  // No globalName: src/index.js does an explicit
  // `window.FlatPPLViewer = window.FlatPPLViewer || {}` merge so a host
  // that pre-populates the namespace isn't clobbered.
  platform: 'browser',
  target: ['es2020'],
  legalComments: 'inline',
};

if (WATCH) {
  const engineCtx  = await esbuild.context(engineBuildOpts);
  const workerCtx  = await esbuild.context(samplerWorkerBuildOpts);
  const viewerCtx  = await esbuild.context(viewerBuildOpts);
  const echartsCtx = await esbuild.context(echartsCustomBuildOpts);
  await Promise.all([engineCtx.rebuild(), workerCtx.rebuild(), viewerCtx.rebuild(),
                     echartsCtx.rebuild()]);
  console.log('  bundled engine        -> vendor/engine.min.js');
  console.log('  bundled sampler-worker -> vendor/sampler-worker.min.js');
  console.log('  bundled viewer        -> vendor/viewer.js');
  console.log('  bundled echarts       -> vendor/echarts.min.js');
  await Promise.all([engineCtx.watch(), workerCtx.watch(), viewerCtx.watch(),
                     echartsCtx.watch()]);
  console.log('  watching packages/engine/ and packages/viewer/src/ for changes (Ctrl+C to exit)…');
} else {
  await Promise.all([
    esbuild.build(engineBuildOpts),
    esbuild.build(samplerWorkerBuildOpts),
    esbuild.build(viewerBuildOpts),
    esbuild.build(echartsCustomBuildOpts),
  ]);
  console.log('  bundled engine        -> vendor/engine.min.js');
  console.log('  bundled sampler-worker -> vendor/sampler-worker.min.js');
  console.log('  bundled viewer        -> vendor/viewer.js');
  console.log('  bundled echarts       -> vendor/echarts.min.js');
}
