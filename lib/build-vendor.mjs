// Build all third-party JS into lib/. Outputs are UMD/IIFE-loadable so
// they work in both the extension host (Node) and the webview (browser),
// which keeps the door open for VS Code Web. Run once after `npm install`
// and whenever a vendored dependency is updated.
//
//   npm install
//   npm run build:vendor
//
// The CI workflow runs the same two commands before `vsce package` so the
// nightly vsix bundles fresh copies. lib/ build outputs are .gitignored;
// only this script (and any future build inputs) live in the repo.

import { readFile, writeFile, copyFile, readdir } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);
const libDir = here;
const nm = join(root, 'node_modules');

// ---------------------------------------------------------------------
// 1. Copy ready-made UMD/min bundles from node_modules into lib/.
//    Each entry is { src: path-relative-to-node_modules, dst: filename
//    visualPanel.js expects in lib/ }.

const COPY_LIBS = [
  { pkg: 'cytoscape',            src: 'cytoscape/dist/cytoscape.min.js',          dst: 'cytoscape.min.js' },
  { pkg: '@dagrejs/dagre',       src: '@dagrejs/dagre/dist/dagre.min.js',         dst: 'dagre.min.js' },
  { pkg: 'cytoscape-dagre',      src: 'cytoscape-dagre/cytoscape-dagre.js',       dst: 'cytoscape-dagre.js' },
  { pkg: 'cytoscape-bubblesets', src: 'cytoscape-bubblesets/build/index.umd.min.js', dst: 'cytoscape-bubblesets.min.js' },
  { pkg: 'cytoscape-layers',     src: 'cytoscape-layers/build/index.umd.min.js',  dst: 'cytoscape-layers.min.js' },
  { pkg: 'echarts',              src: 'echarts/dist/echarts.min.js',              dst: 'echarts.min.js' },
];

for (const { pkg, src, dst } of COPY_LIBS) {
  const from = join(nm, src);
  const to = join(libDir, dst);
  if (!existsSync(from)) {
    console.error(`  ! missing: ${pkg} (looked for ${src})`);
    process.exit(1);
  }
  await copyFile(from, to);
  console.log(`  copied ${pkg} -> lib/${dst}`);
}

// ---------------------------------------------------------------------
// 2. Generate THIRD-PARTY-LICENSES.md from the actual bundled deps.
//    Walks the lib copies above and concatenates each package's LICENSE
//    file with attribution headers, so the file in the repo always
//    matches what's bundled into the vsix.

const bundledNames = new Set(COPY_LIBS.map(c => c.pkg));

let out = '# Third-Party Licenses\n\n';
out += 'This extension bundles the following third-party libraries into the\n';
out += 'distributed `.vsix`. Each retains its original license. Generated\n';
out += 'automatically from `node_modules/` by `lib/build-vendor.mjs`; do not\n';
out += 'edit by hand — re-run `npm run build:vendor` instead.\n\n';

const sorted = [...bundledNames].sort();
for (const name of sorted) {
  const pkgDir = join(nm, name);
  if (!existsSync(pkgDir)) continue;
  const meta = JSON.parse(await readFile(join(pkgDir, 'package.json'), 'utf8'));
  const license = meta.license || '(unspecified)';
  const homepage = meta.homepage || meta.repository?.url || '';
  const licText = await findLicenseText(pkgDir);
  out += '---\n\n';
  out += `## ${name}\n\n`;
  out += `- **Version:** ${meta.version}\n`;
  out += `- **License:** ${license}\n`;
  if (homepage) out += `- **Source:** ${homepage}\n`;
  out += '\n';
  if (licText) out += licText.trim() + '\n\n';
}

await writeFile(join(root, 'THIRD-PARTY-LICENSES.md'), out);
console.log('  wrote THIRD-PARTY-LICENSES.md');

// ---------------------------------------------------------------------

async function findLicenseText(pkgDir) {
  const files = await readdir(pkgDir);
  const candidates = files.filter(f => /^LICEN[CS]E(\..*)?$/i.test(f));
  if (candidates.length === 0) return '';
  // Prefer LICENSE without an extension over LICENSE.md etc.
  candidates.sort((a, b) => a.length - b.length);
  return await readFile(join(pkgDir, candidates[0]), 'utf8');
}
