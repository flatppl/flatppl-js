// Build all third-party JS, the FlatPPL engine bundle, and synced grammars
// into ./lib/ and ./grammars/ — the artifacts that get packaged into the
// VS Code extension's .vsix. Outputs are UMD/IIFE-loadable so they work in
// both the extension host (Node) and the webview (browser), keeping the
// door open for VS Code Web. Run once after `npm install` and whenever a
// vendored dependency, the engine source, or the grammar source changes.
//
//   npm install
//   npm run build:vendor      # one-shot build (everything)
//   npm run watch:vendor      # rebuilds engine bundle on engine/ changes
//
// The CI workflow runs `npm ci && npm run build:vendor` before vsce package
// so the nightly vsix bundles fresh copies. lib/, grammars/, and the
// generated THIRD-PARTY-LICENSES.md are .gitignored; only this script
// (and its inputs in node_modules / sibling repos) live in the repo.
//
// Watch mode is engine-source-only: the third-party copies, grammar sync,
// and license generation only happen on first run. Re-run `npm run
// build:vendor` (no watch) when those inputs change.
//
// Repo layout this script assumes:
//
//   flatppl-js/
//   ├── packages/
//   │   ├── engine/                       — @flatppl/engine source
//   │   └── vscode-extension/             — this package
//   │       ├── build-vendor.mjs          — this script
//   │       ├── lib/                      — outputs (gitignored)
//   │       └── grammars/textmate/        — synced from flatppl-grammars
//   └── node_modules/                     — workspace-hoisted devDeps
//
// `flatppl-grammars` is a sibling repo at ../../../flatppl-grammars when
// cloned alongside this one. If absent (CI), we fetch the pinned ref from
// github.com/flatppl/flatppl-grammars instead.

import { readFile, writeFile, copyFile, readdir, mkdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import * as esbuild from 'esbuild';

const here     = dirname(fileURLToPath(import.meta.url));   // packages/vscode-extension/
const repoRoot = dirname(dirname(here));                     // flatppl-js/
const enginePkg = join(repoRoot, 'packages', 'engine');      // packages/engine/
const libDir   = join(here, 'lib');
const gramDir  = join(here, 'grammars');
const nm       = join(repoRoot, 'node_modules');             // hoisted by npm workspaces

// flatppl-grammars sibling: the natural sibling layout where developers
// clone all FlatPPL repos next to each other. When this exists we copy
// from it directly; otherwise we fetch the pinned ref from GitHub.
const grammarsSibling = join(dirname(repoRoot), 'flatppl-grammars');
// Pin to a branch (`main`) during dev; switch to a commit SHA or release tag
// for stable nightly builds. The GitHub archive URL has separate forms for
// branches/tags and bare refs; we use `refs/heads/<branch>` here, which
// is the canonical form GitHub documents and which `<bare-ref>.tar.gz`
// shortcuts no longer reliably support.
const grammarsPin = 'main';
const grammarsRemote = `https://github.com/flatppl/flatppl-grammars/archive/refs/heads/${grammarsPin}.tar.gz`;

const WATCH = process.argv.includes('--watch');

await mkdir(libDir, { recursive: true });
await mkdir(gramDir, { recursive: true });

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
    console.error(`  ! missing: ${pkg} (looked for ${src} under ${nm})`);
    process.exit(1);
  }
  await copyFile(from, to);
  console.log(`  copied ${pkg} -> lib/${dst}`);
}

// ---------------------------------------------------------------------
// 2. Sync grammars from flatppl-grammars (host-neutral artifact repo).
//    Sibling-clone first (fastest dev loop), GitHub fetch otherwise.

await syncGrammars();

// ---------------------------------------------------------------------
// 3. Bundle the FlatPPL engine into a single browser-loadable IIFE so the
//    webview can run the same parser/analyzer/DAG logic the extension
//    host uses. The engine source lives in ../engine/ (workspace sibling)
//    and is pure-JS CommonJS; esbuild handles the CJS→IIFE shape. Output
//    exports a global `FlatPPLEngine` matching engine/index.js's exports.
//    A footer wires module.exports too so the same bundle can be Node-
//    required (tests, etc.) and browser-loaded uniformly.

const engineBuildOpts = {
  entryPoints: [join(enginePkg, 'index.js')],
  outfile: join(libDir, 'engine.min.js'),
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

if (WATCH) {
  // Watch mode: rebuild engine.min.js on every change under packages/engine/.
  // The process keeps running until killed (Ctrl+C). Reload the VS Code
  // window after a rebuild to pick up the new bundle in the webview.
  const ctx = await esbuild.context(engineBuildOpts);
  await ctx.rebuild();
  console.log('  bundled engine -> lib/engine.min.js');
  await ctx.watch();
  console.log('  watching packages/engine/ for changes (Ctrl+C to exit)…');
} else {
  await esbuild.build(engineBuildOpts);
  console.log('  bundled engine -> lib/engine.min.js');
}

// ---------------------------------------------------------------------
// 4. Generate THIRD-PARTY-LICENSES.md from the actually bundled deps.
//    Walks node_modules for each package whose UMD we copied, plus the
//    engine source location, and concatenates each LICENSE file with
//    attribution headers. The file lands at the package root (where vsce
//    will pick it up); .gitignored, regenerated on every build.

const bundledNames = new Set(COPY_LIBS.map(c => c.pkg));

let out = '# Third-Party Licenses\n\n';
out += 'This extension bundles the following third-party libraries into the\n';
out += 'distributed `.vsix`. Each retains its original license. Generated\n';
out += 'automatically from `node_modules/` by `build-vendor.mjs`; do not\n';
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

await writeFile(join(here, 'THIRD-PARTY-LICENSES.md'), out);
console.log('  wrote THIRD-PARTY-LICENSES.md');

// ---------------------------------------------------------------------

async function syncGrammars() {
  // Always start with a clean grammars/ so deletions in the source repo
  // propagate. (Cheap — only a few small JSON files.)
  await rm(gramDir, { recursive: true, force: true });
  await mkdir(join(gramDir, 'textmate'), { recursive: true });

  if (existsSync(grammarsSibling)) {
    console.log(`  grammars: using sibling clone at ${grammarsSibling}`);
    await copyDirRecursive(join(grammarsSibling, 'textmate'), join(gramDir, 'textmate'));
  } else {
    console.log(`  grammars: fetching pinned ref '${grammarsPin}' from GitHub`);
    await fetchAndExtractGrammars(grammarsRemote, gramDir);
  }
}

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

async function fetchAndExtractGrammars(url, dstDir) {
  // Pull the tarball, write to a temp file, extract the whole archive to a
  // temp directory, then copy the `textmate/` subdir into dstDir. This
  // two-step "extract then copy" approach avoids portability issues with
  // tar's wildcard handling — GNU tar requires `--wildcards` to expand
  // patterns in member names, while bsdtar / macOS tar handles them
  // differently. Letting tar extract everything and using JS to pick the
  // wanted subdir works the same on every platform.
  const { tmpdir } = await import('node:os');
  const { spawn } = await import('node:child_process');

  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) {
    throw new Error(`failed to fetch grammars: ${res.status} ${res.statusText} ${url}`);
  }

  const stamp = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  const tmpFile = join(tmpdir(), `flatppl-grammars-${stamp}.tar.gz`);
  const tmpExtract = join(tmpdir(), `flatppl-grammars-${stamp}`);

  try {
    // Buffer the response to disk. Tarballs from GitHub archives are small
    // (<1 MB for grammar repos), so a single in-memory ArrayBuffer is fine.
    const buf = Buffer.from(await res.arrayBuffer());
    await writeFile(tmpFile, buf);

    // Extract the whole archive to a temp directory. tar -xzf is universally
    // supported across GNU tar, bsdtar, macOS, Windows-with-MSYS, and the
    // CI ubuntu-latest runners.
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
    // `<repo>-<ref>/`. Find it (it's the only entry) and copy its
    // textmate/ subdir over.
    const entries = await readdir(tmpExtract);
    if (entries.length === 0) {
      throw new Error(`grammar archive extracted empty: ${tmpExtract}`);
    }
    const topLevel = join(tmpExtract, entries[0]);
    await copyDirRecursive(join(topLevel, 'textmate'), join(dstDir, 'textmate'));
  } finally {
    // Best-effort cleanup; leave artifacts in place if anything errors.
    await rm(tmpFile,    { force: true }).catch(() => {});
    await rm(tmpExtract, { recursive: true, force: true }).catch(() => {});
  }
}

async function findLicenseText(pkgDir) {
  const files = await readdir(pkgDir);
  const candidates = files.filter(f => /^LICEN[CS]E(\..*)?$/i.test(f));
  if (candidates.length === 0) return '';
  // Prefer LICENSE without an extension over LICENSE.md etc.
  candidates.sort((a, b) => a.length - b.length);
  return await readFile(join(pkgDir, candidates[0]), 'utf8');
}
