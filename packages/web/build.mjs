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
import { existsSync, watch as fsWatch } from 'node:fs';
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
// 2. Viewer bundle. From Phase 4 the viewer lives as ES modules
//    (packages/viewer/src/index.js + main.js + future submodules);
//    esbuild bundles src/index.js → vendor/viewer.js in IIFE format,
//    with the global merge (`window.FlatPPLViewer = window.FlatPPLViewer
//    || {}`) and DOMContentLoaded auto-mount wired explicitly inside
//    index.js (no `globalName:` — that would clobber a pre-existing
//    object the host may have populated). `minify:false` keeps the
//    bundled output diffable in dev and CSP-clean (no eval).
//    Bundling happens below alongside the engine/worker/codemirror
//    builds, so all four bundles share the same `if (WATCH)` /
//    one-shot path.

// Toolbar icons. We pull from packages/vscode-extension/media/ so the
// gallery and the VS Code extension stay visually consistent — same
// icon set, single source of truth. Update the extension's SVG and
// the next gallery build picks it up.
const extMedia = join(repoRoot, 'packages', 'vscode-extension', 'media');
const mediaDst = join(distDir, 'media');
await mkdir(mediaDst, { recursive: true });
const SHARED_ICONS = ['visualize-module-dark.svg'];
for (const name of SHARED_ICONS) {
  const from = join(extMedia, name);
  if (!existsSync(from)) {
    console.error(`  ! missing shared icon: ${name} (looked under ${extMedia})`);
    process.exit(1);
  }
  await copyFile(from, join(mediaDst, name));
  console.log(`  copied ${name} -> dist/media/${name}`);
}

// ---------------------------------------------------------------------
// 3. Build the page entry-point and app sources from src/ into dist/.
//
// The page-shell <script> tags reference plain .js filenames (e.g.
// `<script src="app.js">`). Source modules live as .ts under src/ now
// (post-TypeScript migration); esbuild.transform strips the types and
// the resulting JS lands at dist/<name>.js so the HTML doesn't change.
// Non-.ts assets (index.html, style.css, …) are copied verbatim.

// Files in src/ that are esbuild entry points (bundled into
// dist/vendor/...) rather than served as-is. Skip them in the
// flat copy / transpile loop to keep dist/ tidy and avoid serving
// ES-module source the page would never actually load.
const SRC_BUNDLE_ENTRIES = new Set([
  'codemirror-bundle-entry.js',
  'codemirror-bundle-entry.ts',
]);

// Per-file transpile-or-copy: each script in src/ is loaded by the page
// as a plain <script src="...">, NOT as an ES module — so the build
// output must keep file boundaries (one .ts → one .js) rather than
// bundling cross-file imports. The IIFEs inside each file install
// their exports onto window.FlatPPL* globals; that contract stays
// intact because esbuild.transform only strips the types. Non-.ts
// assets (index.html, style.css, …) are copied verbatim. Returns
// false if the entry was skipped (bundle entry, .d.ts, missing file).
async function transpileOrCopySrcFile(name) {
  if (SRC_BUNDLE_ENTRIES.has(name)) return false;
  // Ambient declaration files have no runtime output — they're for tsc
  // only. Skip them so dist/ doesn't get a meaningless `.d.js` sibling.
  if (name.endsWith('.d.ts')) return false;
  const srcPath = join(here, 'src', name);
  if (!existsSync(srcPath)) return false;
  if (name.endsWith('.ts')) {
    const src = await readFile(srcPath, 'utf8');
    const result = await esbuild.transform(src, {
      loader: 'ts',
      target: 'es2020',
      sourcefile: name,
    });
    const outName = name.slice(0, -3) + '.js';
    await writeFile(join(distDir, outName), result.code);
    console.log(`  transpiled src/${name} -> dist/${outName}`);
  } else {
    await copyFile(srcPath, join(distDir, name));
    console.log(`  copied src/${name} -> dist/${name}`);
  }
  return true;
}

const SRC_FILES = await readdir(join(here, 'src'));
for (const name of SRC_FILES) {
  await transpileOrCopySrcFile(name);
}

// ---------------------------------------------------------------------
// 4. Hand-curated demo content (gallery-specific .flatppl programs,
//    feature tests, etc.) Recursive copy so subdirectories under demo/
//    are preserved.

async function syncDemoTree({ clean = true } = {}) {
  if (!existsSync(join(here, 'demo'))) return;
  // Avoid removing dist/demo when called from the watcher — serve.mjs
  // holds a recursive fs.watch on dist/demo for the live-reload SSE
  // channel, and that watcher dies (ENOENT) if its target briefly
  // disappears. At startup the directory may already have been
  // cleaned out together with the rest of dist/ (`rm -rf dist`), so
  // a fresh full build still starts from scratch.
  if (clean) await rm(join(distDir, 'demo'), { recursive: true, force: true });
  await copyDirRecursive(join(here, 'demo'), join(distDir, 'demo'));
  console.log('  copied demo/ -> dist/demo/');
}

// Skip the rm at startup when running under --watch — serve.mjs's
// recursive watcher on dist/ doesn't survive `rm -rf` of any
// subdirectory. The watcher rebuild path passes `clean: false`
// for the same reason.
await syncDemoTree({ clean: !WATCH });

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

// CodeMirror 6 bundle for the playground editor. Loaded LAZILY by
// the gallery (only when __FLATPPL_CONFIG__.playground is true), so
// non-playground deploys never request it. We still always emit
// it — the flag flips at runtime, and a static-host deploy can't
// know in advance which deploys want the playground.
const codemirrorBuildOpts = {
  entryPoints: [join(here, 'src', 'codemirror-bundle-entry.ts')],
  outfile: join(vendorDir, 'codemirror.min.js'),
  bundle: true,
  minify: true,
  format: 'iife',
  platform: 'browser',
  target: ['es2020'],
  legalComments: 'inline',
};

const viewerBuildOpts = {
  entryPoints: [join(viewerPkg, 'src', 'index.ts')],
  outfile: join(vendorDir, 'viewer.js'),
  bundle: true,
  // minify:false — the viewer bundle is reviewed/dev-diffed routinely
  // and the host webview's CSP forbids eval, which some minify
  // transforms can emit. Same trade-off ARCHITECTURE.md notes for
  // dev-diffability.
  minify: false,
  format: 'iife',
  // No globalName: src/index.js does an explicit
  // `window.FlatPPLViewer = window.FlatPPLViewer || {}` merge so a
  // host that pre-populates the namespace isn't clobbered.
  platform: 'browser',
  target: ['es2020'],
  legalComments: 'inline',
};

if (WATCH) {
  const engineCtx     = await esbuild.context(engineBuildOpts);
  const workerCtx     = await esbuild.context(samplerWorkerBuildOpts);
  const codemirrorCtx = await esbuild.context(codemirrorBuildOpts);
  const viewerCtx     = await esbuild.context(viewerBuildOpts);
  await Promise.all([engineCtx.rebuild(), workerCtx.rebuild(),
                     codemirrorCtx.rebuild(), viewerCtx.rebuild()]);
  console.log('  bundled engine        -> dist/vendor/engine.min.js');
  console.log('  bundled sampler-worker -> dist/vendor/sampler-worker.min.js');
  console.log('  bundled codemirror     -> dist/vendor/codemirror.min.js');
  console.log('  bundled viewer        -> dist/vendor/viewer.js');
  await Promise.all([engineCtx.watch(), workerCtx.watch(),
                     codemirrorCtx.watch(), viewerCtx.watch()]);

  // esbuild's context.watch() only re-fires the four bundles above.
  // The src/ transpile-or-copy loop and the demo/ tree copy run ONCE
  // at startup, so without these additional fs.watch handlers edits
  // to index.html, style.css, src/*.ts, demo/*.flatppl, etc. would
  // not propagate to dist/ — and the dev server (which serves dist/)
  // would keep showing stale content. We watch src/ flat (no nested
  // dirs in practice) and demo/ recursive (it contains the curated
  // .flatppl programs). Both are debounced with a tiny coalescing
  // window because fs.watch can fire multiple events per save
  // (rename + change, editor swap-file pattern, etc.).
  watchDirDebounced(join(here, 'src'), { recursive: false }, (filename) => {
    if (!filename) return;
    transpileOrCopySrcFile(filename).catch((err) => {
      console.error(`  ! src/${filename} rebuild failed:`, err.message);
    });
  });
  watchDirDebounced(join(here, 'demo'), { recursive: true }, async () => {
    // demo/ tree changed — re-sync the whole tree (it's small and
    // hand-curated) and regenerate models.json since the manifest
    // lists every .flatppl file under demo/. `clean: false` skips
    // the rm step so serve.mjs's recursive watcher on dist/demo
    // doesn't see its target vanish mid-rebuild. Stale files
    // wouldn't be pruned this way, but the dev workflow doesn't
    // create them often, and a full `npm run build` always starts
    // from a clean slate.
    try {
      await syncDemoTree({ clean: false });
      await generateManifest();
    } catch (err) {
      console.error('  ! demo/ resync failed:', err.message);
    }
  });
  console.log('  watching for changes (Ctrl+C to exit)…');
} else {
  await Promise.all([
    esbuild.build(engineBuildOpts),
    esbuild.build(samplerWorkerBuildOpts),
    esbuild.build(codemirrorBuildOpts),
    esbuild.build(viewerBuildOpts),
  ]);
  console.log('  bundled engine        -> dist/vendor/engine.min.js');
  console.log('  bundled sampler-worker -> dist/vendor/sampler-worker.min.js');
  console.log('  bundled codemirror     -> dist/vendor/codemirror.min.js');
  console.log('  bundled viewer        -> dist/vendor/viewer.js');
}

// ---------------------------------------------------------------------
// Helpers

// Wrap fs.watch with a small per-path debounce. fs.watch on Linux can
// emit two events (rename + change) for a single save; editors that
// swap files via tmp + rename can emit even more. Coalescing into a
// single trailing-edge callback keeps the rebuild log readable and
// avoids racing transpile/copy operations on the same file.
function watchDirDebounced(dir, options, onChange) {
  if (!existsSync(dir)) return;
  const timers = new Map();
  const DELAY_MS = 40;
  fsWatch(dir, options, (_eventType, filename) => {
    // filename can be null on some platforms; we still dispatch with
    // null so directory-scope handlers (like demo/) re-sync wholesale.
    const key = filename || '';
    const existing = timers.get(key);
    if (existing) clearTimeout(existing);
    timers.set(key, setTimeout(() => {
      timers.delete(key);
      try {
        onChange(filename);
      } catch (err) {
        console.error(`  ! watcher error for ${dir}/${filename}:`, err.message);
      }
    }, DELAY_MS));
  });
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

async function syncExamples() {
  const dst = join(distDir, 'examples');
  // When running under --watch alongside serve.mjs (npm run dev),
  // serve.mjs holds a recursive fs.watch on the dist/ tree for live-
  // reload. `rm -rf` on dist/examples kills that watcher (ENOENT
  // scandir) — same trap that syncDemoTree avoids in watch mode.
  // In watch mode we therefore merge-copy without the prior rm.
  // One-shot builds (`npm run build`) start from `rm -rf dist`
  // outside this script anyway, so cleanliness isn't lost.
  if (!WATCH) {
    await rm(dst, { recursive: true, force: true });
  }
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

/** Recursively list FlatPPL surface-variant files under rootDir,
    sorted alphabetically. Paths are relative to rootDir (forward
    slashes). Matches all three of the spec's surface variants —
    .flatppl (canonical), .flatppy (Python-AST), .flatppj (Julia-AST)
    — so a gallery deploy surfaces every source the engine can parse. */
async function listFlatpplLikeFiles(rootDir) {
  const VARIANT_EXTENSIONS = ['.flatppl', '.flatppy', '.flatppj'];
  const out = [];
  async function walk(dir, prefix) {
    if (!existsSync(dir)) return;
    const entries = await readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
      const childRelPath = prefix ? prefix + '/' + ent.name : ent.name;
      if (ent.isDirectory()) {
        await walk(join(dir, ent.name), childRelPath);
      } else if (ent.isFile()
                 && VARIANT_EXTENSIONS.some((x) => ent.name.endsWith(x))) {
        out.push(childRelPath);
      }
    }
  }
  await walk(rootDir, '');
  out.sort();
  return out;
}

// Keep the extension in the gallery title — the variant is part of
// what users want to identify the file by ("minimal.flatppl" vs
// "minimal.flatppy" vs "minimal.flatppj" all live side-by-side, and
// the extension is the only thing distinguishing them).
function basenameTitle(path) {
  const i = path.lastIndexOf('/');
  return i < 0 ? path : path.slice(i + 1);
}

async function generateManifest() {
  const demoFiles     = await listFlatpplLikeFiles(join(distDir, 'demo'));
  const exampleFiles  = await listFlatpplLikeFiles(join(distDir, 'examples'));

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
