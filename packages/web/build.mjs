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

import { copyFile, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { existsSync, statSync, watch as fsWatch } from 'node:fs';
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

// Model/representation file types the gallery surfaces (Phase 1: FlatPPL +
// Markdown). The surface registry (src/surfaces.ts) routes each to its
// right-pane surface; KEEP THIS IN STEP with surfaces.ts `typeForPath`.
// Later types (pyhf / HS3 / Stan / Julia / Python) are a one-line
// extension here once their surfaces (or the convert command) land.
// Declared at top-level so it's initialized before the manifest pass runs.
const MODEL_EXTENSIONS = ['.flatppl', '.md', '.markdown', '.hs3.json', '.pyhf.json'];

// flatppl-wasm-api convert artifact (the `wasm-pack --target web` output:
// an ESM glue + the wasm binary). Provisioned into dist/vendor/ by
// provisionWasmConvert() — sibling-build or download, mirroring the LSP.
// Absent is fine: the gallery hides the Convert command when it's missing.
const WASM_GLUE = 'flatppl_wasm_api.js';
const WASM_BIN  = 'flatppl_wasm_api_bg.wasm';
const WASM_RELEASE_BASE =
  'https://github.com/flatppl/flatppl-rust/releases/download/nightly';

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

// flatppl-grammars sibling: same sibling-first / GitHub-tarball pattern as the
// examples sync. Provides the canonical TextMate grammar AND the CodeMirror
// highlighter module (codemirror/textmate-highlight.ts).
// GRAMMARS_DIR (set by CI to a checked-out flatppl-grammars) takes precedence;
// otherwise the natural sibling layout for local dev. Tarball fallback if neither.
const grammarsSibling = process.env.GRAMMARS_DIR || join(dirname(repoRoot), 'flatppl-grammars');
const grammarsPin = 'main';
const grammarsRemote = `https://github.com/flatppl/flatppl-grammars/archive/refs/heads/${grammarsPin}.tar.gz`;
const srcVendorDir = join(here, 'src', 'vendor');

const WATCH = process.argv.includes('--watch');

await mkdir(vendorDir, { recursive: true });

// ---------------------------------------------------------------------
// 1. Copy ready-made UMD/min bundles from node_modules into dist/vendor/.

const COPY_LIBS = [
  { pkg: 'cytoscape',            src: 'cytoscape/dist/cytoscape.min.js',             dst: 'cytoscape.min.js' },
  { pkg: '@dagrejs/dagre',       src: '@dagrejs/dagre/dist/dagre.min.js',            dst: 'dagre.min.js' },
  { pkg: 'cytoscape-dagre',      src: 'cytoscape-dagre/dist/cytoscape-dagre.js',     dst: 'cytoscape-dagre.js' },
  { pkg: 'cytoscape-bubblesets', src: 'cytoscape-bubblesets/build/index.umd.min.js', dst: 'cytoscape-bubblesets.min.js' },
  { pkg: 'cytoscape-layers',     src: 'cytoscape-layers/build/index.umd.min.js',     dst: 'cytoscape-layers.min.js' },
  { pkg: 'echarts',              src: 'echarts/dist/echarts.min.js',                 dst: 'echarts.min.js' },
  { pkg: 'vscode-oniguruma',     src: 'vscode-oniguruma/release/onig.wasm',           dst: 'onig.wasm' },
  // Temml: MathML styling + the (small, optional) script-capital font.
  // Temml-Local.css sets a font-family chain that picks up locally-
  // installed math fonts (Cambria Math / STIX Two Math / Noto Sans
  // Math / browser default), so no CDN / Google Fonts is involved.
  // `Temml.woff2` is only fetched if the page actually renders a
  // `\mathcal{...}` glyph (lazy per @font-face spec); ~9 KB.
  { pkg: 'temml',                src: 'temml/dist/Temml-Local.css',                  dst: 'temml.css' },
  { pkg: 'temml',                src: 'temml/dist/Temml.woff2',                      dst: 'Temml.woff2' },
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
// 1b. Provision the flatppl-wasm-api convert artifact into dist/vendor/.
await provisionWasmConvert();

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
  // Skip directories (e.g. src/vendor/, the vendored-grammar drop dir): the
  // per-file transpile/copy loop and the src/ watcher both reach here, and
  // copyFile/transform on a directory throws EISDIR.
  if (!statSync(srcPath).isFile()) return false;
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

const SRC_FILES = await readdir(join(here, 'src'), { withFileTypes: true });
for (const ent of SRC_FILES) {
  if (!ent.isFile()) continue;  // skip src/vendor/ and any other subdirs
  await transpileOrCopySrcFile(ent.name);
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
const hasCmHighlighter = await syncGrammars();

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

// Bundle the vendored FlatPPL CodeMirror highlighter (from flatppl-grammars)
// into an IIFE that bundles vscode-textmate + vscode-oniguruma and publishes
// window.FlatPPLTextmate. CodeMirror is NOT bundled here — the module receives
// the editor's CM instance via window.FlatPPLEditorBundle at call time.
const textmateHighlightBuildOpts = {
  entryPoints: [join(srcVendorDir, 'textmate-highlight.ts')],
  outfile: join(vendorDir, 'textmate-highlight.js'),
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

// All browser bundles, in emit order. The TextMate highlighter is only
// included when syncGrammars() vendored the CM module (hasCmHighlighter);
// otherwise a stub already sits at dist/vendor/textmate-highlight.js.
const BUNDLES = [
  { opts: engineBuildOpts,        label: 'engine             -> dist/vendor/engine.min.js' },
  { opts: samplerWorkerBuildOpts, label: 'sampler-worker     -> dist/vendor/sampler-worker.min.js' },
  { opts: codemirrorBuildOpts,    label: 'codemirror         -> dist/vendor/codemirror.min.js' },
  ...(hasCmHighlighter
    ? [{ opts: textmateHighlightBuildOpts, label: 'textmate-highlight -> dist/vendor/textmate-highlight.js' }]
    : []),
  { opts: viewerBuildOpts,        label: 'viewer             -> dist/vendor/viewer.js' },
];

if (WATCH) {
  const ctxs = await Promise.all(BUNDLES.map(b => esbuild.context(b.opts)));
  await Promise.all(ctxs.map(c => c.rebuild()));
  for (const b of BUNDLES) console.log(`  bundled ${b.label}`);
  await Promise.all(ctxs.map(c => c.watch()));

  // esbuild's context.watch() only re-fires the bundles above.
  // The src/ transpile-or-copy loop and the demo/ tree copy run ONCE
  // at startup, so without these additional fs.watch handlers edits
  // to index.html, style.css, src/*.ts, demo/*.flatppl, etc. would
  // not propagate to dist/ — and the dev server (which serves dist/)
  // would keep showing stale content. We watch src/ non-recursively
  // (src/vendor/ exists but its vendored contents are re-bundled by
  // esbuild's watch context above; transpileOrCopySrcFile skips
  // directories, so a 'vendor' event is a safe no-op) and demo/
  // recursive (it contains the curated .flatppl programs). Both are
  // debounced with a tiny coalescing window because fs.watch can fire
  // multiple events per save (rename + change, editor swap-file
  // pattern, etc.).
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
  await Promise.all(BUNDLES.map(b => esbuild.build(b.opts)));
  for (const b of BUNDLES) console.log(`  bundled ${b.label}`);
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

// flatppl-wasm-api convert artifact provisioning. Convert is REQUIRED by
// default: provision the artifact (CI-staged dir → sibling wasm-pack build →
// nightly download) and bake the build-time flag, OR FAIL THE BUILD with an
// actionable message. No silent degradation — a build that exits 0 has a
// PREDICTABLE convert state. Opt out explicitly with FLATPPL_CONVERT=off for a
// deterministic convert-less gallery. The page reads the flag from the
// generated build-flags.js (no runtime probe).
async function provisionWasmConvert() {
  if (process.env.FLATPPL_CONVERT === 'off') {
    // Explicit opt-out → deterministic convert-less build.
    await rm(join(vendorDir, WASM_GLUE), { force: true });
    await rm(join(vendorDir, WASM_BIN), { force: true });
    await writeConvertFlag(false);
    console.log('  flatppl-wasm-api: convert DISABLED (FLATPPL_CONVERT=off)');
    return;
  }
  const source = await tryProvisionWasm();
  if (source) {
    await writeConvertFlag(true);
    console.log(`  flatppl-wasm-api: convert ENABLED (${source})`);
    return;
  }
  throw new Error(
    'flatppl-wasm-api: convert is required but its wasm artifact could not be provisioned.\n'
    + '  Provide it one of these ways:\n'
    + '    - install the wasm toolchain so it builds from the flatppl-rust sibling:\n'
    + '        rustup target add wasm32-unknown-unknown && cargo install wasm-pack\n'
    + '    - or point FLATPPL_WASM_DIR at a directory holding a prebuilt artifact\n'
    + '  Or build a convert-less gallery explicitly:\n'
    + '        FLATPPL_CONVERT=off npm run build');
}

// Place the convert artifact into dist/vendor/ from the first working source.
// Returns a short description of that source, or null if none worked. Never
// throws — the caller decides whether "none" is fatal.
async function tryProvisionWasm() {
  const staged = process.env.FLATPPL_WASM_DIR;
  if (staged && existsSync(join(staged, WASM_GLUE))) {
    await copyFile(join(staged, WASM_GLUE), join(vendorDir, WASM_GLUE));
    await copyFile(join(staged, WASM_BIN), join(vendorDir, WASM_BIN));
    return 'staged ' + staged;
  }
  const rustSibling = process.env.FLATPPL_RUST_DIR
    || join(dirname(repoRoot), 'flatppl-rust');
  const crateDir = join(rustSibling, 'crates', 'wasm-api');
  if (existsSync(crateDir) && (await hasCmd('wasm-pack'))) {
    const pkg = join(crateDir, 'pkg');
    try {
      await runCmd('wasm-pack', ['build', '--target', 'web', '--release',
        '--no-typescript', '--out-dir', pkg, crateDir]);
      await copyFile(join(pkg, WASM_GLUE), join(vendorDir, WASM_GLUE));
      await copyFile(join(pkg, WASM_BIN), join(vendorDir, WASM_BIN));
      return 'built from sibling flatppl-rust';
    } catch (e) {
      console.warn(`  flatppl-wasm-api: wasm-pack build failed (${e.message}); trying download`);
    }
  }
  try {
    for (const name of [WASM_GLUE, WASM_BIN]) {
      const res = await fetch(`${WASM_RELEASE_BASE}/${name}`);
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${name}`);
      await writeFile(join(vendorDir, name), Buffer.from(await res.arrayBuffer()));
    }
    return 'downloaded from nightly';
  } catch (e) {
    console.warn(`  flatppl-wasm-api: download unavailable (${e.message})`);
    return null;
  }
}

// Bake the build-time convert flag into the page (no runtime probe):
// dist/build-flags.js merges { convert: <bool> } into __FLATPPL_CONFIG__.
async function writeConvertFlag(on) {
  await writeFile(join(distDir, 'build-flags.js'),
    '// Generated by build.mjs — build-time feature flags. Do not edit.\n'
    + 'window.__FLATPPL_CONFIG__ = Object.assign(window.__FLATPPL_CONFIG__ || {}, '
    + `{ convert: ${on ? 'true' : 'false'} });\n`);
}

function hasCmd(cmd) {
  return new Promise((resolve) => {
    const p = spawn(cmd, ['--version'], { stdio: 'ignore' });
    p.on('error', () => resolve(false));
    p.on('exit', (code) => resolve(code === 0));
  });
}

function runCmd(cmd, args) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: 'inherit' });
    p.on('error', reject);
    p.on('exit', (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}`))));
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
      throw new Error(`sibling exists but ${src} not found — flatppl-examples may have moved its content`);
    }
    await copyDirRecursive(src, dst);
  } else {
    console.log(`  examples: fetching pinned ref '${examplesPin}' from GitHub`);
    await fetchAndExtractExamples(examplesRemote, dst);
  }
}

// Fetch a .tar.gz, extract the whole archive into a fresh temp dir, and
// return its single top-level directory (GitHub archives unpack into one
// `<repo>-<ref>/` dir) plus a cleanup() to remove the temp tree. The
// caller copies whatever subdir it needs out of `topLevel`, then awaits
// cleanup() in a finally. Two-step "extract then copy" (vs a tar wildcard)
// avoids portability issues across GNU tar / bsdtar / macOS / Windows
// MSYS / CI ubuntu-latest. Shared by syncExamples and syncGrammars.
async function fetchAndExtractTarball(url, prefix) {
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

async function fetchAndExtractExamples(url, dstDir) {
  const { topLevel, cleanup } = await fetchAndExtractTarball(url, 'flatppl-examples-');
  try {
    const src = join(topLevel, 'examples');
    if (!existsSync(src)) {
      throw new Error(`expected examples/ subdir under ${topLevel}, not found`);
    }
    await copyDirRecursive(src, dstDir);
  } finally {
    await cleanup();
  }
}

async function syncGrammars() {
  // Vendor flatppl.tmLanguage.json -> dist/vendor/ (fetched at runtime) and
  // codemirror/textmate-highlight.ts -> src/vendor/ (esbuild entry below).
  // Clean the vendored drop dir so an upstream rename can't leave a stale file
  // that esbuild's pinned entryPoint would keep bundling. Safe: this runs once
  // at startup before any watcher attaches.
  //
  // Source resolution mirrors syncExamples: GRAMMARS_DIR / sibling clone (no
  // tarball, no network), else the pinned GitHub tarball. The tmLanguage JSON
  // is always required (throws if missing). The CM highlighter module is
  // optional; when absent we emit a no-op stub at dist/vendor/textmate-
  // highlight.js so index.html's unconditional <script> never 404s, and
  // return false so the caller skips the (now pointless) esbuild bundle.
  await rm(srcVendorDir, { recursive: true, force: true });
  await mkdir(srcVendorDir, { recursive: true });
  const tmDst = join(vendorDir, 'flatppl.tmLanguage.json');
  const modDst = join(srcVendorDir, 'textmate-highlight.ts');
  const stubDst = join(vendorDir, 'textmate-highlight.js');

  // Resolve `top` — the directory holding textmate/ and codemirror/.
  let top, cleanup = () => {};
  if (existsSync(grammarsSibling)) {
    top = grammarsSibling;
    console.log(`  grammars: using ${process.env.GRAMMARS_DIR ? 'GRAMMARS_DIR' : 'sibling'} at ${grammarsSibling}`);
  } else {
    console.log(`  grammars: fetching pinned ref '${grammarsPin}' from GitHub`);
    ({ topLevel: top, cleanup } = await fetchAndExtractTarball(grammarsRemote, 'flatppl-grammars-'));
  }

  try {
    const tmSrc = join(top, 'textmate', 'flatppl.tmLanguage.json');
    const modSrc = join(top, 'codemirror', 'textmate-highlight.ts');
    if (!existsSync(tmSrc)) throw new Error(`grammars: required ${tmSrc} not found`);
    await copyFile(tmSrc, tmDst);
    if (existsSync(modSrc)) {
      await copyFile(modSrc, modDst);
      console.log('  grammars: copied tmLanguage + CM highlighter');
      return true;
    }
    await writeFile(stubDst,
      '/* FlatPPL TextMate highlighter unavailable: the grammars source lacks '
      + 'codemirror/textmate-highlight.ts. Editor degrades to plain text + '
      + 'binding overlay. Stub kept so index.html\'s <script> tag does not 404. */\n');
    console.warn('  grammars: CM highlighter (codemirror/textmate-highlight.ts) absent; wrote stub, editor degrades to plain text + binding overlay');
    return false;
  } finally {
    await cleanup();
  }
}

// MODEL_EXTENSIONS is declared at top-level (above) so it's initialized
// before the manifest pass; `modelTypeForPath` mirrors surfaces.ts.
function modelTypeForPath(path) {
  const p = path.toLowerCase();
  if (p.endsWith('.flatppl')) return 'flatppl';
  if (p.endsWith('.md') || p.endsWith('.markdown')) return 'markdown';
  if (p.endsWith('.hs3.json')) return 'hs3';
  if (p.endsWith('.pyhf.json')) return 'pyhf';
  return 'unknown';
}

/** Recursively list model/representation files under rootDir, sorted
    alphabetically. Paths are relative to rootDir (forward slashes).
    Matches MODEL_EXTENSIONS — the gallery is multi-format, no longer
    FlatPPL-only. */
async function listModelFiles(rootDir) {
  const out = [];
  async function walk(dir, prefix) {
    if (!existsSync(dir)) return;
    const entries = await readdir(dir, { withFileTypes: true });
    for (const ent of entries) {
      const childRelPath = prefix ? prefix + '/' + ent.name : ent.name;
      if (ent.isDirectory()) {
        await walk(join(dir, ent.name), childRelPath);
      } else if (ent.isFile()
          && MODEL_EXTENSIONS.some((e) => ent.name.toLowerCase().endsWith(e))) {
        out.push(childRelPath);
      }
    }
  }
  await walk(rootDir, '');
  out.sort();
  return out;
}

// Use the basename (not the full path) as the gallery title so the
// tree renders cleanly. The .flatppl extension stays in the label
// so the file type is obvious at a glance.
function basenameTitle(path) {
  const i = path.lastIndexOf('/');
  return i < 0 ? path : path.slice(i + 1);
}

async function generateManifest() {
  const demoFiles     = await listModelFiles(join(distDir, 'demo'));
  const exampleFiles  = await listModelFiles(join(distDir, 'examples'));

  // `type` is carried for the gallery (surface routing / future icons +
  // model→representations grouping). The runtime still derives type from
  // the path via surfaces.ts `typeForPath`, so a hand-written manifest
  // without `type` keeps working.
  const entries = [];
  for (const path of demoFiles) {
    entries.push({ path: 'demo/' + path,     title: basenameTitle(path), type: modelTypeForPath(path) });
  }
  for (const path of exampleFiles) {
    entries.push({ path: 'examples/' + path, title: basenameTitle(path), type: modelTypeForPath(path) });
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
