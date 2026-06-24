// @flatppl/web — file-type + tree-bucket decisions (single source of truth).
//
// The mapping from a path to its surface type, the gallery's left-pane
// folder bucketing, and the per-folder default-open state all used to be
// duplicated: once in build.mjs (manifest generation) and again in the
// runtime scripts (surfaces.ts, app.ts). That drift is exactly what
// ../CLAUDE.md warns about ("KEEP THIS IN STEP with surfaces.ts"). This
// module is the one home for those pure decisions.
//
// Consumed two ways from one source:
//   - build.mjs imports it directly (Node ESM) for manifest filtering /
//     typing and the build-flag parse.
//   - The runtime gets it as a global: build.mjs bundles this file to
//     dist/vendor/file-types.js (esbuild IIFE, globalName FlatPPLFileTypes),
//     loaded as a <script> before the web sub-modules. surfaces.ts /
//     app.ts call window.FlatPPLFileTypes.* — no second copy of the logic.
//   - test/file-types.test.mjs imports it directly to lock the behaviour.
//
// Pure: no DOM, no window, no Node APIs — safe in all three contexts.

// Model/representation file types the gallery surfaces. Adding a new format
// (a `.stan`, …) is a one-line change HERE — both the build's manifest filter
// and the runtime's surface routing pick it up. HS3 / pyhf models are JSON and
// carry the explicit `.hs3.json` / `.pyhf.json` extensions.
export const MODEL_EXTENSIONS = [
  '.flatppl', '.md', '.markdown', '.hs3.json', '.pyhf.json',
];

/** Map a path to its surface type. Empty/missing defaults to 'flatppl'
 *  (the gallery is FlatPPL-first); an unrecognised extension is 'unknown'
 *  (routes to the placeholder surface). hs3/pyhf are named distinctly so
 *  they stay uploadable + reserve the slot for a future native surface. */
export function typeForPath(path) {
  if (!path) return 'flatppl';
  const p = String(path).toLowerCase();
  if (p.endsWith('.flatppl')) return 'flatppl';
  if (p.endsWith('.md') || p.endsWith('.markdown')) return 'markdown';
  if (p.endsWith('.hs3.json')) return 'hs3';
  if (p.endsWith('.pyhf.json')) return 'pyhf';
  return 'unknown';
}

/** Map a manifest entry path to its left-pane folder bucket. demo/ + test-cases/
 *  land in the feature-test "Test cases" folder (demo/ is the legacy prefix);
 *  everything else is a plain example. */
export function bucketKeyForPath(path) {
  const p = String(path || '');
  if (p.indexOf('test-cases/') === 0) return 'test-cases';
  if (p.indexOf('demo/') === 0) return 'test-cases';
  return 'examples';
}

/** Default open/closed state for a folder when the user has no persisted
 *  preference. Examples open by default; User opens when it has content;
 *  everything else (Test cases) starts collapsed. */
export function defaultFolderOpen(key, itemCount) {
  if (key === 'examples') return true;
  if (key === 'user') return itemCount > 0;
  return false;
}

/** Parse the FLATPPL_HIDE_TEST_EXAMPLES build env into a boolean. On for
 *  any truthy value other than the explicit off-ish ones, so CI can set
 *  '1' / 'true' and a stray '0' / 'off' / 'false' / '' reads as off. */
export function hideTestExamplesEnv(value) {
  return !!value && value !== '0' && value !== 'off' && value !== 'false';
}
