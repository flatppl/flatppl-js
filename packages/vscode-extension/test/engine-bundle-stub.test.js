// Bundles packages/engine with the stdlib stub and asserts the result loads in
// a scope with NO `require` (like the webview) — i.e. no @stdlib require runs at
// init — and that @stdlib is absent from the output. This guards the size win
// and the load-cleanliness the plain-external approach broke.
const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const path = require('node:path');
const esbuild = require('esbuild');
const { stubStdlibPlugin } = require('../build-stdlib-stub.cjs');

const ENGINE_ENTRY = path.join(__dirname, '..', '..', 'engine', 'index.ts');

async function bundleEngine(plugins) {
  const r = await esbuild.build({
    entryPoints: [ENGINE_ENTRY],
    bundle: true,
    write: false,
    format: 'iife',
    globalName: 'FlatPPLEngine',
    platform: 'browser',
    target: ['es2020'],
    plugins,
  });
  return r.outputFiles[0].text;
}

test('stubbed engine bundle loads with no `require` and excludes @stdlib', async () => {
  const code = await bundleEngine([stubStdlibPlugin]);
  // No @stdlib *source* in the bundle — the stub replaces every @stdlib module
  // with a ~80-byte throw-on-call Proxy. esbuild emits the namespace key
  // "stdlib-stub:@stdlib/..." in chunk comments (that is expected), but real
  // stdlib internals (e.g. `lanczos`, the gamma kernel) must be absent.
  assert.ok(code.includes('stdlib-stub:'), 'stub plugin must have run (namespace marker present)');
  assert.ok(!code.includes('lanczos'), 'real @stdlib gamma source must not be in the bundle');
  assert.ok(!code.includes('gammaRatio'), 'real @stdlib source must not be in the bundle');
  // Loads in a scope without `require` (webview-like) — no throw at init.
  const sandbox = { globalThis: {} };
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox);
  assert.equal(typeof sandbox.FlatPPLEngine, 'object', 'FlatPPLEngine global must be defined');
  // density is a module object; logDensity is the callable entry point.
  assert.equal(typeof sandbox.FlatPPLEngine.density, 'object', 'FlatPPLEngine.density module must be defined');
  assert.equal(typeof sandbox.FlatPPLEngine.density.logDensity, 'function', 'FlatPPLEngine.density.logDensity must be callable');
});
