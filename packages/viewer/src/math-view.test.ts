// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const { buildMathRequest, focusedBindingName, mathModelKey, mathWasmUrl } = await import('./math-view.ts');

// The math pane shows flatppl-rust's rendered document; this pure layer
// only builds the request and holds the small rules the pane needs
// (flatppl-dev/math-view-design.md §4).

test('buildMathRequest defaults path, bundle and formats per the contract', () => {
  assert.deepEqual(buildMathRequest({ source: 'x = 1' }), {
    source: 'x = 1', path: 'model.flatppl', bundle: {}, formats: ['mathml'], document: true,
  });
});

test('buildMathRequest passes the host path and the resolved-path bundle through', () => {
  const req = buildMathRequest({
    source: 'm = load_module("lib/h.flatppl")',
    path: 'models/m.flatppl',
    bundleSources: { 'models/lib/h.flatppl': 'h = 1' },
  });
  assert.equal(req.path, 'models/m.flatppl');
  assert.deepEqual(req.bundle, { 'models/lib/h.flatppl': 'h = 1' });
  assert.deepEqual(req.formats, ['mathml']);
});

test('buildMathRequest treats a null bundle and a null path as absent', () => {
  const req = buildMathRequest({ source: 'x = 1', path: null, bundleSources: null });
  assert.equal(req.path, 'model.flatppl');
  assert.deepEqual(req.bundle, {});
});

test('focusedBindingName follows the plot binding, else the sub-DAG root, and nothing in module view', () => {
  const MODULE = '<module>';
  assert.equal(focusedBindingName({ currentPlotBindingName: 'tau', currentState: { targetName: 'posterior' }, MODULE_TARGET: MODULE }), 'tau');
  assert.equal(focusedBindingName({ currentPlotBindingName: null, currentState: { targetName: 'posterior' }, MODULE_TARGET: MODULE }), 'posterior');
  assert.equal(focusedBindingName({ currentPlotBindingName: null, currentState: { targetName: MODULE }, MODULE_TARGET: MODULE }), null);
  assert.equal(focusedBindingName({ currentPlotBindingName: null, currentState: null, MODULE_TARGET: MODULE }), null);
});

test('mathModelKey changes with the analysed source, the path and the bundle, and only those', () => {
  const k = mathModelKey('x = 1', 'm.flatppl', { 'lib/h.flatppl': 'h = 1' });
  assert.equal(k, mathModelKey('x = 1', 'm.flatppl', { 'lib/h.flatppl': 'h = 1' }));
  assert.notEqual(k, mathModelKey('x = 2', 'm.flatppl', { 'lib/h.flatppl': 'h = 1' }));
  assert.notEqual(k, mathModelKey('x = 1', 'n.flatppl', { 'lib/h.flatppl': 'h = 1' }));
  assert.notEqual(k, mathModelKey('x = 1', 'm.flatppl', { 'lib/h.flatppl': 'h = 2' }));
  assert.equal(mathModelKey('x = 1', null, null), mathModelKey('x = 1', undefined, undefined));
});

test('mathWasmUrl resolves a relative glue URL against the page, keeps absolute ones, and is null when unset', () => {
  assert.equal(mathWasmUrl({ wasmApiUrl: 'vendor/flatppl_wasm_api.js' }, 'https://live.flatppl.org/'), 'https://live.flatppl.org/vendor/flatppl_wasm_api.js');
  assert.equal(mathWasmUrl({ wasmApiUrl: 'vendor/flatppl_wasm_api.js' }, 'https://x.org/gallery/index.html'), 'https://x.org/gallery/vendor/flatppl_wasm_api.js');
  assert.equal(mathWasmUrl({ wasmApiUrl: 'https://cdn.example/api.js' }, 'https://x.org/'), 'https://cdn.example/api.js');
  assert.equal(mathWasmUrl({ wasmApiUrl: '  ' }, 'https://x.org/'), null);
  assert.equal(mathWasmUrl({}, 'https://x.org/'), null);
  assert.equal(mathWasmUrl(null, 'https://x.org/'), null);
});

