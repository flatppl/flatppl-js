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

const {
  buildMathRequest, focusedBindingName, mathModelKey, mathWasmUrl,
  MATH_DOCUMENT_FORMATS, mathDocumentFormat, buildMathExportRequest, mathExportFileName,
} = await import('./math-view.ts');

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


// ---- whole documents (`export_math`, design §4 "Whole documents") ------

test('the document catalogue names the four formats the contract fixes', () => {
  assert.deepEqual(MATH_DOCUMENT_FORMATS.map((f) => f.document), ['html', 'md', 'tex', 'typ']);
  assert.equal(mathDocumentFormat('tex').label, 'LaTeX');
  assert.equal(mathDocumentFormat('docx'), null);
});

test('buildMathExportRequest carries the pane\'s source, path and bundle plus the format', () => {
  assert.deepEqual(buildMathExportRequest({
    source: 'x = 1', path: 'examples/m.flatppl', bundleSources: { 'examples/d.flatppl': 'y = 2' }, document: 'typ',
  }), {
    source: 'x = 1', path: 'examples/m.flatppl', bundle: { 'examples/d.flatppl': 'y = 2' }, document: 'typ',
  });
  // No path / bundle: the same defaults as the pane's request.
  assert.deepEqual(buildMathExportRequest({ source: '', document: 'html' }),
    { source: '', path: 'model.flatppl', bundle: {}, document: 'html' });
});

test('mathExportFileName is the model stem with the format extension', () => {
  assert.equal(mathExportFileName('examples/hep/model.flatppl', 'html'), 'model.html');
  assert.equal(mathExportFileName('/home/u/work/eight_schools.flatppl', 'tex'), 'eight_schools.tex');
  assert.equal(mathExportFileName('C:\\models\\m.flatppl', 'typ'), 'm.typ');
  // Only the .flatppl suffix is stripped; other dots stay.
  assert.equal(mathExportFileName('a.b.flatppl', 'md'), 'a.b.md');
  assert.equal(mathExportFileName('notes', 'md'), 'notes.md');
  // No path: the default module name.
  assert.equal(mathExportFileName(null, 'html'), 'model.html');
  assert.throws(() => mathExportFileName('m.flatppl', 'docx'), /unknown math document format/);
});
