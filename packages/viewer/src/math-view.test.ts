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

const { buildMathRequest, composeMathRows, rowNameFor } = await import('./math-view.ts');

// The math pane consumes flatppl-rust's `render_math` contract
// (flatppl-dev/math-view-design.md §4 plus the Rust session's deltas):
// one row per module-level binding, a decomposition being ONE row named
// after its first target with every name in `names`; `order` lists row
// names; diagnostics carry a binding name ("" = module-level). The pure
// composition below is what turns that into rows the pane renders.

const response = {
  order: ['y_data', 'a', 'posterior'],
  bindings: [
    { name: 'y_data', names: ['y_data'], kind: 'value', mathml: '<math data-flatppl-binding="y_data"><mi>y</mi></math>', refs: ['y_data'], loc: { start: 10, end: 20 }, annotation: '8 values' },
    { name: 'a', names: ['a', 'b'], kind: 'draw', mathml: '<math data-flatppl-binding="a"><mi>a</mi></math>', refs: ['a', 'b', 'M'] },
    { name: 'posterior', names: ['posterior'], kind: 'measure', mathml: '<math data-flatppl-binding="posterior"><mi>p</mi></math>', refs: ['posterior', 'L'] },
  ],
  diagnostics: [
    { binding: '', message: 'format `typst` is not available yet' },
    { binding: 'b', message: 'no index range: inference failed' },
    { binding: 'gone', message: 'orphan message' },
  ],
};

test('buildMathRequest defaults path, bundle and formats per the contract', () => {
  assert.deepEqual(buildMathRequest({ source: 'x = 1' }), {
    source: 'x = 1', path: 'model.flatppl', bundle: {}, formats: ['mathml'],
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

test('composeMathRows keeps the response order and carries the row fields', () => {
  const { rows } = composeMathRows(response, {});
  assert.deepEqual(rows.map((r) => r.name), ['y_data', 'a', 'posterior']);
  assert.deepEqual(rows[1].names, ['a', 'b']);
  assert.equal(rows[0].kind, 'value');
  assert.equal(rows[0].annotation, '8 values');
  assert.equal(rows[1].annotation, null);
  assert.equal(rows[0].mathml, response.bindings[0].mathml);
  assert.deepEqual(rows[2].refs, ['posterior', 'L']);
});

test('the focused row is the one whose names include the focus — also a secondary decomposition name', () => {
  assert.deepEqual(composeMathRows(response, { focus: 'posterior' }).rows.map((r) => r.focused), [false, false, true]);
  assert.deepEqual(composeMathRows(response, { focus: 'b' }).rows.map((r) => r.focused), [false, true, false]);
  assert.deepEqual(composeMathRows(response, { focus: null }).rows.map((r) => r.focused), [false, false, false]);
  assert.deepEqual(composeMathRows(response, { focus: 'nope' }).rows.map((r) => r.focused), [false, false, false]);
});

test('per-binding diagnostics attach to their row by any of its names; the rest are module-level', () => {
  const { rows, moduleDiagnostics } = composeMathRows(response, {});
  assert.deepEqual(rows[1].diagnostics, ['no index range: inference failed']);
  assert.deepEqual(rows[0].diagnostics, []);
  // Module-level ("" binding) and orphans (a binding no row claims) both
  // surface at module level — a diagnostic is never dropped silently.
  assert.deepEqual(moduleDiagnostics, [
    'format `typst` is not available yet',
    'gone: orphan message',
  ]);
});

test('composeMathRows resolves the source line and doc-comment per row through the callbacks', () => {
  const { rows } = composeMathRows(response, {
    lineOf: (name) => (name === 'a' ? 15 : name === 'posterior' ? 25 : null),
    docOf: (name) => (name === 'posterior' ? { markup: 'md', lines: ['The posterior.'] } : null),
  });
  assert.equal(rows[0].line, null);
  assert.equal(rows[1].line, 15);
  assert.equal(rows[2].line, 25);
  assert.equal(rows[0].doc, null);
  assert.deepEqual(rows[2].doc, { markup: 'md', lines: ['The posterior.'] });
});

test('rowNameFor maps any binding name to its row, or null when there is none', () => {
  assert.equal(rowNameFor(response, 'b'), 'a');
  assert.equal(rowNameFor(response, 'a'), 'a');
  assert.equal(rowNameFor(response, 'posterior'), 'posterior');
  assert.equal(rowNameFor(response, 'M'), null);
});

test('a row listed in order but missing from bindings is skipped, not fabricated', () => {
  const { rows } = composeMathRows({ ...response, order: ['y_data', 'ghost', 'a'] }, {});
  assert.deepEqual(rows.map((r) => r.name), ['y_data', 'a']);
});
