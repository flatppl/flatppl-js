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

const { buildMathRequest, composeMathRows, rowNameFor, rowHtml, moduleDocHtml, notationHtml, focusedBindingName, mathModelKey, mathWasmUrl } = await import('./math-view.ts');

// The math pane consumes flatppl-rust's `render_math` contract
// (flatppl-dev/math-view-design.md §4 plus the Rust session's deltas):
// one row per module-level binding, a decomposition being ONE row named
// after its first target with every name in `names`; `order` lists row
// names; `refs` lists the OTHER bindings a row refers to; diagnostics carry a
// binding name ("" = module-level). The pure
// composition below is what turns that into rows the pane renders.

const response = {
  order: ['y_data', 'a', 'posterior'],
  bindings: [
    { name: 'y_data', names: ['y_data'], kind: 'value', mathml: '<math data-flatppl-binding="y_data"><mi>y</mi></math>', refs: [], loc: { start: 10, end: 20 }, annotation: '8 values' },
    { name: 'a', names: ['a', 'b'], kind: 'draw', mathml: '<math data-flatppl-binding="a"><mi>a</mi></math>', refs: ['M'] },
    { name: 'posterior', names: ['posterior'], kind: 'measure', mathml: '<math data-flatppl-binding="posterior"><mi>p</mi></math>', refs: ['L'], doc: { html: '<p>The posterior.</p>', block: true } },
  ],
  diagnostics: [
    { binding: '', message: 'format `typst` is not available yet' },
    { binding: 'b', message: 'no index range: inference failed' },
    { binding: 'gone', message: 'orphan message' },
  ],
};

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

test('composeMathRows keeps the response order and carries the row fields', () => {
  const { rows } = composeMathRows(response, {});
  assert.deepEqual(rows.map((r) => r.name), ['y_data', 'a', 'posterior']);
  assert.deepEqual(rows[1].names, ['a', 'b']);
  assert.equal(rows[0].kind, 'value');
  assert.equal(rows[0].annotation, '8 values');
  assert.equal(rows[1].annotation, null);
  assert.equal(rows[0].mathml, response.bindings[0].mathml);
  assert.deepEqual(rows[2].refs, ['L']);   // other bindings only (own names excluded, per contract)
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

test('composeMathRows resolves the source line per row through the callback and carries the rendered doc', () => {
  const { rows } = composeMathRows(response, {
    lineOf: (name) => (name === 'a' ? 15 : name === 'posterior' ? 25 : null),
  });
  assert.equal(rows[0].line, null);
  assert.equal(rows[1].line, 15);
  assert.equal(rows[2].line, 25);
  assert.equal(rows[0].doc, null);
  assert.deepEqual(rows[2].doc, { html: '<p>The posterior.</p>', block: true });
});

test('a doc without a block flag is a caption; a doc without html is no doc', () => {
  const res = { order: ['x'], bindings: [{ name: 'x', names: ['x'], kind: 'value', mathml: '<math/>', refs: [], doc: { html: '<p>c</p>' } }], diagnostics: [] };
  assert.deepEqual(composeMathRows(res, {}).rows[0].doc, { html: '<p>c</p>', block: false });
  res.bindings[0].doc = { block: true };
  assert.equal(composeMathRows(res, {}).rows[0].doc, null);
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

test('a binding present in bindings but absent from order surfaces as a module-level diagnostic', () => {
  const { rows, moduleDiagnostics } = composeMathRows({ ...response, order: ['y_data', 'a'] }, {});
  assert.deepEqual(rows.map((r) => r.name), ['y_data', 'a']);
  assert.ok(moduleDiagnostics.some((d) => d.includes('posterior')), moduleDiagnostics.join(' | '));
});

test('a binding without a mathml fragment renders empty and says so, never the text "undefined"', () => {
  const res = { order: ['x'], bindings: [{ name: 'x', names: ['x'], kind: 'value', refs: ['x'] }], diagnostics: [] };
  const { rows } = composeMathRows(res, {});
  assert.equal(rows[0].mathml, '');
  assert.equal(rows[0].diagnostics.length, 1);
  assert.ok(!rowHtml(rows[0]).includes('undefined'));
});

test('rowHtml escapes the binding name in the data attribute and the annotation / diagnostics in text', () => {
  const html = rowHtml({
    name: 'a"b<c', names: ['a"b<c'], kind: 'value', mathml: '<math><mi>a</mi></math>', refs: [],
    annotation: '<8 & "values">', diagnostics: ['bad <thing>'], focused: true, line: 3, doc: null,
  });
  assert.ok(html.includes('data-binding="a&quot;b&lt;c"'), html);
  assert.ok(html.includes('<math><mi>a</mi></math>'));
  assert.ok(html.includes('&lt;8 &amp; "values"&gt;'));   // text position: quotes stay
  assert.ok(html.includes('bad &lt;thing&gt;'));
  assert.ok(html.includes('class="math-row focused"'));
  assert.ok(!html.includes('math-row-doc'));
});

test('rowHtml places the Rust-rendered doc fragment verbatim above the equation, block docs marked as prose', () => {
  const base = {
    name: 'x', names: ['x'], kind: 'value', mathml: '<math/>', refs: [], annotation: null, diagnostics: [],
    focused: false, line: null,
  };
  const caption = rowHtml({ ...base, doc: { html: '<p>The <em>mean</em> <math><mi>μ</mi></math>.</p>', block: false } });
  assert.ok(caption.includes('<div class="math-row-doc"><p>The <em>mean</em> <math><mi>μ</mi></math>.</p></div><div class="math-row-eq">'));
  const prose = rowHtml({ ...base, doc: { html: '<h2>Why</h2><p>Because.</p>', block: true } });
  assert.ok(prose.includes('<div class="math-row-doc block"><h2>Why</h2>'));
  assert.ok(!rowHtml({ ...base, doc: { html: '', block: false } }).includes('math-row-doc'));
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

test('the module introduction is the title (escaped) as h1 plus the rendered body, or nothing', () => {
  const html = moduleDocHtml({ title: 'Eight Schools <Rubin>', html: '<p>Effects <math><mi>y</mi></math>.</p>' });
  assert.equal(html, '<div class="math-module-doc"><h1>Eight Schools &lt;Rubin&gt;</h1><p>Effects <math><mi>y</mi></math>.</p></div>');
  assert.equal(moduleDocHtml({ title: null, html: '<p>Untitled.</p>' }), '<div class="math-module-doc"><p>Untitled.</p></div>');
  assert.equal(moduleDocHtml({ title: 'Only a title', html: '' }), '<div class="math-module-doc"><h1>Only a title</h1></div>');
  assert.equal(moduleDocHtml(null), '');
  assert.equal(moduleDocHtml(undefined), '');
  assert.equal(moduleDocHtml({ title: null, html: '' }), '');
});

test('notation starts collapsed with the rendered symbol and note verbatim and no code spelling, or is omitted when empty', () => {
  assert.equal(notationHtml([{ name: 'Normal', mathml: '<mi>μ</mi>', note: '<p>Mean <math><mi>μ</mi></math> &amp; scale.</p>' }]),
    '<details class="math-notation"><summary>Notation</summary><dl>'
    + '<dt><math><mi>μ</mi></math></dt><dd><p>Mean <math><mi>μ</mi></math> &amp; scale.</p></dd></dl></details>');
  assert.equal(notationHtml([]), '');
  assert.equal(notationHtml(undefined), '');
});
