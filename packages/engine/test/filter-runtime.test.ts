'use strict';

// Spec §07: filter(predicate, data) — keep elements for which the
// predicate returns true. The orchestrator pre-eval pass resolves
// the predicate's body via env.__resolveFnBody and the sampler's
// evaluateCall walks data per element.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const { toJS } = require('./_value-helpers.ts');

function buildFixed(source: any) {
  const lifted = processSource(source);
  return orchestrator.buildDerivations(lifted.bindings);
}

// =====================================================================
// Inline fn(...) predicates
// =====================================================================

test('filter: fn(_ > 0) keeps positive elements', () => {
  const r = buildFixed(`
data = [1.0, -2.5, 3.0, -4.0, 5.0]
result = filter(fn(_ > 0.0), data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('result')), [1.0, 3.0, 5.0]);
});

test('filter: in-range predicate keeps interior values', () => {
  // fn(_ in interval(lo, hi)) would be the spec-canonical form, but
  // `in` + set IRs aren't yet value-evaluable through the worker's
  // evaluateCall (TODO: wire `in` for value contexts). The functionof
  // form with a shared placeholder is the inline equivalent — each
  // `_` in fn() is a DISTINCT positional param (spec §sec:fn), so we
  // use functionof + a single placeholder to express "the same value
  // appears twice in the body".
  const r = buildFixed(`
lo = 0.0
hi = 10.0
in_range = functionof(land(_x_ >= lo, _x_ <= hi), x = _x_)
data = [-5.0, 2.5, 7.5, 12.0, 9.9]
result = filter(in_range, data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('result')), [2.5, 7.5, 9.9]);
});

test('filter: empty data ⇒ empty result', () => {
  const r = buildFixed(`
data = []
result = filter(fn(_ > 0.0), data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('result')), []);
});

test('filter: predicate always-false ⇒ empty result', () => {
  const r = buildFixed(`
data = [1.0, 2.0, 3.0]
result = filter(fn(_ > 100.0), data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('result')), []);
});

test('filter: predicate always-true ⇒ original vector', () => {
  const r = buildFixed(`
data = [1.0, 2.0, 3.0]
result = filter(fn(_ > -100.0), data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('result')), [1.0, 2.0, 3.0]);
});

// =====================================================================
// Named function predicate
// =====================================================================

test('filter: named functionof binding as predicate', () => {
  const r = buildFixed(`
is_positive = functionof(x > 0.0, x = x)
data = [1.0, -1.0, 2.0, -2.0]
x = 0.0
result = filter(is_positive, data)
`);
  // We define `x` as a separate fixed binding because functionof's
  // boundary x = x captures the binding by name. The predicate body
  // references the bound parameter, not the module-level x.
  assert.deepEqual(toJS(r.fixedValues.get('result')), [1.0, 2.0]);
});

// =====================================================================
// Closure over surrounding bindings
// =====================================================================

test('filter: predicate closes over fixed-phase bindings', () => {
  // Threshold is a separate fixed-phase scalar that the predicate
  // references via free name. The pre-eval env carries it through.
  const r = buildFixed(`
threshold = 2.0
data = [1.0, 2.0, 3.0, 4.0]
result = filter(fn(_ > threshold), data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('result')), [3.0, 4.0]);
});

// =====================================================================
// Chaining with other value ops
// =====================================================================

// =====================================================================
// reduce / scan — binary higher-order ops sharing the filter machinery
// =====================================================================

test('reduce(fn(_ + _), [1, 2, 3, 4]) ⇒ 10', () => {
  const r = buildFixed(`
data = [1.0, 2.0, 3.0, 4.0]
total = reduce(fn(_ + _), data)
`);
  assert.equal(r.fixedValues.get('total'), 10);
});

test('reduce closes over fixed-phase bindings', () => {
  const r = buildFixed(`
factor = 2.0
data = [1.0, 2.0, 3.0]
weighted_sum = reduce(fn(_ + factor * _), data)
`);
  // 1 + 2 * 2 = 5; 5 + 2 * 3 = 11
  assert.equal(r.fixedValues.get('weighted_sum'), 11);
});

test('reduce: empty vector ⇒ runtime error', () => {
  // No initial accumulator and no elements to fold — undefined fold.
  const r = buildFixed(`
data = []
total = reduce(fn(_ + _), data)
`);
  // pre-eval silently skips on throws — total stays unevaluated.
  assert.equal(r.fixedValues.get('total'), undefined);
});

test('scan(fn(_ + _), 0, [1, 2, 3, 4]) ⇒ [1, 3, 6, 10]', () => {
  const r = buildFixed(`
data = [1.0, 2.0, 3.0, 4.0]
cumsum = scan(fn(_ + _), 0.0, data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('cumsum')), [1, 3, 6, 10]);
});

test('scan: empty data ⇒ empty result', () => {
  const r = buildFixed(`
data = []
cumsum = scan(fn(_ + _), 0.0, data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('cumsum')), []);
});

test('scan: cumulative max (max(acc, x))', () => {
  const r = buildFixed(`
data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]
running_max = scan(fn(max(_, _)), 0.0, data)
`);
  assert.deepEqual(toJS(r.fixedValues.get('running_max')), [3, 3, 4, 4, 5, 9, 9]);
});

// =====================================================================
// broadcast — elementwise function application over arrays
// =====================================================================

test('broadcast: unary positional ⇒ map over single array', () => {
  const r = buildFixed(`
xs = [1.0, 2.0, 3.0, 4.0]
ys = broadcast(fn(_ * 2.0 + 1.0), xs)
`);
  assert.deepEqual(toJS(r.fixedValues.get('ys')), [3, 5, 7, 9]);
});

test('broadcast: binary positional ⇒ zip-map over two arrays', () => {
  const r = buildFixed(`
xs = [1.0, 2.0, 3.0]
ys = [10.0, 20.0, 30.0]
sums = broadcast(fn(_ + _), xs, ys)
`);
  assert.deepEqual(toJS(r.fixedValues.get('sums')), [11, 22, 33]);
});

test('broadcast: closure over fixed-phase scalar', () => {
  const r = buildFixed(`
scale = 10.0
xs = [1.0, 2.0, 3.0]
ys = broadcast(fn(scale * _), xs)
`);
  assert.deepEqual(toJS(r.fixedValues.get('ys')), [10, 20, 30]);
});

test('broadcast: empty input ⇒ empty output', () => {
  const r = buildFixed(`
xs = []
ys = broadcast(fn(_ + 1.0), xs)
`);
  assert.deepEqual(toJS(r.fixedValues.get('ys')), []);
});

test('broadcasted(f): direct application becomes broadcast', () => {
  const r = buildFixed(`
xs = [1.0, 2.0, 3.0]
ys = [10.0, 20.0, 30.0]
sums = broadcasted(add)(xs, ys)
`);
  assert.deepEqual(toJS(r.fixedValues.get('sums')), [11, 22, 33]);
});

test('broadcasted(f): via named binding (HEP-style bcadd / bcmul)', () => {
  const r = buildFixed(`
bcadd = broadcasted(add)
bcmul = broadcasted(mul)
mu = 2.0
sig = [1.0, 2.0, 3.0]
bkg = [10.0, 20.0, 30.0]
expected = bcadd(bcmul([mu, mu, mu], sig), bkg)
`);
  // bcmul([2,2,2], [1,2,3]) = [2, 4, 6]; bcadd with [10, 20, 30] = [12, 24, 36]
  assert.deepEqual(toJS(r.fixedValues.get('expected')), [12, 24, 36]);
});

test('broadcast: length mismatch ⇒ pre-eval silently skips on throw', () => {
  const r = buildFixed(`
xs = [1.0, 2.0, 3.0]
ys = [10.0, 20.0]
sums = broadcast(fn(_ + _), xs, ys)
`);
  // pre-eval silently swallows the throw → sums stays unevaluated.
  assert.equal(r.fixedValues.get('sums'), undefined);
});

test('filter: integrates with bincounts (region-restricted observation)', () => {
  const r = buildFixed(`
in_region = functionof(land(_x_ >= 2.0, _x_ <= 5.0), x = _x_)
data = [1.0, 2.5, 3.5, 4.5, 6.0, 8.0]
region_data = filter(in_region, data)
counts = bincounts([2.0, 3.0, 4.0, 5.0], region_data)
`);
  // region_data = [2.5, 3.5, 4.5]; bins [2,3] [3,4] [4,5] → [1, 1, 1]
  assert.deepEqual(toJS(r.fixedValues.get('region_data')), [2.5, 3.5, 4.5]);
  assert.deepEqual(toJS(r.fixedValues.get('counts')), [1, 1, 1]);
});
