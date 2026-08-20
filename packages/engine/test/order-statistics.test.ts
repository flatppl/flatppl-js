'use strict';

// Spec §07 order statistics: `median(xs)` and `quantile(xs, p)`.
//
//   median  — x_((n+1)/2) for odd n, ½(x_(n/2) + x_(n/2+1)) for even n
//   quantile — h = (n−1)p + 1, k = ⌊h⌋,
//              x_(k) + (h − k)(x_(k+1) − x_(k)), second term vanishing
//              at k = n
//
// Every expected value below is the ORACLE value from Julia's
// Statistics.median / Statistics.quantile (the latter is Hyndman-Fan
// type 7, the numpy `method='linear'` default), not engine output. The
// spec formula was checked against that oracle bit-for-bit on each case
// listed here.
//
// median is order-invariant, so it also reduces a table column-wise and
// is an eligible aggregate reduction (§04 §sec:aggregate). quantile is
// two-argument and is neither.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const sampler = require('../sampler.ts');
const lowerMod = require('../lower.ts');

const ARITH_OPS = sampler._internal.ARITH_OPS;

function ev(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'unexpected errors: ' + JSON.stringify(errs));
  return buildDerivations(r.bindings).fixedValues;
}

function errors(src: string) {
  return processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

// Evaluate a bare RHS through the lowered IR, so a runtime throw
// surfaces here instead of being swallowed by fixed-eval's try/catch.
function evalRHS(src: string, binding: string, env: any) {
  const ctx = processSource(src);
  const b = ctx.bindings.get(binding);
  assert.ok(b, `binding ${binding} not found`);
  return sampler.evaluateExpr(lowerMod.lowerExpr(b.node.value), env);
}

const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-12,
  `${a} !== ${b}`);

// =====================================================================
// median — the oracle table
// =====================================================================

const MEDIAN_CASES: [number[], number][] = [
  [[5], 5],                            // n = 1
  [[1, 3], 2],                         // n = 2, mean of both
  [[1, 2, 3, 4], 2.5],                 // even n
  [[1, 2, 3, 4, 5], 3],                // odd n
  [[7, 1, 5, 3], 4],                   // unsorted input
  [[2, 4, 4, 4, 5, 5, 7, 9], 4.5],     // repeats
];

for (const [xs, want] of MEDIAN_CASES) {
  test(`median(${JSON.stringify(xs)}) = ${want}`, () => {
    const src = `m = median([${xs.map((x) => x.toFixed(1)).join(', ')}])`;
    assert.equal(ev(src).get('m'), want);
  });
}

test('median reduces every element of a rank-2 array (like maximum)', () => {
  // Six entries, sorted [1, 2, 3, 4, 5, 9] → ½(3 + 4).
  const fv = ev(`
M = rowstack([[1.0, 2.0, 9.0], [3.0, 4.0, 5.0]])
m = median(M)
`);
  assert.equal(fv.get('m'), 3.5);
});

test('median leaves its operand buffer unsorted', () => {
  // The sort must run on a copy: the operand may be a live Value buffer
  // another binding still reads.
  const v = { shape: [4], data: Float64Array.from([7, 1, 5, 3]) };
  assert.equal(ARITH_OPS.median(v), 4);
  assert.deepEqual(Array.from(v.data), [7, 1, 5, 3]);
});

// =====================================================================
// quantile — the oracle table
// =====================================================================

const QUANTILE_CASES: [number[], number, number][] = [
  // n = 1: k = n on every p, so the interpolation term must vanish —
  // x_(k+1) does not exist.
  [[5], 0, 5],
  [[5], 0.25, 5],
  [[5], 0.5, 5],
  [[5], 1, 5],
  // n = 2.
  [[1, 3], 0, 1],
  [[1, 3], 0.25, 1.5],
  [[1, 3], 0.5, 2],
  [[1, 3], 0.75, 2.5],
  [[1, 3], 1, 3],
  // Even n.
  [[1, 2, 3, 4], 0, 1],
  [[1, 2, 3, 4], 0.25, 1.75],
  [[1, 2, 3, 4], 0.5, 2.5],
  [[1, 2, 3, 4], 0.75, 3.25],
  [[1, 2, 3, 4], 1, 4],
  // Odd n. p = 0.25 lands exactly on the knot h = 2, so the result is
  // the order statistic x_(2) itself; p = 0.2 = 1/n does not.
  [[1, 2, 3, 4, 5], 0.2, 1.8],
  [[1, 2, 3, 4, 5], 0.25, 2],
  [[1, 2, 3, 4, 5], 0.5, 3],
  // Unsorted input.
  [[7, 1, 5, 3], 0.1, 1.6],
  [[7, 1, 5, 3], 0.5, 4],
  [[7, 1, 5, 3], 0.9, 6.4],
  // Repeats.
  [[2, 4, 4, 4, 5, 5, 7, 9], 0.25, 4],
  [[2, 4, 4, 4, 5, 5, 7, 9], 0.5, 4.5],
  [[2, 4, 4, 4, 5, 5, 7, 9], 0.75, 5.5],
];

for (const [xs, p, want] of QUANTILE_CASES) {
  test(`quantile(${JSON.stringify(xs)}, ${p}) = ${want}`, () => {
    const src = `q = quantile([${xs.map((x) => x.toFixed(1)).join(', ')}], ${p})`;
    close(ev(src).get('q') as number, want);
  });
}

// §07 pins quantile as an EXPRESSION, so its floating-point rounding is the
// normative answer and an algebraically equivalent rearrangement is a
// deviation. Every other pinned case agrees under both forms, and `close`
// uses a 1e-12 tolerance, so this is the one case that can see the
// difference — assert it exactly.
//
//   spec  x_(k) + (h−k)(x_(k+1) − x_(k))  = 1 + 0.5·(1e16 − 1) = 5000000000000001
//   lerp  b − (b−a)(1−t)                  =                      5000000000000000
//
// The second form is what numpy's `_lerp` switches to for t ≥ 0.5. numpy
// 2.5.1 percentile(method='linear') and Julia both return ...000; the spec
// formula, recomputed independently in Python, returns ...001.
test('quantile pins the spec expression, not an equivalent lerp (§07)', () => {
  const q = ev('q = quantile([1.0e16, 1.0], 0.5)').get('q') as number;
  assert.strictEqual(q, 5000000000000001);
  assert.notStrictEqual(q, 5000000000000000);
});

test('quantile(xs, 0) = minimum, quantile(xs, 1) = maximum (§07)', () => {
  const fv = ev(`
xs = [7.0, 1.0, 5.0, 3.0]
lo = quantile(xs, 0.0)
hi = quantile(xs, 1.0)
mn = minimum(xs)
mx = maximum(xs)
`);
  assert.equal(fv.get('lo'), fv.get('mn'));
  assert.equal(fv.get('hi'), fv.get('mx'));
});

test('quantile(xs, 0.5) = median(xs) at both parities (§07)', () => {
  const fv = ev(`
odd = [1.0, 2.0, 3.0, 4.0, 5.0]
even = [1.0, 2.0, 3.0, 4.0]
q_odd = quantile(odd, 0.5)
m_odd = median(odd)
q_even = quantile(even, 0.5)
m_even = median(even)
`);
  assert.equal(fv.get('q_odd'), fv.get('m_odd'));
  assert.equal(fv.get('q_even'), fv.get('m_even'));
});

// =====================================================================
// Domain errors — §07 `real arrays, interval(0, 1)`
// =====================================================================

test('quantile: a literal p outside the closed interval is a static error', () => {
  assert.deepEqual(errors('bad = quantile([1.0, 2.0], 1.5)'),
    ['quantile: p must lie in interval(0, 1) (spec §07); got 1.5']);
  assert.deepEqual(errors('bad = quantile([1.0, 2.0], -0.1)'),
    ['quantile: p must lie in interval(0, 1) (spec §07); got -0.1']);
});

test('quantile: p at either endpoint is in domain (§03: closed interval)', () => {
  assert.deepEqual(errors('a = quantile([1.0, 2.0], 0.0)'), []);
  assert.deepEqual(errors('b = quantile([1.0, 2.0], 1.0)'), []);
});

test('quantile: a computed p outside the interval throws at evaluation', () => {
  // Not statically visible, so the runtime owns this one.
  assert.throws(
    () => ARITH_OPS.quantile({ shape: [2], data: Float64Array.from([1, 2]) }, 1.5),
    /p must lie in interval\(0, 1\)/);
});

test('quantile: a table argument is a static error (not a table reduction)', () => {
  const errs = errors(`
t = table(x = [3.0, 1.0, 5.0])
bad = quantile(t, 0.5)
`);
  assert.equal(errs.length, 1);
  assert.match(errs[0], /quantile: argument must be a real array/);
});

test('quantile: a table reaching the runtime throws rather than returning NaN', () => {
  assert.throws(() => ARITH_OPS.quantile({ __table__: true, columns: {}, nrows: 0 }, 0.5),
    /does not list quantile among the table reductions/);
});

// =====================================================================
// Empty input — §07 does not define an empty reduction. median and
// quantile have no order statistics to return, so both give NaN, which
// is what `mean([])` already gives (0/0). Recorded in TODO-flatppl-js.md
// as a spec gap; pinned here so a change is deliberate.
// =====================================================================

test('median / quantile of an empty array ⇒ NaN (matches mean([]))', () => {
  const empty = { shape: [0], data: new Float64Array(0) };
  assert.ok(Number.isNaN(ARITH_OPS.median(empty)));
  assert.ok(Number.isNaN(ARITH_OPS.quantile(empty, 0.5)));
  assert.ok(Number.isNaN(ARITH_OPS.mean(empty)));
});

// =====================================================================
// Table form — §07's "Table reductions" paragraph names median
// =====================================================================

test('median over a table reduces column-wise into a record', () => {
  const fv = ev(`
t = table(x = [3.0, 1.0, 5.0, 9.0], y = [10.0, 20.0, 30.0, 40.0])
m = median(t)
`);
  const m: any = fv.get('m');
  assert.equal(m.x, 4);    // sorted [1, 3, 5, 9] → ½(3 + 5)
  assert.equal(m.y, 25);   // ½(20 + 30)
});

test('median of a table is typed real per column, not the column type', () => {
  // Even n averages two order statistics, so an integer column can
  // reduce to a half-integer.
  const r = processSource(`
t = table(x = [3.0, 1.0, 5.0, 9.0])
m = median(t)
`);
  const t: any = r.bindings.get('m').inferredType;
  assert.equal(t.kind, 'record');
  assert.equal(t.fields.x.kind, 'scalar');
  assert.equal(t.fields.x.prim, 'real');
});

// =====================================================================
// Aggregate eligibility — §04 §sec:aggregate lists median
// =====================================================================

test('aggregate(median, [.i], …) reduces each row', () => {
  const M = { shape: [2, 3], data: Float64Array.from([1, 2, 9, 3, 4, 5]) };
  const out = evalRHS('r = aggregate(median, [.i], A[.i, .j])', 'r', { A: M });
  // Row 1 = [1, 2, 9] → 2 ; row 2 = [3, 4, 5] → 4.
  assert.deepEqual(Array.from(out.data || out), [2, 4]);
});

test('aggregate(median, [], …) reduces to a scalar over every axis', () => {
  const M = { shape: [2, 3], data: Float64Array.from([1, 2, 9, 3, 4, 5]) };
  const out = evalRHS('r = aggregate(median, [], A[.i, .j])', 'r', { A: M });
  assert.equal(out, 3.5);   // sorted [1, 2, 3, 4, 5, 9] → ½(3 + 4)
});

test('aggregate(median, …) types as real even over an integer body', () => {
  const r = processSource(`
A = rowstack([[1, 2], [3, 4]])
s = aggregate(median, [], A[.i, .j])
`);
  const t: any = r.bindings.get('s').inferredType;
  assert.equal(t.kind, 'scalar');
  assert.equal(t.prim, 'real');
});

test('aggregate rejects quantile — it has no f_reduction form', () => {
  const errs = errors(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
r = aggregate(quantile, [.i], A[.i, .j])
`);
  assert.ok(errs.some((m: string) => /must be one of: /.test(m)),
    'expected the eligible-reduction diagnostic, got ' + JSON.stringify(errs));
});
