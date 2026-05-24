'use strict';

// Spec §07 reductions × input-domain coverage:
//
//   sum, mean, prod          — real OR complex arrays (any rank)
//   var, std                 — real arrays (any rank)
//   maximum, minimum         — real arrays (any rank)
//   cumsum, cumprod          — vectors (rank-1)
//   lengthof                 — vectors, tables
//   sizeof                   — vectors, arrays (any rank)
//
// One test per (op, domain) combination. The engine-side coverage
// gap before this file: rank-≥2 reductions and complex sum/mean/prod
// silently fell through the rank-1 type signatures.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');
const sampler = require('../sampler.ts');
const valueLib = require('../value.ts');
const ARITH_OPS = sampler._internal.ARITH_OPS;

function ev(src: string) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'unexpected errors: ' + JSON.stringify(errs));
  return orchestrator.buildDerivations(lifted.bindings).fixedValues;
}

// =====================================================================
// sum / mean / prod — real (rank-1)
// =====================================================================

test('sum on rank-1 real vector', () => {
  const fv = ev('s = sum([1.0, 2.0, 3.0, 4.0])');
  assert.equal(fv.get('s'), 10);
});

test('mean on rank-1 real vector', () => {
  const fv = ev('m = mean([1.0, 2.0, 3.0, 4.0])');
  assert.equal(fv.get('m'), 2.5);
});

test('prod on rank-1 real vector', () => {
  const fv = ev('p = prod([1.0, 2.0, 3.0, 4.0])');
  assert.equal(fv.get('p'), 24);
});

// =====================================================================
// sum / mean / prod — real (rank-2 matrix; spec says "arrays")
// =====================================================================

test('sum on rank-2 real matrix reduces over every entry', () => {
  const fv = ev(`
M = rowstack([[1.0, 2.0], [3.0, 4.0]])
s = sum(M)
`);
  assert.equal(fv.get('s'), 10);
});

test('mean on rank-2 real matrix', () => {
  const fv = ev(`
M = rowstack([[1.0, 2.0], [3.0, 4.0]])
m = mean(M)
`);
  assert.equal(fv.get('m'), 2.5);
});

test('prod on rank-2 real matrix', () => {
  const fv = ev(`
M = rowstack([[1.0, 2.0], [3.0, 4.0]])
p = prod(M)
`);
  assert.equal(fv.get('p'), 24);
});

// =====================================================================
// sum / mean / prod — real (rank-3)
// =====================================================================

test('sum on rank-3 real tensor (fill(1, [2,3,4]) ⇒ 24)', () => {
  const fv = ev(`
T = fill(1.0, [2, 3, 4])
s = sum(T)
`);
  assert.equal(fv.get('s'), 24);
});

// =====================================================================
// sum / mean / prod — complex (rank-1)
// =====================================================================

test('sum on complex rank-1 (real-only entries) ≡ regular sum', () => {
  // ARITH_OPS-level test (complex value literals through surface
  // syntax flow through complex() one-by-one and the literal-list
  // path doesn't yet produce a complex-typed Value).
  const v = valueLib.complexValue(
    new Float64Array([1, 2, 3]),
    new Float64Array([0, 0, 0]),
    [3]);
  const s = ARITH_OPS.sum(v);
  assert.deepEqual(s, { re: 6, im: 0 });
});

test('sum on complex rank-1 with imaginary parts', () => {
  const v = valueLib.complexValue(
    new Float64Array([1, 2, 3]),
    new Float64Array([4, 5, 6]),
    [3]);
  const s = ARITH_OPS.sum(v);
  assert.deepEqual(s, { re: 6, im: 15 });
});

test('mean on complex rank-1', () => {
  const v = valueLib.complexValue(
    new Float64Array([2, 4, 6]),
    new Float64Array([10, 20, 30]),
    [3]);
  const m = ARITH_OPS.mean(v);
  assert.deepEqual(m, { re: 4, im: 20 });
});

test('prod on complex rank-1 — multiplicative complex algebra', () => {
  // (1 + 0i) * (2 + 1i) * (0 + 1i)
  //   = (2 + 1i) * (0 + 1i)
  //   = (2*0 − 1*1) + (2*1 + 1*0) i = -1 + 2i
  const v = valueLib.complexValue(
    new Float64Array([1, 2, 0]),
    new Float64Array([0, 1, 1]),
    [3]);
  const p = ARITH_OPS.prod(v);
  assert.ok(Math.abs(p.re - -1) < 1e-12);
  assert.ok(Math.abs(p.im -  2) < 1e-12);
});

// =====================================================================
// var / std / maximum / minimum — real arrays of any rank
// =====================================================================

test('var on rank-1', () => {
  const fv = ev('v = var([1.0, 2.0, 3.0, 4.0])');
  // population variance: mean=2.5; SS = 2.25+0.25+0.25+2.25 = 5; /4 = 1.25
  assert.ok(Math.abs(fv.get('v') - 1.25) < 1e-12);
});

test('std on rank-1 = sqrt(var)', () => {
  const fv = ev('s = std([1.0, 2.0, 3.0, 4.0])');
  assert.ok(Math.abs(fv.get('s') - Math.sqrt(1.25)) < 1e-12);
});

test('var on rank-2 reduces over all entries', () => {
  const fv = ev(`
M = rowstack([[1.0, 2.0], [3.0, 4.0]])
v = var(M)
`);
  // mean=2.5; SS=5; /4 = 1.25
  assert.ok(Math.abs(fv.get('v') - 1.25) < 1e-12);
});

test('maximum on rank-1', () => {
  const fv = ev('m = maximum([3.0, 1.0, 7.0, 5.0])');
  assert.equal(fv.get('m'), 7);
});

test('minimum on rank-2', () => {
  const fv = ev(`
M = rowstack([[3.0, 1.0], [7.0, 5.0]])
m = minimum(M)
`);
  assert.equal(fv.get('m'), 1);
});

// =====================================================================
// cumsum / cumprod — vectors only per spec §07
// =====================================================================

test('cumsum on rank-1', () => {
  const fv = ev('cs = cumsum([1.0, 2.0, 3.0, 4.0])');
  const cs: any = fv.get('cs');
  assert.deepEqual(Array.from(cs.data), [1, 3, 6, 10]);
});

test('cumprod on rank-1', () => {
  const fv = ev('cp = cumprod([1.0, 2.0, 3.0, 4.0])');
  const cp: any = fv.get('cp');
  assert.deepEqual(Array.from(cp.data), [1, 2, 6, 24]);
});

// =====================================================================
// lengthof — vectors, tables
// =====================================================================

test('lengthof on rank-1 vector', () => {
  const fv = ev('n = lengthof([1.0, 2.0, 3.0, 4.0])');
  assert.equal(fv.get('n'), 4);
});

test('lengthof on shape-explicit Value vector', () => {
  // linspace produces a rank-1 Value.
  const fv = ev('n = lengthof(linspace(0.0, 1.0, 7))');
  assert.equal(fv.get('n'), 7);
});

test('lengthof on a table-like record (row count = first column length)', () => {
  // Table runtime isn't first-class yet but a record-of-columns
  // produced by `record(...)` lets us pin the spec-correct behaviour:
  // lengthof returns the row count.
  const fv = ev(`
t = record(a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0])
n = lengthof(t)
`);
  assert.equal(fv.get('n'), 3);
});

// =====================================================================
// sizeof — vectors, arrays (any rank)
// =====================================================================

test('sizeof on rank-1 ⇒ [length]', () => {
  const fv = ev('s = sizeof([1.0, 2.0, 3.0])');
  const s: any = fv.get('s');
  assert.deepEqual(Array.from(s.data), [3]);
});

test('sizeof on rank-2 ⇒ [m, n]', () => {
  const fv = ev(`
M = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
s = sizeof(M)
`);
  const s: any = fv.get('s');
  assert.deepEqual(Array.from(s.data), [3, 2]);
});

test('sizeof on rank-3 ⇒ [a, b, c]', () => {
  const fv = ev(`
T = fill(0.0, [2, 3, 4])
s = sizeof(T)
`);
  const s: any = fv.get('s');
  assert.deepEqual(Array.from(s.data), [2, 3, 4]);
});

// =====================================================================
// Sanity / regression: F = sum((C - D) .^ 2) on a matrix
// =====================================================================

test('sum / mean of rank-2 zero matrix is 0 (the user F/G pattern)', () => {
  const fv = ev(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[5.0, 6.0], [7.0, 8.0]])
D = A * B
C = rowstack([[19.0, 22.0], [43.0, 50.0]])
F = sum((C - D) .^ 2)
G = mean((C - D) .^ 2)
`);
  assert.ok(Math.abs(fv.get('F')) < 1e-20);
  assert.ok(Math.abs(fv.get('G')) < 1e-20);
});
