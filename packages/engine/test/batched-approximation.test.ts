'use strict';

// Tests for Phase 7a: batched approximation functions (polynomial,
// bernstein, stepwise) under evaluateExprN.
//
// When the x argument is per-atom (Float64Array(N) or Value shape=[N])
// and the coefficient-class arguments are atom-indep, the dispatcher
// runs ONE tight loop over the N-atom batch instead of N JS-function
// calls through the per-atom fallback. The result matches the scalar
// implementation exactly.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const valueLib = require('..').value;

const lit = (v: any) => ({ kind: 'lit', value: v });
const ref = (n: any) => ({ kind: 'ref', ns: 'self', name: n });
const litArr = (arr: any) => ({ kind: 'call', op: 'vector', args: arr.map(lit) });

function close(a: any, b: any, tol?: any) {
  tol = tol == null ? 1e-12 : tol;
  return Math.abs(a - b) <= tol;
}

// =====================================================================
// polynomial — batched against the closed-form Horner
// =====================================================================

test('polynomial: per-atom x via Float64Array, atom-indep coefficients', () => {
  // p(x) = 1 + 2x + 3x²; evaluate at x = [0, 1, 2, 3] → [1, 6, 17, 34]
  const x = new Float64Array([0, 1, 2, 3]);
  const ir = {
    kind: 'call', op: 'polynomial',
    kwargs: { coefficients: litArr([1, 2, 3]), x: ref('x') },
  };
  const r = sampler.evaluateExprN(ir, { x }, 4, {});
  assert.ok(r instanceof Float64Array);
  assert.deepEqual(Array.from(r), [1, 6, 17, 34]);
});

test('polynomial: per-atom x via Value shape=[N] → returns Value', () => {
  const x = valueLib.batchedScalar(new Float64Array([0, 1, 2]));
  const ir = {
    kind: 'call', op: 'polynomial',
    kwargs: { coefficients: litArr([5, 0, 1]), x: ref('x') },
  };
  const r = sampler.evaluateExprN(ir, { x }, 3, {});
  // p(x) = 5 + x² → [5, 6, 9]
  assert.ok(valueLib.isValue(r));
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [5, 6, 9]);
});

test('polynomial: atom-indep x falls through (per-atom fallback / one-shot)', () => {
  // Both args atom-indep → not batched, returns scalar via one-shot eval.
  const ir = {
    kind: 'call', op: 'polynomial',
    kwargs: { coefficients: litArr([1, 2, 3]), x: lit(2) },
  };
  const r = sampler.evaluateExprN(ir, null, 4, {});
  // p(2) = 1 + 4 + 12 = 17
  assert.equal(r, 17);
});

test('polynomial: zero polynomial → all zeros', () => {
  const x = new Float64Array([1, 2, 3]);
  const ir = {
    kind: 'call', op: 'polynomial',
    kwargs: { coefficients: litArr([]), x: ref('x') },
  };
  const r = sampler.evaluateExprN(ir, { x }, 3, {});
  assert.deepEqual(Array.from(r), [0, 0, 0]);
});

// =====================================================================
// bernstein — batched against the scalar bernstein implementation
// =====================================================================

test('bernstein: per-atom x', () => {
  // Bernstein basis with coeffs = [0, 1] (linear interpolation a=0,b=1
  // on [0,1]) — output is just x itself.
  const x = new Float64Array([0, 0.25, 0.5, 0.75, 1.0]);
  const ir = {
    kind: 'call', op: 'bernstein',
    kwargs: { coefficients: litArr([0, 1]), x: ref('x') },
  };
  const r = sampler.evaluateExprN(ir, { x }, 5, {});
  for (let i = 0; i < 5; i++) {
    assert.ok(close(r[i], x[i]), `bernstein(linear, x=${x[i]}) = ${r[i]}`);
  }
});

test('bernstein: matches scalar implementation pointwise', () => {
  // Generic check: batched result equals N scalar calls.
  const x = new Float64Array([0.1, 0.3, 0.5, 0.7, 0.9]);
  const coeffs = [2, 5, 3, 7];
  const irBatched = {
    kind: 'call', op: 'bernstein',
    kwargs: { coefficients: litArr(coeffs), x: ref('x') },
  };
  const r = sampler.evaluateExprN(irBatched, { x }, 5, {});
  // Compare each entry to a single-point evaluateExpr call.
  for (let i = 0; i < 5; i++) {
    const irScalar = {
      kind: 'call', op: 'bernstein',
      kwargs: { coefficients: litArr(coeffs), x: lit(x[i]) },
    };
    const expected = sampler.evaluateExpr(irScalar, {});
    assert.ok(close(r[i], expected, 1e-12),
      `mismatch at i=${i}: batched=${r[i]} scalar=${expected}`);
  }
});

test('bernstein: x = 1.0 boundary picks last coefficient', () => {
  const x = new Float64Array([1.0]);
  const ir = {
    kind: 'call', op: 'bernstein',
    kwargs: { coefficients: litArr([2, 5, 7]), x: ref('x') },
  };
  const r = sampler.evaluateExprN(ir, { x }, 1, {});
  assert.equal(r[0], 7);
});

// =====================================================================
// stepwise — batched piecewise constant
// =====================================================================

test('stepwise: per-atom x', () => {
  // edges = [0, 1, 2, 3], values = [10, 20, 30]
  //   x ∈ [0, 1) → 10
  //   x ∈ [1, 2) → 20
  //   x ∈ [2, 3] → 30
  const x = new Float64Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]);
  const ir = {
    kind: 'call', op: 'stepwise',
    kwargs: { edges: litArr([0, 1, 2, 3]), values: litArr([10, 20, 30]), x: ref('x') },
  };
  const r = sampler.evaluateExprN(ir, { x }, 6, {});
  assert.deepEqual(Array.from(r), [10, 20, 20, 30, 30, 30]);
});

test('stepwise: out-of-range produces NaN', () => {
  const x = new Float64Array([-1, 0, 5]);
  const ir = {
    kind: 'call', op: 'stepwise',
    kwargs: { edges: litArr([0, 1, 2]), values: litArr([10, 20]), x: ref('x') },
  };
  const r = sampler.evaluateExprN(ir, { x }, 3, {});
  assert.ok(Number.isNaN(r[0]));
  assert.equal(r[1], 10);
  assert.ok(Number.isNaN(r[2]));
});

test('stepwise: edges/values length mismatch throws', () => {
  const x = new Float64Array([0.5]);
  const ir = {
    kind: 'call', op: 'stepwise',
    kwargs: { edges: litArr([0, 1, 2]), values: litArr([10]), x: ref('x') },
  };
  assert.throws(() => sampler.evaluateExprN(ir, { x }, 1, {}),
    /edges length must equal values length \+ 1/);
});

// =====================================================================
// Regression: per-atom fallback for x with downstream refs
// =====================================================================

test('regression: polynomial with x as a per-atom expression composes correctly', () => {
  // x = ref(a) * 2; polynomial of x.
  // a = [1, 2, 3] → x = [2, 4, 6]; p(x) = 1 + x → [3, 5, 7]
  const a = new Float64Array([1, 2, 3]);
  const ir = {
    kind: 'call', op: 'polynomial',
    kwargs: {
      coefficients: litArr([1, 1]),
      x: { kind: 'call', op: 'mul', args: [ref('a'), lit(2)] },
    },
  };
  const r = sampler.evaluateExprN(ir, { a }, 3, {});
  assert.deepEqual(Array.from(r), [3, 5, 7]);
});
