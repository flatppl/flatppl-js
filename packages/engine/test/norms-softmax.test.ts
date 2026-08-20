'use strict';

// Spec §07 Norms and normalization: l1norm, l2norm, linfnorm, l1unit,
// l2unit, logsumexp, softmax, logsoftmax. All pure vector→scalar or
// vector→vector reductions dispatched through ARITH_OPS.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const { toJS } = require('./_value-helpers.ts');

function lit(v: any)        { return { kind: 'lit', value: v }; }
function vec(...vs: any[])    { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function call(op: any, v: any)   { return { kind: 'call', op, args: [v] }; }
const evRaw = (ir: any) => sampler.evaluateExpr(ir, {});
const ev = (ir: any) => toJS(evRaw(ir));

function arrClose(a: any, b: any, tol?: any) {
  tol = tol == null ? 1e-12 : tol;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!(Math.abs(a[i] - b[i]) <= tol)) return false;
  }
  return true;
}

// =====================================================================
// l1norm / l2norm
// =====================================================================

test('l1norm: Σ|v_i|', () => {
  assert.equal(ev(call('l1norm', vec(1, -2, 3, -4))), 10);
});

test('l2norm: classic 3-4-5 right triangle', () => {
  assert.equal(ev(call('l2norm', vec(3, 4))), 5);
});

test('l1norm / l2norm on empty vector ⇒ 0', () => {
  assert.equal(ev(call('l1norm', vec())), 0);
  assert.equal(ev(call('l2norm', vec())), 0);
});

// =====================================================================
// linfnorm — max_i |v_i|
// =====================================================================

test('linfnorm: largest magnitude wins, sign ignored', () => {
  assert.equal(ev(call('linfnorm', vec(-3, 2, -7))), 7);
  assert.equal(ev(call('linfnorm', vec(1, 1, 1))), 1);
});

test('linfnorm on empty vector ⇒ 0 (same convention as l1norm / l2norm)', () => {
  // Oracle: LinearAlgebra.norm(Float64[], Inf) == 0.0.
  assert.equal(ev(call('linfnorm', vec())), 0);
});

test('linfnorm ARITH_OPS complex branch takes each modulus (not reachable from source)', () => {
  // |3 + 4i| = 5 beats |1 + 0i| = 1. Oracle: norm([3+4im, 1], Inf) == 5.
  //
  // The Value is hand-built and ARITH_OPS is called directly because there
  // is NO surface route: `types.ts` gives linfnorm `array(1, …, REAL)`, and
  // the engine has no real-or-complex array signature at all, so
  // `linfnorm([complex(3.0, 4.0), …])` is a static type error. §07 gives
  // all three norms the domain `real/complex vectors`, so the complex half
  // is unimplemented at the surface for l1norm / l2norm / linfnorm alike.
  // This test pins the runtime branch only; it does not certify the domain.
  // Closing the gap needs a new type form (TODO-flatppl-js.md).
  const v = {
    shape: [2],
    data: Float64Array.from([3, 1]),
    im: Float64Array.from([4, 0]),
    dtype: 'complex',
  };
  assert.equal(sampler._internal.ARITH_OPS.linfnorm(v), 5);
});

// =====================================================================
// l1unit / l2unit
// =====================================================================

test('l1unit: sums to 1 (absolute) — uniform probability vector', () => {
  const r = ev(call('l1unit', vec(2, 2, 2, 2)));
  assert.ok(arrClose(r, [0.25, 0.25, 0.25, 0.25]));
});

test('l2unit: norm of result is 1', () => {
  const r = ev(call('l2unit', vec(3, 4)));
  assert.ok(Math.abs(Math.hypot(...r) - 1) < 1e-12);
  // Direction preserved: [3, 4] / 5 = [0.6, 0.8]
  assert.ok(arrClose(r, [0.6, 0.8]));
});

test('l1unit / l2unit on zero-norm vector throws', () => {
  assert.throws(() => ev(call('l1unit', vec(0, 0, 0))), /zero-norm/);
  assert.throws(() => ev(call('l2unit', vec(0, 0, 0))), /zero-norm/);
});

// Empty input — no quotient is ever evaluated, so the result is
// vacuously the empty vector (empty-array ruling, flatppl-dev/
// empty-arrays-ruling.md), not the zero-norm error. Was a throw.
test('l1unit / l2unit on empty vector ⇒ the empty vector, not the zero-norm error', () => {
  assert.deepEqual(ev(call('l1unit', vec())), []);
  assert.deepEqual(ev(call('l2unit', vec())), []);
});

// =====================================================================
// logsumexp — numerically stable log Σ exp
// =====================================================================

test('logsumexp: log Σ exp on small uniform vector', () => {
  // logsumexp([0, 0, 0]) = log(3)
  assert.ok(Math.abs(ev(call('logsumexp', vec(0, 0, 0))) - Math.log(3)) < 1e-12);
});

test('logsumexp: numerically stable at large entries', () => {
  // Direct exp(1000) would overflow; logsumexp must give 1000 + log(1).
  assert.ok(Math.abs(ev(call('logsumexp', vec(1000))) - 1000) < 1e-12);
  // [1000, 1000, 1000]: result = 1000 + log(3).
  assert.ok(Math.abs(ev(call('logsumexp', vec(1000, 1000, 1000))) -
    (1000 + Math.log(3))) < 1e-12);
});

test('logsumexp: empty vector ⇒ -Infinity', () => {
  assert.equal(ev(call('logsumexp', vec())), -Infinity);
});

// =====================================================================
// softmax / logsoftmax
// =====================================================================

test('softmax: uniform input ⇒ uniform output', () => {
  const r = ev(call('softmax', vec(0, 0, 0)));
  assert.ok(arrClose(r, [1/3, 1/3, 1/3], 1e-12));
});

test('softmax: sums to 1', () => {
  const r = ev(call('softmax', vec(1.0, 2.0, 3.0, 0.5)));
  const sum = r.reduce((s: any, v: any) => s + v, 0);
  assert.ok(Math.abs(sum - 1) < 1e-12);
});

test('softmax: shift-invariance — adding a constant to all entries gives the same output', () => {
  const r1 = ev(call('softmax', vec(1, 2, 3)));
  const r2 = ev(call('softmax', vec(101, 102, 103)));
  assert.ok(arrClose(r1, r2, 1e-12));
});

test('logsoftmax: exp ∘ logsoftmax = softmax', () => {
  const v = vec(0.5, -1.0, 2.0, 0.25);
  const ls = ev(call('logsoftmax', v));
  const s  = ev(call('softmax', v));
  const expLs = ls.map((x: any) => Math.exp(x));
  assert.ok(arrClose(expLs, s, 1e-12));
});

test('logsoftmax: each entry equals v_i − logsumexp(v)', () => {
  const v   = vec(1.0, 2.0, 3.0);
  const ls  = ev(call('logsoftmax', v));
  const lse = ev(call('logsumexp', v));
  const expected = [1 - lse, 2 - lse, 3 - lse];
  assert.ok(arrClose(ls, expected, 1e-12));
});

// Empty input — softmax / logsoftmax give the empty vector (empty-array
// ruling), already conformant; pinned here so a future change is
// deliberate.
test('softmax([]) / logsoftmax([]) ⇒ the empty vector', () => {
  assert.deepEqual(ev(call('softmax', vec())), []);
  assert.deepEqual(ev(call('logsoftmax', vec())), []);
});
