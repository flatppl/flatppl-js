'use strict';

// Spec §07 Norms and normalization: l1norm, l2norm, l1unit, l2unit,
// logsumexp, softmax, logsoftmax. All pure vector→scalar or
// vector→vector reductions dispatched through ARITH_OPS.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');

function lit(v)        { return { kind: 'lit', value: v }; }
function vec(...vs)    { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function call(op, v)   { return { kind: 'call', op, args: [v] }; }
const ev = (ir) => sampler.evaluateExpr(ir, {});

function arrClose(a, b, tol) {
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
  const sum = r.reduce((s, v) => s + v, 0);
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
  const expLs = ls.map(x => Math.exp(x));
  assert.ok(arrClose(expLs, s, 1e-12));
});

test('logsoftmax: each entry equals v_i − logsumexp(v)', () => {
  const v   = vec(1.0, 2.0, 3.0);
  const ls  = ev(call('logsoftmax', v));
  const lse = ev(call('logsumexp', v));
  const expected = [1 - lse, 2 - lse, 3 - lse];
  assert.ok(arrClose(ls, expected, 1e-12));
});
