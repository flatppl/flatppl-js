'use strict';

// Tests for complex arithmetic — Phase B of "type inference + complex
// arithmetic". Adds scalar complex values (plain `{re, im}` JS objects)
// + auto-promotion of real → complex at op boundaries.
//
// Out of scope for this commit (deferred follow-up):
//   - Per-atom complex Values (atom-batched shape=[N] of complex):
//     would extend Value with optional `.im` Float64Array (matching
//     TF.js's separate-real/imag storage).
//   - Worker-side evaluateN handling of complex-result expressions.
//   - Materialiser support for complex-valued bindings.
//
// This commit gives correct scalar complex arithmetic at the
// ARITH_OPS / evaluateExpr layer, which is enough for fixed-phase
// expressions and within-expression complex sub-computations.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const { ARITH_OPS } = sampler._internal;

const lit = v => ({ kind: 'lit', value: v });
const cnst = n => ({ kind: 'const', name: n });
const call = (op, ...args) => ({ kind: 'call', op, args });

function close(a, b, tol) {
  tol = tol == null ? 1e-12 : tol;
  return Math.abs(a - b) <= tol;
}
function cclose(a, b, tol) {
  return close(a.re, b.re, tol) && close(a.im, b.im, tol);
}

// =====================================================================
// Constructor + accessors
// =====================================================================

test('complex(re, im) constructor', () => {
  assert.deepEqual(ARITH_OPS.complex(3, 4), { re: 3, im: 4 });
});

test('real(z) and imag(z)', () => {
  const z = { re: 3, im: 4 };
  assert.equal(ARITH_OPS.real(z), 3);
  assert.equal(ARITH_OPS.imag(z), 4);
  // Identity on real numbers.
  assert.equal(ARITH_OPS.real(5), 5);
  assert.equal(ARITH_OPS.imag(5), 0);
});

test('conj flips imag sign', () => {
  assert.deepEqual(ARITH_OPS.conj({ re: 3, im: 4 }), { re: 3, im: -4 });
  // Identity on real.
  assert.equal(ARITH_OPS.conj(5), 5);
});

test('cis(theta) = cos(theta) + i sin(theta)', () => {
  const z = ARITH_OPS.cis(Math.PI / 2);
  assert.ok(close(z.re, 0));
  assert.ok(close(z.im, 1));
  const w = ARITH_OPS.cis(Math.PI);
  assert.ok(close(w.re, -1));
  assert.ok(close(w.im, 0));
});

test('im constant resolves to (0, 1)', () => {
  // Use sampler.evaluateExpr with a const-IR.
  const r = sampler.evaluateExpr(cnst('im'), {});
  assert.deepEqual(r, { re: 0, im: 1 });
});

// =====================================================================
// Arithmetic operators with auto-promotion
// =====================================================================

test('add: complex + complex', () => {
  const r = ARITH_OPS.add({ re: 1, im: 2 }, { re: 3, im: 4 });
  assert.deepEqual(r, { re: 4, im: 6 });
});

test('add: real + complex auto-promotes', () => {
  const r = ARITH_OPS.add(5, { re: 1, im: 2 });
  assert.deepEqual(r, { re: 6, im: 2 });
});

test('sub: complex - real', () => {
  const r = ARITH_OPS.sub({ re: 5, im: 3 }, 2);
  assert.deepEqual(r, { re: 3, im: 3 });
});

test('mul: (1+2i)(3+4i) = -5 + 10i', () => {
  const r = ARITH_OPS.mul({ re: 1, im: 2 }, { re: 3, im: 4 });
  assert.ok(cclose(r, { re: -5, im: 10 }));
});

test('mul: real * complex broadcasts', () => {
  const r = ARITH_OPS.mul(2, { re: 1, im: 3 });
  assert.deepEqual(r, { re: 2, im: 6 });
});

test('div: (1+i)/(1-i) = i', () => {
  const r = ARITH_OPS.div({ re: 1, im: 1 }, { re: 1, im: -1 });
  assert.ok(cclose(r, { re: 0, im: 1 }));
});

test('neg of complex', () => {
  assert.deepEqual(ARITH_OPS.neg({ re: 3, im: -2 }), { re: -3, im: 2 });
});

test('pow: i^2 = -1', () => {
  const r = ARITH_OPS.pow({ re: 0, im: 1 }, 2);
  assert.ok(cclose(r, { re: -1, im: 0 }, 1e-12));
});

// =====================================================================
// Elementary functions on complex
// =====================================================================

test('abs of complex: |3+4i| = 5', () => {
  assert.equal(ARITH_OPS.abs({ re: 3, im: 4 }), 5);
});

test('abs2 of complex: |3+4i|^2 = 25', () => {
  assert.equal(ARITH_OPS.abs2({ re: 3, im: 4 }), 25);
});

test('exp of complex: exp(iπ) = -1 (Euler identity)', () => {
  const r = ARITH_OPS.exp({ re: 0, im: Math.PI });
  assert.ok(cclose(r, { re: -1, im: 0 }, 1e-10));
});

test('log of complex: log(-1) = iπ (principal branch)', () => {
  const r = ARITH_OPS.log({ re: -1, im: 0 });
  assert.ok(close(r.re, 0));
  assert.ok(close(r.im, Math.PI, 1e-12));
});

test('sqrt of complex: sqrt(-1) = i', () => {
  const r = ARITH_OPS.sqrt({ re: -1, im: 0 });
  assert.ok(close(r.re, 0));
  assert.ok(close(r.im, 1, 1e-12));
});

test('sqrt of positive real: sqrt(4) = 2', () => {
  // Real input goes through the real path (Math.sqrt).
  assert.equal(ARITH_OPS.sqrt(4), 2);
});

test('Euler identity: exp(i·θ) = cis(θ)', () => {
  const theta = 0.7;
  const lhs = ARITH_OPS.exp({ re: 0, im: theta });
  const rhs = ARITH_OPS.cis(theta);
  assert.ok(cclose(lhs, rhs, 1e-12));
});

test('conj(z) · z = |z|^2 (real)', () => {
  const z = { re: 3, im: 4 };
  const r = ARITH_OPS.mul(ARITH_OPS.conj(z), z);
  // result should be {re: 25, im: 0}
  assert.ok(close(r.re, 25));
  assert.ok(close(r.im, 0));
});

// =====================================================================
// Through evaluateExpr (end-to-end)
// =====================================================================

test('evaluateExpr: complex(2.0, 3.0)', () => {
  const r = sampler.evaluateExpr(call('complex', lit(2), lit(3)), {});
  assert.deepEqual(r, { re: 2, im: 3 });
});

test('evaluateExpr: 2 + 3 * im (surface form of complex literal)', () => {
  const ir = call('add', lit(2), call('mul', lit(3), cnst('im')));
  const r = sampler.evaluateExpr(ir, {});
  assert.deepEqual(r, { re: 2, im: 3 });
});

test('evaluateExpr: |3 + 4i|^2 via abs2', () => {
  const ir = call('abs2',
    call('add', lit(3), call('mul', lit(4), cnst('im'))));
  const r = sampler.evaluateExpr(ir, {});
  assert.equal(r, 25);
});

// =====================================================================
// Real-only path regression — adding/multiplying reals never produces
// complex unintentionally.
// =====================================================================

test('regression: real-only arithmetic stays real', () => {
  assert.equal(ARITH_OPS.add(2, 3), 5);
  assert.equal(ARITH_OPS.mul(2, 3), 6);
  assert.equal(ARITH_OPS.sub(5, 3), 2);
  assert.equal(typeof ARITH_OPS.add(2, 3), 'number');
});
