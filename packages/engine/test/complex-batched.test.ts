'use strict';

// Chunk 3 of the complex-values thread: complex flowing through the
// batched ARITH_OPS_N dispatch (shape=[N] per-atom complex scalars and
// the complex constructors/accessors). Exercises ARITH_OPS_N directly —
// the same entry point worker.evaluateN / per-atom dispatch use.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const value = require('../value.ts');
const { ARITH_OPS_N } = sampler._internal;
const {
  complexValue, batchedScalar, scalar, vector, readComplex,
  isComplexValue, isValue,
} = value;

const N = 3;
function reim(v) {
  const c = readComplex(v);
  return [Array.from(c.re), Array.from(c.im)];
}

// ---- complex constructor over batched real inputs --------------------

test('complex(reBatched, imBatched) → complex Value shape=[N]', () => {
  const re = batchedScalar([1, 2, 3]);
  const im = batchedScalar([4, 5, 6]);
  const z = ARITH_OPS_N.complex([re, im], N);
  assert.ok(isComplexValue(z));
  assert.deepEqual(z.shape, [N]);
  assert.deepEqual(reim(z), [[1, 2, 3], [4, 5, 6]]);
});

test('complex(scalar, scalar) atom-indep → complex Value shape=[]', () => {
  const z = ARITH_OPS_N.complex([scalar(3), scalar(2)], N);
  assert.ok(isComplexValue(z));
  assert.deepEqual(z.shape, []);
  assert.deepEqual(reim(z), [[3], [2]]);
});

// ---- batched complex arithmetic --------------------------------------

test('add: complex[N] + complex[N] (planar re/im)', () => {
  const a = complexValue([1, 2, 3], [10, 20, 30], [N]);
  const b = complexValue([4, 5, 6], [1, 1, 1], [N]);
  const s = ARITH_OPS_N.add([a, b], N);
  assert.ok(isComplexValue(s) && s.shape[0] === N);
  assert.deepEqual(reim(s), [[5, 7, 9], [11, 21, 31]]);
});

test('mul: complex[N] * complex[N] (full complex multiply per atom)', () => {
  const a = complexValue([1, 0, 2], [0, 1, 0], [N]);   // 1, i, 2
  const b = complexValue([0, 0, 3], [1, 1, 0], [N]);   // i, i, 3
  const p = ARITH_OPS_N.mul([a, b], N);
  // 1*i = i ; i*i = -1 ; 2*3 = 6
  assert.deepEqual(reim(p), [[0, -1, 6], [1, 0, 0]]);
});

test('real batched + complex batched → promotes to complex', () => {
  const r = batchedScalar([1, 2, 3]);
  const c = complexValue([10, 10, 10], [5, 5, 5], [N]);
  const s = ARITH_OPS_N.add([r, c], N);
  assert.ok(isComplexValue(s));
  assert.deepEqual(reim(s), [[11, 12, 13], [5, 5, 5]]);
});

test('neg: complex[N] negates both parts', () => {
  const a = complexValue([1, -2, 3], [-4, 5, -6], [N]);
  const n = ARITH_OPS_N.neg([a], N);
  assert.deepEqual(reim(n), [[-1, 2, -3], [4, -5, 6]]);
});

// ---- accessors degrade to real Values --------------------------------

test('abs2 of complex[N] → REAL Value shape=[N]', () => {
  const z = complexValue([3, 0, 1], [4, 2, 1], [N]);   // |.|² = 25,4,2
  const r = ARITH_OPS_N.abs2([z], N);
  assert.ok(isValue(r) && !isComplexValue(r));
  assert.deepEqual(Array.from(r.data), [25, 4, 2]);
});

test('real / imag / conj of complex[N]', () => {
  const z = complexValue([1, 2, 3], [4, 5, 6], [N]);
  assert.deepEqual(Array.from(ARITH_OPS_N.real([z], N).data), [1, 2, 3]);
  assert.deepEqual(Array.from(ARITH_OPS_N.imag([z], N).data), [4, 5, 6]);
  assert.deepEqual(reim(ARITH_OPS_N.conj([z], N)), [[1, 2, 3], [-4, -5, -6]]);
});

test('conj / real on a REAL batched value stays real (no-op path)', () => {
  const r = batchedScalar([7, 8, 9]);
  const c = ARITH_OPS_N.conj([r], N);
  assert.ok(isValue(c) && !isComplexValue(c));
  assert.deepEqual(Array.from(c.data), [7, 8, 9]);
});

test('cis(thetaBatched) → unit-modulus complex Value', () => {
  const theta = batchedScalar([0, Math.PI / 2, Math.PI]);
  const z = ARITH_OPS_N.cis([theta], N);
  assert.ok(isComplexValue(z));
  const { re, im } = readComplex(z);
  assert.ok(Math.abs(re[0] - 1) < 1e-12 && Math.abs(im[0]) < 1e-12);
  assert.ok(Math.abs(re[1]) < 1e-12 && Math.abs(im[1] - 1) < 1e-12);
  assert.ok(Math.abs(re[2] + 1) < 1e-12 && Math.abs(im[2]) < 1e-12);
});

// ---- spec §03 driving example: A_total = A_sig*c + A_bkg; abs2 -------

test('amplitude model: abs2(A_sig*coupling + A_bkg) per atom', () => {
  const A_sig  = complexValue([1, 2], [1, 0], [2]);
  const coupling = complexValue([0, 1], [1, 0], [2]);   // i, 1
  const A_bkg  = complexValue([1, 1], [0, 0], [2]);
  const prod = ARITH_OPS_N.mul([A_sig, coupling], 2);
  const tot  = ARITH_OPS_N.add([prod, A_bkg], 2);
  const I    = ARITH_OPS_N.abs2([tot], 2);
  // atom0: (1+i)*i = -1+i ; +1 = 0+i ; |.|²=1
  // atom1: (2)*1 = 2 ; +1 = 3 ; |.|²=9
  assert.ok(!isComplexValue(I));
  assert.deepEqual(Array.from(I.data), [1, 9]);
});

// ---- shape-rich complex still routes via value-ops (chunk 2) ----------

test('add of complex shape=[N,k] routes through value-ops, not guarded', () => {
  const a = complexValue([1, 2, 3, 4], [0, 0, 0, 0], [2, 2]);
  const b = complexValue([10, 10, 10, 10], [1, 1, 1, 1], [2, 2]);
  const s = ARITH_OPS_N.add([a, b], 2);
  assert.ok(isComplexValue(s));
  assert.deepEqual(s.shape, [2, 2]);
  assert.deepEqual(reim(s), [[11, 12, 13, 14], [1, 1, 1, 1]]);
});

test('abs2 of shape-rich complex → real Value, shape preserved', () => {
  const z = complexValue([1, 2, 3, 4], [1, 1, 1, 1], [2, 2]);
  const r = ARITH_OPS_N.abs2([z], 2);
  assert.ok(isValue(r) && !isComplexValue(r));
  assert.deepEqual(r.shape, [2, 2]);
  // |1+i|²=2, |2+i|²=5, |3+i|²=10, |4+i|²=17
  assert.deepEqual(Array.from(r.data), [2, 5, 10, 17]);
});

test('exp / conj / real of a shape-rich complex (per-atom vector)', () => {
  const z = complexValue([0, 1], [Math.PI, 0], [1, 2]);  // (iπ, 1)
  const e = ARITH_OPS_N.exp([z], 1);                       // (e^{iπ}=-1, e)
  assert.ok(isComplexValue(e) && e.shape[0] === 1);
  const ce = readComplex(e);
  assert.ok(Math.abs(ce.re[0] + 1) < 1e-12 && Math.abs(ce.im[0]) < 1e-12);
  assert.ok(Math.abs(ce.re[1] - Math.E) < 1e-12);
  const c = readComplex(ARITH_OPS_N.conj([z], 1));
  assert.deepEqual(Array.from(c.im), [-Math.PI, -0]);
  const re = ARITH_OPS_N.real([z], 1);
  assert.ok(!isComplexValue(re));
  assert.deepEqual(Array.from(re.data), [0, 1]);
  assert.deepEqual(re.shape, [1, 2]);
});

test('complex(reVec, imVec) constructor over shape-rich real inputs', () => {
  const A = value.withShape(Float64Array.from([1, 2, 3, 4]), [2, 2]);
  const B = value.withShape(Float64Array.from([5, 6, 7, 8]), [2, 2]);
  const z = ARITH_OPS_N.complex([A, B], 2);
  assert.ok(isComplexValue(z));
  assert.deepEqual(z.shape, [2, 2]);
  assert.deepEqual(reim(z), [[1, 2, 3, 4], [5, 6, 7, 8]]);
});

// ---- divide (spec §07 function-form of `/`) ---------------------------

test('divide: batched real → fast Float64Array broadcast Value', () => {
  const a = batchedScalar([10, 20, 30]);
  const b = batchedScalar([2, 4, 5]);
  const r = ARITH_OPS_N.divide([a, b], 3);
  assert.ok(isValue(r) && !isComplexValue(r));
  assert.deepEqual(Array.from(r.data), [5, 5, 6]);
});

test('divide: batched complex (was broken — per-atom fallback)', () => {
  const a = complexValue([1, 4], [2, 0], [2]);   // 1+2i, 4
  const b = complexValue([0, 2], [1, 0], [2]);   // i,   2
  const r = ARITH_OPS_N.divide([a, b], 2);
  assert.ok(isComplexValue(r));
  // (1+2i)/i = 2-i ; 4/2 = 2
  assert.deepEqual(reim(r), [[2, 2], [-1, 0]]);
});
