'use strict';

// Tests for the atom-batched cross — Phase 2d of the shape-explicit
// refactor. Covers the per-atom shape combinations needed for
// MvNormal-style models (`mu + L * z` with shape=[N, n] z).
//
// IMPORTANT — atom-count convention in this test file:
// The atom count N is chosen larger than every matrix / vector
// intrinsic dimension in each test, mirroring real models (N >>
// k/m/n; typical N is ~1024). This avoids the fundamental shape
// ambiguity between an atom-indep matrix shape=[m, n] and an atom-
// batched vector shape=[N=m, n]: when shape[0] coincidentally equals
// N the dispatcher cannot distinguish them. In practice this never
// happens (matrices have a handful of rows; N is thousands).
//
// Coverage:
//
//   1. mulN: matrix(m,n) × shape=[N, n] → shape=[N, m]
//      (the L * z core for MvNormal sampling).
//   2. mulN: transpose(matrix) × shape=[N, n] honours the tag.
//   3. mulN delegates to atom-indep mul when no operand has atom axis.
//   4. mulN rejects unsupported atom combinations (clear message).
//   5. addN: shape=[k] + shape=[N, k] → shape=[N, k] (broadcast).
//   6. addN: shape=[N, k] + shape=[N, k] → elementwise.
//   7. addN: rejection on per-atom shape mismatch / orientation mismatch.
//   8. subN: symmetric to addN, non-commutative scalar broadcast.
//   9. negN: pointwise at any rank (atom-batched included).
//  10. Full pipeline: `mu + L * z` end-to-end through the ARITH_OPS_N
//      dispatcher.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const valueLib = require('..').value;
const valueOps = require('../value-ops');
const sampler = require('../sampler');

const { scalar, vector, matrix, batchedVector, transpose,
        getTag } = valueLib;

// =====================================================================
// mulN: matrix × atom-batched vector
// =====================================================================

test('mulN: matrix(2,2) × shape=[N, 2] → shape=[N, 2]', () => {
  // Identity matrix is a no-op on every atom's vector.
  // N=5 atoms, each a 2-vector — N > all matrix dims, no ambiguity.
  const N = 5;
  const I = matrix([1, 0, 0, 1], 2, 2);
  const flat = new Float64Array([
    1, 2,  3, 4,  5, 6,  7, 8,  9, 10,
  ]);
  const Z = batchedVector(flat, 2);
  const r = valueOps.mulN(I, Z, N);
  assert.deepEqual(r.shape, [N, 2]);
  assert.deepEqual(Array.from(r.data), Array.from(flat));
});

test('mulN: non-identity matrix per-atom matvec', () => {
  // L = [[2, 0], [1, 3]] (lower triangular).
  // N=4 atoms: [1, 0], [0, 1], [1, 1], [2, -1] → expected per-atom
  // matvec: [2, 1], [0, 3], [2, 4], [4, -1]
  const N = 4;
  const L = matrix([2, 0, 1, 3], 2, 2);
  const Z = batchedVector(new Float64Array([
    1, 0,  0, 1,  1, 1,  2, -1,
  ]), 2);
  const r = valueOps.mulN(L, Z, N);
  assert.deepEqual(r.shape, [N, 2]);
  assert.deepEqual(Array.from(r.data), [2, 1, 0, 3, 2, 4, 4, -1]);
});

test('mulN: transpose(matrix) × shape=[N, n] uses the tag', () => {
  // M = [[1, 2], [3, 4]]; M^T = [[1, 3], [2, 4]].
  // N=4 atoms: [1, 0], [0, 1], [1, 1], [2, 3].
  // M^T @ atoms:
  //   [1, 0] → [1, 2]
  //   [0, 1] → [3, 4]
  //   [1, 1] → [4, 6]
  //   [2, 3] → [11, 16]
  const N = 4;
  const M = matrix([1, 2, 3, 4], 2, 2);
  const Z = batchedVector(new Float64Array([
    1, 0,  0, 1,  1, 1,  2, 3,
  ]), 2);
  const r = valueOps.mulN(transpose(M), Z, N);
  assert.deepEqual(r.shape, [N, 2]);
  assert.deepEqual(Array.from(r.data), [1, 2, 3, 4, 4, 6, 11, 16]);
});

test('mulN: shape mismatch (matrix cols ≠ batched-vec dim) rejected', () => {
  const N = 1024;
  const M = matrix([1, 2, 3, 4], 2, 2);  // shape=[2, 2]
  const Z = batchedVector(new Float64Array(N * 3), 3);  // shape=[N, 3]
  assert.throws(
    () => valueOps.mulN(M, Z, N),
    /matrix×batchedVector shape mismatch/);
});

test('mulN: delegates to atom-indep mul when no atom axis', () => {
  // Both args lack a leading-N axis (vectors don't have one when
  // they're atom-indep). Result should equal `mul` on the same args.
  const N = 1024;
  const M = matrix([1, 2, 3, 4], 2, 2);
  const v = vector([5, 6]);
  const r1 = valueOps.mulN(M, v, N);
  const r2 = valueOps.mul(M, v);
  assert.deepEqual(Array.from(r1.data), Array.from(r2.data));
});

test('mulN: unsupported atom-batched matrix × atom-batched vector', () => {
  // Phase 6+ territory.
  const N = 64;
  const Mb = { shape: [N, 2, 2], data: new Float64Array(N * 4) };
  const Zb = batchedVector(new Float64Array(N * 2), 2);  // shape=[N, 2]
  assert.throws(
    () => valueOps.mulN(Mb, Zb, N),
    /unsupported atom-batched/);
});

// =====================================================================
// addN: atom-indep + atom-batched broadcast
// =====================================================================

test('addN: shape=[k] + shape=[N, k] → broadcast', () => {
  // N=4 atoms, mu=[10, 20]. Z atoms = [1,2],[3,4],[5,6],[7,8].
  const N = 4;
  const mu = vector([10, 20]);
  const Z = batchedVector(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 2);
  const r = valueOps.addN(mu, Z, N);
  assert.deepEqual(r.shape, [N, 2]);
  assert.deepEqual(Array.from(r.data), [11, 22, 13, 24, 15, 26, 17, 28]);
});

test('addN: shape=[N, k] + shape=[k] (symmetric)', () => {
  const N = 4;
  const mu = vector([10, 20]);
  const Z = batchedVector(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 2);
  const r = valueOps.addN(Z, mu, N);
  assert.deepEqual(Array.from(r.data), [11, 22, 13, 24, 15, 26, 17, 28]);
});

test('addN: shape=[N, k] + shape=[N, k] → elementwise per atom', () => {
  const N = 4;
  const Z1 = batchedVector(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 2);
  const Z2 = batchedVector(new Float64Array([10, 20, 30, 40, 50, 60, 70, 80]), 2);
  const r = valueOps.addN(Z1, Z2, N);
  assert.deepEqual(r.shape, [N, 2]);
  assert.deepEqual(Array.from(r.data), [11, 22, 33, 44, 55, 66, 77, 88]);
});

test('addN: per-atom shape mismatch rejected', () => {
  const N = 4;
  const mu = vector([10, 20, 30]);  // shape [3]
  const Z = batchedVector(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 2);
  assert.throws(
    () => valueOps.addN(Z, mu, N),
    /per-atom shape mismatch/);
});

test('addN: orientation mismatch rejected', () => {
  // mu is row vector (T tag); Z atom-batched columns. Dispatcher should
  // refuse to broadcast across opposite orientations.
  const N = 4;
  const muRow = transpose(vector([10, 20]));
  const Z = batchedVector(new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]), 2);
  assert.throws(
    () => valueOps.addN(Z, muRow, N),
    /opposite orientation/);
});

test('addN: delegates to atom-indep add when no atom axis', () => {
  const N = 1024;
  const a = vector([1, 2, 3]);
  const b = vector([10, 20, 30]);
  const r = valueOps.addN(a, b, N);
  assert.deepEqual(Array.from(r.data), [11, 22, 33]);
});

// =====================================================================
// subN
// =====================================================================

test('subN: shape=[N, k] - shape=[k] (non-commutative)', () => {
  const N = 4;
  const Z = batchedVector(new Float64Array([10, 20, 30, 40, 50, 60, 70, 80]), 2);
  const mu = vector([1, 2]);
  const r = valueOps.subN(Z, mu, N);
  assert.deepEqual(Array.from(r.data), [9, 18, 29, 38, 49, 58, 69, 78]);
});

test('subN: shape=[k] - shape=[N, k] (argument order matters)', () => {
  const N = 4;
  const mu = vector([1, 2]);
  const Z = batchedVector(new Float64Array([10, 20, 30, 40, 50, 60, 70, 80]), 2);
  const r = valueOps.subN(mu, Z, N);
  assert.deepEqual(Array.from(r.data), [-9, -18, -29, -38, -49, -58, -69, -78]);
});

// =====================================================================
// negN: pointwise at any rank
// =====================================================================

test('negN: scalar', () => {
  assert.equal(valueOps.negN(scalar(5), 10).data[0], -5);
});

test('negN: vector / atom-indep', () => {
  const r = valueOps.negN(vector([1, -2, 3]), 4);
  assert.deepEqual(Array.from(r.data), [-1, 2, -3]);
});

test('negN: atom-batched shape=[N, k]', () => {
  const N = 4;
  const Z = batchedVector(new Float64Array([1, -2, -3, 4, 5, -6, -7, 8]), 2);
  const r = valueOps.negN(Z, N);
  assert.deepEqual(r.shape, [N, 2]);
  assert.deepEqual(Array.from(r.data), [-1, 2, 3, -4, -5, 6, 7, -8]);
});

// =====================================================================
// End-to-end pipeline: `mu + L * z` through ARITH_OPS_N dispatch
// =====================================================================

const { ARITH_OPS_N } = sampler._internal;

test('pipeline: mu + L * z end-to-end (MvNormal sample pattern)', () => {
  // mu shape=[2]; L shape=[2,2] lower triangular;
  // z is per-atom shape=[N, 2]. Result: shape=[N, 2] per-atom samples.
  const N = 4;
  const mu = vector([100, 200]);
  // L = [[2, 0], [1, 3]] — Cholesky of some 2x2 covariance.
  const L = matrix([2, 0, 1, 3], 2, 2);
  const z = batchedVector(new Float64Array([
    1, 0,  0, 1,  1, 1,  2, -1,
  ]), 2);

  // Step 1: Lz. shape=[4, 2].
  // L @ atoms:
  //   [1, 0] → [2, 1]
  //   [0, 1] → [0, 3]
  //   [1, 1] → [2, 4]
  //   [2, -1] → [4, -1]
  const Lz = ARITH_OPS_N.mul([L, z], N);
  assert.deepEqual(Lz.shape, [N, 2]);
  assert.deepEqual(Array.from(Lz.data), [2, 1, 0, 3, 2, 4, 4, -1]);

  // Step 2: mu + Lz. shape=[4, 2]. Broadcast mu over atoms.
  const sample = ARITH_OPS_N.add([mu, Lz], N);
  assert.deepEqual(sample.shape, [N, 2]);
  assert.deepEqual(Array.from(sample.data),
    [102, 201, 100, 203, 102, 204, 104, 199]);
});

test('pipeline: atom-indep batched scalar still on scalar path', () => {
  // Make sure shape=[N] inputs continue going through broadcast2.
  const N = 4;
  const a = new Float64Array([1, 2, 3, 4]);
  const r = ARITH_OPS_N.add([a, 10], N);
  // Phase-1 fast path: bare Float64Array result.
  assert.ok(r instanceof Float64Array);
  assert.deepEqual(Array.from(r), [11, 12, 13, 14]);
});
