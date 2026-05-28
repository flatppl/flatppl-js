'use strict';

// Tests for value-ops.mul — shape-dispatched, tag-aware multiplication.
// Phase 2b of the shape-explicit refactor (TODO-flatppl-js.md).
//
// Coverage:
//
//   1. scalar × scalar → scalar
//   2. scalar × vector / matrix → broadcast (tag preserved)
//   3. matrix × matrix → matmul (all four Klein-4 tag combinations
//      on the LHS/RHS swap bit; conjugate bit is observationally a
//      no-op for real values but is plumbed through composition).
//   4. matrix × vector → matvec (matrix tagged N/T; column vector
//      input only).
//   5. vector × matrix → vec-mat (row vector tagged T on the left).
//   6. transpose(vector) × vector → scalar (inner product).
//   7. vector × transpose(vector) → matrix (outer product).
//   8. Error cases:
//      - vector × vector (both N or both T)
//      - matrix × row-vector
//      - column-vector × matrix
//      - dimension mismatches
//   9. Dispatch through ARITH_OPS.mul (the scalar IR primitive)
//      and through ARITH_OPS_N.mul (the batched dispatcher).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const valueLib = require('..').value;
const valueOps = require('../value-ops.ts');
const sampler = require('../sampler.ts');

const { scalar, vector, matrix, batchedScalar,
        transpose, adjoint, getTag } = valueLib;

// =====================================================================
// scalar × anything
// =====================================================================

test('mul: scalar × scalar → scalar', () => {
  const r = valueOps.mul(scalar(2), scalar(3));
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], 6);
});

test('mul: scalar × vector → vector (tag preserved)', () => {
  const v = vector([1, 2, 3]);
  const r = valueOps.mul(scalar(10), v);
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [10, 20, 30]);
  assert.equal(getTag(r), 'N');
});

test('mul: scalar × transpose(vector) → row vector (tag T preserved)', () => {
  const rv = transpose(vector([1, 2, 3]));
  const r = valueOps.mul(scalar(2), rv);
  assert.deepEqual(r.shape, [3]);
  assert.equal(getTag(r), 'T');
  assert.deepEqual(Array.from(r.data), [2, 4, 6]);
});

test('mul: scalar × matrix → matrix (tag preserved)', () => {
  const M = matrix([1, 2, 3, 4], 2, 2);
  const r = valueOps.mul(scalar(5), M);
  assert.deepEqual(r.shape, [2, 2]);
  assert.deepEqual(Array.from(r.data), [5, 10, 15, 20]);
});

test('mul: scalar × transpose(matrix) → keeps transpose tag', () => {
  const Mt = transpose(matrix([1, 2, 3, 4, 5, 6], 2, 3));
  // Mt has shape=[3, 2], tag='T', data still the [2,3] layout.
  const r = valueOps.mul(scalar(2), Mt);
  assert.deepEqual(r.shape, [3, 2]);
  assert.equal(getTag(r), 'T');
});

test('mul: vector × scalar (symmetric) → vector', () => {
  const r = valueOps.mul(vector([1, 2, 3]), scalar(4));
  assert.deepEqual(Array.from(r.data), [4, 8, 12]);
});

// =====================================================================
// vector × vector (inner / outer / error)
// =====================================================================

test('mul: transpose(v1) × v2 → scalar (inner product)', () => {
  const u = transpose(vector([1, 2, 3]));  // row
  const v = vector([4, 5, 6]);              // column
  const r = valueOps.mul(u, v);
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], 1 * 4 + 2 * 5 + 3 * 6);  // 32
});

test('mul: v1 × transpose(v2) → matrix (outer product)', () => {
  const u = vector([1, 2, 3]);              // column
  const v = transpose(vector([10, 20]));    // row
  const r = valueOps.mul(u, v);
  assert.deepEqual(r.shape, [3, 2]);
  // Expected:
  //   1*10  1*20
  //   2*10  2*20
  //   3*10  3*20
  assert.deepEqual(Array.from(r.data), [10, 20, 20, 40, 30, 60]);
});

test('mul: column × column rejected with clear message', () => {
  assert.throws(
    () => valueOps.mul(vector([1, 2]), vector([3, 4])),
    /vector \* vector is not defined/);
});

test('mul: row × row rejected', () => {
  const a = transpose(vector([1, 2]));
  const b = transpose(vector([3, 4]));
  assert.throws(
    () => valueOps.mul(a, b),
    /transpose\(v1\) \* transpose\(v2\) is not defined/);
});

test('mul: inner-product length mismatch rejected', () => {
  assert.throws(
    () => valueOps.mul(transpose(vector([1, 2, 3])), vector([4, 5])),
    /inner-product vector length mismatch/);
});

// =====================================================================
// matrix × vector (matvec)
// =====================================================================

test('mul: matrix(m,n) × vector(n) → vector(m)', () => {
  // M = [[1,2,3],[4,5,6]] (2×3); v = [10,20,30] → [140, 320]
  const M = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  const v = vector([10, 20, 30]);
  const r = valueOps.mul(M, v);
  assert.deepEqual(r.shape, [2]);
  assert.deepEqual(Array.from(r.data), [140, 320]);
});

test('mul: transpose(matrix) × vector → matvec on the transposed view', () => {
  // M = [[1,2,3],[4,5,6]] (2×3); M^T (3×2); M^T @ [7, 8] → [1*7+4*8, 2*7+5*8, 3*7+6*8] = [39, 54, 69]
  const Mt = transpose(matrix([1, 2, 3, 4, 5, 6], 2, 3));
  const r = valueOps.mul(Mt, vector([7, 8]));
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [39, 54, 69]);
});

test('mul: matrix × (transposed/row vector) is rejected', () => {
  const M = matrix([1, 2, 3, 4], 2, 2);
  const rv = transpose(vector([1, 2]));
  assert.throws(
    () => valueOps.mul(M, rv),
    /matrix \* \(transposed.*vector\) is not defined/);
});

test('mul: matrix × vector dim mismatch rejected', () => {
  const M = matrix([1, 2, 3, 4], 2, 2);
  const v = vector([1, 2, 3]);
  assert.throws(
    () => valueOps.mul(M, v),
    /dimension mismatch/);
});

// =====================================================================
// vector × matrix (row vector × matrix → row vector)
// =====================================================================

test('mul: transpose(vector) × matrix → row vector', () => {
  // [1, 2] @ [[1,2,3],[4,5,6]] = [1+8, 2+10, 3+12] = [9, 12, 15]
  const u = transpose(vector([1, 2]));
  const M = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  const r = valueOps.mul(u, M);
  assert.deepEqual(r.shape, [3]);
  assert.equal(getTag(r), 'T');
  assert.deepEqual(Array.from(r.data), [9, 12, 15]);
});

test('mul: column vector × matrix rejected', () => {
  const u = vector([1, 2]);
  const M = matrix([1, 2, 3, 4], 2, 2);
  assert.throws(
    () => valueOps.mul(u, M),
    /\(column vector\) \* matrix is not defined/);
});

// =====================================================================
// matrix × matrix (all four Klein-4 transpose-bit combinations)
// =====================================================================

test('mul: matrix × matrix (both N) — basic matmul', () => {
  // A = [[1,2],[3,4]]; B = [[5,6],[7,8]]; A@B = [[19,22],[43,50]]
  const A = matrix([1, 2, 3, 4], 2, 2);
  const B = matrix([5, 6, 7, 8], 2, 2);
  const r = valueOps.mul(A, B);
  assert.deepEqual(r.shape, [2, 2]);
  assert.deepEqual(Array.from(r.data), [19, 22, 43, 50]);
});

test('mul: transpose(A) × B — aSwap path', () => {
  // A = [[1,2],[3,4]]; A^T = [[1,3],[2,4]]
  // A^T @ [[5,6],[7,8]] = [[1*5+3*7, 1*6+3*8],[2*5+4*7, 2*6+4*8]]
  //                     = [[26,30],[38,44]]
  const A = matrix([1, 2, 3, 4], 2, 2);
  const B = matrix([5, 6, 7, 8], 2, 2);
  const r = valueOps.mul(transpose(A), B);
  assert.deepEqual(r.shape, [2, 2]);
  assert.deepEqual(Array.from(r.data), [26, 30, 38, 44]);
});

test('mul: A × transpose(B) — bSwap path', () => {
  // B^T = [[5,7],[6,8]]; A @ B^T = [[1*5+2*6, 1*7+2*8],[3*5+4*6, 3*7+4*8]]
  //                              = [[17,23],[39,53]]
  const A = matrix([1, 2, 3, 4], 2, 2);
  const B = matrix([5, 6, 7, 8], 2, 2);
  const r = valueOps.mul(A, transpose(B));
  assert.deepEqual(Array.from(r.data), [17, 23, 39, 53]);
});

test('mul: transpose(A) × transpose(B) — both swap', () => {
  // A^T @ B^T = (B @ A)^T. B @ A = [[5*1+6*3, 5*2+6*4],[7*1+8*3, 7*2+8*4]]
  //                              = [[23,34],[31,46]]
  // → (B @ A)^T = [[23,31],[34,46]]
  const A = matrix([1, 2, 3, 4], 2, 2);
  const B = matrix([5, 6, 7, 8], 2, 2);
  const r = valueOps.mul(transpose(A), transpose(B));
  assert.deepEqual(Array.from(r.data), [23, 31, 34, 46]);
});

test('mul: matrix × matrix dim mismatch', () => {
  const A = matrix([1, 2, 3, 4], 2, 2);   // [2,2]
  const B = matrix([1, 2, 3, 4, 5, 6], 2, 3);  // [2,3]
  // A(2,2) × B(2,3) — inner dim 2 matches; OK.
  assert.doesNotThrow(() => valueOps.mul(A, B));
  // B(2,3) × A(2,2) — inner dim 3 ≠ 2.
  assert.throws(
    () => valueOps.mul(B, A),
    /dimension mismatch/);
});

test('mul: rectangular matmul (m≠n≠p)', () => {
  // A is (2,3), B is (3,4), result (2,4).
  const A = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  const B = matrix([1, 2, 3, 4,
                    5, 6, 7, 8,
                    9, 10, 11, 12], 3, 4);
  const r = valueOps.mul(A, B);
  assert.deepEqual(r.shape, [2, 4]);
  // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
  //       = [38, 44, 50, 56]
  // Row 1: [4+25+54, 8+30+60, 12+35+66, 16+40+72] = [83, 98, 113, 128]
  assert.deepEqual(Array.from(r.data), [38, 44, 50, 56, 83, 98, 113, 128]);
});

test('mul: adjoint(matrix) of real matrix behaves like transpose', () => {
  // For real values, A^* (adjoint) = A^T. Tag is 'A' but observationally
  // the data reads identically. Verify the result matches transpose case.
  const A = matrix([1, 2, 3, 4], 2, 2);
  const B = matrix([5, 6, 7, 8], 2, 2);
  const r1 = valueOps.mul(adjoint(A), B);
  const r2 = valueOps.mul(transpose(A), B);
  assert.deepEqual(Array.from(r1.data), Array.from(r2.data));
});

// =====================================================================
// Dispatch through ARITH_OPS.mul and ARITH_OPS_N.mul
// =====================================================================

const { ARITH_OPS, ARITH_OPS_N } = sampler._internal;

test('ARITH_OPS.mul: bare scalars still on the JS fast path', () => {
  assert.equal(ARITH_OPS.mul(2, 3), 6);
  assert.equal(typeof ARITH_OPS.mul(2, 3), 'number');
});

test('ARITH_OPS.mul: Value scalar × Value scalar → rank-0 Value', () => {
  // engine-concepts §20 / TODO Phase 1: when either input is a Value,
  // the dispatcher routes through value-ops.mul unconditionally —
  // rank-0 × rank-0 produces a rank-0 Value (no bare-number unwrap).
  // Callers that need a JS scalar use valueLib.asScalar(...).
  const r = ARITH_OPS.mul(scalar(4), scalar(5));
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], 20);
});

test('ARITH_OPS.mul: shape-rich Value routes to value-ops', () => {
  const A = matrix([1, 2, 3, 4], 2, 2);
  const v = vector([1, 1]);
  const r = ARITH_OPS.mul(A, v);
  assert.deepEqual(r.shape, [2]);
  // [[1,2],[3,4]] @ [1,1] = [3, 7]
  assert.deepEqual(Array.from(r.data), [3, 7]);
});

test('ARITH_OPS_N.mul: shape-rich Value bypasses broadcast2 → matmul', () => {
  // The IR-level batched dispatcher: when N=4 and a Value has shape
  // [2,2] (intrinsic matrix, not atom-batched), route directly to
  // value-ops.mul. The batched-scalar broadcast2 path is bypassed.
  const A = matrix([1, 0, 0, 1], 2, 2);  // identity
  const v = vector([7, 8]);
  const r = ARITH_OPS_N.mul([A, v], 4);
  assert.deepEqual(r.shape, [2]);
  assert.deepEqual(Array.from(r.data), [7, 8]);
});

test('ARITH_OPS_N.mul: bare scalars (no Value) still use broadcast2', () => {
  // Regression guard: the legacy scalar path must not regress when
  // there are no Value inputs.
  const N = 4;
  const a = new Float64Array([1, 2, 3, 4]);
  const r = ARITH_OPS_N.mul([a, 2], N);
  assert.ok(r instanceof Float64Array);
  assert.deepEqual(Array.from(r), [2, 4, 6, 8]);
});
