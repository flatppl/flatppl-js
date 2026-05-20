'use strict';

// Tests for Value-aware linear-algebra ops — Phase 3 of the shape-
// explicit refactor. Every linalg op in ARITH_OPS now dispatches on
// Value input: transpose / adjoint use the free Klein-4 tag flip,
// others bridge through nested-array form to reach the textbook
// algorithms (det / inv / linsolve / Cholesky / Gauss–Jordan).
//
// Coverage:
//
//   1. transpose / adjoint on Value: O(1) tag flip, data preserved.
//   2. trace / det / logabsdet: Value scalar result.
//   3. diagmat / self_outer / inv / lower_cholesky / linsolve:
//      Value matrix / vector result.
//   4. row_gram / col_gram routed through tag-aware mul (zero transpose
//      allocation).
//   5. transpose(M) input to a non-tag-aware op (e.g. cholesky on the
//      Gram of a transposed matrix) materialises correctly via
//      _valueToNested.
//   6. Round-trip: _valueToNested + _nestedToValue.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const valueLib = require('..').value;
const valueOps = require('../value-ops.ts');
const sampler = require('../sampler.ts');

const { scalar, vector, matrix, transpose, adjoint, getTag } = valueLib;
const { ARITH_OPS } = sampler._internal;

function close(a, b, tol) {
  tol = tol == null ? 1e-12 : tol;
  return Math.abs(a - b) <= tol;
}

function dataClose(a, b, tol) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (!close(a[i], b[i], tol)) return false;
  return true;
}

// =====================================================================
// _valueToNested / _nestedToValue bridge
// =====================================================================

test('_valueToNested: matrix N-tag → nested row-major', () => {
  const v = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  assert.deepEqual(valueOps._valueToNested(v), [[1, 2, 3], [4, 5, 6]]);
});

test('_valueToNested: matrix T-tag reads via index permutation', () => {
  // Underlying data is [2, 3] row-major; transpose makes logical [3, 2].
  // Logical (i, j) = original (j, i).
  const v = transpose(matrix([1, 2, 3, 4, 5, 6], 2, 3));
  assert.deepEqual(valueOps._valueToNested(v), [[1, 4], [2, 5], [3, 6]]);
});

test('_valueToNested: vector → flat JS array', () => {
  const v = vector([1, 2, 3]);
  assert.deepEqual(valueOps._valueToNested(v), [1, 2, 3]);
});

test('_nestedToValue: round-trip with matrix', () => {
  const original = matrix([1, 2, 3, 4], 2, 2);
  const nested = valueOps._valueToNested(original);
  const recovered = valueOps._nestedToValue(nested);
  assert.deepEqual(recovered.shape, [2, 2]);
  assert.deepEqual(Array.from(recovered.data), Array.from(original.data));
});

// =====================================================================
// transpose / adjoint on Value (free tag flip)
// =====================================================================

test('ARITH_OPS.transpose on Value: returns Value with toggled tag, data shared', () => {
  const M = matrix([1, 2, 3, 4], 2, 2);
  const Mt = ARITH_OPS.transpose(M);
  assert.equal(getTag(Mt), 'T');
  assert.equal(Mt.data, M.data, 'transpose on Value must share data buffer');
  assert.deepEqual(Mt.shape, [2, 2]);  // square matrix; shape unchanged
});

test('ARITH_OPS.transpose on Value: rectangular swaps shape', () => {
  const M = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  const Mt = ARITH_OPS.transpose(M);
  assert.deepEqual(Mt.shape, [3, 2]);
});

test('ARITH_OPS.adjoint on Value: tag → A, data shared', () => {
  const M = matrix([1, 2, 3, 4], 2, 2);
  const Ma = ARITH_OPS.adjoint(M);
  assert.equal(getTag(Ma), 'A');
  assert.equal(Ma.data, M.data);
});

test('ARITH_OPS.transpose on nested array (legacy): unchanged', () => {
  const M = [[1, 2, 3], [4, 5, 6]];
  const Mt = ARITH_OPS.transpose(M);
  assert.deepEqual(Mt, [[1, 4], [2, 5], [3, 6]]);
});

// =====================================================================
// Scalar-result ops (trace / det / logabsdet)
// =====================================================================

test('ARITH_OPS.trace on Value: returns scalar Value', () => {
  const M = matrix([1, 2, 3, 4], 2, 2);
  const r = ARITH_OPS.trace(M);
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], 1 + 4);
});

test('ARITH_OPS.det on Value (2x2 ad-bc)', () => {
  const M = matrix([1, 2, 3, 4], 2, 2);  // det = 1*4 - 2*3 = -2
  const r = ARITH_OPS.det(M);
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], -2);
});

test('ARITH_OPS.logabsdet on Value', () => {
  const M = matrix([1, 0, 0, 4], 2, 2);  // det = 4, log|det| = log 4
  const r = ARITH_OPS.logabsdet(M);
  assert.equal(r.shape.length, 0);
  assert.ok(close(r.data[0], Math.log(4)));
});

// =====================================================================
// Matrix-result ops
// =====================================================================

test('ARITH_OPS.diagmat on Value vector → vector-backed diag Value', () => {
  const v = vector([1, 2, 3]);
  const D = ARITH_OPS.diagmat(v);
  assert.deepEqual(D.shape, [3, 3]);
  assert.ok(valueLib.isDiagStored(D), 'diag structure, O(m) storage');
  assert.deepEqual(Array.from(D.data), [1, 2, 3], 'stores the diagonal');
  assert.deepEqual(Array.from(valueLib.densify(D).data),
    [1, 0, 0, 0, 2, 0, 0, 0, 3]);
});

test('ARITH_OPS.self_outer on Value vector → Value matrix', () => {
  const v = vector([1, 2, 3]);
  const O = ARITH_OPS.self_outer(v);
  assert.deepEqual(O.shape, [3, 3]);
  assert.deepEqual(Array.from(O.data),
    [1, 2, 3,
     2, 4, 6,
     3, 6, 9]);
});

test('ARITH_OPS.inv on Value: A * inv(A) ≈ I', () => {
  const A = matrix([4, 7, 2, 6], 2, 2);
  const invA = ARITH_OPS.inv(A);
  assert.deepEqual(invA.shape, [2, 2]);
  // Verify A · inv(A) ≈ I via Value-aware mul.
  const I = valueOps.mul(A, invA);
  assert.ok(dataClose(I.data, [1, 0, 0, 1], 1e-12));
});

test('ARITH_OPS.lower_cholesky on Value: L · Lᵀ = A', () => {
  // A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] — classic SPD example.
  const A = matrix([4, 12, -16,
                    12, 37, -43,
                    -16, -43, 98], 3, 3);
  const L = ARITH_OPS.lower_cholesky(A);
  assert.deepEqual(L.shape, [3, 3]);
  // Reconstruct: L @ L^T (uses tag-flip transpose; verifies matmul tag path).
  const reconstructed = valueOps.mul(L, valueLib.transpose(L));
  assert.ok(dataClose(reconstructed.data, A.data, 1e-10));
});

test('ARITH_OPS.linsolve on Value: A · x = b verified', () => {
  // A x = b; solve for x then verify A @ x ≈ b.
  const A = matrix([3, 2, 1, 4], 2, 2);
  const b = vector([7, 10]);
  const x = ARITH_OPS.linsolve(A, b);
  assert.deepEqual(x.shape, [2]);
  const reconstructed = valueOps.mul(A, x);
  assert.ok(dataClose(reconstructed.data, b.data, 1e-10));
});

// =====================================================================
// Gram matrices (row_gram / col_gram via tag-aware mul)
// =====================================================================

test('ARITH_OPS.row_gram on Value: A · Aᵀ', () => {
  // A is 2x3. row_gram(A) is 2x2 = A @ A^T.
  // A = [[1, 2, 3], [4, 5, 6]]; A @ A^T:
  //   [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
  //   [4*1+5*2+6*3, 4*4+5*5+6*6] = [32, 77]
  const A = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  const G = ARITH_OPS.row_gram(A);
  assert.deepEqual(G.shape, [2, 2]);
  assert.deepEqual(Array.from(G.data), [14, 32, 32, 77]);
});

test('ARITH_OPS.col_gram on Value: Aᵀ · A', () => {
  // A is 2x3. col_gram(A) is 3x3 = A^T @ A.
  // A^T @ A column entries: (i, j) = sum_k A[k][i] * A[k][j]
  //   (0,0) = 1*1+4*4 = 17;  (0,1) = 1*2+4*5 = 22;  (0,2) = 1*3+4*6 = 27
  //   (1,1) = 2*2+5*5 = 29;  (1,2) = 2*3+5*6 = 36;  (2,2) = 3*3+6*6 = 45
  const A = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  const G = ARITH_OPS.col_gram(A);
  assert.deepEqual(G.shape, [3, 3]);
  assert.deepEqual(Array.from(G.data),
    [17, 22, 27,
     22, 29, 36,
     27, 36, 45]);
});

// =====================================================================
// Mixing Value transpose tag with linalg ops that bridge through
// nested-array form. _valueToNested materialises the tag's effect.
// =====================================================================

test('lower_cholesky(row_gram(transpose(A))) — tag flows through bridge', () => {
  // row_gram(A^T) = A^T @ A = col_gram(A), which is SPD if A has full column rank.
  // A = [[2, 1], [1, 2], [0, 1]] (3x2). col_gram is 2x2 SPD.
  const A = matrix([2, 1,
                    1, 2,
                    0, 1], 3, 2);
  const At = ARITH_OPS.transpose(A);
  const G = ARITH_OPS.row_gram(At);    // (A^T) @ (A^T)^T = A^T @ A = col_gram(A)
  // G entries: (0,0) = 4+1+0 = 5; (0,1) = 2+2+0 = 4; (1,1) = 1+4+1 = 6
  assert.deepEqual(Array.from(G.data), [5, 4, 4, 6]);
  // Now Cholesky on this. Should produce a valid lower-triangular factor.
  const L = ARITH_OPS.lower_cholesky(G);
  assert.deepEqual(L.shape, [2, 2]);
  // L @ L^T ≈ G
  const reconstructed = valueOps.mul(L, valueLib.transpose(L));
  assert.ok(dataClose(reconstructed.data, G.data, 1e-10));
});
