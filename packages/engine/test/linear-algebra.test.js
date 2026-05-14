'use strict';

// Spec §07 Linear algebra: transpose, adjoint, trace, diagmat,
// self_outer, det, logabsdet, inv, linsolve, lower_cholesky,
// row_gram, col_gram. All operate on nested JS arrays (matrices) and
// flat arrays (vectors); textbook algorithms via sampler.evaluateExpr.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler');

function lit(v)        { return { kind: 'lit', value: v }; }
function vec(...vs)    { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function mat(...rows)  { return { kind: 'call', op: 'vector', args: rows.map(r => vec(...r)) }; }
function call(op, ...args) { return { kind: 'call', op, args }; }
const ev = (ir) => sampler.evaluateExpr(ir, {});

function matClose(A, B, tol) {
  tol = tol == null ? 1e-12 : tol;
  if (A.length !== B.length) return false;
  for (let i = 0; i < A.length; i++) {
    if (A[i].length !== B[i].length) return false;
    for (let j = 0; j < A[i].length; j++) {
      if (!(Math.abs(A[i][j] - B[i][j]) <= tol)) return false;
    }
  }
  return true;
}

// =====================================================================
// transpose / adjoint / trace
// =====================================================================

test('transpose: row/col swap', () => {
  assert.deepEqual(
    ev(call('transpose', mat([1, 2, 3], [4, 5, 6]))),
    [[1, 4], [2, 5], [3, 6]]);
});

test('adjoint of a real matrix ≡ transpose', () => {
  const A = mat([1, 2], [3, 4]);
  assert.deepEqual(ev(call('adjoint', A)), ev(call('transpose', A)));
});

test('trace: sum of diagonal entries', () => {
  assert.equal(ev(call('trace', mat([1, 0, 0], [0, 2, 0], [0, 0, 3]))), 6);
  assert.equal(ev(call('trace', mat([5, 99], [99, -5]))), 0);
});

test('trace: rejects non-square matrix', () => {
  assert.throws(() => ev(call('trace', mat([1, 2, 3], [4, 5, 6]))),
    /square/);
});

// =====================================================================
// diagmat / self_outer
// =====================================================================

test('diagmat: vector → diagonal matrix', () => {
  assert.deepEqual(ev(call('diagmat', vec(1, 2, 3))),
    [[1, 0, 0], [0, 2, 0], [0, 0, 3]]);
});

test('self_outer: v · vᵀ', () => {
  // [1, 2, 3] outer [1, 2, 3] = [[1, 2, 3], [2, 4, 6], [3, 6, 9]]
  assert.deepEqual(ev(call('self_outer', vec(1, 2, 3))),
    [[1, 2, 3], [2, 4, 6], [3, 6, 9]]);
});

// =====================================================================
// det / logabsdet
// =====================================================================

test('det of 2×2: ad − bc', () => {
  // det([[4, 2], [1, 3]]) = 12 − 2 = 10
  assert.equal(ev(call('det', mat([4, 2], [1, 3]))), 10);
});

test('det of identity = 1', () => {
  assert.equal(ev(call('det', mat([1, 0, 0], [0, 1, 0], [0, 0, 1]))), 1);
});

test('det of singular matrix = 0', () => {
  // Rows [1,2] and [2,4] are linearly dependent.
  assert.equal(ev(call('det', mat([1, 2], [2, 4]))), 0);
});

test('logabsdet: matches log|det|', () => {
  // det = 10, log|10| ≈ 2.303
  assert.ok(Math.abs(ev(call('logabsdet', mat([4, 2], [1, 3]))) - Math.log(10)) < 1e-12);
});

test('logabsdet: singular matrix ⇒ -Infinity', () => {
  assert.equal(ev(call('logabsdet', mat([1, 2], [2, 4]))), -Infinity);
});

// =====================================================================
// inv / linsolve
// =====================================================================

test('inv: A · inv(A) ≈ I', () => {
  // A = [[4, 2], [1, 3]] → inv = [[0.3, -0.2], [-0.1, 0.4]]
  const invA = ev(call('inv', mat([4, 2], [1, 3])));
  assert.ok(matClose(invA, [[0.3, -0.2], [-0.1, 0.4]]));
});

test('linsolve: A · x = b — verified by checking A·x = b', () => {
  // [[4, 2], [1, 3]] · x = [10, 11] → x = [0.8, 3.4]
  const x = ev(call('linsolve', mat([4, 2], [1, 3]), vec(10, 11)));
  // Verify: 4·0.8 + 2·3.4 = 3.2 + 6.8 = 10 ✓; 0.8 + 3·3.4 = 0.8 + 10.2 = 11 ✓
  assert.ok(Math.abs(x[0] - 0.8) < 1e-12);
  assert.ok(Math.abs(x[1] - 3.4) < 1e-12);
});

test('linsolve: singular matrix ⇒ runtime error', () => {
  assert.throws(
    () => ev(call('linsolve', mat([1, 2], [2, 4]), vec(1, 2))),
    /singular/);
});

// =====================================================================
// lower_cholesky
// =====================================================================

test('lower_cholesky: L · Lᵀ = A for a 2×2 SPD matrix', () => {
  // A = [[4, 2], [2, 3]] → L = [[2, 0], [1, √2]]
  const L = ev(call('lower_cholesky', mat([4, 2], [2, 3])));
  assert.ok(Math.abs(L[0][0] - 2) < 1e-12);
  assert.ok(Math.abs(L[0][1] - 0) < 1e-12);
  assert.ok(Math.abs(L[1][0] - 1) < 1e-12);
  assert.ok(Math.abs(L[1][1] - Math.SQRT2) < 1e-12);
});

test('lower_cholesky: not positive definite ⇒ runtime error', () => {
  // Diagonal has a negative entry — definitely not PD.
  assert.throws(
    () => ev(call('lower_cholesky', mat([1, 0], [0, -1]))),
    /positive definite/);
});

// =====================================================================
// row_gram / col_gram
// =====================================================================

test('row_gram(A) = A · Aᵀ', () => {
  // A = [[1, 2], [3, 4]]; A·Aᵀ = [[5, 11], [11, 25]]
  assert.deepEqual(ev(call('row_gram', mat([1, 2], [3, 4]))),
    [[5, 11], [11, 25]]);
});

test('col_gram(A) = Aᵀ · A', () => {
  // A = [[1, 2], [3, 4]]; Aᵀ·A = [[10, 14], [14, 20]]
  assert.deepEqual(ev(call('col_gram', mat([1, 2], [3, 4]))),
    [[10, 14], [14, 20]]);
});
