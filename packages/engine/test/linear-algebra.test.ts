'use strict';

// Spec §07 Linear algebra: transpose, adjoint, trace, diagmat,
// self_outer, det, logabsdet, inv, linsolve, lower_cholesky,
// row_gram, col_gram. All operate on nested JS arrays (matrices) and
// flat arrays (vectors); textbook algorithms via sampler.evaluateExpr.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const valueLib = require('../value.ts');
const { toJS } = require('./_value-helpers.ts');

function lit(v: any)        { return { kind: 'lit', value: v }; }
function vec(...vs: any[])    { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
// Per spec §03, a literal nested-vector `[[…],[…]]` is a vector-of-
// vectors, NOT a matrix — `rowstack` is the explicit lift to rank-2.
// Use `rowstack(vector(vector(…), vector(…)))` so the matrix ops here
// see a proper matrix Value.
function mat(...rows: any[]) {
  const vov = { kind: 'call', op: 'vector', args: rows.map(r => vec(...r)) };
  return { kind: 'call', op: 'rowstack', args: [vov] };
}
function call(op: any, ...args: any[]) { return { kind: 'call', op, args }; }
const evRaw = (ir: any) => sampler.evaluateExpr(ir, {});
const ev = (ir: any) => toJS(evRaw(ir));

function matClose(A: any, B: any, tol?: any) {
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

test('diagmat: vector → vector-backed diagonal structure', () => {
  const D = evRaw(call('diagmat', vec(1, 2, 3)));
  assert.ok(valueLib.isDiagStored(D), 'diagmat yields a diag Value');
  assert.deepEqual(Array.from(D.data), [1, 2, 3], 'stores the diagonal');
  assert.deepEqual(Array.from(valueLib.densify(D).data),
    [1, 0, 0, 0, 2, 0, 0, 0, 3]);
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

// =====================================================================
// cross (3-D vector cross product, spec §07)
// =====================================================================

test('cross: canonical basis vectors — e1 × e2 = e3', () => {
  assert.deepEqual(ev(call('cross', vec(1, 0, 0), vec(0, 1, 0))), [0, 0, 1]);
  assert.deepEqual(ev(call('cross', vec(0, 1, 0), vec(0, 0, 1))), [1, 0, 0]);
  assert.deepEqual(ev(call('cross', vec(0, 0, 1), vec(1, 0, 0))), [0, 1, 0]);
});

test('cross: antisymmetry — b × a = −(a × b)', () => {
  const ab = ev(call('cross', vec(1, 2, 3), vec(4, 5, 6)));
  const ba = ev(call('cross', vec(4, 5, 6), vec(1, 2, 3)));
  for (let k = 0; k < 3; k++) assert.equal(ba[k], -ab[k]);
  // Spec formula: cross([1,2,3], [4,5,6]) = [2·6−3·5, 3·4−1·6, 1·5−2·4]
  //                                       = [-3, 6, -3]
  assert.deepEqual(ab, [-3, 6, -3]);
});

test('cross: parallel vectors → zero', () => {
  // a × (λ·a) = 0 for any λ.
  assert.deepEqual(ev(call('cross', vec(1, 2, 3), vec(2, 4, 6))), [0, 0, 0]);
});

test('cross: result is orthogonal to both inputs', () => {
  const a = vec(1, 2, 3), b = vec(4, 5, 6);
  const c = ev(call('cross', a, b));
  function dot(x: number[], y: number[]) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
  }
  assert.equal(dot(c, [1, 2, 3]), 0);
  assert.equal(dot(c, [4, 5, 6]), 0);
});

test('cross: rejects non-length-3 inputs', () => {
  assert.throws(() => ev(call('cross', vec(1, 2), vec(0, 1, 0))),
    /length-3/);
  assert.throws(() => ev(call('cross', vec(0, 1, 0), vec(1, 2, 3, 4))),
    /length-3/);
});

test('cross: complex inputs — bilinear over ℂ (no conjugation)', () => {
  // i·e1 × e2 = i·e3, NOT −i·e3 (would be the Hermitian variant).
  // Build complex vectors directly via the complex() constructor.
  const a = call('vector',
    call('complex', lit(0), lit(1)),  // i
    call('complex', lit(0), lit(0)),
    call('complex', lit(0), lit(0)));
  const b = call('vector',
    call('complex', lit(0), lit(0)),
    call('complex', lit(1), lit(0)),
    call('complex', lit(0), lit(0)));
  const out = evRaw(call('cross', a, b));
  // Expected result: components (0, 0, i). With bilinearity:
  //   (i·e1) × (e2) = i·(e1 × e2) = i·e3 ⇒ re=[0,0,0], im=[0,0,1].
  assert.ok(valueLib.isComplexValue(out));
  assert.deepEqual(out.shape, [3]);
  assert.deepEqual(Array.from(out.data), [0, 0, 0]);
  assert.deepEqual(Array.from(out.im),   [0, 0, 1]);
});

test('cross: complex bilinearity — cross(α·a, β·b) = αβ·cross(a, b)', () => {
  // Pick α = 1+i, β = 1−i  ⇒  αβ = 2 (real). Cross product
  // direction unchanged; magnitudes scale by 2.
  function cVec(reIm: Array<[number, number]>) {
    return call('vector',
      ...reIm.map(([r, i]) => call('complex', lit(r), lit(i))));
  }
  const a = cVec([[1, 1], [2, 2], [3, 3]]);          // (1+i) · [1,2,3]
  const b = cVec([[4, -4], [5, -5], [6, -6]]);       // (1−i) · [4,5,6]
  const out = evRaw(call('cross', a, b));
  // 2 · cross([1,2,3], [4,5,6]) = 2·[−3, 6, −3] = [−6, 12, −6].
  assert.deepEqual(Array.from(out.data), [-6, 12, -6]);
  assert.deepEqual(Array.from(out.im),   [0, 0, 0]);
});
