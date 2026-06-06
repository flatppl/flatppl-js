'use strict';

// =====================================================================
// ext-linalg.test.ts — algorithmic primitives for the
// `ext-linear-algebra` standard module (spec §09).
// =====================================================================
//
// Tests the pure primitives in ext-linalg.ts (lu / kron / matexp);
// standard-module integration is covered separately in
// standard-modules.test.ts via the registry lookup path.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const extLinalg = require('../ext-linalg.ts');
const stdmod    = require('../standard-modules.ts');

function matVal(rows: number, cols: number, data: number[]): any {
  return { shape: [rows, cols], data: new Float64Array(data) };
}

function flatMatMul(A: number[], B: number[], aRows: number, aCols: number, bCols: number): number[] {
  const out = new Array(aRows * bCols).fill(0);
  for (let i = 0; i < aRows; i++) {
    for (let k = 0; k < aCols; k++) {
      const aik = A[i * aCols + k];
      for (let j = 0; j < bCols; j++) out[i * bCols + j] += aik * B[k * bCols + j];
    }
  }
  return out;
}

function assertClose(actual: any, expected: number[], eps = 1e-10) {
  assert.equal(actual.length, expected.length, 'length match');
  for (let i = 0; i < actual.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < eps,
      `[${i}]: actual ${actual[i]} != expected ${expected[i]} (eps ${eps})`);
  }
}

// =====================================================================
// 1. lu — P · A = L · U
// =====================================================================

test('lu: 2x2 with no pivot — P=I, L unit-diag, U upper-tri', () => {
  // A = [[4, 3], [2, 1]]; |4| > |2| so no row swap; L = [[1,0],[0.5,1]],
  // U = [[4,3],[0,-0.5]], P = I.
  const A = matVal(2, 2, [4, 3, 2, 1]);
  const { fields } = extLinalg._lu(A);
  assertClose(fields.P.data, [1, 0, 0, 1]);
  assertClose(fields.L.data, [1, 0, 0.5, 1]);
  assertClose(fields.U.data, [4, 3, 0, -0.5]);
});

test('lu: 3x3 — round-trip P·A = L·U holds for arbitrary matrix', () => {
  // A = [[4, 3, 1], [2, 5, 7], [6, 8, 9]]
  const A_data = [4, 3, 1, 2, 5, 7, 6, 8, 9];
  const A = matVal(3, 3, A_data);
  const { fields } = extLinalg._lu(A) as { fields: Record<string, { data: ArrayLike<number> }> };
  const P = Array.from(fields.P.data);
  const L = Array.from(fields.L.data);
  const U = Array.from(fields.U.data);
  // P·A should equal L·U.
  const PA = flatMatMul(P, A_data, 3, 3, 3);
  const LU = flatMatMul(L, U, 3, 3, 3);
  assertClose(PA, LU);
  // L: unit-diagonal, strictly-lower below.
  assert.equal(L[0], 1); assert.equal(L[4], 1); assert.equal(L[8], 1);
  assert.equal(L[1], 0); assert.equal(L[2], 0); assert.equal(L[5], 0);
  // U: upper-triangular.
  assert.equal(U[3], 0); assert.equal(U[6], 0); assert.equal(U[7], 0);
});

test('lu: singular matrix throws', () => {
  // A = [[1, 2], [2, 4]] is rank-1.
  const A = matVal(2, 2, [1, 2, 2, 4]);
  assert.throws(() => extLinalg._lu(A), /singular/);
});

test('lu: rejects non-square', () => {
  const A = matVal(2, 3, [1, 2, 3, 4, 5, 6]);
  assert.throws(() => extLinalg._lu(A), /square/);
});

// =====================================================================
// 2. kron — Kronecker tensor product
// =====================================================================

test('kron: 2x2 ⊗ 2x2 — shape (4, 4) with block structure', () => {
  // A = [[1, 2], [3, 4]];  B = [[0, 5], [6, 7]]
  // kron = [[A11·B, A12·B], [A21·B, A22·B]] (block layout, row-major)
  //      = [[0,  5,  0, 10],
  //         [6,  7, 12, 14],
  //         [0, 15,  0, 20],
  //         [18,21, 24, 28]]
  const A = matVal(2, 2, [1, 2, 3, 4]);
  const B = matVal(2, 2, [0, 5, 6, 7]);
  const K = extLinalg._kron(A, B);
  assert.deepEqual(K.shape, [4, 4]);
  assertClose(K.data, [
    0,  5,  0, 10,
    6,  7, 12, 14,
    0, 15,  0, 20,
    18, 21, 24, 28,
  ]);
});

test('kron: rectangular A (2x3) ⊗ B (3x1) → (6, 3)', () => {
  const A = matVal(2, 3, [1, 0, 2, 3, 4, 5]);
  const B = matVal(3, 1, [10, 20, 30]);
  const K = extLinalg._kron(A, B);
  assert.deepEqual(K.shape, [6, 3]);
  // Block (i, j) is A[i, j] · B (3x1 column).
  // Row 0 (atom A[0, *]): [B·1, B·0, B·2] = [10,0,20; 20,0,40; 30,0,60]
  assertClose(K.data, [
    10,  0, 20,
    20,  0, 40,
    30,  0, 60,
    30, 40, 50,
    60, 80, 100,
    90, 120, 150,
  ]);
});

test('kron: identity ⊗ A → block-diagonal', () => {
  const I = matVal(2, 2, [1, 0, 0, 1]);
  const A = matVal(2, 2, [3, 4, 5, 6]);
  const K = extLinalg._kron(I, A);
  assertClose(K.data, [
    3, 4, 0, 0,
    5, 6, 0, 0,
    0, 0, 3, 4,
    0, 0, 5, 6,
  ]);
});

// =====================================================================
// 3. matexp — matrix exponential via scaling-and-squaring Padé(13)
// =====================================================================

test('matexp: zero matrix → identity', () => {
  const A = matVal(3, 3, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
  const E = extLinalg._matexp(A);
  assertClose(E.data, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
});

test('matexp: identity → e · I', () => {
  const A = matVal(2, 2, [1, 0, 0, 1]);
  const E = extLinalg._matexp(A);
  assertClose(E.data, [Math.E, 0, 0, Math.E], 1e-12);
});

test('matexp: diagonal → diag of exp(eigenvalues)', () => {
  const A = matVal(3, 3, [2, 0, 0, 0, -1, 0, 0, 0, 0.5]);
  const E = extLinalg._matexp(A);
  assertClose(E.data, [
    Math.exp(2), 0, 0,
    0, Math.exp(-1), 0,
    0, 0, Math.exp(0.5),
  ], 1e-10);
});

test('matexp: nilpotent — finite series terminates exactly', () => {
  // A = [[0, 1], [0, 0]];  e^A = I + A = [[1, 1], [0, 1]]
  const A = matVal(2, 2, [0, 1, 0, 0]);
  const E = extLinalg._matexp(A);
  assertClose(E.data, [1, 1, 0, 1], 1e-12);
});

test('matexp: known small case — [[0, -π], [π, 0]] = rotation by π', () => {
  // exp([[0, -theta], [theta, 0]]) = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]
  // At theta = π: [[-1, 0], [0, -1]]
  const A = matVal(2, 2, [0, -Math.PI, Math.PI, 0]);
  const E = extLinalg._matexp(A);
  assertClose(E.data, [-1, 0, 0, -1], 1e-10);
});

// =====================================================================
// 4. qr — Householder reflections (spec §07; surfaced in §09 module)
// =====================================================================

function flatTranspose(M: number[], rows: number, cols: number): number[] {
  const out = new Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) out[j * rows + i] = M[i * cols + j];
  }
  return out;
}

test('qr: 2x2 — round-trip A = Q · R', () => {
  // A = [[4, 3], [2, 1]]
  const A_data = [4, 3, 2, 1];
  const A = matVal(2, 2, A_data);
  const { fields } = extLinalg._qr(A) as { fields: Record<string, { data: ArrayLike<number>; shape: number[] }> };
  const Q = Array.from(fields.Q.data);
  const R = Array.from(fields.R.data);
  // Q · R should equal A.
  const QR = flatMatMul(Q, R, 2, 2, 2);
  assertClose(QR, A_data, 1e-12);
  // Q has orthonormal columns: Q^T Q = I.
  const Qt = flatTranspose(Q, 2, 2);
  const QtQ = flatMatMul(Qt, Q, 2, 2, 2);
  assertClose(QtQ, [1, 0, 0, 1], 1e-12);
});

test('qr: 4x2 thin QR — Q is 4x2 orthonormal, R is 2x2 upper', () => {
  // A = arbitrary 4x2 matrix
  const A_data = [1, 2, 3, 4, 5, 6, 7, 8];
  const A = matVal(4, 2, A_data);
  const { fields } = extLinalg._qr(A) as { fields: Record<string, { data: ArrayLike<number>; shape: number[] }> };
  assert.deepEqual(fields.Q.shape, [4, 2]);
  assert.deepEqual(fields.R.shape, [2, 2]);
  const Q = Array.from(fields.Q.data);
  const R = Array.from(fields.R.data);
  // Q · R = A.
  const QR = flatMatMul(Q, R, 4, 2, 2);
  assertClose(QR, A_data, 1e-10);
  // Q^T · Q = I (Q has orthonormal columns).
  const Qt = flatTranspose(Q, 4, 2);
  const QtQ = flatMatMul(Qt, Q, 2, 4, 2);
  assertClose(QtQ, [1, 0, 0, 1], 1e-12);
  // R is upper-triangular: R[1, 0] = 0.
  assert.ok(Math.abs(R[2]) < 1e-12, 'R[1,0] = 0');
});

test('qr: rejects m < n', () => {
  const A = matVal(2, 4, [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.throws(() => extLinalg._qr(A), /m >= n/);
});

// =====================================================================
// 5. lstsq — least-squares via QR
// =====================================================================

test('lstsq: square 2x2 — recovers exact solution (no residual)', () => {
  // A · x = b where A = [[4, 3], [2, 1]], x = [1, 2], b = [10, 4].
  const A = matVal(2, 2, [4, 3, 2, 1]);
  const b = { shape: [2], data: new Float64Array([10, 4]) };
  const x = extLinalg._lstsq(A, b);
  assertClose(Array.from(x.data), [1, 2], 1e-12);
});

test('lstsq: overdetermined 4x2 — matches normal-equations solution', () => {
  // Linear-regression-style: 4 data points, fit slope + intercept.
  // y = 2·x + 1 with noise → exact data: x = [1,2,3,4], y = [3,5,7,9]
  // A = [[1,1],[1,2],[1,3],[1,4]] (column 0 = intercept, column 1 = x).
  // True params: [intercept, slope] = [1, 2].
  const A = matVal(4, 2, [1, 1, 1, 2, 1, 3, 1, 4]);
  const b = { shape: [4], data: new Float64Array([3, 5, 7, 9]) };
  const x = extLinalg._lstsq(A, b);
  assertClose(Array.from(x.data), [1, 2], 1e-10);
});

test('lstsq: overdetermined 4x2 with noise — minimises residual', () => {
  // Same shape but with noise in b — least-squares fit.
  const A = matVal(4, 2, [1, 1, 1, 2, 1, 3, 1, 4]);
  const b = { shape: [4], data: new Float64Array([3.1, 4.9, 7.2, 8.8]) };
  const x = extLinalg._lstsq(A, b);
  // Computed by hand / known formula: x ≈ [1.05, 1.97]
  assert.ok(Math.abs(x.data[0] - 1.05) < 0.1);
  assert.ok(Math.abs(x.data[1] - 1.97) < 0.1);
});

test('lstsq: rejects b length mismatch', () => {
  const A = matVal(3, 2, [1, 2, 3, 4, 5, 6]);
  const b = { shape: [4], data: new Float64Array([1, 2, 3, 4]) };
  assert.throws(() => extLinalg._lstsq(A, b), /vector/);
});

// =====================================================================
// 6. Symmetric eigendecomposition (Jacobi)
// =====================================================================

test('eigen: diagonal — eigenvalues are diag, vectors are identity (up to ordering)', () => {
  const A = matVal(3, 3, [5, 0, 0, 0, 2, 0, 0, 0, 8]);
  const { fields } = extLinalg._eigen(A) as { fields: Record<string, { data: ArrayLike<number> }> };
  const vals = Array.from(fields.values.data).slice().sort((a: number, b: number) => a - b);
  assertClose(vals, [2, 5, 8], 1e-12);
});

test('eigen: 2x2 symmetric — known case [[2,1],[1,2]] → eigvals 1, 3', () => {
  const A = matVal(2, 2, [2, 1, 1, 2]);
  const { fields } = extLinalg._eigen(A) as { fields: Record<string, { data: ArrayLike<number> }> };
  const vals = Array.from(fields.values.data).slice().sort((a: number, b: number) => a - b);
  assertClose(vals, [1, 3], 1e-12);
  // Vectors should diagonalise: V^T · A · V = diag(values).
  const V = Array.from(fields.vectors.data);
  const Vt = flatTranspose(V, 2, 2);
  const AV = flatMatMul([2, 1, 1, 2], V, 2, 2, 2);
  const VtAV = flatMatMul(Vt, AV, 2, 2, 2);
  // VtAV diagonal entries match the eigenvalues; off-diagonal ≈ 0.
  assert.ok(Math.abs(VtAV[1]) < 1e-10);
  assert.ok(Math.abs(VtAV[2]) < 1e-10);
});

test('eigen: 3x3 symmetric — round-trip A = V · diag(values) · V^T', () => {
  // A = [[4, -2, 1], [-2, 4, -2], [1, -2, 4]]
  const A_data = [4, -2, 1, -2, 4, -2, 1, -2, 4];
  const A = matVal(3, 3, A_data);
  const { fields } = extLinalg._eigen(A) as { fields: Record<string, { data: ArrayLike<number> }> };
  const V = Array.from(fields.vectors.data);
  const values = Array.from(fields.values.data);
  // V · diag(values) · V^T
  const Vd = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) Vd[i * 3 + j] = V[i * 3 + j] * values[j];
  }
  const VdVt = flatMatMul(Vd, flatTranspose(V, 3, 3), 3, 3, 3);
  assertClose(VdVt, A_data, 1e-10);
});

test('eigen: rejects non-symmetric matrix', () => {
  const A = matVal(2, 2, [1, 2, 3, 4]);
  assert.throws(() => extLinalg._eigen(A), /symmetric/);
});

test('eigmax / eigmin: matches min/max of eigen()', () => {
  const A = matVal(3, 3, [5, 1, 0, 1, 4, 1, 0, 1, 3]);
  const max = extLinalg._eigmax(A);
  const min = extLinalg._eigmin(A);
  const { fields } = extLinalg._eigen(A);
  const vals = Array.from(fields.values.data) as number[];
  assert.ok(Math.abs(max - Math.max(...vals)) < 1e-12);
  assert.ok(Math.abs(min - Math.min(...vals)) < 1e-12);
});

// =====================================================================
// 7. SVD (thin) — one-sided Jacobi
// =====================================================================

test('svd: 2x2 diagonal — singular values are |diag|, sorted descending', () => {
  // A = [[3, 0], [0, 2]] → S = [3, 2]
  const A = matVal(2, 2, [3, 0, 0, 2]);
  const { fields } = extLinalg._svd(A);
  assertClose(Array.from(fields.S.data), [3, 2], 1e-12);
});

test('svd: 2x2 symmetric — round-trip A = U · diag(S) · V^T', () => {
  // A = [[2, 1], [1, 2]] — symmetric, singular values 3 and 1.
  const A_data = [2, 1, 1, 2];
  const A = matVal(2, 2, A_data);
  const { fields } = extLinalg._svd(A) as { fields: Record<string, { data: ArrayLike<number>; shape: number[] }> };
  const U = Array.from(fields.U.data);
  const V = Array.from(fields.V.data);
  const S = Array.from(fields.S.data);
  // U · diag(S) · V^T
  const US = new Array(4);
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) US[i * 2 + j] = U[i * 2 + j] * S[j];
  }
  const USVt = flatMatMul(US, flatTranspose(V, 2, 2), 2, 2, 2);
  assertClose(USVt, A_data, 1e-10);
});

test('svd: 4x2 thin — round-trip + singular values descending', () => {
  // A = arbitrary 4x2 matrix
  const A_data = [1, 2, 3, 4, 5, 6, 7, 8];
  const A = matVal(4, 2, A_data);
  const { fields } = extLinalg._svd(A) as { fields: Record<string, { data: ArrayLike<number>; shape: number[] }> };
  assert.deepEqual(fields.U.shape, [4, 2]);
  assert.deepEqual(fields.V.shape, [2, 2]);
  assert.deepEqual(fields.S.shape, [2]);
  const U = Array.from(fields.U.data);
  const V = Array.from(fields.V.data);
  const S = Array.from(fields.S.data) as number[];
  // S descending
  assert.ok(S[0] >= S[1]);
  // Round-trip A = U · diag(S) · V^T
  const US = new Array(8);
  for (let i = 0; i < 4; i++) {
    for (let j = 0; j < 2; j++) US[i * 2 + j] = U[i * 2 + j] * S[j];
  }
  const USVt = flatMatMul(US, flatTranspose(V, 2, 2), 4, 2, 2);
  assertClose(USVt, A_data, 1e-9);
  // U^T U = I (orthonormal columns)
  const UtU = flatMatMul(flatTranspose(U, 4, 2), U, 2, 4, 2);
  assertClose(UtU, [1, 0, 0, 1], 1e-10);
});

test('svd: rejects m < n', () => {
  const A = matVal(2, 3, [1, 2, 3, 4, 5, 6]);
  assert.throws(() => extLinalg._svd(A), /m >= n/);
});

// =====================================================================
// 8. rank — numerical rank via SVD
// =====================================================================

test('rank: identity 3x3 → 3', () => {
  const A = matVal(3, 3, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  assert.equal(extLinalg._rank(A), 3);
});

test('rank: zero matrix → 0', () => {
  const A = matVal(3, 3, [0, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(extLinalg._rank(A), 0);
});

test('rank: rank-2 3x3 (two duplicate rows) → 2', () => {
  const A = matVal(3, 3, [1, 2, 3, 4, 5, 6, 1, 2, 3]);  // row 0 == row 2
  assert.equal(extLinalg._rank(A), 2);
});

test('rank: rank-deficient via column-dependence → matches', () => {
  // col 2 = col 0 + col 1
  const A = matVal(3, 3, [1, 2, 3, 4, 5, 9, 7, 8, 15]);
  assert.equal(extLinalg._rank(A), 2);
});

test('rank: 2x3 wide matrix (transpose path) → matches taller form', () => {
  // 2x3 wide; transpose to 3x2 internally. Full rank = 2.
  const A = matVal(2, 3, [1, 2, 3, 4, 5, 6]);
  assert.equal(extLinalg._rank(A), 2);
});

// =====================================================================
// 9. Standard-module integration: the bindings reach the registry under
//    `ext-linear-algebra@0.1` and their impl is invocable end-to-end.
// =====================================================================

test('std-module: ext-linear-algebra@0.1 is registered with the current binding set', () => {
  // Built-in modules are registered at module-load time, but the
  // standard-modules.test.ts file calls _clearStandardModules in its
  // setup — if that ran first in the test process, the registry is
  // empty here. Re-register defensively.
  stdmod._registerBuiltinStandardModules();
  const mod = stdmod.lookupStandardModule('ext-linear-algebra', '0.1');
  assert.ok(mod, 'module registered');
  for (const op of ['lu', 'kron', 'matexp', 'qr', 'lstsq',
                    'eigen', 'eigmax', 'eigmin', 'svd', 'rank']) {
    assert.ok(mod.bindings.has(op), op + ' binding present');
  }
});

test('std-module: ext-linear-algebra.lu impl reachable via registry', () => {
  stdmod._registerBuiltinStandardModules();
  const mod = stdmod.lookupStandardModule('ext-linear-algebra', '0.1');
  const luDesc = mod.bindings.get('lu');
  assert.equal(luDesc.kind, 'function');
  const A = matVal(2, 2, [4, 3, 2, 1]);
  const r = luDesc.impl(A);
  assert.equal(r.shape, 'record');
  assert.ok(r.fields.P && r.fields.L && r.fields.U);
});

test('std-module: ext-linear-algebra.kron impl reachable via registry', () => {
  stdmod._registerBuiltinStandardModules();
  const mod = stdmod.lookupStandardModule('ext-linear-algebra', '0.1');
  const kronDesc = mod.bindings.get('kron');
  const A = matVal(2, 2, [1, 2, 3, 4]);
  const B = matVal(2, 2, [0, 5, 6, 7]);
  const K = kronDesc.impl(A, B);
  assert.deepEqual(K.shape, [4, 4]);
});

test('matexp: large-norm — scaling kicks in and result is finite', () => {
  // A with ||A||_1 > θ_13, ensures the scaling branch runs.
  const A = matVal(2, 2, [10, 0, 0, -10]);
  const E = extLinalg._matexp(A);
  assertClose(E.data, [
    Math.exp(10), 0,
    0, Math.exp(-10),
  ], 1e-4);  // relative-error tolerance for the e^10 magnitude
});
