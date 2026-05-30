'use strict';

// =====================================================================
// batched-matmul.test.ts — rank-2 × rank-2 atom-batched matmul variant
// =====================================================================
//
// P6 follow-up (TODO-flatppl-js.md). The variant registers as
// `mul(rank-2, rank-2)` atom-aware → `_matBatchedMatMul`. Three
// input patterns produce the same [N, m, p] output:
//   - A=[N, m, n] × B=[n, p]
//   - A=[m, n]    × B=[N, n, p]
//   - A=[N, m, n] × B=[N, n, p]
//
// Pins the variant fires AND produces the same values as per-atom
// matmul via `_matMatMul`. The dispatcher's atom-aware matcher is
// the wire; the variant supplies the impl.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const valueOps = require('../value-ops.ts');
const valueLib = require('../value.ts');

function flatRowMajor(rows: number[][]): Float64Array {
  const r = rows.length, c = rows[0].length;
  const out = new Float64Array(r * c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) out[i * c + j] = rows[i][j];
  }
  return out;
}

function rank2Value(rows: number[][]) {
  return { shape: [rows.length, rows[0].length], data: flatRowMajor(rows) };
}

function rank3Value(atoms: number[][][]) {
  const N = atoms.length, m = atoms[0].length, n = atoms[0][0].length;
  const out = new Float64Array(N * m * n);
  let k = 0;
  for (let a = 0; a < N; a++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) out[k++] = atoms[a][i][j];
    }
  }
  return { shape: [N, m, n], data: out };
}

// =====================================================================
// 1. Both atom-batched: A=[N, m, n] × B=[N, n, p] → [N, m, p]
// =====================================================================

test('_matBatchedMatMul: both atom-batched per-atom matmul', () => {
  const N = 3, m = 2, n = 2, p = 2;
  const A = rank3Value([
    [[1, 2], [3, 4]],
    [[1, 0], [0, 1]],
    [[2, 0], [0, 2]],
  ]);
  const B = rank3Value([
    [[5, 6], [7, 8]],
    [[1, 2], [3, 4]],
    [[1, 1], [1, 1]],
  ]);
  const out = valueOps._matBatchedMatMul(A, B, N);
  assert.deepEqual(out.shape, [N, m, p]);
  // Atom 0: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
  assert.equal(out.data[0], 19);
  assert.equal(out.data[1], 22);
  assert.equal(out.data[2], 43);
  assert.equal(out.data[3], 50);
  // Atom 1: identity × [[1,2],[3,4]] = [[1,2],[3,4]]
  assert.equal(out.data[4], 1);
  assert.equal(out.data[5], 2);
  assert.equal(out.data[6], 3);
  assert.equal(out.data[7], 4);
  // Atom 2: 2*I × [[1,1],[1,1]] = [[2,2],[2,2]]
  assert.equal(out.data[8], 2);
  assert.equal(out.data[11], 2);
});

// =====================================================================
// 2. A batched, B shared
// =====================================================================

test('_matBatchedMatMul: A=[N,m,n] × B=[n,p] (shared B)', () => {
  const N = 2;
  const A = rank3Value([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
  ]);
  const B = rank2Value([[1, 0], [0, 1]]);    // identity, shape [2, 2]
  const out = valueOps._matBatchedMatMul(A, B, N);
  assert.deepEqual(out.shape, [N, 2, 2]);
  // Output equals A unchanged (identity right-mul).
  assert.deepEqual(Array.from(out.data), [1, 2, 3, 4, 5, 6, 7, 8]);
});

// =====================================================================
// 3. A shared, B batched
// =====================================================================

test('_matBatchedMatMul: A=[m,n] × B=[N,n,p] (shared A)', () => {
  const N = 2;
  const A = rank2Value([[2, 0], [0, 2]]);   // 2 * identity, shape [2, 2]
  const B = rank3Value([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
  ]);
  const out = valueOps._matBatchedMatMul(A, B, N);
  assert.deepEqual(out.shape, [N, 2, 2]);
  // Output equals 2*B.
  assert.deepEqual(Array.from(out.data),
    [2, 4, 6, 8, 10, 12, 14, 16]);
});

// =====================================================================
// 4. Conformance: equals per-atom _matMatMul over N samples
// =====================================================================

test('_matBatchedMatMul: equals per-atom _matMatMul (conformance oracle)', () => {
  const N = 5, m = 3, n = 4, p = 2;
  // Generate N random atom-pairs.
  const atomsA: number[][][] = [];
  const atomsB: number[][][] = [];
  for (let a = 0; a < N; a++) {
    const Am: number[][] = [];
    for (let i = 0; i < m; i++) {
      const row: number[] = [];
      for (let j = 0; j < n; j++) row.push((a * 13 + i * 7 + j * 3) % 17);
      Am.push(row);
    }
    const Bm: number[][] = [];
    for (let i = 0; i < n; i++) {
      const row: number[] = [];
      for (let j = 0; j < p; j++) row.push((a * 5 + i * 11 + j * 2) % 13);
      Bm.push(row);
    }
    atomsA.push(Am); atomsB.push(Bm);
  }
  const A = rank3Value(atomsA);
  const B = rank3Value(atomsB);
  const batched = valueOps._matBatchedMatMul(A, B, N);
  // Per-atom oracle.
  for (let a = 0; a < N; a++) {
    const Aa = rank2Value(atomsA[a]);
    const Ba = rank2Value(atomsB[a]);
    const perAtom = valueOps._matMatMul(Aa, Ba);
    for (let k = 0; k < m * p; k++) {
      assert.equal(batched.data[a * m * p + k], perAtom.data[k],
        `atom ${a} flat[${k}]: batched=${batched.data[a * m * p + k]} vs per-atom=${perAtom.data[k]}`);
    }
  }
});
