'use strict';

// =====================================================================
// optimizer-linalg.test.ts
// =====================================================================
//
// Small dense linear-algebra helpers the optimizer core needs:
// a symmetric eigensolver (CMA-ES samples via C^{1/2} = B·diag(d)·Bᵀ and
// updates the step-size path via C^{-1/2}), plus matvec. Pure, dependency-
// free; tested directly against the eigen-equation A·vⱼ = λⱼ·vⱼ.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const la = require('../optimizer/linalg.ts');

function matvec(A: any, v: any) {
  const n = A.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) out[i] += A[i][j] * v[j];
  return out;
}

test('symEig: diagonal matrix returns its diagonal as eigenvalues', () => {
  const { values } = la.symEig([[3, 0], [0, 7]]);
  const sorted = values.slice().sort((a: number, b: number) => a - b);
  assert.ok(Math.abs(sorted[0] - 3) < 1e-9, `got ${sorted[0]}`);
  assert.ok(Math.abs(sorted[1] - 7) < 1e-9, `got ${sorted[1]}`);
});

test('symEig: satisfies the eigen-equation A·vⱼ = λⱼ·vⱼ with orthonormal vectors', () => {
  // A symmetric, non-diagonal, positive definite.
  const A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]];
  const { values, vectors } = la.symEig(A);
  const n = 3;
  // Each column j of `vectors` is an eigenvector for values[j].
  for (let j = 0; j < n; j++) {
    const vj = vectors.map((row: any) => row[j]);
    const Av = matvec(A, vj);
    for (let i = 0; i < n; i++) {
      assert.ok(Math.abs(Av[i] - values[j] * vj[i]) < 1e-7,
        `eigen-eq fail col ${j} row ${i}: ${Av[i]} vs ${values[j] * vj[i]}`);
    }
    // unit norm
    const norm2 = vj.reduce((s: number, x: number) => s + x * x, 0);
    assert.ok(Math.abs(norm2 - 1) < 1e-7, `vector ${j} not unit: ${norm2}`);
  }
  // columns orthogonal
  const dot01 = vectors.reduce((s: number, row: any) => s + row[0] * row[1], 0);
  assert.ok(Math.abs(dot01) < 1e-7, `cols 0,1 not orthogonal: ${dot01}`);
});

test('symEig and covariance factors preserve relative accuracy at different scales', () => {
  for (const scale of [1, 1e-16, 1e16]) {
    const A = [[2 * scale, scale], [scale, 2 * scale]];
    const { values, vectors, sqrt, invSqrt } = la.matSqrtAndInvSqrt(A);
    const sorted = values.slice().sort((a: number, b: number) => a - b);
    assert.ok(Math.abs(sorted[0] / scale - 1) < 1e-12);
    assert.ok(Math.abs(sorted[1] / scale - 3) < 1e-12);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
      let av = 0, ss = 0, si = 0;
      for (let k = 0; k < 2; k++) {
        av += (A[i][k] / scale) * vectors[k][j];
        ss += (sqrt[i][k] / Math.sqrt(scale)) * (sqrt[k][j] / Math.sqrt(scale));
        si += sqrt[i][k] * invSqrt[k][j];
      }
      assert.ok(Math.abs(av - (values[j] / scale) * vectors[i][j]) < 1e-12);
      assert.ok(Math.abs(ss - A[i][j] / scale) < 1e-12);
      assert.ok(Math.abs(si - (i === j ? 1 : 0)) < 1e-12);
    }
  }
});

test('covariance factor retains its eigenvalue floor for a zero matrix', () => {
  const { values, sqrt, invSqrt } = la.matSqrtAndInvSqrt([[0, 0], [0, 0]]);
  assert.deepEqual(values, [0, 0]);
  assert.ok(Math.abs(sqrt[0][0] / 1e-15 - 1) < 1e-12);
  assert.ok(Math.abs(invSqrt[0][0] / 1e15 - 1) < 1e-12);
  assert.deepEqual(la.symEig([[1e-40]]), { values: [1e-40], vectors: [[1]] });
  assert.deepEqual(la.symEig([[1e200, 0], [0, 1e-200]]).values, [1e200, 1e-200]);
});

test('matSqrtAndInvSqrt: B·diag(d)·Bᵀ reconstructs A^{1/2}, and the inverse undoes it', () => {
  const A = [[4, 1], [1, 3]];
  const { sqrt, invSqrt } = la.matSqrtAndInvSqrt(A);
  // sqrt·sqrt ≈ A
  const s2 = [
    [sqrt[0][0] * sqrt[0][0] + sqrt[0][1] * sqrt[1][0], sqrt[0][0] * sqrt[0][1] + sqrt[0][1] * sqrt[1][1]],
    [sqrt[1][0] * sqrt[0][0] + sqrt[1][1] * sqrt[1][0], sqrt[1][0] * sqrt[0][1] + sqrt[1][1] * sqrt[1][1]],
  ];
  assert.ok(Math.abs(s2[0][0] - 4) < 1e-7 && Math.abs(s2[0][1] - 1) < 1e-7
    && Math.abs(s2[1][1] - 3) < 1e-7, `sqrt² != A: ${JSON.stringify(s2)}`);
  // sqrt·invSqrt ≈ I
  const id = [
    [sqrt[0][0] * invSqrt[0][0] + sqrt[0][1] * invSqrt[1][0], sqrt[0][0] * invSqrt[0][1] + sqrt[0][1] * invSqrt[1][1]],
    [sqrt[1][0] * invSqrt[0][0] + sqrt[1][1] * invSqrt[1][0], sqrt[1][0] * invSqrt[0][1] + sqrt[1][1] * invSqrt[1][1]],
  ];
  assert.ok(Math.abs(id[0][0] - 1) < 1e-7 && Math.abs(id[0][1]) < 1e-7
    && Math.abs(id[1][0]) < 1e-7 && Math.abs(id[1][1] - 1) < 1e-7,
    `sqrt·invSqrt != I: ${JSON.stringify(id)}`);
});
