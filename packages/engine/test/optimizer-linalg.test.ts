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
