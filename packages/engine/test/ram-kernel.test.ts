'use strict';
// packages/engine/test/ram-kernel.test.ts
// Unit test for the rank-1 Cholesky update/downdate helper: after updating the
// factor S (C = S Sᵀ) to the factor of C + alpha·v vᵀ, the recomputed product
// must match a DENSE Cholesky of the same target matrix (independent oracle) to
// machine precision — for both an update (alpha>0) and a downdate (alpha<0).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { cholRank1Update } = require('../ram-kernel.ts');

// Reference dense Cholesky (lower). Throws if not PD.
function denseChol(A: Float64Array, dim: number): Float64Array {
  const L = new Float64Array(dim * dim);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * dim + j];
      for (let k = 0; k < j; k++) s -= L[i * dim + k] * L[j * dim + k];
      if (i === j) { assert.ok(s > 0, 'reference target is PD'); L[i * dim + j] = Math.sqrt(s); }
      else L[i * dim + j] = s / L[j * dim + j];
    }
  }
  return L;
}
function SSt(S: Float64Array, dim: number): Float64Array {
  const C = new Float64Array(dim * dim);
  for (let i = 0; i < dim; i++) for (let j = 0; j < dim; j++) {
    let s = 0; for (let k = 0; k < dim; k++) s += S[i * dim + k] * S[j * dim + k];
    C[i * dim + j] = s;
  }
  return C;
}
function maxdiff(a: Float64Array, b: Float64Array): number { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; }

const dim = 3;
const S0 = new Float64Array([2, 0, 0, 0.5, 1.3, 0, -0.4, 0.2, 0.9]); // lower-tri, PD

for (const alpha of [0.7, -0.3]) {
  test(`cholRank1Update matches dense recompute (alpha=${alpha})`, () => {
    const v = new Float64Array([0.6, -0.9, 0.4]);
    const C = SSt(S0, dim);
    const target = C.map((x, i) => x + alpha * v[Math.floor(i / dim)] * v[i % dim]);
    const Lref = denseChol(target, dim);
    const S = Float64Array.from(S0);
    const ok = cholRank1Update(S, Float64Array.from(v), alpha, dim);
    assert.equal(ok, true, 'update/downdate stayed PD');
    assert.ok(maxdiff(S, Lref) < 1e-12, `S vs dense factor maxdiff ${maxdiff(S, Lref)}`);
    assert.ok(maxdiff(SSt(S, dim), target) < 1e-12, `S Sᵀ vs target maxdiff ${maxdiff(SSt(S, dim), target)}`);
  });
}

test('cholRank1Update reports PD loss on an over-large downdate', () => {
  const S = Float64Array.from(S0);
  // Downdate by a huge vector along the first axis → C[0,0] goes negative.
  const v = new Float64Array([100, 0, 0]);
  const ok = cholRank1Update(S, v, -1, dim);
  assert.equal(ok, false, 'downdate that breaks PD returns false');
});
