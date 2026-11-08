// A per-atom matrix times a per-atom matrix of a different cell shape,
// [N, m, n] × [N, n, p]. `mulN` sent two batches to its per-atom `mul` only
// when their shapes were equal, so `A * Z` with A [2, 2] and Z [2, 3] per
// atom failed "mulN: unsupported atom-batched shape combination". posteriorDB
// hierarchical_gp reaches it as `lower_cholesky(cov) * GP_std`.
// Oracle: the matrix product per atom, written out.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function matmul(a: Float64Array, m: number, n: number, b: Float64Array, p: number): number[] {
  const out = new Array(m * p).fill(0);
  for (let i = 0; i < m; i++) for (let k = 0; k < p; k++) {
    for (let j = 0; j < n; j++) out[i * p + k] += a[i * n + j] * b[j * p + k];
  }
  return out;
}

test('per-atom [2,2] × [2,3] and lower_cholesky(…) × [2,3] are the per-atom products', async () => {
  const src = 'v ~ iid(Normal(0.0, 1.0), 2)\nA = v * transpose(v) + diagmat([1.0, 1.0])\n'
    + 'Z ~ iid(Normal(0.0, 1.0), [2, 3])\nY = A * Z\nL = lower_cholesky(A)\nW = L * Z\n';
  const N = 6;
  const { ctx } = makeMatCtx(src, { sampleCount: N });
  const A = (await ctx.getMeasure('A')).value.data;
  const Z = (await ctx.getMeasure('Z')).value.data;
  const L = (await ctx.getMeasure('L')).value.data;
  const Y = (await ctx.getMeasure('Y')).value;
  const W = (await ctx.getMeasure('W')).value;
  assert.deepEqual(Y.shape, [N, 2, 3]);
  assert.deepEqual(W.shape, [N, 2, 3]);
  for (let i = 0; i < N; i++) {
    const a = A.subarray(4 * i, 4 * i + 4), z = Z.subarray(6 * i, 6 * i + 6), l = L.subarray(4 * i, 4 * i + 4);
    // L is the Cholesky factor of A: L·Lᵀ = A, upper triangle zero.
    assert.ok(Math.abs(l[1]) < 1e-15, 'lower triangular');
    const llt = [l[0] * l[0], l[0] * l[2], l[2] * l[0], l[2] * l[2] + l[3] * l[3]];
    llt.forEach((x, k) => assert.ok(Math.abs(x - a[k]) < 1e-12, `atom ${i} LLᵀ[${k}]`));
    const y = matmul(a, 2, 2, z, 3), w = matmul(l, 2, 2, z, 3);
    for (let k = 0; k < 6; k++) {
      assert.ok(Math.abs(Y.data[6 * i + k] - y[k]) < 1e-12, `Y atom ${i} cell ${k}`);
      assert.ok(Math.abs(W.data[6 * i + k] - w[k]) < 1e-12, `W atom ${i} cell ${k}`);
    }
  }
});
