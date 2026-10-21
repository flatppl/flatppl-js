'use strict';

// §06 iid copies the whole reified sub-DAG, including stochastic ancestors.
// Gaussian convolution gives Var(x) = 2² + .6² = 4.36. Independent copies
// have covariance zero. Constructor parameters instead remain shared.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
function covariance(a: Float64Array, b: Float64Array): number {
  const ma = a.reduce((s, x) => s + x, 0) / a.length;
  const mb = b.reduce((s, x) => s + x, 0) / b.length;
  return a.reduce((s, x, i) => s + (x - ma) * (b[i] - mb), 0) / a.length;
}

for (const rootSeed of [717, 818]) {
  for (const base of ['lawof(x)', 'weighted(2.0, lawof(x))', 'lawof(u)']) {
    test(`iid copies ancestors of ${base}, seed ${rootSeed}`, async () => {
      const n = 30000;
      const { ctx } = makeMatCtx(`
z ~ Normal(0.0, 2.0)
x ~ Normal(z, 0.6)
u = x + 1.0
M = iid(${base}, 2)
`, { sampleCount: n, rootSeed });
      const flat = (await ctx.getMeasure('M')).samples;
      const a = Float64Array.from({ length: n }, (_, i) => flat[2 * i]);
      const b = Float64Array.from({ length: n }, (_, i) => flat[2 * i + 1]);
      // About six standard errors, far below the defective covariance 4.
      assert.ok(Math.abs(covariance(a, b)) < 0.16, `Cov = ${covariance(a, b)}, oracle 0`);
      for (const col of [a, b]) {
        assert.ok(Math.abs(covariance(col, col) - 4.36) < 0.24);
      }
    });
  }
}

test('iid constructor parameters retain the canonical shared draw', async () => {
  const n = 30000;
  const { ctx } = makeMatCtx(`
z ~ Normal(0.0, 2.0)
M = iid(Normal(z, 0.6), 2)
`, { sampleCount: n, rootSeed: 919 });
  const flat = (await ctx.getMeasure('M')).samples;
  const z = (await ctx.getMeasure('z')).samples;
  const a = Float64Array.from({ length: n }, (_, i) => flat[2 * i]);
  const b = Float64Array.from({ length: n }, (_, i) => flat[2 * i + 1]);
  assert.ok(Math.abs(covariance(a, b) - 4) < 0.24);
  for (const col of [a, b]) {
    const noise = Float64Array.from(col, (x, i) => x - z[i]);
    assert.ok(Math.abs(covariance(noise, noise) - 0.36) < 0.025);
  }
});

for (const location of ['z', 'z + 1.0']) {
  test(`iid scopes absorbed ancestry separately from free location ${location}`, async () => {
    const n = 8192;
    const { ctx } = makeMatCtx(`
z ~ Normal(0.0, 2.0)
x ~ Normal(z, 0.6)
offset = ${location}
M = iid(locscale(lawof(x), offset, 1.0), 2)
`, { sampleCount: n, rootSeed: 717 });
    const flat = (await ctx.getMeasure('M')).samples;
    const offset = (await ctx.getMeasure('offset')).samples;
    const a = Float64Array.from({ length: n }, (_, i) => flat[2 * i]);
    const b = Float64Array.from({ length: n }, (_, i) => flat[2 * i + 1]);
    const residuals = [a, b].map(col => Float64Array.from(col, (x, i) => x - offset[i]));
    assert.ok(Math.abs(covariance(a, b) - 4) < 0.65);
    assert.ok(Math.abs(covariance(residuals[0], residuals[1])) < 0.4);
    for (const residual of residuals) {
      assert.ok(Math.abs(covariance(residual, residual) - 4.36) < 0.5);
      assert.ok(Math.abs(covariance(residual, offset)) < 0.4);
    }
  });
}

test('iid respects a named kernel boundary while copying its captured base', async () => {
  const n = 30000;
  const { ctx } = makeMatCtx(`
z ~ Normal(0.0, 2.0)
x ~ Normal(z, 0.6)
y ~ Normal(z, 1.0)
K = kernelof(y, z = z)
C = kchain(lawof(x), K)
M = iid(C, 2)
`, { sampleCount: n, rootSeed: 717 });
  const flat = (await ctx.getMeasure('M')).samples;
  const a = Float64Array.from({ length: n }, (_, i) => flat[2 * i]);
  const b = Float64Array.from({ length: n }, (_, i) => flat[2 * i + 1]);
  assert.ok(Math.abs(covariance(a, b)) < 0.2);
  for (const col of [a, b]) assert.ok(Math.abs(covariance(col, col) - 5.36) < 0.3);
});
