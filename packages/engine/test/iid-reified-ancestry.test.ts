'use strict';

// §06 iid copies the whole reified sub-DAG, including stochastic ancestors.
// Gaussian convolution gives Var(x) = 2² + .6² = 4.36. Independent copies
// have covariance zero. Constructor parameters instead remain shared.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { isEngineLimitation } = require('../limitations.ts');

// A MODEL error, not an engine limitation: no engine can make one node both
// fresh per coordinate and shared across them. Assert the code so the two
// verdicts cannot be swapped by a reworded sentence.
function assertBothRoles(err: any, name: string): boolean {
  assert.equal(err.code, 'IID_REIFIED_EXTERNAL_OVERLAP', err.message);
  assert.equal(isEngineLimitation(err), false);
  assert.match(err.message, new RegExp('"' + name + '"'));
  assert.match(err.message, /invalid, not merely unsupported/);
  return true;
}

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

test('iid refuses a captured ancestor also used as an external wrapper input', async () => {
  const { ctx } = makeMatCtx(`
z ~ Normal(0.0, 2.0)
x ~ Normal(z, 0.6)
M = iid(pushfwd(fn(0.0 * _ + z), lawof(x)), 2)
`, { sampleCount: 256, rootSeed: 717 });
  // The output is Dirac(parent z). One cache cannot represent both parent z
  // and its captured copies. Refuse instead of returning independent z's.
  await assert.rejects(() => ctx.getMeasure('M'),
    (e: any) => assertBothRoles(e, 'z'));
});

test('iid detects captured overlap through an external value expression', async () => {
  const { ctx } = makeMatCtx(`
z ~ Normal(0.0, 2.0)
x ~ Normal(z, 0.6)
offset = z + 1.0
M = iid(pushfwd(fn(0.0 * _ + offset), lawof(x)), 2)
`, { sampleCount: 256, rootSeed: 717 });
  await assert.rejects(() => ctx.getMeasure('M'),
    (e: any) => assertBothRoles(e, 'z'));
});

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
