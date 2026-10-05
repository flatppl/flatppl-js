'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');
const { smcSample } = require('../smc-sample.ts');
const { amisSample } = require('../amis-sample.ts');
const { makeDemczKernel } = require('../demcz-kernel.ts');
const { makeEllipticalSliceKernel } = require('../elliptical-slice-kernel.ts');
const { makeSliceKernel } = require('../slice-kernel.ts');
const { runMcmc } = require('../mcmc-driver.ts');
const { runNested } = require('../nested-sample.ts');
const { buildPriorTransform } = require('../prior-transform.ts');
const { mulberry32 } = require('../optimizer/cmaes.ts');
const rng = require('../rng.ts');
const sampler = require('../sampler.ts');

test('AMIS full and reduced mixtures retain a normalized Gaussian target', () => {
  const mv = { dim: 1, logPosterior: (x: Float64Array) =>
    -Math.log(0.5 * Math.sqrt(2 * Math.PI)) - 2 * (x[0] - 2) ** 2 };
  for (const eps of [0, Infinity]) {
    const result = amisSample(mv, { seed: 413, amisSamples: 500, amisIters: 8, amisEpsK: eps });
    assert.equal(result.K, eps === 0 ? 8 : 1);
    let mass = 0, first = 0, second = 0;
    for (let i = 0; i < result.samples.length; i++) {
      const w = Math.exp(result.logW[i]) / result.samples.length;
      const x = result.samples[i][0];
      mass += w;
      first += w * x;
      second += w * (x - 2) ** 2;
    }
    assert.ok(Math.abs(mass - 1) < 0.08, `mass ${mass}`);
    assert.ok(Math.abs(first / mass - 2) < 0.05, `mean ${first / mass}`);
    assert.ok(Math.abs(second / mass - 0.25) < 0.04, `variance ${second / mass}`);
  }
});

const HIERARCHICAL = `
x ~ normalize(weighted(fn(exp(_)), Normal(0, 1)))
y ~ Normal(x, 0.01)
prior = lawof(record(x=x, y=y))
obs ~ Normal(y-x, 0.1)
L = likelihoodof(kernelof(obs, x=x, y=y), 0.0)
posterior = bayesupdate(L, prior)
`;

test('joint prior initialization retains weighted ancestry and SMC evidence', async () => {
  const { ctx } = ctxFor(HIERARCHICAL, 12000);
  const mv = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const points = mv.initFromPrior(12000, mulberry32(12));
  let mean = 0, residual = 0;
  for (const p of points) {
    const t = mv.constrainAll(p);
    mean += t.x / points.length;
    residual += (t.y - t.x) ** 2 / points.length;
  }
  // Completing the square gives x~N(1,1); y-x~N(0,.01²).
  assert.ok(Math.abs(mean - 1) < 0.08, `weighted mean ${mean}`);
  assert.ok(Math.abs(residual - 0.0001) < 0.00002, `residual ${residual}`);
  const result = smcSample(mv, { seed: 12, smcParticles: 12000, smcSteps: 2, smcCESS: 0.01 });
  const logZ = -0.5 * Math.log(2 * Math.PI * (0.01 ** 2 + 0.1 ** 2));
  assert.ok(Math.abs(result.logZ - logZ) < 0.01, `evidence ${result.logZ}`);
});

test('constructor joint initialization keeps repeated factors independent', async () => {
  const { ctx } = ctxFor(`
x = elementof(reals)
y = elementof(reals)
D = Normal(0,1)
prior = joint(x=D, y=D)
obs ~ Normal(x-y, 1)
L = likelihoodof(kernelof(obs, x=x, y=y), 0)
posterior = bayesupdate(L, prior)
`, 4000);
  const mv = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const points = mv.initFromPrior(4000, mulberry32(7));
  const variance = points.reduce((s: number, p: Float64Array) => s + (p[0] - p[1]) ** 2, 0) / points.length;
  assert.ok(Math.abs(variance - 2) < 0.2, `independent-difference variance ${variance}`);
});

test('missing prior pools remain usable for scoring but never become zero draws', async () => {
  const { ctx } = ctxFor(HIERARCHICAL, 100);
  const get = ctx.getMeasure;
  ctx.getMeasure = (name: string) => {
    if (name === 'y') throw new Error('forward pool unavailable');
    return get(name);
  };
  const mv = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  assert.ok(Number.isFinite(mv.logPosteriorConstrained({ x: 1, y: 1 })));
  assert.throws(() => mv.initFromPrior(2, mulberry32(1)));
});

test('prior initialization rejects incompatible atom and weight axes', async () => {
  for (const mismatch of ['atoms', 'weights']) {
    const { ctx } = ctxFor(HIERARCHICAL, 100);
    const get = ctx.getMeasure;
    ctx.getMeasure = async (name: string) => {
      const m = await get(name);
      if (name !== 'y') return m;
      return mismatch === 'atoms' ? { ...m, samples: m.samples.subarray(0, 50) }
        : { ...m, logWeights: new Float64Array(200) };
    };
    const mv = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
    assert.throws(() => mv.initFromPrior(2, mulberry32(1)));
  }
});

test('DEMCz difference proposal is symmetric under exact RNG quadrature', () => {
  const counts = new Map<number, number>();
  for (let a = 0; a < 12; a++) for (let b = 0; b < 12; b++) {
    const kernel = makeDemczKernel({ gamma: 1, b: 0 });
    const ensemble = [0, 1, 3, 7].map(x => Float64Array.of(x));
    const uniforms = [(a + 0.5) / 12, (b + 0.5) / 12, 0.5, 0.5, 0.5];
    let i = 0;
    kernel.step(ensemble, new Float64Array(4), { dim: 1, logPosterior: () => 0 },
      () => uniforms[i++ % uniforms.length], kernel.init(4, 1), 'sample');
    counts.set(ensemble[0][0], (counts.get(ensemble[0][0]) || 0) + 1);
  }
  for (const [delta, count] of counts) assert.equal(counts.get(-delta), count);
});

test('nested replacement preserves the constrained uniform mean near boundaries', () => {
  const n = 40;
  let sum = 0;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const uniforms = [0, (i + 0.5) / n, 0.75, (j + 0.5) / n, 0.5];
    let k = 0;
    const r = runNested((u: Float64Array) => u[0], 1, (x: number) => x,
      { nLive: 2, maxIter: 1, dlogz: 1e-12, sliceSweeps: 1, prng: () => uniforms[k++] });
    sum += r.samples[1];
  }
  assert.ok(Math.abs(sum / (n * n) - 0.5) < 1e-12);
});

function philox(seed: number) {
  const key = rng.keyFromSeed(seed);
  return sampler.makePhiloxPrngAdapter(rng.stateFromKey(key[0], key[1]));
}
function normal(prng: () => number) {
  return Math.sqrt(-2 * Math.log(Math.max(prng(), 1e-300))) * Math.cos(2 * Math.PI * prng());
}
const centeredPool = (n: number) => Array.from({ length: n }, (_, i) => Float64Array.of(i % 2 ? 1 : -1));

test('AMIS proposal moments use each covariance diagonal once, initially and after fitting', () => {
  const result = amisSample({ dim: 1, initFromPrior: centeredPool,
    logPosterior: (x: Float64Array) => -(x[0] ** 2) / 2 },
  { seed: 4, amisSamples: 4, amisIters: 2, amisEpsK: 0, amisFloorFrac: 0 });
  const prng = philox(4), initialVariance = 4 + 1e-6;
  const first = Array.from({ length: 4 }, () => Math.sqrt(initialVariance) * normal(prng));
  const weights = first.map(x => Math.exp(-0.5 * x * x * (1 - 1 / initialVariance)));
  const sum = weights.reduce((a, b) => a + b, 0);
  const mu = first.reduce((s, x, i) => s + weights[i] * x / sum, 0);
  const variance = first.reduce((s, x, i) => s + weights[i] * (x - mu) ** 2 / sum, 0) + 1e-6;
  const expected = first.concat(Array.from({ length: 4 }, () => mu + Math.sqrt(variance) * normal(prng)));
  expected.forEach((x, i) => assert.ok(Math.abs(result.samples[i][0] - x) < 1e-12));
});

test('SMC proposal covariance matches the weighted population covariance', () => {
  const proposals: Float64Array[][] = [];
  const result = smcSample({ dim: 1, initFromPrior: centeredPool,
    logPriorLikBatch: (xs: Float64Array[]) => {
      proposals.push(xs);
      return { prior: new Float64Array(xs.length), lik: new Float64Array(xs.length) };
    } }, { seed: 4, smcParticles: 4, smcSteps: 2 });
  const prng = philox(4); prng(); // systematic-resampling offset
  for (let i = 0; i < 2; i++) {
    const expected = result.samples[i][0] + 2.38 * Math.sqrt(1 + 1e-9) * normal(prng);
    assert.ok(Math.abs(proposals[1][i][0] - expected) < 1e-12);
  }
});

test('elliptical slice fitted reference generates the empirical covariance', () => {
  let proposal = NaN;
  const mv = { initFromPrior: centeredPool, logPosteriorBatch: (xs: Float64Array[]) => {
    proposal = xs[0][0]; return xs.map(x => -(x[0] ** 2) / 2);
  } };
  const kernel = makeEllipticalSliceKernel();
  const uniforms = [0.5, 0, 0.5, 0.25]; let i = 0;
  kernel.step([Float64Array.of(0)], Float64Array.of(0), mv, () => uniforms[i++], kernel.init(1, 1, {}, mv), 'sample');
  assert.ok(Math.abs(proposal - Math.sqrt((1 + 1e-9) * 2 * Math.log(2))) < 1e-12);
});

test('slice initialization uses the sampler seed rather than ambient randomness', () => {
  const mv = { dim: 1, names: ['x'], constrainAll: (x: Float64Array) => ({ x: x[0] }),
    logPosterior: (x: Float64Array) => -(x[0] ** 2) / 2,
    initFromPrior: (n: number, prng: () => number) => Array.from({ length: n }, () => Float64Array.of(prng())) };
  const saved = Math.random;
  const run = (width: number) => {
    let i = 0; Math.random = () => 0.5 + (i++ % 2 ? width : -width);
    return runMcmc(mv, makeSliceKernel(), { nWalkers: 2, warmup: 0, draws: 10, seed: 42,
      initPositions: [Float64Array.of(0), Float64Array.of(0)] }).drawsByName.x;
  };
  try { assert.deepEqual(run(0.1), run(0.4)); } finally { Math.random = saved; }
});

test('truncated Uniform prior resolves its interval support before quantile mapping', () => {
  const { ctx } = ctxFor(`
x ~ normalize(truncate(Uniform(interval(0,2)), interval(0.5,1.5)))
prior = lawof(record(x=x))
y ~ Normal(x,1)
L = likelihoodof(kernelof(y,x=x),0)
posterior = bayesupdate(L,prior)
`, 10);
  const pt = buildPriorTransform(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(pt.transform(Float64Array.of(0.25)).x - 0.75) < 1e-12);
});

test('AMIS prior-scale regularization protects covariance in every direction', () => {
  const result = amisSample({ dim: 2,
    initFromPrior: (n: number) => Array.from({ length: n }, (_, i) => Float64Array.of(i % 2 ? 1 : -1, i % 2 ? 1 : -1)),
    logPosterior: (x: Float64Array) => -1e6 * (x[0] - x[1]) ** 2 - x[0] ** 2 / 2,
  }, { seed: 4, amisSamples: 3, amisIters: 2, amisEpsK: 0, amisFloorFrac: 0.01 });
  const prng = philox(4);
  const z = Array.from({ length: 6 }, () => [normal(prng), normal(prng)]);
  // Three emitted points identify the affine Gaussian proposal. Differences
  // eliminate its mean; solve the 2x2 system for its Cholesky rows.
  const a = z[4][0] - z[3][0], b = z[4][1] - z[3][1];
  const c = z[5][0] - z[3][0], d = z[5][1] - z[3][1];
  const x = result.samples;
  const row = (j: number) => {
    const u = x[4][j] - x[3][j], v = x[5][j] - x[3][j];
    return [(u * d - b * v) / (a * d - b * c), (a * v - u * c) / (a * d - b * c)];
  };
  const r0 = row(0), r1 = row(1);
  const perpendicularVariance = ((r0[0] - r1[0]) ** 2 + (r0[1] - r1[1]) ** 2) / 2;
  assert.ok(perpendicularVariance >= 0.04 - 1e-10, `directional variance ${perpendicularVariance}`);
});
