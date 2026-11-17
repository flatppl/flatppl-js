'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { REGISTRY } = require('../sampler-registry.ts');
const fixtures = require('./fixtures/hep-sampler-cdfs.json').cases;
const prefix = 'h = standard_module("particle-physics", "0.1")\n';

test('normalized HEP laws draw against independent CDFs, including every ARGUS slope branch', async () => {
  const n = 16384, seed = 117;
  for (const c of fixtures) {
    const { ctx } = makeMatCtx(prefix + `K = h.${c.name}\nx ~ iid(K(${c.params}), ${n})`,
      { sampleCount: 1, rootSeed: seed });
    const samples = (await ctx.getMeasure('x')).samples;
    assert.equal(samples.length, n);
    for (let j = 0; j < c.points.length; j++) {
      const p = c.cdf[j], count = samples.reduce((s: number, x: number) => s + +(x <= c.points[j]), 0);
      const se = Math.sqrt(p * (1 - p) / n), tolerance = 6 * se + 2 / n;
      assert.ok(Math.abs(count / n - p) <= tolerance,
        `${c.name}(${c.params}), seed=${seed}, n=${n}, x=${c.points[j]}: ${count / n} vs ${p}, SE=${se}`);
    }
  }
});

test('Crystal Ball tails retain the normalized score for near-one and large exponents', async () => {
  // Independent scipy.stats.crystalball.logpdf values. The power itself
  // overflows at n=1000, although the normalized log density stays ordinary.
  for (const [n, expected] of [[1.01, -5.4948100861839215], [1000, -4.310845293915195]]) {
    const { ctx } = makeMatCtx(prefix + `M = h.CrystalBall(n=${n}, alpha=1.5, sigma=1.0, m0=0.0)\nlp = logdensityof(M, -3.0)`);
    const actual = (await ctx.getMeasure('lp')).samples[0];
    assert.ok(Math.abs(actual - expected) < 2e-12, `n=${n}: ${actual}`);
  }
});

test('HEP draws preserve explicit RNG state across successive bulk calls', async () => {
  for (const c of fixtures.filter((c: any, i: number, a: any[]) => a.findIndex(v => v.name === c.name) === i)) {
    const { ctx } = makeMatCtx(prefix + `
M = h.${c.name}(${c.params})
a, s = rand(iid(M, 8), rnginit(117))
b, t = rand(iid(M, 8), s)
all_draws, r = rand(iid(M, 16), rnginit(117))
joined = cat(a, b)
`, { sampleCount: 1 });
    assert.deepEqual(Array.from((await ctx.getMeasure('joined')).samples),
      Array.from((await ctx.getMeasure('all_draws')).samples), c.name);
  }
});

test('HEP analytical CDFs and quantiles match independent references', () => {
  for (const c of fixtures) {
    const dist = new REGISTRY[c.name].Ctor(...c.params);
    for (let j = 0; j < c.points.length; j++) {
      assert.ok(Math.abs(dist.cdf(c.points[j]) - c.cdf[j]) < 2e-8, `${c.name} CDF`);
    }
    const median = dist.quantile(0.5);
    assert.ok(Math.abs(dist.cdf(median) - 0.5) < 2e-8, `${c.name} median`);
    assert.equal(dist.cdf(dist.support[0]), 0, `${c.name} lower support`);
    assert.equal(dist.cdf(dist.support[1]), 1, `${c.name} upper support`);
    assert.equal(dist.quantile(0), dist.support[0], `${c.name} zero quantile`);
    assert.equal(dist.quantile(1), dist.support[1], `${c.name} unit quantile`);
  }
});

test('HEP draws retain runtime parameters and ordinary outer broadcast shapes', async () => {
  const { ctx } = makeMatCtx(prefix + `
shift ~ Normal(0, 10)
kernel = h.BifurcatedNormal
M = kernel.(shift .+ [0.0, 10.0], [1.0, 1.0], [1.0, 1.0])
x ~ M
z = x .- shift
`, { sampleCount: 4096, rootSeed: 117 });
  const x = (await ctx.getMeasure('x')).value;
  assert.deepEqual(x.shape, [4096, 2]);
  const shift = (await ctx.getMeasure('shift')).samples;
  const z = (await ctx.getMeasure('z')).value;
  const counts = [0, 0];
  for (let i = 0; i < 4096; i++) {
    for (let j = 0; j < 2; j++) assert.equal(z.data[2 * i + j], x.data[2 * i + j] - shift[i]);
    for (let j = 0; j < 2; j++) counts[j] += +(z.data[2 * i + j] - 10 * j <= 0.5);
  }
  // Equal widths give exact standard-normal residuals conditional on shift.
  const probability = 0.6914624612740131, se = Math.sqrt(probability * (1 - probability) / 4096);
  for (const count of counts) assert.ok(Math.abs(count / 4096 - probability) < 6 * se);
});
