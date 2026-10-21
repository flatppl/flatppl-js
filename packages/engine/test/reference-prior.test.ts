'use strict';

// §06 explicit improper priors. Oracles are density
// products and the exact Normal location posterior, not another engine.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const LOCATION = `
b = elementof(reals)
prior = joint(b = Lebesgue(reals))
y ~ Normal(b, 1)
K = kernelof(y, b = b)
L = likelihoodof(pars -> K(b = pars.b), 2)
posterior = bayesupdate(L, prior)
`;

test('explicit reference and proper priors preserve density and mass', async () => {
  const { proc, ctx } = ctxFor(`
b = elementof(cartpow(reals, 2))
s ~ Normal(0, 1)
improper = joint(b = Lebesgue(cartpow(reals, 2)))
proper = lawof((s = s))
prior = joint(improper, proper)
score = logdensityof(prior, record(b = [-3, 5], s = 1))
zero = logdensityof(Lebesgue(reals), 7)
mass = totalmass(prior)
`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal((await ctx.getMeasure('zero')).samples[0], 0);
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0]
    - (-Math.log(2 * Math.PI) / 2 - 0.5)) < 1e-12);
  assert.equal((await ctx.getMeasure('mass')).samples[0], Infinity);
});

test('a Lebesgue location prior gives the Normal likelihood posterior', async () => {
  const { proc, ctx } = ctxFor(LOCATION + `
score = logdensityof(posterior, record(b = 0.5))
`, 2000);
  assert.equal(proc.loweredModule.bindings.get('K').inferredType.result.mass, 'normalized');
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0]
    - (-Math.log(2 * Math.PI) / 2 - 1.5 ** 2 / 2)) < 1e-12);
  const posterior = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'mh', chains: 4, draws: 500, warmup: 300, seed: 12,
    initPositions: [0, 1, 3, 4].map((x) => Float64Array.of(x)),
  });
  const draws = posterior.fields.b.samples;
  const mean = draws.reduce((s: number, x: number) => s + x, 0) / draws.length;
  const variance = draws.reduce((s: number, x: number) => s + (x - mean) ** 2, 0) / draws.length;
  assert.ok(Math.abs(mean - 2) < 0.15, `Normal(2,1) mean: ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.25, `Normal(2,1) variance: ${variance}`);
});

test('RAM initializes a location posterior without drawing its Lebesgue prior', async () => {
  const { ctx } = ctxFor(LOCATION, 3200);
  const posterior = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'ram', chains: 4, draws: 800, warmup: 500, seed: 12,
  });
  const draws = posterior.fields.b.samples;
  const mean = draws.reduce((s: number, x: number) => s + x, 0) / draws.length;
  const variance = draws.reduce((s: number, x: number) => s + (x - mean) ** 2, 0) / draws.length;
  assert.ok(Math.abs(mean - 2) < 0.15, `Normal(2,1) mean: ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.25, `Normal(2,1) variance: ${variance}`);
  await assert.rejects(() => ctx.getMeasure('prior'), /no prior sampler/);
});

test('elliptical slice corrects its reference to the Normal location posterior', async () => {
  const { ctx } = ctxFor(LOCATION, 3200);
  const posterior = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'elliptical-slice-sampler', chains: 4, draws: 800, warmup: 0, seed: 12,
  });
  const draws = posterior.fields.b.samples;
  const mean = draws.reduce((s: number, x: number) => s + x, 0) / draws.length;
  const variance = draws.reduce((s: number, x: number) => s + (x - mean) ** 2, 0) / draws.length;
  assert.ok(Math.abs(mean - 2) < 0.15, `Normal(2,1) mean: ${mean}`);
  assert.ok(Math.abs(variance - 1) < 0.25, `Normal(2,1) variance: ${variance}`);
  await assert.rejects(() => ctx.getMeasure('prior'), /no prior sampler/);
});

test('density-only MCMC retries inside the posterior support and replays seeded starts', async () => {
  const source = `
b = elementof(reals)
prior = joint(b = Lebesgue(reals))
y ~ Uniform(interval(b, b + 1))
K = kernelof(y, b = b)
L = likelihoodof(pars -> K(b = pars.b), 2)
posterior = bayesupdate(L, prior)
`;
  const opts = { backend: 'ram', chains: 4, warmup: 0, draws: 20, seed: 18 };
  const run = async () => {
    const { ctx } = ctxFor(source, 80);
    return (await materialiser.materialiseMeasure('posterior', ctx, opts)).fields.b.samples;
  };
  const first = await run();
  assert.ok(first.every((x: number) => x >= 1 && x <= 2));
  assert.deepEqual(first, await run());
});

test('reference MCMC bounds initialization attempts and accepts explicit distant starts', async () => {
  const { ctx } = ctxFor(`
b = elementof(reals)
prior = joint(b = Lebesgue(reals))
y ~ Uniform(interval(b, b + 1))
K = kernelof(y, b = b)
L = likelihoodof(pars -> K(b = pars.b), 100)
posterior = bayesupdate(L, prior)
`, 80);
  const opts = { backend: 'ram', chains: 4, warmup: 0, draws: 20, seed: 18 };
  await assert.rejects(() => materialiser.materialiseMeasure('posterior', ctx, opts),
    /after 100 initial positions.*Supply finite initPositions/);
  const posterior = await materialiser.materialiseMeasure('posterior', ctx, {
    ...opts, initPositions: [99.1, 99.3, 99.6, 99.9].map(x => Float64Array.of(x)),
  });
  assert.ok(posterior.fields.b.samples.every((x: number) => x >= 99 && x <= 100));
});

