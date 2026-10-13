'use strict';

// §09 regression.Flat and §04 improper prior sources. Oracles are density
// products and the exact Normal location posterior, not another engine.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const REG = 'reg = standard_module("regression", "0.1")\n';
const LOCATION = REG + `
b ~ reg.Flat()
prior = lawof(record(b = b))
y ~ Normal(b, 1)
K = kernelof(y, b = b)
L = likelihoodof(K, 2)
posterior = bayesupdate(L, prior)
`;

test('Flat iid and mixed laws preserve their scores and infinite mass', async () => {
  const { proc, ctx } = ctxFor(REG + `
Flat = reg.Flat
b ~ iid(Flat(), 2)
s ~ Normal(0, 1)
prior = lawof(record(b = b, s = s))
score = logdensityof(prior, record(b = [-3, 5], s = 1))
zero = logdensityof(Flat(), 7)
mass = totalmass(prior)
`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal((await ctx.getMeasure('zero')).samples[0], 0);
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0]
    - (-Math.log(2 * Math.PI) / 2 - 0.5)) < 1e-12);
  assert.equal((await ctx.getMeasure('mass')).samples[0], Infinity);
});

test('Flat with a unit Normal likelihood gives the Normal location posterior', async () => {
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

test('RAM initializes a Flat location posterior without drawing its prior', async () => {
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

test('elliptical slice corrects a density-only reference to the Flat location posterior', async () => {
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

test('Flat MCMC retries inside the posterior support and replays seeded starts', async () => {
  const source = REG + `
b ~ reg.Flat()
prior = lawof(record(b = b))
y ~ Uniform(interval(b, b + 1))
K = kernelof(y, b = b)
L = likelihoodof(K, 2)
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

test('Flat MCMC bounds initialization attempts and accepts explicit distant starts', async () => {
  const { ctx } = ctxFor(REG + `
b ~ reg.Flat()
prior = lawof(record(b = b))
y ~ Uniform(interval(b, b + 1))
K = kernelof(y, b = b)
L = likelihoodof(K, 100)
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

test('improper ancestors cannot disappear through projection or constant maps', async () => {
  const { proc, ctx } = ctxFor(REG + `
a ~ reg.Flat()
b ~ Normal(0, 1)
r = record(a = a, b = b)
projected = lawof(r.b)
collapsed = lawof(0 * a)
score = logdensityof(projected, 1)
mass = totalmass(collapsed)
`, 1);
  // These measures have infinite total mass, even though their outputs only
  // retain a proper coordinate or a constant. Unsupported integrals may refuse.
  assert.notEqual(proc.loweredModule.bindings.get('projected').inferredType.mass, 'normalized');
  assert.notEqual(proc.loweredModule.bindings.get('collapsed').inferredType.mass, 'normalized');
  await assert.rejects(() => ctx.getMeasure('score'), /infinite mass|marginali|no prior sampler/);
  await assert.rejects(() => ctx.getMeasure('mass'), /infinite mass|marginali|no prior sampler/);
});

test('Flat refuses prior draws and prior-drawing importance sampling', async () => {
  const { ctx } = ctxFor(LOCATION + 'sample, state = rand(reg.Flat(), rnginit(7))', 8);
  await assert.rejects(() => ctx.getMeasure('sample'), /no prior sampler/);
  await assert.rejects(() => materialiser.materialiseMeasure('posterior', ctx,
    { backend: 'is' }), /no prior sampler/);
  for (const backend of ['amis', 'smc']) {
    await assert.rejects(() => materialiser.materialiseMeasure('posterior', ctx,
      { backend, chains: 4, draws: 2, warmup: 0 }), (error: Error) => {
      assert.ok(error.message.includes(`backend '${backend}'`));
      assert.match(error.message, /sampleable joint prior/);
      assert.doesNotMatch(error.message, /initPositions/);
      return true;
    });
  }
});

test('Flat source admission does not admit arbitrary unnormalized reweightings', () => {
  const proc = processSource(REG + 'b ~ weighted(2, reg.Flat())');
  assert.ok(proc.diagnostics.some((d: any) => d.severity === 'error'
    && d.message.startsWith('draw requires')));
});

test('jointchain retains the Flat base while scoring the conditional factor', async () => {
  const { ctx } = ctxFor(REG + `
b = elementof(reals)
K = functionof(Normal(b, 1), b = b)
prior = jointchain(b = reg.Flat(), s = K)
x ~ prior
score = logdensityof(lawof(x), record(b = 0.5, s = 1.5))
mass = totalmass(prior)
`, 1);
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0]
    - (-Math.log(2 * Math.PI) / 2 - 0.5)) < 1e-12);
  assert.equal((await ctx.getMeasure('mass')).samples[0], Infinity);
});

test('functionof cuts Flat ancestors from a joint response without changing the closed law', async () => {
  const { proc, ctx } = ctxFor(REG + `
b ~ reg.Flat()
M = joint(Normal(b, 1), Normal(0, 1))
K = functionof(M, b = b)
C = K(2)
x ~ C
sample, state = rand(C, rnginit(7))
score = logdensityof(C, [2, 0])
replicated = iid(M, 2)
conditional_replicated = functionof(replicated, b = b)
y ~ Normal(b, 1)
P = lawof(y)
conditional_law = functionof(P, b = b)
from_law ~ P
conditional_copy = kernelof(from_law, b = b)
a ~ reg.Flat()
U = joint(Normal(b, 1), Normal(a, 1))
partial = functionof(U, b = b)
`, 1);
  const bindings = proc.loweredModule.bindings;
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.notEqual(bindings.get('M').inferredType.mass, 'normalized');
  assert.notEqual(bindings.get('partial').inferredType.result.mass, 'normalized');
  assert.equal(bindings.get('K').inferredType.result.mass, 'normalized');
  assert.equal(bindings.get('C').inferredType.mass, 'normalized');
  for (const name of ['conditional_replicated', 'conditional_law', 'conditional_copy']) {
    assert.equal(bindings.get(name).inferredType.result.mass, 'normalized', name);
  }
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0]
    + Math.log(2 * Math.PI)) < 1e-12);
  await ctx.getMeasure('x');
  const sample = await ctx.getMeasure('sample');
  assert.equal(sample.samples.length, 2);
  assert.ok(sample.samples.every(Number.isFinite));
});

test('identity calls preserve Flat permission without admitting other improper measures', async () => {
  const { proc, ctx } = ctxFor(REG + `
identity(M) = M
P = identity(identity(reg.Flat()))
x ~ P
score = logdensityof(lawof(x), 0)
`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal(proc.loweredModule.bindings.get('P').inferredType.mass, 'locallyfinite');
  assert.equal((await ctx.getMeasure('score')).samples[0], 0);
  for (const source of ['Lebesgue(reals)', 'weighted(2, reg.Flat())']) {
    const rejected = processSource(REG + `identity(M) = M\nx ~ identity(${source})`);
    assert.ok(rejected.diagnostics.some((d: any) => d.message.startsWith('draw requires')), source);
  }
});

test('loaded modules preserve Flat source eligibility and reject imported Lebesgue', async () => {
  const model = REG + `flat = reg.Flat()\nb ~ flat\nprior = lawof(record(b = b))`;
  const { proc, ctx } = ctxFor(`m = load_module("m.flatppl")
x ~ m.flat
z ~ m.prior
scalar_score = logdensityof(lawof(x), 0)
record_score = logdensityof(lawof(z), record(b = 0))
`, 1, { bundle: { sources: { 'm.flatppl': model } } });
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal((await ctx.getMeasure('scalar_score')).samples[0], 0);
  assert.equal((await ctx.getMeasure('record_score')).samples[0], 0);
  const rejected = processSource('m = load_module("m.flatppl")\nx ~ m.flat',
    { bundle: { sources: { 'm.flatppl': 'flat = Lebesgue(reals)' } } });
  assert.ok(rejected.diagnostics.some((d: any) => d.message.startsWith('draw requires')));
});

test('an unsupported imported measure boundary stays failed at the call site', () => {
  const model = REG + 'source = reg.Flat()\nfactory = functionof(source, M = source)';
  for (const actual of ['reg.Flat()', 'Normal(0,1)', 'Lebesgue(reals)', 'weighted(2,Normal(0,1))']) {
    const proc = processSource(REG + `m = load_module("m.flatppl")\nP = m.factory(${actual})\nx ~ P`,
      { bundle: { sources: { 'm.flatppl': model } } });
    const binding = proc.loweredModule.bindings.get('P');
    assert.equal(binding.inferredType.kind, 'failed', actual);
    assert.notEqual(binding.admittedPrior, true, actual);
    assert.ok(proc.diagnostics.some((d: any) => d.severity === 'error'
      && /boundary.*M.*(fixed phase|must be a value)/i.test(d.message)), actual);
  }
});
