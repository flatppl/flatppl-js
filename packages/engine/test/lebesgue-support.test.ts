'use strict';

// §06 reference supports retain volume and the canonical coordinate charts.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');
const normal = (x: number) => -(Math.log(2 * Math.PI) + x * x) / 2;

test('Lebesgue support density and volume follow continuous reference geometry', async () => {
  const cases = [
    ['posreals', '2', '-1', Infinity],
    ['interval(2, inf)', '3', '1', Infinity],
    ['interval(-inf, -2)', '-3', '-1', Infinity],
    ['interval(2, 5)', '3', '1', 3],
    ['cartpow(interval(2, 5), 2)', '[3, 4]', '[3, 6]', 9],
    ['cartpow(cartpow(interval(0, 2), 2), 2)', '[[1,1],[1,1]]', '[[1,1],[1,3]]', 16],
    ['cartprod(interval(0, 2), interval(0, 3))', '[1, 2]', '[1, 4]', 6],
    ['cartprod(a = interval(0, 2), b = posreals)', 'record(a=1,b=2)', 'record(a=1,b=-1)', Infinity],
    ['stdsimplex(3)', '[0.2, 0.3, 0.5]', '[-0.2, 0.3, 0.9]', 0.5],
    ['stdsimplex(3)', '[0.2, 0.3, 0.5]', '[0.2, 0.3, 0.6]', 0.5],
    ['ordered(3)', '[-1, 0, 2]', '[-1, 2, 0]', Infinity],
    ['zerosum(3)', '[1, -1, 0]', '[inf, -inf, 0]', Infinity],
    ['zerosum(3)', '[1, -1, 0]', '[-1, 0, 2]', Infinity],
    ['zerosum(1)', '[0]', '[inf]', 1],
  ];
  for (const [support, inside, outside, volume] of cases) {
    const { proc, ctx } = ctxFor(`
S = ${support}
M = Lebesgue(support = S)
inside = logdensityof(M, ${inside})
outside = logdensityof(M, ${outside})
mass = totalmass(M)
`, 1);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), [], String(support));
    assert.equal((await ctx.getMeasure('inside')).samples[0], 0, String(support));
    assert.equal((await ctx.getMeasure('outside')).samples[0], -Infinity, String(support));
    const actual = (await ctx.getMeasure('mass')).samples[0];
    assert.ok(actual === volume || Math.abs(actual - Number(volume)) < 1e-12, `${support}: ${actual}`);
  }
});

test('dependent Lebesgue bounds and iid keep zero reference contribution', async () => {
  const { proc, ctx } = ctxFor(`
a = elementof(reals)
K = functionof(iid(Lebesgue(interval(a, a + 3)), 2), a)
prior = jointchain(a = Normal(0, 1), b = pars -> K(a = pars.a))
inside = logdensityof(prior, record(a = 1, b = [2, 3]))
outside = logdensityof(prior, record(a = 1, b = [2, 5]))
`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.ok(Math.abs((await ctx.getMeasure('inside')).samples[0] - normal(1)) < 1e-12);
  assert.equal((await ctx.getMeasure('outside')).samples[0], -Infinity);
  const conditional = ctxFor(`
upper = elementof(posreals)
M = Lebesgue(interval(0, upper))
mass = totalmass(M)
f = functionof(mass, upper = upper)
volume = f(3)
`, 1).ctx;
  assert.ok(Math.abs((await conditional.getMeasure('volume')).samples[0] - 3) < 1e-12);
});

test('Lebesgue supports use the existing MCMC charts and reference Jacobians', async () => {
  const cases = [
    ['posreals', 'Normal(b,1)', '0', [1], 1, 0],
    ['interval(2, 6)', 'Normal(b,1)', '0', [4], 1, 0],
    ['ordered(3)', 'Normal.(b,1)', '[0,0,0]', [0, 1, 2], 3, 0],
    ['zerosum(3)', 'Normal.(b,1)', '[0,0,0]', [0, 0, 0], 2, 0],
    ['stdsimplex(3)', 'Normal.(b,1)', '[0,0,0]', [1 / 3, 1 / 3, 1 / 3], 2, -3 * Math.log(3)],
    ['cartpow(posreals, 2)', 'Normal.(b,1)', '[0,0]', [1, 1], 2, 0],
    ['cartprod(interval(2, 6), posreals)', 'Normal.(b,1)', '[0,0]', [4, 1], 2, 0],
  ];
  for (const [support, likelihood, observed, point, dim, jacobian] of cases) {
    const { ctx } = ctxFor(`
b = elementof(${support})
prior = joint(b = Lebesgue(${support}))
y ~ ${likelihood}
K = kernelof(y, b = b)
L = likelihoodof(pars -> K(b = pars.b), ${observed})
posterior = bayesupdate(L, prior)
`, 1);
    const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
    assert.equal(view.dim, dim, String(support));
    const p = point as number[];
    const actual = view.logPosterior(new Float64Array(Number(dim)));
    const expected = p.reduce((sum: number, x: number) => sum + normal(x), 0) + Number(jacobian);
    assert.ok(Math.abs(actual - expected) < 1e-10, `${support}: ${actual} vs ${expected}`);
  }
});

test('positive Lebesgue supports posterior MCMC and still refuses prior-based inference', async () => {
  const { ctx } = ctxFor(`
b = elementof(posreals)
prior = joint(b = Lebesgue(posreals))
y ~ Normal(b, 1)
K = kernelof(y, b = b)
L = likelihoodof(pars -> K(b = pars.b), 0)
posterior = bayesupdate(L, prior)
`, 2000);
  const posterior = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'mh', chains: 4, draws: 500, warmup: 300, seed: 12,
    initPositions: [0.25, 0.5, 1, 2].map(x => Float64Array.of(x)),
  });
  const draws = posterior.fields.b.samples;
  assert.ok(draws.every((x: number) => x > 0));
  const mean = draws.reduce((s: number, x: number) => s + x, 0) / draws.length;
  assert.ok(Math.abs(mean - Math.sqrt(2 / Math.PI)) < 0.12, `half-Normal mean: ${mean}`);
  await assert.rejects(() => materialiser.materialiseMeasure('posterior', ctx, { backend: 'is' }), /no prior sampler/);
});

for (const backend of ['ram', 'slice', 'elliptical-slice-sampler']) {
  test(`positive Lebesgue initializes ${backend} through the existing free-coordinate chart`, async () => {
    const { ctx } = ctxFor(`
b = elementof(posreals)
prior = joint(b = Lebesgue(posreals))
y ~ Normal(b, 1)
K = kernelof(y, b = b)
L = likelihoodof(pars -> K(b = pars.b), 0)
posterior = bayesupdate(L, prior)
`, 3200);
    const posterior = await materialiser.materialiseMeasure('posterior', ctx, {
      backend, chains: 4, draws: 800, warmup: 500, seed: 12,
    });
    const draws = posterior.fields.b.samples;
    assert.ok(draws.every((x: number) => Number.isFinite(x) && x > 0));
    const mean = draws.reduce((s: number, x: number) => s + x, 0) / draws.length;
    const second = draws.reduce((s: number, x: number) => s + x * x, 0) / draws.length;
    assert.ok(Math.abs(mean - Math.sqrt(2 / Math.PI)) < 0.12, `half-Normal mean: ${mean}`);
    assert.ok(Math.abs(second - 1) < 0.2, `half-Normal second moment: ${second}`);
  });
}
