'use strict';

const assert = require('node:assert/strict');
const { inBothModes } = require('./_perf-helpers.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');

const normal = (x: number, mu: number, sigma: number) =>
  -.5 * Math.log(2 * Math.PI) - Math.log(sigma) - .5 * ((x - mu) / sigma) ** 2;
const near = (actual: number, expected: number) => assert.ok(Math.abs(actual - expected) < 1e-11,
  `${actual} != ${expected}`);

inBothModes('Normal plates keep live scales and outer atoms separate', 'density.sharedParameters', async () => {
  const points = [{ mu: [0, 1, 2], sigma: .5 }, { mu: [2, -1, 0], sigma: 2 }];
  for (const observed of [[.2, -.7, 1.1], [1, 2, -1]]) {
    const { ctx } = ctxFor(`
mu ~ iid(Normal(0, 2), 3)
sigma ~ Exponential(1)
y ~ Normal.(mu, sigma)
P = lawof(record(mu = mu, sigma = sigma))
K = kernelof(y, mu = mu, sigma = sigma)
L = likelihoodof(K, ${JSON.stringify(observed)})
posterior = bayesupdate(L, P)
`, 1);
    const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
    const expected = points.map(p => observed.reduce((s, y, i) => s + normal(y, p.mu[i], p.sigma), 0));
    for (const i of [0, 1, 0]) near(scorer.likOf(points[i]), expected[i]);
    scorer.priorLikBatch(points).lik.forEach((v: number, i: number) => near(v, expected[i]));
  }
});

inBothModes('Normal matrix plates distinguish singleton and varying scales', 'density.sharedParameters', async () => {
  for (const scales of [[[2]], [[.5, 2]]]) {
    const { ctx } = ctxFor(`
means = rowstack([[0, 1], [2, 3]])
scales = rowstack(${JSON.stringify(scales)})
score = logdensityof(Normal.(means, scales), rowstack([[.2, 1.4], [2.7, 2.8]]))
`, 3);
    const result = await ctx.getMeasure('score');
    const observed = [.2, 1.4, 2.7, 2.8];
    const expected = observed.reduce((s, x, i) => s + normal(x, i, scales[0][i % scales[0].length]), 0);
    near((result.value?.data || result.samples)[0], expected);
  }
});

inBothModes('prepared Normal components retain independent and shared mixture choices', 'density.sharedParameters', async () => {
  const ys = [-.4, .8, 1.2];
  const a = ys.map(y => normal(y, 0, .7)), b = ys.map(y => normal(y, 2, 1.5));
  const mix = (x: number, y: number) => Math.log(.3 * Math.exp(x) + .7 * Math.exp(y));
  const { ctx } = ctxFor(`
A = Normal(0, .7)
B = Normal(2, 1.5)
independent = logdensityof(iid(superpose(weighted(.3, A), weighted(.7, B)), 3), ${JSON.stringify(ys)})
shared = logdensityof(superpose(weighted(.3, iid(A, 3)), weighted(.7, iid(B, 3))), ${JSON.stringify(ys)})
plain = logdensityof(iid(A, 3), ${JSON.stringify(ys)})
`, 1);
  for (const [name, expected] of [
    ['independent', a.reduce((s, x, i) => s + mix(x, b[i]), 0)],
    ['shared', mix(a.reduce((s, x) => s + x, 0), b.reduce((s, x) => s + x, 0))],
    ['plain', a.reduce((s, x) => s + x, 0)],
  ] as const) {
    const result = await ctx.getMeasure(name);
    near((result.value?.data || result.samples)[0], expected);
  }
});

inBothModes('IID preparation keeps distribution parameters live across families', 'density.sharedParameters', async () => {
  const observed = [.2, .4, .7];
  // Integer-shape densities provide independent normalization constants.
  const cases = [
    { law: 'Beta(2, a)', points: [2, 3], density: (x: number, a: number) =>
      Math.log(a * (a + 1)) + Math.log(x) + (a - 1) * Math.log1p(-x) },
    { law: 'Gamma(3, a)', points: [1, 2], density: (x: number, a: number) =>
      3 * Math.log(a) - Math.log(2) + 2 * Math.log(x) - a * x },
    { law: 'Exponential(a)', points: [1, 2], density: (x: number, a: number) => Math.log(a) - a * x },
    { law: 'StudentT(a)', points: [1, 3], density: (x: number, a: number) => a === 1
      ? -Math.log(Math.PI) - Math.log1p(x * x)
      : Math.log(2 / (Math.PI * Math.sqrt(3))) - 2 * Math.log1p(x * x / 3) },
  ];
  for (const { law, points, density } of cases) {
    const { ctx } = ctxFor(`
a ~ Exponential(1)
y ~ iid(${law}, 3)
P = lawof(record(a = a))
K = kernelof(y, a = a)
L = likelihoodof(K, ${JSON.stringify(observed)})
posterior = bayesupdate(L, P)
`, 1);
    const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
    const expected = points.map(a => observed.reduce((s, x) => s + density(x, a), 0));
    for (const i of [0, 1, 0]) near(scorer.likOf({ a: points[i] }), expected[i]);
    scorer.priorLikBatch(points.map(a => ({ a }))).lik.forEach((v: number, i: number) => near(v, expected[i]));
    const model = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
    const free = points.map(a => model.unconstrainAll({ a }));
    // Exponential(1) prior and the canonical positive-domain log-Jacobian.
    const posterior = expected.map((v, i) => v - points[i] + Math.log(points[i]));
    free.forEach((z, i) => near(model.logPosterior(z), posterior[i]));
    model.logPosteriorBatch(free).forEach((v: number, i: number) => near(v, posterior[i]));
  }
});

inBothModes('prepared non-Normal mixtures retain component supports and weights', 'density.sharedParameters', async () => {
  const observed = [0, .3, .8, 1];
  const expected = observed.reduce((s, x) => s + Math.log(.4 * 12 * x * (1 - x) ** 2 + .6), 0);
  const { ctx } = ctxFor(`
A = Beta(2, 3)
B = Beta(1, 1)
score = logdensityof(iid(superpose(weighted(.4, A), weighted(.6, B)), 4), ${JSON.stringify(observed)})
`, 1);
  const result = await ctx.getMeasure('score');
  near((result.value?.data || result.samples)[0], expected);
});
