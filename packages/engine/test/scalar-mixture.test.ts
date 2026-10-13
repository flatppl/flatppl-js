'use strict';

const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { inBothModes } = require('./_perf-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const density = require('../density.ts');

const close = (actual: number, expected: number) =>
  assert.ok(Math.abs(actual - expected) < 1e-11, `${actual} != ${expected}`);

inBothModes('mixture choice stays outside or inside the iid plate', 'density.scalarMixture', async () => {
  const { ctx } = ctxFor(`
M = normalize(superpose(Bernoulli(.2), Bernoulli(.8)))
A = normalize(superpose(iid(Bernoulli(.2), 2), iid(Bernoulli(.8), 2)))
independent = logdensityof(iid(M, 2), [false, true])
shared = logdensityof(A, [false, true])
rest = logdensityof(joint(A, Bernoulli(.3)), [false, true, true])
`, 1);
  // Independent choices: .5*.5. Shared choice: (.8*.2 + .2*.8)/2.
  close((await ctx.getMeasure('independent')).samples[0], Math.log(.25));
  close((await ctx.getMeasure('shared')).samples[0], Math.log(.16));
  close((await ctx.getMeasure('rest')).samples[0], Math.log(.16 * .3));
});

inBothModes('mixture parameters and normalization stay live across calls', 'density.scalarMixture', async () => {
  const { ctx } = ctxFor(`
a ~ Exponential(1)
b ~ Exponential(1)
p ~ Uniform(unitinterval)
M = normalize(superpose(weighted(a, Bernoulli(p)), weighted(b, Bernoulli(.8))))
y ~ iid(M, 2)
P = lawof(record(a=a, b=b, p=p))
K = kernelof(y, a=a, b=b, p=p)
L = likelihoodof(K, [false, true])
post = bayesupdate(L, P)
`, 1);
  const scorer = await buildLogPi(ctx, ctx.derivations.post);
  const point = { a: 2, b: 3, p: .1 };
  const points = [{ ...point }, { a: 20, b: 30, p: .1 }, { a: 3, b: 1, p: .6 }];
  const likelihood = ({ a, b, p }: typeof point) => {
    const q = (a * p + b * .8) / (a + b);
    return Math.log(q * (1 - q));
  };
  for (const current of [...points, points[0]]) {
    Object.assign(point, current);
    close(scorer.likOf(point), likelihood(current));
  }
  const batch = scorer.priorLikBatch(points).lik;
  points.forEach((current, i) => close(batch[i], likelihood(current)));
  assert.throws(() => scorer.probeLik({ ...point, a: 0, b: 0 }), /ZERO total mass/);
  assert.throws(() => scorer.probeLik({ ...point, a: -1, b: 3 }), /negative|non-negative/);
  close(scorer.likOf(points[0]), likelihood(points[0]));
});

inBothModes('scalar mixtures preserve zero weights, support and tilted leaf scores', 'density.scalarMixture', async () => {
  const lit = (value: number) => ({ kind: 'lit', value });
  const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });
  const measure = call('superpose', call('weighted', lit(0), call('Beta', lit(.5), lit(1))),
    call('Beta', lit(1), lit(1)));
  close(density.logDensity(measure, 0, {}), 0);
  assert.equal(density.logDensity(measure, 2, {}), -Infinity);
  const { ctx } = ctxFor(`
T = normalize(logweighted(fn(1000 * _), Bernoulli(.2)))
rare = logdensityof(normalize(superpose(T, T)), false)
C = normalize(superpose(weighted(abs2(complex(1, 2)), Bernoulli(.2)), Bernoulli(.8)))
complex_weight = logdensityof(C, true)
`, 1);
  close((await ctx.getMeasure('rare')).samples[0], -1000 + Math.log(4));
  close((await ctx.getMeasure('complex_weight')).samples[0], Math.log((5 * .2 + .8) / 6));
});

inBothModes('selectors and variate-dependent weights keep their scoring law', 'density.scalarMixture', async () => {
  const { ctx } = ctxFor(`
c ~ Bernoulli(.3)
y ~ ifelse(c, Bernoulli(.2), Bernoulli(.8))
P = lawof(record(c=c, y=y))
selected = logdensityof(P, record(c=true, y=true))
f(x) = x*x
g(x) = (x-1)*(x-1)
M = superpose(weighted(f, Normal(0, 1)), weighted(g, Normal(1, 1)))
callable_weight = logdensityof(M, .5)
`, 1);
  close((await ctx.getMeasure('selected')).samples[0], Math.log(.3 * .2));
  close((await ctx.getMeasure('callable_weight')).samples[0], Math.log(.5) - .5*Math.log(2*Math.PI) - .125);
});
