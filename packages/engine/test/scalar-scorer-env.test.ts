'use strict';

const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { inBothModes } = require('./_perf-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const normal = (x: number, mu: number, sigma: number) =>
  -Math.log(sigma * Math.sqrt(2 * Math.PI)) - .5 * ((x - mu) / sigma) ** 2;

inBothModes('scalar scoring reuses fixed values and retains required derived priors', 'mcmc.scalarEnv', async () => {
  for (const derivedPrior of [false, true]) {
    const { ctx } = ctxFor(`
data = [1, 2]
a ~ Normal(0, 1)
scale = exp(a)
b ~ Normal(0, ${derivedPrior ? 'scale' : '2'})
mean = a + b
y ~ iid(Normal(mean, 1), 2)
P = lawof(record(a=a, b=b))
K = kernelof(y, a=a, b=b)
L = likelihoodof(K, data)
post = bayesupdate(L, P)
`, 1);
    const scorer = await buildLogPi(ctx, ctx.derivations.post);
    const points = [{ a: .2, b: -.3 }, { a: -.6, b: 1.2 }];
    const batch = scorer.logPiBatch(points);
    for (const [i, p] of points.entries()) {
      const prior = normal(p.a, 0, 1) + normal(p.b, 0, derivedPrior ? Math.exp(p.a) : 2);
      const lik = normal(1, p.a + p.b, 1) + normal(2, p.a + p.b, 1);
      for (const [actual, expected] of [[scorer.priorOf(p), prior], [scorer.likOf(p), lik],
        [scorer.logPi(p), prior + lik], [batch[i], prior + lik]]) {
        assert.ok(Math.abs(actual - expected) < 1e-11, `${actual} != ${expected}`);
      }
    }
  }
});
