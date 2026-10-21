'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

test('normalized mixtures of iid laws retain one shared component choice', async () => {
  const { ctx } = ctxFor(`
p ~ Uniform(unitinterval)
counts(n) = iid(Binomial(n, p), 2)
sizes = [2, 3]
weights = [1, 1]
y ~ normalize(ksuperpose(counts, weights)(sizes))
K = kernelof(y, p = p)
L = likelihoodof(pars -> K(p = pars.p), [1, 2])
score = logdensityof(L, record(p = 0.4))
`, 1);
  // One common n: 1/2 * sum_n Binom(1 | n,p) Binom(2 | n,p).
  const expected = Math.log((0.48 * 0.16 + 0.432 * 0.288) / 2);
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0] - expected) < 1e-12);
});

test('normalizing an iid matrix cancels each live log mass in log space', async () => {
  const { ctx } = ctxFor(`
t ~ Uniform(interval(900, 1100))
y ~ normalize(iid(logweighted(t, Normal(0, 1)), [2, 2]))
K = kernelof(y, t = t)
L = likelihoodof(pars -> K(t = pars.t), rowstack([[0, 1], [2, 3]]))
score = logdensityof(L, record(t = 1000))
`, 1);
  const expected = -2 * Math.log(2 * Math.PI) - 7;
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0] - expected) < 1e-10);
});
