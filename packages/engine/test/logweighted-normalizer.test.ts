'use strict';

// logweighted(f,M) and weighted(exp(f),M) define the same measure (§06).
// Exponential tilting of Normal(0,1) by exp(x) yields Normal(1,1).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

test('fixed callable logweights use the same deterministic normalizer as weights', async () => {
  const exact = -0.5 * Math.log(2 * Math.PI) - 0.5 * 0.5 ** 2;
  for (const count of [1, 32]) {
    const { proc, ctx } = ctxFor(`flatppl_compat = "0.1"
m = normalize(logweighted(fn(_), Normal(0.0, 1.0)))
lp = logdensityof(m, 0.5)
`, count);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const got = (await ctx.getMeasure('lp')).samples[0];
    assert.ok(Math.abs(got - exact) < 1e-8, `${got} versus ${exact} at N=${count}`);
  }
});

test('MCMC uses the same fixed leaf normalizer for both weight spellings', async () => {
  const exact = -0.5 * Math.log(2 * Math.PI) - 0.5 * 0.5 ** 2;
  for (const weight of ['weighted(fn(exp(_))', 'logweighted(fn(_)']) {
    const { proc, ctx } = ctxFor(`flatppl_compat = "0.1"
theta ~ Uniform(interval(0.5, 2.0))
m = normalize(${weight}, Normal(0.0, 1.0)))
y ~ m
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = 0.5))
posterior = bayesupdate(L, lawof(theta))
`, 1);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const deriv = Object.values(ctx.derivations).find((d: any) => d.kind === 'bayesupdate');
    const { likOf } = await buildLogPi(ctx, deriv);
    for (const theta of [0.5, 2.0]) {
      const got = likOf({ theta });
      assert.ok(Math.abs(got - exact) < 1e-8, `${got} versus ${exact}`);
    }
  }
});

test('normalization cancels large positive and negative logweight offsets', async () => {
  const exact = -0.5 * Math.log(2 * Math.PI) - 0.5 * 0.5 ** 2;
  for (const offset of [709, 710, -740]) {
    const { ctx } = ctxFor(`flatppl_compat = "0.1"
theta ~ Uniform(interval(0.5, 2.0))
m = normalize(logweighted(fn(${offset}.0 + _), Normal(0.0, 1.0)))
lp = logdensityof(m, 0.5)
y ~ m
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = 0.5))
posterior = bayesupdate(L, lawof(theta))
`, 1);
    const got = (await ctx.getMeasure('lp')).samples[0];
    assert.ok(Math.abs(got - exact) < 1e-8, `${got} versus ${exact} at offset=${offset}`);
    const deriv = Object.values(ctx.derivations).find((d: any) => d.kind === 'bayesupdate');
    const { likOf } = await buildLogPi(ctx, deriv);
    assert.ok(Math.abs(likOf({ theta: 1 }) - exact) < 1e-8);
  }
});

test('logweight scaling permits a zero weight at the median', async () => {
  // E[X²]=1 for standard Normal, so x²*phi(x) already has mass one.
  const exact = -0.5 * Math.log(2 * Math.PI) - 0.5 * 0.5 ** 2 + Math.log(0.25);
  const { ctx } = ctxFor(`flatppl_compat = "0.1"
m = normalize(logweighted(x -> -740.0 + log(x*x), Normal(0.0, 1.0)))
lp = logdensityof(m, 0.5)
`, 1);
  const got = (await ctx.getMeasure('lp')).samples[0];
  assert.ok(Math.abs(got - exact) < 1e-8, `${got} versus ${exact}`);
});
