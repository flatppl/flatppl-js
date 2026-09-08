'use strict';

// exp(x) times Normal(theta,1) has mass exp(theta + 1/2).
// A pooled divisor changes likelihood ratios. Until this mass is supported,
// both inference paths must refuse it instead of returning a wrong score.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const H = 'flatppl_compat = "0.1"\n';
const refusal = /normalize.*parameter-dependent.*mass/i;

for (const weight of ['weighted(fn(exp(_))', 'logweighted(fn(_)']) {
  const model = H + 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + `m = normalize(${weight}, Normal(theta, 1.0)))\n`
    + 'y ~ m\nK = kernelof(record(y = y), theta = theta)\n'
    + 'L = likelihoodof(K, record(y = 0.5))\n'
    + 'lp = logdensityof(L, 0.5)\nposterior = bayesupdate(L, lawof(theta))\n';

  test(`${weight}: likelihood refuses a pooled latent-base normalizer`, async () => {
    const { proc, ctx } = ctxFor(model, 32);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    await assert.rejects(async () => ctx.getMeasure('lp'), refusal);
  });

  test(`${weight}: MCMC refuses the same unsupported normalizer`, async () => {
    const { proc, ctx } = ctxFor(model, 32);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const deriv = Object.values(ctx.derivations).find((d: any) => d.kind === 'bayesupdate');
    assert.ok(deriv);
    await assert.rejects(async () => buildLogPi(ctx, deriv), refusal);
  });
}

test('a fixed application retains the exact tilted Normal density', async () => {
  const { proc, ctx } = ctxFor(H + `
theta ~ Uniform(interval(0.5, 2.0))
m = normalize(weighted(fn(exp(_)), Normal(theta, 1.0)))
K = functionof(m, theta = theta)
M = K(theta = 1.0)
lp = logdensityof(M, 0.5)
`, 32);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const got = (await ctx.getMeasure('lp')).samples[0];
  const exact = -0.5 * Math.log(2 * Math.PI) - 1.5 ** 2 / 2;
  assert.ok(Math.abs(got - exact) < 1e-8, `${got} versus ${exact}`);
});

test('a named weight boundary does not capture the stochastic module binding', async () => {
  const { proc, ctx } = ctxFor(H + `
z ~ Normal(0.0, 1.0)
w = functionof(0.0 * z, v = z)
m = normalize(logweighted(w, Normal(0.0, 1.0)))
lp = logdensityof(m, 0.5)
`, 32);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  // The weight's z is its variate, not the module draw. exp(0*z)=1,
  // so this fallback preserves Normal(0,1) and has exactly unit mass.
  const got = (await ctx.getMeasure('lp')).samples[0];
  const exact = -0.5 * Math.log(2 * Math.PI) - 0.5 ** 2 / 2;
  assert.ok(Math.abs(got - exact) < 1e-12, `${got} versus ${exact}`);
});

test('a weight helper free capture prevents a pooled normalizer', async () => {
  const { proc, ctx } = ctxFor(H + `
theta ~ Uniform(interval(0.5, 2.0))
h = z -> exp(theta * z[1])
m = normalize(weighted(v -> h(v), Dirichlet([2.0, 2.0])))
y ~ m
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = [0.5, 0.5]))
lp = logdensityof(L, 1.0)
`, 32);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  // The Dirichlet base is fixed. Only the helper's free theta changes
  // the integral, so checking the base parameters alone misses this case.
  await assert.rejects(async () => ctx.getMeasure('lp'), refusal);
});
