'use strict';

// A kernel body that draws a latent the prior does not carry (`mu` below) has
// a likelihood that integrates that latent out (spec §06 "Density of composed
// measures"). `buildLogPi` has no such marginal. The latent stayed unbound in
// the scoring env, `likWith` swallowed the unbound-reference error to
// -Infinity, and an MH run failed at initialisation with "supply finite
// initPositions", which names the wrong cause. The refusal now comes from the
// scorer's setup and names the latent.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const materialiser = require('../materialiser.ts');

const LATENT = `
theta ~ Normal(0, 1)
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), theta = theta)
`;
const FORMS: Record<string, string> = {
  direct: 'L = likelihoodof(K, record(y = 0.5))\nprior = lawof(theta)\nposterior = bayesupdate(L, prior)\n',
  adapter: 'L = likelihoodof(pars -> K(theta = pars.theta), record(y = 0.5))\n'
    + 'prior = lawof(record(theta = theta))\nposterior = bayesupdate(L, prior)\n',
};
const logN = (x: number, m: number, s: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

for (const [form, tail] of Object.entries(FORMS)) {
  test(`${form}: buildLogPi refuses a latent the prior does not carry, naming it`, async () => {
    const src = LATENT + tail;
    assert.deepEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
    const { ctx } = ctxFor(src, 4);
    await assert.rejects(() => buildLogPi(ctx, ctx.derivations.posterior), (e: any) => {
      assert.match(e.message, /the likelihood marginalises the stochastic ancestor\(s\) mu\b/);
      assert.match(e.message, /spec §06/);
      assert.equal(e.modelRefusal, true);
      return true;
    });
  });

  test(`${form}: an MH run refuses with the latent, not with an initialisation failure`, async () => {
    const { ctx } = ctxFor(LATENT + tail, 200);
    await assert.rejects(
      () => materialiser.materialiseMeasure('posterior', ctx,
        { backend: 'mh', chains: 2, warmup: 50, draws: 50, seed: 1 }),
      (e: any) => {
        assert.match(e.message, /marginalises the stochastic ancestor\(s\) mu\b/);
        assert.doesNotMatch(e.message, /initPositions/);
        return true;
      });
  });
}

test('a latent the prior carries scores the closed form', async () => {
  const src = `
mu ~ Normal(0, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), mu = mu)
L = likelihoodof(K, record(y = 0.5))
prior = lawof(mu)
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ mu: 0.3 }) - logN(0.5, 0.3, 1)) < 1e-12);
});

test('a draw independent of the boundary is a variate, not a latent, and still scores', async () => {
  const src = `
theta ~ Normal(0, 1)
z ~ Normal(1, 2)
w ~ Normal(theta, 1)
K = kernelof(record(w = w, z = z), theta = theta)
L = likelihoodof(K, record(w = 0.5, z = 0.7))
prior = lawof(theta)
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ theta: 0.3 }) - (logN(0.5, 0.3, 1) + logN(0.7, 1, 2))) < 1e-12);
});
