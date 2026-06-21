'use strict';
// nested-iid-broadcast-latent.test.ts — a NESTED iid-of-broadcast latent
// `beta ~ iid(Normal.(mu, sigma), L)` is an [L, D] matrix (L outer iid copies of
// the D-vector the broadcast `Normal.(mu,sigma)` produces). Two bugs blocked it:
//   1. matIid's composite-inner fallback recorded the value shape as [N, ...L]
//      (dropping the inner broadcast's D axis) → shape [N,L] disagreed with the
//      N·L·D data length → flattenNestedVariate mis-counted the inner axis.
//   2. flattenNestedVariate handled a multi-axis block only as a nested JS Array;
//      a flat [L,D] Value/Float64Array (the scorer's flatToScorerPt form) fell
//      through to "expected Array at depth 0, got object".
// After both: the materialised value is [N, L, D] and the prior density scores.
//
// Oracle (Distributions.jl, independent): at mu=[0.3,-0.5], sigma=[1.2,0.8],
// beta=[[-0.2,0.4],[0.1,-0.6],[0.5,0.0]] (beta[l,d] ~ Normal(mu[d], sigma[d])),
//   Σ_d logpdf(Normal(0,100), mu_d) + Σ_d logpdf(Exponential(1), sigma_d)
//     + Σ_{l,d} logpdf(Normal(mu_d, sigma_d), beta_{l,d}) = -19.389920487386135.

const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const TOL = 1e-9;
const SRC = `
mu ~ iid(Normal(0, 100), 2)
sigma ~ iid(Exponential(1), 2)
beta ~ iid(Normal.(mu, sigma), 3)
prior = lawof(record(mu = mu, sigma = sigma, beta = beta))
k = (b_row) -> Normal(sum(b_row), 1)
z ~ k.(beta)
forward_kernel = kernelof(record(z = z), mu = mu, sigma = sigma, beta = beta)
L_model = likelihoodof(forward_kernel, record(z = [0.0, 0.0, 0.0]))
posterior = bayesupdate(L_model, prior)
`;
function postDeriv(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') return v;
  }
  return null;
}

test('iid-of-broadcast latent materialises as [N, L, D]', async () => {
  const { ctx } = ctxFor(SRC, 4);
  const m = await ctx.getMeasure('beta');
  assert.deepEqual(m.value.shape, [4, 3, 2], `beta shape: ${JSON.stringify(m.value.shape)}`);
  assert.equal(m.value.data.length, 24, `beta data length: ${m.value.data.length}`);
});

test('prior density of an [L,D] matrix latent matches the Distributions.jl oracle (1e-9)', async () => {
  const { ctx } = ctxFor(SRC, 1);
  const { priorOf } = await buildLogPi(ctx, postDeriv(ctx));
  const pt = {
    mu: Float64Array.from([0.3, -0.5]),
    sigma: Float64Array.from([1.2, 0.8]),
    beta: { shape: [3, 2], data: Float64Array.from([-0.2, 0.4, 0.1, -0.6, 0.5, 0.0]) },
  };
  const got = priorOf(pt);
  assert.ok(Math.abs(got - (-19.389920487386135)) <= TOL, `priorOf: got ${got}, oracle -19.389920487386135`);
});
