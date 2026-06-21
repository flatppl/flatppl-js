'use strict';
// is-broadcast-vector-obs.test.ts — the IS reweighting path must score a
// kernel-broadcast-of-builtin likelihood (`y ~ Bernoulli.(p)`) over a VECTOR
// observation, where the per-observation parameter `p` derives from a per-atom
// latent. walkBroadcast computed `paramUsesAtom` from a STATIC scan of the
// refArrays names only; in the IS path the derived per-atom `p` is threaded via
// baseEnv, so the scan missed it → collectionAxesOf kept the leading N (atom)
// axis → axes=[N, Nobs] while the shared observed is [Nobs] → flattenNestedVariate
// "axis 0 wants dim=N but got Array of length <Nobs>". Fix: also treat a rank-≥2
// param value whose leading axis is the atom axis as atom-batched (unambiguous).
//
// Oracle (Distributions.jl, independent): the Bernoulli log-likelihood of the
// hierarchical-logistic model at mu=[0.3,-0.5], sigma=[1.2,0.8],
// beta=[[-0.2,0.4],[0.1,-0.6],[0.5,0.0]], with p_i = invlogit(x_i · beta[group_i])
// over the 9 observations, is -7.897129698035907.

const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const TOL = 1e-9;
const SRC = `
x_data = [[1.0,0.5],[1.0,-1.2],[1.0,0.3],[1.0,1.8],[1.0,-0.7],[1.0,2.1],[1.0,0.9],[1.0,-1.5],[1.0,0.2]]
ll_data = [1, 1, 1, 2, 2, 2, 3, 3, 3]
y_data  = [1, 0, 0, 1, 0, 1, 1, 0, 0]
mu ~ iid(Normal(0, 100), 2)
sigma ~ iid(Exponential(1), 2)
beta ~ iid(Normal.(mu, sigma), 3)
prior = lawof(record(mu = mu, sigma = sigma, beta = beta))
logit_p = (x_row, group) -> invlogit(sum(x_row .* get(beta, group)))
p = logit_p.(x_data, ll_data)
y ~ Bernoulli.(p)
forward_kernel = kernelof(record(y = y), mu = mu, sigma = sigma, beta = beta)
L_model = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L_model, prior)
`;
function postDeriv(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') return v;
  }
  return null;
}

test('broadcast-Bernoulli vector-observation likelihood matches the Distributions.jl oracle (1e-9)', async () => {
  const { ctx } = ctxFor(SRC, 1);
  const { likOf } = await buildLogPi(ctx, postDeriv(ctx));
  const pt = {
    mu: Float64Array.from([0.3, -0.5]),
    sigma: Float64Array.from([1.2, 0.8]),
    beta: { shape: [3, 2], data: Float64Array.from([-0.2, 0.4, 0.1, -0.6, 0.5, 0.0]) },
  };
  const got = likOf(pt);
  assert.ok(Math.abs(got - (-7.897129698035907)) <= TOL, `likOf: got ${got}, oracle -7.897129698035907`);
});

test('IS reweighting materialises the posterior (count=N broadcast over a shared vector observation)', async () => {
  const { ctx } = ctxFor(SRC, 200);   // default backend = IS
  const m = await ctx.getMeasure('posterior');
  assert.ok(m.fields && m.fields.beta, 'posterior has a beta field');
  const s = m.fields.beta.samples || (m.fields.beta.value && m.fields.beta.value.data);
  assert.ok(s && s.length > 0, 'beta field carries samples');
});
