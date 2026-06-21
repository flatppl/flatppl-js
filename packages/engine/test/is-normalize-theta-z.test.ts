'use strict';
// Oracle (Distributions.jl): normalize(superpose(weighted(theta, N1),
// weighted(theta, N2))) has Z=2θ ⇒ density (N1+N2)/2, θ-INDEPENDENT. So the IS
// posterior mean of theta must equal its PRIOR mean (the likelihood does not move
// it): Beta(2,2) prior ⇒ mean 0.5. A constant-baked Z leaves a spurious +N·logθ
// that biases the IS weights toward high theta ⇒ IS weighted mean >> 0.5.
const { test }   = require('node:test');
const assert     = require('node:assert/strict');
const { ctxFor } = require('./density/regression-baseline.test.ts');

const SRC = `
theta ~ Beta(2, 2)
mu ~ iid(Normal(0, 5), 2)
mix = normalize(superpose(weighted(theta, Normal(mu[1], 1)), weighted(theta, Normal(mu[2], 1))))
y ~ iid(mix, 5)
prior = lawof(record(theta = theta, mu = mu))
forward_kernel = kernelof(record(y = y), theta = theta, mu = mu)
L = likelihoodof(forward_kernel, record(y = [0.1, -0.2, 0.3, 1.0, -1.0]))
posterior = bayesupdate(L, prior)
`;

// Compute the IS-weighted mean of theta from logWeights + prior samples.
// getMeasure('posterior') returns a recordMeasure whose fields hold the PRIOR
// samples and whose logWeights carry the IS reweighting.
function isMean(prior_samples: Float64Array, logWeights: Float64Array): number {
  let maxLw = -Infinity;
  for (let i = 0; i < logWeights.length; i++) if (logWeights[i] > maxLw) maxLw = logWeights[i];
  let sw = 0, swt = 0;
  for (let i = 0; i < logWeights.length; i++) {
    const w = Math.exp(logWeights[i] - maxLw);
    sw += w; swt += w * prior_samples[i];
  }
  return swt / sw;
}

test('IS posterior of theta is prior-mean (Z=2θ cancels) — per-θ normalizer', async () => {
  const { ctx } = ctxFor(SRC, 10000);
  const prior = await ctx.getMeasure('prior');
  const post  = await ctx.getMeasure('posterior');
  const theta_prior = prior.fields.theta.samples;
  const lw    = post.logWeights;
  const mu    = isMean(theta_prior, lw);
  // θ-independent likelihood ⇒ IS posterior == Beta(2,2) prior ⇒ mean 0.5.
  // The constant-Z bug biases to ~0.78 (Beta(7,2) effective prior).
  assert.ok(Math.abs(mu - 0.5) < 0.03, `IS weighted mean(theta)=${mu}, expected ≈0.5`);
});
