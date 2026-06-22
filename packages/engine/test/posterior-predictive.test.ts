'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./density/regression-baseline.test.ts');
const pp = require('../posterior-predictive.ts');

// Conjugate Normal: prior mu~Normal(0,2), y_i~Normal(mu,1), n=5.
const Y = [1.0, 2.0, 0.5, 1.5, 1.2];
const SRC = `
y_data = [1.0, 2.0, 0.5, 1.5, 1.2]
mu ~ Normal(0.0, 2.0)
prior = lawof(record(mu = mu))
y ~ iid(Normal(mu, 1.0), 5)
forward_kernel = kernelof(record(y = y), mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

function closedForm() {
  const m0 = 0, s0 = 2, sigma = 1, n = Y.length;
  const ybar = Y.reduce((a, b) => a + b, 0) / n;
  const sn2 = 1 / (1 / (s0 * s0) + n / (sigma * sigma));
  const mn = sn2 * (m0 / (s0 * s0) + (n * ybar) / (sigma * sigma));
  return { mean: mn, std: Math.sqrt(sn2 + sigma * sigma) };
}

function weightedMeanStd(samples: any, logWeights: any) {
  const n = samples.length;
  const lw = logWeights || new Array(n).fill(0);
  let mx = -Infinity; for (const v of lw) if (v > mx) mx = v;
  let Z = 0; for (const v of lw) Z += Math.exp(v - mx);
  let mean = 0; for (let i = 0; i < n; i++) mean += (Math.exp(lw[i] - mx) / Z) * samples[i];
  let varSum = 0; for (let i = 0; i < n; i++) { const d = samples[i] - mean; varSum += (Math.exp(lw[i] - mx) / Z) * d * d; }
  return { mean, std: Math.sqrt(varSum) };
}

test('posterior-predictive matches the closed-form conjugate-Normal predictive', async () => {
  const { ctx } = ctxFor(SRC, 20000);
  let d = null; for (const n of Object.keys(ctx.derivations)) if (ctx.derivations[n] && ctx.derivations[n].kind === 'bayesupdate') d = ctx.derivations[n];
  assert.ok(d, 'bayesupdate derivation present');
  const posterior = await ctx.getMeasure('posterior');

  const ppc = await pp.buildPosteriorPredictive(d, ctx, posterior);
  assert.ok(ppc && ppc.fields && ppc.fields.y, 'PPC built for field y');
  assert.deepEqual(ppc.fields.y.observed, Y, 'observed vector resolved');

  const yRep = ppc.fields.y.yRep;
  assert.ok(yRep.samples.length >= 20000, 'y_rep pools draws × replicates');
  const got = weightedMeanStd(yRep.samples, yRep.logWeights);
  const want = closedForm();   // mean ≈ 1.181, std ≈ 1.091
  assert.ok(Math.abs(got.mean - want.mean) < 0.05, `predictive mean ${got.mean} vs ${want.mean}`);
  assert.ok(Math.abs(got.std - want.std) < 0.05, `predictive std ${got.std} vs ${want.std}`);
});

// Inline composite-likelihood model: locscale(StudentT) with FIXED nu (no
// heavy-tail explosion), so the IS-weighted predictive mean is stable.
const COMP = `
y_data = [2.0, -1.0, 3.0, 0.5, -2.0]
mu ~ Normal(0.0, 5.0)
sigma ~ Uniform(interval(0.5, 5.0))
nu = 8.0
prior = lawof(record(mu = mu, sigma = sigma))
y ~ iid(locscale(StudentT(nu), mu, sigma), 5)
forward_kernel = kernelof(record(y = y), mu = mu, sigma = sigma)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

test('PPC handles a composite (locscale StudentT) likelihood, weighted predictive near the data centre', async () => {
  const { ctx } = ctxFor(COMP, 20000);
  let d = null; for (const n of Object.keys(ctx.derivations)) if (ctx.derivations[n] && ctx.derivations[n].kind === 'bayesupdate') d = ctx.derivations[n];
  const posterior = await ctx.getMeasure('posterior');
  const ppc = await pp.buildPosteriorPredictive(d, ctx, posterior);
  assert.ok(ppc && ppc.fields && ppc.fields.y, 'PPC built for composite field y');
  assert.equal(ppc.fields.y.observed.length, 5);
  const { mean } = weightedMeanStd(ppc.fields.y.yRep.samples, ppc.fields.y.yRep.logWeights);
  // data centre = 0.5; weighted posterior-predictive mean should land near it.
  assert.ok(mean > -2 && mean < 3, `composite weighted predictive mean ${mean} near data centre 0.5`);
});

test('buildPosteriorPredictive returns null when body is not a decomposable record', async () => {
  // A scalar-observation likelihood (no record body) → null, not a throw.
  const src = `
mu ~ Normal(0.0, 2.0)
prior = lawof(record(mu = mu))
y ~ Normal(mu, 1.0)
forward_kernel = kernelof(y, mu = mu)
L = likelihoodof(forward_kernel, 1.3)
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 100);
  let d = null; for (const n of Object.keys(ctx.derivations)) if (ctx.derivations[n] && ctx.derivations[n].kind === 'bayesupdate') d = ctx.derivations[n];
  const posterior = await ctx.getMeasure('posterior');
  const ppc = await pp.buildPosteriorPredictive(d, ctx, posterior);
  assert.equal(ppc, null);
});
