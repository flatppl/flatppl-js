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

test('PPC covers a bare (non-record) forward output — single predictive field', async () => {
  // A scalar-observation likelihood whose forward output is a BARE variate `y`
  // (not a record). The old shape-decoder returned null here; the materialiser
  // path produces a single-field PPC. Conjugate check: mu~Normal(0,2), one
  // observation y=1.3, sigma=1 → posterior-predictive mean ≈ 1.04.
  const src = `
mu ~ Normal(0.0, 2.0)
prior = lawof(record(mu = mu))
y ~ Normal(mu, 1.0)
forward_kernel = kernelof(y, mu = mu)
L = likelihoodof(forward_kernel, 1.3)
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 20000);
  let d = null; for (const n of Object.keys(ctx.derivations)) if (ctx.derivations[n] && ctx.derivations[n].kind === 'bayesupdate') d = ctx.derivations[n];
  assert.ok(d, 'bayesupdate derivation present');
  const posterior = await ctx.getMeasure('posterior');
  const ppc = await pp.buildPosteriorPredictive(d, ctx, posterior);
  assert.ok(ppc && ppc.fields, 'PPC built for a bare forward output');
  const names = Object.keys(ppc.fields);
  assert.equal(names.length, 1, 'single predictive field');
  const f = ppc.fields[names[0]];
  assert.deepEqual(f.observed, [1.3], 'scalar observed resolved');
  const s = f.yRep.samples;
  assert.ok(s.length > 0 && Array.from(s).every(Number.isFinite), 'finite y_rep');
  const { mean } = weightedMeanStd(s, f.yRep.logWeights);
  assert.ok(Math.abs(mean - 1.04) < 0.1, `bare-output predictive mean ${mean} vs ~1.04`);
});

test('PPC handles locscale with a named fixed-phase shift constant', async () => {
  // Regression: the forward-body evaluateN path must inject fixed-phase constants
  // (like shift0=2.0) via addFixedRefArrays, not just posterior params.
  // Without the fix, shift0 is absent from fwdRefArrays, the worker evaluates
  // the bijection body against an empty ref for shift0, and returns NaN.
  const src = `
y_data = [3.0, 1.0, 4.0, 2.0, 3.5]
shift0 = 2.0
nu = 8.0
sigma ~ Uniform(interval(0.5, 5.0))
prior = lawof(record(sigma = sigma))
y ~ iid(locscale(StudentT(nu), shift0, sigma), 5)
forward_kernel = kernelof(record(y = y), sigma = sigma)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 8000);
  let d = null; for (const n of Object.keys(ctx.derivations)) if (ctx.derivations[n] && ctx.derivations[n].kind === 'bayesupdate') d = ctx.derivations[n];
  assert.ok(d, 'bayesupdate derivation present');
  const posterior = await ctx.getMeasure('posterior');
  const ppc = await pp.buildPosteriorPredictive(d, ctx, posterior);
  assert.ok(ppc && ppc.fields && ppc.fields.y, 'PPC built with a fixed-const shift');
  const s = ppc.fields.y.yRep.samples;
  assert.ok(s.length > 0, 'y_rep samples present');
  // Every finite sample must be a real number (not NaN-poisoned by the dropped const).
  // Without the fix, shift0 is missing from the forward-body refArrays and the
  // worker returns NaN for each sample.
  const finite = Array.from(s).filter(Number.isFinite);
  assert.ok(finite.length > s.length * 0.9, `most y_rep finite (got ${finite.length}/${s.length})`);
});

test('PPC covers a superpose-mixture observation likelihood (old shape-decoder declined it)', async () => {
  // A superpose (additive mixture) observation model. The old builder only
  // handled leaf + locscale-pushfwd inner dists, so it returned null for a
  // superpose; the materialiser path samples the mixture generically. This is
  // the generality win — a PPC for a model built with the standard construct
  // whose inner distribution is a composite the whitelist did not cover.
  const src = `
y_data = [0.5, 3.2, 1.1, 4.0, 0.8]
mu ~ Normal(0.0, 5.0)
prior = lawof(record(mu = mu))
y ~ iid(superpose(Normal(mu, 1.0), Normal(mu + 3.0, 1.0)), 5)
forward_kernel = kernelof(record(y = y), mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 8000);
  let d = null; for (const n of Object.keys(ctx.derivations)) if (ctx.derivations[n] && ctx.derivations[n].kind === 'bayesupdate') d = ctx.derivations[n];
  assert.ok(d, 'bayesupdate derivation present');
  const posterior = await ctx.getMeasure('posterior');
  const ppc = await pp.buildPosteriorPredictive(d, ctx, posterior);
  assert.ok(ppc && ppc.fields && ppc.fields.y, 'PPC built for a superpose-mixture likelihood');
  const s = ppc.fields.y.yRep.samples;
  assert.ok(s.length > 0, 'y_rep present');
  const arr = Array.from(s) as number[];
  assert.ok(arr.every((v) => Number.isFinite(v)), 'all y_rep finite');
});
