'use strict';

// Closed-form regression tests for the measure-algebra ops that have a
// tractable analytical reference distribution. Existing test files cover
// individual primitives (jointchain density, chain MC marginal density,
// pushfwd-as-LogNormal, etc.); this file adds the *cross-component
// sample-statistics* and *posterior-importance* checks that pin
// the sampling path against the closed-form measure-theoretic answer.
//
// Each test uses a small Normal-only model so the analytical reference
// is unambiguous. Tolerances are 3σ-ish at SAMPLE_COUNT = 8192.
//
// Coverage:
//   - joint(M1, M2) — components independent ⇒ empirical Cov ≈ 0
//   - jointchain(prior, K) — variates (θ, y) with y|θ ~ N(θ, 1):
//       Cov(θ, y) = Var(θ),  Var(y) = Var(θ) + 1
//   - bayesupdate(L, prior) — Normal-Normal conjugate posterior
//       prior θ ~ N(0, 1), likelihood N(y_obs; θ, 1), y_obs = 2
//       ⇒ posterior θ ~ N(1, 0.5).  Importance-weighted moments match.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker');

const SAMPLE_COUNT = 8192;
const ROOT_SEED    = 0xC10D5F;  // distinct from other test files

function makeCtx(source) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// =====================================================================
// Statistics helpers — operate on Float64Array samples, optional logW
// =====================================================================

function unweightedMean(xs) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function unweightedVar(xs) {
  const m = unweightedMean(xs);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - m) * (xs[i] - m);
  return s / xs.length;
}

function unweightedCov(xs, ys) {
  const mx = unweightedMean(xs);
  const my = unweightedMean(ys);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - mx) * (ys[i] - my);
  return s / xs.length;
}

// Weighted statistics with logWeights normalised by their logSumExp.
function weightedMean(xs, logW) {
  if (!logW) return unweightedMean(xs);
  let lse = -Infinity;
  for (let i = 0; i < logW.length; i++) {
    if (logW[i] > lse) lse = logW[i];
  }
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = Math.exp(logW[i] - lse);
    num += w * xs[i];
    den += w;
  }
  return num / den;
}

function weightedVar(xs, logW) {
  if (!logW) return unweightedVar(xs);
  const m = weightedMean(xs, logW);
  let lse = -Infinity;
  for (let i = 0; i < logW.length; i++) {
    if (logW[i] > lse) lse = logW[i];
  }
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    const w = Math.exp(logW[i] - lse);
    num += w * (xs[i] - m) * (xs[i] - m);
    den += w;
  }
  return num / den;
}

// =====================================================================
// joint(M1, M2): independent product ⇒ cross-covariance vanishes
// =====================================================================

test('joint positional: components independent — empirical Cov(M1, M2) ≈ 0', async () => {
  // Closed form: M1 ⊗ M2 has Cov(X, Y) = 0 by construction. With
  // SAMPLE_COUNT = 8192, the SE of the sample covariance under iid
  // unit-variance Gaussians is roughly 1/√N ≈ 0.011 — 3σ ≈ 0.035.
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 0.0, sigma = 1.0)
J  = joint(M1, M2)
`);
  const J = await ctx.getMeasure('J');
  assert.ok(J.elems && J.elems.length === 2, 'tuple measure with two elems');
  const xs = J.elems[0].samples;
  const ys = J.elems[1].samples;
  // Marginals: each ~ N(0, 1).
  assert.ok(Math.abs(unweightedMean(xs)) < 0.05,
    'M1 marginal mean ≈ 0, got ' + unweightedMean(xs));
  assert.ok(Math.abs(unweightedMean(ys)) < 0.05,
    'M2 marginal mean ≈ 0, got ' + unweightedMean(ys));
  assert.ok(Math.abs(unweightedVar(xs) - 1) < 0.10,
    'M1 marginal var ≈ 1, got ' + unweightedVar(xs));
  assert.ok(Math.abs(unweightedVar(ys) - 1) < 0.10,
    'M2 marginal var ≈ 1, got ' + unweightedVar(ys));
  // Independence: cross-covariance vanishes.
  const cov = unweightedCov(xs, ys);
  assert.ok(Math.abs(cov) < 0.05,
    'joint should produce independent components ⇒ Cov ≈ 0, got ' + cov);
});

// =====================================================================
// jointchain(prior, K): variates (θ, y) with y = θ + ε, ε ~ N(0, 1)
// ⇒ Cov(θ, y) = Var(θ) = 1, Var(y) = 2, Var(θ) = 1
// =====================================================================

test('jointchain 2-arg: Cov(θ, y) = Var(θ), Var(y) = Var(θ) + 1 (closed-form)', async () => {
  // jointchain pattern:  θ ~ N(0,1);  y | θ ~ N(θ, 1).
  // Marginal y is N(0, √2), so Var(y) = 2. By construction
  // y = θ + ε with ε independent of θ, ε ~ N(0,1), giving
  //   Cov(θ, y) = Var(θ) = 1
  //   Corr(θ, y) = 1 / √2 ≈ 0.7071
  const ctx = makeCtx(`
theta = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta = theta))
obs_dist = joint(y = Normal(mu = theta, sigma = 1.0))
K = functionof(obs_dist, theta = theta)
joint_model = jointchain(prior, K)
`);
  const m = await ctx.getMeasure('joint_model');
  assert.ok(m.fields, 'joint_model materialises as record measure');
  const thetas = m.fields.theta.samples;
  const ys     = m.fields.y.samples;
  // Marginal moments.
  assert.ok(Math.abs(unweightedMean(thetas)) < 0.05,
    'E[θ] ≈ 0, got ' + unweightedMean(thetas));
  assert.ok(Math.abs(unweightedMean(ys))     < 0.06,
    'E[y] ≈ 0, got ' + unweightedMean(ys));
  const varTheta = unweightedVar(thetas);
  const varY     = unweightedVar(ys);
  assert.ok(Math.abs(varTheta - 1) < 0.10,
    'Var(θ) ≈ 1, got ' + varTheta);
  assert.ok(Math.abs(varY - 2) < 0.20,
    'Var(y) = Var(θ) + 1 ≈ 2 (Normal(0, √2) marginal), got ' + varY);
  // Cross-covariance — the core jointchain identity.
  const cov = unweightedCov(thetas, ys);
  assert.ok(Math.abs(cov - 1) < 0.10,
    'Cov(θ, y) = Var(θ) ≈ 1 by jointchain structure, got ' + cov);
});

// =====================================================================
// bayesupdate: Normal-Normal conjugate posterior
// =====================================================================

test('bayesupdate: Normal-Normal conjugate ⇒ posterior N(1, 0.5) at y_obs = 2', async () => {
  // Closed-form conjugate prior:
  //   θ ~ N(μ₀=0, σ₀²=1),  y | θ ~ N(θ, σ²=1),  y_obs = 2
  //   σ²_post = (σ² σ₀²) / (σ² + σ₀²) = 0.5
  //   μ_post  = σ²_post · (μ₀/σ₀² + y_obs/σ²) = 0.5 · 2 = 1.0
  // Importance sampling reweights prior atoms by L(θ_i) = N(y_obs; θ_i, 1):
  // the weighted mean and variance of the prior atoms converge to
  // μ_post and σ²_post.
  //
  // Tolerances are forgiving because importance sampling with prior =
  // proposal can be high-variance when the posterior is concentrated
  // relative to the prior (n_eff < N). With σ_prior = σ_lik = 1 the
  // overlap is decent, but we still budget ~10% absolute error.
  // Use the same lawof(record(...))+functionof(joint(y=...),...) pattern
  // as the bayesian_inference fixtures — that's the shape the bayesupdate
  // classifier recognises today (record-shaped prior + record-shaped obs).
  const ctx = makeCtx(`
mu = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(mu = mu))
obs_dist = joint(y = Normal(mu = mu, sigma = 1.0))
K = functionof(obs_dist, mu = mu)
L = likelihoodof(K, record(y = 2.0))
posterior = bayesupdate(L, prior)
`);
  const post = await ctx.getMeasure('posterior');
  assert.ok(post.fields && post.fields.mu, 'posterior should be a record measure with mu field');
  assert.ok(post.logWeights, 'posterior atoms should carry logWeights');

  const mus = post.fields.mu.samples;
  const lw  = post.logWeights;
  // Weighted moments against analytical posterior N(1, 0.5).
  const muHat   = weightedMean(mus, lw);
  const varHat  = weightedVar(mus, lw);
  assert.ok(Math.abs(muHat - 1.0) < 0.10,
    'posterior mean μ_post = 1.0, got ' + muHat);
  assert.ok(Math.abs(varHat - 0.5) < 0.10,
    'posterior variance σ²_post = 0.5, got ' + varHat);

  // n_eff should be a meaningful fraction of N — sanity check that
  // the reweighting wasn't degenerate.
  assert.ok(post.n_eff > SAMPLE_COUNT * 0.3,
    'n_eff > 30% of N (reasonable IS overlap), got ' + post.n_eff);
});
