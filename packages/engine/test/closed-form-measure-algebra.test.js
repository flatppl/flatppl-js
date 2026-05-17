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

// =====================================================================
// superpose density (engine-concepts §11 — the discrete-selector
// `select` path). superpose is *additive* and *un-normalised*:
//   ν = Σ_k M_k  ⇒  p_ν(x) = Σ_k p_{M_k}(x)
// so logdensityof(superpose(...), x) = logsumexp_k logp_{M_k}(x),
// exactly (no Monte-Carlo, no −logN — the EXACT discrete sibling of
// the kchain MC marginal). Closed-form Normal references.
// =====================================================================

function normalLogpdf(x, mu, sigma) {
  return -Math.log(sigma) - 0.5 * Math.log(2 * Math.PI)
    - (x - mu) * (x - mu) / (2 * sigma * sigma);
}

test('superpose density: log p(x) = log[ p_A(x) + p_B(x) ] (raw additive)', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 4.0, sigma = 1.0)
S = superpose(A, B)
lp = logdensityof(S, 1.0)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(
    Math.exp(normalLogpdf(1.0, 0, 1)) + Math.exp(normalLogpdf(1.0, 4, 1)));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-10,
    `superpose logp: got ${m.samples[0]}, expected ${expected}`);
});

test('superpose density: superpose(m, m) = log 2 + logp_m (additivity)', async () => {
  const ctx = makeCtx(`
m = Normal(mu = 2.0, sigma = 0.5)
T = superpose(m, m)
lpt = logdensityof(T, 1.3)
lpm = logdensityof(m, 1.3)
`);
  const [T, M] = await Promise.all([ctx.getMeasure('lpt'), ctx.getMeasure('lpm')]);
  assert.ok(Math.abs(T.samples[0] - (Math.LN2 + M.samples[0])) < 1e-10,
    `expected log2 + logp_m = ${Math.LN2 + M.samples[0]}, got ${T.samples[0]}`);
  // And against the closed form directly.
  assert.ok(Math.abs(M.samples[0] - normalLogpdf(1.3, 2, 0.5)) < 1e-10);
});

test('superpose density: weighted summands ⇒ log Σ w_k p_k', async () => {
  // superpose(weighted(0.25, A), weighted(0.75, B)) at x=1.3.
  // Un-normalised: density = 0.25·p_A + 0.75·p_B (NOT divided by Σw —
  // that's what normalize() would do; superpose alone does not).
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 2.0)
S = superpose(weighted(0.25, A), weighted(0.75, B))
lp = logdensityof(S, 1.3)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(
    0.25 * Math.exp(normalLogpdf(1.3, 0, 1))
    + 0.75 * Math.exp(normalLogpdf(1.3, 5, 2)));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-10,
    `weighted superpose logp: got ${m.samples[0]}, expected ${expected}`);
});

test('superpose density: 3-component superpose sums all branches', async () => {
  const ctx = makeCtx(`
A = Normal(mu = -3.0, sigma = 1.0)
B = Normal(mu =  0.0, sigma = 0.5)
C = Normal(mu =  3.0, sigma = 2.0)
S = superpose(A, B, C)
lp = logdensityof(S, 0.4)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(
    Math.exp(normalLogpdf(0.4, -3, 1))
    + Math.exp(normalLogpdf(0.4, 0, 0.5))
    + Math.exp(normalLogpdf(0.4, 3, 2)));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-10,
    `3-component superpose logp: got ${m.samples[0]}, expected ${expected}`);
});

test('superpose density: totalmass additivity stays consistent with density', async () => {
  // superpose(weighted(2, m), weighted(3, m)) has total mass 2+3 = 5
  // (existing materialiser invariant) AND density = log[(2+3)·p_m] =
  // log 5 + logp_m — the two views must agree.
  const ctx = makeCtx(`
m  = Normal(mu = 0.0, sigma = 1.0)
S  = superpose(weighted(2.0, m), weighted(3.0, m))
lp = logdensityof(S, 0.6)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(5) + normalLogpdf(0.6, 0, 1);
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-10,
    `got ${m.samples[0]}, expected log5 + logp = ${expected}`);
});

// =====================================================================
// measure-valued ifelse density (engine-concepts §11). ifelse(c,a,b)
// with c ~ Bernoulli(p) is the 2-branch discrete-selector mixture;
// marginalising the (anonymous) selector gives the EXACT closed-form
//   log p(x) = log[ p·p_a(x) + (1−p)·p_b(x) ]
// =====================================================================

test('ifelse density: log[ p·p_A(x) + (1−p)·p_B(x) ] (Bernoulli selector)', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 2.0)
c = draw(Bernoulli(p = 0.3))
M = ifelse(c, A, B)
lp = logdensityof(M, 1.3)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(
    0.3 * Math.exp(normalLogpdf(1.3, 0, 1))
    + 0.7 * Math.exp(normalLogpdf(1.3, 5, 2)));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-10,
    `ifelse logp: got ${m.samples[0]}, expected ${expected}`);
});

test('ifelse density: p=0.5 ⇒ log[ ½(p_A + p_B) ]', async () => {
  const ctx = makeCtx(`
A = Normal(mu = -2.0, sigma = 1.0)
B = Normal(mu =  2.0, sigma = 1.0)
c = draw(Bernoulli(p = 0.5))
M = ifelse(c, A, B)
lp = logdensityof(M, 0.0)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(
    0.5 * Math.exp(normalLogpdf(0.0, -2, 1))
    + 0.5 * Math.exp(normalLogpdf(0.0, 2, 1)));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-10,
    `got ${m.samples[0]}, expected ${expected}`);
});

test('ifelse density: identical branches ⇒ logp_m for ANY p (invariant)', async () => {
  // ifelse(c, m, m): mixture = p·p_m + (1−p)·p_m = p_m regardless of p.
  const ctx = makeCtx(`
m = Normal(mu = 1.0, sigma = 0.7)
c = draw(Bernoulli(p = 0.137))
M = ifelse(c, m, m)
lp  = logdensityof(M, 0.4)
lpm = logdensityof(m, 0.4)
`);
  const [M, Mm] = await Promise.all([ctx.getMeasure('lp'), ctx.getMeasure('lpm')]);
  assert.ok(Math.abs(M.samples[0] - Mm.samples[0]) < 1e-10,
    `ifelse(c,m,m) should equal logp_m: got ${M.samples[0]} vs ${Mm.samples[0]}`);
  assert.ok(Math.abs(Mm.samples[0] - normalLogpdf(0.4, 1, 0.7)) < 1e-10);
});

test('ifelse density: degenerate p=1 ⇒ branch A; p=0 ⇒ branch B', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 9.0, sigma = 1.0)
cT = draw(Bernoulli(p = 1.0))
cF = draw(Bernoulli(p = 0.0))
MT = ifelse(cT, A, B)
MF = ifelse(cF, A, B)
lpT = logdensityof(MT, 0.5)
lpF = logdensityof(MF, 0.5)
`);
  const [T, F] = await Promise.all([ctx.getMeasure('lpT'), ctx.getMeasure('lpF')]);
  // p=1: log[1·p_A + 0·p_B] = logp_A (the −Inf branch drops out).
  assert.ok(Math.abs(T.samples[0] - normalLogpdf(0.5, 0, 1)) < 1e-10,
    `p=1 ⇒ logp_A, got ${T.samples[0]}`);
  assert.ok(Math.abs(F.samples[0] - normalLogpdf(0.5, 9, 1)) < 1e-10,
    `p=0 ⇒ logp_B, got ${F.samples[0]}`);
});

test('ifelse density ≡ superpose(weighted(p,A), weighted(1−p,B)) density', async () => {
  // Cross-construct consistency: ifelse and superpose ride the SAME
  // select core, so the two spellings of the same mixture must give
  // bit-comparable densities.
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 4.0, sigma = 1.5)
c = draw(Bernoulli(p = 0.4))
viaIf = ifelse(c, A, B)
viaSup = superpose(weighted(0.4, A), weighted(0.6, B))
lpIf  = logdensityof(viaIf, 2.1)
lpSup = logdensityof(viaSup, 2.1)
`);
  const [I, S] = await Promise.all([ctx.getMeasure('lpIf'), ctx.getMeasure('lpSup')]);
  assert.ok(Math.abs(I.samples[0] - S.samples[0]) < 1e-10,
    `ifelse=${I.samples[0]} vs superpose=${S.samples[0]} — shared core must agree`);
});
