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
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 8192;
const ROOT_SEED    = 0xC10D5F;  // distinct from other test files

function makeCtx(source: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
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

function unweightedMean(xs: any) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

function unweightedVar(xs: any) {
  const m = unweightedMean(xs);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - m) * (xs[i] - m);
  return s / xs.length;
}

function unweightedCov(xs: any, ys: any) {
  const mx = unweightedMean(xs);
  const my = unweightedMean(ys);
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += (xs[i] - mx) * (ys[i] - my);
  return s / xs.length;
}

// Weighted statistics with logWeights normalised by their logSumExp.
function weightedMean(xs: any, logW: any) {
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

function weightedVar(xs: any, logW: any) {
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

function normalLogpdf(x: any, mu: any, sigma: any) {
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

// ---- ifelse SAMPLING (matSelect gather) vs closed-form mixture -----
// X = ifelse(c~Bernoulli(p), A, B):
//   E[X]   = p·μ_A + (1−p)·μ_B
//   E[X²]  = p·(σ_A²+μ_A²) + (1−p)·(σ_B²+μ_B²)
//   Var[X] = E[X²] − E[X]²
//   P(X from branch A) = p
test('ifelse sampling: mixture mean / variance / branch fraction (closed-form)', async () => {
  const p = 0.3, muA = 0, sgA = 1, muB = 10, sgB = 2;
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 10.0, sigma = 2.0)
c = draw(Bernoulli(p = 0.3))
M = ifelse(c, A, B)
x = draw(M)
`);
  const m = await ctx.getMeasure('x');
  const xs = m.samples;
  const EX  = p * muA + (1 - p) * muB;                 // = 7
  const EX2 = p * (sgA * sgA + muA * muA)
            + (1 - p) * (sgB * sgB + muB * muB);
  const VX  = EX2 - EX * EX;
  const meanHat = unweightedMean(xs);
  const varHat  = unweightedVar(xs);
  // Branch-A fraction: A-mass sits near 0, B-mass near 10 → split @5.
  let nA = 0;
  for (let i = 0; i < xs.length; i++) if (xs[i] < 5) nA++;
  const fracA = nA / xs.length;
  assert.ok(Math.abs(meanHat - EX) < 0.15,
    `mixture mean: got ${meanHat}, expected ${EX}`);
  assert.ok(Math.abs(varHat - VX) / VX < 0.10,
    `mixture variance: got ${varHat}, expected ${VX}`);
  assert.ok(Math.abs(fracA - p) < 0.02,
    `branch-A fraction: got ${fracA}, expected p=${p}`);
});

test('ifelse sampling: p=1 ⇒ all branch A; p=0 ⇒ all branch B', async () => {
  const ctx = makeCtx(`
A = Normal(mu = -5.0, sigma = 0.5)
B = Normal(mu =  5.0, sigma = 0.5)
cT = draw(Bernoulli(p = 1.0))
cF = draw(Bernoulli(p = 0.0))
xT = draw(ifelse(cT, A, B))
xF = draw(ifelse(cF, A, B))
`);
  const [T, F] = await Promise.all([ctx.getMeasure('xT'), ctx.getMeasure('xF')]);
  assert.ok(Math.abs(unweightedMean(T.samples) - (-5)) < 0.1,
    `p=1 ⇒ branch A (μ=−5), got ${unweightedMean(T.samples)}`);
  assert.ok(Math.abs(unweightedMean(F.samples) - 5) < 0.1,
    `p=0 ⇒ branch B (μ=+5), got ${unweightedMean(F.samples)}`);
});

// ---- MC-weight ifelse: NON-closed-form selector condition ----------
// engine-concepts §11. The condition `c = z > 0` (z ~ Normal) is a
// {0,1} selector with NO closed-form P(true) — classifyIfelse can't
// resolve a Bernoulli p. The mixture is still STRUCTURALLY exact; the
// per-branch weight P(true) is estimated ONCE from the materialised
// selector ensemble (the general materialiser runtime-weight
// resolver) and substituted as a constant logweight. With z~N(0,1),
// P(z>0)=0.5, so the density must match the closed-form symmetric
// mixture to within the p̂ Monte-Carlo error (statistical, not exact).
test('MC-weight ifelse: non-closed-form condition density ≈ estimated mixture', async () => {
  const ctx = makeCtx(`
z = draw(Normal(mu = 0.0, sigma = 1.0))
c = z > 0.0
A = Normal(mu = -3.0, sigma = 1.0)
B = Normal(mu =  3.0, sigma = 1.0)
M = ifelse(c, A, B)
lp = logdensityof(M, 0.5)
`);
  const m = await ctx.getMeasure('lp');
  // True P(z>0) = 0.5; analytic symmetric mixture density at x=0.5.
  const expected = Math.log(
    0.5 * Math.exp(normalLogpdf(0.5, -3, 1))
    + 0.5 * Math.exp(normalLogpdf(0.5, 3, 1)));
  // p̂ SE ≈ 1/(2√N) ≈ 0.0055 (N=8192); local d(logp)/dp ≈ −1.8 here,
  // so 3σ on logp ≈ 0.03 — allow a comfortable 0.08 margin.
  assert.ok(Number.isFinite(m.samples[0])
    && Math.abs(m.samples[0] - expected) < 0.08,
    `MC-weight ifelse logp: got ${m.samples[0]}, expected ≈ ${expected}`);
});

test('MC-weight ifelse: density ≈ closed-form Bernoulli(0.5) ifelse (same structure)', async () => {
  // The non-closed-form-condition spelling and the closed-form
  // Bernoulli spelling are the SAME mixture; only the weight's
  // PROVENANCE differs (MC-estimated vs analytic). They must agree to
  // within the estimator error — proves the structure is exact and
  // only the weight is approximated.
  const ctx = makeCtx(`
z = draw(Normal(mu = 0.0, sigma = 1.0))
cMC = z > 0.0
cCF = draw(Bernoulli(p = 0.5))
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 4.0, sigma = 2.0)
lpMC = logdensityof(ifelse(cMC, A, B), 1.7)
lpCF = logdensityof(ifelse(cCF, A, B), 1.7)
`);
  const [MC, CF] = await Promise.all(
    [ctx.getMeasure('lpMC'), ctx.getMeasure('lpCF')]);
  assert.ok(Math.abs(MC.samples[0] - CF.samples[0]) < 0.05,
    `MC-weight=${MC.samples[0]} vs closed-form=${CF.samples[0]} `
    + '— same structure, weight only estimated');
});

test('MC-weight ifelse: sampling mixture mean / variance (closed-form)', async () => {
  // Sampling never needed P(true) (matSelect gathers by the realised
  // condition), so the generated mixture is exact regardless of the
  // density-weight estimation. z~N(0,1) ⇒ symmetric 50/50 split of
  // Normal(−3,1) and Normal(3,1): E[X]=0, Var[X]=0.5(1+9)·2/2=10.
  const ctx = makeCtx(`
z = draw(Normal(mu = 0.0, sigma = 1.0))
c = z > 0.0
A = Normal(mu = -3.0, sigma = 1.0)
B = Normal(mu =  3.0, sigma = 1.0)
x = draw(ifelse(c, A, B))
`);
  const xs = (await ctx.getMeasure('x')).samples;
  const meanHat = unweightedMean(xs);
  const varHat  = unweightedVar(xs);
  let nA = 0;
  for (let i = 0; i < xs.length; i++) if (xs[i] < 0) nA++;
  const fracA = nA / xs.length;
  assert.ok(Math.abs(meanHat - 0) < 0.15,
    `mixture mean: got ${meanHat}, expected 0`);
  assert.ok(Math.abs(varHat - 10) / 10 < 0.10,
    `mixture variance: got ${varHat}, expected 10`);
  assert.ok(Math.abs(fracA - 0.5) < 0.03,
    `branch-A fraction: got ${fracA}, expected 0.5`);
});

// =====================================================================
// retain mode (engine-concepts §11): when the selector binding is
// observed in an enclosing joint, walkSelect reads the threaded value
// and scores `walk(branch_k, x)` — picking the conditional, NOT
// logsumexp-ing the mixture. The selector's prior mass is paid by the
// joint's selector field separately (avoids double-counting).
// =====================================================================

test('retain mode: joint(c,ifelse) density = logBernoulli(c) + branch density', async () => {
  // Construction:
  //   c ~ Bernoulli(0.3); m = ifelse(c, A, B); joint(c=c, m=m)
  // Observed (c=1, m=0.5):
  //   density of c field = log P(c=1) = log 0.3
  //   density of m field (retain — c is in env): pick branch 0 (A
  //     since c=1 is truthy in K=2 Bernoulli ifelse), score
  //     logp_A(0.5). NO logsumexp, NO double-count of log P(c).
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 1.0)
c = draw(Bernoulli(p = 0.3))
m = ifelse(c, A, B)
jc = joint(c = c, m = m)
lp = logdensityof(jc, record(c = 1.0, m = 0.5))
`);
  const M = await ctx.getMeasure('lp');
  const expected = Math.log(0.3) + normalLogpdf(0.5, 0, 1);
  assert.ok(Math.abs(M.samples[0] - expected) < 1e-10,
    `retain joint logp: got ${M.samples[0]}, expected ${expected}`);
});

test('retain mode: c=0 picks branch B (Bernoulli false ⇒ second branch)', async () => {
  // Symmetric to the c=1 case: c=0 (falsy) routes to branch 1 (B).
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 1.0)
c = draw(Bernoulli(p = 0.3))
m = ifelse(c, A, B)
jc = joint(c = c, m = m)
lp = logdensityof(jc, record(c = 0.0, m = 5.2))
`);
  const M = await ctx.getMeasure('lp');
  const expected = Math.log(0.7) + normalLogpdf(5.2, 5, 1);
  assert.ok(Math.abs(M.samples[0] - expected) < 1e-10,
    `retain c=0 logp: got ${M.samples[0]}, expected ${expected}`);
});

test('retain mode: joint(i, xs[i]) — K-way Categorical (1-based)', async () => {
  // Categorical selector (1-based per FlatPPL spec), retain x's
  // conditional given the observed selector.
  //   i ~ Categorical([0.2, 0.3, 0.5]);
  //   xs = [draw(M1), draw(M2), draw(M3)];
  //   x = xs[i];
  //   joint(i=i, x=x) observed (i=2, x=4.5) ⇒ second branch (M2):
  //   density = log P(i=2) + logp_M2(4.5) = log 0.3 + logN(4.5; 5, 1).
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 5.0, sigma = 1.0)
M3 = Normal(mu = 9.0, sigma = 1.0)
i  = draw(Categorical(p = [0.2, 0.3, 0.5]))
xs = [draw(M1), draw(M2), draw(M3)]
x  = xs[i]
jc = joint(i = i, x = x)
lp = logdensityof(jc, record(i = 2.0, x = 4.5))
`);
  const M = await ctx.getMeasure('lp');
  const expected = Math.log(0.3) + normalLogpdf(4.5, 5, 1);
  assert.ok(Math.abs(M.samples[0] - expected) < 1e-10,
    `retain Categorical logp: got ${M.samples[0]}, expected ${expected}`);
});

test('retain mode: differs from marginalize (no enclosing joint ⇒ logsumexp)', async () => {
  // Same A, B, p — but observing m alone (no joint, no c in env) ⇒
  // marginalised mixture density. Verifies the retain branch in
  // walkSelect only fires when the selector is in scope.
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 1.0)
c = draw(Bernoulli(p = 0.3))
m = ifelse(c, A, B)
lp_marginal = logdensityof(m, 0.5)
`);
  const M = await ctx.getMeasure('lp_marginal');
  const expected = Math.log(
    0.3 * Math.exp(normalLogpdf(0.5, 0, 1))
    + 0.7 * Math.exp(normalLogpdf(0.5, 5, 1)));
  assert.ok(Math.abs(M.samples[0] - expected) < 1e-10,
    `marginalised logp: got ${M.samples[0]}, expected ${expected}`);
});

// =====================================================================
// normalized mixture (engine-concepts §11): the spec's canonical
//   mix = normalize(superpose(weighted(w1, M1), weighted(w2, M2)))
// normalize(M) is lowered to logweighted(−log Z, M) with CLOSED-FORM
// Z = Σ w_k. Probability mixture ⇒ Z=1 (0-shift no-op); an
// unnormalized base divides every atom by Z exactly.
// =====================================================================

test('normalized mixture: density = log[ w1·p_A + w2·p_B ] (Σw=1, Z=1)', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 2.0)
mix = normalize(superpose(weighted(0.25, A), weighted(0.75, B)))
lp = logdensityof(mix, 1.3)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(
    0.25 * Math.exp(normalLogpdf(1.3, 0, 1))
    + 0.75 * Math.exp(normalLogpdf(1.3, 5, 2)));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-10,
    `normalized mixture logp: got ${m.samples[0]}, expected ${expected}`);
});

test('normalized mixture: normalize(superpose(2·m, 3·m)) ≡ m (Z=5 divided out)', async () => {
  const ctx = makeCtx(`
m  = Normal(mu = 1.0, sigma = 0.7)
mn = normalize(superpose(weighted(2.0, m), weighted(3.0, m)))
lpn = logdensityof(mn, 0.4)
lpm = logdensityof(m, 0.4)
`);
  const [Mn, Mm] = await Promise.all([ctx.getMeasure('lpn'), ctx.getMeasure('lpm')]);
  assert.ok(Math.abs(Mn.samples[0] - Mm.samples[0]) < 1e-10,
    `normalize(2m+3m) must equal m: got ${Mn.samples[0]} vs ${Mm.samples[0]}`);
  assert.ok(Math.abs(Mm.samples[0] - normalLogpdf(0.4, 1, 0.7)) < 1e-10);
});

test('normalized mixture: integrates to 1 (trapezoid over a wide grid)', async () => {
  // ∫ p_mix(x) dx ≈ 1 for a proper normalized mixture. Evaluate the
  // closed-form density on a fine grid and trapezoid-integrate.
  const ctx = makeCtx(`
A = Normal(mu = -2.0, sigma = 0.8)
B = Normal(mu =  3.0, sigma = 1.3)
mix = normalize(superpose(weighted(0.4, A), weighted(0.6, B)))
`);
  // logdensityof at each grid point (one binding per point keeps the
  // harness simple; closed-form ⇒ exact, no sampling).
  const lo = -12, hi = 15, n = 6000, h = (hi - lo) / n;
  let src = `
A = Normal(mu = -2.0, sigma = 0.8)
B = Normal(mu =  3.0, sigma = 1.3)
mix = normalize(superpose(weighted(0.4, A), weighted(0.6, B)))
`;
  for (let i = 0; i <= n; i++) {
    src += `p${i} = logdensityof(mix, ${(lo + i * h).toFixed(6)})\n`;
  }
  const c2 = makeCtx(src);
  let integral = 0;
  for (let i = 0; i <= n; i++) {
    const lp = (await c2.getMeasure('p' + i)).samples[0];
    const w = (i === 0 || i === n) ? 0.5 : 1.0;
    integral += w * Math.exp(lp);
  }
  integral *= h;
  assert.ok(Math.abs(integral - 1.0) < 2e-3,
    `normalized mixture must integrate to 1, got ${integral}`);
});

// =====================================================================
// stochastic-phase array indexing (engine-concepts §11) — the draw-
// style spelling of a discrete mixture:
//   i ~ Categorical(w); xs = [draw(M1),…]; x = xs[i]
// recognised onto the SAME select core. Density (selector
// marginalised) = logsumexp_k(log w_k + logp_{M_k}); sampling gathers
// branch i per atom. Categorical is spec 1-based.
// =====================================================================

test('xs[i] density: K-component categorical mixture (closed-form)', async () => {
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 5.0, sigma = 1.0)
M3 = Normal(mu = 9.0, sigma = 1.0)
i  = draw(Categorical(p = [0.2, 0.3, 0.5]))
xs = [draw(M1), draw(M2), draw(M3)]
x  = xs[i]
lp = logdensityof(x, 4.0)
`);
  const m = await ctx.getMeasure('lp');
  const expected = Math.log(
    0.2 * Math.exp(normalLogpdf(4.0, 0, 1))
    + 0.3 * Math.exp(normalLogpdf(4.0, 5, 1))
    + 0.5 * Math.exp(normalLogpdf(4.0, 9, 1)));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-9,
    `xs[i] mixture logp: got ${m.samples[0]}, expected ${expected}`);
});

test('xs[i] ≡ explicit superpose(weighted…) density (shared core)', async () => {
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 5.0, sigma = 2.0)
i  = draw(Categorical(p = [0.35, 0.65]))
xs = [draw(M1), draw(M2)]
viaIdx = xs[i]
viaSup = normalize(superpose(weighted(0.35, M1), weighted(0.65, M2)))
lpIdx = logdensityof(viaIdx, 2.4)
lpSup = logdensityof(viaSup, 2.4)
`);
  const [I, S] = await Promise.all([ctx.getMeasure('lpIdx'), ctx.getMeasure('lpSup')]);
  assert.ok(Math.abs(I.samples[0] - S.samples[0]) < 1e-10,
    `xs[i]=${I.samples[0]} vs normalize(superpose)=${S.samples[0]}`);
});

test('xs[i] sampling: mixture mean + per-branch fractions = w (closed-form)', async () => {
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 5.0, sigma = 1.0)
M3 = Normal(mu = 9.0, sigma = 1.0)
i  = draw(Categorical(p = [0.2, 0.3, 0.5]))
xs = [draw(M1), draw(M2), draw(M3)]
x  = draw(xs[i])
`);
  const m = await ctx.getMeasure('x');
  const xs = m.samples;
  const EX = 0.2 * 0 + 0.3 * 5 + 0.5 * 9;            // = 6.0
  let n1 = 0, n2 = 0, n3 = 0;
  for (let k = 0; k < xs.length; k++) {
    if (xs[k] < 2.5) n1++; else if (xs[k] < 7) n2++; else n3++;
  }
  const N = xs.length;
  assert.ok(Math.abs(unweightedMean(xs) - EX) < 0.15,
    `mixture mean: got ${unweightedMean(xs)}, expected ${EX}`);
  assert.ok(Math.abs(n1 / N - 0.2) < 0.025
    && Math.abs(n2 / N - 0.3) < 0.025
    && Math.abs(n3 / N - 0.5) < 0.025,
    `branch fractions [${n1 / N}, ${n2 / N}, ${n3 / N}] vs [0.2,0.3,0.5]`);
});

// =====================================================================
// broadcast(logdensityof, M, points) — evaluate a tractable density
// at many points. flatppl-js EAGER reference realisation
// (engine-concepts §11): maps the trusted single-point logdensityof
// over the points; tractable M ⇒ NO sampling. Result is a value
// array (one logp per point).
// =====================================================================

test('broadcast(logdensityof, M, pts): plain leaf == analytic logpdf vector', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
pts = [0.0, 1.0, 2.5, 5.0]
lps = broadcast(logdensityof, A, pts)
`);
  const m = await ctx.getMeasure('lps');
  const P = [0.0, 1.0, 2.5, 5.0];
  assert.equal(m.samples.length, P.length, 'one logp per point');
  for (let i = 0; i < P.length; i++) {
    assert.ok(Math.abs(m.samples[i] - normalLogpdf(P[i], 0, 1)) < 1e-10,
      `point ${P[i]}: got ${m.samples[i]}, expected ${normalLogpdf(P[i], 0, 1)}`);
  }
});

test('broadcast(logdensityof, mixture, pts): per-point closed-form mixture', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 2.0)
mix = normalize(superpose(weighted(0.3, A), weighted(0.7, B)))
pts = [-1.0, 0.0, 1.0, 2.5, 5.0, 8.0]
lps = broadcast(logdensityof, mix, pts)
`);
  const m = await ctx.getMeasure('lps');
  const P = [-1.0, 0.0, 1.0, 2.5, 5.0, 8.0];
  for (let i = 0; i < P.length; i++) {
    const exp = Math.log(
      0.3 * Math.exp(normalLogpdf(P[i], 0, 1))
      + 0.7 * Math.exp(normalLogpdf(P[i], 5, 2)));
    assert.ok(Math.abs(m.samples[i] - exp) < 1e-10,
      `point ${P[i]}: got ${m.samples[i]}, expected ${exp}`);
  }
});

test('broadcast(logdensityof, ifelse, pts) ≡ broadcast over normalize(superpose) — shared core through broadcast', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 2.0)
c = draw(Bernoulli(p = 0.3))
viaIf = ifelse(c, A, B)
viaSup = normalize(superpose(weighted(0.3, A), weighted(0.7, B)))
pts = [0.0, 1.3, 4.0, 6.5]
lpsIf  = broadcast(logdensityof, viaIf, pts)
lpsSup = broadcast(logdensityof, viaSup, pts)
`);
  const [I, S] = await Promise.all([ctx.getMeasure('lpsIf'), ctx.getMeasure('lpsSup')]);
  assert.equal(I.samples.length, 4);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(I.samples[i] - S.samples[i]) < 1e-10,
      `point ${i}: ifelse=${I.samples[i]} vs normalize(superpose)=${S.samples[i]}`);
  }
});

test('broadcast(logdensityof, mix, pts): inline point vector trapezoids to 1', async () => {
  // ∫ p_mix dx ≈ 1 via a single broadcast over a fine inline grid —
  // exercises broadcast(logdensityof,…) with an inline (non-binding)
  // points expression and confirms it is a proper density.
  const lo = -10, hi = 14, n = 4000, h = (hi - lo) / n;
  let grid = '';
  for (let i = 0; i <= n; i++) {
    grid += (i ? ', ' : '') + (lo + i * h).toFixed(6);
  }
  const ctx = makeCtx(`
A = Normal(mu = -1.0, sigma = 0.9)
B = Normal(mu =  4.0, sigma = 1.4)
mix = normalize(superpose(weighted(0.45, A), weighted(0.55, B)))
lps = broadcast(logdensityof, mix, [${grid}])
`);
  const m = await ctx.getMeasure('lps');
  let integral = 0;
  for (let i = 0; i <= n; i++) {
    const w = (i === 0 || i === n) ? 0.5 : 1.0;
    integral += w * Math.exp(m.samples[i]);
  }
  integral *= h;
  assert.ok(Math.abs(integral - 1.0) < 2e-3,
    `broadcast density must integrate to 1, got ${integral}`);
});

// =====================================================================
// Reference-measure guard (engine-concepts §11/§12). logsumexp-ing a
// continuous (Lebesgue) branch with a discrete/atomic (counting /
// point-mass) one is dimensionally meaningless — the spike-and-slab
// trap. The guard refuses such a logdensityof with a clear error,
// while leaving SAMPLING (well-defined) allowed, and never rejects a
// homogeneous mixture (zero-false-positive).
// =====================================================================

test('ref-measure guard: spike-and-slab ifelse(c,Normal,Dirac) density is refused', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
S = Dirac(value = 0.0)
c = draw(Bernoulli(p = 0.3))
M = ifelse(c, A, S)
lp = logdensityof(M, 0.0)
`);
  await assert.rejects(() => ctx.getMeasure('lp'),
    /incommensurable reference measures|spike-and-slab/);
});

test('ref-measure guard: spike-and-slab SAMPLING stays well-defined (density-only guard)', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
S = Dirac(value = 7.0)
c = draw(Bernoulli(p = 0.25))
x = draw(ifelse(c, A, S))
`);
  const m = await ctx.getMeasure('x');           // must NOT throw
  // ifelse(c, A, S): c true (prob p=0.25) → A=Normal; c false
  // (prob 1−p=0.75) → S=Dirac@7. The atom at exactly 7.0 carries
  // ≈ 1−p = 0.75 of the mass.
  let atom = 0;
  for (let i = 0; i < m.samples.length; i++) if (m.samples[i] === 7.0) atom++;
  const frac = atom / m.samples.length;
  assert.ok(Math.abs(frac - 0.75) < 0.04,
    `spike (Dirac/false-branch) fraction ≈ 1−p = 0.75, got ${frac}`);
});

test('ref-measure guard: superpose(Normal, Poisson) density is refused (continuous ⊕ discrete)', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Poisson(rate = 2.0)
lp = logdensityof(superpose(A, B), 1.0)
`);
  await assert.rejects(() => ctx.getMeasure('lp'),
    /incommensurable reference measures/);
});

test('ref-measure guard: homogeneous discrete superpose(Poisson,Bernoulli) is allowed + correct', async () => {
  const ctx = makeCtx(`
A = Poisson(rate = 2.0)
B = Bernoulli(p = 0.5)
lp = logdensityof(superpose(A, B), 1.0)
`);
  const m = await ctx.getMeasure('lp');          // must NOT throw
  // raw additive: p_Pois(1;2) + p_Bern(1;0.5) = 2·e^-2 + 0.5
  const expected = Math.log(2 * Math.exp(-2) + 0.5);
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-9,
    `homogeneous-discrete superpose: got ${m.samples[0]}, expected ${expected}`);
});

test('ref-measure guard: homogeneous continuous superpose(Normal,Cauchy) is allowed', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Cauchy(location = 0.0, scale = 1.0)
lp = logdensityof(superpose(A, B), 0.5)
`);
  const m = await ctx.getMeasure('lp');          // must NOT throw
  const expected = Math.log(
    Math.exp(normalLogpdf(0.5, 0, 1))
    + (1 / (Math.PI * (1 + 0.5 * 0.5))));
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-9,
    `Normal+Cauchy superpose: got ${m.samples[0]}, expected ${expected}`);
});

test('ref-measure guard: nested heterogeneity (branch is itself mixed) is refused', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 3.0, sigma = 1.0)
S = Dirac(value = 0.0)
inner = superpose(B, S)
c = draw(Bernoulli(p = 0.5))
M = ifelse(c, A, inner)
lp = logdensityof(M, 0.0)
`);
  await assert.rejects(() => ctx.getMeasure('lp'),
    /incommensurable reference measures|spike-and-slab/);
});

// =====================================================================
// broadcast(logdensityof, M, pts): the batched closed-form pass
// (points-batched density) must be BIT-IDENTICAL to the single-point
// reference route — the invariant of that optimisation, pinned here
// independently of any analytic reference.
// =====================================================================

test('broadcast(logdensityof) batched ≡ per-point single logdensityof (bit-identical)', async () => {
  const grid = [-3.0, -1.25, 0.0, 0.4, 2.7, 5.0, 8.5];
  let src = `
A = Normal(mu = 0.0, sigma = 1.3)
B = Normal(mu = 5.0, sigma = 2.0)
c = draw(Bernoulli(p = 0.35))
mix = ifelse(c, A, B)
lps = broadcast(logdensityof, mix, [${grid.join(', ')}])
`;
  // one single-point logdensityof binding per grid point
  grid.forEach((x, i) => { src += `p${i} = logdensityof(mix, ${x})\n`; });
  const ctx = makeCtx(src);
  const batched = await ctx.getMeasure('lps');
  assert.equal(batched.samples.length, grid.length);
  for (let i = 0; i < grid.length; i++) {
    const single = (await ctx.getMeasure('p' + i)).samples[0];
    assert.ok(Math.abs(batched.samples[i] - single) < 1e-12,
      `point ${grid[i]}: batched ${batched.samples[i]} vs single-point `
      + `reference ${single} (must be bit-identical)`);
  }
});

// =====================================================================
// §11 (2a): INLINE-measure ifelse branches — ifelse(c, Normal(..),
// Normal(..)) (sampleable-leaf branches, no named bindings). Same
// select core; must match the named-branch form exactly. Value-valued
// ifelse stays declined (evaluator path) — covered by logical-ops
// tests; the suite-green gate guards that.
// =====================================================================

test('inline-branch ifelse: density == named-branch ifelse == analytic', async () => {
  const ctx = makeCtx(`
A = Normal(mu = 0.0, sigma = 1.0)
B = Normal(mu = 5.0, sigma = 2.0)
c = draw(Bernoulli(p = 0.3))
named  = ifelse(c, A, B)
inline = ifelse(c, Normal(mu = 0.0, sigma = 1.0), Normal(mu = 5.0, sigma = 2.0))
lpN = logdensityof(named, 1.3)
lpI = logdensityof(inline, 1.3)
`);
  const [N, I] = await Promise.all([ctx.getMeasure('lpN'), ctx.getMeasure('lpI')]);
  const expected = Math.log(
    0.3 * Math.exp(normalLogpdf(1.3, 0, 1))
    + 0.7 * Math.exp(normalLogpdf(1.3, 5, 2)));
  assert.ok(Math.abs(I.samples[0] - expected) < 1e-10,
    `inline-branch ifelse density: got ${I.samples[0]}, expected ${expected}`);
  assert.ok(Math.abs(I.samples[0] - N.samples[0]) < 1e-12,
    `inline must equal named-branch ifelse: ${I.samples[0]} vs ${N.samples[0]}`);
});

test('inline-branch ifelse: sampling mixture mean/variance (closed-form)', async () => {
  const p = 0.3, muA = -8, sgA = 1, muB = 8, sgB = 1;     // well-separated
  const ctx = makeCtx(`
c = draw(Bernoulli(p = 0.3))
x = draw(ifelse(c, Normal(mu = -8.0, sigma = 1.0), Normal(mu = 8.0, sigma = 1.0)))
`);
  const xs = (await ctx.getMeasure('x')).samples;
  const EX = p * muA + (1 - p) * muB;                       // = 3.2
  const EX2 = p * (sgA * sgA + muA * muA) + (1 - p) * (sgB * sgB + muB * muB);
  const VX = EX2 - EX * EX;
  let nA = 0;
  for (let i = 0; i < xs.length; i++) if (xs[i] < 0) nA++;   // clean split
  assert.ok(Math.abs(unweightedMean(xs) - EX) < 0.15,
    `mean: got ${unweightedMean(xs)}, expected ${EX}`);
  assert.ok(Math.abs(unweightedVar(xs) - VX) / VX < 0.10,
    `variance: got ${unweightedVar(xs)}, expected ${VX}`);
  assert.ok(Math.abs(nA / xs.length - p) < 0.02,
    `branch-A fraction ≈ p = ${p}, got ${nA / xs.length}`);
});

// =====================================================================
// bayesupdate: Gamma-Poisson conjugate posterior (multi-observation)
// =====================================================================
//
// Closed-form conjugate prior:
//   λ ~ Gamma(α₀, β₀),  y_i | λ ~ Poisson(λ) IID, i = 1..n
//   ⇒ λ | y ~ Gamma(α₀ + Σy_i, β₀ + n)
//
// With α₀=2, β₀=1, counts_data=[2,3,7,6,4], Σy = 22, n = 5:
//   posterior λ ~ Gamma(24, 6)
//   E[λ | data] = 24/6 = 4
//   Var[λ | data] = 24/36 = 2/3
//
// Importance sampling reweights prior atoms by L(λ_i) = Π Poisson(y_k; λ_i):
// the weighted moments converge to the closed-form Gamma(24, 6) moments.
//
// Also exercises classifyIid resolving `iid(Poisson(λ), n)` where
// `n = lengthof(counts_data)` is a binding ref to a fixed-phase
// expression — the regression covered structurally in
// iid-multi-axis.test.ts. Here we additionally pin the numerical
// posterior moments end-to-end.

// Closed-form log-pdf helpers.
function logGamma(x: number): number {
  // Stirling-series approximation good enough for ratios across atoms
  // — Numerical Recipes / Lanczos coefficients would tighten this
  // further but the differences we test against cancel the bulk.
  let g = 0, z = x;
  while (z < 7) { g -= Math.log(z); z += 1; }
  // Stirling for log Γ(z) at z ≥ 7
  const z2 = 1 / (z * z);
  g += (z - 0.5) * Math.log(z) - z + 0.5 * Math.log(2 * Math.PI)
     + (1/12 - (1/360 - (1/1260 - 1/(1680*z2)) * z2) * z2) / z;
  return g;
}
function gammaLogPdf(x: number, shape: number, rate: number): number {
  if (x <= 0) return -Infinity;
  return shape * Math.log(rate) - logGamma(shape)
       + (shape - 1) * Math.log(x) - rate * x;
}

test('bayesupdate: Gamma-Poisson conjugate ⇒ posterior Gamma(24, 6)', async () => {
  // Spec §08 Gamma(shape, rate); per-observation Poisson IID over the
  // count data. Forward kernel boundary-includes lambda so the
  // posterior atoms carry λ samples + importance weights from
  // the per-atom log-likelihood L(λ) = Π Poisson(counts_k | λ).
  const ctx = makeCtx(`
counts_data = [2, 3, 7, 6, 4]
n = lengthof(counts_data)
lambda = draw(Gamma(shape = 2.0, rate = 1.0))
prior = lawof(record(lambda = lambda))
y_dist = joint(y = iid(Poisson(rate = lambda), n))
K = functionof(y_dist, lambda = lambda)
L = likelihoodof(K, record(y = counts_data))
posterior = bayesupdate(L, prior)
`);
  const post = await ctx.getMeasure('posterior');
  assert.ok(post.fields && post.fields.lambda, 'posterior is a record measure with lambda');
  assert.ok(post.logWeights, 'posterior atoms carry logWeights');

  const lambdas = post.fields.lambda.samples;
  const lw     = post.logWeights;

  // Plan B: weighted moments against analytical Gamma(24, 6).
  const muHat   = weightedMean(lambdas, lw);
  const varHat  = weightedVar(lambdas, lw);
  // Closed-form: mean = 24/6 = 4, var = 24/36 = 2/3 ≈ 0.6667.
  assert.ok(Math.abs(muHat - 4.0) < 0.10,
    'posterior E[λ] = 24/6 = 4.0, got ' + muHat);
  assert.ok(Math.abs(varHat - (2/3)) < 0.10,
    'posterior Var[λ] = 24/36 = 2/3, got ' + varHat);

  // Plan A: log-density consistency. For each posterior atom λ_i,
  //   log p_engine(λ_i) = log p_prior(λ_i) + lw_i
  // should differ from
  //   log p_closed(λ_i) = log Gamma(λ_i; 24, 6)
  // by an additive normalising constant that's the SAME for every
  // atom (engine returns unnormalised log-posterior; closed-form is
  // the normalised Gamma(24, 6)). Check that the per-atom difference
  // is consistent across atoms — i.e. spread (max-min) is small.
  let minDiff =  Infinity, maxDiff = -Infinity;
  for (let i = 0; i < lambdas.length; i++) {
    const lam = lambdas[i];
    if (!(lam > 0)) continue;          // skip any pathological atoms
    const logPrior  = gammaLogPdf(lam, 2.0, 1.0);
    const logPostEn = logPrior + lw[i];
    const logPostCl = gammaLogPdf(lam, 24.0, 6.0);
    const d = logPostEn - logPostCl;
    if (d < minDiff) minDiff = d;
    if (d > maxDiff) maxDiff = d;
  }
  // Numerically the spread should be near floating-point fuzz; allow
  // 1e-6 to absorb Stirling-series rounding in logGamma above.
  assert.ok(maxDiff - minDiff < 1e-6,
    'log p_engine − log p_Gamma(24,6) varies across atoms — engine '
    + 'posterior shape mismatches closed-form. spread=' + (maxDiff - minDiff));

  // n_eff sanity — Gamma(2,1) is concentrated near small λ; data with
  // mean 4.4 pulls the posterior away from the prior mode (2), so ESS
  // will be lossy but should still be a meaningful fraction.
  assert.ok(post.n_eff > SAMPLE_COUNT * 0.10,
    'n_eff > 10% of N (importance overlap reasonable), got ' + post.n_eff);
});

// =====================================================================
// bayesupdate: Normal-InverseGamma linear regression
//   sigma² ~ InverseGamma(5, 5);   sigma = sqrt(sigma²)
//   alpha, beta | sigma ~ Normal(0, 3 sigma)
//   y_i ~ Normal(alpha + beta x_i, sigma)     (broadcast Normal kernel)
//
// NIG conjugate posterior parameters (closed form):
//   μ_n  ≈ (1.1355, 1.8739)
//   a_n  = a_0 + n/2 = 5 + 2 = 7
//   b_n  ≈ 5.3035
//   E[sigma²|data] = b_n / (a_n − 1) ≈ 0.8839
//
// Density exercises walkBroadcast (per-atom × per-element Normal logpdf
// over the broadcast(Normal, means, sigma) variate).
// =====================================================================

test('bayesupdate: NIG linear regression ⇒ posterior moments match closed form', async () => {
  const ctx = makeCtx(`
x_data = [1.1, 1.5, 1.3, 1.4]
y_data = [3.2, 4.1, 3.4, 3.9]
sigma2 = draw(InverseGamma(shape = 5.0, scale = 5.0))
sigma  = sqrt(sigma2)
alpha  = draw(Normal(mu = 0.0, sigma = sigma * 3.0))
beta   = draw(Normal(mu = 0.0, sigma = sigma * 3.0))
prior  = lawof(record(alpha = alpha, beta = beta, sigma = sigma))
means  = alpha .+ beta .* x_data
y      = draw(broadcast(Normal, means, sigma))
forward_kernel = kernelof(record(y = y),
                          alpha = alpha, beta = beta, sigma = sigma)
L         = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`);
  const post = await ctx.getMeasure('posterior');
  assert.ok(post.fields && post.fields.alpha && post.fields.beta && post.fields.sigma,
    'posterior is a record measure with alpha/beta/sigma');
  assert.ok(post.logWeights, 'posterior atoms carry logWeights');

  const alphas = post.fields.alpha.samples;
  const betas  = post.fields.beta.samples;
  const sigmas = post.fields.sigma.samples;
  const lw     = post.logWeights;

  // Plan B: weighted moments against the NIG closed form.
  const aHat   = weightedMean(alphas, lw);
  const bHat   = weightedMean(betas,  lw);
  // E[sigma² | data] = b_n / (a_n − 1) = 5.3035 / 6 ≈ 0.8839.
  let sigma2Sum = 0, lse = -Infinity;
  for (let i = 0; i < lw.length; i++) if (lw[i] > lse) lse = lw[i];
  let den = 0;
  for (let i = 0; i < lw.length; i++) {
    const w = Math.exp(lw[i] - lse);
    sigma2Sum += w * sigmas[i] * sigmas[i];
    den += w;
  }
  const sigma2Hat = sigma2Sum / den;

  // Importance-sampling variance can be loose when the prior overlaps
  // the posterior modestly; allow a few % absolute tolerance on each
  // moment. n_eff sanity check below pins the overlap quality.
  assert.ok(Math.abs(aHat - 1.1355) < 0.20,
    'posterior E[α] = μ_n[1] ≈ 1.1355, got ' + aHat);
  assert.ok(Math.abs(bHat - 1.8739) < 0.20,
    'posterior E[β] = μ_n[2] ≈ 1.8739, got ' + bHat);
  assert.ok(Math.abs(sigma2Hat - 0.8839) < 0.20,
    'posterior E[σ²] = b_n/(a_n−1) ≈ 0.8839, got ' + sigma2Hat);

  // n_eff: the prior is moderately diffuse relative to the posterior
  // (3-σ scale on the coefficients), so a fair fraction of atoms
  // should overlap meaningfully.
  assert.ok(post.n_eff > SAMPLE_COUNT * 0.02,
    'n_eff > 2% of N (importance overlap), got ' + post.n_eff);
});
