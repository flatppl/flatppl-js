'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildPriorTransform } = require('../prior-transform.ts');

// Independent prior: mu ~ Normal(1,2), rate ~ Exponential(0.5). No likelihood
// dependence needed for the prior transform, but the model needs a posterior
// derivation to enumerate latents — use a trivial observed likelihood.
const SRC = `
flatppl_compat = "0.1"
mu ~ Normal(1.0, 2.0)
lam ~ Exponential(0.5)
prior = lawof(record(mu = mu, lam = lam))
y ~ Normal.(mu, lam)
K = kernelof(record(y = y), mu = mu, lam = lam)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;

function ks(a: number[], b: number[]): number {           // two-sample KS statistic
  const A = a.slice().sort((x, y) => x - y), B = b.slice().sort((x, y) => x - y);
  const all = A.concat(B).sort((x, y) => x - y);
  let d = 0;
  for (const v of all) {
    const fa = A.filter((x) => x <= v).length / A.length;
    const fb = B.filter((x) => x <= v).length / B.length;
    d = Math.max(d, Math.abs(fa - fb));
  }
  return d;
}

test('prior transform: scalar independent latents match forward draws (KS)', () => {
  const { ctx } = ctxFor(SRC, 100);
  const d = ctx.derivations['posterior'];
  const pt = buildPriorTransform(ctx, d);
  assert.equal(pt.dim, 2);
  assert.deepEqual(pt.latentNames, ['mu', 'lam']);
  // Deterministic quasi-forward draws through T over a stratified grid on [0,1]^2.
  const N = 400;
  const muT: number[] = [], lamT: number[] = [];
  for (let i = 0; i < N; i++) {
    const u = new Float64Array([ (i + 0.5) / N, ((i * 7 + 3) % N + 0.5) / N ]);
    const rec = pt.transform(u);
    muT.push(rec.mu); lamT.push(rec.lam);
  }
  // Analytic forward draws via the same quantile ladder from independent U — the
  // oracle here is the closed-form prior CDF, realised as a reference sample.
  const muRef: number[] = [], lamRef: number[] = [];
  for (let i = 0; i < N; i++) {
    const p = (i + 0.5) / N;
    muRef.push(1.0 + 2.0 * Math.SQRT2 * require('@stdlib/math-base-special-erfinv')(2 * p - 1)); // Normal(1,2) ppf
    lamRef.push(-Math.log1p(-p) / 0.5);                                                          // Exp(0.5) ppf
  }
  assert.ok(ks(muT, muRef) < 0.1, `mu KS ${ks(muT, muRef)}`);
  assert.ok(ks(lamT, lamRef) < 0.1, `lam KS ${ks(lamT, lamRef)}`);
});

test('prior transform: coordSupports reflect each latent\'s real support', () => {
  const { ctx } = ctxFor(SRC, 100);
  const d = ctx.derivations['posterior'];
  const pt = buildPriorTransform(ctx, d);
  assert.equal(pt.coordSupports.length, 2);
  assert.equal(pt.coordSupports[0].kind, 'real');       // mu ~ Normal(1,2)
  assert.notEqual(pt.coordSupports[1].kind, 'real');    // lam ~ Exponential(0.5)
  assert.equal(pt.coordSupports[1].kind, 'positive');
});

test('prior transform: iid(Normal(0,1), 4) yields a 4-vector, coords standard-normal', () => {
  const SRC_IID = `
flatppl_compat = "0.1"
z ~ iid(Normal(0.0, 1.0), 4)
prior = lawof(record(z = z))
y ~ Normal.(z, 1.0)
K = kernelof(record(y = y), z = z)
L = likelihoodof(K, record(y = [0.0, 0.0, 0.0, 0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC_IID, 100);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.equal(pt.dim, 4);
  const rec = pt.transform(new Float64Array([0.5, 0.5, 0.975, 0.025]));
  assert.ok(rec.z instanceof Float64Array && rec.z.length === 4);
  assert.ok(Math.abs(rec.z[0]) < 1e-9 && Math.abs(rec.z[1]) < 1e-9);   // Φ⁻¹(0.5)=0
  assert.ok(rec.z[2] > 1.9 && rec.z[2] < 2.0);                          // Φ⁻¹(0.975)≈1.96
  assert.ok(rec.z[3] < -1.9 && rec.z[3] > -2.0);                        // Φ⁻¹(0.025)≈−1.96
});

test('prior transform: conditional iid (funnel) — theta tracks realised mu,sig', () => {
  // Brief's model used `sig ~ HalfNormal(1.0)`, but HalfNormal is absent from
  // SAMPLEABLE_DISTRIBUTIONS (ir-shared.ts) — the worker/classifier gate the
  // rest of buildDerivations uses — so `sig ~ HalfNormal(1.0)` fails
  // classification (bayesupdate boundary error) before prior-transform even
  // runs; that's an engine gap orthogonal to this task, not something to fix
  // here. HalfNormal remains a valid *inner* base measure for the ladder
  // (ARG_NAMES / inverse-cdf.ts already support it — see the coordSupports
  // test above using Exponential similarly). Swapped in `Exponential(1.0)`
  // for the scale latent, which is on both SAMPLEABLE_DISTRIBUTIONS and
  // ARG_NAMES, to keep the conditional-iid wiring under test without
  // tripping the unrelated classification gap.
  const SRC_FUNNEL = `
flatppl_compat = "0.1"
mu ~ Normal(0.0, 1.0)
sig ~ Exponential(1.0)
theta ~ iid(Normal(mu, sig), 3)
prior = lawof(record(mu = mu, sig = sig, theta = theta))
y ~ Normal.(theta, 1.0)
K = kernelof(record(y = y), mu = mu, sig = sig, theta = theta)
L = likelihoodof(K, record(y = [0.0, 0.0, 0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx: c2 } = ctxFor(SRC_FUNNEL, 100);
  const pt2 = buildPriorTransform(c2, c2.derivations['posterior']);
  assert.deepEqual(pt2.latentNames, ['mu', 'sig', 'theta']);
  assert.equal(pt2.dim, 5);   // 1 + 1 + 3
  // u = [Φ(mu-coord), Exponential(sig-coord), theta coords...]. 0.9772499 /
  // 0.0227501 are Φ(±2) (verified via @stdlib erfinv below, not eyeballed —
  // the brief's own "≈1.96" annotation for these u's was off: Φ(1.96)=0.975,
  // not 0.9772499).
  const u = new Float64Array([0.8413447, 0.5, 0.5, 0.9772499, 0.0227501]);
  const rec = pt2.transform(u);
  const mu = rec.mu, sig = rec.sig;         // mu = Φ⁻¹(0.8413)=1 ; sig = Exponential(1) median = ln2 ≈ 0.693147
  assert.ok(Math.abs(mu - 1) < 1e-4, `mu ${mu}`);
  assert.ok(Math.abs(sig - Math.LN2) < 1e-4, `sig ${sig}`);
  const erfinv = require('@stdlib/math-base-special-erfinv');
  const probit = (p: number) => Math.SQRT2 * erfinv(2 * p - 1);
  const z = probit(0.9772499);              // independent oracle for the theta1/theta2 coord, ≈2.0
  // theta[0] = mu + sig·Φ⁻¹(0.5) = mu ; theta[1] = mu + sig·z ; theta[2] = mu − sig·z
  assert.ok(Math.abs(rec.theta[0] - mu) < 1e-4, `theta0 ${rec.theta[0]}`);
  assert.ok(Math.abs(rec.theta[1] - (mu + sig * z)) < 1e-3, `theta1 ${rec.theta[1]}`);
  assert.ok(Math.abs(rec.theta[2] - (mu - sig * z)) < 1e-3, `theta2 ${rec.theta[2]}`);
});

test('prior transform: named derived binding between parent draw and child latent params', () => {
  // `logm = log(m)` sits between the parent draw `m` and the child latent
  // `b ~ LogNormal(logm, 0.5)`. Without refreshDerived seeding `env.logm`,
  // resolveParams throws (unresolved ref) — this is the load-bearing case
  // the funnel test above cannot exercise (there, children reference direct
  // draws already threaded into env by tasks 2-3). `m` uses Exponential(1.0)
  // rather than the brief's HalfNormal(1.0) for the same classification-gate
  // reason noted above.
  const SRC_DERIVED = `
flatppl_compat = "0.1"
m ~ Exponential(1.0)
logm = log(m)
b ~ LogNormal(logm, 0.5)
prior = lawof(record(m = m, b = b))
y ~ Normal.(b, 1.0)
K = kernelof(record(y = y), m = m, b = b)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC_DERIVED, 100);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.deepEqual(pt.latentNames, ['m', 'b']);
  assert.equal(pt.dim, 2);
  // u = [0.5, 0.5]: m = Exponential(1) median = ln2; Φ⁻¹(0.5) = 0 so
  // b = exp(logm + 0.5·0) = exp(log(m)) = m (independent oracle: identity,
  // not a re-derivation of the engine's own LogNormal quantile).
  const u = new Float64Array([0.5, 0.5]);
  const rec = pt.transform(u);
  assert.ok(Math.abs(rec.m - Math.LN2) < 1e-6, `m ${rec.m}`);   // Exponential(1) median = ln2
  assert.ok(Math.abs(rec.b - rec.m) < 1e-6, `b ${rec.b} vs m ${rec.m}`);
});

test('prior transform: finite two-sided truncation — z at u=0.5 is the symmetric-interval median (0)', () => {
  // normalize(truncate(Normal(0,1), interval(-1,1))) — a truncated standard
  // normal on a symmetric interval. Independent closed-form oracle: its
  // median is 0 by symmetry (F(-1) and F(1) are equidistant from 0.5 around
  // Φ(0)=0.5), regardless of the engine's own quantile ladder.
  const SRC_TRUNC = `
flatppl_compat = "0.1"
z ~ normalize(truncate(Normal(0.0, 1.0), interval(-1.0, 1.0)))
prior = lawof(record(z = z))
y ~ Normal.(z, 1.0)
K = kernelof(record(y = y), z = z)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC_TRUNC, 100);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.deepEqual(pt.latentNames, ['z']);
  assert.equal(pt.dim, 1);
  const zAtHalf = pt.transform(new Float64Array([0.5])).z;
  assert.ok(Math.abs(zAtHalf) < 1e-9, `z at u=0.5 should be ≈0, got ${zAtHalf}`);
  // Monotonicity in the cube coordinate, and containment within the bounds.
  const zLo = pt.transform(new Float64Array([0.1])).z;
  const zHi = pt.transform(new Float64Array([0.9])).z;
  assert.ok(zHi > zLo, `transform must be monotone: z(0.9)=${zHi} should exceed z(0.1)=${zLo}`);
  assert.ok(zLo > -1 && zLo < 1, `z(0.1)=${zLo} must lie within (-1,1)`);
  assert.ok(zHi > -1 && zHi < 1, `z(0.9)=${zHi} must lie within (-1,1)`);
});

test('prior transform: pushfwd applies fn forward to the base quantile', () => {
  const SRC_PF = `
flatppl_compat = "0.1"
b ~ pushfwd(fn(20.0 * _), Beta(2.0, 2.0))
prior = lawof(record(b = b))
y ~ Normal.(b, 1.0)
K = kernelof(record(y = y), b = b)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC_PF, 100);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.equal(pt.dim, 1);
  assert.ok(Math.abs(pt.transform(new Float64Array([0.5])).b - 10) < 1e-4);
  // monotone increasing in u
  assert.ok(pt.transform(new Float64Array([0.9])).b > pt.transform(new Float64Array([0.1])).b);
});

test('prior transform: composite-iid Beta block realises per-row quantiles', () => {
  const MODEL = `
G = 2
N = 3
n_data = [[10, 10, 10], [10, 10, 10]]
r_data = [[7, 8, 6], [5, 6, 4]]
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
binomial_row_K = (n_row, p_row) -> Binomial.(n_row, p_row)
r ~ binomial_row_K.(n_data, p)
prior = lawof(record(p = p))
forward_kernel = kernelof(record(r = r), p = p)
L = likelihoodof(forward_kernel, record(r = r_data))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(MODEL, 50);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.equal(pt.dim, 6);                                  // G·N = 2·3
  const { quantile } = require('../inverse-cdf.ts');
  // u=0.5 for every coord: element (g,j) = Beta⁻¹(0.5; a[g], b[g]). Row 0 uses (2,5); row 1 uses (3,4).
  const rec = pt.transform(new Float64Array(6).fill(0.5));
  assert.ok(rec.p instanceof Float64Array && rec.p.length === 6);
  for (let g = 0; g < 2; g++) for (let j = 0; j < 3; j++) {
    const want = quantile('Beta', 0.5, { alpha: [2, 3][g], beta: [5, 4][g] });
    assert.ok(Math.abs(rec.p[g * 3 + j] - want) < 1e-9, `p[${g},${j}] = ${rec.p[g*3+j]} vs ${want}`);
  }
});

test('prior transform: unsupported (discrete) prior form refuses loudly', () => {
  const SRC_BAD = `
flatppl_compat = "0.1"
n ~ Poisson(3.0)
prior = lawof(record(n = n))
y ~ Normal.(n, 1.0)
K = kernelof(record(y = y), n = n)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC_BAD, 100);
  assert.throws(() => {
    const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
    pt.transform(new Float64Array(pt.dim).fill(0.5));   // in case the throw is lazy (per-realise)
  }, /not invertible|not eligible|not yet supported|no quantile/);
});

test('prior transform: eight-schools full fixture — tau is HalfCauchy via normalize(truncate)', () => {
  const src = fs.readFileSync(path.join(__dirname, 'fixtures/baseline/eight-schools.flatppl'), 'utf8');
  const { ctx } = ctxFor(src, 100);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.deepEqual(pt.latentNames, ['mu', 'tau', 'theta']);
  assert.equal(pt.dim, 10);                     // 1 + 1 + 8
  // tau = truncate(Cauchy(0,5), [0,inf)) quantile at u=0.5 → 5 (half-Cauchy median).
  const u = new Float64Array(10).fill(0.5);
  const rec = pt.transform(u);
  assert.ok(Math.abs(rec.tau - 5) < 1e-4, `tau ${rec.tau}`);
  assert.ok(rec.tau > 0, 'tau positive-support');
});

test('prior transform: partial-pooling fixture — Uniform(interval(0,1)) phi at u=0.5 is 0.5', () => {
  // GAP A fixture: `phi ~ Uniform(interval(0.0, 1.0))` gives measure IR
  // {op:'Uniform', args:[{op:'interval', args:[lit 0, lit 1]}]} — one interval
  // arg, not two positional scalars. Independent oracle: Uniform(0,1)'s
  // quantile is the identity, so phi(u) = u exactly.
  const src = fs.readFileSync(path.join(__dirname, 'fixtures/baseline/partial-pooling.flatppl'), 'utf8');
  const { ctx } = ctxFor(src, 50);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.deepEqual(pt.latentNames, ['phi', 'kappa', 'theta']);
  assert.equal(pt.dim, 10);   // phi(1) + kappa(1) + theta(N=8)
  const rec = pt.transform(new Float64Array(pt.dim).fill(0.5));
  assert.ok(Math.abs(rec.phi - 0.5) < 1e-12, `phi at u=0.5 should be 0.5, got ${rec.phi}`);
  assert.ok(rec.phi > 0 && rec.phi < 1, `phi=${rec.phi} must lie in (0,1)`);
  // Monotone in the phi coordinate (index 0), holding the rest at 0.5, and
  // stays within the unit interval across the grid.
  let prev = -Infinity;
  for (let k = 1; k < 20; k++) {
    const u = new Float64Array(pt.dim).fill(0.5);
    u[0] = k / 20;
    const phi = pt.transform(u).phi;
    assert.ok(phi > prev, `phi not strictly monotone at u[0]=${k / 20}: ${phi} <= ${prev}`);
    assert.ok(phi > 0 && phi < 1, `phi=${phi} out of (0,1) at u[0]=${k / 20}`);
    prev = phi;
  }
});

test('prior transform: linear-regression fixture — InverseGamma(5,5) sigma2 quantile + finite composition through sigma=sqrt(sigma2)', () => {
  // GAP B fixture: `sigma2 ~ InverseGamma(5, 5)` was refused ("not invertible")
  // before InverseGamma got a quantile/cdf. Oracle: scipy.stats.invgamma(a=5,
  // scale=5).ppf(0.5) = 1.070455477822771 (computed independently via scipy;
  // see inverse-cdf.test.ts for the same constant against the full p-grid).
  const src = fs.readFileSync(path.join(__dirname, 'fixtures/baseline/linear-regression.flatppl'), 'utf8');
  const { ctx } = ctxFor(src, 50);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.deepEqual(pt.latentNames, ['sigma2', 'alpha', 'beta']);
  assert.equal(pt.dim, 3);
  const SIGMA2_PPF_HALF = 1.070455477822771;   // scipy oracle
  const rec = pt.transform(new Float64Array([0.5, 0.7, 0.3]));
  assert.ok(Math.abs(rec.sigma2 - SIGMA2_PPF_HALF) < 1e-9, `sigma2 ${rec.sigma2} vs scipy ${SIGMA2_PPF_HALF}`);
  // alpha ~ Normal(0, sigma*3) with sigma=sqrt(sigma2) — a derived binding
  // sitting between the sigma2 draw and the alpha/beta latents. Independent
  // closed-form check (not the engine's own probit call): alpha = 3·σ·Φ⁻¹(0.7).
  const erfinv = require('@stdlib/math-base-special-erfinv');
  const probit = (p: number) => Math.SQRT2 * erfinv(2 * p - 1);
  const sigma = Math.sqrt(SIGMA2_PPF_HALF);
  const alphaExpected = 3 * sigma * probit(0.7);
  const betaExpected = 3 * sigma * probit(0.3);
  assert.ok(Number.isFinite(rec.alpha), `alpha must be finite, got ${rec.alpha}`);
  assert.ok(Number.isFinite(rec.beta), `beta must be finite, got ${rec.beta}`);
  assert.ok(Math.abs(rec.alpha - alphaExpected) < 1e-6, `alpha ${rec.alpha} vs expected ${alphaExpected}`);
  assert.ok(Math.abs(rec.beta - betaExpected) < 1e-6, `beta ${rec.beta} vs expected ${betaExpected}`);
});

test('prior transform: StudentT/Cauchy/ChiSquared latents run (no refusal), medians match scipy', () => {
  // Previously refused ("not invertible") — all three have closed-form/library
  // quantiles now. Independent oracle (scipy, computed separately — see
  // inverse-cdf.test.ts for the full ppf grid): StudentT(nu) median = 0 by
  // symmetry; ChiSquared(k) median = scipy .ppf(0.5); Cauchy(loc,scale) median
  // = loc by symmetry.
  const SRC_TCC = `
flatppl_compat = "0.1"
t ~ StudentT(3.0)
c ~ Cauchy(0.0, 1.0)
x ~ ChiSquared(4.0)
prior = lawof(record(t = t, c = c, x = x))
y ~ Normal.(t, 1.0)
K = kernelof(record(y = y), t = t, c = c, x = x)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC_TCC, 100);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.deepEqual(pt.latentNames, ['t', 'c', 'x']);
  assert.equal(pt.dim, 3);
  const rec = pt.transform(new Float64Array([0.5, 0.5, 0.5]));
  const CHISQ4_PPF_HALF = 3.3566939800333224;   // scipy.stats.chi2(df=4).ppf(0.5)
  assert.ok(Math.abs(rec.t) < 1e-9, `t at u=0.5 should be 0 (StudentT median), got ${rec.t}`);
  assert.ok(Math.abs(rec.c) < 1e-9, `c at u=0.5 should be 0 (Cauchy(0,1) median), got ${rec.c}`);
  assert.ok(Math.abs(rec.x - CHISQ4_PPF_HALF) < 1e-9, `x at u=0.5 should be scipy chi2(4) median ${CHISQ4_PPF_HALF}, got ${rec.x}`);
});

test('prior transform: every coordinate is monotone non-decreasing in its cube coord', () => {
  const src = fs.readFileSync(path.join(__dirname, 'fixtures/baseline/eight-schools.flatppl'), 'utf8');
  const { ctx } = ctxFor(src, 100);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  const base = new Float64Array(pt.dim).fill(0.5);
  for (let c = 0; c < pt.dim; c++) {
    let prevFlat = -Infinity;
    for (let k = 1; k < 40; k++) {
      const u = base.slice(); u[c] = k / 40;
      const rec = pt.transform(u);
      // flatten record to the coordinate list to read coord c
      const flat: number[] = [];
      for (const nm of pt.latentNames) {
        const v = rec[nm];
        if (v instanceof Float64Array) for (const x of v) flat.push(x); else flat.push(v);
      }
      assert.ok(flat[c] >= prevFlat - 1e-9, `coord ${c} not monotone at u=${k/40}: ${flat[c]} < ${prevFlat}`);
      prevFlat = flat[c];
    }
  }
});

test('prior transform: hierarchical-logistic (kwargs Gamma + locscale StudentT pushfwd)', () => {
  // Exercises two real-model forms that a plain positional/functionof reader misses:
  //   sigma_a ~ Gamma(shape = 4.0, rate = 2.0)   — keyword-arg call (kwargs, not positional)
  //   b       ~ locscale(StudentT(3.0), 0.0, 2.5) — pushfwd via a location-scale BIJECTION
  //                                                 binding (args[0] is a bijection, not a bare functionof)
  const src = fs.readFileSync(path.join(__dirname, 'fixtures/baseline/hierarchical-logistic.flatppl'), 'utf8');
  const { ctx } = ctxFor(src, 20);
  const pt = buildPriorTransform(ctx, ctx.derivations['posterior']);
  assert.deepEqual(pt.latentNames, ['mu_a', 'sigma_a', 'a', 'b']);
  const rec = pt.transform(new Float64Array(pt.dim).fill(0.5));
  // mu_a = Normal(0,1) median = 0.
  assert.ok(Math.abs(rec.mu_a) < 1e-9, `mu_a ${rec.mu_a}`);
  // sigma_a = Gamma(shape=4, rate=2) median (kwargs path) — scipy.stats.gamma(a=4, scale=0.5).ppf(0.5).
  assert.ok(Math.abs(rec.sigma_a - 1.8360303744254485) < 1e-6, `sigma_a ${rec.sigma_a}`);
  // b = locscale(StudentT(3), 0, 2.5) at u=0.5 = 2.5·t₃⁻¹(0.5) + 0 = 0 (pushfwd via bijection).
  assert.ok(Math.abs(rec.b) < 1e-9, `b ${rec.b}`);
  // a ~ iid(Normal(mu_a, sigma_a), 3): each = mu_a + sigma_a·Φ⁻¹(0.5) = mu_a = 0.
  assert.ok(rec.a instanceof Float64Array && rec.a.length === 3);
  for (const v of rec.a) assert.ok(Math.abs(v) < 1e-9, `a elem ${v}`);
});
