'use strict';

// End-to-end tests for the six new univariate distributions added in
// the §08 fill-in:
//   - ChiSquared(k)
//   - VonMises(mu, kappa)
//   - Laplace(location, scale)
//   - Geometric(p)
//   - NegativeBinomial(alpha, beta)
//   - NegativeBinomial2(mu, psi)
//
// For each: classifier produces a draw step, samples are drawn from
// the right support / domain, the empirical mean ≈ the analytical
// mean within the sampler's noise. We also pin density values for
// VonMises (against a hand-computed reference) and the two negative-
// binomial parameterisations against each other at the equivalence
// point (α ↔ ψ, β ↔ ψ/μ).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 4096;
const ROOT_SEED    = 0xC0FFEEEE;

function makeCtx(source: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
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

function mean(xs: any) {
  let s = 0;
  for (let i = 0; i < xs.length; i++) s += xs[i];
  return s / xs.length;
}

// ---------------------------------------------------------------------
// ChiSquared
// ---------------------------------------------------------------------

test('ChiSquared(k): classifier OK + samples > 0', async () => {
  const ctx = makeCtx(`m = ChiSquared(k = 4)`);
  const m = await ctx.getMeasure('m');
  assert.ok(m.samples, 'matSample must produce .samples');
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(m.samples[i] > 0, `samples[${i}] = ${m.samples[i]} must be > 0`);
  }
});

test('ChiSquared(k): empirical mean ≈ k', async () => {
  const ctx = makeCtx(`m = ChiSquared(k = 5)`);
  const m = await ctx.getMeasure('m');
  const mu = mean(m.samples);
  assert.ok(Math.abs(mu - 5) < 0.3,
    `empirical mean = ${mu}, expected ≈ 5 (within 0.3)`);
});

// ---------------------------------------------------------------------
// VonMises
// ---------------------------------------------------------------------

test('VonMises(mu, kappa): classifier OK + samples within (μ−π, μ+π]', async () => {
  const ctx = makeCtx(`m = VonMises(mu = 0, kappa = 2)`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    const x = m.samples[i];
    assert.ok(x > -Math.PI - 1e-9 && x <= Math.PI + 1e-9,
      `samples[${i}] = ${x} must be in (−π, π]`);
  }
});

test('VonMises: density value pinned at canonical point (mu=0, kappa=1, x=0)', () => {
  const sampler = require('../sampler.ts');
  // At x = μ: logpdf = κ·cos(0) − log(2π) − log(I_0(κ)) = 1 − log(2π) − log(I_0(1)).
  // I_0(1) = 1.2660658777520084, so the exact value is -1.0737914249165241 (matches
  // Distributions.jl / numpyro). The engine's Chebyshev I_0 approximation is within
  // ~2.3e-8, so pin the TRUE value with a tolerance just looser than that — not the
  // old loose 1e-3 against a wrong target (flatppl-js#22).
  const reg = sampler._internal.REGISTRY.VonMises;
  // Reach into the logpdfFn directly (mu, kappa).
  const lp = reg.logpdfFn(0, 0, 1);
  assert.ok(Math.abs(lp - (-1.0737914249165241)) < 1e-6,
    `VonMises logpdf(x=0, mu=0, kappa=1) = ${lp}, expected ≈ -1.0737914249`);
});

test('VonMises: kappa=0 reduces to uniform density 1/(2π) at any x', () => {
  const sampler = require('../sampler.ts');
  const reg = sampler._internal.REGISTRY.VonMises;
  const expected = -Math.log(2 * Math.PI);
  for (const x of [-1.0, 0.0, 0.5, 2.0]) {
    const lp = reg.logpdfFn(x, 0, 0);
    assert.ok(Math.abs(lp - expected) < 1e-9,
      `VonMises logpdf at kappa=0 should be uniform ${expected}, got ${lp}`);
  }
});

// ---------------------------------------------------------------------
// Laplace
// ---------------------------------------------------------------------

test('Laplace(location, scale): classifier OK + samples real', async () => {
  const ctx = makeCtx(`m = Laplace(location = 0, scale = 1)`);
  const m = await ctx.getMeasure('m');
  assert.ok(m.samples.length === SAMPLE_COUNT);
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(Number.isFinite(m.samples[i]));
  }
});

test('Laplace(0, 1): empirical mean ≈ 0', async () => {
  const ctx = makeCtx(`m = Laplace(location = 0, scale = 1)`);
  const m = await ctx.getMeasure('m');
  const mu = mean(m.samples);
  assert.ok(Math.abs(mu) < 0.1,
    `Laplace(0,1) empirical mean = ${mu}, expected ≈ 0`);
});

test('Laplace: density at x=μ equals 1/(2b)', () => {
  const sampler = require('../sampler.ts');
  const reg = sampler._internal.REGISTRY.Laplace;
  // density(μ; μ, b) = 1/(2b) → log = -log(2b)
  const lp = reg.logpdfFn(0, 0, 1);
  assert.ok(Math.abs(lp - (-Math.log(2))) < 1e-12,
    `Laplace logpdf(0; 0, 1) = ${lp}, expected ${-Math.log(2)}`);
});

// ---------------------------------------------------------------------
// Geometric
// ---------------------------------------------------------------------

test('Geometric(p): samples are non-negative integers', async () => {
  const ctx = makeCtx(`m = Geometric(p = 0.4)`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    const k = m.samples[i];
    assert.ok(Number.isInteger(k) && k >= 0,
      `samples[${i}] = ${k} must be a non-negative integer`);
  }
});

test('Geometric(p=0.5): empirical mean ≈ (1-p)/p = 1', async () => {
  const ctx = makeCtx(`m = Geometric(p = 0.5)`);
  const m = await ctx.getMeasure('m');
  const mu = mean(m.samples);
  assert.ok(Math.abs(mu - 1) < 0.15,
    `Geometric(0.5) mean = ${mu}, expected ≈ 1`);
});

test('Geometric: pmf(0; p) = p, pmf(1; p) = p(1-p)', () => {
  const sampler = require('../sampler.ts');
  const reg = sampler._internal.REGISTRY.Geometric;
  const p = 0.3;
  assert.ok(Math.abs(reg.logpdfFn(0, p) - Math.log(p)) < 1e-12);
  assert.ok(Math.abs(reg.logpdfFn(1, p) - (Math.log(p) + Math.log(1 - p))) < 1e-12);
});

// ---------------------------------------------------------------------
// NegativeBinomial / NegativeBinomial2
// ---------------------------------------------------------------------

test('NegativeBinomial(alpha, beta): samples are non-negative integers', async () => {
  const ctx = makeCtx(`m = NegativeBinomial(alpha = 3, beta = 2)`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    const k = m.samples[i];
    assert.ok(Number.isInteger(k) && k >= 0,
      `samples[${i}] = ${k} must be a non-negative integer`);
  }
});

test('NegativeBinomial: empirical mean ≈ α/β', async () => {
  // alpha=3, beta=2 → mean = 1.5
  const ctx = makeCtx(`m = NegativeBinomial(alpha = 3, beta = 2)`);
  const m = await ctx.getMeasure('m');
  const mu = mean(m.samples);
  assert.ok(Math.abs(mu - 1.5) < 0.2,
    `NegativeBinomial mean = ${mu}, expected ≈ 1.5`);
});

test('NegativeBinomial2(mu, psi): empirical mean ≈ μ', async () => {
  const ctx = makeCtx(`m = NegativeBinomial2(mu = 4, psi = 2)`);
  const m = await ctx.getMeasure('m');
  const mu = mean(m.samples);
  assert.ok(Math.abs(mu - 4) < 0.4,
    `NegativeBinomial2(mu=4, psi=2) mean = ${mu}, expected ≈ 4`);
});

test('NegativeBinomial / NegativeBinomial2 agree at α=ψ, β=ψ/μ', () => {
  // pmf(k; α=ψ, β=ψ/μ) should equal pmf(k; μ, ψ).
  const sampler = require('../sampler.ts');
  const reg1 = sampler._internal.REGISTRY.NegativeBinomial;
  const reg2 = sampler._internal.REGISTRY.NegativeBinomial2;
  const mu = 5.0, psi = 3.0;
  const alpha = psi;
  const beta  = psi / mu;
  for (let k = 0; k < 15; k++) {
    const lp1 = reg1.logpdfFn(k, alpha, beta);
    const lp2 = reg2.logpdfFn(k, mu, psi);
    assert.ok(Math.abs(lp1 - lp2) < 1e-10,
      `k=${k}: NB.logpdf=${lp1} vs NB2.logpdf=${lp2}`);
  }
});

test('NegativeBinomial / NegativeBinomial2: finite logpdf for non-integer shape', () => {
  // Spec: alpha (NB) and psi (NB2) are elementof(posreals) — non-integer is valid, and
  // the generalized binomial coefficient is finite. Regression for the bug where the
  // integer-only binomcoefln returned NaN for non-integer shape (flatppl-js#20).
  // Oracle values are the closed-form spec densities, derived independently:
  //   NB(alpha=2.5, beta=1)  @ k=3  = ln Γ(5.5) - ln Γ(2.5) - ln Γ(4) + 2.5·ln(1/2) + 3·ln(1/2)
  //   NB2(mu=4,  psi=2.5)    @ k=3  = ln Γ(5.5) - ln Γ(4) - ln Γ(2.5) + 3·ln(4/6.5) + 2.5·ln(2.5/6.5)
  const sampler = require('../sampler.ts');
  const nb  = sampler._internal.REGISTRY.NegativeBinomial.logpdfFn(3, 2.5, 1.0);
  const nb2 = sampler._internal.REGISTRY.NegativeBinomial2.logpdfFn(3, 4.0, 2.5);
  assert.ok(Number.isFinite(nb),  `NB(2.5,1) logpdf must be finite, got ${nb}`);
  assert.ok(Number.isFinite(nb2), `NB2(4,2.5) logpdf must be finite, got ${nb2}`);
  assert.ok(Math.abs(nb  - (-1.9309378651619566)) < 1e-12, `NB(2.5,1)@3 logpdf = ${nb}`);
  assert.ok(Math.abs(nb2 - (-1.9639304319959507)) < 1e-12, `NB2(4,2.5)@3 logpdf = ${nb2}`);
});

test('Pareto(shape, scale): logpdf matches Distributions.jl, support is [scale, ∞)', () => {
  // Oracle: Pareto(α, θ) pdf = α·θ^α / x^(α+1) for x ≥ θ; logpdf values from
  // Distributions.jl Pareto(α=shape, θ=scale).
  const sampler = require('../sampler.ts');
  const reg = sampler._internal.REGISTRY.Pareto;
  const lp = (x: any, a: any, sc: any) => reg.logpdfFn(x, a, sc);
  assert.ok(Math.abs(lp(3.0, 2.5, 1.5) - (-1.915189508193818)) < 1e-12, 'Pareto(2.5,1.5)@3');
  assert.ok(Math.abs(lp(1.5, 2.5, 1.5) - (0.5108256237659909)) < 1e-12, 'Pareto(2.5,1.5)@scale = log(α/θ)');
  assert.ok(Math.abs(lp(5.0, 3.0, 2.0) - (-3.259697819388456)) < 1e-12, 'Pareto(3,2)@5');
  assert.equal(lp(1.0, 2.5, 1.5), -Infinity, 'below scale ⇒ -Inf');
});

test('Pareto(shape, scale): samples are all ≥ scale', async () => {
  const ctx = makeCtx(`m = Pareto(shape = 2.5, scale = 1.5)`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(m.samples[i] >= 1.5, `samples[${i}] = ${m.samples[i]} must be ≥ scale=1.5`);
  }
});

test('Pareto: Ctor pdf/cdf/quantile and both sampler branches (α=2.5, θ=1.5)', () => {
  const sampler = require('../sampler.ts');
  const R = sampler._internal.REGISTRY.Pareto;
  const d = new R.Ctor(2.5, 1.5);
  const close = (a: any, b: any) => Math.abs(a - b) < 1e-12;
  // pdf = (α/x)·(θ/x)^α ; below scale ⇒ 0
  assert.ok(close(d.pdf(3.0), (2.5 / 3.0) * Math.pow(1.5 / 3.0, 2.5)), 'pdf');
  assert.equal(d.pdf(1.0), 0, 'pdf below scale = 0');
  // cdf = 1 − (θ/x)^α ; below scale ⇒ 0; quantile inverts it
  assert.ok(close(d.cdf(3.0), 1 - Math.pow(1.5 / 3.0, 2.5)), 'cdf');
  assert.equal(d.cdf(1.0), 0, 'cdf below scale = 0');
  assert.equal(d.quantile(0), 1.5, 'quantile(0) = scale');
  assert.equal(d.quantile(1), Infinity, 'quantile(1) = ∞');
  assert.ok(close(d.quantile(d.cdf(3.0)), 3.0), 'quantile∘cdf = id');
  // sampler: static and parametric branches, deterministic prng=0.5, both ≥ scale
  const prng = () => 0.5;
  const expected = 1.5 * Math.pow(1 - 0.5, -1 / 2.5);
  assert.ok(close(R.randFn.factory(2.5, 1.5, { prng })(), expected), 'static sampler');
  assert.ok(close(R.randFn.factory({ prng })(2.5, 1.5), expected), 'parametric sampler');
});

// ---------------------------------------------------------------------
// Spec equivalences
// ---------------------------------------------------------------------

test('ChiSquared(k) ≡ Gamma(k/2, 0.5) in density', () => {
  const sampler = require('../sampler.ts');
  const regC = sampler._internal.REGISTRY.ChiSquared;
  const regG = sampler._internal.REGISTRY.Gamma;
  const k = 5;
  for (const x of [0.5, 1.0, 2.5, 8.0]) {
    const lpC = regC.logpdfFn(x, k);
    const lpG = regG.logpdfFn(x, k / 2, 0.5);
    assert.ok(Math.abs(lpC - lpG) < 1e-10,
      `x=${x}: ChiSquared.logpdf=${lpC} vs Gamma(k/2,0.5).logpdf=${lpG}`);
  }
});
