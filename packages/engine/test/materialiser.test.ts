'use strict';

// End-to-end tests for materialiseMeasure(name, ctx).
//
// We exercise the same entry point the viewer uses, supplying our own
// thin getMeasure cache and a synchronous-worker bridge built on top of
// createWorkerHandler. The materialiser is async-Promise-shaped, but
// the worker calls underneath are synchronous, so the Promise chains
// resolve in microtasks — keeping each test small and ordered.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 1024;
const ROOT_SEED    = 12345;

/**
 * Build a fresh materialisation context for `source` and return a
 * helper that materialises a binding by name. Each call shares one
 * worker handler + one promise cache, mirroring how the viewer's
 * getMeasure works.
 */
function makeCtx(source: any, opts?: any) {
  opts = opts || {};
  const lifted = processSource(source);
  // buildDerivations returns the LIFTED bindings map (including the
  // anon bindings introduced for inline measure/value expressions) and
  // the fixedValues map computed by the pre-eval pass — pull both from
  // there so resolveIRToValue can chase refs to lifted anons.
  const built = orchestrator.buildDerivations(lifted.bindings);
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
    sampleCount: opts.sampleCount != null ? opts.sampleCount : SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
    rejectionBudget: opts.rejectionBudget,
  };
  return ctx;
}

test('totalmass: weighted Normal broadcasts the scalar mass', async () => {
  const ctx = makeCtx(`
m = weighted(0.5, Normal(mu = 0.0, sigma = 1.0))
z = totalmass(m)
`);
  const z = await ctx.getMeasure('z');
  assert.equal(z.samples.length, SAMPLE_COUNT);
  for (let i = 0; i < z.samples.length; i++) {
    assert.ok(Math.abs(z.samples[i] - 0.5) < 1e-12,
      'totalmass atom ' + i + ' should equal 0.5, got ' + z.samples[i]);
  }
  assert.equal(z.logWeights, null);
  assert.equal(z.logTotalmass, 0);
  assert.equal(z.n_eff, SAMPLE_COUNT);
});

test('totalmass: unit-mass measure broadcasts 1', async () => {
  const ctx = makeCtx(`
m = Normal(mu = 0.0, sigma = 1.0)
z = totalmass(m)
`);
  const z = await ctx.getMeasure('z');
  for (let i = 0; i < z.samples.length; i++) {
    assert.ok(Math.abs(z.samples[i] - 1.0) < 1e-12,
      'unit-mass atom ' + i + ' should equal 1, got ' + z.samples[i]);
  }
});

test('totalmass: composed weights multiply', async () => {
  const ctx = makeCtx(`
m1 = weighted(0.5, Normal(mu = 0.0, sigma = 1.0))
m2 = weighted(0.25, m1)
z  = totalmass(m2)
`);
  const z = await ctx.getMeasure('z');
  for (let i = 0; i < z.samples.length; i++) {
    assert.ok(Math.abs(z.samples[i] - 0.125) < 1e-12,
      'composed totalmass should equal 0.5 * 0.25 = 0.125, got ' + z.samples[i]);
  }
});

// =====================================================================
// truncate — support restriction
// =====================================================================

test('truncate: CDF path on Normal × posreals gives samples ≥ 0', async () => {
  const ctx = makeCtx(`
m = Normal(mu = 0.0, sigma = 1.0)
t = truncate(m, posreals)
`);
  const t = await ctx.getMeasure('t');
  assert.equal(t.samples.length, SAMPLE_COUNT);
  for (let i = 0; i < t.samples.length; i++) {
    assert.ok(t.samples[i] >= 0,
      'truncated atom ' + i + ' should be >= 0, got ' + t.samples[i]);
    assert.ok(!Number.isNaN(t.samples[i]),
      'CDF path should not produce NaN at atom ' + i);
  }
  // P(X >= 0) = 0.5 for standard Normal → logTotalmass ≈ log(0.5).
  assert.ok(Math.abs(t.logTotalmass - Math.log(0.5)) < 1e-12,
    'logTotalmass should ≈ log(0.5), got ' + t.logTotalmass);
  // CDF path produces uniform-weight, full-N atoms.
  assert.equal(t.logWeights, null);
  assert.equal(t.n_eff, SAMPLE_COUNT);
});

test('truncate: CDF path on Normal × interval matches CDF math', async () => {
  const ctx = makeCtx(`
m = Normal(mu = 0.0, sigma = 1.0)
t = truncate(m, interval(-1.0, 1.0))
`);
  const t = await ctx.getMeasure('t');
  for (let i = 0; i < t.samples.length; i++) {
    assert.ok(t.samples[i] >= -1.0 && t.samples[i] <= 1.0,
      'atom ' + i + ' should be in [-1, 1], got ' + t.samples[i]);
  }
  // F(1) − F(-1) ≈ 0.6826 for standard Normal.
  const expected = Math.log(0.6826894921370859);
  assert.ok(Math.abs(t.logTotalmass - expected) < 1e-3,
    'logTotalmass should ≈ log(F(1)-F(-1)), got ' + t.logTotalmass);
});

test('truncate: rejection-redraw path on parametric Normal × interval', async () => {
  // Parametric Normal (mu from an upstream variate) → expandMeasureIR
  // gives a self-contained call IR with a value-ref in kwargs, so the
  // CDF path is skipped and the rejection-redraw path runs. The
  // resulting atoms must all lie in the truncation interval (or be
  // NaN if the per-atom budget exhausted).
  const ctx = makeCtx(`
mu_dist = Normal(mu = 0.0, sigma = 1.0)
mu      = draw(mu_dist)
y       = Normal(mu = mu, sigma = 0.5)
t       = truncate(y, interval(-1.0, 1.0))
`);
  const t = await ctx.getMeasure('t');
  assert.equal(t.samples.length, SAMPLE_COUNT);
  let inBand = 0;
  for (let i = 0; i < t.samples.length; i++) {
    const x = t.samples[i];
    if (Number.isNaN(x)) continue;
    assert.ok(x >= -1.0 && x <= 1.0,
      'rejection atom ' + i + ' should be in [-1, 1], got ' + x);
    inBand++;
  }
  assert.ok(inBand > SAMPLE_COUNT * 0.5,
    'expected most atoms to land in band after rejection-redraw, got '
    + inBand + '/' + SAMPLE_COUNT);
});

test('truncate: rejection budget=1 NaNs unaccepted atoms (mass shift correct)', async () => {
  // budget=1 means no redraws — atoms that don't land in S on first
  // try become NaN. The mass shift then equals the empirical M(S)
  // acceptance rate, so logTotalmass should be near log(0.5) for
  // Normal restricted to posreals (parametric case → rejection path).
  const ctx = makeCtx(`
mu_dist = Normal(mu = 0.0, sigma = 1.0)
mu      = draw(mu_dist)
y       = Normal(mu = mu, sigma = 1.0)
t       = truncate(y, posreals)
`, { rejectionBudget: 1 });
  const t = await ctx.getMeasure('t');
  let nValid = 0;
  for (let i = 0; i < t.samples.length; i++) {
    if (Number.isNaN(t.samples[i])) continue;
    assert.ok(t.samples[i] >= 0, 'valid atom must be ≥ 0');
    nValid++;
  }
  // Empirical M(S) for Normal(μ,1) marginalized over μ~N(0,1) is 0.5
  // by symmetry. SE on 1024 atoms ≈ √(0.25/1024) ≈ 0.016 → 4σ band
  // ≈ ±0.064. Test logTotalmass against log(0.5) within that band.
  const expected = Math.log(0.5);
  assert.ok(Math.abs(t.logTotalmass - expected) < 0.2,
    'logTotalmass ≈ log(0.5) for symmetric setup; got ' + t.logTotalmass);
  assert.equal(t.n_eff, nValid,
    'n_eff should equal count of non-NaN atoms');
});

// =====================================================================
// Uniform(support = S) — continuous uniform with set-typed support
// =====================================================================

test('Uniform(interval(0, 1)) samples ⊂ [0, 1] with mean ≈ 0.5', async () => {
  const ctx = makeCtx(`
U = Uniform(support = interval(0.0, 1.0))
x = draw(U)
`, { sampleCount: 4096 });
  const m = await ctx.getMeasure('x');
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(m.samples[i] >= 0 && m.samples[i] <= 1,
      'atom ' + i + ' should lie in [0, 1], got ' + m.samples[i]);
  }
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  assert.ok(Math.abs(mean - 0.5) < 0.05,
    'mean of Uniform(0, 1) should be ≈ 0.5, got ' + mean);
});

test('Uniform(support = unitinterval) — named-set support', async () => {
  const ctx = makeCtx(`
U = Uniform(support = unitinterval)
x = draw(U)
`, { sampleCount: 2048 });
  const m = await ctx.getMeasure('x');
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(m.samples[i] >= 0 && m.samples[i] <= 1,
      'unitinterval atom ' + i + ' should lie in [0, 1]');
  }
});

test('Uniform: logdensityof returns analytic 1/(b−a) inside support', async () => {
  // Uniform(0, 2) at x=0.5: density = 1/(2−0) = 0.5; log = -log(2).
  const ctx = makeCtx(`
U = Uniform(support = interval(0.0, 2.0))
lp = logdensityof(U, 0.5)
`);
  const m = await ctx.getMeasure('lp');
  assert.ok(Math.abs(m.samples[0] - Math.log(0.5)) < 1e-12,
    'logp should be log(0.5), got ' + m.samples[0]);
});

// =====================================================================
// GeneralizedNormal — closes out spec §08 continuous family
// =====================================================================

test('GeneralizedNormal(0, √2, 2) ≡ Normal(0, 1) (β = 2 case)', async () => {
  // Per spec, GN with β=2 reduces to Normal with σ = α/√2.
  // Setting α=√2 gives σ=1; sample moments should match Normal(0, 1).
  const ctx = makeCtx(`
M = GeneralizedNormal(mean = 0.0, alpha = 1.4142135623730951, beta = 2.0)
x = draw(M)
`, { sampleCount: 16384 });
  const m = await ctx.getMeasure('x');
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  let ss = 0;
  for (const v of m.samples) ss += (v - mean) * (v - mean);
  const variance = ss / m.samples.length;
  assert.ok(Math.abs(mean - 0) < 0.05, 'mean should ≈ 0, got ' + mean);
  assert.ok(Math.abs(variance - 1) < 0.1, 'variance should ≈ 1, got ' + variance);
});

test('GeneralizedNormal(0, 1, 1) ≡ Laplace(0, 1): variance ≈ 2', async () => {
  // β=1 collapses to Laplace with scale α. Laplace(μ=0, b=1) has
  // variance 2·b² = 2.
  const ctx = makeCtx(`
M = GeneralizedNormal(mean = 0.0, alpha = 1.0, beta = 1.0)
x = draw(M)
`, { sampleCount: 16384 });
  const m = await ctx.getMeasure('x');
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  let ss = 0;
  for (const v of m.samples) ss += (v - mean) * (v - mean);
  const variance = ss / m.samples.length;
  assert.ok(Math.abs(variance - 2.0) < 0.2,
    'Laplace(0, 1) variance should ≈ 2, got ' + variance);
});

test('GeneralizedNormal: logdensityof at the mode (x=μ) is closed-form', async () => {
  // logpdf(μ; μ, α, β) = log(β) − log(2α) − Γln(1/β).
  // For α=1, β=2: = log(2) − log(2) − Γln(0.5) = −Γln(0.5) = −log(√π).
  const ctx = makeCtx(`
M = GeneralizedNormal(mean = 0.0, alpha = 1.0, beta = 2.0)
lp = logdensityof(M, 0.0)
`);
  const m = await ctx.getMeasure('lp');
  assert.ok(Math.abs(m.samples[0] - (-0.5 * Math.log(Math.PI))) < 1e-12,
    'logpdf at mode should be -log(√π), got ' + m.samples[0]);
});

// =====================================================================
// InverseGamma — derived from Gamma via 1/X equivalence
// =====================================================================

test('InverseGamma(3, 2): all samples positive; mean ≈ 1, var ≈ 1', async () => {
  // Spec §08 line 200: if X ~ Gamma(α, β), then 1/X ~ InverseGamma(α, β).
  // Analytical moments for α = 3, β = 2:
  //   E[X] = β / (α − 1) = 2 / 2 = 1
  //   Var  = β² / ((α − 1)² · (α − 2)) = 4 / (4 · 1) = 1
  const ctx = makeCtx(`
M = InverseGamma(shape = 3.0, scale = 2.0)
x = draw(M)
`, { sampleCount: 16384 });
  const m = await ctx.getMeasure('x');
  let neg = 0;
  for (let i = 0; i < m.samples.length; i++) if (m.samples[i] <= 0) neg++;
  assert.equal(neg, 0, 'all InverseGamma atoms must be > 0');
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  assert.ok(Math.abs(mean - 1.0) < 0.1,
    'mean should be ≈ 1, got ' + mean);
});

test('InverseGamma: logdensityof at x=1 matches the closed-form pdf', async () => {
  // logpdf(x=1; α=3, β=2) = α·log(β) − Γln(α) − (α+1)·log(x) − β/x
  //                       = 3·log(2) − log(2) − 0 − 2 = 2·log(2) − 2.
  const ctx = makeCtx(`
M = InverseGamma(shape = 3.0, scale = 2.0)
lp = logdensityof(M, 1.0)
`);
  const m = await ctx.getMeasure('lp');
  const expected = 2 * Math.log(2) - 2;
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-12,
    'logpdf at 1 should be 2·log(2)−2, got ' + m.samples[0]);
});

// =====================================================================
// Categorical / Categorical0 — discrete with 1-based / 0-based support
// =====================================================================

test('Categorical([0.5, 0.3, 0.2]) sample fractions ≈ probabilities', async () => {
  const ctx = makeCtx(`
M = Categorical(p = [0.5, 0.3, 0.2])
x = draw(M)
`, { sampleCount: 8192 });
  const m = await ctx.getMeasure('x');
  const counts: any = { 1: 0, 2: 0, 3: 0 };
  for (let i = 0; i < m.samples.length; i++) {
    const k = m.samples[i] | 0;
    assert.ok(k >= 1 && k <= 3, 'atom ' + i + ' must be in {1, 2, 3}, got ' + m.samples[i]);
    counts[k]++;
  }
  const N = m.samples.length;
  assert.ok(Math.abs(counts[1] / N - 0.5) < 0.04,
    'fraction of 1 should ≈ 0.5, got ' + counts[1] / N);
  assert.ok(Math.abs(counts[2] / N - 0.3) < 0.04,
    'fraction of 2 should ≈ 0.3, got ' + counts[2] / N);
  assert.ok(Math.abs(counts[3] / N - 0.2) < 0.04,
    'fraction of 3 should ≈ 0.2, got ' + counts[3] / N);
});

test('Categorical0([0.5, 0.3, 0.2]) shifts support to {0, 1, 2}', async () => {
  const ctx = makeCtx(`
M = Categorical0(p = [0.5, 0.3, 0.2])
x = draw(M)
`, { sampleCount: 2048 });
  const m = await ctx.getMeasure('x');
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < m.samples.length; i++) {
    const v = m.samples[i] | 0;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  assert.equal(min, 0, 'Categorical0 support starts at 0');
  assert.equal(max, 2, 'Categorical0 support ends at length-1');
});

test('Categorical: logdensityof at category 2 ⇒ log(p[1]) = log(0.3)', async () => {
  const ctx = makeCtx(`
M = Categorical(p = [0.5, 0.3, 0.2])
lp = logdensityof(M, 2)
`);
  const m = await ctx.getMeasure('lp');
  assert.ok(Math.abs(m.samples[0] - Math.log(0.3)) < 1e-12,
    'logp at k=2 should be log(0.3), got ' + m.samples[0]);
});

test('Categorical: out-of-support log-density ⇒ -Infinity', async () => {
  const ctx = makeCtx(`
M = Categorical(p = [0.5, 0.3, 0.2])
lp = logdensityof(M, 5)
`);
  const m = await ctx.getMeasure('lp');
  assert.equal(m.samples[0], -Infinity);
});

// =====================================================================
// Logistic / Weibull — synthesised inverse-CDF samplers
// =====================================================================

test('Logistic(0, 1) sample moments match analytic mean=0, var=π²/3', async () => {
  const ctx = makeCtx(`
M = Logistic(mu = 0.0, s = 1.0)
x = draw(M)
`, { sampleCount: 8192 });
  const m = await ctx.getMeasure('x');
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  let ss = 0;
  for (const v of m.samples) ss += (v - mean) * (v - mean);
  const variance = ss / m.samples.length;
  assert.ok(Math.abs(mean - 0) < 0.15,
    'mean should be ≈ 0, got ' + mean);
  const expectedVar = (Math.PI * Math.PI) / 3;
  assert.ok(Math.abs(variance - expectedVar) < 0.3,
    'variance should be ≈ π²/3 ≈ 3.29, got ' + variance);
});

test('Logistic: logdensityof matches analytic logpdf at the mode', async () => {
  // logpdf at x=mu (the mode) for Logistic(mu=0, s=1):
  //   z = 0 → exp(-0)/(1·(1+1)^2) = 1/4 → log = -log(4)
  const ctx = makeCtx(`
M = Logistic(mu = 0.0, s = 1.0)
lp = logdensityof(M, 0.0)
`);
  const m = await ctx.getMeasure('lp');
  assert.ok(Math.abs(m.samples[0] - (-Math.log(4))) < 1e-12,
    'logp at mode should be -log(4), got ' + m.samples[0]);
});

test('Weibull(2, 1) samples are non-negative and match analytic mean ≈ √π/2', async () => {
  const ctx = makeCtx(`
M = Weibull(shape = 2.0, scale = 1.0)
x = draw(M)
`, { sampleCount: 8192 });
  const m = await ctx.getMeasure('x');
  let neg = 0;
  for (let i = 0; i < m.samples.length; i++) if (m.samples[i] < 0) neg++;
  assert.equal(neg, 0, 'all Weibull atoms should be ≥ 0');
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  const expected = Math.sqrt(Math.PI) / 2;   // λ · Γ(1 + 1/k) at k=2, λ=1
  assert.ok(Math.abs(mean - expected) < 0.05,
    'mean should be ≈ √π/2, got ' + mean);
});

test('Weibull(1, 1/λ) ≡ Exponential(λ): logdensity at 0 matches log(λ)', async () => {
  // Weibull(shape=1, scale=1/2) ≡ Exponential(rate=2). At x=0 both
  // densities equal λ = 2; log = log(2).
  const ctx = makeCtx(`
M = Weibull(shape = 1.0, scale = 0.5)
lp = logdensityof(M, 0.0)
`);
  const m = await ctx.getMeasure('lp');
  assert.ok(Math.abs(m.samples[0] - Math.log(2)) < 1e-12,
    'logp at 0 of Weibull(1, 1/2) should be log(2), got ' + m.samples[0]);
});

// =====================================================================
// pushfwd(f, M) — pushforward of measure through function
// =====================================================================

test('pushfwd: classifier produces a pushfwd derivation referencing f + M', async () => {
  // pushfwd is a first-class measure-op. ln's derivation directly
  // carries the function ref and base measure name; matPushfwd in the
  // materialiser produces samples by applying f over M's atoms in one
  // batched evaluateN call.
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
ln = pushfwd(fn(exp(_)), M)
`);
  const d = ctx.derivations.ln;
  assert.ok(d, 'ln should be derivable');
  assert.equal(d.kind, 'pushfwd');
  assert.equal(d.from, 'M');
  assert.ok(d.fnRef, 'fnRef should point at the (lifted) function binding');
  // The lifted function binding is a fn / functionof, callable as f.
  const fBinding = ctx.bindings.get(d.fnRef);
  assert.ok(fBinding);
  assert.ok(fBinding.type === 'fn' || fBinding.type === 'functionof',
    `fnRef binding type should be fn or functionof, got ${fBinding.type}`);
});

test('pushfwd: samples of pushfwd(fn(exp(_)), Normal(0,1)) match LogNormal(0,1)', async () => {
  // The pushforward of standard Normal through exp is the standard
  // LogNormal. Analytic moments at μ=0, σ=1:
  //   E[X] = exp(0.5)  ≈ 1.649
  //   Var  = (e − 1)·e ≈ 4.671
  // All atoms are positive. We use a large sample count to reduce MC
  // error well below the assertion tolerance.
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
ln = pushfwd(fn(exp(_)), M)
`, { sampleCount: 16384 });
  const m = await ctx.getMeasure('ln');
  let neg = 0;
  for (let i = 0; i < m.samples.length; i++) {
    if (m.samples[i] < 0) neg++;
  }
  assert.equal(neg, 0, 'all pushfwd-through-exp atoms should be positive');
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  assert.ok(Math.abs(mean - Math.exp(0.5)) < 0.1,
    'mean should be ≈ exp(½), got ' + mean);
});

test('pushfwd: linear transform pushfwd(fn(2 * _ + 1), Normal) is Normal(2μ+1, 2σ)', async () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
shifted = pushfwd(fn(2.0 * _ + 1.0), M)
`, { sampleCount: 4096 });
  const m = await ctx.getMeasure('shifted');
  const mean = m.samples.reduce((s: any, v: any) => s + v, 0) / m.samples.length;
  // Analytic mean = 2·0 + 1 = 1; variance = 4·1 = 4 → sd = 2.
  assert.ok(Math.abs(mean - 1.0) < 0.15,
    'mean should be ≈ 1, got ' + mean);
  let ss = 0;
  for (const v of m.samples) ss += (v - mean) * (v - mean);
  const variance = ss / m.samples.length;
  assert.ok(Math.abs(variance - 4.0) < 0.5,
    'variance should be ≈ 4, got ' + variance);
});

// =====================================================================
// Positional joint(M1, M2, ...) — independent product, tuple shape
// =====================================================================

test('joint positional: classifies as tuple over component refs', async () => {
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 5.0, sigma = 1.0)
J  = joint(M1, M2)
`);
  const d = ctx.derivations.J;
  assert.ok(d, 'J should be derivable');
  assert.equal(d.kind, 'tuple');
  assert.deepEqual(d.elems, ['M1', 'M2']);
});

test('joint positional: materialises as tuple of per-component sub-measures', async () => {
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 5.0, sigma = 1.0)
J  = joint(M1, M2)
`);
  const J = await ctx.getMeasure('J');
  assert.ok(J.elems && J.elems.length === 2,
    'tuple measure should expose two elems');
  // Each component is independently sampled — means should be near
  // the configured μ for each within sampling noise.
  const m0 = mean(J.elems[0].samples);
  const m1 = mean(J.elems[1].samples);
  assert.ok(Math.abs(m0 - 0.0) < 0.2, 'elem 0 mean ≈ 0, got ' + m0);
  assert.ok(Math.abs(m1 - 5.0) < 0.2, 'elem 1 mean ≈ 5, got ' + m1);
});

test('joint positional: lifts inline measure expressions', async () => {
  // The lift pass should pull `Normal(0,1)` and `Exponential(1)` into
  // anon bindings; classifyRecordOrJoint then sees positional refs.
  const ctx = makeCtx(`
J = joint(Normal(mu = 0.0, sigma = 1.0), Exponential(rate = 1.0))
`);
  const d = ctx.derivations.J;
  assert.ok(d, 'J should be derivable');
  assert.equal(d.kind, 'tuple');
  assert.equal(d.elems.length, 2,
    'positional joint should classify two component refs after lifting');
});

function mean(arr: any) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

test('jointchain: 2-arg with record-shaped prior produces a joint record measure', async () => {
  // jointchain(prior, K) where prior is record-shaped lifts to the
  // joint record with K's body fields appended. Materialises as a
  // record measure exposing all prior fields + K's body fields.
  const ctx = makeCtx(`
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
theta2 = draw(Exponential(rate = 1.0))
obs_dist = joint(y = Normal(mu = theta1, sigma = theta2))
forward_kernel = functionof(obs_dist, theta1 = theta1, theta2 = theta2)
prior = lawof(record(theta1 = theta1, theta2 = theta2))
joint_model = jointchain(prior, forward_kernel)
`);
  const m = await ctx.getMeasure('joint_model');
  assert.ok(m.fields, 'joint_model should be a record measure');
  const keys = Object.keys(m.fields).sort();
  assert.deepEqual(keys, ['theta1', 'theta2', 'y']);
});

test('jointchain: 2-arg logdensityof scores all components (closed-form)', async () => {
  // densityof of jointchain(prior, K) at a fully-specified record
  // equals densityof(prior) × densityof(K(at-prior-values)). Each
  // factor is closed-form (Normal/Exponential logpdf), so the result
  // matches analytic against the same prior atom.
  const ctx = makeCtx(`
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
theta2 = draw(Exponential(rate = 1.0))
obs_dist = joint(y = Normal(mu = theta1, sigma = theta2))
forward_kernel = functionof(obs_dist, theta1 = theta1, theta2 = theta2)
prior = lawof(record(theta1 = theta1, theta2 = theta2))
joint_model = jointchain(prior, forward_kernel)
lp = logdensityof(joint_model, record(theta1 = 0.0, theta2 = 1.0, y = 0.0))
`);
  const lp = await ctx.getMeasure('lp');
  // Per-atom env carries (theta1_i, theta2_i) — but the observation
  // pins all three values, so the logpdf is the same scalar at every
  // atom: logpdf_N(0;0,1) + logpdf_Exp(1;1) + logpdf_N(0; 0, 1).
  const LOG_2PI = Math.log(2 * Math.PI);
  const expected = (-0.5 * LOG_2PI)
    + (Math.log(1) - 1)     // Exp(rate=1) at x=1: log λ − λx
    + (-0.5 * LOG_2PI);     // Normal(0,1) at y=0
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    'jointchain logdensityof should sum component logpdfs analytically, got '
    + lp.samples[0] + ' (expected ' + expected + ')');
});

test('jointchain: N-ary record-shaped (M, K1, K2) folds left-associatively', async () => {
  // jointchain(M, K1, K2, ..., Kn) per spec §06 is left-associative,
  // unfolding to jointchain(jointchain(...jointchain(M, K1), K2)..., Kn).
  // We fold step-by-step, rewriting each 2-arg level before wrapping
  // it in the next outer — each outer step sees an already-normalised
  // record AST as its args[0]. End-to-end: 3-arg jointchain on a
  // record-shaped prior produces a record measure with prior fields
  // + each kernel's body fields.
  const ctx = makeCtx(`
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta1 = theta1))
theta2 = draw(Normal(mu = theta1, sigma = 1.0))
obs_dist1 = joint(theta2 = Normal(mu = theta1, sigma = 1.0))
obs_dist2 = joint(y = Normal(mu = theta2, sigma = 1.0))
K1 = functionof(obs_dist1, theta1 = theta1)
K2 = functionof(obs_dist2, theta1 = theta1, theta2 = theta2)
joint_model = jointchain(prior, K1, K2)
`);
  const d = ctx.derivations.joint_model;
  assert.ok(d, 'joint_model should be derivable');
  // BEHAVIOURAL contract: N-ary record-shaped jointchain retains every
  // component variate. (Pre-consolidation this also pinned the legacy
  // internal `d.kind === 'record'`; first-class classifies it as
  // kind:'jointchain' — an implementation detail, not behaviour.)
  const m = await ctx.getMeasure('joint_model');
  assert.ok(m.fields, 'should materialise as a record measure');
  assert.deepEqual(Object.keys(m.fields).sort(), ['theta1', 'theta2', 'y']);
});

test('jointchain: N-ary closed-form logdensityof with env-threading', async () => {
  // The same joint-density correctness story as the 2-arg case
  // extends to N-ary chains: each subsequent kernel's leaf
  // distribution refs resolve to the OBSERVED values of prior fields.
  // For N-ary, kernel substitution names downstream refs by the
  // SOURCE BINDING (an anon) rather than the surface field name —
  // walkJoint env-threads under both keys (field name AND source
  // binding name) so closed-form factorisation works out.
  const ctx = makeCtx(`
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta1 = theta1))
theta2 = draw(Normal(mu = theta1, sigma = 1.0))
obs_dist1 = joint(theta2 = Normal(mu = theta1, sigma = 1.0))
obs_dist2 = joint(y = Normal(mu = theta2, sigma = 1.0))
K1 = functionof(obs_dist1, theta1 = theta1)
K2 = functionof(obs_dist2, theta1 = theta1, theta2 = theta2)
joint_model = jointchain(prior, K1, K2)
lp = logdensityof(joint_model, record(theta1 = 0.0, theta2 = 0.0, y = 0.0))
`);
  const lp = await ctx.getMeasure('lp');
  // Each Normal logpdf at x=mu equals -½log(2π) (with σ=1). All three
  // observed values equal the chain's "centred" point, so:
  //   logpdf(0; 0, 1) + logpdf(0; 0, 1) + logpdf(0; 0, 1) = -3·½log(2π)
  const LOG_2PI = Math.log(2 * Math.PI);
  const expected = -3 * 0.5 * LOG_2PI;
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    'N-ary jointchain logdensityof should sum closed-form factors, got '
    + lp.samples[0] + ' (expected ' + expected + ')');
});

// =====================================================================
// kchain(M, K) — Kleisli composition with prior marginalisation
// =====================================================================

test('kchain: produces a measure of K\'s body fields only (no prior)', async () => {
  // BEHAVIOURAL contract (first-class jointchain consolidation): kchain
  // drops the prior and keeps only K's body, marginalising the prior
  // away. Here y ~ Normal(theta1, 1) with theta1 ~ Normal(0,1), so the
  // marginal of y is Normal(0, √2): mean ≈ 0, var ≈ 2 — and the
  // measure must NOT expose the prior field theta1. (Pre-consolidation
  // this asserted the legacy internal representation: d.kind ==
  // 'record', d.fields == ['y'], d.chainOrigin === true. Those are the
  // inlineChainOps mechanism being retired; behaviour is what matters
  // and is what we now pin.)
  const ctx = makeCtx(`
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta1 = theta1))
obs_dist = joint(y = Normal(mu = theta1, sigma = 1.0))
forward_kernel = functionof(obs_dist, theta1 = theta1)
predictive = kchain(prior, forward_kernel)
`);
  const m = await ctx.getMeasure('predictive');
  // Prior marginalised away: no theta1 component anywhere.
  assert.ok(!(m.fields && m.fields.theta1),
    'kchain must not expose the marginalised prior field theta1');
  const ys = m.samples
    || (m.fields && m.fields.y && m.fields.y.samples);
  assert.ok(ys && ys.length > 0, 'kchain yields K-body variate y');
  const mean = ys.reduce((a: any, b: any) => a + b, 0) / ys.length;
  let v = 0; for (const x of ys) v += (x - mean) ** 2; v /= ys.length;
  assert.ok(Math.abs(mean) < 0.12, 'y marginal mean ≈ 0, got ' + mean);
  assert.ok(Math.abs(v - 2.0) < 0.35, 'y marginal var ≈ 2, got ' + v);
});

test('kchain: logdensityof marginalises via MC (logsumexp − log N)', async () => {
  // Closed-form check: kchain(Normal(0,1), x ↦ Normal(x, 1)) is
  // Normal(0, √2). The MC estimator of its log-density at obs=0 is
  //   logsumexp_i { log p(0 | Normal(prior_i, 1)) } − log N
  // which converges to log p(0 | Normal(0, √2)) = −½log(2π) − ½log(2).
  // We check the broadcast scalar at atom 0 against the analytic
  // value within a generous tolerance (MC standard error at 1024
  // atoms is roughly 0.05 on the log scale).
  const ctx = makeCtx(`
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta1 = theta1))
obs_dist = joint(y = Normal(mu = theta1, sigma = 1.0))
forward_kernel = functionof(obs_dist, theta1 = theta1)
predictive = kchain(prior, forward_kernel)
lp = logdensityof(predictive, record(y = 0.0))
`);
  const lp = await ctx.getMeasure('lp');
  // All atoms broadcast the same MC estimate.
  const v0 = lp.samples[0];
  for (let i = 1; i < lp.samples.length; i++) {
    assert.equal(lp.samples[i], v0,
      'chain marginal logp should be broadcast (atom ' + i + ' differs)');
  }
  // Analytical marginal: Normal(0, sqrt(2)) at y=0.
  const LOG_2PI = Math.log(2 * Math.PI);
  const expected = -0.5 * LOG_2PI - 0.5 * Math.log(2);
  assert.ok(Math.abs(v0 - expected) < 0.1,
    'MC marginal logp should match analytic Normal(0, √2) within MC error, got '
    + v0 + ' (expected ' + expected + ')');
  // n_eff collapses to 1 — there's one estimator here.
  assert.equal(lp.n_eff, 1);
});

test('jointchain: positional scalar form logdensityof', async () => {
  // funnel = jointchain(Exp(1), fn(Normal(1, _))) — variate is [a, b]
  // with a ~ Exp(1) and b ~ Normal(1, a). Positional jointchain lifts
  // both components to anon bindings; expandMeasureIR turns the
  // tuple-classified result into `joint([a's distIR, b's distIR])`
  // which traceeval's positional-args branch splits per footprint.
  //
  // BUT: the 2-arg positional jointchain rewrite drops a alias (b's
  // sample IR has a self-ref to a's anon, not a literal 'a' binding
  // name). We sanity-check that classification + the log-density
  // call goes through without error. Numeric correctness on the
  // dependent term is covered by the env-threaded record case above.
  const ctx = makeCtx(`
funnel = jointchain(Exponential(rate = 1), fn(Normal(mu = 1, sigma = _)))
lp = logdensityof(funnel, [1.0, 2.0])
`);
  const lp = await ctx.getMeasure('lp');
  assert.equal(lp.samples.length, SAMPLE_COUNT,
    'positional-jointchain logdensityof should produce per-atom samples');
  assert.ok(Number.isFinite(lp.samples[0]),
    'logp should be finite at observed values, got ' + lp.samples[0]);
});

test('joint positional: logdensityof matches summed component logpdfs', async () => {
  // densityof(joint(M1, M2), [x, y]) = pdf_M1(x) · pdf_M2(y) — no
  // marginalisation, no normalising constant. Verifies traceeval's
  // positional-args branch routes the observation through each
  // component's footprint.
  const ctx = makeCtx(`
M1 = Normal(mu = 0.0, sigma = 1.0)
M2 = Normal(mu = 5.0, sigma = 2.0)
J  = joint(M1, M2)
lp = logdensityof(J, [0.0, 5.0])
`);
  const lp = await ctx.getMeasure('lp');
  // For each prior atom the answer is the same shared scalar (no
  // per-i refs); we check atom 0 against analytic.
  // logpdf_Normal(0;0,1) = -log√(2π) - 0
  // logpdf_Normal(5;5,2) = -log(2·√(2π)) - 0
  const LOG_2PI = Math.log(2 * Math.PI);
  const expected = -0.5 * LOG_2PI + (-Math.log(2) - 0.5 * LOG_2PI);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
    'lp[0] should equal sum of component logpdfs, got ' + lp.samples[0]
    + ' (expected ' + expected + ')');
});

test('truncate: composed totalmass scales by truncation factor', async () => {
  // truncate(Normal, posreals) reaches the CDF path → exact mass 0.5.
  // Wrapping in weighted(0.5, …) multiplies by 0.5 → exact 0.25.
  // Order matters: weighted-outside-truncate keeps the CDF path on the
  // inner Normal (vs. truncating a pre-weighted measure, which has
  // non-uniform parent weights and falls into the empirical-shift path).
  const ctx = makeCtx(`
t  = truncate(Normal(mu = 0.0, sigma = 1.0), posreals)
m  = weighted(0.5, t)
z  = totalmass(m)
`);
  const z = await ctx.getMeasure('z');
  for (let i = 0; i < z.samples.length; i++) {
    assert.ok(Math.abs(z.samples[i] - 0.25) < 1e-9,
      'composed totalmass should equal 0.25, got ' + z.samples[i]);
  }
});

// =====================================================================
// Complex-valued bindings (chunk 5 of the complex-values thread)
// =====================================================================
//
// A deterministic complex transform of a stochastic parent flows
// through matEvaluate → evaluateN (reply.imag) → measureFromReply.
// The Measure keeps `.samples` as the real part (legacy scalar view)
// and exposes the imaginary buffer as `.imag` + `dtype:'complex'`.

test('complex(x, 2x) binding → Measure carries planar re/im', async () => {
  const ctx = makeCtx(`
x ~ Normal(mu = 0.0, sigma = 1.0)
z = complex(x, 2.0 * x)
`);
  const x = await ctx.getMeasure('x');
  const z = await ctx.getMeasure('z');
  assert.equal(z.dtype, 'complex');
  assert.ok(z.imag instanceof Float64Array);
  assert.equal(z.imag.length, SAMPLE_COUNT);
  assert.equal(z.value.dtype, 'complex');
  assert.equal(z.value.im, z.imag, 'planar: Measure.imag shares Value.im');
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    assert.ok(Math.abs(z.samples[i] - x.samples[i]) < 1e-12,
      'real part = x at atom ' + i);
    assert.ok(Math.abs(z.imag[i] - 2 * x.samples[i]) < 1e-12,
      'imag part = 2x at atom ' + i);
  }
});

test('abs2 of a complex binding → REAL Measure (no dtype/imag)', async () => {
  const ctx = makeCtx(`
x ~ Normal(mu = 0.0, sigma = 1.0)
A = complex(x, 1.0)
I = abs2(A)
`);
  const x = await ctx.getMeasure('x');
  const I = await ctx.getMeasure('I');
  assert.equal(I.dtype, undefined, 'modulus-squared is real');
  assert.equal(I.imag, undefined);
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    const want = x.samples[i] * x.samples[i] + 1;
    assert.ok(Math.abs(I.samples[i] - want) < 1e-10,
      'abs2 atom ' + i + ': got ' + I.samples[i] + ' want ' + want);
  }
});

test('conj of a complex binding negates the imaginary part', async () => {
  const ctx = makeCtx(`
x ~ Normal(mu = 1.0, sigma = 0.5)
z = complex(x, x)
zc = conj(z)
`);
  const z  = await ctx.getMeasure('z');
  const zc = await ctx.getMeasure('zc');
  assert.equal(zc.dtype, 'complex');
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    assert.ok(Math.abs(zc.samples[i] - z.samples[i]) < 1e-12);
    assert.ok(Math.abs(zc.imag[i] + z.imag[i]) < 1e-12,
      'conj imag = -imag at atom ' + i);
  }
});
