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
const { createWorkerHandler } = require('../worker');

const SAMPLE_COUNT = 1024;
const ROOT_SEED    = 12345;

/**
 * Build a fresh materialisation context for `source` and return a
 * helper that materialises a binding by name. Each call shares one
 * worker handler + one promise cache, mirroring how the viewer's
 * getMeasure works.
 */
function makeCtx(source, opts) {
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

function mean(arr) {
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
  assert.equal(d.kind, 'record');
  assert.deepEqual(Object.keys(d.fields).sort(), ['theta1', 'theta2', 'y']);
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
// chain(M, K) — Kleisli composition with prior marginalisation
// =====================================================================

test('chain: produces a measure of K\'s body fields only (no prior)', async () => {
  // chain(M, K) drops the prior fields and keeps only K's body —
  // semantically marginalising the prior away. The materialiser
  // samples K(prior_atom_i) per atom; the binding's chainOrigin
  // flag survives so density evaluation knows to MC-marginalise.
  const ctx = makeCtx(`
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta1 = theta1))
obs_dist = joint(y = Normal(mu = theta1, sigma = 1.0))
forward_kernel = functionof(obs_dist, theta1 = theta1)
predictive = chain(prior, forward_kernel)
`);
  const d = ctx.derivations.predictive;
  assert.ok(d, 'predictive should be derivable');
  assert.equal(d.kind, 'record');
  assert.deepEqual(Object.keys(d.fields), ['y']);
  assert.ok(d.chainOrigin === true,
    'chainOrigin flag should be set so matLogdensityof picks up marginalisation');
});

test('chain: logdensityof marginalises via MC (logsumexp − log N)', async () => {
  // Closed-form check: chain(Normal(0,1), x ↦ Normal(x, 1)) is
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
predictive = chain(prior, forward_kernel)
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
