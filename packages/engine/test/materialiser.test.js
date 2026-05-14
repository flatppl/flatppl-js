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
  const { derivations } = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });

  const cache = new Map();
  const ctx = {
    derivations,
    bindings:    lifted.bindings,
    fixedValues: lifted.fixedValues || new Map(),
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
