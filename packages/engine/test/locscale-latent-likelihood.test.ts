'use strict';

// pushfwd/bijection density with LATENT (kernel-fed) bijection params
// (TODO-flatppl-js.md "latent kernel-fed bijection params"). A
// location-scale Student-t likelihood with latent location+scale routes
// through `locscale` → `pushfwd(<affine bijection>, StudentT)`, whose
// fwd/inv/logvolume bodies reference the latent loc/scale. In a likelihood
// kernel those latents arrive per-atom in refArrays, so the bijection must
// be evaluated per atom (see density.walkPushfwd).
//
// Oracle (INDEPENDENT — scipy.stats.t, not the engine):
//   t.logpdf(1.5, df=3, loc=0.5, scale=2.0) = -1.8541214455305277
//   t.logpdf(3.0, df=3, loc=0.5, scale=2.0) = -2.5325528906644554
// (== t3.logpdf((y-0.5)/2.0) - log(2.0).)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

// buildCtx that SURFACES worker errors — a masked worker error would
// otherwise crash later (mat-density applyReduce on undefined samples)
// and hide the real `unbound self reference` message.
function buildCtx(src: any, N: any) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [], 'unexpected diagnostics: ' + JSON.stringify(errs));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    getMeasure: (n: any) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => {
      const r = w.handle(m);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
  };
  return ctx;
}

const LATENT_SRC = (obsVal: any) =>
  `mu ~ Normal(0, 10)\n`
  + `sigma ~ Exponential(1)\n`
  + `obs ~ locscale(StudentT(3.0), mu, sigma)\n`
  + `fk = kernelof(record(obs = obs), mu = mu, sigma = sigma)\n`
  + `observed_data = ${obsVal}\n`
  + `L = likelihoodof(fk, record(obs = observed_data))\n`
  + `ld = logdensityof(L, record(mu = 0.5, sigma = 2.0))\n`;

test('locscale likelihood with latent loc+scale scores against scipy oracle (y=1.5)', async () => {
  const ctx = buildCtx(LATENT_SRC('1.5'), 64);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - (-1.8541214455305277)) < 1e-12,
    `got ${m.samples[0]}, expected -1.8541214455305277`);
});

test('locscale likelihood with latent loc+scale scores against scipy oracle (y=3.0)', async () => {
  const ctx = buildCtx(LATENT_SRC('3.0'), 64);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - (-2.5325528906644554)) < 1e-12,
    `got ${m.samples[0]}, expected -2.5325528906644554`);
});

test('zero-axis locscale broadcast holds its measure constant and scores live scalar parameters', async () => {
  // §04: a measure is held constant, and no collections means one call.
  // Both spellings therefore give Normal(mu + 0.5, 2*tau), with the
  // likelihood's explicit parameter point supplying mu=0.5 and tau=2.
  const oracle = -0.5 * Math.log(2 * Math.PI) - Math.log(4) - 0.5 * (0.5 / 4) ** 2;
  for (const expression of [
    'locscale.(base_measure, mu + 0.5, 2 * tau)',
    'broadcast(locscale, Normal(0, 1), mu + 0.5, 2 * tau)',
  ]) {
    const ctx = buildCtx(
      'mu ~ Normal(0, 10)\n'
      + 'tau ~ LogNormal(0, 1)\n'
      + 'base_measure = Normal(0, 1)\n'
      + `obs ~ ${expression}\n`
      + 'fk = kernelof(record(obs = obs), mu = mu, tau = tau)\n'
      + 'L = likelihoodof(fk, record(obs = 1.5))\n'
      + 'ld = logdensityof(L, record(mu = 0.5, tau = 2))\n', 8);
    const score = await ctx.getMeasure('ld');
    assert.ok(Math.abs(score.samples[0] - oracle) < 1e-12);
    const draws = await ctx.getMeasure('obs');
    assert.equal(draws.samples.length, 8);
    assert.ok(Array.from(draws.samples).every(Number.isFinite));
  }
  const vectorBase = processSource('M = locscale.(iid(Normal(0, 1), 2), 1, 4)');
  assert.ok(vectorBase.diagnostics.some((d: any) => d.severity === 'error'));
  // The iid variate is held constant, not broadcast over. Put the scalar
  // transform inside iid to obtain two independent Normal(1, 4) coordinates.
  const iid = buildCtx('M = iid(locscale.(Normal(0, 1), 1, 4), 2)\n'
    + 'ld = logdensityof(M, [1.5, 1.5])\n', 8);
  const jointScore = await iid.getMeasure('ld');
  assert.ok(Math.abs(jointScore.samples[0] - 2 * oracle) < 1e-12);
});

test('locscale broadcast respects a user callable and an explicit base constructor', async () => {
  const ctx = buildCtx(
    'locscale(m, a, b) = a + b\n'
    + 'y = locscale.(Normal(0, 1), 0, 0)\n'
    + 'M = base.locscale.(Normal(0, 1), 0, 2)\n'
    + 'ld = logdensityof(M, 1)\n', 8);
  const y = await ctx.getMeasure('y');
  assert.equal(y.samples[0], 0);
  const degenerate = processSource('M = base.locscale.(Normal(0, 1), 0, 0)');
  assert.ok(degenerate.diagnostics.some((d: any) => d.severity === 'error'));
  const score = await ctx.getMeasure('ld');
  const oracle = -0.5 * Math.log(2 * Math.PI) - Math.log(2) - 0.5 * (1 / 2) ** 2;
  assert.ok(Math.abs(score.samples[0] - oracle) < 1e-12);
});

// Translation has zero log-volume. Its explicit shift is a kernel input.
// The independent Normal(0.5,1) oracle at 1.5 is -1.4189385332046727.
// Keep an unused second input to exercise the multiple-input binding path.
test('per-atom translation in a likelihood with a latent shift', async () => {
  const ctx = buildCtx(
    `shift ~ Normal(0.0, 1.0)\n`
    + `pad ~ Normal(0.0, 1.0)\n`
    + `obs ~ locscale(Normal(0.0, 1.0), shift, 1.0)\n`
    + `fk = kernelof(record(obs = obs), shift = shift, pad = pad)\n`
    + `observed_data = 1.5\n`
    + `L = likelihoodof(fk, record(obs = observed_data))\n`
    + `ld = logdensityof(L, record(shift = 0.5, pad = 0.0))\n`, 64);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - (-1.4189385332046727)) < 1e-12,
    `got ${m.samples[0]}, expected -1.4189385332046727`);
});

// The later field reads the earlier observation through the density overlay.
// Independent oracle: norm.logpdf(1.5,0,1) + norm.logpdf(0.7,2,1)
// = -3.8078770664093455.
test('per-atom translation with an earlier observed field in its base', async () => {
  const ctx = buildCtx(
    `shift ~ Normal(0.0, 1.0)\n`
    + `pad ~ Normal(0.0, 1.0)\n`
    + `a ~ Normal(0.0, 1.0)\n`
    + `b ~ locscale(Normal(a, 1.0), shift, 1.0)\n`
    + `fk = kernelof(record(a = a, b = b), shift = shift, pad = pad)\n`
    + `L = likelihoodof(fk, record(a = 1.5, b = 0.7))\n`
    + `ld = logdensityof(L, record(shift = 0.5, pad = 0.0))\n`, 64);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - (-3.8078770664093455)) < 1e-12,
    `got ${m.samples[0]}, expected -3.8078770664093455`);
});

// Constant (literal) locscale in a likelihood: bijection bodies reference
// no latent param, so needsPerAtom is false and the atom-independent fast
// path runs. locscale(StudentT(3), 0.5, 2.0) with a CONSTANT loc/scale —
// same oracle as the latent case at y=1.5, reached via the unchanged path.
// Fallback form: empty-input kernelof fails at materialisation (record()
// not evaluable in sampler context), so we give the kernel one unrelated
// input `d` that the bijection bodies do NOT reference. The essential
// property: needsPerAtom is false (bijection refs no refArrays name).
test('constant-bijection locscale likelihood still uses the fast path (oracle unchanged)', async () => {
  const ctx = buildCtx(
    `dummy ~ Normal(0, 1)\n`
    + `obs ~ locscale(StudentT(3.0), 0.5, 2.0)\n`
    + `fk = kernelof(record(obs = obs), d = dummy)\n`
    + `observed_data = 1.5\n`
    + `L = likelihoodof(fk, record(obs = observed_data))\n`
    + `ld = logdensityof(L, record(d = 0.0))\n`, 64);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - (-1.8541214455305277)) < 1e-12,
    `got ${m.samples[0]}, expected -1.8541214455305277`);
});
