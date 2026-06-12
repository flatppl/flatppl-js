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

// Per-atom path with a SCALAR (literal-0.0) logVolume. The latent oracle
// tests above route through locscale → an affine bijection whose logVolume
// is a FUNCTION (−log scale); they never exercise the per-atom loop's
// scalar-logVolume slot (density.ts walkPushfwd, `bij.logVolume.kind ===
// 'scalar'` inside the needsPerAtom branch). Here we build the bijection
// EXPLICITLY: a pure translation `_ + shift` whose inverse `_ - shift`
// references the latent `shift`, with the volume-preserving literal logVolume
// 0.0. Because f_inv references the latent kernel input `shift`, walkPushfwd
// takes the per-atom branch; the scalar-0 logVolume then exercises line 1049.
//
// pushfwd(_ + 0.5)(Normal(0,1)) = Normal(0.5, 1); the likelihood scores the
// observation 1.5 at the conditioned shift=0.5.
// Oracle (INDEPENDENT — scipy.stats.norm.logpdf(1.5, loc=0.5, scale=1.0)):
//   -1.4189385332046727
//
// Surface-form note: the prescribed `shift = shift` single-input kernel makes
// `feedInputs` route `shift` through `measureToPerAtomRecords` (an array of
// {shift} records), which the per-atom slice path can't index as a scalar
// column. A SECOND, unrelated latent input (`pad`, which the bijection bodies
// do NOT reference) makes feedInputs materialise both as plain Float64Array
// columns — the same column shape the working locscale latents above arrive
// as — without altering the scored density (Normal(0,1) base + a +0.5 shift,
// pad untouched). This mirrors the constant-bijection test's `dummy` device.
test('per-atom scalar-logVolume bijection in a likelihood (translation, latent shift)', async () => {
  const ctx = buildCtx(
    `shift ~ Normal(0.0, 1.0)\n`
    + `pad ~ Normal(0.0, 1.0)\n`
    + `bij = bijection(fn(_ + shift), fn(_ - shift), 0.0)\n`
    + `obs ~ pushfwd(bij, Normal(0.0, 1.0))\n`
    + `fk = kernelof(record(obs = obs), shift = shift, pad = pad)\n`
    + `observed_data = 1.5\n`
    + `L = likelihoodof(fk, record(obs = observed_data))\n`
    + `ld = logdensityof(L, record(shift = 0.5, pad = 0.0))\n`, 64);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - (-1.4189385332046727)) < 1e-12,
    `got ${m.samples[0]}, expected -1.4189385332046727`);
});

// Per-atom path with a non-null OVERLAY. When the latent-bijection pushfwd is
// a LATER field of a joint-record observation whose base references an EARLIER
// field, that earlier field's value is threaded in via the env overlay. The
// per-atom loop must layer that overlay onto each atom's env (density.ts
// walkPushfwd, `if (overlayKeys)` inside the needsPerAtom branch, 1015-1019) —
// the prior tests have a null overlay (first/only field), so this is the only
// model that reaches it.
//
//   a ~ Normal(0,1)                       (field 1 — observed 1.5, overlaid)
//   b ~ pushfwd(_+shift, Normal(a, 1))    (field 2 — base refs prior field a)
// pushfwd(+0.5)(Normal(a,1)) = Normal(a+0.5, 1); scored at a=1.5, b=0.7.
// Oracle (INDEPENDENT — scipy.stats.norm):
//   norm.logpdf(1.5,0,1) + norm.logpdf(0.7, 2.0, 1) = -3.8078770664093455
test('per-atom bijection with a non-null overlay (later-field pushfwd refs prior field)', async () => {
  const ctx = buildCtx(
    `shift ~ Normal(0.0, 1.0)\n`
    + `pad ~ Normal(0.0, 1.0)\n`
    + `bij = bijection(fn(_ + shift), fn(_ - shift), 0.0)\n`
    + `a ~ Normal(0.0, 1.0)\n`
    + `b ~ pushfwd(bij, Normal(a, 1.0))\n`
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
