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
