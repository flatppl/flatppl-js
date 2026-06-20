'use strict';

// packages/engine/test/mh-backend.test.ts
// End-to-end test: backend:'mh' posterior path through the materialiser.
// Conjugate Normal-Normal: mu ~ Normal(0, 10); y ~ Normal(mu, 1) observed at 5.
// Analytic posterior: mu | y ~ Normal(postMean, postVar) where
//   postVar  = 1 / (1/sigma2_prior + 1/sigma2_lik) = 1 / (1/100 + 1/1)
//   postMean = postVar * (y / sigma2_lik)           = postVar * 5

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

// Build a materialiser ctx for `src` with N IS samples.
function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) return { errs, ctx: null };
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule && lifted.loweredModule.moduleRegistry
      ? lifted.loweredModule.moduleRegistry : null,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: N,
    rootSeed: 42,
    rootKey: 42,
  };
  return { errs: [], ctx };
}

// Conjugate Normal-Normal: mu ~ N(0,10); y | mu ~ N(mu,1), observed y=5.
// Analytic posterior: mu | y ~ N(postMean, postVar)
// Uses functionof/draw pattern from closed-form-measure-algebra.test.ts.
const MODEL = `
mu = draw(Normal(mu = 0.0, sigma = 10.0))
prior = lawof(record(mu = mu))
obs_dist = joint(y = Normal(mu = mu, sigma = 1.0))
K = functionof(obs_dist, mu = mu)
L = likelihoodof(K, record(y = 5.0))
posterior = bayesupdate(L, prior)
`;

// Public API: materialiser.materialiseMeasure(name, ctx, inferenceOpts).
// This is the real production path — no ctx manipulation needed.
test('backend:mh recovers the conjugate posterior mean via public API', async () => {
  const { errs, ctx } = setupCtx(MODEL, 100);
  assert.equal(errs.length, 0, `parse errors: ${errs.map((e: any) => e.message).join('; ')}`);
  const m = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'mh', chains: 4, warmup: 1000, draws: 1000, seed: 1,
  });
  // m is a scalar measure from mhSample; draws live in m.samples
  const draws = m.samples;
  assert.ok(draws && draws.length > 0, 'mh posterior produced samples');
  let mean = 0;
  for (let i = 0; i < draws.length; i++) mean += draws[i];
  mean /= draws.length;
  const postVar = 1 / (1 + 1 / 100);
  const postMean = 5 * postVar;
  assert.ok(
    Math.abs(mean - postMean) < 0.15,
    `mh posterior mean ${mean.toFixed(4)} vs analytic ${postMean.toFixed(4)} (tol 0.15)`,
  );
});

test('backend defaults to is (IS posterior carries logWeights)', async () => {
  const { ctx } = setupCtx(MODEL, 200);
  // No inferenceOpts → IS path, unchanged behaviour.
  const m = await materialiser.materialiseMeasure('posterior', ctx);
  assert.ok('logWeights' in m, 'IS posterior carries logWeights');
  assert.ok(m.logWeights !== null, 'IS logWeights is not null (reweighted)');
});

test('mh result carries diagnostics with acceptRate and rHat', async () => {
  const { errs, ctx } = setupCtx(MODEL, 100);
  assert.equal(errs.length, 0, `parse errors: ${errs.map((e: any) => e.message).join('; ')}`);
  const m = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'mh', chains: 4, warmup: 500, draws: 500, seed: 2,
  });
  assert.ok(m.diagnostics, 'measure carries diagnostics field');
  const { acceptRate, perParam } = m.diagnostics;
  assert.ok(typeof acceptRate === 'number' && acceptRate > 0 && acceptRate < 1,
    `acceptRate ${acceptRate} should be in (0, 1)`);
  assert.ok(perParam && typeof perParam === 'object', 'diagnostics.perParam is present');
  // For the conjugate model, mu is the only latent.
  assert.ok('mu' in perParam, 'perParam has entry for latent mu');
  const { rHat, essBulk } = perParam.mu;
  assert.ok(Number.isFinite(rHat), `rHat ${rHat} should be finite`);
  assert.ok(rHat < 1.5, `rHat ${rHat} should be near 1 for well-mixed conjugate model`);
  assert.ok(Number.isFinite(essBulk) && essBulk > 0, `essBulk ${essBulk} should be positive finite`);
});

test('backend emcee recovers the conjugate posterior mean and returns diagnostics', async () => {
  const { ctx, errs } = setupCtx(MODEL, 4000);   // MODEL: the conjugate mu~Normal(0,10); posterior=bayesupdate(...)
  assert.equal(errs.length, 0);
  const m = await materialiser.materialiseMeasure('posterior', ctx, { backend: 'emcee', walkers: 10, warmup: 1000, draws: 1000, seed: 5 });
  const draws = m.value ? m.value.data : m.samples;
  let mean = 0; for (let i = 0; i < draws.length; i++) mean += draws[i]; mean /= draws.length;
  const postVar = 1/(1+1/100), postMean = 5*postVar;
  assert.ok(Math.abs(mean - postMean) < 0.15, `emcee mean ${mean} vs ${postMean}`);
  assert.ok(m.diagnostics && typeof m.diagnostics.acceptRate === 'number', 'emcee result carries diagnostics');
});

test('mismatched kwarg name throws rather than silently using 0', () => {
  // Exercise the REAL guard in model-spec.ts buildPosteriorSpec's logLikelihood.
  // Parse the canonical conjugate model, extract the real bayesupdate derivation,
  // call buildPosteriorSpec(d, ctx) to get the production logLikelihood, then
  // assert it throws when theta is missing the required kwarg 'mu'.
  const modelSpec = require('../model-spec.ts');
  const { errs, ctx } = setupCtx(MODEL, 100);
  assert.equal(errs.length, 0, `parse errors: ${errs.map((e: any) => e.message).join('; ')}`);
  // The real bayesupdate derivation produced by orchestrator.buildDerivations.
  const d = ctx.derivations['posterior'];
  assert.ok(d && d.kind === 'bayesupdate', 'posterior derivation must be bayesupdate');
  const spec = modelSpec.buildPosteriorSpec(d, ctx);
  assert.ok(typeof spec.logLikelihood === 'function', 'spec must carry logLikelihood');
  // Calling with { x: 1 } (missing required kwarg 'mu') must throw the guard error.
  assert.throws(
    () => spec.logLikelihood({ x: 1 }),
    (err: any) => err.message.includes("kwarg 'mu'"),
    'mismatched kwarg name should throw a clear error from the real guard',
  );
  // Calling with the correct key must not throw.
  assert.doesNotThrow(
    () => spec.logLikelihood({ mu: 1 }),
    'matched kwarg name must not throw',
  );
});
