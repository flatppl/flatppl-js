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

// Mirror the setupCtx harness from test-models.test.ts, extended to accept
// inferenceOpts on getMeasure and stash them on ctx so matBayesupdate can read them.
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
    getMeasure: (n: string, inferenceOpts?: any) => {
      // Thread inferenceOpts onto ctx so matBayesupdate can read ctx.inferenceOpts.
      // Clear it after the call so IS-default tests never see a stale option.
      if (inferenceOpts) ctx.inferenceOpts = inferenceOpts;
      // Do NOT cache MH requests — each call with opts is a fresh posterior run.
      if (inferenceOpts) {
        return materialiser.materialiseMeasure(n, ctx)
          .finally(() => { delete ctx.inferenceOpts; });
      }
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

test('backend:mh recovers the conjugate posterior mean', async () => {
  const { errs, ctx } = setupCtx(MODEL, 100);
  assert.equal(errs.length, 0, `parse errors: ${errs.map((e: any) => e.message).join('; ')}`);
  const m = await ctx.getMeasure('posterior', {
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
  const m = await ctx.getMeasure('posterior');   // no backend opt → IS path
  assert.ok('logWeights' in m, 'IS posterior carries logWeights');
  assert.ok(m.logWeights !== null, 'IS logWeights is not null (reweighted)');
});
