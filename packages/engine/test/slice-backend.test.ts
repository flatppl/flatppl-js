'use strict';
// packages/engine/test/slice-backend.test.ts
// End-to-end backend:'slice' posterior path. Conjugate Normal-Normal:
// mu ~ Normal(0,10); y ~ Normal(mu,1) observed at 5. Oracle: closed-form
// posterior mean/var, mu | y ~ Normal(5·postVar, postVar), postVar=1/(1+1/100).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const errs = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) return { errs, ctx: null };
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule && lifted.loweredModule.moduleRegistry ? lifted.loweredModule.moduleRegistry : null,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => { const r = worker.handle(m); return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r); },
    sampleCount: N, rootSeed: 42, rootKey: 42,
  };
  return { errs: [], ctx };
}

const MODEL = `
mu = draw(Normal(mu = 0.0, sigma = 10.0))
prior = lawof(record(mu = mu))
obs_dist = joint(y = Normal(mu = mu, sigma = 1.0))
K = functionof(obs_dist, mu = mu)
L = likelihoodof(K, record(y = 5.0))
posterior = bayesupdate(L, prior)
`;

test('backend:slice recovers the conjugate posterior mean/var', async () => {
  const { errs, ctx } = setupCtx(MODEL, 2000);
  assert.equal(errs.length, 0, `parse errors: ${errs.map((e: any) => e.message).join('; ')}`);
  const m = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'slice', chains: 4, warmup: 500, draws: 1000, seed: 1,
  });
  const draws = m.fields.mu.samples;
  assert.ok(draws && draws.length > 0, 'slice posterior produced samples');
  let mean = 0; for (let i = 0; i < draws.length; i++) mean += draws[i]; mean /= draws.length;
  let v = 0; for (let i = 0; i < draws.length; i++) v += (draws[i] - mean) ** 2; v /= draws.length;
  const postVar = 1 / (1 + 1 / 100), postMean = 5 * postVar;
  assert.ok(Math.abs(mean - postMean) < 0.15, `slice mean ${mean.toFixed(4)} vs analytic ${postMean.toFixed(4)}`);
  assert.ok(Math.abs(Math.sqrt(v) - Math.sqrt(postVar)) < 0.1, `slice sd ${Math.sqrt(v).toFixed(4)} vs analytic ${Math.sqrt(postVar).toFixed(4)}`);
});

test('slice result carries diagnostics with finite rHat and method tag', async () => {
  const { ctx } = setupCtx(MODEL, 500);
  const m = await materialiser.materialiseMeasure('posterior', ctx, { backend: 'slice', chains: 4, warmup: 500, draws: 500, seed: 2 });
  assert.ok(m.diagnostics && m.diagnostics.perParam, 'diagnostics present');
  assert.ok(Number.isFinite(m.diagnostics.perParam.mu.rHat) && m.diagnostics.perParam.mu.rHat < 1.5, 'mu rHat finite/near 1');
  assert.ok(Number.isFinite(m.diagnostics.perParam.mu.essBulk) && m.diagnostics.perParam.mu.essBulk > 0, 'essBulk positive');
});
