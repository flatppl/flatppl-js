'use strict';
// packages/engine/test/ram-freeze-then-parallel.test.ts
// RAM reuses the freeze-then-parallel pool path: phase 1 runs warmup on one
// worker and surfaces the adapted proposal (S, exposed as {L, scale}) + warmed
// positions; phase 2 samples full-length chains from that frozen proposal with
// no re-adaptation. Oracle: conjugate Normal-Normal posterior mean, mu | y=5.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const MODEL = `
mu = draw(Normal(mu = 0.0, sigma = 10.0))
prior = lawof(record(mu = mu))
obs_dist = joint(y = Normal(mu = mu, sigma = 1.0))
K = functionof(obs_dist, mu = mu)
L = likelihoodof(K, record(y = 5.0))
posterior = bayesupdate(L, prior)
`;

function setupCtx(N: number) {
  const lifted = processSource(MODEL);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), moduleRegistry: null,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => { const r = worker.handle(m); return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r); },
    sampleCount: N, rootSeed: 42, rootKey: 42,
  };
  return ctx;
}

test('ram freeze-then-parallel: warmup surfaces the frozen proposal + warmed positions', async () => {
  const m = await materialiser.materialiseMeasure('posterior', setupCtx(64), {
    backend: 'ram', chains: 4, warmup: 400, draws: 1, seed: 1, mcmcPhase: 'warmup',
  });
  const w = m.diagnostics && m.diagnostics.warmup;
  assert.ok(w, 'warmup phase attaches diagnostics.warmup');
  assert.ok(w.L instanceof Float64Array && w.L.length === 1, 'scalar mu → 1×1 factor S');
  assert.ok(Number.isFinite(w.L[0]) && w.L[0] > 0, 'proposal factor is finite/positive');
  assert.ok(Number.isFinite(w.scale) && w.scale > 0, 'scale finite/positive');
  assert.equal(w.endPositions.length, 4, 'one warmed end-position per chain');
});

test('ram freeze-then-parallel: sampling from the frozen proposal recovers the mean', async () => {
  const postVar = 1 / (1 + 1 / 100), postMean = 5 * postVar;
  const warm = await materialiser.materialiseMeasure('posterior', setupCtx(64), {
    backend: 'ram', chains: 4, warmup: 500, draws: 1, seed: 1, mcmcPhase: 'warmup',
  });
  const wd = warm.diagnostics.warmup;
  const m = await materialiser.materialiseMeasure('posterior', setupCtx(4000), {
    backend: 'ram', chains: 2, warmup: 0, draws: 1000, seed: 2, mcmcPhase: 'sample',
    initAdapt: { L: wd.L, scale: wd.scale }, initPositions: [wd.endPositions[0], wd.endPositions[1]],
  });
  const draws = m.fields.mu.samples;
  let mean = 0; for (let i = 0; i < draws.length; i++) mean += draws[i]; mean /= draws.length;
  assert.ok(Math.abs(mean - postMean) < 0.15, `ram freeze mean ${mean.toFixed(4)} vs analytic ${postMean.toFixed(4)}`);
});
