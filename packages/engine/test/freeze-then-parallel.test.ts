'use strict';

// packages/engine/test/freeze-then-parallel.test.ts
// Freeze-then-parallel mh path (used by the viewer worker pool): phase 1 runs the
// full warmup on ONE worker and surfaces the adapted proposal ({L, scale}) + the
// warmed chain positions; phase 2 samples full-length chains from that frozen
// proposal with no re-adaptation — the chains distribute across workers. This
// exercises the engine-side wiring (mat-density phase threading + driver
// initAdapt/initPositions + adaptState/endPositions return) through the
// materialiser. Oracle: conjugate Normal-Normal posterior mean, mu | y=5.

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

test('freeze-then-parallel mh: warmup phase surfaces frozen proposal + warmed positions', async () => {
  const m = await materialiser.materialiseMeasure('posterior', setupCtx(64), {
    backend: 'mh', chains: 4, warmup: 400, draws: 1, seed: 1, mcmcPhase: 'warmup',
  });
  const w = m.diagnostics && m.diagnostics.warmup;
  assert.ok(w, 'warmup phase attaches diagnostics.warmup');
  assert.ok(w.L instanceof Float64Array && w.L.length === 1, 'scalar mu → 1×1 Cholesky factor');
  assert.ok(Number.isFinite(w.L[0]) && w.L[0] > 0, 'proposal factor is finite/positive');
  assert.ok(Number.isFinite(w.scale) && w.scale > 0, 'adapted step scale is finite/positive');
  assert.equal(w.endPositions.length, 4, 'one warmed end-position per chain');
});

test('freeze-then-parallel mh: sampling from the frozen proposal recovers the posterior mean', async () => {
  const postVar = 1 / (1 + 1 / 100), postMean = 5 * postVar;
  const warm = await materialiser.materialiseMeasure('posterior', setupCtx(64), {
    backend: 'mh', chains: 4, warmup: 500, draws: 1, seed: 1, mcmcPhase: 'warmup',
  });
  const wd = warm.diagnostics.warmup;
  // Sample two full-length chains from the frozen proposal, resuming warmed positions.
  const m = await materialiser.materialiseMeasure('posterior', setupCtx(4000), {
    backend: 'mh', chains: 2, warmup: 0, draws: 1000, seed: 2, mcmcPhase: 'sample',
    initAdapt: { L: wd.L, scale: wd.scale }, initPositions: [wd.endPositions[0], wd.endPositions[1]],
  });
  const draws = m.fields.mu.samples;
  let mean = 0; for (let i = 0; i < draws.length; i++) mean += draws[i]; mean /= draws.length;
  assert.ok(Number.isFinite(mean), 'sample-phase mean is finite');
  assert.ok(Math.abs(mean - postMean) < 0.15, `freeze-then-parallel mean ${mean.toFixed(4)} vs analytic ${postMean.toFixed(4)} (tol 0.15)`);
});
