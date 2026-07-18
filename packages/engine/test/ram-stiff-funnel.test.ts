'use strict';
// packages/engine/test/ram-stiff-funnel.test.ts
// Regression for the RAM freeze-then-parallel collapse on a stiff/hierarchical
// posterior. Model: the BAT.jl signal-plus-background rare-event search
// (hierarchical per-dataset background rates B ~ iid(LogNormal(log(m_B) −
// σ_B²/2, σ_B), 5) — a funnel: σ_B's posterior sd is ~0.03, orders of magnitude
// tighter than the other coordinates).
//
// Bug: RAM's rank-1 proposal starts isotropic and adapted too slowly to discover
// that stiff scale within warmup, so the FROZEN block proposal over-stepped σ_B
// on every draw → acceptRate === 0 → all samples pinned at the warmed position →
// the viewer rendered `record(S = 0.4692, sigma_B = 0.0260, …)` (a constant) in
// place of a posterior. Fix: warmup re-anchors S to the empirical posterior
// covariance (× the optimal RW factor), which the rank-1 update refines — so the
// stiff direction gets its true scale. The hard assertion is that the
// frozen-proposal chains actually MOVE (acceptance healthy, samples not constant).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SRC = fs.readFileSync(path.join(__dirname, 'fixtures/bat-signal-background.flatppl'), 'utf8');

function setupCtx(N: number) {
  const lifted = processSource(SRC);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    moduleRegistry: lifted.loweredModule && lifted.loweredModule.moduleRegistry
      ? lifted.loweredModule.moduleRegistry : null,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n); const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p; },
    sendWorker: (m: any) => { const r = worker.handle(m); return r && r.type === 'error' ? Promise.reject(new Error(r.message)) : Promise.resolve(r); },
    sampleCount: N, rootSeed: 42, rootKey: 42,
  };
  return ctx;
}

function notConstant(s: Float64Array): boolean { for (let i = 1; i < s.length; i++) if (s[i] !== s[0]) return true; return false; }

test('ram freeze-then-parallel does not collapse on the BAT hierarchical posterior', async () => {
  // Phase 1: warmup on one worker → frozen proposal {L, scale} + warmed positions.
  const warm = await materialiser.materialiseMeasure('posterior', setupCtx(1), {
    backend: 'ram', chains: 4, walkers: 4, warmup: 1000, draws: 1, seed: 7, mcmcPhase: 'warmup',
  });
  const wd = warm.diagnostics.warmup;
  // Phase 2: sample full chains from the frozen proposal, resuming warmed positions.
  const m = await materialiser.materialiseMeasure('posterior', setupCtx(4000), {
    backend: 'ram', chains: 2, walkers: 2, warmup: 0, draws: 1000, seed: 99, mcmcPhase: 'sample',
    initAdapt: { L: wd.L, scale: wd.scale }, initPositions: [wd.endPositions[0], wd.endPositions[1]],
  });
  // The bug produced acceptRate === 0 with every field constant. Require movement.
  assert.ok(m.diagnostics.acceptRate > 0.05, `acceptRate ${m.diagnostics.acceptRate} — frozen proposal did not move (regression: constant posterior)`);
  for (const k of ['S', 'sigma_B', 'm_B', 'lam']) {
    assert.ok(notConstant(m.fields[k].samples), `field ${k} is constant — chains frozen`);
  }
  // The stiff coordinate σ_B must explore, not pin — its posterior sd is ~0.03.
  const b = m.fields.sigma_B.samples;
  let mean = 0; for (let i = 0; i < b.length; i++) mean += b[i]; mean /= b.length;
  let v = 0; for (let i = 0; i < b.length; i++) v += (b[i] - mean) * (b[i] - mean); v = Math.sqrt(v / b.length);
  assert.ok(v > 0.005, `sigma_B std ${v.toFixed(5)} — stiff coordinate collapsed`);
});
