'use strict';
// packages/engine/test/ram-quality.test.ts
// Regression for the RAM (Robust Adaptive Metropolis) sampler-quality bug on
// eight-schools: the frozen, per-warmup-adapted proposal was systematically
// oversized (acceptRate well below the RWM target of 0.234) because
// `ram-kernel.ts` had no fast, direction-independent scale correction —
// warmup relied solely on the slow rank-1 (Vihola 2012) shape update plus a
// periodic re-anchor to kappa·√(empirical covariance), and that re-anchor
// could also land on the LAST warmup sweep with zero refinement time left.
// Both left split-R̂ elevated (poor mixing) and, on other models, the chain
// pinned near its warmup end-position. Fixed in ram-kernel.ts: `scale` is now
// a genuinely Robbins-Monro-adapted multiplier (mirroring mh-kernel.ts), and
// the periodic re-anchor reserves a settle window before the freeze.
//
// Oracle here is the mh backend on the SAME model/seed: mh was never affected
// by this bug (independent adaptation design) and its posterior means are the
// cross-check that RAM converges to the same target, not just that its own
// diagnostics look healthy.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { ctxFor } = require('./_ctx-factory.ts');
const { materialiser } = require('..');

const FIXTURE = path.join(__dirname, 'fixtures/baseline/eight-schools.flatppl');
const SRC = fs.readFileSync(FIXTURE, 'utf8');

// chains/warmup/draws large enough for eight-schools' half-Cauchy funnel (tau)
// to settle within split-R̂ < 1.1 deterministically at this seed; ~1s runtime.
const OPTS = { chains: 4, warmup: 3000, draws: 2000, seed: 7 };

test('backend:ram on eight-schools: acceptRate near target and split-R̂ < 1.1 for every latent', async () => {
  const { ctx } = ctxFor(SRC, 4000);
  const m = await materialiser.materialiseMeasure('posterior', ctx, Object.assign({ backend: 'ram' }, OPTS));

  // Direct pin on the broken mechanism: a proposal frozen from an under-
  // corrected warmup shows up FIRST as acceptRate well below the RW-MH target
  // (0.234 for dim > 1) — the regression measured acceptRate as low as 0.01-0.18
  // here, vs mh's healthy ~0.2-0.3.
  const { acceptRate, perParam } = m.diagnostics;
  assert.ok(acceptRate > 0.15 && acceptRate < 0.35, `acceptRate ${acceptRate} should be near the 0.234 RW-MH target`);

  for (const name of Object.keys(perParam)) {
    const { rHat } = perParam[name];
    assert.ok(Number.isFinite(rHat) && rHat < 1.1, `${name} split-R̂ ${rHat} should be < 1.1`);
  }
});

test('backend:ram on eight-schools: mu/tau posterior means match the mh reference', async () => {
  const mean = (arr: Float64Array) => { let s = 0; for (let i = 0; i < arr.length; i++) s += arr[i]; return s / arr.length; };

  const { ctx: ramCtx } = ctxFor(SRC, 4000);
  const ram = await materialiser.materialiseMeasure('posterior', ramCtx, Object.assign({ backend: 'ram' }, OPTS));
  const { ctx: mhCtx } = ctxFor(SRC, 4000);
  const mh = await materialiser.materialiseMeasure('posterior', mhCtx, Object.assign({ backend: 'mh' }, OPTS));

  const ramMu = mean(ram.fields.mu.samples), mhMu = mean(mh.fields.mu.samples);
  const ramTau = mean(ram.fields.tau.samples), mhTau = mean(mh.fields.tau.samples);

  // Generous tolerance: tau's MC error is large (half-Cauchy funnel, modest
  // ESS), so this checks RAM lands on the SAME posterior mh does, not that
  // either estimate is precise.
  assert.ok(Math.abs(ramMu - mhMu) < 2.5, `ram mu ${ramMu.toFixed(3)} vs mh ${mhMu.toFixed(3)}`);
  assert.ok(Math.abs(ramTau - mhTau) < 3, `ram tau ${ramTau.toFixed(3)} vs mh ${mhTau.toFixed(3)}`);
  // Both backends should also land in the eight-schools literature ballpark.
  assert.ok(ramMu > 1 && ramMu < 9, `ram mu ${ramMu.toFixed(3)} outside the eight-schools ballpark`);
  assert.ok(ramTau > 0.5 && ramTau < 9, `ram tau ${ramTau.toFixed(3)} outside the eight-schools ballpark`);
});
