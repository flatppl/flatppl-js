'use strict';

// =====================================================================
// jointchain-composite batch-flatten — calibration oracle (Phase 8 leg 4)
// =====================================================================
//
// `test/fixtures/hierarchical-state-space.flatppl` (G=3 groups, AR-1):
//   x_0 ~ Normal(x0_per_group[g], sigma_init=0.1)
//   x_k ~ Normal(x_{k-1}, sigma_step=0.5)   for k = 1..3   (the carry)
//   y = broadcast(group_chain, x0 = x0_per_group)          → [N, 3, 4]
//
// A jointchain-bodied kernel-broadcast is a Markov chain per cell. Phase 8
// leg 4 folds it as a SCAN (`_executeJointChainScan`): the cell axis K=3
// folds into each step's sampleN (count = N·3, one call per step), and the
// steps run sequentially with the carry — step k-1's flat [count] column
// (already in (i,j) order) binds directly as step k's input refArray.
// K·C per-cell calls collapse to C.
//
// The decisive carry-correctness check is the INCREMENT distribution:
// x_k - x_{k-1} ~ Normal(0, sigma_step) only if step k actually consumes
// step k-1's variate. A broken carry would not calibrate to σ=0.5.
// Also pins the fold path (single [kernel_broadcast 3] ladder), the shape,
// the per-group x_0 mean, and random-walk variance growth
// Var(x_k) ≈ sigma_init² + k·sigma_step².

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const axisStackMod = require('../axis-stack.ts');
const { createWorkerHandler } = require('../worker.ts');

test('jointchain scan fold: AR-1 carry calibrates (increment σ=σ_step) at shape [N,3,4]', async () => {
  const N = 8000;
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'hierarchical-state-space.flatppl'), 'utf-8');
  // The fixture (verbatim from flatppl-examples) trips the cosmetic
  // multi-line doc-comment diagnostic; the analyzer recovers. Assert the
  // model is derivable rather than diagnostic-clean.
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  assert.ok(built.derivations && built.derivations.y, 'y has a derivation');
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 9 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
    sampleCount: N, rootKey: [9, 0],
  };

  // (a) Fold path: a jointchain body adds no inner axis → single ladder.
  const stack = axisStackMod.bindingAxisStack('y', ctx);
  assert.deepEqual(stack, [{ source: 'kernel_broadcast', size: 3, name: 'x0_per_group' }],
    'y carries the single [kernel_broadcast 3] ladder (scan fold path)');

  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [N, 3, 4], 'shape [N, groups, chain length]');

  const G = 3, C = 4;
  const d = m.value.data;
  const x0 = [0.0, 0.5, 1.0];
  const sigmaInit = 0.1, sigmaStep = 0.5;

  // (b) Carry correctness: increments x_k - x_{k-1} ~ Normal(0, σ_step).
  let si = 0, sq = 0, ni = 0;
  for (let g = 0; g < G; g++) {
    for (let k = 1; k < C; k++) {
      for (let i = 0; i < N; i++) {
        const inc = d[i * G * C + g * C + k] - d[i * G * C + g * C + (k - 1)];
        si += inc; sq += inc * inc; ni++;
      }
    }
  }
  const incMean = si / ni, incStd = Math.sqrt(sq / ni - incMean * incMean);
  assert.ok(Math.abs(incMean) < 0.05, `increment mean ≈ 0; got ${incMean.toFixed(4)}`);
  assert.ok(Math.abs(incStd - sigmaStep) < 0.05,
    `increment std ≈ σ_step=${sigmaStep} (proves the carry threads x_{k-1} into x_k); got ${incStd.toFixed(4)}`);

  // (c) Per-group x_0 mean ≈ x0_per_group[g]; random-walk variance growth.
  for (let g = 0; g < G; g++) {
    let s0 = 0;
    for (let i = 0; i < N; i++) s0 += d[i * G * C + g * C + 0];
    assert.ok(Math.abs(s0 / N - x0[g]) < 0.05, `group ${g} x_0 mean ≈ ${x0[g]}; got ${(s0 / N).toFixed(3)}`);
    let m3 = 0;
    for (let i = 0; i < N; i++) m3 += d[i * G * C + g * C + 3];
    m3 /= N;
    let v3 = 0;
    for (let i = 0; i < N; i++) { const x = d[i * G * C + g * C + 3] - m3; v3 += x * x; }
    v3 /= N;
    const expVar = sigmaInit * sigmaInit + 3 * sigmaStep * sigmaStep;   // ≈ 0.76
    assert.ok(Math.abs(v3 - expVar) < 0.12, `group ${g} Var(x_3) ≈ ${expVar.toFixed(2)}; got ${v3.toFixed(3)}`);
  }
});
