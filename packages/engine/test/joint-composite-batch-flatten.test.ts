'use strict';

// =====================================================================
// joint-composite batch-flatten — calibration oracle (Phase 8 leg 3)
// =====================================================================
//
//   a_means = [1, 2, 3];  b_rates = [0.5, 1, 2]
//   obs_kernel = kernelof(joint(a = Normal(mu = am, sigma = 1),
//                               b = Exponential(rate = br)), am = am, br = br)
//   y = broadcast(obs_kernel, am = a_means, br = b_rates)
//
// A joint-bodied kernel-broadcast: K=3 cells, each drawing ONE joint
// product variate of C=2 independent scalar components. The components
// are the variate STRUCTURE (eventShape), not a parallel axis — so the
// fold (`_executeJointCompositeBatchFlatten`) does NOT fold them into the
// batch; it batch-flattens each component over the cell axis K (count =
// N·K, one sampleN per component) and concatenates → [N, K, C]. That is
// the "product" leg: K·C per-cell worker calls collapse to C.
//
// Pins (a) the fold path — y carries the single-entry `[kernel_broadcast
// 3]` ladder (a joint body adds no inner axis), the dispatcher's all-
// scalar condition — and (b) calibration: y[i,j,0] ~ Normal(a_means[j],1)
// and y[i,j,1] ~ Exponential(b_rates[j]).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const axisStackMod = require('../axis-stack.ts');
const { createWorkerHandler } = require('../worker.ts');

const SRC = `flatppl_compat = "0.1"
a_means = [1.0, 2.0, 3.0]
b_rates = [0.5, 1.0, 2.0]
obs_kernel = kernelof(joint(a = Normal(mu = am, sigma = 1.0), b = Exponential(rate = br)), am = am, br = br)
y = broadcast(obs_kernel, am = a_means, br = b_rates)`;

test('joint fold: y calibrates per cell to (Normal a, Exponential b) at shape [N,3,2]', async () => {
  const N = 6000;
  const lifted = processSource(SRC);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'parses cleanly: ' + errs.map((d: any) => d.message).join('; '));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 5 });
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
    sampleCount: N, rootKey: [5, 0],
  };

  // (a) Fold path: a joint body adds no inner axis, so y carries just the
  //     outer kernel_broadcast entry (the all-scalar dispatcher folds).
  const stack = axisStackMod.bindingAxisStack('y', ctx);
  assert.deepEqual(stack, [{ source: 'kernel_broadcast', size: 3, name: 'a_means' }],
    'y carries the single [kernel_broadcast 3] ladder (joint fold path)');

  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [N, 3, 2], 'shape [N, cells, components]');
  assert.equal(m.value.data.length, N * 3 * 2, 'flat buffer N·3·2');

  // (b) Calibration: component a ~ Normal(a_means[j], 1); b ~ Exp(b_rates[j]).
  const aMeans = [1, 2, 3];
  const bRates = [0.5, 1, 2];
  const d = m.value.data;
  for (let j = 0; j < 3; j++) {
    let sa = 0, sb = 0;
    for (let i = 0; i < N; i++) { sa += d[i * 6 + j * 2 + 0]; sb += d[i * 6 + j * 2 + 1]; }
    const meanA = sa / N, meanB = sb / N;
    assert.ok(Math.abs(meanA - aMeans[j]) < 0.1,
      `cell ${j} component a mean ≈ ${aMeans[j]}; got ${meanA.toFixed(3)}`);
    assert.ok(Math.abs(meanB - 1 / bRates[j]) < 0.15,
      `cell ${j} component b mean ≈ ${(1 / bRates[j]).toFixed(3)} (Exp mean); got ${meanB.toFixed(3)}`);
  }
});
