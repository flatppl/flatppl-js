'use strict';

// =====================================================================
// nested-broadcast batch-flatten — calibration oracle (Phase 8 leg 2)
// =====================================================================
//
// `test/fixtures/nested-broadcast.flatppl` (patients × visits):
//   visit_means        = [10, 20, 30, 40]      % per-visit baseline (shared)
//   sigmas_per_patient = [0.5, 1.0, 1.5]        % per-patient noise scale
//   patient_kernel = kernelof(broadcast(Normal, mu = visit_means,
//                                       sigma = sigma_g), sigma_g = sigma_g)
//   y = broadcast(patient_kernel, sigma_g = sigmas_per_patient)
//
// y is the canonical NESTED broadcast: an outer kernel-broadcast over
// patients whose body is itself a broadcast over visits. Phase 8 leg 2
// folds the two parallel axes (outer K_outer=3 × inner K_inner=4) into
// ONE count = N·3·4 batch (`_executeNestedBroadcastBatchFlatten`), reading
// the static `[kernel_broadcast 3, broadcast 4]` ladder the dissolver
// records on `y` (enabler: axisStack recursion into kernel bodies).
//
// This test pins (a) that the FOLD PATH is taken — y carries the literal
// two-entry ladder, the dispatcher's condition — and (b) the calibration:
// y[i, p, v] ~ Normal(visit_means[v], sigmas_per_patient[p]), so each
// patient row has σ ≈ sigmas[p] and each visit column has mean ≈
// visit_means[v]. The §03 shape is [N, 3, 4] (N atoms × patients × visits).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const axisStackMod = require('../axis-stack.ts');
const { createWorkerHandler } = require('../worker.ts');

test('nested-broadcast fold: y calibrates per (patient σ, visit μ) at shape [N,3,4]', async () => {
  const N = 4000;
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'nested-broadcast.flatppl'), 'utf-8');
  const lifted = processSource(src);
  // NB: the fixture (copied verbatim from flatppl-examples) trips the
  // cosmetic "one doc-comment per binding" parser diagnostic on its
  // multi-line `%` comments; the analyzer recovers and the model
  // materialises. We assert the model is DERIVABLE rather than
  // diagnostic-clean (not ours to reformat the shared fixture).
  const built = orchestrator.buildDerivations(lifted.bindings);
  assert.ok(built.derivations && built.derivations.y, 'y has a derivation');
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 11 });
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
    sampleCount: N, rootKey: [11, 0],
  };

  // (a) The fold path: y carries the static two-axis ladder, so the
  //     dispatcher routes to `_executeNestedBroadcastBatchFlatten` rather
  //     than the per-cell reference.
  const stack = axisStackMod.bindingAxisStack('y', ctx);
  assert.deepEqual(stack, [
    { source: 'kernel_broadcast', size: 3, name: 'sigmas_per_patient' },
    { source: 'broadcast', size: 4, name: 'visit_means' },
  ], 'y carries the literal [kernel_broadcast 3, broadcast 4] ladder (fold path)');

  const m = await ctx.getMeasure('y');

  // (b) §03 shape: N atoms × 3 patients × 4 visits.
  assert.deepEqual(m.value.shape, [N, 3, 4], 'shape [N, patients, visits]');
  assert.equal(m.value.data.length, N * 3 * 4, 'flat buffer N·3·4');

  // (c) Calibration: y[i,p,v] ~ Normal(visit_means[v], sigmas[p]).
  const visitMeans = [10, 20, 30, 40];
  const sigmas = [0.5, 1.0, 1.5];
  const d = m.value.data;
  for (let p = 0; p < 3; p++) {
    for (let v = 0; v < 4; v++) {
      let s = 0;
      for (let i = 0; i < N; i++) s += d[i * 12 + p * 4 + v];
      const mean = s / N;
      let q = 0;
      for (let i = 0; i < N; i++) { const x = d[i * 12 + p * 4 + v] - mean; q += x * x; }
      const sd = Math.sqrt(q / N);
      assert.ok(Math.abs(mean - visitMeans[v]) < 0.15,
        `(p=${p},v=${v}) mean ≈ ${visitMeans[v]}; got ${mean.toFixed(3)}`);
      assert.ok(Math.abs(sd - sigmas[p]) < 0.12,
        `(p=${p},v=${v}) sd ≈ ${sigmas[p]} (patient scale); got ${sd.toFixed(3)}`);
    }
  }
});
