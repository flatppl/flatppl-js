'use strict';

// =====================================================================
// bare MvNormal broadcast batch-flatten — calibration oracle (Phase 8)
// =====================================================================
//
//   mus   = [[1, 5], [10, 20]]      % per-cell mean vectors  (K=2, n=2)
//   Sigma = [[1, 0.5], [0.5, 2]]    % shared covariance
//   y = broadcast(MvNormal, mu = mus, cov = Sigma)           → [N, 2, 2]
//
// `broadcast(MvNormal, mu_per_cell, cov)` is a batched multivariate
// observation (K groups, each an MvNormal with a group-specific mean,
// shared cov). Phase 8 folds the cell axis K into the affine forward:
// one shared Cholesky, the per-cell means laid out to [count = N·K, n],
// and ONE `affineAtomBatchedForward` over count matvecs L against every
// (i,j) position — replacing the former per-cell loop (K affine forwards
// + z-slicing). The intrinsic n axis is NOT folded (§03).
//
// Pins per-cell calibration: y[i,j,:] ~ MvNormal(mus[j], Sigma) — so each
// cell's sample mean matches mus[j] and each cell's sample covariance
// matches the shared Sigma. A broken per-cell-mu layout would smear the
// means; a broken affine would distort the covariance.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

const SRC = `flatppl_compat = "0.1"
mus = [[1.0, 5.0], [10.0, 20.0]]
Sigma = [[1.0, 0.5], [0.5, 2.0]]
y = broadcast(MvNormal, mu = mus, cov = Sigma)`;

test('MvNormal broadcast fold: per-cell means + shared cov at shape [N,2,2]', async () => {
  const N = 20000;
  const lifted = processSource(SRC);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'parses cleanly: ' + errs.map((d: any) => d.message).join('; '));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 3 });
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
    sampleCount: N, rootKey: [3, 0],
  };

  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [N, 2, 2], 'shape [N, cells, event-dim]');

  const mus = [[1, 5], [10, 20]];
  const Sigma = [[1, 0.5], [0.5, 2]];
  const d = m.value.data;
  const K = 2, n = 2;
  for (let j = 0; j < K; j++) {
    let m0 = 0, m1 = 0;
    for (let i = 0; i < N; i++) { m0 += d[(i * K + j) * n + 0]; m1 += d[(i * K + j) * n + 1]; }
    m0 /= N; m1 /= N;
    assert.ok(Math.abs(m0 - mus[j][0]) < 0.1 && Math.abs(m1 - mus[j][1]) < 0.1,
      `cell ${j} mean ≈ [${mus[j]}]; got [${m0.toFixed(2)}, ${m1.toFixed(2)}]`);
    let v00 = 0, v11 = 0, v01 = 0;
    for (let i = 0; i < N; i++) {
      const a = d[(i * K + j) * n + 0] - m0, b = d[(i * K + j) * n + 1] - m1;
      v00 += a * a; v11 += b * b; v01 += a * b;
    }
    v00 /= N; v11 /= N; v01 /= N;
    assert.ok(Math.abs(v00 - Sigma[0][0]) < 0.1, `cell ${j} var0 ≈ ${Sigma[0][0]}; got ${v00.toFixed(3)}`);
    assert.ok(Math.abs(v11 - Sigma[1][1]) < 0.15, `cell ${j} var1 ≈ ${Sigma[1][1]}; got ${v11.toFixed(3)}`);
    assert.ok(Math.abs(v01 - Sigma[0][1]) < 0.1, `cell ${j} cov01 ≈ ${Sigma[0][1]}; got ${v01.toFixed(3)}`);
  }
});
