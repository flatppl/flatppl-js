'use strict';

// =====================================================================
// vector-obs-mvnormal.test.ts — Phase 5.1 Session 4
// =====================================================================
//
// Pins kernel-broadcast over a bare vector-output distribution
// (MvNormal in Session 4). The path:
//
//   1. Classifier recognises `broadcast(MvNormal, mu = mu_per_group,
//      cov = cov_shared)` as a `kernelbroadcast` derivation (via the
//      new `VECTOR_OUTPUT_DISTRIBUTIONS` gate).
//   2. matKernelBroadcast routes through
//      `_executeBareVectorOutputBroadcast` (Session 4 addition) which
//      builds per-cell mu slices, applies the registry's `affine`
//      atom-batched forward to iid Normal base draws, and stitches
//      into a [N, K, n] atom-major Value.
//
// Closes engine-concepts §22.2(d) — "composite recognisers handle
// multivariate components for free" — end-to-end with a real
// kernel-broadcast fixture.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function readFixture(name: string): string {
  const p = path.join(__dirname, 'fixtures', name);
  return fs.readFileSync(p, 'utf-8');
}

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
    sampleCount: N,
    rootSeed: 42,
  };
  return { ctx, derivations: built.derivations, bindings: built.bindings };
}

// =====================================================================
// 1. Classifier recognises bare-MvNormal kernel-broadcast
// =====================================================================

test('vector-obs MvNormal: classifier routes broadcast(MvNormal, …) as kernelbroadcast', () => {
  const src = readFixture('vector-obs-mvnormal.flatppl');
  const { derivations } = setupCtx(src, 10);
  assert.ok(derivations.y, 'y has a derivation');
  // `y ~ broadcast(...)` lifts the measure to an anon binding and
  // `y` becomes an alias of it. Walk the alias chain to the
  // underlying derivation; that's where the kernelbroadcast lives.
  let kbName = 'y';
  while (derivations[kbName] && derivations[kbName].kind === 'alias') {
    kbName = derivations[kbName].from;
  }
  assert.equal(derivations[kbName].kind, 'kernelbroadcast',
    'bare MvNormal broadcast classifies as kernelbroadcast (not '
    + 'rejected by the SAMPLEABLE_DISTRIBUTIONS gate). Lifted binding: '
    + kbName);
  assert.equal(derivations[kbName].distOp, 'MvNormal');
});

// =====================================================================
// 2. Materialiser produces [N, K, n] atom-major
// =====================================================================

test('vector-obs MvNormal: materialises to [N, K, n] atom-major', async () => {
  const src = readFixture('vector-obs-mvnormal.flatppl');
  const N = 50;
  const { ctx } = setupCtx(src, N);
  const m = await ctx.getMeasure('y');
  assert.ok(m && m.value && Array.isArray(m.value.shape),
    'y materialises to a shape-tagged Value');
  // K=3 groups, n=2 dims per MvNormal cell — fixture pins these.
  assert.deepEqual(Array.from(m.value.shape), [N, 3, 2],
    'shape [N, K, n] atom-major');
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      `output sample data[${i}] = ${m.value.data[i]} should be finite`);
  }
});

// =====================================================================
// 3. Per-group sample mean ≈ mu_per_group within calibrated margin
// =====================================================================
//
// 4-sigma margin on the std-error of a sample mean: with shared cov
// (diag(cov) = [1.0, 0.5]), the per-cell std-error of the sample mean
// is sqrt(cov[k,k]/N).

test('vector-obs MvNormal: per-group sample mean ≈ mu_per_group', async () => {
  const src = readFixture('vector-obs-mvnormal.flatppl');
  const N = 4000;
  const { ctx } = setupCtx(src, N);
  const m = await ctx.getMeasure('y');
  const data = m.value.data;     // [N, K=3, n=2] atom-major
  const muExpected = [[0.0, 0.0], [1.0, -1.0], [-2.0, 3.0]];
  const diagCov = [1.0, 0.5];
  for (let j = 0; j < 3; j++) {
    for (let k = 0; k < 2; k++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += data[i * 3 * 2 + j * 2 + k];
      const mean = sum / N;
      const stderr = Math.sqrt(diagCov[k] / N);
      const margin = 4 * stderr;
      assert.ok(Math.abs(mean - muExpected[j][k]) < margin,
        `group ${j} dim ${k}: sample mean ${mean.toFixed(4)} vs `
        + `expected ${muExpected[j][k]} (margin ${margin.toFixed(4)})`);
    }
  }
});

// =====================================================================
// 4. Per-group sample covariance ≈ shared cov within calibrated margin
// =====================================================================
//
// 4-sigma margin on a sample variance: var(s²) ≈ 2σ⁴/(N-1); std-error
// = σ² · sqrt(2/(N-1)). Off-diagonal covariance variance is bounded by
// the same expression scaled by the product of diagonals.

test('vector-obs MvNormal: per-group sample cov ≈ cov_shared', async () => {
  const src = readFixture('vector-obs-mvnormal.flatppl');
  const N = 5000;
  const { ctx } = setupCtx(src, N);
  const m = await ctx.getMeasure('y');
  const data = m.value.data;
  const muExpected = [[0.0, 0.0], [1.0, -1.0], [-2.0, 3.0]];
  const covExpected = [[1.0, 0.3], [0.3, 0.5]];
  const adaptive = Math.sqrt(2 / (N - 1));
  for (let j = 0; j < 3; j++) {
    // Per-group sample (co)variance, centred at expected mu.
    const acc = [[0, 0], [0, 0]];
    for (let i = 0; i < N; i++) {
      const a = data[i * 3 * 2 + j * 2 + 0] - muExpected[j][0];
      const b = data[i * 3 * 2 + j * 2 + 1] - muExpected[j][1];
      acc[0][0] += a * a;
      acc[0][1] += a * b;
      acc[1][0] += b * a;
      acc[1][1] += b * b;
    }
    for (let p = 0; p < 2; p++) {
      for (let q = 0; q < 2; q++) {
        const sampleCov = acc[p][q] / N;
        const scale = Math.sqrt(Math.abs(covExpected[p][p] * covExpected[q][q]));
        const margin = 4 * scale * adaptive;
        assert.ok(Math.abs(sampleCov - covExpected[p][q]) < margin,
          `group ${j} cov[${p},${q}]: sample ${sampleCov.toFixed(4)} vs `
          + `expected ${covExpected[p][q]} (margin ${margin.toFixed(4)})`);
      }
    }
  }
});
