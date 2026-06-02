'use strict';

// =====================================================================
// nested-broadcast-mvnormal-inner.test.ts — Phase 5.1 Session 5b
// =====================================================================
//
// Pins the DETECTOR + executor scaffolding for nested-broadcast with
// a VECTOR_OUTPUT inner head (MvNormal). Closes the INNER half of
// engine-concepts §22.2(d) at the recognition layer:
//
//   1. detectNestedBroadcastKernelBinding accepts MvNormal as inner
//      head alongside scalar SAMPLEABLE_DISTRIBUTIONS, recording
//      innerIsVectorOutput + innerEventDim.
//   2. The composite-body-recognizers nested-broadcast variant
//      forwards the new fields.
//   3. _executeNestedBroadcastComposite dispatches vector-output inner
//      to the Phase 8 fold `_executeNestedBroadcastVectorFold` (the
//      per-cell reference is retired): the two parallel axes fold into
//      ONE affine over count = N·K_outer·K_inner, output shape
//      [N, K_outer, K_inner * eventDim].
//
// Test 4 closes the formerly-deferred end-to-end exercise: the fold
// materialises + calibrates (per-(outer, inner) mean + shared cov).
// Atom-INDEP scope (per-cell mu resolves statically) — a latent outer
// mean threaded per-atom into the inner draw still defers to the
// matPushfwd vector-base extension.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator } = require('..');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

function readFixture(name: string): string {
  const p = path.join(__dirname, 'fixtures', name);
  return fs.readFileSync(p, 'utf-8');
}

function liftAndBuild(src: string) {
  const lifted = processSource(src);
  return orchestrator.buildDerivations(lifted.bindings);
}

// =====================================================================
// 1. Detector accepts MvNormal as inner head + records vector-output flag
// =====================================================================

test('nested-broadcast-mvnormal-inner detector: MvNormal flagged as vector-output', () => {
  const src = readFixture('nested-broadcast-mvnormal-inner.flatppl');
  const built = liftAndBuild(src);
  const kbShape = require('../kernel-broadcast-shape.ts');
  const desc = kbShape.detectNestedBroadcastKernelBinding(
    'inner_kernel', built.bindings);
  assert.ok(desc, 'inner_kernel matches nested-broadcast detector with '
    + 'VECTOR_OUTPUT_DISTRIBUTIONS gate extension');
  assert.equal(desc.innerDistOp, 'MvNormal');
  assert.equal(desc.innerIsVectorOutput, true,
    'MvNormal flagged as vector-output inner head');
  assert.ok(Number.isNaN(desc.innerEventDim) || desc.innerEventDim === 2,
    'inner eventDim: 2 (literal mu) or NaN (placeholder); got '
    + desc.innerEventDim);
});

// =====================================================================
// 2. Composite-body recognizer forwards the new fields
// =====================================================================

test('nested-broadcast-mvnormal-inner composite-body: variant carries innerIsVectorOutput + innerEventDim', () => {
  const src = readFixture('nested-broadcast-mvnormal-inner.flatppl');
  const built = liftAndBuild(src);
  const compositeBodies = require('../composite-body-recognizers.ts');
  const ctx = { bindings: built.bindings };
  const result = compositeBodies.tryRecognizeCompositeBody(
    { distOp: 'inner_kernel' }, ctx);
  assert.ok(result, 'composite-body recognizer matches');
  assert.equal(result.kind, 'nested_broadcast');
  assert.equal(result.innerDistOp, 'MvNormal');
  assert.equal(result.innerIsVectorOutput, true);
});

// =====================================================================
// 3. Scalar-inner nested-broadcast continues to work end-to-end
// =====================================================================
//
// Regression-only — confirms the detector relaxation + executor changes
// for vector-output didn't break the existing Phase 4.4 path. The
// `hierarchical-models.test.ts` "nested-broadcast" test exercises this
// in detail; we re-pin the detector shape here for completeness.

test('nested-broadcast scalar-inner: detector flags non-vector-output', () => {
  const src = readFixture('nested-broadcast.flatppl');   // Phase 4.4
                                                          // baseline
  const built = liftAndBuild(src);
  const kbShape = require('../kernel-broadcast-shape.ts');
  const desc = kbShape.detectNestedBroadcastKernelBinding(
    'patient_kernel', built.bindings);
  assert.ok(desc, 'baseline nested-broadcast still recognised');
  assert.equal(desc.innerDistOp, 'Normal');
  assert.equal(desc.innerIsVectorOutput, false,
    'scalar Normal stays non-vector-output (no regression)');
  assert.equal(desc.innerEventDim, 1);
});

// =====================================================================
// 4. End-to-end fold: vector-output inner materialises + calibrates
// =====================================================================
//
// The Phase 8 vector fold (`_executeNestedBroadcastVectorFold`) is now the
// only path — the per-cell reference is retired. It folds the two parallel
// axes (K_outer × K_inner) into ONE affine forward over count =
// N·K_outer·K_inner: one shared Cholesky of cov_shared, the per-(outer,
// inner)-cell means laid out, ONE `affineAtomBatchedForward`. The fixture's
// outer param is unused, so every outer cell repeats the same per-inner
// MvNormal(mu_inner_per_cell[k], cov_shared); a broken per-cell mu layout
// would smear the means, a broken affine would distort the covariance.

test('nested-broadcast-mvnormal-inner fold: per-(outer,inner) mean + shared cov at [N,3,8]', async () => {
  const N = 20000;
  const src = readFixture('nested-broadcast-mvnormal-inner.flatppl');
  const built = liftAndBuild(src);
  assert.ok(built.derivations && built.derivations.y, 'y has a derivation');
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 7 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (nm: string) => {
      if (cache.has(nm)) return cache.get(nm);
      const p = materialiser.materialiseMeasure(nm, ctx);
      cache.set(nm, p);
      return p;
    },
    sendWorker: (msg: any) => Promise.resolve(worker.handle(msg)),
    sampleCount: N, rootKey: [7, 0],
  };

  const Kout = 3, Kin = 4, n = 2;
  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [N, Kout, Kin * n],
    'shape [N, K_outer, K_inner * eventDim]');

  const muInner = [[0, 0], [1, -1], [-2, 3], [0.5, 0.5]];
  const Sigma = [[1, 0.3], [0.3, 0.5]];
  const d = m.value.data;
  const W = Kout * Kin * n;                       // per-atom stride = 24

  // (a) Per-(outer, inner) cell sample mean ≈ mu_inner[k] (outer-independent).
  for (let j = 0; j < Kout; j++) {
    for (let k = 0; k < Kin; k++) {
      const base = j * Kin * n + k * n;           // (j, k) offset in the atom
      let m0 = 0, m1 = 0;
      for (let i = 0; i < N; i++) { m0 += d[i * W + base]; m1 += d[i * W + base + 1]; }
      m0 /= N; m1 /= N;
      assert.ok(Math.abs(m0 - muInner[k][0]) < 0.08 && Math.abs(m1 - muInner[k][1]) < 0.08,
        `(out=${j}, in=${k}) mean ≈ [${muInner[k]}]; got [${m0.toFixed(2)}, ${m1.toFixed(2)}]`);
    }
  }

  // (b) Sample covariance at one cell ≈ cov_shared (the shared affine).
  const base = 1 * Kin * n + 2 * n;               // outer cell 1, inner cell 2
  let mm0 = 0, mm1 = 0;
  for (let i = 0; i < N; i++) { mm0 += d[i * W + base]; mm1 += d[i * W + base + 1]; }
  mm0 /= N; mm1 /= N;
  let v00 = 0, v11 = 0, v01 = 0;
  for (let i = 0; i < N; i++) {
    const a = d[i * W + base] - mm0, b = d[i * W + base + 1] - mm1;
    v00 += a * a; v11 += b * b; v01 += a * b;
  }
  v00 /= N; v11 /= N; v01 /= N;
  assert.ok(Math.abs(v00 - Sigma[0][0]) < 0.1, `var0 ≈ ${Sigma[0][0]}; got ${v00.toFixed(3)}`);
  assert.ok(Math.abs(v11 - Sigma[1][1]) < 0.08, `var1 ≈ ${Sigma[1][1]}; got ${v11.toFixed(3)}`);
  assert.ok(Math.abs(v01 - Sigma[0][1]) < 0.08, `cov01 ≈ ${Sigma[0][1]}; got ${v01.toFixed(3)}`);
});
