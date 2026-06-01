'use strict';

// =====================================================================
// joint-mvnormal-component.test.ts — Phase 5.1 Session 5a
// =====================================================================
//
// Pins joint composite-body kernel-broadcast with a VECTOR_OUTPUT
// component (MvNormal). The d-joint half of engine-concepts §22.2(d) —
// "composite recognisers handle multivariate components for free" —
// closes here:
//
//   1. detectJointKernelBinding accepts MvNormal as a component
//      alongside scalar SAMPLEABLE_DISTRIBUTIONS, recording
//      isVectorOutput + eventDim.
//   2. _executeJointComposite dispatches vector components to a
//      registry-backed per-cell materialiser (_sampleVectorOutputAtCell)
//      that consumes the affine bijection entry — same hot path
//      matMvNormal uses, run per joint cell × component.
//   3. Output Value shape becomes [N, K, sum_c(eventDim_c)] atom-major
//      with per-component event-dim slot widths.
//
// Together with Phase 5.1 Session 4 (`039031a`) bare-MvNormal kernel-
// broadcast, this closes the d HEAD case + the d COMPONENT case. The
// remaining d-piece (nested_broadcast inner MvNormal) defers until a
// fixture motivates it.

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
// 1. Detector accepts MvNormal as a joint component
// =====================================================================

test('joint-mvnormal-component: detector marks MvNormal component as vector-output', () => {
  const src = readFixture('joint-mvnormal-component.flatppl');
  const { bindings } = setupCtx(src, 10);
  const kbShape = require('../kernel-broadcast-shape.ts');
  const desc = kbShape.detectJointKernelBinding('joint_kernel', bindings);
  assert.ok(desc, 'joint_kernel matches the joint-composite detector');
  assert.equal(desc.components.length, 2, 'two components: loc + obs');
  // Components may be in either order depending on detector — find by op.
  const loc = desc.components.find((c: any) => c.distOp === 'MvNormal');
  const obs = desc.components.find((c: any) => c.distOp === 'Normal');
  assert.ok(loc, 'MvNormal component recognised');
  assert.equal(loc.isVectorOutput, true,
    'MvNormal flagged as vector-output');
  // MvNormal mu in this fixture is a kernel-placeholder (`mu = m`)
  // not a literal vector. _resolveVectorEventDim only traces literal
  // arrays + one-level refs to literals; it accepts NaN otherwise.
  // The materialiser resolves the actual eventDim from the
  // substituted-Value shape at runtime — see test 2 below for
  // end-to-end shape verification.
  assert.ok(Number.isNaN(loc.eventDim) || loc.eventDim === 2,
    'MvNormal eventDim is statically NaN (placeholder mu) or 2 '
    + '(when traceable to literal); got ' + loc.eventDim);
  assert.ok(obs, 'Normal component still recognised');
  assert.equal(obs.isVectorOutput, false,
    'scalar Normal stays non-vector-output');
  assert.equal(obs.eventDim, 1);
});

// =====================================================================
// 2. Materialiser dispatches per-cell vector-output components
// =====================================================================

test('joint-mvnormal-component: materialises to [N, K, totalEventDim] atom-major', async () => {
  const src = readFixture('joint-mvnormal-component.flatppl');
  const N = 50;
  const { ctx } = setupCtx(src, N);
  const m = await ctx.getMeasure('y');
  assert.ok(m && m.value && Array.isArray(m.value.shape),
    'y materialises to a shape-tagged Value');
  // K=3 groups; per-cell event dim = 2 (loc) + 1 (obs) = 3.
  assert.deepEqual(Array.from(m.value.shape), [N, 3, 3],
    'shape [N, K, totalEventDim] atom-major');
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      `output sample data[${i}] = ${m.value.data[i]} should be finite`);
  }
});

// =====================================================================
// 3. Per-group MvNormal location mean ≈ mu_per_group
// =====================================================================

test('joint-mvnormal-component: MvNormal loc mean ≈ mu_per_group', async () => {
  const src = readFixture('joint-mvnormal-component.flatppl');
  const N = 4000;
  const { ctx } = setupCtx(src, N);
  const m = await ctx.getMeasure('y');
  const data = m.value.data;     // [N, K=3, totalEventDim=3] atom-major
  const muExpected = [[0.0, 0.0], [1.0, -1.0], [-2.0, 3.0]];
  const diagCov = [1.0, 0.5];
  // Detector ordering follows the joint's surface field order, which the
  // fixture writes as `loc` (MvNormal, eventDim=2) then `obs` (Normal,
  // eventDim=1). So columns [0, 1] of the stitch are the MvNormal loc.
  for (let g = 0; g < 3; g++) {
    for (let k = 0; k < 2; k++) {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += data[i * 3 * 3 + g * 3 + k];
      const mean = sum / N;
      const stderr = Math.sqrt(diagCov[k] / N);
      const margin = 4 * stderr;
      assert.ok(Math.abs(mean - muExpected[g][k]) < margin,
        `group ${g} loc dim ${k}: sample mean ${mean.toFixed(4)} vs `
        + `expected ${muExpected[g][k]} (margin ${margin.toFixed(4)})`);
    }
  }
});

// =====================================================================
// 4. Scalar Normal obs mean ≈ 0 (well within margin)
// =====================================================================

test('joint-mvnormal-component: Normal obs mean ≈ 0 (4-sigma margin)', async () => {
  const src = readFixture('joint-mvnormal-component.flatppl');
  const N = 4000;
  const { ctx } = setupCtx(src, N);
  const m = await ctx.getMeasure('y');
  const data = m.value.data;
  const sigmaObs = 0.7;
  // Column [2] of the stitch is the scalar Normal obs.
  for (let g = 0; g < 3; g++) {
    let sum = 0;
    for (let i = 0; i < N; i++) sum += data[i * 3 * 3 + g * 3 + 2];
    const mean = sum / N;
    const stderr = sigmaObs / Math.sqrt(N);
    const margin = 4 * stderr;
    assert.ok(Math.abs(mean) < margin,
      `group ${g} obs: sample mean ${mean.toFixed(4)} should be near 0 `
      + `(margin ${margin.toFixed(4)})`);
  }
});

// =====================================================================
// Composition proof (Phase 5.1 Session 5f) — gate widening does NOT
// disturb the joint composite path.
// =====================================================================
//
// 5f-1 widened the MvNormal lift gate to fire for module-level refs with
// a statically-known inferredType. The MvNormal HERE is a joint
// component whose `mu = m` is a kernel-broadcast cell placeholder (a
// `%local`, not a module binding with inferredType), so the gate skips
// it — no `__bij` binding is synthesised, the MvNormal IR node survives,
// and the joint detector's VECTOR_OUTPUT_DISTRIBUTIONS path stays the
// materialiser. Teaching the composite detectors to recognise a
// lowered `pushfwd(<bij>, iid)` component (so they could consume the
// §22 decomposition uniformly) is 5g scope; 5f must leave this path
// untouched, which this pins.

test('joint-mvnormal-component: gate widening leaves composite path intact (no __bij)', () => {
  const src = readFixture('joint-mvnormal-component.flatppl');
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const synthBij = Array.from(built.bindings.keys())
    .filter((n: any) => /^__bij/.test(n));
  assert.equal(synthBij.length, 0,
    'kernel-placeholder mu does not trigger the 5f-1 gate — MvNormal '
    + 'component stays on the composite/matMvNormal path, not lowered '
    + 'to pushfwd');
});
