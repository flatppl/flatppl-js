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
//   3. _executeNestedBroadcastComposite is wired to dispatch
//      vector-output inner via _sampleVectorOutputAtCell + per-cell
//      event-dim aware stitching ([N, K_outer, K_inner * eventDim]).
//
// **End-to-end exercise deferred to Session 5c+:** a meaningful
// nested-broadcast pattern requires the outer kernel param to thread
// into the inner MvNormal kwarg (mu / cov), which means atom-dep
// MvNormal params — not yet supported in Session 5b MVP. The
// detector + executor extensions land in this commit as the
// recognition layer that Session 5c's atom-dep extension will
// consume.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator } = require('..');

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
