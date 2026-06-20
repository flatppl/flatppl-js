'use strict';

// =====================================================================
// profile-plan-pivot-range.test.ts
// =====================================================================
//
// Unit tests for `pivotCenteredRange` — the pure range helper behind the
// viewer's "auto-fit domain" button. Given the current pivot value and
// the source/prior auto-width ([autoLo, autoHi] from resolveSweepRange /
// fourSigmaQuantileRange), it re-centers a window of that width on the
// pivot, honoring the support kind (bounded intervals keep their natural
// bounds; half-bounded supports clamp at 0; integer leaves snap out).
//
// This is the testable core; the viewer's runAutoDomain is the async
// DOM/worker glue around it.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const eng = require('../index.ts');
const pivotCenteredRange = eng.orchestrator.pivotCenteredRange;

test('reals: centers the auto-width window on the pivot', () => {
  // width = 2, pivot = 5 → [4, 6]
  assert.deepEqual(pivotCenteredRange(5, 0, 2, { kind: 'reals' }), [4, 6]);
  // negative pivot
  assert.deepEqual(pivotCenteredRange(-3, -1, 1, { kind: 'reals' }), [-4, -2]);
});

test('posreals: clamps the low edge at 0 when the window would go negative', () => {
  // width = 4, pivot = 0.5 → raw [-1.5, 2.5] → clamp lo → [0, 2.5]
  assert.deepEqual(pivotCenteredRange(0.5, 0, 4, { kind: 'posreals' }), [0, 2.5]);
  // well clear of 0 → no clamp
  assert.deepEqual(pivotCenteredRange(10, 0, 4, { kind: 'posreals' }), [8, 12]);
});

test('nonnegreals: clamps the low edge at 0 like posreals', () => {
  assert.deepEqual(pivotCenteredRange(1, 0, 4, { kind: 'nonnegreals' }), [0, 3]);
});

test('bounded interval: keeps its natural bounds (pivot-independent)', () => {
  // The natural domain of a bounded axis IS the interval — re-centering
  // would clip it. pivot near an edge must not move the bounds.
  assert.deepEqual(pivotCenteredRange(9, 0, 10, { kind: 'interval' }), [0, 10]);
  assert.deepEqual(pivotCenteredRange(0.9, 0, 1, { kind: 'unitinterval' }), [0, 1]);
});

test('integer leaf: snaps the window outward to integer bounds', () => {
  // width = 3, pivot = 5 → raw [3.5, 6.5] → floor/ceil → [3, 7]
  assert.deepEqual(pivotCenteredRange(5, 0, 3, { kind: 'integers', isInt: true }), [3, 7]);
});

test('degenerate auto-width: falls back to a sensible window', () => {
  // width = 0 → fall back to |pivot| (=10) → [5, 15]
  assert.deepEqual(pivotCenteredRange(10, 2, 2, { kind: 'reals' }), [5, 15]);
  // width = 0, pivot = 0 → fall back to 1 → [-0.5, 0.5]
  assert.deepEqual(pivotCenteredRange(0, 0, 0, { kind: 'reals' }), [-0.5, 0.5]);
});

test('non-finite pivot: returns the auto-width unchanged (no-op)', () => {
  assert.deepEqual(pivotCenteredRange(NaN, 1, 2, { kind: 'reals' }), [1, 2]);
  assert.deepEqual(pivotCenteredRange(Infinity, 1, 2, { kind: 'reals' }), [1, 2]);
});
