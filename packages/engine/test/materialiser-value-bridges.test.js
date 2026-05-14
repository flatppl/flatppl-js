'use strict';

// Tests for the Phase 4a Value ↔ Measure bridges: `valueOf(m)` builds
// a Value view of a Measure (sharing storage with .samples), and
// `measureFromValue(v, extras)` is the reverse constructor.
//
// Phase 4a only introduces the helpers; handlers don't populate
// `.value` yet (that's 4b). valueOf falls back to constructing the
// Value from .samples + .dims for unmigrated handlers.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const materialiser = require('..').materialiser;
const valueLib     = require('..').value;
const { valueOf, measureFromValue } = materialiser;

// =====================================================================
// valueOf — Measure → Value view
// =====================================================================

test('valueOf: scalar-atom Measure → shape=[N] Value, storage shared', () => {
  const samples = new Float64Array([1, 2, 3, 4]);
  const m = { samples: samples, logWeights: null, logTotalmass: 0, n_eff: 4 };
  const v = valueOf(m);
  assert.deepEqual(v.shape, [4]);
  assert.equal(v.data, samples, 'data must SHARE storage with samples');
});

test('valueOf: vector-atom Measure (dims=[k]) → shape=[N, k] Value', () => {
  // 3 atoms × 2 components — atom-major flat storage.
  const samples = new Float64Array([1, 2, 3, 4, 5, 6]);
  const m = { samples: samples, dims: [2], logWeights: null,
              logTotalmass: 0, n_eff: 3 };
  const v = valueOf(m);
  assert.deepEqual(v.shape, [3, 2]);
  assert.equal(v.data, samples);
});

test('valueOf: matrix-atom Measure (dims=[m, n]) → shape=[N, m, n] Value', () => {
  // 2 atoms × 2x2 matrix = 8 elements.
  const samples = new Float64Array(8);
  const m = { samples: samples, dims: [2, 2], logWeights: null,
              logTotalmass: 0, n_eff: 2 };
  const v = valueOf(m);
  assert.deepEqual(v.shape, [2, 2, 2]);
});

test('valueOf: returns null for record-shaped Measure (no top-level .samples)', () => {
  const m = {
    fields: { a: { samples: new Float64Array(4) }, b: { samples: new Float64Array(4) } },
    logTotalmass: 0,
    n_eff: 4,
  };
  assert.equal(valueOf(m), null);
});

test('valueOf: prefers existing .value field (Phase 4b ready)', () => {
  // When a handler has already populated .value (Phase 4b migration),
  // valueOf returns it directly rather than rebuilding from .samples.
  const explicit = valueLib.scalar(42);  // shape=[]; intentionally weird
  const m = {
    samples: new Float64Array([42]),
    value: explicit,
    logWeights: null, logTotalmass: 0, n_eff: 1,
  };
  assert.equal(valueOf(m), explicit);
});

test('valueOf: null on missing measure / non-samples shapes', () => {
  assert.equal(valueOf(null), null);
  assert.equal(valueOf(undefined), null);
  assert.equal(valueOf({}), null);
});

// =====================================================================
// measureFromValue — Value → Measure
// =====================================================================

test('measureFromValue: scalar-atom Value shape=[N] → standard Measure', () => {
  const v = valueLib.batchedScalar(new Float64Array([1, 2, 3, 4]));
  const m = measureFromValue(v);
  assert.equal(m.samples, v.data, 'samples must share storage with value.data');
  assert.equal(m.value, v);
  assert.equal(m.logWeights, null);
  assert.equal(m.logTotalmass, 0);
  assert.equal(m.n_eff, 4);
  assert.equal(m.dims, undefined);
});

test('measureFromValue: vector-atom Value shape=[N, k] → dims=[k]', () => {
  const v = valueLib.batchedVector(new Float64Array([1, 2, 3, 4, 5, 6]), 2);
  const m = measureFromValue(v);
  assert.deepEqual(m.dims, [2]);
  assert.equal(m.samples.length, 6);
  assert.equal(m.value, v);
});

test('measureFromValue: matrix-atom shape=[N, m, n] → dims=[m, n]', () => {
  const v = { shape: [2, 3, 4], data: new Float64Array(24) };
  const m = measureFromValue(v);
  assert.deepEqual(m.dims, [3, 4]);
  assert.equal(m.samples.length, 24);
});

test('measureFromValue: passes through extras (logWeights, logTotalmass, n_eff)', () => {
  const v = valueLib.batchedScalar(new Float64Array([1, 2, 3]));
  const w = new Float64Array([-1, -1, -1]);
  const m = measureFromValue(v, {
    logWeights: w,
    logTotalmass: -2.5,
    n_eff: 2,
  });
  assert.equal(m.logWeights, w);
  assert.equal(m.logTotalmass, -2.5);
  assert.equal(m.n_eff, 2);
});

test('measureFromValue: rejects shape=[] (no atom axis)', () => {
  assert.throws(
    () => measureFromValue(valueLib.scalar(3)),
    /no atom axis/);
});

test('measureFromValue: rejects non-Value', () => {
  assert.throws(
    () => measureFromValue(new Float64Array([1, 2, 3])),
    /not a Value/);
});

// =====================================================================
// Round-trip: valueOf(measureFromValue(v)) ≡ v (modulo identity)
// =====================================================================

test('round-trip: measureFromValue then valueOf returns the same value', () => {
  const v = valueLib.batchedVector(new Float64Array([1, 2, 3, 4]), 2);
  const m = measureFromValue(v);
  const recovered = valueOf(m);
  // measureFromValue stashes the original Value in .value, so valueOf
  // returns it directly (same reference, free).
  assert.equal(recovered, v);
});
