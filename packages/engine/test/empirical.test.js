'use strict';

// Tests for engine/empirical.js — pure-numeric helpers around the
// EmpiricalMeasure { samples, logWeights } shape.
//
// Coverage:
//   - logSumExp: identities, stability, edge cases
//   - totalLogMass: 0 for null-uniform, logSumExp for explicit
//   - effectiveSampleSize: N for uniform, 1 for degenerate, in-between
//   - materialiseUniform: pass-through for explicit, fills -log(N)
//     for null
//   - systematicResample: indices in range, reproducibility,
//     distribution preservation, single-uniform-call contract
//   - multinomialResample: same coverage shape

const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  logSumExp,
  totalLogMass,
  effectiveSampleSize,
  materialiseUniform,
  systematicResample,
  multinomialResample,
} = require('../empirical');

// =====================================================================
// logSumExp
// =====================================================================

test('logSumExp: empty → -Infinity', () => {
  assert.equal(logSumExp([]), -Infinity);
});

test('logSumExp: single element → that element', () => {
  assert.equal(logSumExp([0]), 0);
  assert.equal(logSumExp([42]), 42);
  assert.equal(logSumExp([-Infinity]), -Infinity);
});

test('logSumExp: log(a)+log(b) identity', () => {
  // logSumExp([log a, log b]) = log(a + b)
  const a = 3, b = 5;
  const got = logSumExp([Math.log(a), Math.log(b)]);
  assert.ok(Math.abs(got - Math.log(a + b)) < 1e-12);
});

test('logSumExp: numerically stable for large values', () => {
  // Without max-shift, exp(1000) overflows. With it we get 1000 + log(2).
  const got = logSumExp([1000, 1000]);
  assert.ok(Math.abs(got - (1000 + Math.log(2))) < 1e-9);
});

test('logSumExp: -Infinity entries are absorbed (treated as zero weight)', () => {
  // log(0 + a + 0 + b) = log(a + b)
  const a = 7, b = 11;
  const got = logSumExp([-Infinity, Math.log(a), -Infinity, Math.log(b)]);
  assert.ok(Math.abs(got - Math.log(a + b)) < 1e-12);
});

test('logSumExp: all -Infinity → -Infinity', () => {
  assert.equal(logSumExp([-Infinity, -Infinity, -Infinity]), -Infinity);
});

// =====================================================================
// totalLogMass
// =====================================================================

test('totalLogMass: null logWeights → 0 (probability measure)', () => {
  // Uniform 1/N over N atoms sums to 1; log is 0. Holds regardless
  // of N — the helper short-circuits without looking at samples.
  const m1 = { samples: new Float64Array(10), logWeights: null };
  const m100 = { samples: new Float64Array(100), logWeights: null };
  assert.equal(totalLogMass(m1), 0);
  assert.equal(totalLogMass(m100), 0);
});

test('totalLogMass: explicit logWeights → logSumExp', () => {
  // Three atoms with weights [1, 2, 3] in linear space → total = 6.
  const m = {
    samples: new Float64Array([10, 20, 30]),
    logWeights: new Float64Array([Math.log(1), Math.log(2), Math.log(3)]),
  };
  assert.ok(Math.abs(totalLogMass(m) - Math.log(6)) < 1e-12);
});

// =====================================================================
// effectiveSampleSize
// =====================================================================

test('effectiveSampleSize: uniform-weight measure → N', () => {
  const m = { samples: new Float64Array(50), logWeights: null };
  assert.equal(effectiveSampleSize(m), 50);
});

test('effectiveSampleSize: explicit uniform → N (matches null)', () => {
  // A materialised uniform weighting should produce the same ESS
  // as the null-uniform shorthand.
  const N = 100;
  const w = new Float64Array(N);
  const c = -Math.log(N);
  for (let i = 0; i < N; i++) w[i] = c;
  const m = { samples: new Float64Array(N), logWeights: w };
  assert.ok(Math.abs(effectiveSampleSize(m) - N) < 1e-9);
});

test('effectiveSampleSize: degenerate (one atom dominates) → ~1', () => {
  // logWeights [0, -Inf, -Inf, …]: only atom 0 has any mass.
  const N = 20;
  const w = new Float64Array(N);
  w.fill(-Infinity);
  w[0] = 0;
  const m = { samples: new Float64Array(N), logWeights: w };
  assert.ok(Math.abs(effectiveSampleSize(m) - 1) < 1e-9);
});

// =====================================================================
// materialiseUniform
// =====================================================================

test('materialiseUniform: null → explicit -log(N) array, samples shared', () => {
  const samples = new Float64Array([1, 2, 3, 4]);
  const m = { samples, logWeights: null };
  const out = materialiseUniform(m);
  assert.equal(out.samples, samples, 'samples must be the same reference');
  assert.ok(out.logWeights instanceof Float64Array);
  assert.equal(out.logWeights.length, 4);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(out.logWeights[i] - (-Math.log(4))) < 1e-12);
  }
});

test('materialiseUniform: pass-through when already explicit', () => {
  const samples = new Float64Array([1, 2, 3]);
  const w = new Float64Array([0, -1, -2]);
  const m = { samples, logWeights: w };
  const out = materialiseUniform(m);
  assert.equal(out, m, 'no-op should return the same object');
});

// =====================================================================
// systematicResample
// =====================================================================

test('systematicResample: indices are in [0, N) and length n', () => {
  const w = new Float64Array([0, 0, 0, 0]); // uniform
  const idx = systematicResample(w, 7, () => 0.5);
  assert.equal(idx.length, 7);
  for (let i = 0; i < 7; i++) {
    assert.ok(idx[i] >= 0 && idx[i] < 4);
  }
});

test('systematicResample: calls prng exactly once', () => {
  const w = new Float64Array([0, 0, 0]);
  let calls = 0;
  systematicResample(w, 100, () => { calls++; return 0.3; });
  assert.equal(calls, 1);
});

test('systematicResample: same prng → same indices (deterministic)', () => {
  const w = new Float64Array([0, -1, -2, -3]);
  const a = systematicResample(w, 50, () => 0.123);
  const b = systematicResample(w, 50, () => 0.123);
  for (let i = 0; i < 50; i++) assert.equal(a[i], b[i]);
});

test('systematicResample: distribution approximates the source weights', () => {
  // Source: weights ∝ [1, 2, 3, 4]. Out of 1000 resamples the histogram
  // should be roughly proportional. Systematic gives exact-ish output
  // for uniform spacing of positions; this is a sanity check, not a
  // tight statistical test.
  const w = new Float64Array([Math.log(1), Math.log(2), Math.log(3), Math.log(4)]);
  const idx = systematicResample(w, 1000, () => 0.5);
  const counts = [0, 0, 0, 0];
  for (let i = 0; i < idx.length; i++) counts[idx[i]]++;
  // Expected: 100, 200, 300, 400. Allow ±2 atoms slack from systematic's
  // deterministic grid alignment.
  assert.ok(Math.abs(counts[0] - 100) <= 2, `counts[0] = ${counts[0]}`);
  assert.ok(Math.abs(counts[1] - 200) <= 2, `counts[1] = ${counts[1]}`);
  assert.ok(Math.abs(counts[2] - 300) <= 2, `counts[2] = ${counts[2]}`);
  assert.ok(Math.abs(counts[3] - 400) <= 2, `counts[3] = ${counts[3]}`);
});

test('systematicResample: empty weights → error', () => {
  assert.throws(() => systematicResample(new Float64Array(0), 10, () => 0.5),
    /no atoms/);
});

test('systematicResample: zero/negative n → error', () => {
  assert.throws(() => systematicResample(new Float64Array([0]), 0, () => 0.5),
    /n must be > 0/);
});

// =====================================================================
// multinomialResample
// =====================================================================

test('multinomialResample: indices in range, length n, prng called n times', () => {
  const w = new Float64Array([0, 0, 0, 0]);
  let calls = 0;
  const idx = multinomialResample(w, 10, () => { calls++; return 0.5; });
  assert.equal(idx.length, 10);
  assert.equal(calls, 10);
  for (let i = 0; i < 10; i++) assert.ok(idx[i] >= 0 && idx[i] < 4);
});

test('multinomialResample: distribution approximates the source (loose)', () => {
  // Multinomial has higher variance than systematic, so the bound is
  // looser. Use a deterministic LCG so the test is reproducible.
  const w = new Float64Array([Math.log(1), Math.log(2), Math.log(3), Math.log(4)]);
  let s = 0xcafebabe;
  function lcg() { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; }
  const idx = multinomialResample(w, 10000, lcg);
  const counts = [0, 0, 0, 0];
  for (let i = 0; i < idx.length; i++) counts[idx[i]]++;
  // Expected: 1000, 2000, 3000, 4000. Allow ±5% (multinomial variance).
  assert.ok(Math.abs(counts[0] - 1000) < 200);
  assert.ok(Math.abs(counts[1] - 2000) < 200);
  assert.ok(Math.abs(counts[2] - 3000) < 200);
  assert.ok(Math.abs(counts[3] - 4000) < 200);
});

test('multinomialResample: degenerate weights → all output indices equal', () => {
  // logWeights = [0, -Inf, -Inf, -Inf] → all mass on atom 0.
  const w = new Float64Array([0, -Infinity, -Infinity, -Infinity]);
  const idx = multinomialResample(w, 50, () => 0.7);
  for (let i = 0; i < 50; i++) assert.equal(idx[i], 0);
});
