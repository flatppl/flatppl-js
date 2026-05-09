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

// =====================================================================
// PSIS k̂ + importance-sampling quality classifier
// =====================================================================
//
// Reference values were generated against scipy.stats.genpareto MLE +
// the loo R package's psis() routine; we accept a ±0.1 tolerance on k̂
// since Zhang-Stephens is an empirical-Bayes posterior mean and minor
// implementation differences (grid resolution, prior choice) shift the
// estimate slightly.

const { paretoKHat, paretoKThreshold, importanceSamplingQuality,
        estimateDof, recordMeasure, arrayMeasure } = require('../empirical');

// Deterministic Pareto sampler: F⁻¹(u) = (1-u)^(-1/α) - 1 has shape
// k̂ = 1/α for the GPD. Use an LCG for reproducibility.
function paretoSamples(alpha, n, seed) {
  let s = (seed || 1) >>> 0;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    const u = (s / 0x100000000);
    out[i] = Math.pow(1 - u, -1 / alpha) - 1;  // shifted so support starts at 0
  }
  return out;
}

test('paretoKHat: pure-Pareto α=4 sample → k̂ ≈ 0.25', () => {
  const w = paretoSamples(4, 5000, 42);
  const k = paretoKHat(w);
  // Zhang-Stephens posterior mean; tolerance generous because of the
  // empirical-Bayes prior and finite-sample variability.
  assert.ok(Math.abs(k - 0.25) < 0.15, `k̂=${k}, expected ≈0.25`);
});

test('paretoKHat: pure-Pareto α=2 sample → k̂ ≈ 0.5', () => {
  const w = paretoSamples(2, 5000, 7);
  const k = paretoKHat(w);
  assert.ok(Math.abs(k - 0.5) < 0.15, `k̂=${k}, expected ≈0.5`);
});

test('paretoKHat: pure-Pareto α=1 sample → k̂ ≈ 1.0', () => {
  const w = paretoSamples(1, 5000, 11);
  const k = paretoKHat(w);
  // α=1 is the boundary of integrable mean; the Zhang-Stephens
  // estimator's finite-sample upward bias is largest here (~0.2).
  assert.ok(Math.abs(k - 1.0) < 0.3, `k̂=${k}, expected ≈1.0`);
});

test('paretoKHat: degenerate input → NaN', () => {
  assert.ok(Number.isNaN(paretoKHat(new Float64Array([1, 1, 1]))));
  assert.ok(Number.isNaN(paretoKHat(new Float64Array([]))));
});

test('paretoKThreshold: matches the canonical sample-size-aware bound', () => {
  // Vehtari et al. 2024: k★ = min(0.7, 1 − 1/log10(N))
  assert.ok(Math.abs(paretoKThreshold(100) - 0.5) < 1e-9);
  assert.ok(Math.abs(paretoKThreshold(10000) - 0.7) < 1e-9);  // capped
  assert.ok(Math.abs(paretoKThreshold(1000) - (1 - 1 / 3)) < 1e-9);
});

test('importanceSamplingQuality: unweighted measure → good with ratio=1', () => {
  const m = { samples: new Float64Array(1000).fill(0.5), logWeights: null };
  const q = importanceSamplingQuality(m, 1);
  assert.equal(q.label, 'good');
  assert.equal(q.ratio, 1);
  assert.ok(Number.isNaN(q.kHat));
});

test('importanceSamplingQuality: nearly-uniform log-weights → good', () => {
  const N = 10000;
  const samples = new Float64Array(N);
  const logWeights = new Float64Array(N);
  // Tiny noise around -log(N); ESS/N stays near 1.
  let s = 0xdeadbeef;
  for (let i = 0; i < N; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    samples[i] = i;
    logWeights[i] = -Math.log(N) + (s / 0x100000000 - 0.5) * 0.01;
  }
  const q = importanceSamplingQuality({ samples, logWeights }, 1);
  assert.equal(q.label, 'good');
  assert.ok(q.ratio > 0.99, `ratio=${q.ratio}`);
});

test('importanceSamplingQuality: heavy-tailed weights → degrades to bad/unusable', () => {
  // Build log-weights from a Pareto-α=1 sample so k̂ should be ≈ 1.0
  // (boundary between bad and unusable). 5000 atoms.
  const N = 5000;
  const samples = new Float64Array(N);
  for (let i = 0; i < N; i++) samples[i] = i;
  const logW = new Float64Array(N);
  const tailSamples = paretoSamples(1, N, 99);
  for (let i = 0; i < N; i++) logW[i] = Math.log(1 + tailSamples[i]);
  const q = importanceSamplingQuality({ samples, logWeights: logW }, 1);
  assert.ok(q.label === 'bad' || q.label === 'unusable',
            `expected bad/unusable, got ${q.label}; k̂=${q.kHat}`);
});

test('importanceSamplingQuality: single-particle dominance → unusable', () => {
  // One atom carries 90% of mass.
  const N = 1000;
  const samples = new Float64Array(N);
  for (let i = 0; i < N; i++) samples[i] = i;
  const logW = new Float64Array(N);
  for (let i = 0; i < N; i++) logW[i] = Math.log(0.1 / (N - 1));
  logW[0] = Math.log(0.9);
  const q = importanceSamplingQuality({ samples, logWeights: logW }, 1);
  assert.equal(q.label, 'unusable');
  assert.ok(q.wmax > 0.5);
});

test('importanceSamplingQuality: small N caps at ok for non-uniform weights', () => {
  // N=50 with mildly non-uniform weights (a real reweighting, not
  // a uniform fill). The small-N cap should prevent green even when
  // ratio looks fine — PSIS k̂ is itself noisy at N<100.
  const N = 50;
  const samples = new Float64Array(N);
  const logW = new Float64Array(N);
  let s = 0xdeadbeef;
  for (let i = 0; i < N; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    samples[i] = i;
    logW[i] = -Math.log(N) + (s / 0x100000000 - 0.5) * 0.5;
  }
  const q = importanceSamplingQuality({ samples, logWeights: logW }, 1);
  assert.notEqual(q.label, 'good');
});

test('importanceSamplingQuality: uniform-within-epsilon weights → unweighted-style result', () => {
  // Even with explicit logWeights, all-equal entries should classify
  // as if logWeights were null: kHat NaN, label 'good', ratio 1.
  // Common when materialiseUniform set logW = -log(N) before any
  // reweighting op was applied.
  const N = 1000;
  const samples = new Float64Array(N);
  const logW = new Float64Array(N).fill(-Math.log(N));
  const q = importanceSamplingQuality({ samples, logWeights: logW }, 1);
  assert.equal(q.label, 'good');
  assert.ok(!Number.isFinite(q.kHat), 'kHat NaN signals uniform/non-applicable');
  assert.equal(q.ratio, 1);
});

test('estimateDof: scalar measure → 1', () => {
  const m = { samples: new Float64Array([1, 2, 3, 4]), logWeights: null };
  assert.equal(estimateDof(m), 1);
});

test('estimateDof: record with two varying scalar fields → 2', () => {
  const f1 = { samples: new Float64Array([1, 2, 3]), logWeights: null };
  const f2 = { samples: new Float64Array([4, 5, 6]), logWeights: null };
  const m = recordMeasure({ a: f1, b: f2 }, null);
  assert.equal(estimateDof(m), 2);
});

test('estimateDof: record with one constant scalar field → 1 (zero-variance dropped)', () => {
  const fVar = { samples: new Float64Array([1, 2, 3]), logWeights: null };
  const fConst = { samples: new Float64Array([7, 7, 7]), logWeights: null };
  const m = recordMeasure({ a: fVar, b: fConst }, null);
  assert.equal(estimateDof(m), 1);
});

test('estimateDof: array(iid) of length 10 across N atoms → 10 (independent slots)', () => {
  const N = 100, K = 10;
  const samples = new Float64Array(N * K);
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < K; s++) samples[i * K + s] = i * K + s;
  }
  const m = arrayMeasure(samples, [K], null);
  assert.equal(estimateDof(m), K);
});
