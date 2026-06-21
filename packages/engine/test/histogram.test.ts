'use strict';

// Tests for engine/histogram.js — pure-numeric histogram helpers used
// by the visualizer's main thread (and previously by the worker).
//
// Coverage:
//   - freedmanDiaconisHistogram: bin uniformity, area normalisation,
//     degenerate (all-equal) fallback, trimQ=0 covers full range
//   - integerHistogram: probabilities sum to 1, atoms are integers
//   - quantileSorted: matches numpy-style linear interpolation
//   - meanSd: matches naive computation

const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  freedmanDiaconisHistogram, integerHistogram, quantileSorted, meanSd,
} = require('../histogram.ts');

test('freedmanDiaconisHistogram: bins are equal-width and area sums near 1', () => {
  // 5000 standard-normal samples via Box-Muller + LCG, deterministic.
  const xs = new Float64Array(5000);
  let s = 12345;
  function lcg() { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; }
  for (let i = 0; i < xs.length; i += 2) {
    const u = Math.max(lcg(), 1e-10), v = lcg();
    const r = Math.sqrt(-2 * Math.log(u));
    xs[i]     = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < xs.length) xs[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  const h = freedmanDiaconisHistogram(xs);
  assert.ok(h.binWidth > 0);
  assert.equal(h.xs.length, h.ys.length);
  assert.equal(h.binEdges.length, h.xs.length + 1);
  // Equal width across bins (within float epsilon).
  for (let i = 0; i < h.xs.length; i++) {
    const w = h.binEdges[i + 1] - h.binEdges[i];
    assert.ok(Math.abs(w - h.binWidth) < 1e-9);
  }
  // Area ≈ 1 - 2*trimQ = 0.99 by default.
  let area = 0;
  for (let i = 0; i < h.ys.length; i++) area += h.ys[i] * h.binWidth;
  assert.ok(area > 0.97 && area < 1.01, `area ${area} not in [0.97, 1.01]`);
});

test('freedmanDiaconisHistogram: degenerate (all-equal) yields single-bin fallback', () => {
  const xs = new Float64Array([3, 3, 3, 3, 3]);
  const h = freedmanDiaconisHistogram(xs);
  assert.equal(h.xs.length, 1);
  assert.equal(h.binWidth, 1);
});

test('freedmanDiaconisHistogram: trimQ=0 keeps all samples in range', () => {
  const xs = new Float64Array([0, 1, 2, 3, 4, 5, 100]);
  const h = freedmanDiaconisHistogram(xs, { trimQ: 0 });
  let total = 0;
  for (let i = 0; i < h.ys.length; i++) total += h.ys[i] * h.binWidth;
  assert.ok(Math.abs(total - 1) < 1e-9, `total area ${total} ≠ 1 with trimQ=0`);
});

test('integerHistogram: probabilities sum to 1, atoms are integers', () => {
  const r = integerHistogram(new Float64Array([0, 1, 1, 2, 2, 2, 3]));
  let s = 0;
  for (let i = 0; i < r.ys.length; i++) s += r.ys[i];
  assert.ok(Math.abs(s - 1) < 1e-12);
  for (let i = 0; i < r.xs.length; i++) assert.ok(Number.isInteger(r.xs[i]));
  assert.equal(r.support[0], 0);
  assert.equal(r.support[1], 3);
});

test('integerHistogram: empty samples returns empty arrays', () => {
  const r = integerHistogram(new Float64Array(0));
  assert.equal(r.xs.length, 0);
  assert.equal(r.ys.length, 0);
});

test('quantileSorted: matches NumPy linear interpolation', () => {
  const a = new Float64Array([1, 2, 3, 4, 5]);
  assert.equal(quantileSorted(a, 0), 1);
  assert.equal(quantileSorted(a, 1), 5);
  assert.equal(quantileSorted(a, 0.5), 3);
  assert.equal(quantileSorted(a, 0.25), 2);
  assert.equal(quantileSorted(a, 0.75), 4);
});

test('meanSd: matches naive computation', () => {
  const samples = [1, 2, 3, 4, 5];
  const { mean, sd } = meanSd(samples);
  assert.equal(mean, 3);
  // Population sd of 1..5 = sqrt(2).
  assert.ok(Math.abs(sd - Math.sqrt(2)) < 1e-12);
});

// =====================================================================
// Weighted-histogram path: opts.logWeights branches in both
// freedmanDiaconisHistogram and integerHistogram.
// =====================================================================

test('integerHistogram: explicit uniform logWeights matches unweighted', () => {
  // logWeights = [-log(N) ... -log(N)] should give bit-for-bit the
  // same result as the count-and-divide-by-N path.
  const samples = new Float64Array([0, 1, 1, 2, 2, 2, 3]);
  const N = samples.length;
  const lw = new Float64Array(N);
  lw.fill(-Math.log(N));
  const a = integerHistogram(samples);
  const b = integerHistogram(samples, { logWeights: lw });
  assert.deepEqual(Array.from(a.xs), Array.from(b.xs));
  for (let i = 0; i < a.ys.length; i++) {
    assert.ok(Math.abs(a.ys[i] - b.ys[i]) < 1e-12,
      `ys[${i}] differs: ${a.ys[i]} vs ${b.ys[i]}`);
  }
});

test('integerHistogram: weighted branch picks up raw atomic mass (mass-faithful)', () => {
  // Three atoms at integers 0, 1, 2 with weights [1, 4, 1] (linear).
  // Per spec §sec:measure-algebra ("operations never rescale"), the
  // histogram reflects the measure's actual atomic mass — bars
  // [1, 4, 1] with total mass 6, NOT a probability-normalised
  // [1/6, 4/6, 1/6]. Run normalize(...) to land on probability scale.
  const samples = new Float64Array([0, 1, 2]);
  const lw = new Float64Array([Math.log(1), Math.log(4), Math.log(1)]);
  const r = integerHistogram(samples, { logWeights: lw });
  assert.equal(r.reference, 'counting');
  assert.deepEqual(Array.from(r.xs), [0, 1, 2]);
  let s = 0;
  for (let i = 0; i < r.ys.length; i++) s += r.ys[i];
  assert.ok(Math.abs(s - 6) < 1e-12, `total mass ${s} should be 6`);
  assert.ok(Math.abs(r.ys[0] - 1) < 1e-12);
  assert.ok(Math.abs(r.ys[1] - 4) < 1e-12);
  assert.ok(Math.abs(r.ys[2] - 1) < 1e-12);
});

test('freedmanDiaconisHistogram: uniform weights produce approximately the same shape as unweighted', () => {
  // 5000 standard-normal samples. The weighted path uses cumulative-
  // weight quantiles (break-points at sum w + 0.5*w_i style), which
  // differs from the rank-based "type-7" quantile by ~half-a-sample
  // in position. So binWidth, edges, and per-bin heights aren't
  // expected to be byte-identical — but the overall histogram shape
  // (bin count within ±2, area ≈ same, peak in roughly the same
  // place) must agree.
  const xs = new Float64Array(5000);
  let s = 12345;
  function lcg() { s = (s * 1664525 + 1013904223) >>> 0; return s / 0x100000000; }
  for (let i = 0; i < xs.length; i += 2) {
    const u = Math.max(lcg(), 1e-10), v = lcg();
    const r = Math.sqrt(-2 * Math.log(u));
    xs[i]     = r * Math.cos(2 * Math.PI * v);
    if (i + 1 < xs.length) xs[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  const lw = new Float64Array(xs.length);
  lw.fill(-Math.log(xs.length));
  const a = freedmanDiaconisHistogram(xs);
  const b = freedmanDiaconisHistogram(xs, { logWeights: lw });
  assert.ok(Math.abs(a.xs.length - b.xs.length) <= 2, 'bin counts within ±2');
  // Total area within rounding; both should integrate to ~1 (= 1 - 2·trimQ).
  function area(h: any) {
    let s = 0;
    for (let i = 0; i < h.ys.length; i++) s += h.ys[i] * h.binWidth;
    return s;
  }
  assert.ok(Math.abs(area(a) - area(b)) < 0.01,
    `areas: unweighted ${area(a)}, weighted ${area(b)}`);
  // Peak in the same neighbourhood (≈ 0 for standard normal).
  function peakX(h: any) {
    let imax = 0;
    for (let i = 1; i < h.ys.length; i++) if (h.ys[i] > h.ys[imax]) imax = i;
    return h.xs[imax];
  }
  assert.ok(Math.abs(peakX(a) - peakX(b)) < 2 * a.binWidth);
});

test('freedmanDiaconisHistogram: heavy weight on one atom shifts the mass', () => {
  // Symmetric samples around 0; put 90% of the weight on the lone
  // atom at +5. The histogram bulk should shift to the right of 0.
  const xs = new Float64Array([-2, -1, 0, 1, 2, 5]);
  // Linear weights: [.02, .02, .02, .02, .02, .9]  → log them.
  const w = [0.02, 0.02, 0.02, 0.02, 0.02, 0.9];
  const lw = Float64Array.from(w.map(Math.log));
  const r = freedmanDiaconisHistogram(xs, { logWeights: lw, trimQ: 0 });
  // Total area = 1 (no trim). Most mass should be concentrated in
  // the bin containing x = 5.
  let area = 0, peak = -Infinity, peakX = NaN;
  for (let i = 0; i < r.ys.length; i++) {
    area += r.ys[i] * r.binWidth;
    if (r.ys[i] > peak) { peak = r.ys[i]; peakX = r.xs[i]; }
  }
  assert.ok(Math.abs(area - 1) < 1e-9, `area ${area} ≠ 1`);
  assert.ok(peakX >= 4 && peakX <= 6, `peak at x = ${peakX} (expected near 5)`);
});

test('freedmanDiaconisHistogram: weighted quantile-based trim ignores low-weight outliers', () => {
  // Bulk of weight on atoms around 0; one wildly outlying atom at
  // +1000 with negligible weight. With weighted trim quantiles,
  // the visible range should NOT extend out to 1000.
  const xs = new Float64Array([-1, 0, 0, 0, 0, 1, 1000]);
  const w = [1, 1, 1, 1, 1, 1, 1e-9];
  const lw = Float64Array.from(w.map(Math.log));
  const r = freedmanDiaconisHistogram(xs, { logWeights: lw });
  // Trim at 0.5% per tail of the *weighted* CDF; the 1000 atom holds
  // nearly zero weight so it's well outside the trim range.
  assert.ok(r.support[1] < 100, `hi support = ${r.support[1]} should be << 1000`);
});

test('normaliseWeights and weightedQuantileSorted are exported and correct', () => {
  const h = require('../histogram.ts');
  assert.equal(typeof h.normaliseWeights, 'function');
  assert.equal(typeof h.weightedQuantileSorted, 'function');
  // Uniform weights over [0,1,2,3,4]: median ~= 2.
  const w = h.normaliseWeights(new Float64Array([0, 0, 0, 0, 0])); // logweights all 0 → uniform
  let sum = 0; for (const x of w) sum += x;
  assert.ok(Math.abs(sum - 1) < 1e-12);
  const med = h.weightedQuantileSorted(new Float64Array([0, 1, 2, 3, 4]), w, 0.5);
  assert.ok(med >= 1.5 && med <= 2.5, `median ${med}`);
});
