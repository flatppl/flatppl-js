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
} = require('../histogram');

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
