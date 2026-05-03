'use strict';

// Pure-numeric histogram helpers, decoupled from the sampler/worker
// stack so they can run on either the main thread (visualPanel) or the
// worker. No stdlib pull-in — all math is JS-native — which is why
// this lives in its own module rather than inside sampler.js or
// worker.js.
//
// Two binning strategies, picked by the caller:
//
//   * Freedman-Diaconis (continuous, lebesgue reference) — bin width
//     2·IQR·n^(-1/3); robust to outliers; equal-width bars overlay
//     cleanly with a smooth PDF curve.
//
//   * Integer atoms (discrete, counting reference) — one bin per
//     integer between min and max(samples).
//
// Both return a uniform `{ xs, ys, support, reference }` shape so the
// rendering path can dispatch on `reference` without caring which
// estimator produced the bars. FD additionally returns `binEdges` and
// `binWidth` so a bar-style chart can size rectangles directly.

/**
 * Equal-width histogram with bin width chosen by the Freedman-Diaconis
 * rule. The visible x-range is trimmed to a quantile interval (default
 * [q0.005, q0.995]) so a single far-away outlier doesn't compress the
 * useful range; samples outside the trim are dropped from the bin
 * counts. Bars are area-normalised to PDF scale so they can be
 * overlaid against a stdlib analytical PDF directly.
 *
 * @param {Float64Array|number[]} samples
 * @param {object} [opts]
 * @param {number} [opts.trimQ=0.005]  trim each tail to this quantile
 * @param {number} [opts.maxBins=200]
 * @param {number} [opts.minBins=8]
 */
function freedmanDiaconisHistogram(samples, opts = {}) {
  const n = samples.length;
  if (n === 0) {
    return {
      xs: new Float64Array(0), ys: new Float64Array(0),
      binEdges: new Float64Array(0), binWidth: 0,
      support: [0, 0], reference: 'lebesgue',
    };
  }
  const sorted = Float64Array.from(samples);
  sorted.sort();

  const trimQ = opts.trimQ != null ? opts.trimQ : 0.005;
  const lo = quantileSorted(sorted, trimQ);
  const hi = quantileSorted(sorted, 1 - trimQ);
  if (!(hi > lo)) {
    // All samples coincide — emit a single 1-wide bin centred on the
    // common value so the chart doesn't crash on zero-width bars.
    const v = sorted[0];
    return {
      xs: new Float64Array([v]),
      ys: new Float64Array([1]),
      binEdges: new Float64Array([v - 0.5, v + 0.5]),
      binWidth: 1,
      support: [v - 0.5, v + 0.5], reference: 'lebesgue',
    };
  }

  const q1 = quantileSorted(sorted, 0.25);
  const q3 = quantileSorted(sorted, 0.75);
  const iqr = q3 - q1;
  let binWidth;
  if (iqr > 0) binWidth = 2 * iqr * Math.pow(n, -1 / 3);
  else         binWidth = (hi - lo) / Math.max(Math.sqrt(n), 1);
  if (!(binWidth > 0)) binWidth = (hi - lo) / 30;

  const minBins = opts.minBins != null ? opts.minBins : 8;
  const maxBins = opts.maxBins != null ? opts.maxBins : 200;
  let nBins = Math.max(minBins, Math.min(maxBins, Math.ceil((hi - lo) / binWidth)));
  binWidth = (hi - lo) / nBins;

  const binEdges = new Float64Array(nBins + 1);
  for (let i = 0; i <= nBins; i++) binEdges[i] = lo + i * binWidth;

  const counts = new Float64Array(nBins);
  for (let i = 0; i < n; i++) {
    const v = samples[i];
    if (v < lo || v > hi) continue;
    let bin = Math.floor((v - lo) / binWidth);
    if (bin >= nBins) bin = nBins - 1;
    if (bin < 0) bin = 0;
    counts[bin]++;
  }
  const norm = 1 / (n * binWidth);
  const ys = new Float64Array(nBins);
  const xs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) {
    ys[i] = counts[i] * norm;
    xs[i] = binEdges[i] + binWidth / 2;
  }
  return { xs, ys, binEdges, binWidth, support: [lo, hi], reference: 'lebesgue' };
}

/**
 * Probability mass function via integer-bin histogram. Bins are unit
 * width centred on each integer atom from min(samples) to max(samples).
 * Heights are normalised to sum to 1 (probability scale).
 */
function integerHistogram(samples) {
  const n = samples.length;
  if (n === 0) {
    return { xs: new Float64Array(0), ys: new Float64Array(0), support: [0, 0], reference: 'counting' };
  }
  let lo = +Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = Math.round(samples[i]);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo + 1;
  const xs = new Float64Array(span);
  const ys = new Float64Array(span);
  for (let i = 0; i < span; i++) xs[i] = lo + i;
  for (let i = 0; i < n; i++) {
    const k = Math.round(samples[i]) - lo;
    ys[k] += 1;
  }
  for (let i = 0; i < span; i++) ys[i] /= n;
  return { xs, ys, support: [lo, hi], reference: 'counting' };
}

/**
 * Sorted-array quantile via linear interpolation. Caller passes an
 * already-sorted typed array.
 */
function quantileSorted(sorted, q) {
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const t = q * (n - 1);
  const i = Math.floor(t);
  const f = t - i;
  if (i + 1 >= n) return sorted[n - 1];
  return sorted[i] * (1 - f) + sorted[i + 1] * f;
}

function meanSd(samples) {
  const n = samples.length;
  if (n === 0) return { mean: NaN, sd: NaN };
  let s = 0;
  for (let i = 0; i < n; i++) s += samples[i];
  const mean = s / n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean;
    v += d * d;
  }
  return { mean, sd: Math.sqrt(v / n) };
}

module.exports = {
  freedmanDiaconisHistogram,
  integerHistogram,
  quantileSorted,
  meanSd,
};
