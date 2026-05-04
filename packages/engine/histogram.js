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
 * Weight-aware: when opts.logWeights is provided, both the trim
 * quantiles and the IQR (which sets the FD bin width) are computed
 * over the *weighted* empirical CDF, and bin heights are accumulated
 * with normalised weights instead of unit counts. With logWeights
 * absent or null, the routine falls back to the unweighted-uniform
 * path which collapses to the original behaviour byte-for-byte.
 *
 * @param {Float64Array|number[]} samples
 * @param {object} [opts]
 * @param {Float64Array} [opts.logWeights]  per-atom log-weights (length matches samples).
 *                                          Pass null/undefined for uniform-weight measures.
 * @param {number} [opts.trimQ=0.005]       trim each tail to this quantile
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

  const lw = opts.logWeights;
  const weighted = lw && lw.length === n;

  // For weighted: pre-compute normalised weights once, sort sample
  // indices by value. For unweighted: sort the values directly (no
  // weight bookkeeping needed). Quantile lookups happen against the
  // sorted view in both cases.
  let sorted, sortedW;
  if (weighted) {
    const norm = normaliseWeights(lw);
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    // Sort by sample value. Indirection lets us keep the per-atom
    // weight aligned with the sorted value.
    idx.sort((a, b) => samples[a] - samples[b]);
    sorted  = new Float64Array(n);
    sortedW = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      sorted[i]  = samples[idx[i]];
      sortedW[i] = norm[idx[i]];
    }
  } else {
    sorted = Float64Array.from(samples);
    sorted.sort();
  }

  const trimQ = opts.trimQ != null ? opts.trimQ : 0.005;
  const qFn = weighted
    ? (q) => weightedQuantileSorted(sorted, sortedW, q)
    : (q) => quantileSorted(sorted, q);
  const lo = qFn(trimQ);
  const hi = qFn(1 - trimQ);
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

  const q1 = qFn(0.25);
  const q3 = qFn(0.75);
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

  // Bin accumulation. Unweighted: each in-trim sample contributes 1
  // to its bin → normalisation factor = 1 / (n * binWidth) gives the
  // pdf-scale height we want. Weighted: each contributes its
  // normalised weight (which sums to 1 across all atoms) → norm
  // factor = 1 / binWidth, since the weighted accumulator already
  // accounts for total mass.
  const counts = new Float64Array(nBins);
  if (weighted) {
    const norm = normaliseWeights(lw);
    for (let i = 0; i < n; i++) {
      const v = samples[i];
      if (v < lo || v > hi) continue;
      let bin = Math.floor((v - lo) / binWidth);
      if (bin >= nBins) bin = nBins - 1;
      if (bin < 0) bin = 0;
      counts[bin] += norm[i];
    }
  } else {
    for (let i = 0; i < n; i++) {
      const v = samples[i];
      if (v < lo || v > hi) continue;
      let bin = Math.floor((v - lo) / binWidth);
      if (bin >= nBins) bin = nBins - 1;
      if (bin < 0) bin = 0;
      counts[bin]++;
    }
  }
  const norm = weighted ? 1 / binWidth : 1 / (n * binWidth);
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
 *
 * Weight-aware: when `opts.logWeights` is provided, each atom
 * contributes its normalised weight to its integer bin instead of a
 * unit count. With null/undefined weights this collapses to the
 * original unweighted-uniform behaviour.
 *
 * @param {Float64Array|number[]} samples
 * @param {object} [opts]
 * @param {Float64Array} [opts.logWeights]
 */
function integerHistogram(samples, opts = {}) {
  const n = samples.length;
  if (n === 0) {
    return { xs: new Float64Array(0), ys: new Float64Array(0), support: [0, 0], reference: 'counting' };
  }
  const lw = opts && opts.logWeights;
  const weighted = lw && lw.length === n;

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
  if (weighted) {
    // Weighted: accumulate normalised weights (which sum to 1) into
    // their integer bins. Result is already on the probability scale
    // — no extra division step.
    const norm = normaliseWeights(lw);
    for (let i = 0; i < n; i++) {
      const k = Math.round(samples[i]) - lo;
      ys[k] += norm[i];
    }
  } else {
    // Unweighted: count atoms, then divide by N so heights sum to 1.
    for (let i = 0; i < n; i++) {
      const k = Math.round(samples[i]) - lo;
      ys[k] += 1;
    }
    for (let i = 0; i < span; i++) ys[i] /= n;
  }
  return { xs, ys, support: [lo, hi], reference: 'counting' };
}

/**
 * Convert per-atom log-weights into linear-space normalised weights
 * (sum = 1), in a numerically stable way: subtract the max log-weight
 * before exp, then divide by the total. Returns a fresh Float64Array.
 *
 * Lives here in histogram.js (rather than being imported from
 * empirical.js) because it's a histogram-implementation detail and we
 * want histogram.js dep-free of empirical.js — they're peers, not in
 * a stack. The duplication with empirical.logSumExp's stability trick
 * is small (~10 lines) and lets the two modules evolve independently.
 */
function normaliseWeights(logWeights) {
  const n = logWeights.length;
  if (n === 0) return new Float64Array(0);
  let max = logWeights[0];
  for (let i = 1; i < n; i++) if (logWeights[i] > max) max = logWeights[i];
  if (!Number.isFinite(max)) {
    // All -Infinity → can't normalise; return uniform as a safe fallback.
    const out = new Float64Array(n);
    out.fill(1 / n);
    return out;
  }
  const out = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i++) {
    out[i] = Math.exp(logWeights[i] - max);
    total += out[i];
  }
  if (total > 0) {
    for (let i = 0; i < n; i++) out[i] /= total;
  }
  return out;
}

/**
 * Weighted quantile from sorted (samples, normalised-weights) arrays.
 * Inverts the weighted empirical CDF: q-th quantile is the value
 * where cumulative weight first reaches q. Linearly interpolates
 * between adjacent atoms to avoid quantile values jumping
 * discontinuously across atoms.
 *
 * Caller passes both arrays already sorted by the underlying sample
 * value (sortedSamples), with sortedNormWeights re-permuted to match.
 * normaliseWeights produces the per-atom weights in the original
 * order; sort with index indirection to preserve the pairing.
 */
function weightedQuantileSorted(sortedSamples, sortedNormWeights, q) {
  const n = sortedSamples.length;
  if (n === 0) return NaN;
  if (n === 1) return sortedSamples[0];
  let cum = 0;
  for (let i = 0; i < n; i++) {
    const cumNext = cum + sortedNormWeights[i];
    if (cumNext >= q) {
      // q lies between cum and cumNext. Linearly interpolate the value
      // between (sortedSamples[i-1], sortedSamples[i]) by where in
      // [cum, cumNext] the target q sits. For i=0 the only sensible
      // answer is sortedSamples[0] (no left neighbour to interp from).
      if (i === 0 || sortedNormWeights[i] <= 0) return sortedSamples[i];
      const t = (q - cum) / sortedNormWeights[i];
      return sortedSamples[i - 1] * (1 - t) + sortedSamples[i] * t;
    }
    cum = cumNext;
  }
  return sortedSamples[n - 1];
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
