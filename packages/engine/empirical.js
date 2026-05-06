'use strict';

// Pure-numeric helpers for the empirical-measure representation used
// throughout the visualizer's sampling stack:
//
//   EmpiricalMeasure
//     { samples:    Float64Array,
//       logWeights: Float64Array | null }      // null = uniform 1/N
//
// This module is the analogue of histogram.js for weights: small,
// stdlib-free, dependency-free helpers around the EmpiricalMeasure
// shape. Everything works in log-space for numerical stability —
// products of likelihoods explode/underflow in linear space, and the
// `null = uniform` convention lets the common-case (variates, plain
// i.i.d. draws) skip allocation entirely.
//
// What's here
//
//   logSumExp(arr)                  — stable log(sum(exp(arr_i)))
//   totalLogMass(measure)           — log(sum of weights), 0 for null-uniform
//   effectiveSampleSize(measure)    — Kish's ESS, N for null-uniform
//   materialiseUniform(measure)     — replace null logWeights with an
//                                     explicit -log(N) array (for ops
//                                     that need an explicit array to
//                                     accumulate into)
//   systematicResample(logWeights, n, prng)
//                                   — preferred resampler; one prng()
//                                     call, low variance
//   multinomialResample(logWeights, n, prng)
//                                   — independent draws; n prng()
//                                     calls; higher variance, kept
//                                     for parity / debugging
//
// Why a separate module from histogram.js? Different concern (weights
// vs. binning) and different consumers (worker-side superpose /
// weighted, plus future PSIS-style diagnostics). Keeping them apart
// also keeps the diff small when histograms grow weighted variants
// in step 3.
//
// Why no RNG dep? The resamplers take a `prng: () => number`
// callback returning U(0,1) values. The caller (worker.js for now)
// wires up Philox via samplerLib.makePhiloxPrngAdapter; main-thread
// callers can pass Math.random for non-deterministic flows. Keeping
// empirical.js stdlib- and rng-free means it compiles into the main
// engine bundle without dragging stdlib in.

/**
 * Numerically-stable log(sum(exp(arr_i))). Returns:
 *   -Infinity when the array is empty or every element is -Infinity
 *   max(arr)  + (small) for finite arrays
 *
 * Invariant: never multiplies through by exp(x) where x might overflow;
 * we factor out the max first.
 *
 * @param {Float64Array | number[]} arr
 * @returns {number}
 */
function logSumExp(arr) {
  const n = arr.length;
  if (n === 0) return -Infinity;
  let max = arr[0];
  for (let i = 1; i < n; i++) if (arr[i] > max) max = arr[i];
  // All-equal-Infinity special cases. -Infinity → -Infinity (sum is
  // 0); +Infinity → +Infinity (sum diverges). NaN propagates.
  if (!Number.isFinite(max)) return max;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.exp(arr[i] - max);
  return max + Math.log(s);
}

/**
 * Total mass of an empirical measure in log-space.
 *
 *   logSumExp(logWeights)   — when explicit weights present
 *   0                        — when logWeights is null (uniform 1/N
 *                              sums to 1, log = 0)
 *
 * Variates and i.i.d. draws (uniform-weight measures) thus always
 * have totalLogMass = 0 (probability measure).
 *
 * @param {{ samples: Float64Array, logWeights: Float64Array | null }} measure
 * @returns {number}
 */
function totalLogMass(measure) {
  if (!measure || !measure.logWeights) return 0;
  return logSumExp(measure.logWeights);
}

/**
 * Kish's effective sample size:
 *
 *    ESS = (sum w_i)^2 / sum w_i^2
 *
 * Computed in log-space:
 *
 *    log_ESS = 2 * logSumExp(logWeights) - logSumExp(2 * logWeights)
 *
 * For uniform-weight measures (logWeights = null), all atoms
 * contribute equally so ESS = N. For a degenerate measure with all
 * mass on one atom, ESS = 1. ESS / N is a good "weight quality"
 * readout — values close to 1 mean weights are nearly uniform; values
 * close to 1/N mean a single atom dominates.
 *
 * @param {{ samples: Float64Array, logWeights: Float64Array | null }} measure
 * @returns {number}
 */
function effectiveSampleSize(measure) {
  const w = measure && measure.logWeights;
  if (!w) return measure.samples.length;
  const N = w.length;
  if (N === 0) return 0;
  const a = logSumExp(w);
  // Square the weights in log-space (multiply by 2). We allocate a
  // small temp array because Float64Array doesn't have a vectorised
  // map; for N ~ 1e5 this is well under a millisecond.
  const tw = new Float64Array(N);
  for (let i = 0; i < N; i++) tw[i] = 2 * w[i];
  const b = logSumExp(tw);
  return Math.exp(2 * a - b);
}

/**
 * Replace a measure's null (implicit-uniform) logWeights with an
 * explicit Float64Array of -log(N). Used by ops that need to mutate
 * weights pointwise (weighted, bayesupdate). The samples array is
 * shared by reference — only the weights branch.
 *
 * Pass-through for measures that already have explicit weights.
 *
 * @param {{ samples: Float64Array, logWeights: Float64Array | null }} measure
 * @returns {{ samples: Float64Array, logWeights: Float64Array }}
 */
function materialiseUniform(measure) {
  if (measure.logWeights) return measure;
  const N = measure.samples.length;
  const w = new Float64Array(N);
  const c = N > 0 ? -Math.log(N) : 0;
  for (let i = 0; i < N; i++) w[i] = c;
  return { samples: measure.samples, logWeights: w };
}

/**
 * Systematic resampling. The standard particle-filter trick for
 * turning a weighted empirical measure with `N` atoms into a uniformly-
 * weighted empirical measure with `n` atoms, in distribution.
 *
 * Algorithm: sort cumulative normalised weights into [0, 1]; draw one
 * uniform offset u0 ∈ [0, 1/n); take the n equally-spaced positions
 * u0, u0 + 1/n, …, u0 + (n-1)/n; output the index whose cumulative-
 * weight bucket each position falls into.
 *
 * Lower variance per resample step than multinomial (picks N atoms
 * with no clustering on a single one) — preferred default in
 * particle filters and other importance-resampling pipelines.
 *
 * @param {Float64Array} logWeights  source measure's per-atom log-weights
 * @param {number} n                 desired output size
 * @param {() => number} prng        returns a uniform in [0, 1); called
 *                                   exactly once
 * @returns {Int32Array} length-n array of source indices in [0, N)
 */
function systematicResample(logWeights, n, prng) {
  const N = logWeights.length;
  if (N === 0) throw new Error('systematicResample: source measure has no atoms');
  if (n <= 0) throw new Error(`systematicResample: n must be > 0 (got ${n})`);

  // Cumulative normalised weights via stable logsumexp. The final
  // cumulative entry is pinned to 1.0 to absorb floating-point
  // round-off; without that, a position close to 1.0 might fall
  // past the last entry and trip the j < N-1 guard incorrectly.
  const lse = logSumExp(logWeights);
  const cum = new Float64Array(N);
  let acc = 0;
  for (let i = 0; i < N; i++) {
    acc += Math.exp(logWeights[i] - lse);
    cum[i] = acc;
  }
  cum[N - 1] = 1.0;

  // Single uniform offset. The N positions u0 + i/n step through the
  // [0, 1) interval at uniform spacing 1/n; the offset randomises
  // the phase. With u0 = 0 we'd always pick the same indices for the
  // same weights; with the offset we get one fresh draw per resample.
  const u0 = prng() / n;
  const indices = new Int32Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const u = u0 + i / n;
    while (j < N - 1 && cum[j] < u) j++;
    indices[i] = j;
  }
  return indices;
}

/**
 * Multinomial resampling. Each output index is an independent draw
 * from the weighted distribution; n calls to prng().
 *
 * Higher variance than systematic (Monte-Carlo noise stacks across
 * draws; can produce repeated/missing atoms more often), so prefer
 * `systematicResample` when applicable. Kept here for parity and as
 * a debugging baseline.
 *
 * @param {Float64Array} logWeights
 * @param {number} n
 * @param {() => number} prng        called n times
 * @returns {Int32Array}
 */
function multinomialResample(logWeights, n, prng) {
  const N = logWeights.length;
  if (N === 0) throw new Error('multinomialResample: source measure has no atoms');
  if (n <= 0) throw new Error(`multinomialResample: n must be > 0 (got ${n})`);

  // Same cumulative-weights setup as systematic.
  const lse = logSumExp(logWeights);
  const cum = new Float64Array(N);
  let acc = 0;
  for (let i = 0; i < N; i++) {
    acc += Math.exp(logWeights[i] - lse);
    cum[i] = acc;
  }
  cum[N - 1] = 1.0;

  // Binary-search each draw into the cumulative bucket. O(n log N)
  // total. We're not assuming the prng outputs are sorted (that
  // assumption is what makes some other resamplers O(n + N)).
  const indices = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const u = prng();
    let lo = 0, hi = N - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cum[mid] < u) lo = mid + 1;
      else              hi = mid;
    }
    indices[i] = lo;
  }
  return indices;
}

// =====================================================================
// Multivariate sample shapes (struct-of-arrays)
// =====================================================================
//
// EmpiricalMeasure generalises to a recursive shape:
//
//   { shape: 'scalar', samples, logWeights }                              (current default)
//   { shape: 'record', fields: { <name>: EmpiricalMeasure }, logWeights }
//   { shape: 'tuple',  elems:  [ EmpiricalMeasure, … ],     logWeights }
//   { shape: 'array',  samples, dims, logWeights }    // flat N*prod(dims)
//
// Top-level `logWeights` lives at the root only — one weight per atom,
// shared across all fields/elements (joint sampling produces atoms,
// not per-field draws). Sub-level EmpiricalMeasures carry their own
// `samples` / `fields` / `elems` but no separate weights — querying
// any leaf's mass goes through the root.
//
// Why SoA (not array-of-records): each field is its own contiguous
// Float64Array, indexed by atom. Marginals (just take a column),
// pair plots (two columns side-by-side), Arrow serialisation
// (column → arrow vector), and `record(t)` ↔ `table(r)` auto-
// conversion (literally the same shape) all fall out for free.
//
// Back-compat: an EmpiricalMeasure without an explicit `shape` field
// is treated as `'scalar'` — matches the pre-multivariate format.
//
// Constructors below produce well-formed multivariate measures.
// They're thin builders: callers populate the fields/elems with
// already-materialised sub-measures.

/** Build a record-shaped measure. `fields` is `{name: subMeasure}`. */
function recordMeasure(fields, logWeights) {
  return { shape: 'record', fields, logWeights: logWeights || null };
}

/** Build a tuple-shaped measure. `elems` is `[subMeasure, ...]`. */
function tupleMeasure(elems, logWeights) {
  return { shape: 'tuple', elems, logWeights: logWeights || null };
}

/**
 * Build an array-shaped measure. Samples are flat (atom-major):
 * `samples[i*stride + j]` is atom i's j-th element. `dims` records
 * the per-atom shape (e.g. `[10]` for `iid(M, 10)`).
 */
function arrayMeasure(samples, dims, logWeights) {
  return { shape: 'array', samples, dims, logWeights: logWeights || null };
}

/** Effective shape — back-compat shim. Untagged measures are scalar. */
function shapeOf(measure) {
  return (measure && measure.shape) || 'scalar';
}

module.exports = {
  logSumExp,
  totalLogMass,
  effectiveSampleSize,
  materialiseUniform,
  systematicResample,
  multinomialResample,
  // Multivariate
  recordMeasure, tupleMeasure, arrayMeasure, shapeOf,
};
