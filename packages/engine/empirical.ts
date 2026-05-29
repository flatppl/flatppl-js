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
function logSumExp(arr: ArrayLike<number>) {
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
function totalLogMass(measure: any) {
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
function effectiveSampleSize(measure: any) {
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
function materialiseUniform(measure: any) {
  if (measure.logWeights) return measure;
  const N = measure.samples.length;
  const w = new Float64Array(N);
  const c = N > 0 ? -Math.log(N) : 0;
  for (let i = 0; i < N; i++) w[i] = c;
  return { samples: measure.samples, logWeights: w };
}

/**
 * Combine the per-atom log-weights of several parent measures into the
 * log-weights of a derived (atom-aligned) measure. The atom-i value of
 * the derived measure is a deterministic function of the atom-i values
 * of its parents, so the IS weight at atom i of the result is the joint
 * IS weight of (parent_1[i], parent_2[i], …) under the per-atom-pair
 * proposal — i.e. the product in linear space, sum in log space — over
 * the INDEPENDENT weighting events the parents bring in.
 *
 * Independence is detected by reference identity of the `logWeights`
 * arrays: two parents whose `logWeights` point at the same Float64Array
 * trace the same weighting event (the typical case when one parent is
 * a deterministic transform of another), so we count those weights
 * once. Distinct array references describe independent events whose IS
 * weights multiply. This contract requires the engine to NEVER clone
 * `logWeights` along an inheritance chain (evaluate-kind, alias, iid,
 * record projection, …): always pass the same reference forward;
 * allocate fresh only when introducing a new weighting event (a draw
 * from a non-trivial measure, weighted / logweighted, bayesupdate).
 *
 * Returns:
 *   - null when every parent has null logWeights (uniform).
 *   - the shared reference when exactly one independent weight stream
 *     is present (preserves reference identity so downstream dedupe
 *     keeps working).
 *   - a fresh Float64Array equal to the per-atom sum when two or more
 *     independent streams are combined.
 *
 * @param {Iterable<{logWeights: Float64Array | null}>} parents
 * @returns {Float64Array | null}
 */
function propagateLogWeights(parents: Iterable<any>) {
  const seen = new Set();
  const unique: Float64Array[] = [];
  for (const p of parents) {
    if (!p || !p.logWeights) continue;
    if (seen.has(p.logWeights)) continue;
    seen.add(p.logWeights);
    unique.push(p.logWeights);
  }
  if (unique.length === 0) return null;
  if (unique.length === 1) return unique[0];
  const N = unique[0].length;
  for (let k = 1; k < unique.length; k++) {
    if (unique[k].length !== N) {
      throw new Error('propagateLogWeights: parent logWeights lengths disagree — '
        + 'every parent must share the engine\'s N-atom axis');
    }
  }
  const out = new Float64Array(N);
  for (const w of unique) {
    for (let i = 0; i < N; i++) out[i] += w[i];
  }
  return out;
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
function systematicResample(logWeights: ArrayLike<number>, n: number, prng: () => number) {
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
function multinomialResample(logWeights: ArrayLike<number>, n: number, prng: () => number) {
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
// **Composite-measure invariant (LOAD-BEARING).** When the outer
// measure is composite (record / tuple) and carries `logWeights`,
// sub-field / sub-element measures MUST carry null logWeights and no
// `n_eff` / `logTotalmass`. The importance-weighting channel lives
// at the OUTER level alone; sub-leaves are SoA storage of per-atom
// samples. The `recordMeasure` / `tupleMeasure` constructors enforce
// this by shallow-copy-and-strip of any sub-measure metadata. They
// recurse so the invariant holds at every nested layer.
//
// **Why.** Per spec §03 records are atoms-of-records (SoA storage),
// not records-of-atoms. The importance-weighting channel is one
// per atom — shared across all fields. Storing per-field
// `logWeights` would either duplicate the outer's (wasted memory +
// drift risk if one path forgets to update both) or carry stale
// upstream weights from the field's binding cache (the original
// rasch-two-parameter symptom: `posterior.fields.diff_sd.logWeights`
// kept the half-Cauchy normalize weights from the prior, missing
// the likelihood update). The cached binding `diff_sd_binding`
// keeps its own weights — `recordMeasure` shallow-copies before
// stripping, so binding-cache integrity is preserved.
//
// **What consumers do.** Reading IS-weighted statistics on a field
// (histogram, ESS, k̂) always reads logWeights from the OUTER
// measure. The viewer's `renderDensityStrips` and `renderCornerGrid`
// already do this; future consumers must follow the same rule.
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

/**
 * Strip per-leaf importance-weighting metadata so the composite-
 * measure invariant holds: logWeights / n_eff / logTotalmass live
 * ONLY at the outer level. Sub-fields and tuple elements are SoA
 * storage of per-atom samples; consumers compute IS-weighted leaf
 * statistics by reading the outer measure's logWeights.
 *
 * Shallow-copies any sub-measure that carries weighting metadata
 * (referent of `fields[k]` / `elems[i]` may be a cached binding
 * measure shared with the orchestrator's per-binding empirical-
 * measure cache — mutating it would corrupt the cache).
 *
 * Nested composites recurse so the invariant holds at every layer:
 * `posterior.fields.X` is clean AND `posterior.fields.X.fields.Y`
 * is clean.
 */
function _stripLeafWeights(m: any): any {
  if (!m || typeof m !== 'object') return m;
  // A leaf carries weighting metadata if any of these are present.
  // Drop them via shallow copy.
  const hasWeights = m.logWeights != null
                  || typeof m.logTotalmass === 'number'
                  || typeof m.n_eff === 'number';
  // Always recurse into nested composites; mutate the copy when we
  // need to.
  let copy: any = null;
  if (m.fields) {
    let changed = false;
    const f2: any = {};
    for (const k in m.fields) {
      const stripped = _stripLeafWeights(m.fields[k]);
      f2[k] = stripped;
      if (stripped !== m.fields[k]) changed = true;
    }
    if (changed) {
      copy = copy || Object.assign({}, m);
      copy.fields = f2;
    }
  }
  if (Array.isArray(m.elems)) {
    let changed = false;
    const e2 = new Array(m.elems.length);
    for (let i = 0; i < m.elems.length; i++) {
      const stripped = _stripLeafWeights(m.elems[i]);
      e2[i] = stripped;
      if (stripped !== m.elems[i]) changed = true;
    }
    if (changed) {
      copy = copy || Object.assign({}, m);
      copy.elems = e2;
    }
  }
  if (hasWeights) {
    copy = copy || Object.assign({}, m);
    copy.logWeights = null;
    // Drop derived metadata that would no longer reflect the
    // outer-level weighting once the composite carries the real
    // outer logWeights. `value` (the Float64Array view) stays —
    // sample storage is shared with the binding cache and not
    // semantically tied to the weighting channel.
    delete copy.logTotalmass;
    delete copy.n_eff;
  }
  return copy || m;
}

/** Build a record-shaped measure. `fields` is `{name: subMeasure}`. */
function recordMeasure(fields: any, logWeights: any) {
  const cleanFields: any = {};
  for (const k in fields) cleanFields[k] = _stripLeafWeights(fields[k]);
  return { shape: 'record', fields: cleanFields, logWeights: logWeights || null };
}

/** Build a tuple-shaped measure. `elems` is `[subMeasure, ...]`. */
function tupleMeasure(elems: any, logWeights: any) {
  const cleanElems = new Array(elems.length);
  for (let i = 0; i < elems.length; i++) cleanElems[i] = _stripLeafWeights(elems[i]);
  return { shape: 'tuple', elems: cleanElems, logWeights: logWeights || null };
}

/**
 * Build an array-shaped measure. Samples are flat (atom-major):
 * `samples[i*stride + j]` is atom i's j-th element. `dims` records
 * the per-atom shape (e.g. `[10]` for `iid(M, 10)`).
 *
 * Also populates `.value` (the shape-tagged Value view) so downstream
 * consumers using the valueOf/measureFromValue contract see a coherent
 * shape. `.value.data` shares storage with `.samples` — no allocation
 * overhead.
 */
function arrayMeasure(samples: any, dims: any, logWeights: any) {
  const m: any = { shape: 'array', samples, dims, logWeights: logWeights || null };
  if (samples && dims && dims.length > 0
      && samples.BYTES_PER_ELEMENT !== undefined) {
    const total = dims.reduce(function (p: number, n: number) { return p * n; }, 1);
    const N = total > 0 ? (samples.length / total) : 0;
    m.value = { shape: [N | 0].concat(dims), data: samples };
  }
  return m;
}

/** Effective shape — back-compat shim. Untagged measures are scalar. */
function shapeOf(measure: any) {
  return (measure && measure.shape) || 'scalar';
}

// =====================================================================
// Importance-sampling quality diagnostic
// =====================================================================
//
// Self-contained quality classifier for an empirical measure with
// optional importance log-weights. Returns:
//
//   { label, ess, ratio, kHat, wmax, dof, N }
//
// where `label` ∈ { 'good', 'ok', 'bad', 'unusable' }. The viewer
// colours the readout span by this label.
//
// Methodology (synthesised from Vehtari et al.'s PSIS recommendations
// and the loo-package conventions):
//   - PSIS k̂ as primary diagnostic — Pareto-tail shape of the upper
//     importance weights. k̂ ≤ 0.5 means finite variance; 0.5 < k̂ ≤ 0.7
//     usable but variance high; 0.7 < k̂ ≤ 1 biased estimates,
//     warning; k̂ > 1 infinite mean, untrustworthy.
//   - Kish ESS / N as secondary — bulk weight uniformity.
//   - max-weight share as a single-particle-dominance check.
//   - dof (effective dimensionality, computed by callers) scales the
//     absolute ESS floor: max(absolute_floor, k·D) — only kicks in
//     for high-D measures (low-D and 1-D scalar measures get the
//     unscaled floor).
//
// Worst-trigger combining: any one diagnostic crossing a worse band
// downgrades the label to that band.

/**
 * Fit a generalised Pareto distribution to the upper tail of an
 * empirical sample, returning the shape parameter k̂ (Vehtari sign
 * convention: k̂ > 0 ⇒ heavy right tail).
 *
 * Uses Zhang & Stephens' (2009) empirical-Bayes posterior-mean
 * estimator — the algorithm the loo R package uses for PSIS k̂
 * diagnostics. Tested against Pareto-tailed synthetic samples to
 * within ~0.05 of the canonical implementation in the relevant
 * range (k̂ ∈ [0.2, 1.5]).
 *
 * @param {Float64Array | number[]} exceedances  upper-tail values
 *        (already shifted by threshold, all > 0). Need not be sorted.
 * @returns {number} k̂; NaN if fit cannot be computed (degenerate input).
 */
function gpdShapeZhangStephens(exceedances: ArrayLike<number>) {
  const n = exceedances.length;
  if (n < 2) return NaN;
  const x = Float64Array.from(exceedances);
  x.sort();
  // Max must be positive and finite for the GPD fit; the smallest
  // exceedance is allowed to be zero (ties at the threshold boundary
  // are common when samples cluster) — log(1 - b·0) = 0 contributes
  // nothing to the profile log-likelihood.
  if (x[n - 1] <= 0 || !Number.isFinite(x[n - 1])) return NaN;
  const m = 30 + Math.floor(Math.sqrt(n));
  // Quantile-mixture grid for b: locations spanning (close to 0, close
  // to 1/x_max). The /3/x_q25 term is the Zhang-Stephens recipe;
  // x_q25 is the 25th-percentile order statistic.
  const xMax = x[n - 1];
  const x25 = x[Math.floor(n / 4 + 0.5)] || x[0];
  const bs = new Float64Array(m);
  for (let j = 1; j <= m; j++) {
    const t = 1 - Math.sqrt(m / (j - 0.5));
    bs[j - 1] = 1 / xMax + t / (3 * x25);
  }
  // Profile log-likelihood at each b. Zhang-Stephens parametrisation:
  //   k(b) = mean(log(1 - b·x_i))
  //   L(b) = n · (log(-b/k(b)) − k(b) − 1)
  // (Same as the loo R package's gpdfit.R.) For heavy-tailed data,
  // the fit prefers b < 0 (so 1 − b·x > 1 and k(b) > 0, matching
  // Vehtari's "k > 0 ⇒ heavy tail" convention). For light-tailed
  // data the optimum b > 0; both branches give -b/k(b) > 0 when
  // sign(b) ≠ sign(k(b)), so log is defined.
  const ll = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    const b = bs[j];
    let kSum = 0;
    for (let i = 0; i < n; i++) {
      const v = 1 - b * x[i];
      if (v <= 0) { kSum = NaN; break; }
      kSum += Math.log(v);
    }
    if (!Number.isFinite(kSum)) { ll[j] = -Infinity; continue; }
    const kj = kSum / n;
    if (kj === 0 || -b / kj <= 0) { ll[j] = -Infinity; continue; }
    ll[j] = n * (Math.log(-b / kj) - kj - 1);
  }
  // Convert log-likelihoods to posterior weights via softmax, then
  // take the posterior mean of b.
  let llMax = -Infinity;
  for (let j = 0; j < m; j++) if (ll[j] > llMax) llMax = ll[j];
  if (!Number.isFinite(llMax)) return NaN;
  let wSum = 0;
  const w = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    w[j] = Math.exp(ll[j] - llMax);
    wSum += w[j];
  }
  if (wSum === 0) return NaN;
  let bBar = 0;
  for (let j = 0; j < m; j++) bBar += (w[j] / wSum) * bs[j];
  // Recover k̂ from b̄ via k(b) = mean(log(1 − b·x_i)) — the same
  // form used inside the profile log-lik above. Loo's gpdfit.R
  // returns the same expression. The sign convention is Vehtari's
  // (positive k̂ ⇒ heavy tail).
  let kSum = 0;
  for (let i = 0; i < n; i++) {
    const v = 1 - bBar * x[i];
    if (v <= 0) return NaN;
    kSum += Math.log(v);
  }
  return kSum / n;
}

/**
 * Compute PSIS-style k̂ for a sample of (positive) importance weights.
 * Selects the upper tail size M = min(N/5, 3·√N) per Vehtari's recipe,
 * then fits a generalised Pareto to the tail via Zhang-Stephens.
 *
 * @param {Float64Array | number[]} weights — non-negative, finite.
 *        Need not be normalised; absolute scale doesn't affect k̂.
 * @returns {number} k̂; NaN on degenerate input or failed fit.
 */
function paretoKHat(weights: ArrayLike<number>) {
  const N = weights.length;
  if (N < 5) return NaN;
  const M = Math.max(5, Math.min(Math.floor(N / 5), Math.floor(3 * Math.sqrt(N))));
  if (M < 2 || M >= N) return NaN;
  const sorted = Float64Array.from(weights);
  sorted.sort();
  // Threshold = (N-M-1)-th order statistic. Take exceedances of
  // top-M weights minus that threshold.
  const threshold = sorted[N - M - 1];
  const tail = new Float64Array(M);
  for (let i = 0; i < M; i++) tail[i] = sorted[N - M + i] - threshold;
  return gpdShapeZhangStephens(tail);
}

/**
 * Sample-size-aware k̂ reliability threshold (Vehtari, Simpson, Yao 2024):
 *   k★ = min(0.7, 1 - 1/log10(N))
 * For N ≥ 10⁴, k★ ≈ 0.7 (the canonical fixed bound).
 * For smaller N, k★ tightens — at N=100, k★ = 0.5.
 */
function paretoKThreshold(N: number) {
  if (N <= 10) return -Infinity;
  return Math.min(0.7, 1 - 1 / Math.log10(N));
}

/**
 * Assess the importance-sampling quality of a (possibly weighted)
 * empirical measure. Returns a structured diagnostic the viewer's
 * sample-stats readout colour-codes.
 *
 * Combining rule: worst trigger across {k̂, ESS, max-weight, sample
 * size}. Effective dimensionality `dof` (computed by the caller from
 * the measure's structure — record fields, array slots, varying-
 * sample axes) scales the absolute ESS floor; pass 1 for a scalar
 * measure or when D is unknown.
 *
 * @param {{samples, logWeights, …}} measure — empirical measure.
 * @param {number} dof — degrees of freedom (≥ 1).
 * @returns {{ label, ess, ratio, kHat, wmax, dof, N }}
 */
function importanceSamplingQuality(measure: any, dof: number) {
  const D = Math.max(1, dof | 0);
  const N = measureAtomCount(measure);
  if (N === 0) {
    return { label: 'unusable', ess: 0, ratio: 0, kHat: NaN,
             wmax: 1, dof: D, N: 0 };
  }
  // ESS only matters when weights are present — and effectiveSampleSize
  // (designed for scalar measures) reads measure.samples.length, which
  // doesn't exist on record/tuple shapes. Short-circuit unweighted
  // measures to ESS=N rather than calling through.
  const ess = measure.logWeights
    ? effectiveSampleSize(measure)
    : N;
  const ratio = N > 0 ? ess / N : 0;

  // Unweighted measure: weights are uniform by construction, k̂ is
  // not meaningful (and the GPD fit on a constant-weight tail is
  // degenerate). Always 'good' for an unweighted measure of any
  // reasonable size. kHat NaN signals "uniform / not applicable" to
  // the viewer; downstream readout drops the diagnostic span entirely
  // for such measures.
  if (!measure.logWeights) {
    if (N < 20) {
      return { label: 'unusable', ess, ratio: 1, kHat: NaN, wmax: 1 / N, dof: D, N };
    }
    return { label: 'good', ess, ratio: 1, kHat: NaN, wmax: 1 / N, dof: D, N };
  }

  // Walk logWeights once: max for normalisation, min/max range for
  // uniform-detection, non-finite check.
  const logW = measure.logWeights;
  let lwMin = Infinity, lwMax = -Infinity;
  let nonFinite = false;
  for (let i = 0; i < N; i++) {
    const lw = logW[i];
    if (!Number.isFinite(lw) && lw !== -Infinity) { nonFinite = true; break; }
    if (lw < lwMin) lwMin = lw;
    if (lw > lwMax) lwMax = lw;
  }
  if (nonFinite) {
    return { label: 'unusable', ess, ratio, kHat: NaN, wmax: 1, dof: D, N };
  }
  // Uniform-within-epsilon: even with explicit logWeights, if every
  // entry equals every other (within float tolerance), the measure
  // is effectively unweighted. Skip the k̂ fit — there's no tail to
  // characterise — and report identically to the no-logWeights case.
  // Common when materialiseUniform set logW = -log(N) preemptively
  // before any reweighting op was actually applied.
  if (lwMax - lwMin < 1e-12) {
    if (N < 20) {
      return { label: 'unusable', ess, ratio: 1, kHat: NaN, wmax: 1 / N, dof: D, N };
    }
    return { label: 'good', ess, ratio: 1, kHat: NaN, wmax: 1 / N, dof: D, N };
  }
  // Normalise weights for max-weight check. Keep work in log-space
  // to avoid underflow in the tail.
  const lse = logSumExp(logW);
  let wmax = 0;
  for (let i = 0; i < N; i++) {
    const w = Math.exp(logW[i] - lse);
    if (w > wmax) wmax = w;
  }

  // PSIS k̂: linearise weights then fit. We pass exp(logW - lse) so
  // weights are normalised to sum to 1 (k̂ is scale-invariant but
  // numerically the bounded range avoids overflow).
  const w = new Float64Array(N);
  for (let i = 0; i < N; i++) w[i] = Math.exp(logW[i] - lse);
  const kHat = paretoKHat(w);

  // Sample-size-aware k̂ threshold.
  const kStar = paretoKThreshold(N);

  // Worst-trigger classification. Order: 0=good, 1=ok, 2=bad, 3=unusable.
  let severity = 0;
  // k̂ band
  if (!Number.isFinite(kHat)) severity = Math.max(severity, 3);
  else if (kHat > 1.0)         severity = Math.max(severity, 3);
  else if (kHat > kStar)       severity = Math.max(severity, 2);
  else if (kHat > 0.5)         severity = Math.max(severity, 1);
  // ESS band — absolute floor scaled by dof. The ratio thresholds
  // are dimension-invariant.
  if (ess < Math.max(20, 2 * D))   severity = Math.max(severity, 3);
  else if (ess < Math.max(50, 5 * D))  severity = Math.max(severity, 2);
  else if (ess < Math.max(100, 10 * D)) severity = Math.max(severity, 1);
  if (ratio < 0.02)  severity = Math.max(severity, 2);
  else if (ratio < 0.1) severity = Math.max(severity, 1);
  // Single-particle dominance.
  if (wmax > 0.5)       severity = Math.max(severity, 3);
  else if (wmax > 0.15) severity = Math.max(severity, 2);
  else if (wmax > 0.05) severity = Math.max(severity, 1);
  // Small-N cap: PSIS k̂ is itself noisy; never call green at N < 100.
  if (N < 100 && severity < 1) severity = 1;
  // Tighter green: k̂ must clear (kStar − 0.1) AND be ≤ 0.5.
  if (severity === 0 && kHat > Math.min(0.5, kStar - 0.1)) severity = 1;

  const label = ['good', 'ok', 'bad', 'unusable'][severity];
  return { label, ess, ratio, kHat, wmax, dof: D, N };
}

// Internal: walk a measure to find any sub-measure's atom count.
function measureAtomCount(measure: any): number {
  if (!measure) return 0;
  if (measure.fields) {
    const ks = Object.keys(measure.fields);
    return ks.length > 0 ? measureAtomCount(measure.fields[ks[0]]) : 0;
  }
  if (Array.isArray(measure.elems) && measure.elems.length > 0) {
    return measureAtomCount(measure.elems[0]);
  }
  if (measure.samples) {
    if (measure.dims && measure.dims.length > 0) {
      const stride = measure.dims.reduce((p: number, n: number) => p * n, 1);
      return stride > 0 ? measure.samples.length / stride : 0;
    }
    return measure.samples.length;
  }
  return 0;
}

/**
 * Heuristic DOF estimate for an empirical measure: the count of
 * scalar leaves whose per-atom samples vary. Slots with constant
 * samples (degenerate atoms — e.g. a Dirac field inside a record)
 * contribute 0. iid(M, n) contributes n (atoms are independent
 * across the iid axis by construction).
 *
 * "Estimate" in the name because the count is a structural one and
 * doesn't account for distribution-level constraints between scalar
 * leaves. A Dirichlet(α, n) has n-1 effective DOF (the simplex
 * sum-to-one constraint), but this function counts n. For our
 * purposes — scaling the ESS floors in importanceSamplingQuality —
 * a small overcount is conservative (asks for slightly more
 * effective samples than strictly necessary) and the simpler walk
 * keeps the helper local.
 *
 * Exact propagation is possible for many constructors: Normal = 1,
 * MvNormal(n) = n, Dirichlet(α, n) = n-1, joint = Σ, iid(M, n) =
 * n·dof(M), pushfwd by injective f preserves, lawof of value-typed
 * e = 0, etc. Threading that metadata through every measure-algebra
 * op is more complex than this estimate justifies for the present
 * use case (ESS-floor scaling in the viewer's sample-stats readout).
 *
 * Even an "exact" DOF is itself an approximation in the general
 * case: rank can vary across the measure's support — e.g. a
 * mixture of components with different intrinsic dimensionalities,
 * or conditional structures where one component has rank-1 and
 * another rank-n. A single global D is a first-order proxy, not
 * a measure-theoretic invariant. For a global threshold-scaling
 * heuristic that's fine; refine the model if a downstream consumer
 * needs per-region rank.
 *
 * Used as the `dof` argument to importanceSamplingQuality; its
 * absolute-ESS floors scale by D so a higher-dimensional measure
 * needs proportionally more effective samples to clear the green/
 * yellow bands.
 */
function estimateDof(measure: any): number {
  if (!measure) return 1;
  if (measure.fields) {
    let d = 0;
    for (const k in measure.fields) d += estimateDof(measure.fields[k]);
    return Math.max(1, d);
  }
  if (Array.isArray(measure.elems)) {
    let d = 0;
    for (const e of measure.elems) d += estimateDof(e);
    return Math.max(1, d);
  }
  if (measure.samples) {
    if (measure.dims && measure.dims.length > 0) {
      const stride = measure.dims.reduce((p: number, n: number) => p * n, 1);
      const N = stride > 0 ? measure.samples.length / stride : 0;
      // Count per-slot variance. A slot with all-equal values across
      // atoms contributes 0 DOF (degenerate / Dirac-like).
      let d = 0;
      for (let s = 0; s < stride; s++) {
        if (slotVaries(measure.samples, N, stride, s)) d++;
      }
      return Math.max(1, d);
    }
    return slotVaries(measure.samples, measure.samples.length, 1, 0) ? 1 : 0;
  }
  return 1;
}

function slotVaries(samples: ArrayLike<number>, N: number, stride: number, s: number) {
  if (N < 2) return false;
  const v0 = samples[s];
  for (let i = 1; i < N; i++) {
    if (samples[i * stride + s] !== v0) return true;
  }
  return false;
}

module.exports = {
  logSumExp,
  totalLogMass,
  effectiveSampleSize,
  materialiseUniform,
  propagateLogWeights,
  systematicResample,
  multinomialResample,
  // Multivariate
  recordMeasure, tupleMeasure, arrayMeasure, shapeOf,
  // Importance-sampling diagnostics
  paretoKHat, paretoKThreshold,
  importanceSamplingQuality, estimateDof,
};
