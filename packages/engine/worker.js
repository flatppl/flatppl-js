'use strict';

// FlatPPL sampler-worker message handler.
//
// This module is the *transport-agnostic* heart of the sampling worker. It
// owns a small piece of mutable state (RNG, env) and exposes a `handle(msg)`
// function that takes a request object and returns a reply object. Wiring
// it up to a Web Worker, Node worker_threads, or an in-process call is the
// caller's job — see `worker-entry.js` for the browser/Node shim.
//
// Why factor it this way?
//   * Tests can drive the handler with plain object literals — no Worker
//     constructor, no postMessage round-trips, no SharedArrayBuffer setup.
//   * The same logic can run synchronously on the main thread when a
//     worker isn't available (e.g. the UI's first render before the worker
//     bundle has finished loading).
//   * Future Rust port can implement the same protocol without inheriting
//     any JS-specific transport details.
//
// State held by a handler instance:
//   - philox  : pure-functional RNG state (rng.js); threaded through every
//               draw so the worker is fully reproducible from its seed.
//   - env     : { name → number } map of values bound on the main thread
//               (parents already sampled, fixed inputs, etc.). The IR
//               coming in has free identifiers that look up here.
//
// Message protocol (request `type` → reply `type`):
//   init     { seed?, env? }              → ready
//   setEnv   { env, merge? }              → ok
//   setSeed  { seed }                     → ok
//   sample   { ir, env?, count }          → samples  { samples: Float64Array }
//   density  { ir, env?, opts? }          → density  { xs, ys, support, reference }
//   evaluate { ir, env? }                 → value    { value }
//   dispose  {}                           → (no reply; worker terminates)
//
// Every request may carry an `id` field; the reply echoes it. Errors come
// back as `{ type: 'error', id, message, stack? }`. Sample arrays are
// returned as Float64Array so the entry shim can transferList them.

const rngLib = require('./rng');
const samplerLib = require('./sampler');

function createWorkerHandler(opts = {}) {
  // Default seed: 0n. Callers should always send `init { seed }` before the
  // first `sample`. Using 0 as the "uninitialized" seed makes accidental
  // missed-init runs deterministic instead of silently random.
  let philox = rngLib.stateFromKey(opts.seed ?? 0);
  let env = { ...(opts.env ?? {}) };

  function handle(msg) {
    const id = msg.id;
    try {
      switch (msg.type) {
        case 'init': {
          // Reset both RNG and env. `init` is the canonical "fresh start"
          // so it always replaces both — `setEnv` and `setSeed` exist for
          // partial updates without a full reset.
          philox = rngLib.stateFromKey(msg.seed ?? 0);
          env = { ...(msg.env ?? {}) };
          return { type: 'ready', id };
        }
        case 'setEnv': {
          // merge=true keeps existing entries (default), merge=false replaces.
          env = msg.merge === false ? { ...(msg.env ?? {}) } : { ...env, ...(msg.env ?? {}) };
          return { type: 'ok', id };
        }
        case 'setSeed': {
          philox = rngLib.stateFromKey(msg.seed ?? 0);
          return { type: 'ok', id };
        }
        case 'sample': {
          // Per-request env overlay: caller can pass extra bindings without
          // clobbering the worker-level env. Worker env wins on conflict so
          // expensive fixed setups don't get accidentally overwritten by a
          // stray per-request key with the same name.
          const callEnv = msg.env ? { ...msg.env, ...env } : env;
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`sample.count must be positive integer (got ${msg.count})`);
          const out = new Float64Array(count);
          for (let i = 0; i < count; i++) {
            const [v, next] = samplerLib.rand(philox, msg.ir, callEnv);
            philox = next;
            out[i] = v;
          }
          return { type: 'samples', id, samples: out };
        }
        case 'density': {
          const callEnv = msg.env ? { ...msg.env, ...env } : env;
          const d = samplerLib.density(msg.ir, callEnv, msg.opts ?? {});
          return { type: 'density', id, ...d };
        }
        case 'evaluate': {
          const callEnv = msg.env ? { ...msg.env, ...env } : env;
          const value = samplerLib.evaluateExpr(msg.ir, callEnv);
          return { type: 'value', id, value };
        }
        case 'sampleChain': {
          // Ancestral sampling: walk an orchestrator-built chain N times,
          // threading a per-draw env through each step, and emit the
          // last step's value into the output array. The orchestrator
          // is responsible for placing steps in topological order so
          // every `ref` resolves against an earlier step's name (or the
          // shared callEnv).
          const callEnv = msg.env ? { ...msg.env, ...env } : { ...env };
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`sampleChain.count must be positive integer (got ${msg.count})`);
          const out = runChain(msg.chain, count, callEnv);
          return { type: 'samples', id, samples: out };
        }
        case 'samplesPlot': {
          // One-shot for the Plot panel: run the chain, return raw samples,
          // and attach both an empirical histogram (always) and a density
          // overlay (analytical when an analyticalIR is supplied, else KDE
          // for continuous, null for discrete since the histogram itself
          // is the empirical pmf).
          //
          // Inputs:
          //   chain         (required) orchestrator step list
          //   count         (required) sample count
          //   discrete      bool       picks histogram-vs-KDE for the empirical curve
          //   analyticalIR  optional   sampleable dist IR for an exact PDF/PMF curve
          //   opts          { gridPoints, trimQ, … } passed through to estimators
          // Output:
          //   { samples, histogram: { xs, ys, support, reference, binWidth? },
          //     density: null | { xs, ys, support, reference, method } }
          const callEnv = msg.env ? { ...msg.env, ...env } : { ...env };
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`samplesPlot.count must be positive integer (got ${msg.count})`);
          const samples = runChain(msg.chain, count, callEnv);
          const opts = msg.opts || {};
          const histogram = msg.discrete
            ? histogramDensity(samples, opts)
            : freedmanDiaconisHistogram(samples, opts);
          let density = null;
          if (msg.analyticalIR) {
            // Exact PDF/PMF via stdlib. callEnv carries any free refs the
            // analytical IR contains (the orchestrator usually only flags
            // analyticalIR when the leaf has all-literal kwargs, so env
            // shouldn't matter — but threading it costs nothing).
            density = samplerLib.density(msg.analyticalIR, callEnv, opts);
            density.method = 'analytical';
          }
          // Intentionally NO KDE fallback here. KDE without boundary
          // correction smears mass past the support of the true
          // distribution (an Exponential's KDE leaks into x<0, etc.),
          // which mis-suggests density where there is none. The
          // histogram alone is honest about the empirical distribution
          // and doesn't pretend to be smooth where the underlying
          // measure isn't. If you want a smooth overlay later, add
          // boundary-aware KDE (reflection / boundary-corrected
          // kernels / log-scale KDE for support [0,∞)) and bring it
          // back behind a different method tag.
          return { type: 'samplesPlot', id, samples, histogram, density };
        }
        case 'densityFromChain': {
          // Run the chain to draw N samples, then estimate the marginal
          // density of the leaf binding from those samples. KDE for
          // continuous (`discrete: false`), integer histogram for
          // discrete (`discrete: true`). The reply mirrors the analytical
          // `density` shape so the UI can render either uniformly.
          const callEnv = msg.env ? { ...msg.env, ...env } : { ...env };
          const count = msg.count | 0;
          if (count <= 0) throw new Error(`densityFromChain.count must be positive integer (got ${msg.count})`);
          const samples = runChain(msg.chain, count, callEnv);
          const opts = msg.opts || {};
          const d = msg.discrete
            ? histogramDensity(samples, opts)
            : kdeDensity(samples, opts);
          return { type: 'density', id, ...d, method: msg.discrete ? 'histogram' : 'kde' };
        }
        case 'dispose': {
          // Caller (entry shim) is expected to close the worker after this.
          philox = null;
          env = null;
          return null;
        }
        default:
          throw new Error(`unknown message type: ${msg.type}`);
      }
    } catch (err) {
      return {
        type: 'error',
        id,
        message: err && err.message ? err.message : String(err),
        stack: err && err.stack ? err.stack : undefined,
      };
    }
  }

  // Test-only inspector. Exposed so tests can assert on RNG advancement
  // without sampling. Don't rely on this from production code.
  function _inspect() {
    return { philox, env };
  }

  /**
   * Walk the chain `count` times, drawing or evaluating per step,
   * threading per-draw env through. Returns Float64Array(count) of
   * the last step's value per draw. Updates the closed-over `philox`
   * state so subsequent worker requests see the advanced RNG.
   */
  function runChain(chain, count, baseEnv) {
    if (!Array.isArray(chain) || chain.length === 0) {
      throw new Error('sampleChain.chain must be a non-empty array');
    }
    const out = new Float64Array(count);
    for (let i = 0; i < count; i++) {
      // Per-draw env: shallow-copy of base so each draw is independent.
      // Earlier-step names get bound here as we walk; later steps can
      // reference them via `ref` IRs in their own subexpressions.
      const drawEnv = { ...baseEnv };
      let lastValue = NaN;
      for (let j = 0; j < chain.length; j++) {
        const step = chain[j];
        if (step.kind === 'sample') {
          const [v, next] = samplerLib.rand(philox, step.ir, drawEnv);
          philox = next;
          drawEnv[step.name] = v;
          lastValue = v;
        } else if (step.kind === 'evaluate') {
          const v = samplerLib.evaluateExpr(step.ir, drawEnv);
          drawEnv[step.name] = v;
          lastValue = v;
        } else {
          throw new Error(`sampleChain: unknown step kind '${step.kind}'`);
        }
      }
      out[i] = lastValue;
    }
    return out;
  }

  return { handle, _inspect };
}

// =====================================================================
// Density estimation — used by `densityFromChain` to turn a sample
// array into a smooth curve for plotting. Pure-numeric, no stdlib.
// Lives at module scope so it can be unit-tested independently of the
// handler closure.
// =====================================================================

/**
 * Gaussian kernel density estimate on a uniform grid. Bandwidth is
 * chosen by Silverman's rule of thumb (h = 1.06 σ n^(-1/5)) by default,
 * which is reasonable for unimodal continuous distributions and at
 * worst slightly oversmooths for heavier-tailed shapes. Tests don't
 * lock in the exact KDE values, only that the curve is non-negative,
 * normalised, and concentrates around the true mode.
 *
 * Grid extends a few bandwidths past the sample [min, max] so the
 * tails decay smoothly to (near-)zero before the axis ends.
 *
 * @param {Float64Array|number[]} samples
 * @param {object} [opts]
 * @param {number} [opts.gridPoints=200]
 * @param {number} [opts.bandwidth]   override Silverman's choice
 * @returns {{ xs: Float64Array, ys: Float64Array, support: [number, number], reference: 'lebesgue' }}
 */
function kdeDensity(samples, opts = {}) {
  const n = samples.length;
  if (n === 0) {
    // Edge case: no samples → empty plot. Caller usually has count > 0
    // but this keeps us from crashing on degenerate input.
    return { xs: new Float64Array(0), ys: new Float64Array(0), support: [0, 0], reference: 'lebesgue' };
  }
  const { mean, sd } = meanSd(samples);
  let h = opts.bandwidth;
  if (!(h > 0)) {
    // Silverman: 1.06 σ n^(-1/5). Falls back to a small floor if all
    // samples coincide (sd=0) so the kernel is finite-width.
    h = sd > 0 ? 1.06 * sd * Math.pow(n, -0.2) : 1.0;
  }
  let lo = +Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = samples[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  // Pad the grid by 3h on each side so kernel tails decay before the axis.
  const padLo = lo - 3 * h;
  const padHi = hi + 3 * h;
  const points = opts.gridPoints | 0 || 200;
  const xs = new Float64Array(points);
  const ys = new Float64Array(points);
  const norm = 1 / (n * h * Math.sqrt(2 * Math.PI));
  for (let i = 0; i < points; i++) {
    const x = padLo + (padHi - padLo) * i / (points - 1);
    xs[i] = x;
    let acc = 0;
    for (let j = 0; j < n; j++) {
      const z = (x - samples[j]) / h;
      acc += Math.exp(-0.5 * z * z);
    }
    ys[i] = acc * norm;
  }
  return { xs, ys, support: [padLo, padHi], reference: 'lebesgue' };
}

/**
 * Probability mass function via integer-bin histogram. Bins are unit
 * width centred on each integer atom from min(samples) to max(samples).
 * Result is normalised to sum to 1 over the support, so it can be
 * plotted alongside analytical pmfs without further scaling.
 *
 * @param {Float64Array|number[]} samples
 * @param {object} [opts]
 * @returns {{ xs: Float64Array, ys: Float64Array, support: [number, number], reference: 'counting' }}
 */
function histogramDensity(samples, opts = {}) {
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
 * Equal-width histogram with bin width chosen by the Freedman-Diaconis
 * rule: h = 2 * IQR * n^(-1/3). Robust to outliers (uses IQR, not sd)
 * and assumes nothing about the distribution shape. The x-range is
 * trimmed to a quantile interval (default [q0.005, q0.995]) so a
 * single far-away outlier doesn't compress the visible range; samples
 * outside the trim are dropped from the bin counts.
 *
 * Bins are normalised so the total area equals the fraction of
 * samples that fell inside the trim range — i.e. the histogram is on
 * the same vertical scale as a PDF, so it can be overlaid directly
 * with a smooth density curve. (If you trim 1% of mass, the bars
 * integrate to 0.99, not 1; the density curve, computed from the
 * full distribution, integrates to 1. That tiny mismatch is the
 * honest cost of trimming and reads as "the bars cover the bulk".)
 *
 * Output shape mirrors histogramDensity (xs/ys/support/reference) so
 * the renderer can dispatch on `reference` uniformly. We additionally
 * return `binWidth` and `binEdges` so a bar-style chart can size the
 * rectangles correctly without re-deriving the spacing.
 *
 * @param {Float64Array|number[]} samples
 * @param {object} [opts]
 * @param {number} [opts.trimQ=0.005]  trim each tail to this quantile
 * @param {number} [opts.maxBins=200]  upper bound on bin count
 * @param {number} [opts.minBins=8]    lower bound — fewer than this hides shape
 * @returns {{ xs: Float64Array, ys: Float64Array, binEdges: Float64Array,
 *            binWidth: number, support: [number, number], reference: 'lebesgue' }}
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
  // Sort once; quantile lookups are O(1) on the sorted array.
  const sorted = Float64Array.from(samples);
  sorted.sort();

  const trimQ = opts.trimQ != null ? opts.trimQ : 0.005;
  const lo = quantileSorted(sorted, trimQ);
  const hi = quantileSorted(sorted, 1 - trimQ);
  if (!(hi > lo)) {
    // All-equal samples — degenerate; emit a single 1-wide bin centred
    // on the value so the chart doesn't crash.
    const v = sorted[0];
    const edges = new Float64Array([v - 0.5, v + 0.5]);
    return {
      xs: new Float64Array([v]),
      ys: new Float64Array([1]),
      binEdges: edges, binWidth: 1,
      support: [v - 0.5, v + 0.5], reference: 'lebesgue',
    };
  }

  const q1 = quantileSorted(sorted, 0.25);
  const q3 = quantileSorted(sorted, 0.75);
  const iqr = q3 - q1;
  // FD bin width — fall back to range/sqrt(n) when IQR collapses (e.g.
  // discrete-ish samples with a sharp mode) so we still get a non-zero
  // width. n_bins is then clamped to [minBins, maxBins] to avoid both
  // single-bar charts and pathological 10000-bin renders.
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
  // PDF normalisation: each bar's height = (count_in_bin / N) / binWidth,
  // so the total area sums to (samples-in-trim / N) ≈ 1 - 2·trimQ.
  const norm = 1 / (n * binWidth);
  const ys = new Float64Array(nBins);
  const xs = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) {
    ys[i] = counts[i] * norm;
    xs[i] = binEdges[i] + binWidth / 2;  // bin centre — handy for line-style overlays
  }
  return { xs, ys, binEdges, binWidth, support: [lo, hi], reference: 'lebesgue' };
}

/**
 * Sorted-array quantile via linear interpolation. Doesn't sort in
 * place — caller must pass an already-sorted array.
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
  let s = 0;
  for (let i = 0; i < n; i++) s += samples[i];
  const mean = s / n;
  let v = 0;
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean;
    v += d * d;
  }
  // Use n (population) rather than n-1 (sample). Bandwidth selection
  // is a heuristic, the bias correction wouldn't move the answer.
  return { mean, sd: Math.sqrt(v / n) };
}

// Helper: collect the transferable buffers in a reply. The browser shim
// uses this to populate the `transferList` argument of postMessage so
// large sample arrays move zero-copy across the worker boundary.
function transferablesOf(reply) {
  if (!reply) return [];
  if (reply.type === 'samples' && reply.samples instanceof Float64Array) {
    return [reply.samples.buffer];
  }
  if (reply.type === 'density') {
    const out = [];
    if (reply.xs instanceof Float64Array) out.push(reply.xs.buffer);
    if (reply.ys instanceof Float64Array) out.push(reply.ys.buffer);
    return out;
  }
  return [];
}

module.exports = {
  createWorkerHandler,
  transferablesOf,
  // Exported for unit-testing the density estimators in isolation
  // (tests/worker.test.js). Not part of the worker protocol surface.
  _internal: { kdeDensity, histogramDensity, freedmanDiaconisHistogram, quantileSorted, meanSd },
};
