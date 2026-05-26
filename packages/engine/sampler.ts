'use strict';

// Sampler — one-step `rand(state, measure, env)` and analytical `density()`
// for FlatPPL's built-in measure-typed distributions.
//
// =====================================================================
// Where this sits
// =====================================================================
//
// This module is the per-step backend the higher-level sampling
// orchestrator builds on. Its scope:
//
//   1. Take a FlatPIR-JSON measure expression (e.g. `Normal(mu=theta1,
//      sigma=1)`) and an environment binding referenced names to concrete
//      values, plus an RNG state, and produce one sample.
//
//   2. Provide analytical density / quantile information for visualization
//      (histogram + density-curve overlay; quantile-based plot ranges).
//
// What it deliberately *doesn't* do (yet, or ever):
//
//   - Walk dependency graphs / decide what to sample first. That's the
//     orchestrator's job, in a higher-level module that consumes this one.
//   - Cache samples. Cache lives in the orchestrator (signature-keyed).
//   - Handle posterior measures (`bayesupdate`, function-weighted
//     `weighted`, multivariate truncation). Per the FlatPPL spec, `rand`
//     refuses these. The sampler returns `Unsampleable` for them; the
//     v1.5 importance-sampling path is added separately.
//
// =====================================================================
// Architecture
// =====================================================================
//
// We use stdlib for both the analytical side (PDF, CDF, quantile, mean,
// variance) and the per-distribution sampling formulas (Box–Muller for
// Normal, ITS for Exponential, Marsaglia–Tsang for Gamma, …). Sampling
// is wired through stdlib's `prng` option, which expects a
// `() → [0, 1)`-returning closure. We bridge our pure-functional Philox
// (state in, value out, new state out) to that stateful interface via a
// small adapter that mutates internally and exposes `getState()` so the
// caller can read the trailing state when sampling completes.
//
// Why use stdlib rather than rolling our own:
//
//   - Saves writing and verifying ~15 distribution-sampler implementations.
//   - Density / quantile / variance / etc. come for free from the same
//     packages — the visualizer needs all of them.
//   - stdlib is well-tested, broadly used, and stays maintained.
//
// Cost: pulling stdlib into the engine package adds ~1 MB to the bundle
// and complicates state ownership (stdlib's `prng` holds onto a closure
// that mutates its captured Philox state). We accept this for v1; if
// bundle size becomes a real issue we can swap stdlib out for hand-rolled
// implementations distribution by distribution behind the same registry
// interface.
//
// =====================================================================
// Distribution registry
// =====================================================================
//
// Per FlatPPL spec, distributions take parameters with specific names.
// stdlib uses partly different conventions (Exponential's rate is called
// `lambda`, Gamma's params are alpha/beta, …). We map at the registry
// boundary — the IR keeps the spec names, the sampler's stdlib calls use
// the translated names. See `PARAM_TRANSLATION` below for the full table.

import type { IRNode } from './engine-types';

const rng = require('./rng.ts');
const valueLib = require('./value.ts');
const valueOps = require('./value-ops.ts');
const perfConfig = require('./perf-config.ts');

// Math special functions from stdlib for the gamma / loggamma /
// probit / invprobit op handlers in ARITH_OPS below. The bulk of
// stdlib (per-distribution Ctor + randFn + logpdfFn) moved to
// sampler-registry.ts in the §17 module split.
const stdlibGamma    = require('@stdlib/math-base-special-gamma');
const stdlibGammaln  = require('@stdlib/math-base-special-gammaln');
const stdlibErfc     = require('@stdlib/math-base-special-erfc');
const stdlibErfcinv  = require('@stdlib/math-base-special-erfcinv');

// Distribution registry — the REGISTRY catalog plus dispatch helpers
// (lookupDistribution, resolveParams, regionBoundsFromIR), the Philox
// PRNG bridge, and the support-bounds normaliser, all hoisted to
// sampler-registry.ts so standard modules can add distributions
// without touching the evaluator core here.
const registry = require('./sampler-registry.ts');
const {
  REGISTRY,
  isKnownDistribution,
  listDistributions,
  lookupDistribution,
  resolveParams,
  regionBoundsFromIR,
  makePhiloxPrngAdapter,
  readSupport,
} = registry;
/**
 * Sample one value from a built-in measure expression.
 *
 * Pure: returns [value, new_state]. The input `state` is not modified.
 *
 * `measureIR` is a FlatPIR-JSON call node (kind: 'call', op: <name>, ...).
 * `env` resolves any references inside parameter expressions to concrete
 * numeric values.
 *
 * Throws if `measureIR.op` is not a known distribution. Callers should
 * check `isKnownDistribution(op)` first when they expect non-builtin
 * measures (or use `canSample()`).
 */
function rand(state: any, measureIR: any, env: any) {
  const entry = lookupDistribution(measureIR);
  const params = resolveParams(measureIR, entry, env);

  // Bridge our pure Philox to stdlib's stateful prng option.
  const prng = makePhiloxPrngAdapter(state);
  const factoryOpts = { prng };
  const sampler = entry.randFn.factory(...params, factoryOpts);

  const value = sampler();
  return [value, prng.getState()];
}

/**
 * Build a reusable sampler closure for `measureIR` with parameters
 * resolved against `env`. Use this when you want to draw N samples
 * from a *fixed-parameter* distribution — building stdlib's factory
 * once and calling its returned function N times is dramatically
 * faster than calling rand() N times (factory creation dominates the
 * per-draw cost; a single `sampler()` call is just one PRNG read +
 * one transform).
 *
 * Returns:
 *   {
 *     draw():    number  — draw one sample, advances internal state
 *     getState(): rng.State — current Philox state for handing back
 *                              to the caller after a batch
 *   }
 *
 * Caller must NOT use this when params depend on per-draw upstreams
 * (e.g. mu = mu_i, sigma = 1) — the factory bakes in the params at
 * construction time. The orchestrator's per-i ref path should keep
 * calling rand() per draw, or rebuild the sampler per chunk.
 */
function makeSampler(state: any, measureIR: any, env: any) {
  const entry = lookupDistribution(measureIR);
  const params = resolveParams(measureIR, entry, env);
  const prng = makePhiloxPrngAdapter(state);
  const sampler = entry.randFn.factory(...params, { prng });
  return {
    draw: sampler,
    getState: () => prng.getState(),
  };
}

/**
 * Build a sampler whose params are supplied per draw rather than baked
 * in at factory time. Use this for the per-i-params path: the stdlib
 * factory closure is built ONCE (with only the prng bound), then each
 * `drawWith(env)` call resolves the IR's param expressions against the
 * current env and invokes the closure with those numerics.
 *
 * Why this matters: stdlib's distribution `factory(...params, opts)`
 * does non-trivial setup work — argument validation, internal
 * lookup-table builds for some discrete dists, etc. When params change
 * per atom, building a fresh factory per draw makes that setup cost
 * dominate (commonly ~10× the actual sampling cost). The
 * `factory(opts)` form returns a closure that accepts params per call,
 * so the setup happens once and per-draw cost collapses to the inner
 * transform + a single evaluateExpr per param.
 *
 * Generic across the whole REGISTRY — no per-distribution code, since
 * every stdlib `random-base-*` module exposes the same dual signature.
 *
 * Returns:
 *   {
 *     drawWith(env): number  — resolve params against `env`, draw one
 *                              sample. Advances internal prng state.
 *     getState():    rng.State — current Philox state for handing back
 *                                to the caller after a batch.
 *   }
 */
function makeParametricSampler(state: any, measureIR: any) {
  const entry = lookupDistribution(measureIR);
  const prng = makePhiloxPrngAdapter(state);
  const sampler = entry.randFn.factory({ prng });
  return {
    drawWith(env: any) {
      const params = resolveParams(measureIR, entry, env);
      return sampler(...params);
    },
    getState: () => prng.getState(),
  };
}

/**
 * Build a parameterized stdlib constructor for analytical queries.
 * Returns the stdlib distribution instance — has methods like .pdf(x),
 * .cdf(x), .quantile(p), .mean, .variance, .stdev, .support, etc.
 *
 * Useful when a caller wants more than density curves (e.g. plot
 * range determination, posterior summaries).
 */
function makeAnalytical(measureIR: any, env: any) {
  const entry = lookupDistribution(measureIR);
  const params = resolveParams(measureIR, entry, env);
  return new entry.Ctor(...params);
}

/**
 * Compute analytical density (or PMF) values for a built-in distribution,
 * suitable for plotting overlay on a sample histogram.
 *
 * Returns:
 *   {
 *     xs: Float64Array,
 *     ys: Float64Array,
 *     support: [low, high],
 *     reference: 'lebesgue' | 'counting',
 *   }
 *
 * For continuous (Lebesgue) distributions, xs is a grid spanning the
 * inter-quantile range [q(qLo), q(qHi)] (defaults: 0.001..0.999, clipped
 * to support if narrower). For discrete (counting) distributions, xs
 * is the integer atoms in that quantile range.
 */
function density(measureIR: any, env: any, opts: any) {
  opts = opts || {};
  const entry = lookupDistribution(measureIR);
  const dist = makeAnalytical(measureIR, env);

  // Plot range can be set three ways, in priority order:
  //   1. opts.range = [lo, hi]   — explicit override. Used when the
  //      caller already has a sample histogram and wants the analytical
  //      curve to align exactly with the bars (no extending past the
  //      first/last bin edge).
  //   2. quantile bounds          — opts.qLo, opts.qHi (defaults
  //      0.001..0.999). Useful when no histogram is available; keeps
  //      heavy-tailed distributions readable by trimming the far tails.
  //   3. distribution support     — fallback when the quantile call
  //      returns ±Infinity (e.g. Cauchy at q=0). Discrete dists may
  //      land here even with finite quantiles.
  let lo, hi;
  if (Array.isArray(opts.range) && opts.range.length === 2) {
    lo = opts.range[0];
    hi = opts.range[1];
  } else {
    const qLo = opts.qLo != null ? opts.qLo : 0.001;
    const qHi = opts.qHi != null ? opts.qHi : 0.999;
    lo = isFinite(dist.quantile(qLo)) ? dist.quantile(qLo) : null;
    hi = isFinite(dist.quantile(qHi)) ? dist.quantile(qHi) : null;
    const support = readSupport(dist);
    if (lo == null) lo = support[0];
    if (hi == null) hi = support[1];
  }

  if (entry.discrete) {
    // Counting reference: evaluate the PMF at integer atoms in the range.
    // stdlib uses `.pmf` for discrete distributions and `.pdf` for
    // continuous; we dispatch by the registry's `discrete` flag.
    const lo_ = Math.ceil(lo);
    const hi_ = Math.floor(hi);
    const n = Math.max(0, hi_ - lo_ + 1);
    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const k = lo_ + i;
      xs[i] = k;
      ys[i] = dist.pmf(k);
    }
    return { xs, ys, support: [lo_, hi_], reference: 'counting' };
  } else {
    // Lebesgue reference: evaluate the PDF on a uniform grid.
    const points = opts.gridPoints != null ? opts.gridPoints : 200;
    const xs = new Float64Array(points);
    const ys = new Float64Array(points);
    const step = (hi - lo) / (points - 1);
    for (let i = 0; i < points; i++) {
      const x = lo + i * step;
      xs[i] = x;
      ys[i] = dist.pdf(x);
    }
    return { xs, ys, support: [lo, hi], reference: 'lebesgue' };
  }
}

// =====================================================================
// Parameter resolution
// =====================================================================
//
// FlatPIR measure-call IR carries kwargs (object) holding the parameter
// expressions, plus optional positional args. We need to: (1) resolve
// each param's expression to a concrete number using `env`, (2) reorder
// to stdlib's positional order, (3) translate names where they differ.

/**
 * Evaluate a FlatPIR-JSON expression to a concrete value in the given
 * env. Supports the subset of expressions that can appear inside
 * distribution parameters: literals, refs, basic arithmetic. Throws on
 * forms it doesn't understand — those need the orchestrator to provide
 * pre-resolved values in env.
 */
function evaluateExpr(ir: IRNode, env: any): any {
  switch (ir.kind) {
    case 'lit':
      return ir.value;
    case 'const':
      return resolveConst(ir.name);
    case 'ref':
      return resolveRef(ir, env);
    case 'axis': {
      // Axis labels are bound by the enclosing aggregate(...) iteration
      // through `env.__axisEnv` (a per-iteration map). The analyzer
      // enforces that axes only appear inside an aggregate; outside that
      // scope env.__axisEnv is absent and we throw — matches the
      // "axes are not values" invariant of spec §05.
      const axisEnv = env && env.__axisEnv;
      const name = (ir as any).name;
      if (!axisEnv || !(name in axisEnv)) {
        throw new Error(`evaluateExpr: axis '.${name}' is not in scope ` +
          `(legal only inside aggregate(...))`);
      }
      return axisEnv[name];
    }
    case 'call':
      return evaluateCall(ir, env);
    default:
      throw new Error(`evaluateExpr: unsupported IR node kind '${(ir as any).kind}'`);
  }
}

// =====================================================================
// Batched IR evaluation — the foundation for per-atom paths across the
// engine (worker.evaluateN, density.applyAtomScalar, future profileN).
// =====================================================================
//
// Contract:
//   evaluateExprN(ir, refArrays, count, baseEnv, opts={}) → value
//
//   - Scalar-typed IR → Float64Array(count) when any input is per-atom;
//     a plain JS scalar (number / boolean) when all inputs are
//     atom-independent. The "scalar-when-possible" fast path means
//     single-point eval (count=1, refArrays=null) returns scalars with
//     no allocation overhead.
//   - Non-scalar-typed IR (vector/matrix/record/tuple) → atom-indep
//     JS value when no per-atom refs touch this subtree; otherwise a
//     length-N JS Array of per-atom values (packed to Float64Array
//     if every element is numeric/boolean).
//
//   refArrays: { name → Float64Array(count) } per-atom value overrides.
//              May be null/empty.
//   baseEnv:   atom-independent env (session env + lifted fixed-phase).
//   opts.overlay: { name → scalar } that wins over BOTH refArrays and
//                 baseEnv at every leaf — used by density.js for
//                 env-threading from consumed observation fields.
//
// Env-precedence at refs (highest first): overlay > refArrays > baseEnv.
//
// Scalar arith ops dispatch through ARITH_OPS_N (batched broadcast).
// Non-scalar ops fall back to per-atom dispatch via the single-point
// `evaluateExpr` — fine for ops where per-atom inputs are uncommon
// (vector/matrix/record builders typically take atom-indep inputs);
// future batched-non-scalar rewrites would eliminate the fallback.

function evaluateExprN(ir: IRNode, refArrays: any, count: any, baseEnv: any, opts: any) {
  const N = count | 0;
  if (N <= 0) throw new Error('evaluateExprN: count must be positive');
  const overlay = (opts && opts.overlay) || null;
  return _evalN(ir, refArrays || null, N, baseEnv || {}, overlay);
}

function _evalN(ir: any, refArrays: any, N: any, baseEnv: any, overlay: any) {
  switch (ir.kind) {
    case 'lit':
      return ir.value;
    case 'const':
      return resolveConst(ir.name);
    case 'ref': {
      if (overlay && Object.prototype.hasOwnProperty.call(overlay, ir.name)) {
        return overlay[ir.name];
      }
      if (refArrays && Object.prototype.hasOwnProperty.call(refArrays, ir.name)) {
        return refArrays[ir.name];
      }
      if (baseEnv != null && ir.name in baseEnv) return baseEnv[ir.name];
      throw new Error(
        `evaluateExprN: unbound ${ir.ns} reference '${ir.name}' — env must ` +
        `provide values for all upstream-resolved names`
      );
    }
    case 'call': {
      const op = ir.op;
      // Scalar-batched dispatch.
      if (op in ARITH_OPS_N) {
        const args = (ir.args || []).map((a: any) => _evalN(a, refArrays, N, baseEnv, overlay));
        return ARITH_OPS_N[op](args, N);
      }
      // Batched approximation functions. Each has the shape
      // (atom-indep coefficients/edges/values, per-atom x). If x batches
      // (Float64Array(N) or Value shape=[N]), collapse the per-atom JS-
      // call overhead into a single tight inner loop. Otherwise fall
      // through to _perAtomFallback (atom-indep eval or per-atom dispatch).
      if (op === 'polynomial' || op === 'bernstein' || op === 'stepwise') {
        const r = _batchedApproximation(op, ir, refArrays, N, baseEnv, overlay);
        if (r !== _BATCH_FELL_THROUGH) return r;
      }
      // Non-batched op: per-atom dispatch through the existing
      // single-point evaluateExpr. Atom-indep when no per-atom refs
      // touch this subtree (cheap one-shot); per-atom loop otherwise.
      return _perAtomFallback(ir, refArrays, N, baseEnv, overlay);
    }
    default:
      throw new Error(`evaluateExprN: unsupported IR node kind '${ir.kind}'`);
  }
}

// Batched approximation function dispatch (polynomial /
// bernstein / stepwise). Each takes atom-indep coefficient-class
// arguments and a per-atom x; if x batches, we run one tight Horner /
// Bernstein-basis / stepwise loop over the entire N-atom batch
// rather than N JS function calls through the per-atom fallback.
//
// The sentinel `_BATCH_FELL_THROUGH` distinguishes "I refused to
// handle this; please use the per-atom fallback" from a legitimate
// `null` / `undefined` result.
const _BATCH_FELL_THROUGH = Symbol('batch-fell-through');

function _batchedApproximation(op: any, ir: any, refArrays: any, N: any, baseEnv: any, overlay: any) {
  const kw = ir.kwargs || {};
  // Each op has the same coefficient-class + x shape; pull both.
  let coeffsIR, xIR, edgesIR;
  if (op === 'polynomial') {
    coeffsIR = kw.coefficients != null ? kw.coefficients : ir.args && ir.args[0];
    xIR      = kw.x            != null ? kw.x            : ir.args && ir.args[1];
  } else if (op === 'bernstein') {
    coeffsIR = kw.coefficients != null ? kw.coefficients : ir.args && ir.args[0];
    xIR      = kw.x            != null ? kw.x            : ir.args && ir.args[1];
  } else { // stepwise
    edgesIR  = kw.edges        != null ? kw.edges        : ir.args && ir.args[0];
    coeffsIR = kw.values       != null ? kw.values       : ir.args && ir.args[1];
    xIR      = kw.x            != null ? kw.x            : ir.args && ir.args[2];
  }
  if (!coeffsIR || !xIR) return _BATCH_FELL_THROUGH;

  // Evaluate the atom-indep operands once. For batching to apply,
  // coefficients/edges/values must not depend on per-atom refs — we
  // detect this by evaluating against baseEnv WITHOUT refArrays.
  // The cheap path: if the IR has no self-refs, evaluation against
  // baseEnv succeeds; otherwise it throws and we fall through.
  let coeffs, edges;
  try {
    coeffs = evaluateExpr(coeffsIR, baseEnv);
    if (op === 'stepwise') edges = evaluateExpr(edgesIR, baseEnv);
  } catch (_) {
    return _BATCH_FELL_THROUGH;
  }
  // Unwrap shape-explicit Value → typed-array data view.
  if (valueLib.isValue(coeffs)) coeffs = coeffs.data;
  if (op === 'stepwise' && valueLib.isValue(edges)) edges = edges.data;
  if (!Array.isArray(coeffs) && !(coeffs && coeffs.BYTES_PER_ELEMENT)) {
    return _BATCH_FELL_THROUGH;
  }
  if (op === 'stepwise'
      && !Array.isArray(edges) && !(edges && edges.BYTES_PER_ELEMENT)) {
    return _BATCH_FELL_THROUGH;
  }

  // Evaluate x. If it's per-atom (Float64Array(N) or Value shape=[N]),
  // batch. Otherwise fall through.
  const xVal = _evalN(xIR, refArrays, N, baseEnv, overlay);
  const xIsBatch = isBatch(xVal, N);
  if (!xIsBatch) {
    // x is atom-indep — the per-atom fallback already short-circuits
    // to a single evaluateExpr call (one-shot path). No reason to
    // run the batched loop here.
    return _BATCH_FELL_THROUGH;
  }
  const xData = xVal instanceof Float64Array ? xVal : xVal.data;

  // Tight inner loops. coeffs is treated as length-k indexed via [];
  // works for Array and Float64Array uniformly.
  const out = new Float64Array(N);
  if (op === 'polynomial') {
    const k = coeffs.length;
    if (k === 0) {
      // Zero polynomial → 0 for every atom; out is already zero-filled.
    } else {
      for (let i = 0; i < N; i++) {
        const x = xData[i];
        let acc = coeffs[k - 1];
        for (let j = k - 2; j >= 0; j--) acc = acc * x + coeffs[j];
        out[i] = acc;
      }
    }
  } else if (op === 'bernstein') {
    const n = coeffs.length - 1;
    if (n < 0) {
      // out already zeros
    } else {
      for (let i = 0; i < N; i++) {
        const x = xData[i];
        const omx = 1 - x;
        if (omx === 0) {
          out[i] = coeffs[n];
          continue;
        }
        let acc = 0, binom = 1, xk = 1, omxn = Math.pow(omx, n);
        for (let k = 0; k <= n; k++) {
          acc += coeffs[k] * binom * xk * omxn;
          xk *= x;
          omxn /= omx;
          binom = binom * (n - k) / (k + 1);
        }
        out[i] = acc;
      }
    }
  } else { // stepwise
    const nBin = coeffs.length;
    if (edges.length !== nBin + 1) {
      throw new Error('stepwise: edges length must equal values length + 1');
    }
    const eLast = edges[nBin];
    for (let i = 0; i < N; i++) {
      const x = xData[i];
      if (x < edges[0] || x > eLast) { out[i] = NaN; continue; }
      let v = NaN;
      for (let b = 0; b < nBin; b++) {
        if (x >= edges[b]
            && (x < edges[b + 1] || (b === nBin - 1 && x === edges[b + 1]))) {
          v = coeffs[b];
          break;
        }
      }
      out[i] = v;
    }
  }
  // Same "kind as inputs" semantics: if xVal was a
  // Value, return a Value; if it was a bare Float64Array, return one.
  if (valueLib.isValue(xVal)) return valueLib.batchedScalar(out);
  return out;
}

function _perAtomFallback(ir: any, refArrays: any, N: any, baseEnv: any, overlay: any) {
  const refNames = refArrays ? Object.keys(refArrays) : [];
  const overlayKeys = overlay ? Object.keys(overlay) : null;
  // Fast path: nothing varies per atom AND overlay is empty → one shot.
  if (refNames.length === 0 && !overlayKeys) {
    return evaluateExpr(ir, baseEnv);
  }
  // Per-atom loop. Build a callEnv that overlays per-atom refArrays
  // values + the overlay (overlay last so it wins over refArrays).
  const callEnv = Object.assign({}, baseEnv);
  if (overlayKeys) for (let j = 0; j < overlayKeys.length; j++) {
    callEnv[overlayKeys[j]] = overlay[overlayKeys[j]];
  }
  // Pre-compute per-ref accessors. refArrays uniformly carry Values
  // internally; bare Float64Array entries are accepted as a back-
  // compat path for tests that pass refArrays directly.
  // shape=[N] (scalar atoms) → `i => data[i]`; shape=[N,
  // ...rest] (vector / matrix atoms) → atom i is a length-prod(rest)
  // sub-Value (subarray view). Computing the access pattern once per
  // ref keeps the inner N-atom loop branch-free.
  const accessors = new Array(refNames.length);
  for (let j = 0; j < refNames.length; j++) {
    const k = refNames[j];
    const v = refArrays[k];
    if (valueLib.isValue(v)) {
      const shape = v.shape;
      const data = v.data;
      if (shape.length === 1) {
        accessors[j] = (i: any) => data[i];
      } else {
        const tailDims = shape.slice(1);
        const tailLen = tailDims.reduce((a: any, b: any) => a * b, 1);
        accessors[j] = (i: any) => ({
          shape: tailDims,
          data: data.subarray(i * tailLen, (i + 1) * tailLen),
        });
      }
    } else {
      // Back-compat: refArrays passed directly as a Float64Array (by
      // tests / external callers that haven't migrated to Values).
      const arr = v;
      accessors[j] = (i: any) => arr[i];
    }
  }
  const out = new Array(N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < refNames.length; j++) {
      const k = refNames[j];
      // overlay wins over refArrays — skip the write when overlay has it.
      if (overlayKeys && Object.prototype.hasOwnProperty.call(overlay, k)) continue;
      callEnv[k] = accessors[j](i);
    }
    out[i] = evaluateExpr(ir, callEnv);
  }
  // Pack to Float64Array if every result is numeric/boolean (the
  // common case for non-scalar ops whose outputs happen to be scalar,
  // e.g. polynomial / get_field returning numbers).
  let allNumeric = true;
  for (let i = 0; i < N; i++) {
    const t = typeof out[i];
    if (t !== 'number' && t !== 'boolean') { allNumeric = false; break; }
  }
  if (allNumeric) {
    const arr = new Float64Array(N);
    for (let i = 0; i < N; i++) arr[i] = +out[i];
    return arr;
  }
  // Per-atom uniform-length numeric array outputs (e.g.
  // softmax / l1unit / l2unit / logsoftmax on per-atom inputs) pack
  // atom-major into a Value shape=[N, k]. Detect uniform shape across
  // atoms; non-uniform results stay as a JS array of per-atom values
  // (the caller chooses how to surface them).
  let allUniformArrays = true;
  const sample0 = out[0];
  let k0 = 0;
  if (Array.isArray(sample0) || (sample0 && sample0.BYTES_PER_ELEMENT !== undefined)) {
    k0 = sample0.length;
  } else if (valueLib.isValue(sample0) && sample0.shape.length === 1) {
    k0 = sample0.shape[0];
  } else {
    allUniformArrays = false;
  }
  for (let i = 0; allUniformArrays && i < N; i++) {
    const s = out[i];
    let len;
    if (Array.isArray(s) || (s && s.BYTES_PER_ELEMENT !== undefined)) {
      len = s.length;
    } else if (valueLib.isValue(s) && s.shape.length === 1) {
      len = s.shape[0];
    } else {
      allUniformArrays = false; break;
    }
    if (len !== k0) { allUniformArrays = false; break; }
    // Also require numeric entries throughout.
    for (let j = 0; allUniformArrays && j < k0; j++) {
      const e = valueLib.isValue(s) ? s.data[j] : s[j];
      if (typeof e !== 'number' && typeof e !== 'boolean') {
        allUniformArrays = false; break;
      }
    }
  }
  if (allUniformArrays && k0 > 0) {
    const data = new Float64Array(N * k0);
    for (let i = 0; i < N; i++) {
      const s = out[i];
      const src = valueLib.isValue(s) ? s.data : s;
      const base = i * k0;
      for (let j = 0; j < k0; j++) data[base + j] = +src[j];
    }
    return { shape: [N, k0], data };
  }
  return out;
}

function resolveConst(name: any) {
  switch (name) {
    case 'pi':  return Math.PI;
    case 'e':   return Math.E;
    case 'inf': return Infinity;
    case '-inf': return -Infinity;
    case 'im':  return { re: 0, im: 1 };  // imaginary unit (spec §03)
    default:
      throw new Error(`evaluateExpr: unknown constant '${name}'`);
  }
}

// =====================================================================
// Complex arithmetic primitives (spec §03 / §07)
// =====================================================================
// Extracted to sampler-complex.ts (engine-concepts §17.5 split).

const _complex = require('./sampler-complex.ts');
const {
  _isComplex, _toComplex,
  _cAdd, _cSub, _cNeg, _cMul, _cDiv, _cConj, _cAbs, _cAbs2,
  _cExp, _cLog, _cSqrt, _cPow,
} = _complex;

function resolveRef(ir: any, env: any) {
  // Both 'self' and '%local' references look up by name in the same env.
  // Cross-module refs (ns = <module>) would require the orchestrator to
  // populate env with the appropriate scoped values; we don't distinguish.
  if (env == null || !(ir.name in env)) {
    throw new Error(
      `evaluateExpr: unbound ${ir.ns} reference '${ir.name}' — env must ` +
      `provide values for all upstream-resolved names`
    );
  }
  return env[ir.name];
}

// Determine whether `v` carries an intrinsic vector/matrix shape
// (i.e. is a Value with rank ≥ 2, or rank 1 with length ≠ atom-count
// N). Used by mul / add / sub dispatchers to route shape-aware paths.
// Bare JS numbers and bare Float64Arrays are NOT rich; they pass
// through the existing scalar-broadcast helpers.
function _isShapeRich(v: any, N?: any) {
  if (!valueLib.isValue(v)) return false;
  const r = v.shape.length;
  if (r === 0) return false;                    // atom-indep scalar
  if (r === 1 && N !== undefined && v.shape[0] === N) return false;  // batched scalar
  return true;
}

// Shape-aware add/sub/neg dispatcher (used inline in the
// ARITH_OPS table below). Same logic as the mul dispatcher: if either
// operand carries an intrinsic vector/matrix shape, route to
// value-ops; otherwise stay on the scalar JS fast path.
//
// Complex extension: when either operand is a complex scalar ({re, im}),
// promote both to complex and dispatch through the complex helper. The
// shape-rich Value path stays real-only for now; per-atom complex
// Values are deferred (storage decision: separate re/im Float64Arrays
// in Value.im — lands when needed).
function _shapeAwareBinop(opName: any, scalarFn: any, a: any, b: any, complexFn: any) {
  if (_isComplex(a) || _isComplex(b)) {
    return complexFn(_toComplex(a), _toComplex(b));
  }
  if (valueLib.isValue(a) || valueLib.isValue(b)) {
    if (_isShapeRich(a) || _isShapeRich(b)) {
      return valueOps[opName](valueLib.asValue(a), valueLib.asValue(b));
    }
    const av = valueLib.isValue(a) ? a.data[0] : a;
    const bv = valueLib.isValue(b) ? b.data[0] : b;
    return scalarFn(av, bv);
  }
  return scalarFn(a, b);
}

const ARITH_OPS = {
  add: (a: any, b: any) => _shapeAwareBinop('add', (x: any, y: any) => x + y, a, b, _cAdd),
  sub: (a: any, b: any) => _shapeAwareBinop('sub', (x: any, y: any) => x - y, a, b, _cSub),
  // mul: shape-dispatched. Bare scalars stay on the JS-multiply fast
  // path; Value inputs with rank ≥ 1 (vectors / matrices, with
  // Klein-4 transpose tag respected) route to value-ops.mul. The
  // dispatcher is reached when:
  //   - ARITH_OPS_N.mul → broadcast2 sees both inputs as not-batched,
  //     unwraps the (any) Value to its scalar via _scalarVal, calls
  //     ARITH_OPS.mul. Hit only when both operands are scalars (the
  //     atom-indep fast path); the (a, b) → a*b primitive still wins.
  //   - ARITH_OPS_N.mul detects a shape-rich Value and routes around
  //     broadcast2 directly to value-ops.mul (see ARITH_OPS_N
  //     construction below).
  //   - evaluateExpr (single-point) hits ARITH_OPS.mul directly. If
  //     either arg is a shape-rich Value it dispatches here.
  mul: (a: any, b: any) => {
    if (_isComplex(a) || _isComplex(b)) {
      return _cMul(_toComplex(a), _toComplex(b));
    }
    if (valueLib.isValue(a) || valueLib.isValue(b)) {
      // Route to value-ops only when at least one operand has
      // intrinsic shape; otherwise keep the scalar JS fast path
      // (Value shape=[] inputs get unwrapped by _scalarVal above and
      // never reach here in that form).
      if (_isShapeRich(a) || _isShapeRich(b)) {
        return valueOps.mul(valueLib.asValue(a), valueLib.asValue(b));
      }
      // Both are atom-indep scalars (one wrapped, one bare or both
      // wrapped). Unwrap and multiply.
      const av = valueLib.isValue(a) ? a.data[0] : a;
      const bv = valueLib.isValue(b) ? b.data[0] : b;
      return av * bv;
    }
    return a * b;
  },
  div: (a: any, b: any) => {
    if (_isComplex(a) || _isComplex(b)) {
      return _cDiv(_toComplex(a), _toComplex(b));
    }
    return a / b;
  },
  divide: (a: any, b: any) => {
    if (_isComplex(a) || _isComplex(b)) {
      return _cDiv(_toComplex(a), _toComplex(b));
    }
    return a / b;
  },
  mod: (a: any, b: any) => a % b,
  neg: (a: any) => {
    if (_isComplex(a)) return _cNeg(a);
    if (valueLib.isValue(a)) {
      if (_isShapeRich(a)) return valueOps.neg(a);
      return -a.data[0];
    }
    return -a;
  },
  pos: (a: any) => _isComplex(a) ? a : +a,
  // identity(x): returns x unchanged. Spec §07 — first-class function
  // form is useful as a default `f` argument (e.g. pushfwd(identity, M)
  // ≡ M) and as a placeholder during interactive editing.
  identity: (a: any) => a,
  // Common unary maths — extend EVALUABLE_OPS in orchestrator.js
  // alongside any addition here so the static gate matches. Complex
  // arguments dispatch to the complex helpers (spec §07: exp / log /
  // sqrt / abs / abs2 extend naturally; sin/cos defer until needed).
  abs:   (a: any) => _isComplex(a) ? _cAbs(a) : Math.abs(a),
  abs2:  (a: any) => _isComplex(a) ? _cAbs2(a) : a * a,
  exp:   (a: any) => _isComplex(a) ? _cExp(a) : Math.exp(a),
  log:   (a: any) => _isComplex(a) ? _cLog(a) : Math.log(a),
  log10: (a: any) => Math.log10(a),
  log1p: (a: any) => Math.log1p(a),
  expm1: (a: any) => Math.expm1(a),
  sqrt:  (a: any) => _isComplex(a) ? _cSqrt(a) : Math.sqrt(a),
  sin:   (a: any) => Math.sin(a),
  cos:   (a: any) => Math.cos(a),
  tan:   (a: any) => Math.tan(a),
  asin:  (a: any) => Math.asin(a),
  acos:  (a: any) => Math.acos(a),
  atan:  (a: any) => Math.atan(a),
  atan2: (y: any, x: any) => Math.atan2(y, x),
  sinh:  (a: any) => Math.sinh(a),
  cosh:  (a: any) => Math.cosh(a),
  tanh:  (a: any) => Math.tanh(a),
  asinh: (a: any) => Math.asinh(a),
  acosh: (a: any) => Math.acosh(a),
  atanh: (a: any) => Math.atanh(a),
  floor: (a: any) => Math.floor(a),
  ceil:  (a: any) => Math.ceil(a),
  round: (a: any) => Math.round(a),
  pow:   (a: any, b: any) => {
    if (_isComplex(a) || _isComplex(b)) {
      return _cPow(_toComplex(a), _toComplex(b));
    }
    return Math.pow(a, b);
  },
  // Complex constructor + accessors (spec §03 / §07).
  complex: (re: any, im: any) => {
    // Two-arg constructor: complex(re, im) → {re, im}. (One-arg form
    // `complex(x)` is the scalar restrictor — identity if already
    // complex; lift if real.)
    if (im === undefined) {
      if (_isComplex(re)) return re;
      if (typeof re === 'number') return { re: re, im: 0 };
      throw new Error('complex: single-arg restrictor requires real or complex input');
    }
    return { re: +re, im: +im };
  },
  real: (z: any) => _isComplex(z) ? z.re : +z,
  imag: (z: any) => _isComplex(z) ? z.im : 0,
  conj: (z: any) => _isComplex(z) ? _cConj(z) : +z,
  cis: (theta: any) => ({ re: Math.cos(+theta), im: Math.sin(+theta) }),
  // Binary scalar reductions (spec §07). Distinct from the variadic
  // `maximum` / `minimum` reductions, which take an array argument.
  min:   (a: any, b: any) => Math.min(a, b),
  max:   (a: any, b: any) => Math.max(a, b),
  // Special functions: gamma function family and link functions (spec
  // §07 "Elementary functions"). Domain checks are minimal here —
  // stdlib returns NaN / Infinity for out-of-domain inputs which the
  // caller can detect via isfinite / isnan.
  gamma:     (a: any) => stdlibGamma(a),
  loggamma:  (a: any) => stdlibGammaln(a),
  // Link functions:
  //   logit(p)    = log(p / (1−p))                            on (0,1)
  //   invlogit(x) = 1 / (1 + exp(−x))                         on ℝ
  //   probit(p)   = Φ⁻¹(p)  via the erfcinv identity          on (0,1)
  //   invprobit(x)= Φ(x)    via the erfc identity             on ℝ
  // probit and invprobit reach ±∞ at the endpoints (spec §07).
  logit:     (p: any) => Math.log(p / (1 - p)),
  invlogit:  (x: any) => 1 / (1 + Math.exp(-x)),
  probit:    (p: any) => -Math.SQRT2 * stdlibErfcinv(2 * p),
  invprobit: (x: any) => 0.5 * stdlibErfc(-x / Math.SQRT2),
  // Comparison ops produce booleans (per spec §07). Spec preserves
  // strict typing — no implicit numeric→boolean cast — so the
  // operands are treated as reals and the result is JS boolean.
  lt:      (a: any, b: any) => a < b,
  le:      (a: any, b: any) => a <= b,
  gt:      (a: any, b: any) => a > b,
  ge:      (a: any, b: any) => a >= b,
  equal:   (a: any, b: any) => a === b,
  unequal: (a: any, b: any) => a !== b,
  // Predicates over reals.
  isfinite: (a: any) => Number.isFinite(a),
  isinf:    (a: any) => !Number.isNaN(a) && !Number.isFinite(a),
  isnan:    (a: any) => Number.isNaN(a),
  iszero:   (a: any) => a === 0,
  // Logic / conditionals (spec §07). FlatPPL booleans are strict — we
  // don't coerce truthy values, the typeinfer pass already requires
  // boolean operands. lxor is exclusive-or; ifelse is the conditional
  // expression returning the first or second branch by cond.
  land:    (a: any, b: any) => a && b,
  lor:     (a: any, b: any) => a || b,
  lxor:    (a: any, b: any) => a !== b,
  lnot:    (a: any) => !a,
  ifelse:  (c: any, a: any, b: any) => c ? a : b,
  // Vector constructor — turns positional args into a JS array.
  // Lower than the typed Float64Array path used for materialised
  // measures: this is for inline `[a, b, c]` literals that surface
  // as (call vector (lit …) (lit …) …) IR. Reductions below take
  // these arrays directly.
  // vector(x1, x2, ...) — per spec §03, a rank-1 array. When all
  // entries are scalar numerics/booleans we lift to a shape-explicit
  // Value (engine-concepts §2.1 contract); when entries are themselves
  // vectors/records/etc. we keep a JS array of those elements —
  // preserving the "vector of vectors" distinction from a matrix that
  // spec §03 establishes (matrices are made explicitly via rowstack /
  // colstack).
  vector: (...xs: any[]) => {
    if (xs.length === 0) return { shape: [0], data: new Float64Array(0) };
    let allScalar = true;
    for (let i = 0; i < xs.length; i++) {
      const t = typeof xs[i];
      if (t !== 'number' && t !== 'boolean') { allScalar = false; break; }
    }
    if (!allScalar) return xs;
    const data = new Float64Array(xs.length);
    for (let i = 0; i < xs.length; i++) {
      data[i] = xs[i] === true ? 1 : xs[i] === false ? 0 : +xs[i];
    }
    return { shape: [xs.length], data: data };
  },
  // cat(...) — structural concatenation per spec §07. Three shape
  // classes; mixing is a runtime error (the type checker rejects
  // most mixes statically but the runtime check guards programmatic
  // cases). Like `vector`, cat lives in ARITH_OPS so fixed-phase
  // pre-eval and fn-body evaluations can compute it, but it is NOT
  // listed in EVALUABLE_OPS — cat of stochastic refs produces a
  // per-atom vector, which doesn't fit the scalar-per-atom worker
  // contract. Vector-valued stochastic bindings have their own
  // tuple / array materialiser paths.
  // rowstack(vs) — turn a vector of vectors into a matrix where the
  // input vectors become rows. colstack does the same but as columns.
  // Spec §07. All input vectors must have the same length.
  // rowstack / colstack lift a vector-of-vectors into a true rank-2
  // matrix. Per spec §03 "vectors of vectors are not interpreted as
  // matrices implicitly, but can be turned into matrices explicitly
  // using rowstack or colstack" — so the result MUST be a shape-
  // explicit Value (rank=2), not just a nested array. Without this,
  // `mul`/`add`/`sub` on rowstack outputs would fall through the
  // valueLib.isValue branch and yield NaN.
  rowstack: (vs: any) => {
    if (!Array.isArray(vs) || vs.length === 0) {
      return { shape: [0, 0], data: new Float64Array(0) };
    }
    // Accept either a JS array of rows (each a JS array or a rank-1
    // Value) or a single Value that's a vector-of-vectors view.
    const m = vs.length;
    const firstRow = vs[0];
    const n = valueLib.isValue(firstRow) ? firstRow.shape[firstRow.shape.length - 1]
                                          : firstRow.length;
    const data = new Float64Array(m * n);
    for (let i = 0; i < m; i++) {
      const row = vs[i];
      const rowLen = valueLib.isValue(row) ? row.shape[row.shape.length - 1] : row.length;
      if (rowLen !== n) {
        throw new Error('rowstack: row length mismatch at index ' + i);
      }
      const rowData = valueLib.isValue(row) ? row.data : row;
      for (let j = 0; j < n; j++) data[i * n + j] = +rowData[j];
    }
    return { shape: [m, n], data: data };
  },
  colstack: (vs: any) => {
    if (!Array.isArray(vs) || vs.length === 0) {
      return { shape: [0, 0], data: new Float64Array(0) };
    }
    const cols = vs.length;
    const firstCol = vs[0];
    const m = valueLib.isValue(firstCol) ? firstCol.shape[firstCol.shape.length - 1]
                                          : firstCol.length;
    const data = new Float64Array(m * cols);
    for (let c = 0; c < cols; c++) {
      const col = vs[c];
      const colLen = valueLib.isValue(col) ? col.shape[col.shape.length - 1] : col.length;
      if (colLen !== m) {
        throw new Error('colstack: column length mismatch at index ' + c);
      }
      const colData = valueLib.isValue(col) ? col.data : col;
      for (let r = 0; r < m; r++) data[r * cols + c] = +colData[r];
    }
    return { shape: [m, cols], data: data };
  },
  // ---- Linear algebra (spec §07) ------------------------------------
  // Ops dispatch on Value
  // inputs. Value path bridges through nested-array form for the
  // matrix-decomposition algorithms (atom-indep one-shot ops; the
  // realloc cost is negligible against the algorithm work). transpose
  // and adjoint use the free Klein-4 tag flip on Value input (no
  // data allocation).
  //
  // Legacy nested-JS-array inputs continue to work unchanged — the
  // existing linear-algebra test suite exercises them via the IR
  // pipeline. Materialisers produce Values, so the
  // Value path becomes the default.
  //
  // Matrices in legacy form are nested JS arrays (row-major):
  // M[i][j] is row i, col j. Vectors are flat JS arrays.
  transpose: (M: any) => {
    if (valueLib.isValue(M)) return valueLib.transpose(M);  // O(1) tag flip
    if (!Array.isArray(M) || M.length === 0) return [];
    const rows = M.length, cols = M[0].length;
    const out = new Array(cols);
    for (let j = 0; j < cols; j++) {
      const row = new Array(rows);
      for (let i = 0; i < rows; i++) row[i] = M[i][j];
      out[j] = row;
    }
    return out;
  },
  // adjoint = conjugate transpose. For real matrices = transpose.
  // (Complex support deferred — but the tag tracks it.)
  adjoint: (M: any) => {
    if (valueLib.isValue(M)) return valueLib.adjoint(M);  // O(1) tag flip
    return ARITH_OPS.transpose(M);
  },
  trace: (M: any): any => {
    if (valueLib.isDiagStored(M) && !M.im) {           // O(n): Σ diagonal
      let s = 0; for (let i = 0; i < M.data.length; i++) s += M.data[i];
      return valueLib.scalar(s);
    }
    if (valueLib.isValue(M)) {
      // Densify any structural overlay, then read the diagonal via the
      // row-major stride (n+1). No nested-JS round-trip.
      const D = valueLib.densify(M);
      const n = D.shape[0];
      if (n !== D.shape[1]) throw new Error('trace: argument must be a square matrix');
      let s = 0;
      for (let i = 0; i < n; i++) s += D.data[i * n + i];
      return valueLib.scalar(s);
    }
    if (!Array.isArray(M)) throw new Error('trace: argument must be a matrix');
    const n = M.length;
    if (n === 0 || M[0].length !== n) {
      throw new Error('trace: argument must be a square matrix');
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += M[i][i];
    return s;
  },
  // diagmat(v) → the m×m diagonal matrix with v on the diagonal.
  // Produces the vector-backed `diag` structured Value (O(m) storage);
  // every diag-aware op fast-paths it, anything else densify()s. A
  // complex diagonal carries its imaginary part on the same vector.
  diagmat: (v: any) => {
    if (valueLib.isValue(v)) {
      if (v.shape.length !== 1) {
        throw new Error('diagmat: argument must be a rank-1 vector, got shape='
          + JSON.stringify(v.shape));
      }
      return valueLib.diagMatrix(v.data, v.im);
    }
    return valueLib.diagMatrix(v instanceof Float64Array ? v : Float64Array.from(v));
  },
  self_outer: (v: any): any => {
    if (valueLib.isValue(v)) {
      const D = valueLib.densify(v);
      if (D.shape.length !== 1) {
        throw new Error('self_outer: argument must be a rank-1 vector, got shape='
          + JSON.stringify(D.shape));
      }
      const n = D.shape[0];
      const out = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) out[i * n + j] = D.data[i] * D.data[j];
      }
      return { shape: [n, n], data: out };
    }
    const n = v.length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array(n);
      for (let j = 0; j < n; j++) row[j] = v[i] * v[j];
      out[i] = row;
    }
    return out;
  },
  // det(A): determinant via LU with partial pivoting. Returns 0 for
  // singular matrices. O(n³).
  det: (A: any): any => {
    if (valueLib.isDiagStored(A) && !A.im) {           // O(n): ∏ diagonal
      let p = 1; for (let i = 0; i < A.data.length; i++) p *= A.data[i];
      return valueLib.scalar(p);
    }
    if (valueLib.isValue(A)) {
      return valueLib.scalar(_detLUValue(valueLib.densify(A)));
    }
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('det: argument must be a non-empty square matrix');
    }
    return _detLU(A);
  },
  // logabsdet(A): log |det(A)|. Computed alongside det via the LU
  // decomposition to keep numerical stability on near-singular inputs.
  logabsdet: (A: any): any => {
    if (valueLib.isDiagStored(A) && !A.im) {           // O(n): Σ log|diag|
      let s = 0; for (let i = 0; i < A.data.length; i++) s += Math.log(Math.abs(A.data[i]));
      return valueLib.scalar(s);
    }
    if (valueLib.isValue(A)) {
      return valueLib.scalar(_logAbsDetLUValue(valueLib.densify(A)));
    }
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('logabsdet: argument must be a non-empty square matrix');
    }
    return _logAbsDetLU(A);
  },
  // inv(A): matrix inverse via LU + back-substitution against I. O(n³).
  // Throws on singular matrices.
  inv: (A: any): any => {
    if (valueLib.isDiagStored(A) && !A.im) {           // O(n): reciprocal
      const d = new Float64Array(A.data.length);
      for (let i = 0; i < d.length; i++) {
        if (A.data[i] === 0) throw new Error('inv: singular diagonal matrix');
        d[i] = 1 / A.data[i];
      }
      return valueLib.diagMatrix(d);
    }
    if (valueLib.isValue(A)) {
      return _invValue(valueLib.densify(A));
    }
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('inv: argument must be a non-empty square matrix');
    }
    return _invGaussJordan(A);
  },
  // linsolve(A, b): solve A x = b for x. b may be a vector (returns
  // vector) or a matrix (returns matrix, solved column-by-column).
  linsolve: (A: any, b: any): any => {
    // Diagonal A: x = b ⊘ diag, O(n), no factorization. Vector b only
    // (the MvNormal/whitening case); matrix b densifies via the
    // generic path below.
    if (valueLib.isDiagStored(A) && !A.im) {
      const d = A.data, m = d.length;
      const bv = valueLib.isValue(b) ? b : null;
      const bd = bv ? bv.data : b;
      if (bd && bd.length === m && (!bv || bv.shape.length === 1)) {
        const x = new Float64Array(m);
        for (let i = 0; i < m; i++) {
          if (d[i] === 0) throw new Error('linsolve: singular diagonal matrix');
          x[i] = bd[i] / d[i];
        }
        return bv ? valueLib.vector(x) : Array.from(x);
      }
    }
    if (valueLib.isValue(A) || valueLib.isValue(b)) {
      // Promote both operands to dense Values so the LU path operates
      // on row-major Float64Array data directly (no nested-JS roundtrip).
      const aV = valueLib.isValue(A) ? valueLib.densify(A) : valueLib.asValue(A);
      const bV = valueLib.isValue(b) ? valueLib.densify(b) : valueLib.asValue(b);
      return _linsolveLUValue(aV, bV);
    }
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('linsolve: A must be a non-empty square matrix');
    }
    return _linsolveLU(A, b);
  },
  // lower_cholesky(A): lower-triangular L with A = L L^T for symmetric
  // positive-definite A. Throws if A is not PD.
  lower_cholesky: (A: any): any => {
    if (valueLib.isDiagStored(A) && !A.im) {
      // A diagonal PD matrix is its own Cholesky structure: L = √diag,
      // still diagonal (lower-triangular). O(n), no factorization.
      const d = new Float64Array(A.data.length);
      for (let i = 0; i < d.length; i++) {
        if (!(A.data[i] > 0)) {
          throw new Error('lower_cholesky: matrix is not positive definite');
        }
        d[i] = Math.sqrt(A.data[i]);
      }
      return valueLib.diagMatrix(d);
    }
    if (valueLib.isValue(A)) {
      return _choleskyValue(valueLib.densify(A));
    }
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('lower_cholesky: argument must be a non-empty square matrix');
    }
    return _cholesky(A);
  },
  // row_gram(A) = A · A^†; col_gram(A) = A^† · A (spec §07 — adjoint /
  // conjugate-transpose, the Hermitian Gram). For real A the conj bit
  // is a numerical no-op so this is identical to the old transpose
  // form; for complex A the adjoint is required for the Gram to be
  // Hermitian. On Value input the matmul goes through value-ops.mul,
  // which folds the Klein-4 conj+swap tags at dispatch (no transpose
  // or conjugate materialisation). Useful for LKJ ↔ LKJCholesky
  // conversions and Gram-matrix priors.
  row_gram: (A: any) => {
    if (valueLib.isValue(A)) return valueOps.mul(A, valueLib.adjoint(A));
    return _matmul(A, ARITH_OPS.transpose(A));
  },
  col_gram: (A: any) => {
    if (valueLib.isValue(A)) return valueOps.mul(valueLib.adjoint(A), A);
    return _matmul(ARITH_OPS.transpose(A), A);
  },
  // array(data, size, dimorder) — n-D array from a flat data vector
  // per spec §07. size is an n-vector of positive dimensions;
  // dimorder is a permutation of [1..n] listing axes from slowest- to
  // fastest-varying as `data` is traversed. dimorder = [1,2,...,n]
  // is row-major (C order); reversed is column-major (Fortran).
  //
  // dimorder doesn't imply memory layout in FlatPPL — it only
  // specifies how the flat data is unpacked into the n-D shape. The
  // result is a nested JS array.
  array: (data: any, size: any, dimorder: any) => {
    // Spec §07 explicit rank-N constructor. Result is a shape-explicit
    // rank-N Value per the §2.1 contract (formerly a nested JS array,
    // which conflated rank-2 results with rank-1-of-rank-1 vector-of-
    // vectors).
    const sizeArr = valueLib.isValue(size) ? Array.from(size.data) : Array.from(size);
    const doArr = valueLib.isValue(dimorder) ? Array.from(dimorder.data)
                                              : Array.from(dimorder);
    const dataArr = valueLib.isValue(data) ? data.data
                  : (data && data.BYTES_PER_ELEMENT ? data : data);
    const n = sizeArr.length;
    if (doArr.length !== n) {
      throw new Error('array: dimorder length ' + doArr.length
        + ' must match size length ' + n);
    }
    let total = 1;
    for (let i = 0; i < n; i++) total *= ((sizeArr[i] as number) | 0);
    const dataLen = (dataArr as any).length;
    if (dataLen !== total) {
      throw new Error('array: prod(size) = ' + total
        + ' does not match data length ' + dataLen);
    }
    if (n === 0) return dataLen > 0 ? (dataArr as any)[0] : null;
    const shape = sizeArr.map((x: any) => (x | 0));
    // Fast path: dimorder is the identity (1, 2, ..., n) — data is
    // already row-major in shape order.
    let identity = true;
    for (let i = 0; i < n; i++) {
      if (((doArr[i] as number) | 0) !== i + 1) { identity = false; break; }
    }
    if (identity) {
      const out = new Float64Array(total);
      for (let k = 0; k < total; k++) out[k] = +(dataArr as any)[k];
      return { shape, data: out };
    }
    // Permuted dimorder — decode each linear-in-input index into the
    // output's row-major position.
    const stridesInDimorder = new Array(n);
    let strideAcc = 1;
    for (let i = n - 1; i >= 0; i--) {
      stridesInDimorder[i] = strideAcc;
      strideAcc *= shape[((doArr[i] as number) | 0) - 1];
    }
    // Row-major strides on the output shape.
    const outStrides = new Array(n);
    let oacc = 1;
    for (let i = n - 1; i >= 0; i--) { outStrides[i] = oacc; oacc *= shape[i]; }
    const coord = new Array(n);
    const out = new Float64Array(total);
    for (let k = 0; k < total; k++) {
      let rem = k;
      for (let i = 0; i < n; i++) {
        const axis = ((doArr[i] as number) | 0) - 1;
        const stride = stridesInDimorder[i];
        coord[axis] = Math.floor(rem / stride) % shape[axis];
        rem %= stride;
      }
      let outIdx = 0;
      for (let a = 0; a < n; a++) outIdx += coord[a] * outStrides[a];
      out[outIdx] = +(dataArr as any)[k];
    }
    return { shape, data: out };
  },
  // fill(x, size) — array of shape `size` filled with x. `size` is
  // either a positive integer (1-D shape) or a vector of positive
  // integers (multi-axis shape). Per spec §07; result is a shape-
  // explicit Value (engine-concepts §2.1).
  fill: (x: any, size: any) => {
    const dims = _sizeAsDims(size);
    const total = dims.reduce((a, b) => a * b, 1);
    const d = new Float64Array(total);
    const v = (x === true) ? 1 : (x === false) ? 0 : +x;
    d.fill(v);
    return { shape: dims, data: d };
  },
  zeros: (size: any) => {
    const dims = _sizeAsDims(size);
    const total = dims.reduce((a, b) => a * b, 1);
    return { shape: dims, data: new Float64Array(total) };
  },
  ones: (size: any) => {
    const dims = _sizeAsDims(size);
    const total = dims.reduce((a, b) => a * b, 1);
    const d = new Float64Array(total);
    d.fill(1);
    return { shape: dims, data: d };
  },
  // eye(n) — n × n identity matrix. Spec §07. Returns a structured
  // (diag-stored) Value.
  eye: (n: any) => {
    const k = n | 0;
    if (k <= 0) return { shape: [0, 0], data: new Float64Array(0) };
    const d = new Float64Array(k);
    d.fill(1);
    return valueLib.diagMatrix(d);
  },
  // onehot(i, n) — length-n basis vector with 1 at position i (1-based
  // per FlatPPL convention). Spec §07.
  onehot: (i: any, n: any) => {
    const idx = i | 0;
    const k = n | 0;
    if (k <= 0) return { shape: [0], data: new Float64Array(0) };
    if (idx < 1 || idx > k) {
      throw new Error('onehot: index ' + idx + ' out of range [1, ' + k + ']');
    }
    const data = new Float64Array(k);
    data[idx - 1] = 1;
    return { shape: [k], data: data };
  },
  // Scalar restrictors (spec §07). Identity at runtime; static typing
  // catches domain violations at type-check time when they're
  // discernible. The runtime versions check the obvious cases.
  boolean: (x: any) => {
    if (x === true || x === false) return x;
    if (x === 0) return false;
    if (x === 1) return true;
    throw new Error('boolean: value ' + x + ' is not a boolean');
  },
  integer: (x: any) => {
    if (Number.isInteger(x)) return x;
    throw new Error('integer: value ' + x + ' is not an integer');
  },
  // linspace(from, to, n) — endpoint-inclusive range of n real numbers
  // evenly spaced from `from` to `to`. n=1 returns [from]; both endpoints
  // are included exactly (not computed via accumulating step). Spec §07.
  // Result is a rank-1 Value (engine-concepts §2.1).
  linspace: (from: any, to: any, n: any) => {
    const k = n | 0;
    if (k <= 0) return { shape: [0], data: new Float64Array(0) };
    const data = new Float64Array(k);
    if (k === 1) { data[0] = +from; return { shape: [1], data }; }
    const lo = +from, hi = +to;
    for (let i = 0; i < k; i++) {
      // Use the parametric form (1−t)·lo + t·hi so the endpoints land
      // exactly on lo and hi without floating-point drift.
      const t = i / (k - 1);
      data[i] = (1 - t) * lo + t * hi;
    }
    return { shape: [k], data };
  },
  // extlinspace(from, to, n) — like linspace but with -inf and +inf
  // prepended/appended. Useful for overflow-bin definitions in binned
  // analyses. Spec §07.
  extlinspace: (from: any, to: any, n: any) => {
    const k = n | 0;
    if (k <= 0) {
      const d0 = new Float64Array(2);
      d0[0] = -Infinity; d0[1] = Infinity;
      return { shape: [2], data: d0 };
    }
    const data = new Float64Array(k + 2);
    data[0] = -Infinity;
    data[k + 1] = Infinity;
    if (k === 1) { data[1] = +from; return { shape: [k + 2], data }; }
    const lo = +from, hi = +to;
    for (let i = 0; i < k; i++) {
      const t = i / (k - 1);
      data[i + 1] = (1 - t) * lo + t * hi;
    }
    return { shape: [k + 2], data };
  },
  // partition(xs, spec) — split a vector into groups. spec may be a
  // positive integer (equal-size groups) or a vector of positive
  // integers (custom group sizes). Spec §07. Returns a vector of
  // sub-vectors (JS arrays of JS arrays).
  // partition(xs, spec) — split a vector into groups. Spec §07 result
  // is a "vector of vectors" — per spec §03 this is NOT a matrix. The
  // runtime rep is a JS array of rank-1 Value vectors.
  partition: (xs: any, spec: any) => {
    const src = valueLib.isValue(xs) ? xs.data : xs;
    const n = src.length;
    if (typeof spec === 'number') {
      const k = spec | 0;
      if (k <= 0) throw new Error('partition: group size must be positive, got ' + k);
      if (n % k !== 0) {
        throw new Error('partition: length ' + n + ' not divisible by group size ' + k);
      }
      const groups = n / k;
      const out = new Array(groups);
      for (let g = 0; g < groups; g++) {
        const d = new Float64Array(k);
        for (let i = 0; i < k; i++) d[i] = +src[g * k + i];
        out[g] = { shape: [k], data: d };
      }
      return out;
    }
    const specSrc = valueLib.isValue(spec) ? spec.data : spec;
    if (Array.isArray(specSrc) || (specSrc && specSrc.BYTES_PER_ELEMENT)) {
      let total = 0;
      for (let i = 0; i < specSrc.length; i++) total += (specSrc[i] | 0);
      if (total !== n) {
        throw new Error('partition: spec sums to ' + total + ' but vector length is ' + n);
      }
      const out = new Array(specSrc.length);
      let cursor = 0;
      for (let g = 0; g < specSrc.length; g++) {
        const sz = specSrc[g] | 0;
        const d = new Float64Array(sz);
        for (let i = 0; i < sz; i++) d[i] = +src[cursor + i];
        out[g] = { shape: [sz], data: d };
        cursor += sz;
      }
      return out;
    }
    throw new Error('partition: spec must be a positive integer or a vector of positive integers');
  },
  // reverse(xs) — reverse element order in a vector. Tables defer for
  // now (no canonical table runtime yet). Returns a rank-1 Value when
  // the input is a flat numeric vector.
  reverse: (xs: any) => {
    if (valueLib.isValue(xs) && xs.shape.length === 1) {
      const n = xs.shape[0];
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[n - 1 - i] = xs.data[i];
      return { shape: [n], data: out };
    }
    if (xs && xs.BYTES_PER_ELEMENT) {
      const n = xs.length;
      const out = new Float64Array(n);
      for (let i = 0; i < n; i++) out[n - 1 - i] = xs[i];
      return { shape: [n], data: out };
    }
    if (Array.isArray(xs)) {
      const out = new Array(xs.length);
      for (let i = 0; i < xs.length; i++) out[xs.length - 1 - i] = xs[i];
      return out;
    }
    throw new Error('reverse: argument must be a vector');
  },
  // tile(A, size) — repeat A along each axis. `size` is a positive
  // integer (rank-1 A) or a vector of positive integers (one per
  // axis, must match A's rank). Result is a rank-equal Value with
  // shape elementwise (A.shape * size). Spec §07.
  tile: (A: any, size: any) => {
    const aV = valueLib.isValue(A) ? valueLib.densify(A) : valueLib.asValue(A);
    const reps = _sizeAsDims(size);
    if (reps.length !== aV.shape.length) {
      throw new Error('tile: size length ' + reps.length
        + ' must match A.rank = ' + aV.shape.length);
    }
    const outShape = aV.shape.map((d: number, i: number) => d * reps[i]);
    const outLen = outShape.reduce((a: number, b: number) => a * b, 1);
    const out = new Float64Array(outLen);
    // Row-major strides for A and out.
    const aStrides = new Array(aV.shape.length);
    let acc = 1;
    for (let i = aV.shape.length - 1; i >= 0; i--) {
      aStrides[i] = acc; acc *= aV.shape[i];
    }
    const oStrides = new Array(outShape.length);
    acc = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
      oStrides[i] = acc; acc *= outShape[i];
    }
    // Walk the OUTPUT index in row-major order; for each coordinate,
    // its position in A is (oi mod A.shape[axis]).
    const coord = new Array(outShape.length).fill(0);
    for (let k = 0; k < outLen; k++) {
      let aIdx = 0;
      for (let ax = 0; ax < outShape.length; ax++) {
        aIdx += (coord[ax] % aV.shape[ax]) * aStrides[ax];
      }
      out[k] = aV.data[aIdx];
      // Increment coord (least-significant axis first).
      for (let ax = outShape.length - 1; ax >= 0; ax--) {
        coord[ax]++;
        if (coord[ax] < outShape[ax]) break;
        coord[ax] = 0;
      }
    }
    return { shape: outShape, data: out };
  },
  // splitblocks(A, blocksize) — split A into a nested array of
  // equal-shaped blocks. blocksize is a scalar (rank-1 A) or a
  // vector of positive integers (must divide A's shape elementwise).
  // The outer-shape is A.shape ⊘ blocksize; each inner block has
  // shape blocksize. Spec §07.
  splitblocks: (A: any, blocksize: any) => {
    const aV = valueLib.isValue(A) ? valueLib.densify(A) : valueLib.asValue(A);
    const bs = _sizeAsDims(blocksize);
    if (bs.length !== aV.shape.length) {
      throw new Error('splitblocks: blocksize length ' + bs.length
        + ' must match A.rank = ' + aV.shape.length);
    }
    const outerShape: number[] = [];
    for (let i = 0; i < bs.length; i++) {
      if (bs[i] <= 0 || aV.shape[i] % bs[i] !== 0) {
        throw new Error('splitblocks: axis ' + i + ' size ' + aV.shape[i]
          + ' is not divisible by blocksize ' + bs[i]);
      }
      outerShape.push(aV.shape[i] / bs[i]);
    }
    // Build nested-array structure: outer is rank=outerShape.length;
    // each leaf is a Value of shape `bs`. Outer is a JS array of
    // JS arrays of Values (matching the spec's "nested array of
    // arrays" wording — true vector-of-vectors per §03).
    const aStrides = new Array(aV.shape.length);
    let acc = 1;
    for (let i = aV.shape.length - 1; i >= 0; i--) {
      aStrides[i] = acc; acc *= aV.shape[i];
    }
    function buildBlock(blockCoord: number[]): any {
      const bsize = bs.reduce((a: number, b: number) => a * b, 1);
      const data = new Float64Array(bsize);
      const inner = new Array(bs.length).fill(0);
      const bStrides = new Array(bs.length);
      let ba = 1;
      for (let i = bs.length - 1; i >= 0; i--) { bStrides[i] = ba; ba *= bs[i]; }
      for (let k = 0; k < bsize; k++) {
        let aIdx = 0;
        for (let ax = 0; ax < bs.length; ax++) {
          aIdx += (blockCoord[ax] * bs[ax] + inner[ax]) * aStrides[ax];
        }
        data[k] = aV.data[aIdx];
        for (let ax = bs.length - 1; ax >= 0; ax--) {
          inner[ax]++;
          if (inner[ax] < bs[ax]) break;
          inner[ax] = 0;
        }
      }
      return { shape: bs.slice(), data };
    }
    function buildOuter(axis: number, blockCoord: number[]): any {
      if (axis === outerShape.length) return buildBlock(blockCoord);
      const out: any[] = [];
      for (let i = 0; i < outerShape[axis]; i++) {
        blockCoord[axis] = i;
        out.push(buildOuter(axis + 1, blockCoord));
      }
      return out;
    }
    return buildOuter(0, new Array(outerShape.length).fill(0));
  },
  // joinblocks(A) — inverse of splitblocks. A is a nested array of
  // equal-shape inner Values; result is a single Value whose shape
  // is the ELEMENTWISE PRODUCT of outer shape and inner shape (same
  // rank required). Spec §07.
  joinblocks: (A: any) => {
    // Derive outer + inner shapes by walking the nested structure.
    const outerShape: number[] = [];
    let cur: any = A;
    while (Array.isArray(cur)) {
      outerShape.push(cur.length);
      cur = cur[0];
    }
    if (!valueLib.isValue(cur) && !(cur && cur.BYTES_PER_ELEMENT)
        && !Array.isArray(cur)) {
      throw new Error('joinblocks: innermost element must be an array/Value, '
        + 'got ' + (typeof cur));
    }
    const innerV = valueLib.isValue(cur) ? cur : valueLib.asValue(cur);
    const innerShape = innerV.shape.slice();
    if (outerShape.length !== innerShape.length) {
      throw new Error('joinblocks: outer rank (' + outerShape.length
        + ') must equal inner rank (' + innerShape.length + ')');
    }
    // Result shape is the ELEMENTWISE product (spec §07).
    const outShape = outerShape.map((d: number, i: number) => d * innerShape[i]);
    const outLen = outShape.reduce((a: number, b: number) => a * b, 1);
    const out = new Float64Array(outLen);
    // Output row-major strides.
    const oStrides = new Array(outShape.length);
    let acc = 1;
    for (let i = outShape.length - 1; i >= 0; i--) {
      oStrides[i] = acc; acc *= outShape[i];
    }
    function fillBlock(node: any, outerIdx: number[]) {
      if (Array.isArray(node)) {
        for (let i = 0; i < node.length; i++) {
          fillBlock(node[i], outerIdx.concat([i]));
        }
        return;
      }
      const block = valueLib.isValue(node) ? node : valueLib.asValue(node);
      if (block.shape.length !== innerShape.length
          || !block.shape.every((d: number, i: number) => d === innerShape[i])) {
        throw new Error('joinblocks: inner blocks must all share the same '
          + 'shape; got ' + JSON.stringify(block.shape)
          + ' vs expected ' + JSON.stringify(innerShape));
      }
      // Result entry at logical coordinate
      //   (outerIdx[ax] * innerShape[ax] + innerCoord[ax])
      // for each axis. Walk every innerCoord and write into out.
      const innerSize = innerShape.reduce((a: number, b: number) => a * b, 1);
      const ic = new Array(innerShape.length).fill(0);
      for (let k = 0; k < innerSize; k++) {
        let pos = 0;
        for (let ax = 0; ax < outShape.length; ax++) {
          pos += (outerIdx[ax] * innerShape[ax] + ic[ax]) * oStrides[ax];
        }
        out[pos] = block.data[k];
        for (let ax = innerShape.length - 1; ax >= 0; ax--) {
          ic[ax]++;
          if (ic[ax] < innerShape[ax]) break;
          ic[ax] = 0;
        }
      }
    }
    fillBlock(A, []);
    return { shape: outShape, data: out };
  },
  // blockdiagmat(mats) — block-diagonal matrix from a vector of
  // matrices. All off-diagonal blocks are zero. Result rows/cols
  // are the sums of input rows/cols. Spec §07.
  blockdiagmat: (mats: any) => {
    // Accept a JS array of matrices (vector-of-matrices per spec §07
    // is vector-of-Value, the post-§2.1 rep) OR an empty rank-1 Value
    // (vector(...) with no args).
    if (valueLib.isValue(mats)) {
      if (mats.shape.length === 1 && mats.shape[0] === 0) {
        return { shape: [0, 0], data: new Float64Array(0) };
      }
      throw new Error('blockdiagmat: argument must be a vector of matrices');
    }
    if (!Array.isArray(mats)) {
      throw new Error('blockdiagmat: argument must be a vector of matrices');
    }
    if (mats.length === 0) {
      return { shape: [0, 0], data: new Float64Array(0) };
    }
    const blocks: any[] = mats.map((m: any) => {
      const v = valueLib.isValue(m) ? valueLib.densify(m) : valueLib.asValue(m);
      if (v.shape.length !== 2) {
        throw new Error('blockdiagmat: each block must be a matrix, got shape='
          + JSON.stringify(v.shape));
      }
      return v;
    });
    let totalRows = 0, totalCols = 0;
    for (const b of blocks) { totalRows += b.shape[0]; totalCols += b.shape[1]; }
    const out = new Float64Array(totalRows * totalCols);
    let r0 = 0, c0 = 0;
    for (const b of blocks) {
      const [m, n] = b.shape;
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
          out[(r0 + i) * totalCols + (c0 + j)] = b.data[i * n + j];
        }
      }
      r0 += m; c0 += n;
    }
    return { shape: [totalRows, totalCols], data: out };
  },
  // bandedmat(v, rows) — matrix with `rows` rows where row i contains
  // v shifted right by i (zeros elsewhere). Result is rows × (rows +
  // length(v) - 1). Spec §07.
  bandedmat: (v: any, rows: any) => {
    const vV = valueLib.isValue(v) ? v : valueLib.asValue(v);
    if (vV.shape.length !== 1) {
      throw new Error('bandedmat: v must be a rank-1 vector');
    }
    const k = vV.shape[0];
    const m = (+rows) | 0;
    if (m <= 0) throw new Error('bandedmat: rows must be a positive integer, got ' + rows);
    const cols = m + k - 1;
    const out = new Float64Array(m * cols);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < k; j++) {
        out[i * cols + i + j] = vV.data[j];
      }
    }
    return { shape: [m, cols], data: out };
  },
  // conv(v, kernel) — 1-D valid-mode convolution. Spec §07:
  //   result_i = ⟨ v[i : i + K - 1], reverse(kernel) ⟩
  // No padding / striding / windowing; requires lengthof(kernel)
  // ≤ lengthof(v). Result length = lengthof(v) − K + 1.
  conv: (v: any, kernel: any) => {
    const vV = valueLib.isValue(v) ? v : valueLib.asValue(v);
    const kV = valueLib.isValue(kernel) ? kernel : valueLib.asValue(kernel);
    if (vV.shape.length !== 1 || kV.shape.length !== 1) {
      throw new Error('conv: both arguments must be rank-1 vectors');
    }
    const n = vV.shape[0], K = kV.shape[0];
    if (K > n) {
      throw new Error('conv: kernel length (' + K + ') > vector length (' + n + ')');
    }
    const outLen = n - K + 1;
    const out = new Float64Array(outLen);
    for (let i = 0; i < outLen; i++) {
      let s = 0;
      // Σ v[i+j] · kernel[K-1-j]  for j=0..K-1
      for (let j = 0; j < K; j++) s += vV.data[i + j] * kV.data[K - 1 - j];
      out[i] = s;
    }
    return { shape: [outLen], data: out };
  },
  // crosscorr(v, kernel) — 1-D valid-mode cross-correlation. Same
  // shape as `conv` but kernel is NOT reversed. Spec §07.
  crosscorr: (v: any, kernel: any) => {
    const vV = valueLib.isValue(v) ? v : valueLib.asValue(v);
    const kV = valueLib.isValue(kernel) ? kernel : valueLib.asValue(kernel);
    if (vV.shape.length !== 1 || kV.shape.length !== 1) {
      throw new Error('crosscorr: both arguments must be rank-1 vectors');
    }
    const n = vV.shape[0], K = kV.shape[0];
    if (K > n) {
      throw new Error('crosscorr: kernel length (' + K + ') > vector length (' + n + ')');
    }
    const outLen = n - K + 1;
    const out = new Float64Array(outLen);
    for (let i = 0; i < outLen; i++) {
      let s = 0;
      for (let j = 0; j < K; j++) s += vV.data[i + j] * kV.data[j];
      out[i] = s;
    }
    return { shape: [outLen], data: out };
  },
  // diag(A, k=0) — extract the k-th diagonal of a matrix A as a
  // vector. k=0 main diagonal, k>0 super-diagonals, k<0 sub-
  // diagonals. Spec §07.
  diag: (A: any, k?: any) => {
    const aV = valueLib.isValue(A) ? valueLib.densify(A) : valueLib.asValue(A);
    if (aV.shape.length !== 2) {
      throw new Error('diag: argument must be a rank-2 matrix, got shape='
        + JSON.stringify(aV.shape));
    }
    const kk = (k == null) ? 0 : (k | 0);
    const [m, n] = aV.shape;
    // Diagonal k starts at (i, j) = (max(-k, 0), max(k, 0)) and runs
    // while both i < m and j < n.
    const iStart = kk < 0 ? -kk : 0;
    const jStart = kk > 0 ?  kk : 0;
    const len = Math.max(0, Math.min(m - iStart, n - jStart));
    const out = new Float64Array(len);
    for (let t = 0; t < len; t++) {
      out[t] = aV.data[(iStart + t) * n + (jStart + t)];
    }
    return { shape: [len], data: out };
  },
  // addaxes(A, n_leading, n_trailing) — reshape A by inserting
  // n_leading singular (size-1) axes before A's axes and n_trailing
  // after (spec §07). Size-1 axes leave numel and the flat buffer
  // unchanged, so this is a ZERO-COPY shape rewrite on the Value —
  // the same lazy-metadata pattern as the Klein-4 transpose tag.
  // It is the explicit, layout-agnostic alternative to implicit
  // NumPy/Julia axis insertion in `broadcast` (spec commit 87c9be1):
  // addaxes(b,1,0) gives NumPy-style alignment, addaxes(b,0,1)
  // Julia-style — the engine never picks for the user.
  addaxes: (A: any, nLeading: any, nTrailing: any) => {
    const _intAxis = (x: any, what: any) => {
      const n = (x && Array.isArray(x.shape) && x.data) ? x.data[0] : +x;
      if (!Number.isInteger(n) || n < 0) {
        throw new Error('addaxes: ' + what +
          ' must be a non-negative integer, got ' + x);
      }
      return n;
    };
    const nl = _intAxis(nLeading, 'n_leading');
    const nt = _intAxis(nTrailing, 'n_trailing');
    const v = valueLib.asValue(A);
    if (valueLib.isTransposeView(v)) {
      // addaxes is a logical-shape rewrite; composing it with a
      // pending transpose/adjoint tag would make "last two axes"
      // ambiguous. Apply addaxes to a non-transposed array.
      throw new Error('addaxes: argument is a transposed/adjoint view; '
        + 'addaxes operates on logical (non-transposed) array shape');
    }
    const shape = new Array(nl + v.shape.length + nt);
    let p = 0;
    for (let i = 0; i < nl; i++) shape[p++] = 1;
    for (let i = 0; i < v.shape.length; i++) shape[p++] = v.shape[i];
    for (let i = 0; i < nt; i++) shape[p++] = 1;
    const out: any = { shape: shape, data: v.data };     // shared buffer
    if (v.dtype) out.dtype = v.dtype;
    if (v.im instanceof Float64Array) out.im = v.im;     // complex rides along
    return out;
  },
  cat: (...xs: any[]) => {
    if (xs.length === 0) return { shape: [0], data: new Float64Array(0) };
    const first = xs[0];
    // cat(scalar, scalar, ...) → rank-1 Value (spec §07).
    if (typeof first === 'number' || typeof first === 'boolean') {
      const data = new Float64Array(xs.length);
      for (let i = 0; i < xs.length; i++) {
        data[i] = xs[i] === true ? 1 : xs[i] === false ? 0 : +xs[i];
      }
      return { shape: [xs.length], data };
    }
    // cat(vector, vector, ...) → rank-1 Value (concatenation along
    // the only axis). Accept any indexable (Value / Float64Array /
    // JS array of numbers).
    if (valueLib.isValue(first)
        || (first && first.BYTES_PER_ELEMENT !== undefined)
        || Array.isArray(first)) {
      let total = 0;
      for (let j = 0; j < xs.length; j++) {
        const v = xs[j];
        total += valueLib.isValue(v) ? v.data.length : v.length;
      }
      const out = new Float64Array(total);
      let pos = 0;
      for (let j = 0; j < xs.length; j++) {
        const v = xs[j];
        const src = valueLib.isValue(v) ? v.data : v;
        for (let i = 0; i < src.length; i++) out[pos++] = +src[i];
      }
      return { shape: [total], data: out };
    }
    if (first && typeof first === 'object') {
      const out: Record<string, any> = {};
      for (let j = 0; j < xs.length; j++) {
        const r = xs[j];
        for (const k in r) {
          if (!Object.prototype.hasOwnProperty.call(r, k)) continue;
          if (k in out) {
            throw new Error("cat: duplicate field '" + k + "'");
          }
          out[k] = r[k];
        }
      }
      return out;
    }
    throw new Error('cat: unsupported argument shape (got '
      + (typeof first) + ')');
  },
  // Reductions over an array. Spec §07:
  //   sum / mean / prod  — real or complex arrays (any rank)
  //   var / std          — real arrays (any rank)
  //   maximum / minimum  — real arrays (any rank)
  //   lengthof           — vectors, tables
  //   sizeof             — vectors, arrays (any rank)
  // For complex Values the reduction operates on the parallel re/im
  // buffers (applying the conjugation bit lazily where needed). All
  // reductions flatten over EVERY entry, so multi-dim inputs reduce
  // to a single scalar.
  sum: (a: any) => {
    if (valueLib.isComplexValue(a)) {
      const cplx = valueLib.readComplex(a);
      let sR = 0, sI = 0;
      for (let i = 0; i < cplx.re.length; i++) {
        sR += cplx.re[i]; sI += cplx.im[i];
      }
      return { re: sR, im: sI };
    }
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return s;
  },
  mean: (a: any) => {
    if (valueLib.isComplexValue(a)) {
      const cplx = valueLib.readComplex(a);
      const n = cplx.re.length;
      let sR = 0, sI = 0;
      for (let i = 0; i < n; i++) { sR += cplx.re[i]; sI += cplx.im[i]; }
      return { re: sR / n, im: sI / n };
    }
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return s / arr.length;
  },
  prod: (a: any) => {
    if (valueLib.isComplexValue(a)) {
      const cplx = valueLib.readComplex(a);
      let pR = 1, pI = 0;
      for (let i = 0; i < cplx.re.length; i++) {
        const r = cplx.re[i], j = cplx.im[i];
        const nR = pR * r - pI * j;
        const nI = pR * j + pI * r;
        pR = nR; pI = nI;
      }
      return { re: pR, im: pI };
    }
    const arr = _arrLike(a);
    let p = 1;
    for (let i = 0; i < arr.length; i++) p *= arr[i];
    return p;
  },
  // lengthof(x) per spec §07: "number of elements (vector) / rows
  // (table)". Vectors are rank-1 arrays — for shape-explicit Values
  // we use shape[0]. Tables (record-of-columns) aren't yet first-class
  // in the engine runtime; if a record-like value reaches here we
  // return the first field's length, matching the spec's "rows" notion.
  lengthof: (a: any) => {
    if (valueLib.isValue(a)) return a.shape.length === 0 ? 1 : a.shape[0];
    if (a && !Array.isArray(a) && typeof a === 'object'
        && a.BYTES_PER_ELEMENT === undefined) {
      // Record-of-columns ⇒ table. Length is the row count = first column's length.
      for (const k in a) {
        if (!Object.prototype.hasOwnProperty.call(a, k)) continue;
        const col = a[k];
        if (valueLib.isValue(col)) return col.shape[0];
        if (col && typeof col.length === 'number') return col.length;
      }
      return 0;
    }
    return _arrLike(a).length;
  },
  // sizeof returns a length-rank vector of the array's per-axis
  // dimensions. For 1-D vectors/typed arrays this is `[lengthof(a)]`;
  // for shape-tagged Values, the dims come from the shape. Spec §07.
  // Returned as a rank-1 Value per the engine-concepts §2.1 contract.
  sizeof: (a: any) => {
    function _packDims(dims: number[]): any {
      const out = new Float64Array(dims.length);
      for (let i = 0; i < dims.length; i++) out[i] = dims[i];
      return { shape: [dims.length], data: out };
    }
    if (valueLib.isValue(a)) return _packDims(a.shape);
    if (a == null) return _packDims([0]);
    if (typeof a === 'number') return _packDims([]);
    if (a.BYTES_PER_ELEMENT && typeof a.length === 'number') {
      return _packDims([a.length]);
    }
    if (Array.isArray(a)) {
      const dims: number[] = [];
      let cur: any = a;
      while (Array.isArray(cur)) {
        dims.push(cur.length);
        cur = cur[0];
      }
      return _packDims(dims);
    }
    return _packDims([]);
  },
  maximum: (a: any) => {
    const arr = _arrLike(a);
    let m = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    return m;
  },
  minimum: (a: any) => {
    const arr = _arrLike(a);
    let m = Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
    return m;
  },
  // Sample variance per spec §07 §sec:functions (line: `var | xs |
  // (1/(n-1)) Σ(xᵢ - x̄)²`). Bessel-corrected — divisor is n-1, not
  // n. n ≤ 1 has no sample variance; return 0 (matches the existing
  // n=0 sentinel and avoids a NaN from 0/0).
  var: (a: any) => {
    const arr = _arrLike(a);
    const n = arr.length;
    if (n <= 1) return 0;
    let s = 0;
    for (let i = 0; i < n; i++) s += arr[i];
    const mu = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = arr[i] - mu; v += d * d; }
    return v / (n - 1);
  },
  // std = sqrt(var). Per spec §07 §sec:functions
  // (`std | xs | √var(x)`) so the Bessel correction in `var` flows
  // through.
  std: (a: any) => {
    const arr = _arrLike(a);
    const n = arr.length;
    if (n <= 1) return 0;
    let s = 0;
    for (let i = 0; i < n; i++) s += arr[i];
    const mu = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = arr[i] - mu; v += d * d; }
    return Math.sqrt(v / (n - 1));
  },
  // cumsum / cumprod — per spec §07, rank-1 results. Returned as
  // shape-explicit Values per engine-concepts §2.1.
  cumsum: (a: any) => {
    const arr = _arrLike(a);
    const n = arr.length;
    const out = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { s += arr[i]; out[i] = s; }
    return { shape: [n], data: out };
  },
  cumprod: (a: any) => {
    const arr = _arrLike(a);
    const n = arr.length;
    const out = new Float64Array(n);
    let p = 1;
    for (let i = 0; i < n; i++) { p *= arr[i]; out[i] = p; }
    return { shape: [n], data: out };
  },
  // Norms and normalization (spec §07). All take a single vector
  // argument. Numerically stable forms — logsumexp uses the standard
  // shift-by-max trick so exp doesn't overflow on large entries.
  l1norm: (a: any) => {
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i]);
    return s;
  },
  l2norm: (a: any) => {
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
    return Math.sqrt(s);
  },
  // l1unit / l2unit — per spec §07, rank-1 normalized vectors.
  l1unit: (a: any) => {
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i]);
    if (s === 0) throw new Error('l1unit: zero-norm vector has no unit form');
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] / s;
    return { shape: [arr.length], data: out };
  },
  l2unit: (a: any) => {
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
    if (s === 0) throw new Error('l2unit: zero-norm vector has no unit form');
    const r = Math.sqrt(s);
    const out = new Float64Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] / r;
    return { shape: [arr.length], data: out };
  },
  logsumexp: (a: any) => {
    const arr = _arrLike(a);
    const n = arr.length;
    if (n === 0) return -Infinity;
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
    if (!Number.isFinite(m)) return m;   // all -Inf → result is -Inf
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.exp(arr[i] - m);
    return m + Math.log(s);
  },
  // softmax / logsoftmax — per spec §07, rank-1 probability /
  // log-probability vectors. Numerically stable via the standard
  // shift-by-max trick.
  softmax: (a: any) => {
    const arr = _arrLike(a);
    const n = arr.length;
    if (n === 0) return { shape: [0], data: new Float64Array(0) };
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
    if (!Number.isFinite(m)) {
      throw new Error('softmax: all-(-Infinity) input is undefined');
    }
    const exps = new Float64Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { exps[i] = Math.exp(arr[i] - m); s += exps[i]; }
    for (let i = 0; i < n; i++) exps[i] /= s;
    return { shape: [n], data: exps };
  },
  logsoftmax: (a: any) => {
    const arr = _arrLike(a);
    const n = arr.length;
    if (n === 0) return { shape: [0], data: new Float64Array(0) };
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
    if (!Number.isFinite(m)) {
      throw new Error('logsoftmax: all-(-Infinity) input is undefined');
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.exp(arr[i] - m);
    const lse = m + Math.log(s);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = arr[i] - lse;
    return { shape: [n], data: out };
  },
};

// _arrLike: unwrap Value to its underlying Float64Array data, or pass
// through Float64Array / JS array. Used by reductions (sum, mean,
// l1norm, etc.) so they handle Value inputs uniformly without
// branching on every loop iteration.
function _arrLike(v: any) {
  if (valueLib.isValue(v)) return v.data;
  return v;
}

// _sizeAsDims: normalize a `size` argument (per spec §07: positive int
// or vector of positive ints) to a JS number[] of axis lengths.
function _sizeAsDims(size: any): number[] {
  if (typeof size === 'number') {
    if (!Number.isInteger(size) || size < 0) {
      throw new Error('size must be a non-negative integer, got ' + size);
    }
    return [size];
  }
  let arr: any = size;
  if (valueLib.isValue(arr)) arr = arr.data;
  if (arr && arr.BYTES_PER_ELEMENT && typeof arr.length === 'number') {
    arr = Array.from(arr as ArrayLike<number>);
  }
  if (!Array.isArray(arr)) {
    throw new Error('size must be an integer or vector of integers');
  }
  const dims: number[] = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const n = arr[i];
    if (!Number.isInteger(n) || n < 0) {
      throw new Error('size: axis ' + (i + 1) + ' is not a non-negative integer (got ' + n + ')');
    }
    dims[i] = n as number;
  }
  return dims;
}

// _buildFilled: nested JS array of shape `dims` with every leaf set to
// `value`. 1-D case yields a flat JS array.
function _buildFilled(dims: number[], value: any): any {
  if (dims.length === 0) return value;
  const n = dims[0];
  const rest = dims.slice(1);
  const out = new Array(n);
  if (rest.length === 0) {
    for (let i = 0; i < n; i++) out[i] = value;
  } else {
    for (let i = 0; i < n; i++) out[i] = _buildFilled(rest, value);
  }
  return out;
}

// =====================================================================
// Batched scalar-arithmetic: ARITH_OPS_N (used by evaluateExprN) is
// generated from this arity table + ARITH_OPS via broadcast helpers.
// Each scalar primitive is defined ONCE (its lambda lives in ARITH_OPS);
// the batched form is mechanically derived.
//
// This is the foundation for batched IR evaluation across the engine:
// worker.evaluateN, density.applyAtomScalar, and any future per-atom
// path dispatch through ARITH_OPS_N. Single-point evaluateExpr keeps
// using ARITH_OPS directly (zero allocation in the scalar-only path).
//
// Architectural note: each batched dispatch goes through ONE indirection
// (the broadcast helper). Future backends (TF.js, WASM SIMD, …) swap
// the broadcast helpers and the leaf logpdf/sample primitives; the
// walker layer stays identical. See "JS-TF backend" notes in TODO.
// =====================================================================

// Which ARITH_OPS entries are scalar-in-scalar-out — eligible for
// batched broadcast through Float64Array. Listed by op name → arity.
// The actual scalar implementation lives in ARITH_OPS above; this table
// drives ARITH_OPS_N construction below.
const _SCALAR_PRIM_ARITY = {
  // Arithmetic. `divide` is the spec §07 function-form of `/` (passed
  // to broadcast/reduce/scan); same scalar semantics as `div`, so it
  // belongs on the batched-broadcast path. Without an entry here it
  // fell to the slow per-atom fallback (real) and broke outright for
  // complex (the fallback can't pack {re,im}).
  add: 2, sub: 2, mul: 2, div: 2, divide: 2, mod: 2, neg: 1, pos: 1, pow: 2,
  // Elementary math
  abs: 1, abs2: 1, exp: 1, log: 1, log10: 1, log1p: 1, expm1: 1, sqrt: 1,
  sin: 1, cos: 1, tan: 1,
  asin: 1, acos: 1, atan: 1, atan2: 2,
  sinh: 1, cosh: 1, tanh: 1,
  asinh: 1, acosh: 1, atanh: 1,
  floor: 1, ceil: 1, round: 1,
  // Pairwise reductions
  min: 2, max: 2,
  // Special functions & link functions
  gamma: 1, loggamma: 1,
  logit: 1, invlogit: 1, probit: 1, invprobit: 1,
  // Comparison
  lt: 2, le: 2, gt: 2, ge: 2, equal: 2, unequal: 2,
  // Predicates
  isfinite: 1, isinf: 1, isnan: 1, iszero: 1,
  // Logic + conditional
  land: 2, lor: 2, lxor: 2, lnot: 1, ifelse: 3,
  // Scalar restrictors
  boolean: 1, integer: 1,
};

// Broadcast helpers — atom-major Float64Array semantics. Each takes a
// scalar function and per-atom-or-scalar inputs; returns either a
// scalar (when ALL inputs are scalar — the single-point fast path with
// no allocation) or Float64Array(N) (when any input is batched).
//
// Polymorphic on Value inputs (
// see TODO-flatppl-js.md and engine/value.js). Each input may be:
//
//   - a bare JS number / boolean              (atom-indep scalar; legacy)
//   - a bare Float64Array of length N         (atom-batched scalar; legacy)
//   - a Value with shape=[]                   (atom-indep scalar; Value form)
//   - a Value with shape=[N]                  (atom-batched scalar; Value form)
//
// "Same kind as inputs" output semantics: when ANY input is a Value
// the result is wrapped as a Value (scalar(out) or batchedScalar(out));
// when all inputs are bare primitives (number / Float64Array) the
// output stays bare. This preserves zero-allocation behaviour for
// single-point evaluation while letting Value-aware callers thread
// shape information through.
//
// `isBatch(v, N)` recognises BOTH bare Float64Array of length N AND
// Value shape=[N]. Anything else (number, Value shape=[], JS array,
// nested array, record) is treated as atom-independent and broadcast
// uniformly.
//
// These three helpers cover every op in _SCALAR_PRIM_ARITY (arities 1,
// 2, 3). Future variadic / N-ary scalar reductions would extend this
// set; non-scalar ops use perAtomDispatch in evaluateExprN below.

const isValueObj = valueLib.isValue;

function isBatch(v: any, N: any) {
  if (v == null) return false;
  // Bare Float64Array of length N (legacy path; also catches any
  // typed array with matching length — same as before).
  if (v.BYTES_PER_ELEMENT !== undefined
      && typeof v.length === 'number' && v.length === N) {
    return true;
  }
  // Value with shape=[N].
  if (isValueObj(v)
      && v.shape.length === 1 && v.shape[0] === N) {
    return true;
  }
  return false;
}

// Underlying Float64Array for a batched input (either bare or wrapped).
function _batchData(v: any) {
  return v instanceof Float64Array ? v : v.data;
}

// JS-number view of an atom-indep scalar (either bare or shape=[] Value).
// Bare arrays / non-batched arrays return unchanged — the scalar fn
// receives them as-is (legacy behaviour: garbage-in-garbage-out for
// shape mismatches at the scalar layer).
function _scalarVal(v: any) {
  if (isValueObj(v) && v.shape.length === 0) return v.data[0];
  return v;
}

// Any input is a Value? Triggers Value-wrapped output.
function _anyValue1(a: any)       { return isValueObj(a); }
function _anyValue2(a: any, b: any)    { return isValueObj(a) || isValueObj(b); }
function _anyValue3(a: any, b: any, c: any) { return isValueObj(a) || isValueObj(b) || isValueObj(c); }

function broadcast1(fn: any, a: any, N: any) {
  const wantValue = _anyValue1(a);
  if (!isBatch(a, N)) {
    const r = fn(_scalarVal(a));
    return wantValue ? valueLib.scalar(r) : r;
  }
  const ad = _batchData(a);
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) out[i] = +fn(ad[i]);
  return wantValue ? valueLib.batchedScalar(out) : out;
}

function broadcast2(fn: any, a: any, b: any, N: any) {
  const aB = isBatch(a, N), bB = isBatch(b, N);
  const wantValue = _anyValue2(a, b);
  if (!aB && !bB) {
    const r = fn(_scalarVal(a), _scalarVal(b));
    return wantValue ? valueLib.scalar(r) : r;
  }
  const out = new Float64Array(N);
  if (aB && bB) {
    const ad = _batchData(a), bd = _batchData(b);
    for (let i = 0; i < N; i++) out[i] = +fn(ad[i], bd[i]);
  } else if (aB) {
    const ad = _batchData(a), bs = _scalarVal(b);
    for (let i = 0; i < N; i++) out[i] = +fn(ad[i], bs);
  } else {
    const as = _scalarVal(a), bd = _batchData(b);
    for (let i = 0; i < N; i++) out[i] = +fn(as, bd[i]);
  }
  return wantValue ? valueLib.batchedScalar(out) : out;
}

function broadcast3(fn: any, a: any, b: any, c: any, N: any) {
  const aB = isBatch(a, N), bB = isBatch(b, N), cB = isBatch(c, N);
  const wantValue = _anyValue3(a, b, c);
  if (!aB && !bB && !cB) {
    const r = fn(_scalarVal(a), _scalarVal(b), _scalarVal(c));
    return wantValue ? valueLib.scalar(r) : r;
  }
  const ad = aB ? _batchData(a) : null;
  const bd = bB ? _batchData(b) : null;
  const cd = cB ? _batchData(c) : null;
  const as = aB ? null : _scalarVal(a);
  const bs = bB ? null : _scalarVal(b);
  const cs = cB ? null : _scalarVal(c);
  const out = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = +fn(aB ? ad[i] : as, bB ? bd[i] : bs, cB ? cd[i] : cs);
  }
  return wantValue ? valueLib.batchedScalar(out) : out;
}

// Build ARITH_OPS_N from the arity table. Each entry is a function
// `(args: Array, N: number) → Float64Array(N) | scalar`. Returns scalar
// when all inputs are atom-independent (the count=1 fast path).
const ARITH_OPS_N: any = {};
for (const op of Object.keys(_SCALAR_PRIM_ARITY)) {
  const arity = (_SCALAR_PRIM_ARITY as any)[op];
  const fn = (ARITH_OPS as any)[op];
  if (typeof fn !== 'function') {
    throw new Error(`ARITH_OPS_N: scalar primitive '${op}' has no ARITH_OPS entry`);
  }
  if (arity === 1) ARITH_OPS_N[op] = (args: any, N: any) => broadcast1(fn, args[0], N);
  else if (arity === 2) ARITH_OPS_N[op] = (args: any, N: any) => broadcast2(fn, args[0], args[1], N);
  else if (arity === 3) ARITH_OPS_N[op] = (args: any, N: any) => broadcast3(fn, args[0], args[1], args[2], N);
  else throw new Error(`ARITH_OPS_N: unsupported arity ${arity} for '${op}'`);
}

// Shape-aware mul / add / sub / neg take precedence over
// the broadcast{1,2}-based dispatch when any operand carries an
// intrinsic vector/matrix shape (Value with rank ≥ 1 whose leading
// dim isn't the atom count). Bare scalars and batched scalars
// (shape=[N]) stay on the scalar broadcast path.
//
// Atom-batched cross: when either operand has a
// leading N-sized axis with rank ≥ 2 (e.g. shape=[N, k] per-atom
// vector, shape=[N, m, n] per-atom matrix), route through the …N
// variants (`valueOps.mulN`, `valueOps.addN`, etc.) which dispatch
// per-atom shapes correctly. `_isShapeRich` (rank ≥ 1, leading dim ≠ N)
// excludes batched scalars (shape=[N]) which stay on broadcast2; an
// additional `_hasAtomAxis` check picks up shape=[N, k…] inputs.
function _shapeAwareCandidate(v: any, N: any) {
  if (!valueLib.isValue(v)) return false;
  const r = v.shape.length;
  if (r === 0) return false;
  if (r === 1 && v.shape[0] === N) return false;  // batched scalar
  return true;  // shape-rich OR atom-batched non-scalar
}
function _wrapShapeAwareBinopN(opName: any, opNameN: any) {
  const fallback = ARITH_OPS_N[opName];
  return (args: any, N: any) => {
    const a = args[0], b = args[1];
    if (_shapeAwareCandidate(a, N) || _shapeAwareCandidate(b, N)) {
      return valueOps[opNameN](valueLib.asValue(a), valueLib.asValue(b), N);
    }
    return fallback(args, N);
  };
}
ARITH_OPS_N.mul = _wrapShapeAwareBinopN('mul', 'mulN');
ARITH_OPS_N.add = _wrapShapeAwareBinopN('add', 'addN');
ARITH_OPS_N.sub = _wrapShapeAwareBinopN('sub', 'subN');
const _negBroadcast = ARITH_OPS_N.neg;
ARITH_OPS_N.neg = (args: any, N: any) => {
  const a = args[0];
  if (_shapeAwareCandidate(a, N)) return valueOps.negN(valueLib.asValue(a), N);
  return _negBroadcast(args, N);
};

// =====================================================================
// Complex batched broadcast (planar re/im, shape=[] or shape=[N])
// =====================================================================
//
// broadcast{1,2,3} coerce per-atom results with `+fn(...)`, which turns
// a complex {re, im} into NaN. Complex *shape-rich* values (shape=[N,k]
// per-atom vectors/matrices) already route through the shape-aware
// value-ops path (complex-aware since chunk 2). The remaining gap is:
//
//   - complex *batched scalars* (shape=[N] of one complex per atom), and
//   - the complex constructors / accessors (complex / real / imag /
//     conj / cis) which never had ARITH_OPS_N entries at all (atom-
//     dependent complex previously threw, per TODO §03).
//
// _cxBroadcast runs the EXISTING scalar ARITH_OPS primitive per atom
// (those already dispatch on {re, im}) and writes into two parallel
// Float64Arrays — the planar layout makes the batched form a pure
// re-pack of the scalar op, exactly as the design intends. Each arg is
// presented to the primitive in its natural scalar form (a JS number
// for real inputs, a {re, im} object for complex), identical to the
// atom-indep evaluateExpr path — so the primitives need no batched
// variant. Result type is detected from the primitive's output
// (number ⇒ real Value, {re, im} ⇒ complex Value).

// Does the complex path apply? True if any arg is complex (scalar
// {re, im} or a complex Value). Constructors (complex / cis) force the
// complex path explicitly at their registration site, since their
// inputs are real but the output is complex.
function _cxInPlay(args: any) {
  for (let i = 0; i < args.length; i++) {
    const v = args[i];
    if (_isComplex(v) || valueLib.isComplexValue(v)) return true;
  }
  return false;
}

// Per-arg accessor: returns a function i → (number | {re, im}) plus an
// `atomBatched` flag. Mirrors isBatch/_scalarVal/_batchData but is
// complex-aware and rejects shape-rich complex (handled elsewhere).
function _cxArgAccessor(v: any, N: any) {
  if (_isComplex(v)) return { batched: false, at: () => v };
  if (valueLib.isComplexValue(v)) {
    if (v.shape.length === 0) {
      const c = valueLib.readComplex(v);
      const z = { re: c.re[0], im: c.im[0] };
      return { batched: false, at: () => z };
    }
    if (v.shape.length === 1 && v.shape[0] === N) {
      const c = valueLib.readComplex(v);   // resolves conj once
      return { batched: true, at: (i: any) => ({ re: c.re[i], im: c.im[i] }) };
    }
    // Shape-rich complex is handled by _cxElementwise before any
    // accessor is built; reaching here would be a dispatch bug.
    throw new Error(
      '_cxArgAccessor: shape-rich complex Value (shape=[' +
      v.shape.join(',') + ']) must be routed through _cxElementwise');
  }
  if (valueLib.isValue(v)) {
    if (v.shape.length === 0) { const s = v.data[0]; return { batched: false, at: () => s }; }
    if (v.shape.length === 1 && v.shape[0] === N) {
      const d = v.data; return { batched: true, at: (i: any) => d[i] };
    }
    const s0 = v.data[0]; return { batched: false, at: () => s0 };
  }
  if (v instanceof Float64Array && v.length === N) {
    return { batched: true, at: (i: any) => v[i] };
  }
  const s = (typeof v === 'boolean') ? (v ? 1 : 0) : +v;
  return { batched: false, at: () => s };
}

// Shape-rich complex/real elementwise: applies the scalar primitive to
// every element of a per-atom vector/matrix (shape=[N, k…] or an
// atom-indep vector shape=[k>1]). Used for complex unary ops (abs2 /
// exp / log / sqrt / real / imag / conj / cis / pos) and the complex
// constructor over real vectors — the "free reshaping" ops that
// shouldn't be limited to scalars. Scalar args broadcast (length-1
// buffers); the governing shape-rich arg's swapped (transpose) bit is
// preserved (readComplex already folded conj, so the result is a pure
// transpose view, never a residual adjoint).
//
// Note: binary add/sub/mul over shape-rich complex never reach here —
// their wrapper routes shape-rich operands to the complex-aware
// value-ops path first. This covers the unary / constructor / scalar-
// op-over-vector cases that the value-ops shape dispatch doesn't.
function _cxRichArgGetter(v: any) {
  if (_isComplex(v)) return () => v;
  if (valueLib.isComplexValue(v)) {
    const c = valueLib.readComplex(v);
    if (c.re.length === 1) { const z = { re: c.re[0], im: c.im[0] }; return () => z; }
    return (i: any) => ({ re: c.re[i], im: c.im[i] });
  }
  if (valueLib.isValue(v)) {
    if (v.data.length === 1) { const s = v.data[0]; return () => s; }
    const d = v.data; return (i: any) => d[i];
  }
  if (v instanceof Float64Array) {
    if (v.length === 1) { const s = v[0]; return () => s; }
    return (i: any) => v[i];
  }
  const s = (typeof v === 'boolean') ? (v ? 1 : 0) : +v;
  return () => s;
}

function _cxElementwise(fn: any, args: any, shape: any, swapped: any) {
  let M = 1;
  for (let d = 0; d < shape.length; d++) M *= shape[d];
  const get = args.map(_cxRichArgGetter);
  const callArgs = new Array(args.length);
  const re = new Float64Array(M);
  let im: Float64Array | null = null;
  for (let i = 0; i < M; i++) {
    for (let k = 0; k < get.length; k++) callArgs[k] = get[k](i);
    const r = fn.apply(null, callArgs);
    if (_isComplex(r)) {
      if (im === null) im = new Float64Array(M);
      re[i] = r.re; im[i] = r.im;
    } else {
      re[i] = +r;
    }
  }
  let out;
  if (im === null) out = { shape: shape.slice(), data: re };
  else out = valueLib.complexValue(re, im, shape);
  if (swapped) out.t = 'T';
  return out;
}

// Is `v` a shape-rich Value for the complex path — a per-atom
// vector/matrix (rank ≥ 2) or an atom-indep vector (rank 1, length ≠ N
// and > 1)? Batched scalars (shape=[N]) and scalars are NOT rich.
function _cxIsRich(v: any, N: any) {
  if (!valueLib.isValue(v)) return false;
  const r = v.shape.length;
  if (r === 0) return false;
  if (r === 1) return v.shape[0] !== N && v.shape[0] > 1;
  return true;
}

function _cxBroadcast(fn: any, args: any, N: any) {
  // Shape-rich elementwise (per-atom vectors/matrices, atom-indep
  // vectors). The first rich arg governs the output shape + swapped
  // bit; remaining shape-rich args must match its element count.
  let richShape = null, richSwapped = false;
  for (let k = 0; k < args.length; k++) {
    if (_cxIsRich(args[k], N)) {
      richShape = args[k].shape;
      richSwapped = valueLib.isTransposeView(args[k]);
      break;
    }
  }
  if (richShape) return _cxElementwise(fn, args, richShape, richSwapped);

  const accs = new Array(args.length);
  let anyBatched = false;
  for (let k = 0; k < args.length; k++) {
    accs[k] = _cxArgAccessor(args[k], N);
    if (accs[k].batched) anyBatched = true;
  }
  if (!anyBatched) {
    const r = fn.apply(null, accs.map((a) => a.at(0)));
    return _isComplex(r)
      ? valueLib.complexValue([r.re], [r.im], [])
      : valueLib.scalar(+r);
  }
  const re = new Float64Array(N);
  let im: Float64Array | null = null;                       // allocated lazily on first complex
  const callArgs = new Array(args.length);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < accs.length; k++) callArgs[k] = accs[k].at(i);
    const r = fn.apply(null, callArgs);
    if (_isComplex(r)) {
      if (im === null) im = new Float64Array(N);
      re[i] = r.re; im[i] = r.im;
    } else {
      re[i] = +r;
    }
  }
  return im === null
    ? valueLib.batchedScalar(re)
    : valueLib.complexValue(re, im, [N]);
}

// Rewire complex-relevant arithmetic ops to take the complex path when
// complex is in play; otherwise keep the zero-alloc real Float64Array
// broadcast. add/sub/mul/neg already have a shape-aware wrapper (which
// handles complex shape-rich via value-ops); we only need to redirect
// their *scalar* fallback (non-shape-rich) to _cxBroadcast.
// add/sub/mul/neg: `realEntry` is the shape-aware wrapper, which
// already routes shape-rich operands (including complex shape=[N,k]) to
// the complex-aware value-ops path. So: shape-rich → realEntry
// (value-ops); else complex scalar → _cxBroadcast; else real scalar →
// realEntry (zero-alloc).
function _cxOrRealShapeAware(opName: any) {
  const realEntry = ARITH_OPS_N[opName];
  const prim = (ARITH_OPS as any)[opName];
  return (args: any, N: any) => {
    for (let i = 0; i < args.length; i++) {
      if (_shapeAwareCandidate(args[i], N)) return realEntry(args, N);
    }
    return _cxInPlay(args) ? _cxBroadcast(prim, args, N) : realEntry(args, N);
  };
}
for (const op of ['add', 'sub', 'mul', 'neg']) {
  if (ARITH_OPS_N[op]) ARITH_OPS_N[op] = _cxOrRealShapeAware(op);
}

// div/divide/pos/pow/abs/abs2/exp/log/sqrt are scalar-only ops with no
// shape-rich value-ops path. When complex is in play they go through
// _cxBroadcast (which itself raises a clear guard for shape-rich
// complex — better than silently producing NaN). Otherwise the
// existing real broadcast entry is preserved unchanged.
function _cxOrReal(opName: any) {
  const realEntry = ARITH_OPS_N[opName];
  const prim = (ARITH_OPS as any)[opName];
  return (args: any, N: any) => _cxInPlay(args)
    ? _cxBroadcast(prim, args, N)
    : realEntry(args, N);
}
for (const op of ['div', 'divide', 'pos', 'pow',
                   'abs', 'abs2', 'exp', 'log', 'sqrt']) {
  if (ARITH_OPS_N[op]) ARITH_OPS_N[op] = _cxOrReal(op);
}

// complex / real / imag / conj / cis had no ARITH_OPS_N entry (atom-
// dependent complex previously threw). Route them unconditionally
// through _cxBroadcast: it degrades to a real Value when inputs and
// output are real (e.g. conj / real on real data), and produces a
// complex Value for the constructors (complex / cis) and for complex
// inputs. Arity is taken from the IR args length at call time.
for (const op of ['complex', 'real', 'imag', 'conj', 'cis']) {
  const prim = (ARITH_OPS as any)[op];
  ARITH_OPS_N[op] = (args: any, N: any) => _cxBroadcast(prim, args, N);
}

// =====================================================================
// Linear-algebra helpers (textbook algorithms; small-matrix sized)
// =====================================================================
// Extracted to sampler-linalg.ts (engine-concepts §17.5 split).

const _linalg = require('./sampler-linalg.ts');
const {
  _luDecomp, _detLU, _logAbsDetLU,
  _luDecompValue, _detLUValue, _logAbsDetLUValue,
  _linsolveLU, _linsolveLUValue,
  _invGaussJordan, _invValue,
  _cholesky, _choleskyValue,
  _matmul,
} = _linalg;


// Helper: build a one-pass reducer that loops over array indices.
// Avoids repeated Array.prototype.reduce overhead for tight inner
// loops on large literal arrays.
function reduce(step: any, init: any) {
  return (arr: any) => {
    let acc = init;
    for (let i = 0; i < arr.length; i++) acc = step(acc, arr[i]);
    return acc;
  };
}

// Map a set IR shape to numeric [lo, hi] bounds for value-level ops
// that accept a region argument (e.g. selectbins). Recognizes literal
// `interval(lo, hi)` and the named real sets. Anything else throws
// — richer set descriptors live behind orchestrator.parseSetIR, which
// the sampler intentionally doesn't depend on.

// =====================================================================
// aggregate(f_reduction, output_axes, expr) — spec §04 §sec:aggregate
// =====================================================================
//
// Multi-axis tensor contraction. Phase = join of input phases (this
// runs per-atom via the standard per-atom-fallback path; no batching
// at this layer in v0.1).
//
// Architecture (engine-concepts §16):
//   AGGREGATE_PATTERNS — list of pattern specialisers (matmul, …)
//     tried in order; first match wins. Gated by the `aggregate`
//     optimisation toggle.
//   _evalAggregateBroadcastReduce — the canonical permute-reshape-
//     broadcast-reduce einsum lowering. Default for everything the
//     specialisers don't catch; ALSO the correctness oracle the
//     dual-mode test runner compares specialisers against (with
//     `aggregate=off` forcing the broadcast-reduce path).

// Axis length inference walks the IR looking for the FIRST get(...) or
// get0(...) whose selector slot k is an axis ref; the length of that
// axis is then the length of the container's dim-k. Walk stops at
// nested aggregate boundaries — those have their own axis scope.
function _inferAggregateAxisLengths(exprIR: any, axisNames: string[], env: any) {
  const lengths: Record<string, number> = {};
  function _shapeOf(val: any): number[] | null {
    if (val == null) return null;
    if (typeof val === 'number' || typeof val === 'boolean') return [];
    if (Array.isArray(val)) {
      // Nested-array form: shape from outer length + recurse for inner.
      const inner: number[] = [];
      if (val.length > 0 && Array.isArray(val[0])) {
        const tail = _shapeOf(val[0]);
        if (tail) for (const x of tail) inner.push(x);
      }
      return [val.length, ...inner];
    }
    if (val && (val as any).BYTES_PER_ELEMENT) return [(val as any).length];
    if (val && Array.isArray((val as any).shape)) return (val as any).shape;
    return null;
  }
  function walk(n: any) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { for (const c of n) walk(c); return; }
    if (n.kind === 'call' && n.op === 'aggregate') return;   // inner scope
    if (n.kind === 'call' && (n.op === 'get' || n.op === 'get0')) {
      const args = n.args || [];
      const container = args[0];
      const sels = args.slice(1);
      let containerVal: any = undefined;
      for (let k = 0; k < sels.length; k++) {
        const s = sels[k];
        if (s && s.kind === 'axis' && !(s.name in lengths)) {
          if (containerVal === undefined) {
            try { containerVal = evaluateExpr(container, env); }
            catch (_) { containerVal = null; }
          }
          if (containerVal == null) continue;
          const shape = _shapeOf(containerVal);
          if (shape && typeof shape[k] === 'number') {
            lengths[s.name] = shape[k];
          }
        }
      }
    }
    for (const k of Object.keys(n)) {
      if (k === 'loc' || k === 'kind' || k === 'op' || k === 'name' || k === 'ns') continue;
      walk(n[k]);
    }
  }
  walk(exprIR);
  // Verify every needed axis got a length; raise a clear error if not.
  const missing = axisNames.filter((a: string) => !(a in lengths));
  if (missing.length > 0) {
    throw new Error(
      `aggregate: could not infer length of axis ${missing.map((a: string) => '.' + a).join(', ')} ` +
      `— each axis must index a known array at least once in expr`);
  }
  return lengths;
}

// Collect axis names that appear in expr, NOT descending into nested
// aggregate(...) bodies (each aggregate has its own closed axis scope
// per spec §05).
function _collectInScopeAxisNames(exprIR: any): Set<string> {
  const names = new Set<string>();
  function walk(n: any) {
    if (!n || typeof n !== 'object') return;
    if (Array.isArray(n)) { for (const c of n) walk(c); return; }
    if (n.kind === 'axis') { names.add(n.name); return; }
    if (n.kind === 'call' && n.op === 'aggregate') return;
    for (const k of Object.keys(n)) {
      if (k === 'loc' || k === 'kind' || k === 'op' || k === 'name' || k === 'ns') continue;
      walk(n[k]);
    }
  }
  walk(exprIR);
  return names;
}

// Pattern table — each entry recognises a specific aggregate shape
// and dispatches to an accelerated implementation. Tried in order;
// first match wins. The broadcast-reduce default
// (`_evalAggregateBroadcastReduce`) is the fallback AND the
// correctness oracle that every specialiser must match. Run tests
// with `perf-config.setOptimization('aggregate', false)` to force the
// broadcast-reduce path and verify equivalence.
//
// Each entry has the shape:
//   match(ir, env): match-state | null
//   execute(ir, env, match): result
// where `match` may carry pre-extracted IR refs (e.g. the two factor
// arrays for matmul) so `execute` doesn't have to re-walk the IR.
const AGGREGATE_PATTERNS: Array<{
  name: string;
  match: (ir: any, env: any) => any;
  execute: (ir: any, env: any, match: any) => any;
}> = [];

function _evalAggregate(ir: any, env: any): any {
  // Pattern dispatch is gated by the `aggregate` optimisation toggle.
  // When disabled, every aggregate flows through the broadcast-reduce
  // default (engine-concepts §16) so equivalence tests (`inBothModes`)
  // can verify each specialiser agrees with the canonical lowering.
  if (perfConfig.getOptimization('aggregate')) {
    for (const p of AGGREGATE_PATTERNS) {
      const m = p.match(ir, env);
      if (m) return p.execute(ir, env, m);
    }
  }
  return _evalAggregateBroadcastReduce(ir, env);
}

// ---------------------------------------------------------------------
// Helper for the matmul-family specialisers below: dispatch the
// nested-array or shape-rich-Value matrix product. Optionally
// transposes either operand before the multiplication. For shape-
// rich Values the transpose is a free Klein-4 tag flip; for nested
// arrays it's `ARITH_OPS.transpose`.
// ---------------------------------------------------------------------
function _matmulDispatch(A: any, B: any, transA: boolean, transB: boolean): any {
  if (valueLib.isValue(A) || valueLib.isValue(B)) {
    let aV = valueLib.asValue(A);
    let bV = valueLib.asValue(B);
    if (transA) aV = valueLib.transpose(aV);
    if (transB) bV = valueLib.transpose(bV);
    return valueOps.mul(aV, bV);
  }
  const aN = transA ? (ARITH_OPS as any).transpose(A) : A;
  const bN = transB ? (ARITH_OPS as any).transpose(B) : B;
  return _matmul(aN, bN);
}

// ---------------------------------------------------------------------
// Specialiser: matmul family (4 transpose variants)
//
//   aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])   ≡  A · B
//   aggregate(sum, [.i, .k], A[.j, .i] * B[.j, .k])   ≡  Aᵀ · B
//   aggregate(sum, [.i, .k], A[.i, .j] * B[.k, .j])   ≡  A · Bᵀ
//   aggregate(sum, [.i, .k], A[.j, .i] * B[.k, .j])   ≡  Aᵀ · Bᵀ
//
// Each operand's axes can be in either order; the matcher determines
// which dim each axis name occupies and sets a transpose flag.
// Scalar multiplication is commutative, so either factor order in
// the `mul` body matches the same logical product.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'matmul-family',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    if (!fIR || fIR.kind !== 'ref' || fIR.name !== 'sum') return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 2) return null;
    if (outAxes[0].kind !== 'axis' || outAxes[1].kind !== 'axis') return null;
    const iName = outAxes[0].name;
    const kName = outAxes[1].name;
    if (iName === kName) return null;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;

    // A's standard matmul layout is `A[.i, .j]` — output dim first,
    // reduce dim second; transposed is `A[.j, .i]`. B's standard is
    // `B[.j, .k]` — reduce dim first, output dim second; transposed
    // is `B[.k, .j]`. The two operands have ASYMMETRIC "normal"
    // conventions, so they get separate classifiers.
    function classifyA(fac: any):
      { src: any; trans: boolean; jName: string } | null
    {
      if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
      if (!fac.args || fac.args.length !== 3) return null;
      const s0 = fac.args[1], s1 = fac.args[2];
      if (!s0 || s0.kind !== 'axis') return null;
      if (!s1 || s1.kind !== 'axis') return null;
      if (s0.name === iName) return { src: fac.args[0], trans: false, jName: s1.name };
      if (s1.name === iName) return { src: fac.args[0], trans: true,  jName: s0.name };
      return null;
    }
    function classifyB(fac: any):
      { src: any; trans: boolean; jName: string } | null
    {
      if (!fac || fac.kind !== 'call' || fac.op !== 'get') return null;
      if (!fac.args || fac.args.length !== 3) return null;
      const s0 = fac.args[1], s1 = fac.args[2];
      if (!s0 || s0.kind !== 'axis') return null;
      if (!s1 || s1.kind !== 'axis') return null;
      if (s1.name === kName) return { src: fac.args[0], trans: false, jName: s0.name };
      if (s0.name === kName) return { src: fac.args[0], trans: true,  jName: s1.name };
      return null;
    }

    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    // Try assigning (f1=A, f2=B), then (f1=B, f2=A).
    for (const [fA, fB] of [[f1, f2], [f2, f1]]) {
      const ca = classifyA(fA);
      const cb = classifyB(fB);
      if (!ca || !cb) continue;
      // Same reduction axis from both sides.
      if (ca.jName !== cb.jName) continue;
      if (ca.jName === iName || ca.jName === kName) continue;
      return { aIR: ca.src, bIR: cb.src, transA: ca.trans, transB: cb.trans };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const B = evaluateExpr(match.bIR, env);
    return _matmulDispatch(A, B, match.transA, match.transB);
  },
});

// ---------------------------------------------------------------------
// Specialiser: matrix-vector multiplication (incl. transposed-A variant)
//
//   aggregate(sum, [.i], A[.i, .j] * v[.j])   ≡  A · v
//   aggregate(sum, [.i], A[.j, .i] * v[.j])   ≡  Aᵀ · v
//
// The single output axis owns A's first or second dim depending on
// which side `.i` appears; the shared axis (.j) iterates over A's
// other dim and over v.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'matvec',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    if (!fIR || fIR.kind !== 'ref' || fIR.name !== 'sum') return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 1) return null;
    if (outAxes[0].kind !== 'axis') return null;
    const iName = outAxes[0].name;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    // The matrix factor has rank-2 indexing (3 args to `get`); the
    // vector factor has rank-1 indexing (2 args). Try both orderings.
    for (const [matGet, vecGet] of [[f1, f2], [f2, f1]]) {
      if (!matGet || matGet.kind !== 'call' || matGet.op !== 'get') continue;
      if (!vecGet || vecGet.kind !== 'call' || vecGet.op !== 'get') continue;
      if (!matGet.args || matGet.args.length !== 3) continue;
      if (!vecGet.args || vecGet.args.length !== 2) continue;
      const mS = [matGet.args[1], matGet.args[2]];
      const vS = vecGet.args[1];
      if (mS[0].kind !== 'axis' || mS[1].kind !== 'axis') continue;
      if (!vS || vS.kind !== 'axis') continue;
      const jName = vS.name;
      if (jName === iName) continue;
      // Determine A's orientation: which dim is .i, which is .j?
      let transA = false;
      if (mS[0].name === iName && mS[1].name === jName) transA = false;
      else if (mS[0].name === jName && mS[1].name === iName) transA = true;
      else continue;
      return { aIR: matGet.args[0], vIR: vecGet.args[0], transA };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const v = evaluateExpr(match.vIR, env);
    if (valueLib.isValue(A) || valueLib.isValue(v)) {
      let aV = valueLib.asValue(A);
      if (match.transA) aV = valueLib.transpose(aV);
      return valueOps.mul(aV, valueLib.asValue(v));
    }
    // Nested-array matvec.
    const Ax = match.transA ? (ARITH_OPS as any).transpose(A) : A;
    const m = Ax.length;
    const n = Ax[0].length;
    const out = new Float64Array(m);
    for (let i = 0; i < m; i++) {
      let s = 0;
      const row = Ax[i];
      for (let j = 0; j < n; j++) s += row[j] * v[j];
      out[i] = s;
    }
    return out;
  },
});

// ---------------------------------------------------------------------
// Specialiser: outer product
//
//   aggregate(<any>, [.i, .j], u[.i] * v[.j])   ≡  u · vᵀ (outer)
//
// No reduce axes → the reduction function is irrelevant (every
// "reduction" is over a single value). The broadcast-reduce default
// produces the right answer but materialises a `_broadcastTo` copy;
// the specialiser computes the outer product directly into one pass.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'outer-product',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [_fIR, axesIR, bodyIR] = args;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 2) return null;
    if (outAxes[0].kind !== 'axis' || outAxes[1].kind !== 'axis') return null;
    const iName = outAxes[0].name;
    const jName = outAxes[1].name;
    if (iName === jName) return null;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    // Two rank-1 indexings, one on `.i` one on `.j`.
    for (const [uGet, vGet] of [[f1, f2], [f2, f1]]) {
      if (!uGet || uGet.kind !== 'call' || uGet.op !== 'get') continue;
      if (!vGet || vGet.kind !== 'call' || vGet.op !== 'get') continue;
      if (!uGet.args || uGet.args.length !== 2) continue;
      if (!vGet.args || vGet.args.length !== 2) continue;
      const us = uGet.args[1], vs = vGet.args[1];
      if (!us || us.kind !== 'axis' || us.name !== iName) continue;
      if (!vs || vs.kind !== 'axis' || vs.name !== jName) continue;
      return { uIR: uGet.args[0], vIR: vGet.args[0] };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const u = evaluateExpr(match.uIR, env);
    const v = evaluateExpr(match.vIR, env);
    // Direct outer product into a nested array. The Value-typed path
    // (atom-batched) routes through valueOps.mul which handles outer
    // via the tvector tag (vector × transposed-vector → matrix per
    // spec §07 linear-algebra rules).
    if (valueLib.isValue(u) || valueLib.isValue(v)) {
      const uV = valueLib.asValue(u);
      const vV = valueLib.asValue(v);
      // Outer = uV (col) · transpose(vV) (row). transpose() flips the
      // Klein-4 tag; valueOps.mul then dispatches to outer.
      return valueOps.mul(uV, valueLib.transpose(vV));
    }
    const m = u.length, n = v.length;
    const out: any[] = new Array(m);
    for (let i = 0; i < m; i++) {
      const row = new Float64Array(n);
      const ui = u[i];
      for (let j = 0; j < n; j++) row[j] = ui * v[j];
      out[i] = row;
    }
    return out;
  },
});

// ---------------------------------------------------------------------
// Specialiser: batched matmul
//
//   aggregate(sum, [.b, .i, .k],
//             A[.b, .i, .j] * B[.b, .j, .k])  ≡ per-batch A·B
//
// Common in atom-batched contexts (sampler runs aggregate per
// atom; the leading axis broadcasts the per-atom data). Avoids the
// full O(Nb · Ni · Nj · Nk) broadcast intermediate the default
// would build — at most O(Nb · max(Ni·Nj, Nj·Nk, Ni·Nk)) per
// stage.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'batched-matmul',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    if (!fIR || fIR.kind !== 'ref' || fIR.name !== 'sum') return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length !== 3) return null;
    if (outAxes[0].kind !== 'axis' || outAxes[1].kind !== 'axis'
        || outAxes[2].kind !== 'axis') return null;
    const bName = outAxes[0].name;
    const iName = outAxes[1].name;
    const kName = outAxes[2].name;
    if (bName === iName || iName === kName || bName === kName) return null;
    if (!bodyIR || bodyIR.kind !== 'call' || bodyIR.op !== 'mul') return null;
    if (!bodyIR.args || bodyIR.args.length !== 2) return null;
    const f1 = bodyIR.args[0], f2 = bodyIR.args[1];
    for (const [aGet, bGet] of [[f1, f2], [f2, f1]]) {
      if (!aGet || aGet.kind !== 'call' || aGet.op !== 'get') continue;
      if (!bGet || bGet.kind !== 'call' || bGet.op !== 'get') continue;
      if (!aGet.args || aGet.args.length !== 4) continue;
      if (!bGet.args || bGet.args.length !== 4) continue;
      const aSels = [aGet.args[1], aGet.args[2], aGet.args[3]];
      const bSels = [bGet.args[1], bGet.args[2], bGet.args[3]];
      if (aSels.some((s: any) => !s || s.kind !== 'axis')) continue;
      if (bSels.some((s: any) => !s || s.kind !== 'axis')) continue;
      // Canonical layout: A[.b, .i, .j], B[.b, .j, .k]. (Other
      // permutations are an extension; this matcher targets the
      // typical case.)
      if (aSels[0].name !== bName) continue;
      if (aSels[1].name !== iName) continue;
      if (bSels[0].name !== bName) continue;
      if (bSels[2].name !== kName) continue;
      const jName = aSels[2].name;
      if (jName === bName || jName === iName || jName === kName) continue;
      if (bSels[1].name !== jName) continue;
      return { aIR: aGet.args[0], bIR: bGet.args[0] };
    }
    return null;
  },
  execute(_ir: any, env: any, match: any): any {
    const A = evaluateExpr(match.aIR, env);
    const B = evaluateExpr(match.bIR, env);
    // Per-batch nested-array matmul. (Value-typed batched matmul
    // would go through valueOps with the batch axis as leading dim;
    // for v0.1 the nested-array path covers the typical user case.)
    const Nb = A.length;
    const out: any[] = new Array(Nb);
    for (let b = 0; b < Nb; b++) {
      out[b] = _matmul(A[b], B[b]);
    }
    return out;
  },
});

// ---------------------------------------------------------------------
// Specialiser: pure axis reduction (single source, no arithmetic)
//
//   aggregate(f, [out_axes…], get(arr, sels…))
//
// Common pattern: column sums, row means, axis-wise max/min, etc.
// All seven reductions are supported via _AGGREGATE_REDUCTIONS. The
// fast path applies the reduction along each non-output dim directly
// on the source — no broadcast intermediate, no per-axis lift.
// ---------------------------------------------------------------------
AGGREGATE_PATTERNS.push({
  name: 'pure-axis-reduction',
  match(ir: any, _env: any): any {
    const args = ir.args || [];
    if (args.length !== 3) return null;
    const [fIR, axesIR, bodyIR] = args;
    const fname = (fIR && fIR.kind === 'ref' && fIR.name) || null;
    if (!fname || !_AGGREGATE_REDUCTIONS[fname]) return null;
    if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') return null;
    const outAxes = axesIR.args || [];
    if (outAxes.length < 1) return null;
    const outAxisNames: string[] = [];
    for (const a of outAxes) {
      if (!a || a.kind !== 'axis') return null;
      outAxisNames.push(a.name);
    }
    // Body must be a single get / get0 — no arithmetic.
    if (!bodyIR || bodyIR.kind !== 'call') return null;
    if (bodyIR.op !== 'get' && bodyIR.op !== 'get0') return null;
    if (!bodyIR.args || bodyIR.args.length < 2) return null;
    // Every selector must be an axis ref (no `only`/`all`/integer in
    // this fast path — those would require pre-applying selectors,
    // which the broadcast-reduce default does already).
    const sels = bodyIR.args.slice(1);
    const selAxisAt: Record<string, number> = {};
    for (let k = 0; k < sels.length; k++) {
      const s = sels[k];
      if (!s || s.kind !== 'axis') return null;
      if (s.name in selAxisAt) return null;  // same axis twice = diagonal extraction (not this fast path)
      selAxisAt[s.name] = k;
    }
    // Every output axis must be present in the body's selectors.
    for (const a of outAxisNames) if (!(a in selAxisAt)) return null;
    return { srcIR: bodyIR.args[0], fname, outAxisNames, selAxisAt, numSourceDims: sels.length };
  },
  execute(_ir: any, env: any, match: any): any {
    const src = evaluateExpr(match.srcIR, env);
    const flat = _toFlat(src);
    const sourceShape = flat.shape;
    const sourceStrides = _rowMajorStrides(sourceShape);
    const reduce = _AGGREGATE_REDUCTIONS[match.fname];

    // Determine per-output-cell traversal: iterate output axes,
    // for each (output-coord) reduce across the remaining dims.
    const outDimSizes = match.outAxisNames.map(
      (a: string) => sourceShape[match.selAxisAt[a]]);
    const outSize = outDimSizes.reduce((p: number, n: number) => p * n, 1);
    const outStrides = match.outAxisNames.map(
      (a: string) => sourceStrides[match.selAxisAt[a]]);

    // Reduce axes = source dims not in output_axes.
    const reduceDims: { sourceDim: number; size: number }[] = [];
    const inOutput = new Set(match.outAxisNames);
    for (const a of Object.keys(match.selAxisAt)) {
      if (!inOutput.has(a)) {
        const d = match.selAxisAt[a];
        reduceDims.push({ sourceDim: d, size: sourceShape[d] });
      }
    }
    const reduceSize = reduceDims.reduce(
      (p: number, x: any) => p * x.size, 1);

    const outData = new Float64Array(outSize);
    const tmp = new Float64Array(reduceSize);
    const outCoord = new Array(match.outAxisNames.length).fill(0);
    const redCoord = new Array(reduceDims.length).fill(0);
    for (let i = 0; i < outSize; i++) {
      let baseOff = 0;
      for (let k = 0; k < outCoord.length; k++) baseOff += outCoord[k] * outStrides[k];
      // Reduce over the reduce-dims at this output cell.
      for (let r = 0; r < reduceDims.length; r++) redCoord[r] = 0;
      for (let r = 0; r < reduceSize; r++) {
        let off = baseOff;
        for (let k = 0; k < reduceDims.length; k++) {
          off += redCoord[k] * sourceStrides[reduceDims[k].sourceDim];
        }
        tmp[r] = flat.data[off];
        // Increment redCoord (last dim fastest).
        for (let k = reduceDims.length - 1; k >= 0; k--) {
          redCoord[k]++;
          if (redCoord[k] < reduceDims[k].size) break;
          redCoord[k] = 0;
        }
      }
      outData[i] = +reduce(tmp);
      // Increment outCoord.
      for (let k = outCoord.length - 1; k >= 0; k--) {
        outCoord[k]++;
        if (outCoord[k] < outDimSizes[k]) break;
        outCoord[k] = 0;
      }
    }
    return _flatToNested(outData, outDimSizes);
  },
});

// Map reduction name → ARITH_OPS implementation.
const _AGGREGATE_REDUCTIONS: Record<string, (a: any) => number> = {
  sum:     (a: any) => (ARITH_OPS as any).sum(a),
  prod:    (a: any) => (ARITH_OPS as any).prod(a),
  mean:    (a: any) => (ARITH_OPS as any).mean(a),
  var:     (a: any) => (ARITH_OPS as any).var(a),
  std:     (a: any) => (ARITH_OPS as any).std(a),
  maximum: (a: any) => (ARITH_OPS as any).maximum(a),
  minimum: (a: any) => (ARITH_OPS as any).minimum(a),
};

// =====================================================================
// Broadcast-reduce default implementation
// =====================================================================
//
// See engine-concepts §16 for the rationale. The pipeline:
//
//   1. Determine canonical axis order = [output_axes…, reduce_axes…].
//   2. For each `get(arr, ...)` in expr, build an "aligned tensor": a
//      Float64Array + shape, where the shape has one dim per canonical
//      axis. Dims the source actually uses carry the axis's length;
//      dims it doesn't are singletons (broadcast-stretchable).
//      Non-axis selectors (integer / `only`) are pre-applied at this
//      stage so the aligned tensor only carries the surviving dims.
//   3. Lift the rest of `expr` to operate on aligned tensors. Each
//      arithmetic op is a broadcast-elementwise sweep over the
//      canonical-order coordinate space — one flat loop per op.
//      Subtrees with no axis refs evaluate to scalars and broadcast
//      as singletons.
//   4. Tail-reduce: with canonical order putting reduce axes last,
//      the reduction is a contiguous-block sweep over the trailing
//      `reduceSize` cells per output slot.
//
// All steps are vectorised tensor primitives — the same ops every
// accelerated backend offers. No per-axis-coord JS function-call
// overhead.

function _shapeProd(shape: number[]): number {
  let p = 1; for (let i = 0; i < shape.length; i++) p *= shape[i]; return p;
}

// Row-major strides for a shape: stride[i] = product of shape[i+1..].
function _rowMajorStrides(shape: number[]): number[] {
  const N = shape.length;
  const out = new Array(N);
  let s = 1;
  for (let i = N - 1; i >= 0; i--) { out[i] = s; s *= shape[i]; }
  return out;
}

// Broadcasting strides: along any dim of size 1, the stride is 0
// (don't advance in that dim — repeat the singleton value).
function _broadcastStrides(srcShape: number[], outShape: number[]): number[] {
  const N = srcShape.length;
  const out = new Array(N);
  let s = 1;
  for (let i = N - 1; i >= 0; i--) {
    out[i] = (srcShape[i] === 1) ? 0 : s;
    s *= srcShape[i];
  }
  return out;
}

// Convert a nested-array (or 1-D typed array) value to {data, shape}.
// Nested → flat: walk in row-major order. Typed-array → wraps as
// shape=[length]. Scalar → shape=[].
function _toFlat(val: any): { data: Float64Array; shape: number[] } {
  if (val == null) return { data: new Float64Array([0]), shape: [] };
  if (typeof val === 'number' || typeof val === 'boolean') {
    return { data: new Float64Array([+val]), shape: [] };
  }
  if (val && (val as any).BYTES_PER_ELEMENT !== undefined) {
    // Typed array — 1-D.
    return { data: val as Float64Array, shape: [val.length] };
  }
  if (Array.isArray(val)) {
    // Determine shape by walking the first element down.
    const shape: number[] = [val.length];
    let probe: any = val[0];
    while (Array.isArray(probe)
           || (probe && probe.BYTES_PER_ELEMENT !== undefined)) {
      shape.push(probe.length);
      probe = probe[0];
    }
    const size = _shapeProd(shape);
    const data = new Float64Array(size);
    let w = 0;
    function fill(x: any) {
      if (Array.isArray(x) || (x && x.BYTES_PER_ELEMENT !== undefined)) {
        for (let i = 0; i < x.length; i++) fill(x[i]);
      } else {
        data[w++] = +x;
      }
    }
    fill(val);
    return { data, shape };
  }
  if (valueLib.isValue(val)) {
    return { data: val.data as Float64Array, shape: val.shape.slice() };
  }
  throw new Error(`aggregate: cannot interpret value of type ${typeof val} as a tensor`);
}

// Flat data + shape → nested-JS-array (rank > 1) or Float64Array (rank
// 1) or scalar (rank 0). Matches the existing output convention from
// `_matmul` and the linspace family.
function _flatToNested(data: Float64Array, shape: number[]): any {
  if (shape.length === 0) return data[0];
  if (shape.length === 1) return data;
  function build(start: number, sh: number[]): any {
    if (sh.length === 1) {
      const out = new Float64Array(sh[0]);
      for (let i = 0; i < sh[0]; i++) out[i] = data[start + i];
      return out;
    }
    const stride = _shapeProd(sh.slice(1));
    const out: any[] = new Array(sh[0]);
    for (let i = 0; i < sh[0]; i++) {
      out[i] = build(start + i * stride, sh.slice(1));
    }
    return out;
  }
  return build(0, shape);
}

// True if `node` (an IR subtree) contains any `kind: 'axis'` node.
// Used for constant-hoisting: axis-free subtrees evaluate once and
// broadcast as singletons. Stops at nested aggregate boundaries
// (those have a closed axis scope per spec §05).
function _containsAxisRef(node: any): boolean {
  if (!node || typeof node !== 'object') return false;
  if (Array.isArray(node)) {
    for (const c of node) if (_containsAxisRef(c)) return true;
    return false;
  }
  if (node.kind === 'axis') return true;
  if (node.kind === 'call' && node.op === 'aggregate') return false;
  for (const k of Object.keys(node)) {
    if (k === 'loc' || k === 'kind' || k === 'op'
        || k === 'name' || k === 'ns') continue;
    if (_containsAxisRef(node[k])) return true;
  }
  return false;
}

// Build an aligned tensor for a `get(arr, sel0, sel1, …)` IR call.
// The resulting shape has `canonicalAxes.length` dims; each is either
// the axis's length (if this source uses that axis) or 1 (singleton,
// broadcast-stretchable). Non-axis selectors (integer indices, `only`)
// are pre-applied and collapse their source dim before the alignment.
function _alignedTensorFromGet(
  getIR: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
): { data: Float64Array; shape: number[] } {
  const args = getIR.args || [];
  const oneBased = getIR.op === 'get';
  const arr = evaluateExpr(args[0], env);
  const sels = args.slice(1);
  const src = _toFlat(arr);
  const sourceStrides = _rowMajorStrides(src.shape);

  // Walk selectors. For each, decide:
  //   - axis ref: remember this dim corresponds to a canonical axis.
  //   - integer / 'only': collapse this dim (advance baseOffset).
  //   - 'all': not supported in aggregate body (per §04 spec the body
  //     uses axis names; `all`/':' would mean "keep an unnamed dim"
  //     which has no defined semantics for the contraction).
  const axisAt: Record<string, number> = {};   // axisName → source dim
  let baseOffset = 0;
  for (let k = 0; k < sels.length; k++) {
    const s = sels[k];
    if (s && s.kind === 'axis') {
      axisAt[s.name] = k;
      continue;
    }
    if (s && s.kind === 'const' && s.name === 'only') {
      if (src.shape[k] !== 1) {
        throw new Error(`aggregate: 'only' selector requires the indexed `
          + `axis to have length 1, got length ${src.shape[k]}`);
      }
      // Index 0 — contributes 0 to baseOffset.
      continue;
    }
    if (s && s.kind === 'const' && s.name === 'all') {
      throw new Error(`aggregate: 'all' / ':' is not supported in an `
        + `aggregate body; use an axis name (.name) instead`);
    }
    // Integer index — evaluate (1-based for `get`, 0-based for `get0`).
    const idx = +evaluateExpr(s, env);
    const idx0 = oneBased ? (idx | 0) - 1 : (idx | 0);
    if (idx0 < 0 || idx0 >= src.shape[k]) {
      throw new Error(`aggregate: index ${oneBased ? idx : idx0} out of bounds `
        + `for axis of length ${src.shape[k]}`);
    }
    baseOffset += idx0 * sourceStrides[k];
  }

  // Build the aligned shape + a per-canonical-position source stride.
  // If the canonical axis isn't in this source, the dim is singleton
  // and contributes 0 to the source offset (broadcast).
  const N = canonicalAxes.length;
  const alignedShape = new Array(N);
  const sourceStrideAt: number[] = new Array(N);
  for (let i = 0; i < N; i++) {
    const axisName = canonicalAxes[i];
    if (axisName in axisAt) {
      alignedShape[i] = axisLengths[axisName];
      sourceStrideAt[i] = sourceStrides[axisAt[axisName]];
    } else {
      alignedShape[i] = 1;
      sourceStrideAt[i] = 0;
    }
  }

  // Materialise the aligned tensor. Iterate output cells in row-major
  // order with a coordinate vector; for each, compute the source
  // offset = baseOffset + Σ coord[i] · sourceStrideAt[i].
  const alignedSize = _shapeProd(alignedShape);
  const out = new Float64Array(alignedSize);
  const coord = new Array(N).fill(0);
  for (let linear = 0; linear < alignedSize; linear++) {
    let srcOff = baseOffset;
    for (let i = 0; i < N; i++) srcOff += coord[i] * sourceStrideAt[i];
    out[linear] = src.data[srcOff];
    // Increment coord, last dim fastest.
    for (let i = N - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < alignedShape[i]) break;
      coord[i] = 0;
    }
  }
  return { data: out, shape: alignedShape };
}

// Inline scalar implementations for the most common binary ops in
// aggregate bodies — bypassing the ARITH_OPS dispatch saves a JS
// function call per element in the broadcast inner loop.
const _AGG_BIN: Record<string, (x: number, y: number) => number> = {
  add:    (x, y) => x + y,
  sub:    (x, y) => x - y,
  mul:    (x, y) => x * y,
  div:    (x, y) => x / y,
  divide: (x, y) => x / y,
  pow:    (x, y) => Math.pow(x, y),
  mod:    (x, y) => x % y,
};

const _AGG_UN: Record<string, (x: number) => number> = {
  neg: (x) => -x,    pos: (x) => +x,
  exp:   Math.exp,    log:   Math.log,    log10: Math.log10,
  log1p: Math.log1p,  expm1: Math.expm1,
  sqrt:  Math.sqrt,   abs:   Math.abs,    abs2: (x) => x * x,
  sin:   Math.sin,    cos:   Math.cos,    tan:   Math.tan,
  asin:  Math.asin,   acos:  Math.acos,   atan:  Math.atan,
  sinh:  Math.sinh,   cosh:  Math.cosh,   tanh:  Math.tanh,
  asinh: Math.asinh,  acosh: Math.acosh,  atanh: Math.atanh,
  floor: Math.floor,  ceil:  Math.ceil,   round: Math.round,
};

// Broadcast-elementwise binary op over two aligned tensors. Each
// already has shape canonicalAxes.length; broadcasting comes from
// per-dim singleton vs full-length stride choice.
function _broadcastBinary(
  a: { data: Float64Array; shape: number[] },
  b: { data: Float64Array; shape: number[] },
  fn: (x: number, y: number) => number,
): { data: Float64Array; shape: number[] } {
  const N = a.shape.length;
  const outShape = new Array(N);
  for (let i = 0; i < N; i++) outShape[i] = Math.max(a.shape[i], b.shape[i]);
  const outSize = _shapeProd(outShape);
  const aStrides = _broadcastStrides(a.shape, outShape);
  const bStrides = _broadcastStrides(b.shape, outShape);
  const out = new Float64Array(outSize);
  const coord = new Array(N).fill(0);
  for (let linear = 0; linear < outSize; linear++) {
    let aOff = 0, bOff = 0;
    for (let i = 0; i < N; i++) { aOff += coord[i] * aStrides[i]; bOff += coord[i] * bStrides[i]; }
    out[linear] = fn(a.data[aOff], b.data[bOff]);
    for (let i = N - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < outShape[i]) break;
      coord[i] = 0;
    }
  }
  return { data: out, shape: outShape };
}

function _broadcastUnary(
  a: { data: Float64Array; shape: number[] },
  fn: (x: number) => number,
): { data: Float64Array; shape: number[] } {
  const out = new Float64Array(a.data.length);
  for (let i = 0; i < a.data.length; i++) out[i] = fn(a.data[i]);
  return { data: out, shape: a.shape.slice() };
}

// Lift an expression IR subtree to an aligned tensor of canonical
// shape. Each leaf becomes either a scalar (axis-free subtree) or an
// aligned tensor (a `get` call); each interior arithmetic op becomes
// a broadcast-elementwise sweep.
function _liftAggregateExpr(
  node: any, canonicalAxes: string[],
  axisLengths: Record<string, number>, env: any,
): { data: Float64Array; shape: number[] } {
  // Constant-hoist: subtree with no axis refs evaluates once and
  // broadcasts as all-singleton.
  if (!_containsAxisRef(node)) {
    const v = +evaluateExpr(node, env);
    const shape = canonicalAxes.map(() => 1);
    return { data: new Float64Array([v]), shape };
  }
  // `get` / `get0`: produces an aligned tensor at this leaf.
  if (node.kind === 'call' && (node.op === 'get' || node.op === 'get0')) {
    return _alignedTensorFromGet(node, canonicalAxes, axisLengths, env);
  }
  // Binary arithmetic: lift both operands, broadcast-apply.
  if (node.kind === 'call' && node.args && node.args.length === 2) {
    const fn = _AGG_BIN[node.op];
    if (fn) {
      const a = _liftAggregateExpr(node.args[0], canonicalAxes, axisLengths, env);
      const b = _liftAggregateExpr(node.args[1], canonicalAxes, axisLengths, env);
      return _broadcastBinary(a, b, fn);
    }
  }
  // Unary scalar math.
  if (node.kind === 'call' && node.args && node.args.length === 1) {
    const fn = _AGG_UN[node.op];
    if (fn) {
      const a = _liftAggregateExpr(node.args[0], canonicalAxes, axisLengths, env);
      return _broadcastUnary(a, fn);
    }
  }
  throw new Error(`aggregate: unsupported op '${node.op || node.kind}' `
    + `in body — broadcast-reduce default supports arithmetic and `
    + `unary math on indexed arrays. Hoist non-broadcasting subterms `
    + `to bindings outside the aggregate.`);
}

function _evalAggregateBroadcastReduce(ir: any, env: any): any {
  const args = ir.args || [];
  if (args.length !== 3) {
    throw new Error(`aggregate: expected 3 args, got ${args.length}`);
  }
  const [fIR, axesIR, exprIR] = args;

  const fname = (fIR.kind === 'ref' && fIR.name)
              || (fIR.kind === 'const' && fIR.name);
  const reduce = fname && _AGGREGATE_REDUCTIONS[fname];
  if (!reduce) throw new Error(`aggregate: unknown reduction '${fname}'`);

  if (!axesIR || axesIR.kind !== 'call' || axesIR.op !== 'vector') {
    throw new Error('aggregate: output_axes must be an array literal of axis names');
  }
  const outAxes = (axesIR.args || []).map((a: any) => {
    if (a.kind !== 'axis') throw new Error('aggregate: output_axes entries must be axis names (.name)');
    return a.name;
  });

  const usedAxes = _collectInScopeAxisNames(exprIR);
  const reduceAxes: string[] = [];
  for (const a of usedAxes) if (!outAxes.includes(a)) reduceAxes.push(a);

  // Canonical axis order: [output…, reduce…]. With reduce axes at the
  // end the final reduction is a tail-reduce over a contiguous block.
  const canonicalAxes = [...outAxes, ...reduceAxes];
  const lengths = _inferAggregateAxisLengths(exprIR, canonicalAxes, env);

  // Lift the body to an aligned tensor over the full canonical shape.
  const lifted = _liftAggregateExpr(exprIR, canonicalAxes, lengths, env);
  // Make sure every dim is materialised (no remaining singletons that
  // need broadcasting; the result tensor has every canonical-axis
  // length as its dim). If the body only depended on output axes and
  // some reduce axes were untouched, that dim is still 1 — reduce-by-
  // singleton is a no-op for sum, multiplication by 1 for prod, etc.
  // — we explicitly handle this below.
  const outShape = outAxes.map((a: string) => lengths[a]);
  const reduceShape = reduceAxes.map((a: string) => lengths[a]);
  const outSize = _shapeProd(outShape);

  // Some lifted dims may still be 1 (singleton) where the canonical
  // dim should be `lengths[a]`. Materialise by stretching to the full
  // shape before reducing — strictly necessary only when the
  // reduction is non-linear in the count (e.g. mean / var / std), but
  // doing it uniformly keeps the reduction code simple.
  if (reduceAxes.length === 0) {
    // No reduction needed. The output IS the lifted tensor (with any
    // singleton dims broadcast to their full length).
    const fullShape = outShape.slice();
    const full = _broadcastTo(lifted, fullShape);
    return _flatToNested(full.data, fullShape);
  }

  const fullShape = canonicalAxes.map((a) => lengths[a]);
  const full = _broadcastTo(lifted, fullShape);

  // Tail-reduce: with `full.shape` = [output…, reduce…], each output
  // cell owns a contiguous run of `reduceSize` elements in the flat
  // buffer. Slice each run and apply the reduction.
  const reduceSize = _shapeProd(reduceShape);
  const outData = new Float64Array(outSize);
  const tmp = new Float64Array(reduceSize);
  for (let i = 0; i < outSize; i++) {
    const base = i * reduceSize;
    for (let j = 0; j < reduceSize; j++) tmp[j] = full.data[base + j];
    outData[i] = +reduce(tmp);
  }
  return _flatToNested(outData, outShape);
}

// Materialise a singleton-broadcast tensor to its full shape — copy
// data so every dim equals its target length. After this the data
// buffer is contiguous in row-major order over `targetShape`.
function _broadcastTo(
  t: { data: Float64Array; shape: number[] },
  targetShape: number[],
): { data: Float64Array; shape: number[] } {
  // Fast path: already at target shape.
  let same = true;
  for (let i = 0; i < targetShape.length; i++) {
    if (t.shape[i] !== targetShape[i]) { same = false; break; }
  }
  if (same) return t;
  const N = targetShape.length;
  const outSize = _shapeProd(targetShape);
  const srcStrides = _broadcastStrides(t.shape, targetShape);
  const out = new Float64Array(outSize);
  const coord = new Array(N).fill(0);
  for (let linear = 0; linear < outSize; linear++) {
    let srcOff = 0;
    for (let i = 0; i < N; i++) srcOff += coord[i] * srcStrides[i];
    out[linear] = t.data[srcOff];
    for (let i = N - 1; i >= 0; i--) {
      coord[i]++;
      if (coord[i] < targetShape[i]) break;
      coord[i] = 0;
    }
  }
  return { data: out, shape: targetShape.slice() };
}

// Resolve a function-positional argument used by higher-order ops
// (filter, reduce, scan) into { body, params } regardless of whether
// the IR is an inline `functionof(...)` call or a self-ref to a named
// fn / functionof / kernelof binding. The orchestrator's pre-eval
// attaches env.__resolveFnBody for the ref path; the inline path
// reads `body` / `params` directly off the functionof IR.
function _resolveFn(fnIR: any, env: any) {
  if (!fnIR) return null;
  if (fnIR.kind === 'ref' && fnIR.ns === 'self') {
    if (typeof env.__resolveFnBody !== 'function') return null;
    const fn = env.__resolveFnBody(fnIR.name);
    if (!fn) return null;
    return fn;
  }
  if (fnIR.kind === 'call' && fnIR.op === 'functionof'
      && Array.isArray(fnIR.params) && fnIR.body) {
    return {
      body: fnIR.body,
      params: fnIR.params,
      paramKwargs: fnIR.paramKwargs,
      paramName: fnIR.params[0],
    };
  }
  return null;
}

// broadcast(callable, inputs…) per spec §04 + commit 87c9be1.
//
//   - Collection inputs (arrays / typed arrays / Value rank ≥ 1) are
//     the only things iterated over.
//   - All collections must have the SAME number of axes — the engine
//     never inserts leading/trailing axes implicitly; `addaxes(...)`
//     is the explicit, layout-agnostic way to align ranks.
//   - Along each axis sizes must be equal or 1; size-1 expands by
//     repetition (singleton broadcast).
//   - Non-collection inputs (scalars, Value rank-0, function/kernel
//     objects) are loop-invariant — held constant. The callable is
//     loop-invariant in exactly this sense; it is not special with
//     respect to iteration.
//   - Records and tuples are not allowed as broadcast inputs.
//   - No collection inputs ⇒ a single call (everything constant).
//
// Result is a nested JS array of the broadcast shape (1-D → a flat
// JS array, matching the rest of the value-evaluator's array forms).
function _broadcastApply(fn: any, inputs: any, env: any): any {
  const P = fn.params.length;
  const slots = new Array(P);
  for (let i = 0; i < P; i++) {
    const v = inputs[i];
    if (valueLib.isValue(v)) {
      if (v.shape.length === 0) slots[i] = { coll: false, val: v.data[0] };
      else slots[i] = { coll: true, V: v };
    } else if (v != null && v.BYTES_PER_ELEMENT !== undefined
               && typeof v.length === 'number') {
      slots[i] = { coll: true, V: valueLib.asValue(v) };
    } else if (Array.isArray(v)) {
      slots[i] = { coll: true, V: valueLib.asValue(v) };
    } else if (typeof v === 'number' || typeof v === 'boolean'
               || (v && typeof v === 'object'
                   && typeof v.re === 'number' && typeof v.im === 'number')) {
      slots[i] = { coll: false, val: v };                  // scalar / complex scalar
    } else if (v && typeof v === 'object'
               && v.body !== undefined && Array.isArray(v.params)) {
      slots[i] = { coll: false, val: v };                  // fn/kernel held constant
    } else if (v && typeof v === 'object') {
      // A plain object with named fields is a record (or tuple).
      throw new Error('broadcast: records and tuples are not allowed as '
        + 'broadcast inputs (spec §04)');
    } else {
      slots[i] = { coll: false, val: v };
    }
  }

  const colls: any[] = [];
  for (let i = 0; i < P; i++) if (slots[i].coll) colls.push(slots[i].V);

  const elemEnv = Object.assign({}, env);
  if (colls.length === 0) {
    // No collection arguments → a single call (spec).
    for (let p = 0; p < P; p++) elemEnv[fn.params[p]] = slots[p].val;
    return evaluateExpr(fn.body, elemEnv);
  }

  // Same number of axes across all collections — no implicit insertion.
  const rank = colls[0].shape.length;
  for (let c = 1; c < colls.length; c++) {
    if (colls[c].shape.length !== rank) {
      throw new Error('broadcast: all collection arguments must have the '
        + 'same number of axes (got ranks ' + rank + ' and '
        + colls[c].shape.length + '); the engine does not insert axes '
        + 'implicitly — use addaxes(...) to align them (spec §04)');
    }
  }
  // Per-axis broadcast size: each collection's size must be that size
  // or 1 (singleton, expanded by repetition).
  const bshape = new Array(rank);
  for (let a = 0; a < rank; a++) {
    let sz = 1;
    for (let c = 0; c < colls.length; c++) {
      const s = colls[c].shape[a];
      if (s !== 1) {
        if (sz !== 1 && sz !== s) {
          throw new Error('broadcast: incompatible sizes on axis ' + a
            + ' (' + sz + ' vs ' + s + '); sizes must be equal or 1');
        }
        sz = s;
      }
    }
    bshape[a] = sz;
  }

  // Row-major strides per collection on its OWN shape; a size-1 axis
  // gets stride 0 so it auto-repeats under the broadcast index.
  const strides = colls.map(function(c) {
    const st = new Array(rank);
    let acc = 1;
    for (let a = rank - 1; a >= 0; a--) {
      st[a] = (c.shape[a] === 1) ? 0 : acc;
      acc *= c.shape[a];
    }
    return st;
  });
  const collOf = new Array(P);
  for (let p = 0, k = 0; p < P; p++) collOf[p] = slots[p].coll ? k++ : -1;

  const idx = new Array(rank).fill(0);
  // Flat row-major pack of leaf evaluations + a sticky "all numeric"
  // flag. If the body of `fn` returns only finite numbers / booleans
  // for every cell, the broadcast result is a shape-explicit Value
  // (engine-concepts §2.1) — same convention as `vector(...)`,
  // `fill(...)`, etc. If any cell returns a non-numeric value
  // (e.g. a complex object, a record), we fall back to the legacy
  // nested-JS-array form.
  const total = bshape.reduce((a: number, b: number) => a * b, 1);
  const flat = new Float64Array(total);
  let allNumeric = true;
  let pos = 0;
  function recur(axis: any): any {
    if (axis === rank) {
      for (let p = 0; p < P; p++) {
        if (collOf[p] < 0) { elemEnv[fn.params[p]] = slots[p].val; continue; }
        const ci = collOf[p], V = colls[ci], st = strides[ci];
        let off = 0;
        for (let a = 0; a < rank; a++) off += idx[a] * st[a];
        elemEnv[fn.params[p]] = (V.dtype === 'complex' && V.im)
          ? { re: V.data[off], im: V.im[off] }
          : V.data[off];
      }
      return evaluateExpr(fn.body, elemEnv);
    }
    const out = new Array(bshape[axis]);
    for (let i = 0; i < bshape[axis]; i++) {
      idx[axis] = i;
      const child: any = recur(axis + 1);
      out[i] = child;
      if (axis === rank - 1) {
        // Leaf level — `child` is one cell's value.
        if (allNumeric) {
          const t = typeof child;
          if (t === 'number') flat[pos] = child;
          else if (t === 'boolean') flat[pos] = child ? 1 : 0;
          else allNumeric = false;
          pos++;
        }
      }
    }
    return out;
  }
  const nested = recur(0);
  if (allNumeric && total === pos) {
    return { shape: bshape.slice(), data: flat };
  }
  return nested;
}

function evaluateCall(ir: any, env: any): any {
  const op = ir.op;
  if (op === 'aggregate') {
    return _evalAggregate(ir, env);
  }
  if (op in ARITH_OPS) {
    const args = (ir.args || []).map((a: any) => evaluateExpr(a, env));
    return (ARITH_OPS as any)[op](...args);
  }
  // tuple_get(<tuple-expr>, <slot lit>) — engine-internal projection
  // emitted by the analyzer's multi-LHS rewriter. Evaluates the tuple
  // child to a JS array and indexes by the literal slot. The slot is
  // always a numeric literal at IR-construction time.
  if (op === 'tuple_get') {
    const args = ir.args || [];
    if (args.length !== 2) {
      throw new Error(`evaluateExpr: tuple_get expects 2 args, got ${args.length}`);
    }
    const t: any = evaluateExpr(args[0], env);
    const i: any = evaluateExpr(args[1], env) | 0;
    if (!Array.isArray(t)) {
      throw new Error(`evaluateExpr: tuple_get target is not an array (got ${typeof t})`);
    }
    return t[i];
  }
  // tuple(...) — JS array of evaluated args. Used for surface
  // `(a, b, ...)` literals and as an intermediate value when downstream
  // code projects via tuple_get.
  if (op === 'tuple') {
    return (ir.args || []).map((a: any) => evaluateExpr(a, env));
  }
  // fixed(x) — identity at runtime. Spec §03 value types: "fixed(x)
  // is semantically identical to identity(x) during FlatPPL code
  // evaluation, it is merely a hint to tooling." Tools (e.g. the
  // viewer's preset-point recognition) inspect the IR's op === 'fixed'
  // tag to honor the hint; the evaluator just unwraps.
  if (op === 'fixed') {
    const args = ir.args || [];
    if (args.length !== 1) {
      throw new Error(`evaluateExpr: fixed expects 1 arg, got ${args.length}`);
    }
    return evaluateExpr(args[0], env);
  }
  // get_field(obj, "name") — record field access. Lowered from
  // surface `obj.field`. Second arg is always a literal string.
  if (op === 'get_field') {
    const args = ir.args || [];
    if (args.length !== 2) {
      throw new Error(`evaluateExpr: get_field expects 2 args, got ${args.length}`);
    }
    const obj: any = evaluateExpr(args[0], env);
    const key: any = args[1] && args[1].kind === 'lit' ? args[1].value : evaluateExpr(args[1], env);
    if (obj == null || typeof obj !== 'object') {
      throw new Error(`evaluateExpr: get_field target is not a record (got ${typeof obj})`);
    }
    return obj[key];
  }
  // get(container, sel, ...) / get0(...) — spec §07 unified element /
  // subset / axis-slice access. All of `v[i]`, `A[i,j]`, `A[:,j]`,
  // `v[[1,3]]`, `get(r,"a")`, tuple integer index lower to `get`
  // (1-based); `get0` is the 0-based variant (`get0(v,0)` → first).
  // Pure deterministic value op over plain JS values (the form
  // fixed-phase pre-eval produces): numbers, JS/typed arrays, records
  // (objects), tuples (arrays). Implemented HERE — in the single
  // deterministic-evaluator authority — rather than as yet another
  // ad-hoc indexer, so fixed-phase expressions containing indexing
  // (`Gamma(shape = tau[1] + 1.0, …)`) no longer dead-end.
  if (op === 'get' || op === 'get0') {
    const args = ir.args || [];
    if (args.length < 2) {
      throw new Error(`evaluateExpr: ${op} expects a container and at least one selector`);
    }
    const oneBased = (op === 'get');
    const container: any = evaluateExpr(args[0], env);
    // `all` (axis slice) and `only` (singleton-axis selector) are
    // selector TOKENS, not values — keep them as sentinels so applyGet
    // can dispatch on identity rather than re-parsing a const IR.
    const ALL  = Symbol('all');
    const ONLY = Symbol('only');
    const sels = args.slice(1).map((a: any) => {
      if (a && a.kind === 'const' && a.name === 'all')  return ALL;
      if (a && a.kind === 'const' && a.name === 'only') return ONLY;
      return evaluateExpr(a, env);
    });
    const isArrayLike = (c: any) => Array.isArray(c) || ArrayBuffer.isView(c)
                                  || valueLib.isValue(c);
    // Slice a Value along its leading axis. Rank-1 returns a scalar
    // number; rank≥2 returns a Value of one rank lower (sharing the
    // underlying data — no copy when the rest of the shape is
    // contiguous in row-major order, which is the engine's convention).
    function _sliceLeading(v: any, idx: number): any {
      let dense = v;
      if (v.t === 'T' || v.t === 'A' || v.struct !== undefined) {
        dense = valueLib.densify(v);
        // densify preserves the transpose tag; for indexing we need
        // physical contiguous-leading-axis data, so apply the tag
        // (rare path — only matters when M came in as transpose-tagged).
        if (dense.t === 'T' || dense.t === 'A') {
          // For rank-2 only — apply transpose physically.
          const m = dense.shape[dense.shape.length - 2];
          const n = dense.shape[dense.shape.length - 1];
          if (dense.shape.length === 2) {
            const src = dense.data;
            const out = new Float64Array(m * n);
            for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
              out[i * n + j] = src[j * m + i];
            }
            dense = { shape: [m, n], data: out };
          }
        }
      }
      const shape = dense.shape;
      if (shape.length === 1) {
        return dense.data[idx];   // scalar number
      }
      const tail = shape.slice(1);
      const tailLen = tail.reduce((a: number, b: number) => a * b, 1);
      const sub = dense.data.subarray(idx * tailLen, (idx + 1) * tailLen);
      return { shape: tail, data: sub };
    }
    const applyGet = (c: any, ss: any): any => {
      if (ss.length === 0) return c;
      const s = ss[0], rest = ss.slice(1);
      if (s === ALL) {                       // whole axis: map over it
        if (!isArrayLike(c)) {
          throw new Error(`evaluateExpr: ${op} ':' axis-slice target is not an array`);
        }
        // For shape-explicit Value, slice along the leading axis and
        // recurse the rest of the selectors over each row.
        if (valueLib.isValue(c)) {
          const n = c.shape[0];
          const out: any[] = [];
          for (let i = 0; i < n; i++) out.push(applyGet(_sliceLeading(c, i), rest));
          return out;
        }
        const ca = c as any;
        const out: any[] = [];
        for (let i = 0; i < ca.length; i++) out.push(applyGet(ca[i], rest));
        return out;
      }
      if (s === ONLY) {                      // singleton axis: must be size 1
        if (!isArrayLike(c)) {
          throw new Error(`evaluateExpr: ${op} '!' (only) axis target is not an array`);
        }
        if (valueLib.isValue(c)) {
          if (c.shape[0] !== 1) {
            throw new Error(`evaluateExpr: ${op}(..., only, ...) requires the indexed `
              + `axis to have length 1, got length ${c.shape[0]}`);
          }
          return applyGet(_sliceLeading(c, 0), rest);
        }
        const ca = c as any;
        if (ca.length !== 1) {
          throw new Error(`evaluateExpr: ${op}(..., only, ...) requires the indexed `
            + `axis to have length 1, got length ${ca.length}`);
        }
        return applyGet(ca[0], rest);
      }
      // Value-typed subset selectors arise from `[i, j, k]` literals
      // that lower to `vector(...)` and now produce a rank-1 Value.
      // Unwrap to its data view.
      if (valueLib.isValue(s) && s.shape.length === 1) {
        return Array.from(s.data).map((si: any) => applyGet(c, [si, ...rest]));
      }
      if (Array.isArray(s)) {                // subset selection
        // All-string subset of a record → a sub-record (spec §07);
        // otherwise an array subset along this axis.
        if (s.every((e) => typeof e === 'string')
            && c != null && typeof c === 'object' && !isArrayLike(c)) {
          const r: Record<string, any> = {};
          for (const k of s) r[k] = applyGet(c[k], rest);
          return r;
        }
        return s.map((si: any) => applyGet(c, [si, ...rest]));
      }
      if (typeof s === 'string') {           // record field
        if (c == null || typeof c !== 'object' || isArrayLike(c)) {
          throw new Error(`evaluateExpr: ${op}(…, "${s}") target is not a record`);
        }
        return applyGet(c[s], rest);
      }
      if (typeof s === 'number' || typeof s === 'boolean') {
        if (!isArrayLike(c)) {
          throw new Error(`evaluateExpr: ${op} index target is not an array (got ${typeof c})`);
        }
        const sn = s as number | boolean;
        const idx = (oneBased ? ((sn as number) | 0) - 1 : ((sn as number) | 0));
        if (valueLib.isValue(c)) {
          const len = c.shape[0];
          if (idx < 0 || idx >= len) {
            throw new Error(`evaluateExpr: ${op} index ${sn} out of `
              + `bounds for length ${len}`);
          }
          return applyGet(_sliceLeading(c, idx), rest);
        }
        const ca = c as any;
        if (idx < 0 || idx >= ca.length) {
          throw new Error(`evaluateExpr: ${op} index ${oneBased ? sn : sn} out of `
            + `bounds for length ${ca.length}`);
        }
        return applyGet(ca[idx], rest);
      }
      throw new Error(`evaluateExpr: ${op} unsupported selector ${String(s)}`);
    };
    return applyGet(container, sels);
  }
  // Shape functions (spec §07 Approximation functions). All three take
  // kwargs so they don't pass through ARITH_OPS; dispatch explicitly.
  // The kwargs are `coefficients` / `values` (a fixed-phase array) and
  // `x` (the evaluation point), plus `edges` for stepwise.
  if (op === 'polynomial') {
    const kw = ir.kwargs || {};
    const coeffsRaw = kw.coefficients != null ? evaluateExpr(kw.coefficients, env)
                                           : evaluateExpr(ir.args[0], env);
    const x = kw.x != null ? evaluateExpr(kw.x, env)
                           : evaluateExpr(ir.args[1], env);
    // Unwrap shape-explicit Value → underlying typed-array view.
    const coeffs = valueLib.isValue(coeffsRaw) ? coeffsRaw.data : coeffsRaw;
    // Σ a_i · x^i, evaluated Horner-style for numerical stability.
    let acc = 0;
    for (let i = coeffs.length - 1; i >= 0; i--) acc = acc * x + coeffs[i];
    return acc;
  }
  if (op === 'bernstein') {
    // Bernstein basis on [0, 1]: f(x) = Σ_{k=0..n} a_k · C(n, k) · x^k · (1-x)^{n-k}
    // where n = length(coefficients) - 1. Numerically stable for x ∈ [0,1].
    const kw = ir.kwargs || {};
    const coeffsRaw: any = kw.coefficients != null ? evaluateExpr(kw.coefficients, env)
                                           : evaluateExpr(ir.args[0], env);
    const x = kw.x != null ? evaluateExpr(kw.x, env)
                           : evaluateExpr(ir.args[1], env);
    const coeffs: any = valueLib.isValue(coeffsRaw) ? coeffsRaw.data : coeffsRaw;
    const n = coeffs.length - 1;
    if (n < 0) return 0;
    // Binomial coefficients via Pascal recurrence (n choose k).
    // O(n) per call after pre-computing the row.
    const oneMinusX = 1 - x;
    let acc = 0, binom = 1;
    let xk = 1, omxn = Math.pow(oneMinusX, n);
    // omxn starts as (1-x)^n; will be divided by (1-x) at each step.
    // Handle x === 1 (oneMinusX = 0) carefully.
    if (oneMinusX === 0) {
      // All bases vanish except the last: f(1) = coeffs[n].
      return coeffs[n];
    }
    for (let k = 0; k <= n; k++) {
      acc += coeffs[k] * binom * xk * omxn;
      xk *= x;
      omxn /= oneMinusX;
      binom = binom * (n - k) / (k + 1);
    }
    return acc;
  }
  if (op === 'selectbins') {
    // selectbins(edges, region, counts) — keep counts for bins whose
    // interval [edges[i], edges[i+1]] intersects `region`. Returns a
    // shorter count array per spec §07: no fractional-bin clipping,
    // bins are either fully included or fully excluded.
    //
    // Region is a set IR. We accept literal `interval(lo, hi)` and the
    // named real sets (`reals` / `posreals` / `nonnegreals` /
    // `unitinterval`) inline here — anything richer would require
    // crossing into orchestrator.parseSetIR territory, which the
    // sampler intentionally doesn't depend on.
    const kw = ir.kwargs || {};
    const edgesRaw    = kw.edges    != null ? evaluateExpr(kw.edges,    env)
                                         : evaluateExpr(ir.args[0],  env);
    const regionIR    = kw.region   != null ? kw.region                 : ir.args[1];
    const countsRaw   = kw.counts   != null ? evaluateExpr(kw.counts,   env)
                                         : evaluateExpr(ir.args[2],  env);
    const edges  = valueLib.isValue(edgesRaw)  ? edgesRaw.data  : edgesRaw;
    const counts = valueLib.isValue(countsRaw) ? countsRaw.data : countsRaw;
    const [lo, hi] = regionBoundsFromIR(regionIR, env);
    const n = counts.length;
    if (edges.length !== n + 1) {
      throw new Error('selectbins: edges length must equal counts length + 1');
    }
    const out: any[] = [];
    for (let i = 0; i < n; i++) {
      // Bin [edges[i], edges[i+1]] intersects [lo, hi] iff
      //   edges[i] ≤ hi  AND  edges[i+1] ≥ lo
      if (edges[i] <= hi && edges[i + 1] >= lo) {
        out.push(counts[i]);
      }
    }
    return out;
  }
  if (op === 'broadcast') {
    // broadcast(f, A, B, ...) per spec §04 higher-order ops. Apply f
    // elementwise over arrays. Two surface shapes accepted:
    //   broadcast(f, A, B, ...)              — positional arrays
    //   broadcast(f, x = A, y = B, ...)      — kwargs naming f's params
    // Each array must have the same length; no auto-broadcast.
    //
    // Kernel-broadcast (stochastic case) is NOT handled here — that
    // would produce an array-valued measure, which lives in the
    // materialiser path, not the value evaluator.
    const args   = ir.args   || [];
    const kwargs = ir.kwargs || {};
    if (args.length < 1) throw new Error('broadcast: no function argument');
    const fn = _resolveFn(args[0], env);
    if (!fn) throw new Error('broadcast: first arg must be a function');
    const kwargKeys = Object.keys(kwargs);
    const sources = new Array(fn.params.length);
    if (kwargKeys.length > 0) {
      // kwargs form: match by surface kwarg name first (paramKwargs),
      // fall back to internal placeholder name (params).
      for (let i = 0; i < fn.params.length; i++) {
        const surface = (fn.paramKwargs && fn.paramKwargs[i]) || fn.params[i];
        if (kwargs[surface] != null) sources[i] = kwargs[surface];
        else if (kwargs[fn.params[i]] != null) sources[i] = kwargs[fn.params[i]];
        else throw new Error('broadcast: no argument for parameter '
          + (surface || fn.params[i]));
      }
    } else {
      const posArgs = args.slice(1);
      if (posArgs.length !== fn.params.length) {
        throw new Error('broadcast: expected ' + fn.params.length
          + ' positional arrays, got ' + posArgs.length);
      }
      for (let i = 0; i < fn.params.length; i++) sources[i] = posArgs[i];
    }
    const inputs: any = sources.map(s => evaluateExpr(s, env));
    return _broadcastApply(fn, inputs, env);
  }
  if (op === 'reduce') {
    // reduce(f, xs) per spec §07. f is a binary function; xs is a
    // non-empty vector. Computes f(...f(f(xs[0], xs[1]), xs[2])..., xs[n-1]).
    // The first element of xs is the initial accumulator (no separate
    // init arg, unlike scan).
    const args = ir.args || [];
    if (args.length !== 2) {
      throw new Error('reduce: expected 2 args (function, xs), got ' + args.length);
    }
    const fn = _resolveFn(args[0], env);
    if (!fn || fn.params.length !== 2) {
      throw new Error('reduce: function arg must be a binary function');
    }
    const xsRaw: any = evaluateExpr(args[1], env);
    const xs: any = valueLib.isValue(xsRaw) ? xsRaw.data : xsRaw;
    if (!Array.isArray(xs) && !(xs && xs.BYTES_PER_ELEMENT)) {
      throw new Error('reduce: xs must be a vector');
    }
    if (xs.length === 0) {
      throw new Error('reduce: empty vector has no initial value');
    }
    const elemEnv = Object.assign({}, env);
    let acc: any = xs[0];
    for (let i = 1; i < xs.length; i++) {
      elemEnv[fn.params[0]] = acc;
      elemEnv[fn.params[1]] = xs[i];
      acc = evaluateExpr(fn.body, elemEnv);
    }
    return acc;
  }
  if (op === 'scan') {
    // scan(f, init, xs) per spec §07. Left scan: produces a vector of
    // intermediate accumulator values, one per element of xs.
    //   out[i] = f(out[i-1], xs[i])   with out[-1] = init
    // Result has the same length as xs.
    const args = ir.args || [];
    if (args.length !== 3) {
      throw new Error('scan: expected 3 args (function, init, xs), got ' + args.length);
    }
    const fn = _resolveFn(args[0], env);
    if (!fn || fn.params.length !== 2) {
      throw new Error('scan: function arg must be a binary function');
    }
    const init = evaluateExpr(args[1], env);
    const xsRaw: any = evaluateExpr(args[2], env);
    const xs: any = valueLib.isValue(xsRaw) ? xsRaw.data : xsRaw;
    if (!Array.isArray(xs) && !(xs && xs.BYTES_PER_ELEMENT)) {
      throw new Error('scan: xs must be a vector');
    }
    // Scan result has the same length as xs and is a rank-1 vector per
    // spec §07 — return a shape-explicit Value.
    const n = xs.length;
    const out: Float64Array = new Float64Array(n);
    const elemEnv = Object.assign({}, env);
    let acc = init;
    for (let i = 0; i < n; i++) {
      elemEnv[fn.params[0]] = acc;
      elemEnv[fn.params[1]] = xs[i];
      acc = evaluateExpr(fn.body, elemEnv);
      out[i] = acc === true ? 1 : acc === false ? 0 : +acc;
    }
    return { shape: [n], data: out };
  }
  if (op === 'filter') {
    // filter(pred, data) per spec §07. pred can be a named function
    // binding (orchestrator.inlineFilterLift lifts inline fn(...) to
    // anon) OR an inline functionof call IR. _resolveFn handles both.
    const args = ir.args || [];
    if (args.length !== 2) {
      throw new Error('filter: expected 2 args (predicate, data), got ' + args.length);
    }
    const fn = _resolveFn(args[0], env);
    if (!fn || fn.params.length !== 1) {
      throw new Error('filter: predicate must be a unary function');
    }
    const dataRaw = evaluateExpr(args[1], env);
    const data = valueLib.isValue(dataRaw) ? dataRaw.data : dataRaw;
    if (!Array.isArray(data) && !(data && data.BYTES_PER_ELEMENT)) {
      throw new Error('filter: data must be a vector (got '
        + (data === null ? 'null' : typeof data) + ')');
    }
    const elemEnv = Object.assign({}, env);
    const kept: any[] = [];
    for (let i = 0; i < data.length; i++) {
      elemEnv[fn.params[0]] = data[i];
      const keep = evaluateExpr(fn.body, elemEnv);
      if (keep) kept.push(data[i]);
    }
    // Filter result is a rank-1 vector per spec §07 — return as Value.
    const out = new Float64Array(kept.length);
    for (let i = 0; i < kept.length; i++) {
      out[i] = kept[i] === true ? 1 : kept[i] === false ? 0 : +kept[i];
    }
    return { shape: [kept.length], data: out };
  }
  if (op === 'bincounts') {
    // bincounts(bins, data) — count data points falling into bins.
    // 1D case: bins is a vector of n+1 edges defining n bins;
    // returns an n-vector of integer counts. Bin semantics: left-
    // closed / right-open for interior bins, the LAST bin is also
    // right-closed so a point exactly at the upper boundary lands in
    // the last bin (spec §07 Binning). Points outside [bins[0], bins[n]]
    // are ignored.
    // Multi-D case (bins is a record of edge vectors): not yet
    // supported — falls through to an explicit error.
    const kw = ir.kwargs || {};
    const binsRaw: any = kw.bins != null ? evaluateExpr(kw.bins, env)
                                 : evaluateExpr(ir.args[0], env);
    const dataRaw = kw.data != null ? evaluateExpr(kw.data, env)
                                 : evaluateExpr(ir.args[1], env);
    const bins = valueLib.isValue(binsRaw) ? binsRaw.data : binsRaw;
    const data = valueLib.isValue(dataRaw) ? dataRaw.data : dataRaw;
    if (!bins || typeof bins.length !== 'number'
        || (bins.length > 0 && typeof bins[0] !== 'number')) {
      throw new Error('bincounts: multi-dimensional binning not yet supported');
    }
    const n: any = bins.length - 1;
    if (n < 0) throw new Error('bincounts: bins must have at least 1 edge');
    // Spec §07 result: rank-1 integer count vector. Returned as a
    // shape-explicit Value (engine-concepts §2.1).
    const counts = new Float64Array(n);
    const last = n - 1;
    const lo = bins[0], hi = bins[n];
    for (let i = 0; i < data.length; i++) {
      const x = data[i];
      if (x < lo || x > hi) continue;
      // Linear scan — adequate for typical bin counts (≤ few hundred).
      for (let j = 0; j < n; j++) {
        if (x >= bins[j] && (x < bins[j + 1] || (j === last && x === bins[j + 1]))) {
          counts[j]++;
          break;
        }
      }
    }
    return { shape: [n], data: counts };
  }
  if (op === 'stepwise') {
    // Piecewise constant: edges has length n+1, values has length n.
    // For x in [edges[i], edges[i+1]) return values[i]; right edge
    // is closed for the last bin.
    const kw = ir.kwargs || {};
    const edgesRaw  = kw.edges  != null ? evaluateExpr(kw.edges,  env)
                                     : evaluateExpr(ir.args[0], env);
    const valuesRaw: any = kw.values != null ? evaluateExpr(kw.values, env)
                                     : evaluateExpr(ir.args[1], env);
    const x      = kw.x      != null ? evaluateExpr(kw.x,      env)
                                     : evaluateExpr(ir.args[2], env);
    const edges  = valueLib.isValue(edgesRaw)  ? edgesRaw.data  : edgesRaw;
    const values: any = valueLib.isValue(valuesRaw) ? valuesRaw.data : valuesRaw;
    const n = values.length;
    if (edges.length !== n + 1) {
      throw new Error('stepwise: edges length must equal values length + 1');
    }
    if (x < edges[0] || x > edges[n]) return NaN;
    // Linear scan — fine for typical bin counts (≤ few hundred); a
    // binary search would help for very long edge vectors.
    for (let i = 0; i < n; i++) {
      if (x >= edges[i] && (x < edges[i + 1] || (i === n - 1 && x === edges[i + 1]))) {
        return values[i];
      }
    }
    return NaN;
  }
  // record(...) — build a JS object from the call's `fields` array
  // (lowered from surface `record(a=x, b=y)`). Field values are
  // evaluated; keys are static names from the fields array.
  if (op === 'record' && Array.isArray(ir.fields)) {
    const out: Record<string, any> = {};
    for (const f of ir.fields) out[f.name] = evaluateExpr(f.value, env);
    return out;
  }
  // rnginit(<bytes>) — produces a fresh Philox state from a byte vector
  // via FNV-1a-based key derivation (rng.seedFromBytes).
  if (op === 'rnginit') {
    const args = ir.args || [];
    if (args.length !== 1) {
      throw new Error(`evaluateExpr: rnginit expects 1 arg, got ${args.length}`);
    }
    const seedRaw: any = evaluateExpr(args[0], env);
    if (!isByteVector(seedRaw)) {
      throw new Error(`evaluateExpr: rnginit seed must be a byte vector (array of integers in 0..255)`);
    }
    const seed = valueLib.isValue(seedRaw) ? Array.from(seedRaw.data) : seedRaw;
    return rng.seedFromBytes(seed);
  }
  // rngstate(<bytes>) — round-trip an externally-serialized state.
  // Round-trip semantics: rngstate(bytesFromState(s)) ≡ s.
  if (op === 'rngstate') {
    const args = ir.args || [];
    if (args.length !== 1) {
      throw new Error(`evaluateExpr: rngstate expects 1 arg, got ${args.length}`);
    }
    const bytesRaw: any = evaluateExpr(args[0], env);
    if (!isByteVector(bytesRaw)) {
      throw new Error(`evaluateExpr: rngstate bytes must be a byte vector (array of integers in 0..255)`);
    }
    const bytes = valueLib.isValue(bytesRaw) ? Array.from(bytesRaw.data) : bytesRaw;
    return rng.stateFromBytes(bytes);
  }
  // rand(<state>, <measure-IR>) — generate a sample from a closed
  // measure with explicit state threading. Per spec §sec:random rand
  // returns a tuple (value, new_state). The measure arg is NOT
  // evaluated as a value — it's passed verbatim to traceeval.walk
  // which handles iid / joint / record / leaf-distribution recursion
  // and threads the rng state. Refused for measures rand can't sample
  // (weighted, logweighted, bayesupdate, multivariate truncation) by
  // traceeval's dispatch — those throw with a clear message.
  if (op === 'rand') {
    return evaluateRand(ir, env);
  }
  // builtin_logdensityof(kernel, kernel_input, x) — FlatPDL primitive,
  // spec §07 §sec:measure-eval-prims. The single per-kernel
  // log-density ABI: dispatch lives in `density-prims.ts`'s
  // `builtinLogdensityof`. `kernel` resolves to a bare distribution
  // name via three accepted IR shapes:
  //   - {kind:'lit', value:'Normal'}              — string-literal name
  //   - {kind:'call', op:'Normal', args:[]}        — no-arg kernel call
  //   - {kind:'ref', name:'Normal'}               — bare identifier
  //     (only when the name is a known distribution; the lowerer
  //     leaves bare distribution names as refs)
  // `kernel_input` is a record matching the kernel's kwarg interface.
  if (op === 'builtin_logdensityof') {
    const args = ir.args || [];
    if (args.length !== 3) {
      throw new Error(`evaluateExpr: builtin_logdensityof expects 3 args `
        + `(kernel, kernel_input, x), got ${args.length}`);
    }
    const kernelName = _resolveKernelName(args[0], env);
    const kernelInput = evaluateExpr(args[1], env);
    const x = evaluateExpr(args[2], env);
    const densityPrims = require('./density-prims.ts');
    return densityPrims.builtinLogdensityof(kernelName, kernelInput, x);
  }
  // Canonical transports (spec §07 §sec:measure-eval-prims). Same
  // signature for all four (kernel, kernel_input, point); the four
  // functions dispatch into density-prims's `builtinTouniform` etc.
  if (op === 'builtin_touniform' || op === 'builtin_fromuniform'
      || op === 'builtin_tonormal' || op === 'builtin_fromnormal') {
    const args = ir.args || [];
    if (args.length !== 3) {
      throw new Error(`evaluateExpr: ${op} expects 3 args `
        + `(kernel, kernel_input, point), got ${args.length}`);
    }
    const kernelName = _resolveKernelName(args[0], env);
    const kernelInput = evaluateExpr(args[1], env);
    const point = evaluateExpr(args[2], env);
    const densityPrims = require('./density-prims.ts');
    if (op === 'builtin_touniform')   return densityPrims.builtinTouniform(kernelName, kernelInput, point);
    if (op === 'builtin_fromuniform') return densityPrims.builtinFromuniform(kernelName, kernelInput, point);
    if (op === 'builtin_tonormal')    return densityPrims.builtinTonormal(kernelName, kernelInput, point);
    return densityPrims.builtinFromnormal(kernelName, kernelInput, point);
  }
  // builtin_sample(rngstate, kernel, kernel_input, [n, m, ...]) — FlatPDL
  // primitive. We synthesise the measure IR `kernel(...kernel_input)`
  // (optionally wrapped in `iid(..., size)` for the trailing dims) and
  // route through traceeval, exactly the path `rand` takes — same
  // RNG-state threading, same per-distribution math.
  if (op === 'builtin_sample') {
    const args = ir.args || [];
    if (args.length < 3) {
      throw new Error(`evaluateExpr: builtin_sample expects at least 3 args `
        + `(rngstate, kernel, kernel_input, [n, m, ...]), got ${args.length}`);
    }
    const state: any = evaluateExpr(args[0], env);
    if (!state || typeof state !== 'object' || !state.key || !state.counter) {
      throw new Error(`evaluateExpr: builtin_sample's rngstate must be an rngstate `
        + `(got ${typeof state})`);
    }
    const kernelName = _resolveKernelName(args[1], env);
    const mIR = _synthSampleMeasureIR(kernelName, args[2], args.slice(3), env);
    const opts: any = {};
    if (env && typeof env.__resolveMeasureRef === 'function') {
      opts.resolveMeasureRef = env.__resolveMeasureRef;
    }
    if (env && typeof env.__resolveValueRef === 'function') {
      opts.resolveValueRef = env.__resolveValueRef;
    }
    const r: any = getTraceeval().walk(state, mIR, env, opts);
    return [r.value, r.state];
  }
  // Calls to other built-ins, user-defined functions, etc. aren't expected
  // inside distribution parameters in the visualizer's scope. The
  // orchestrator should pre-evaluate those and supply concrete numbers
  // via env if the model uses them.
  throw new Error(
    `evaluateExpr: call op '${op}' not evaluable in sampler context — ` +
    `the orchestrator should pre-resolve this`
  );
}

// Predicate guarding rnginit / rngstate: argument must be an iterable of
// integers in [0, 255]. Float64Arrays whose entries happen to be small
// integers are also accepted; shape-explicit Values are unwrapped.
function isByteVector(x: any) {
  if (valueLib.isValue(x)) x = x.data;
  if (!x || typeof x !== 'object' || typeof x.length !== 'number') return false;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    if (typeof v !== 'number' || !Number.isFinite(v)) return false;
    if (v < 0 || v > 255 || Math.floor(v) !== v) return false;
  }
  return true;
}

// Local require to break the cyclic dependency (traceeval.js requires
// sampler.js for the leaf-distribution machinery; we lazy-import here so
// the back-reference doesn't blow up module loading).
let _traceeval: any = null;
function getTraceeval() {
  if (!_traceeval) _traceeval = require('./traceeval.ts');
  return _traceeval;
}

function evaluateRand(ir: any, env: any): any {
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`evaluateExpr: rand expects 2 args (state, measure), got ${args.length}`);
  }
  const state: any = evaluateExpr(args[0], env);
  if (!state || typeof state !== 'object' || !state.key || !state.counter) {
    throw new Error(`evaluateExpr: rand's first arg must be an rngstate (got ${typeof state})`);
  }
  // resolveMeasureRef: when the measure arg is a self-ref to another
  // binding (e.g. `m_alias`), traceeval needs an IR for that binding.
  // The orchestrator supplies a closure when calling us; the bare-
  // sampler path resolves only literal measure calls inline. If a ref
  // shows up without resolveMeasureRef, traceeval throws a clear error.
  //
  // resolveValueRef: parallel hook for value-position refs inside the
  // measure's distribution params (e.g. `Normal(mu = ref a)` where `a`
  // is a stochastic ancestor). When env doesn't pre-carry the value
  // the resolver samples it on demand, threading state. Same passthrough
  // pattern: the orchestrator builds the closure, evaluateRand just
  // forwards what's been parked on env.
  const opts: any = {};
  if (env && typeof env.__resolveMeasureRef === 'function') {
    opts.resolveMeasureRef = env.__resolveMeasureRef;
  }
  if (env && typeof env.__resolveValueRef === 'function') {
    opts.resolveValueRef = env.__resolveValueRef;
  }
  const r: any = getTraceeval().walk(state, args[1], env, opts);
  return [r.value, r.state];
}

// Build a synthetic measure IR `kernel(...kernelInputKwargs)` from a
// kernel name + the kernel_input IR + optional trailing dim IRs. Used
// by `builtin_sample(rngstate, kernel, kernel_input, n, m, ...)` to
// route through `traceeval.walk` — same path `rand(state, m)` takes.
//
// `kernelInputIR` MUST be either:
//   - an inline `record(...)` call (fields → kernel call kwargs), or
//   - a call whose `.kwargs` field is already in the right shape (which
//     the surface grammar doesn't emit but tests may build directly).
// Refs to record-typed bindings are an open follow-up; for now we
// throw a clear error so users see the supported surface.
function _synthSampleMeasureIR(kernelName: string, kernelInputIR: any,
                                dimIRs: any[], _env: any): any {
  let kwargs: Record<string, any>;
  if (kernelInputIR && kernelInputIR.kind === 'call'
      && kernelInputIR.op === 'record'
      && Array.isArray(kernelInputIR.fields)) {
    kwargs = {};
    for (const f of kernelInputIR.fields) kwargs[f.name] = f.value;
  } else if (kernelInputIR && kernelInputIR.kind === 'call'
             && kernelInputIR.kwargs
             && typeof kernelInputIR.kwargs === 'object') {
    kwargs = kernelInputIR.kwargs;
  } else {
    throw new Error('builtin_sample: kernel_input must be an inline record(...) '
      + 'literal (refs to record-typed bindings are an open follow-up)');
  }
  const distIR: any = { kind: 'call', op: kernelName, kwargs };
  if (dimIRs.length === 0) return distIR;
  // Wrap in iid(M, size). Size is a literal vector when there are
  // multiple trailing dims; bare expr when there is exactly one.
  const sizeIR = (dimIRs.length === 1)
    ? dimIRs[0]
    : { kind: 'call', op: 'vector', args: dimIRs };
  return { kind: 'call', op: 'iid', args: [distIR, sizeIR] };
}

// Resolve the `kernel` arg of `builtin_logdensityof(kernel, …)` to a
// bare distribution name. Per spec §07 §sec:measure-eval-prims, kernel
// is a built-in kernel constructor (the spec restricts to built-ins; an
// arbitrary measure expression is disallowed). We accept the three
// shapes the lowerer can produce:
//   - {kind:'lit', value:'<name>'}              — string literal
//   - {kind:'call', op:'<name>', args:[…]}       — kernel call (we
//     read the op; any args/kwargs on this node are ignored because the
//     `kernel_input` arg carries the real input)
//   - {kind:'ref', name:'<name>'}                — bare identifier
// `<name>` must be a known kernel (REGISTRY entry or a multivariate
// MV_DENSITY_FNS entry); anything else is rejected.
function _resolveKernelName(ir: any, _env: any): string {
  if (!ir) throw new Error('builtin_logdensityof: kernel arg is missing');
  let name: string | null = null;
  if (ir.kind === 'lit' && typeof ir.value === 'string') name = ir.value;
  else if (ir.kind === 'call' && typeof ir.op === 'string') name = ir.op;
  else if (ir.kind === 'ref' && typeof ir.name === 'string') name = ir.name;
  if (name == null) {
    throw new Error('builtin_logdensityof: kernel must be a bare distribution name, '
      + `got IR kind=${ir.kind}`);
  }
  const densityPrims = require('./density-prims.ts');
  if (!densityPrims.isBuiltinKernel(name)) {
    throw new Error(`builtin_logdensityof: '${name}' is not a known built-in kernel`);
  }
  return name;
}

module.exports = {
  // Primary API
  rand,
  makeSampler,
  makeParametricSampler,
  density,
  makeAnalytical,
  evaluateExpr,
  evaluateExprN,
  isBatch,

  // Shared with traceeval.js — both modules need to dispatch on the
  // distribution registry and resolve param expressions against an env.
  lookupDistribution,
  resolveParams,
  makePhiloxPrngAdapter,

  // Introspection
  isKnownDistribution,
  listDistributions,

  // Internal — exported for tests.
  _internal: {
    REGISTRY,
    ARITH_OPS,
    ARITH_OPS_N,
    makePhiloxPrngAdapter,
    resolveParams,
    lookupDistribution,
  },
};
