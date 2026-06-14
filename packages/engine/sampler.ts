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
  resolveParamsN,
  hasRandNFn,
  regionBoundsFromIR,
  makePhiloxPrngAdapter,
  makeBulkUniformPrngAdapter,
  readSupport,
} = registry;

// Unified op-declaration registry + dispatcher (engine-concepts §18).
// Phase 2: declared ops route through `opsModule.dispatch` instead of
// `ARITH_OPS[op](...)` at evaluateCall time. The ARITH_OPS entries
// remain for ops that haven't migrated yet; the conformance suite
// (test/ops-conformance.test.ts) pins both paths to agree. Migration
// finishes when no op's ARITH_OPS entry is needed any more.
const opsModule = require('./ops.ts');
require('./ops-declarations.ts');     // side-effect: trigger op registrations
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
 * in at factory time. This is the per-i-params FALLBACK path: the stdlib
 * factory closure is built ONCE (with only the prng bound), then each
 * `drawWith(env)` call resolves the IR's param expressions against the
 * current env and invokes the closure with those numerics. The randNFn
 * family (Normal/LogNormal/Uniform/Exponential) is folded onto
 * makeBulkSampler's `perI` mode instead; this remains the per-i
 * realisation for everything else (Gamma/Beta/Dirac/… and non-interval
 * Uniforms).
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
 * Bulk-sample `n` iid draws from a fixed-parameter distribution in one
 * call (commit 3 of the RNG splitting thread — Option C hybrid).
 *
 * Two paths:
 *
 *   1. If the registry entry exposes `randNFn`, take the explicit
 *      batched path: one `philoxN*` cipher batch + vectorised
 *      transform, no per-atom JS function dispatch. Used for Normal /
 *      LogNormal (Box-Muller — NOT bit-equivalent to scalar randFn)
 *      and Uniform / Exponential (bit-equivalent to scalar randFn).
 *
 *   2. Otherwise fall back to the scalar randFn loop, but feed it
 *      through `makeBulkUniformPrngAdapter` so the Philox cipher
 *      cost is amortized across N draws. Rejection samplers (Gamma,
 *      Beta, …) consume variable uniforms per atom — the adapter
 *      handles that transparently via its on-demand refill.
 *
 * `out` is optional; if provided the caller's Float64Array(n) is
 * reused, avoiding a buffer alloc on the hot path.
 *
 * Returns `{ samples: Float64Array, state: PhiloxState }`. State is
 * threaded back so the caller can continue without losing position.
 *
 * Per-draw-upstream params (per-i kwargs) ARE supported via the optional
 * trailing `perI = { refArrays, count, repeat }` arg: for the randNFn
 * family it resolves each param into a per-atom column (resolveParamsN),
 * atom-major-expands for repeat>1, and runs one philoxN* draw through the
 * SAME per-element transform as the static path — the single batched leaf
 * math (engine-concepts §11, Q2 fold). A non-randNFn dist or a Uniform
 * whose support isn't interval(lo,hi) returns `{ unhandled: true }` so the
 * caller falls back to makeParametricSampler. Omit `perI` for static params.
 */
// Expand a per-atom param column to atom-major length n = count*repeat:
// atom i's value is shared across its `repeat` inner draws, matching the
// output layout out[i*repeat + j] (the iid(M, repeat) per-outer-atom
// fan-out). A scalar column is atom-independent and stays scalar (the
// transform reads it as a stride-0 broadcast — no allocation). When
// repeat === 1 the column is already length count === n, so it's
// returned unchanged.
function _expandPerAtomColumn(col: any, count: number, repeat: number): any {
  if (typeof col === 'number') return col;
  if (repeat === 1) return col;
  const n = count * repeat;
  const exp = new Float64Array(n);
  for (let i = 0; i < count; i++) {
    const v = col[i];
    const base = i * repeat;
    for (let j = 0; j < repeat; j++) exp[base + j] = v;
  }
  return exp;
}

function makeBulkSampler(
  state: any, measureIR: any, env: any, n: number, out?: Float64Array,
  perI?: { refArrays: any, count: number, repeat?: number },
) {
  const entry = lookupDistribution(measureIR);

  // Per-i batched mode: at least one distribution parameter varies per
  // outer atom (it references an upstream draw). Only the randNFn family
  // can batch this — one philoxN* draw + the per-element transform loop
  // (the SAME loop body the static path uses, fed per-atom param columns
  // instead of scalars). This retires the worker's per-draw stdlib
  // randFn.factory (ziggurat) second realisation for those dists
  // (engine-concepts §11, Q2 fold). Anything we can't batch — a
  // non-randNFn dist, or a Uniform whose support isn't interval(lo, hi)
  // — returns { unhandled: true } so the worker keeps its per-draw
  // scalar makeParametricSampler loop (one endpoint, not one algorithm).
  if (perI) {
    if (typeof entry.randNFn !== 'function') return { unhandled: true } as any;
    const cols = resolveParamsN(measureIR, entry, perI.refArrays, perI.count, env);
    if (cols === null) return { unhandled: true } as any;
    const repeat = perI.repeat || 1;
    const expanded = cols.map((c: any) => _expandPerAtomColumn(c, perI.count, repeat));
    const r = entry.randNFn(state, expanded, n, out);
    return { samples: r.out, state: r.state };
  }

  const params = resolveParams(measureIR, entry, env);

  if (typeof entry.randNFn === 'function') {
    // Path 1: explicit batched. randNFn handles the philoxN* call +
    // transform internally, returning { state, out }.
    const r = entry.randNFn(state, params, n, out);
    return { samples: r.out, state: r.state };
  }

  // Path 2: bulk-uniform adapter feeds the scalar randFn loop. The
  // adapter pre-fills N*4 uniforms in one cipher batch and serves
  // them via .prng() — closure dispatch becomes a buffer read, and
  // rejection samplers refill on demand.
  const prng = makeBulkUniformPrngAdapter(state, n);
  const sampler = entry.randFn.factory(...params, { prng });
  const dest = out ?? new Float64Array(n);
  for (let i = 0; i < n; i++) dest[i] = sampler();
  return { samples: dest, state: prng.getState() };
}

/**
 * Batched leaf-distribution draw — the FlatPDL `builtin_sample` leaf
 * endpoint, realised over the in-process bulk sampler.
 *
 * Draws prod(dims) iid samples from a SCALAR built-in leaf distribution
 * in ONE `makeBulkSampler` batch (no per-draw walker loop), then shapes
 * the flat result to the FlatPDL `builtin_sample` / spec-§06 `iid`
 * output contract:
 *
 *   dims = []        → a scalar number                      (one draw)
 *   dims = [n]       → a length-n JS number[]               (vector of scalars)
 *   dims = [n,m,...] → a shape-explicit Value {shape, data} (rank ≥ 2, spec §03)
 *
 * State is THREADED (not split): returns the advanced rng state so the
 * caller continues — matching `rand`'s `(value, new_state)` and
 * `builtin_sample`'s `(X, new_rngstate)` contracts (spec §07). This is
 * the single batched realisation of the leaf endpoint shared by
 * `builtin_sample`, `rand(state, iid(<leaf>, dims))`, and a bare scalar
 * leaf draw `rand(state, <leaf>)` (walkLeaf delegates here at dims=[]),
 * so all leaf paths in the value-position rand primitive agree
 * bit-for-bit and leaves are batched draws per engine-concepts §11.
 *
 * Caller owns param resolution: any value-position refs in `distIR`'s
 * kwargs must already be resolved into `env` (iid semantics — the leaf's
 * params are drawn ONCE and shared across all prod(dims) draws). `distIR`
 * must be a REGISTRY (scalar) distribution call; multivariate kernels are
 * derived measures handled by the materialiser, not this leaf endpoint.
 */
function sampleLeafN(state: any, distIR: any, env: any, dims: number[]) {
  const total = dims.reduce((p: number, d: number) => p * d, 1);
  // 0-dim and n-dim share one path: a scalar draw is just a batch of 1,
  // so the leaf math (randNFn / bulk-uniform) is identical regardless of
  // whether the caller asked for a scalar or a vector.
  const n = (dims.length === 0) ? 1 : total;
  const r = makeBulkSampler(state, distIR, env, n);
  if (dims.length === 0) return { value: r.samples[0], state: r.state };
  // 1-D: a plain JS number[] (a vector of scalars, spec §03) — the
  // shape-explicit Value form is reserved for rank ≥ 2. Matches the
  // historic per-draw walkIid output so consumers index it unchanged.
  if (dims.length === 1) return { value: Array.from(r.samples), state: r.state };
  // Multi-axis: an explicit rank-≥2 array as a shape-explicit Value
  // (spec §03 "vectors of vectors are not matrices" — flat storage + a
  // dims descriptor; the form downstream mul/add/sub recognise).
  return { value: { shape: dims.slice(), data: r.samples }, state: r.state };
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
// Extracted to sampler-eval-batched.ts (sampler split; module map in ARCHITECTURE.md).
// The module exposes the ARITH_OPS_N table + dispatcher functions;
// sampler.ts calls `initARITHOPSN(ARITH_OPS)` below (after ARITH_OPS is
// fully defined) to populate the table.

const _evalBatched = require('./sampler-eval-batched.ts');
const {
  evaluateExprN, ARITH_OPS_N, isBatch,
} = _evalBatched;

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
// Extracted to sampler-complex.ts (sampler split; module map in ARCHITECTURE.md).

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

// Shape-aware add/sub/neg dispatcher (used inline in the
// ARITH_OPS table below). engine-concepts §20 / TODO Phase 1:
// whenever either operand is a Value, route through value-ops —
// no bare-number unwrap branch. valueOps handles rank-0 + rank-N
// via NumPy-style broadcasting (stride-0 read on rank-0 operands),
// so a rank-0 constant Value flowing into a batched op produces
// the right result without pre-replicating it to a length-N buffer.
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
    return valueOps[opName](valueLib.asValue(a), valueLib.asValue(b));
  }
  return scalarFn(a, b);
}

// Apply a column-wise reduction to a table (spec §07 "Table reductions":
// "When `sum`, `mean`, or `var` is applied to a table, the reduction
// operates column-wise and returns a record whose fields are the column
// names and values are the per-column reductions"). Used by the
// reduction-table entries (sum, mean, var, std, prod, maximum, minimum)
// to dispatch table inputs without each entry re-implementing the
// column iteration.
function _tableReduce(t: any, reduceCol: (col: any) => any): any {
  const out: Record<string, any> = {};
  for (const k in t.columns) out[k] = reduceCol(t.columns[k]);
  return out;
}

const ARITH_OPS = {
  add: (a: any, b: any) => _shapeAwareBinop('add', (x: any, y: any) => x + y, a, b, _cAdd),
  sub: (a: any, b: any) => _shapeAwareBinop('sub', (x: any, y: any) => x - y, a, b, _cSub),
  // mul: engine-concepts §20 / TODO Phase 1 — whenever either operand
  // is a Value, route through value-ops.mul (which handles rank-0,
  // rank-1 vec/transposed-vec inner/outer, rank-2 matmul/matvec via
  // Klein-4 transpose-tag dispatch, plus rank-0 broadcasting). Both
  // bare numbers → scalar JS fast path. No bare-number scalar unwrap
  // branch — Value scalars produce Value outputs.
  mul: (a: any, b: any) => {
    if (_isComplex(a) || _isComplex(b)) {
      return _cMul(_toComplex(a), _toComplex(b));
    }
    if (valueLib.isValue(a) || valueLib.isValue(b)) {
      return valueOps.mul(valueLib.asValue(a), valueLib.asValue(b));
    }
    return a * b;
  },
  // Spec §07: `div(a, b) = ⌊a/b⌋` — integer floor division (integer
  // domain, integer result), distinct from `/` which lowers to `divide`
  // (true division). Reachable only via an explicit `div(a, b)` call
  // (the `/` operator lowers to `divide`).
  div: (a: any, b: any) => Math.floor(a / b),
  divide: (a: any, b: any) => {
    if (_isComplex(a) || _isComplex(b)) {
      return _cDiv(_toComplex(a), _toComplex(b));
    }
    return a / b;
  },
  mod: (a: any, b: any) => a % b,
  neg: (a: any) => {
    if (_isComplex(a)) return _cNeg(a);
    if (valueLib.isValue(a)) return valueOps.neg(a);
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
  // Migrated to ops-declarations.ts as kind='variadic' (engine-
  // concepts §18 Phase 5b).
  vector: (...xs: any[]) => opsModule.dispatch('vector', xs),
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
    // Accept BOTH input forms (engine-concepts §2.1 C7):
    //   - Legacy JS array of inner Values / Float64Arrays / JS arrays
    //   - Tagged nested-vector Value: shape=[N, k], outerRank=1.
    //     This is what `vector(V1, V2, …)` now produces when its
    //     args are themselves Values.
    // Both denote the same vec-of-vecs; output is always a flat
    // rank-2 matrix Value (no outerRank tag — every axis is a
    // loop axis at the matrix level).
    if (valueLib.isNestedVectorValue(vs) && vs.shape.length === 2) {
      // Fast path: storage layout is already row-major-rows. Strip
      // the outerRank tag to expose the value AS a matrix.
      return { shape: vs.shape.slice(), data: vs.data };
    }
    if (!Array.isArray(vs) || vs.length === 0) {
      return { shape: [0, 0], data: new Float64Array(0) };
    }
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
    // Symmetric to rowstack but writes the input columns into the
    // output matrix's columns (i.e. transposes the storage layout).
    // For a tagged nested-vector Value the row-major storage holds
    // each "column" contiguously, so we must walk it explicitly
    // rather than just stripping the tag.
    if (valueLib.isNestedVectorValue(vs) && vs.shape.length === 2) {
      const colsN = vs.shape[0];
      const m = vs.shape[1];
      const out = new Float64Array(m * colsN);
      for (let c = 0; c < colsN; c++) {
        for (let i = 0; i < m; i++) {
          out[i * colsN + c] = vs.data[c * m + i];
        }
      }
      return { shape: [m, colsN], data: out };
    }
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
  // Migrated to ops-declarations.ts as kind='rank-polymorphic'
  // (engine-concepts §18 Phase 5a). The dispatcher hands the input
  // to `logical` as-is — atom-batched callers wrap with explicit
  // `broadcast(fn(transpose(_)), …)`.
  transpose: (M: any) => opsModule.dispatch('transpose', [M]),
  adjoint:   (M: any) => opsModule.dispatch('adjoint',   [M]),
  // Migrated to ops-declarations.ts (engine-concepts §18 Phase 2);
  // these entries delegate so direct ARITH_OPS callers (mat-*, tests)
  // keep working with the single-sourced implementation.
  trace:      (M: any): any => opsModule.dispatch('trace', [M]),
  diagmat:    (v: any): any => opsModule.dispatch('diagmat', [v]),
  self_outer: (v: any): any => opsModule.dispatch('self_outer', [v]),
  // Migrated to ops-declarations.ts (engine-concepts §18 Phase 2);
  // these entries delegate so direct ARITH_OPS callers keep working.
  cross:     (a: any, b: any): any => opsModule.dispatch('cross', [a, b]),
  det:       (A: any): any => opsModule.dispatch('det', [A]),
  logabsdet: (A: any): any => opsModule.dispatch('logabsdet', [A]),
  inv:       (A: any): any => opsModule.dispatch('inv', [A]),
  // Engine-internal metricsum symmetry guard (engine-concepts §23) —
  // dispatch shim mirroring `inv`. Throws on non-symmetric / non-
  // square metrics; returns the input unchanged on success.
  _ms_check_symmetric: (A: any): any => opsModule.dispatch('_ms_check_symmetric', [A]),
  // Migrated to ops-declarations.ts as kind='rank-polymorphic'
  // (engine-concepts §18 Phase 5a): A is square matrix, b is
  // vector OR matrix. The dispatcher passes both through to logical
  // as-is.
  linsolve: (A: any, b: any): any => opsModule.dispatch('linsolve', [A, b]),
  // Migrated to ops-declarations.ts (engine-concepts §18 Phase 2).
  lower_cholesky: (A: any): any => opsModule.dispatch('lower_cholesky', [A]),
  row_gram:       (A: any): any => opsModule.dispatch('row_gram',       [A]),
  col_gram:       (A: any): any => opsModule.dispatch('col_gram',       [A]),
  // quadform(A, x) — quadratic form xᵀ A x (spec §07). Implemented
  // as the closed-form composition `dot(x, mul(A, x))`. Numerically
  // stable; reuses the existing matvec + inner-product dispatch
  // (auto-batched: if x is atom-batched, mul(A, x) routes through
  // the rank-2 × rank-1 atom-aware variant and the dot folds via
  // the elementwise + sum pair).
  quadform:       (A: any, x: any): any => {
    valueLib.requireMatrix(valueLib.asValue(A), 'quadform');
    const Ax = valueOps.mul(valueLib.asValue(A), valueLib.asValue(x));
    // dot(x, Ax) = sum(x ⊙ Ax).
    const elem = valueOps.mulElem(valueLib.asValue(x), Ax);
    // Reduce to scalar — sum over all dims (rank-1 input has only
    // one).
    const data = elem.data;
    let s = 0;
    for (let i = 0; i < data.length; i++) s += data[i];
    return s;
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
  // Migrated to ops-declarations.ts as kind='variadic' (engine-
  // concepts §18 Phase 5b).
  cat: (...xs: any[]) => opsModule.dispatch('cat', xs),
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
    if (a && a.__table__ === true) return _tableReduce(a, (ARITH_OPS as any).sum);
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
    if (a && a.__table__ === true) return _tableReduce(a, (ARITH_OPS as any).mean);
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
    if (a && a.__table__ === true) return _tableReduce(a, (ARITH_OPS as any).prod);
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
    // First-class table (spec §03): row count is explicit on the marker.
    if (a && a.__table__ === true) return a.nrows || 0;
    if (a && !Array.isArray(a) && typeof a === 'object'
        && a.BYTES_PER_ELEMENT === undefined) {
      // Legacy record-of-columns shape. Length = first column's length.
      // Kept for the rare callers that hand-construct this form.
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
  // indicesof / indicesof0 (spec §07) — axis indices for arrays /
  // vectors / tables. Implementation materialises a Float64Array
  // (engines may use lazy integer ranges per spec; JS has no
  // built-in range type, so a packed array is the simplest
  // realisation). 1-D / table → single rank-1 Value of length N.
  // Multi-axis array → tuple of per-axis index Values.
  indicesof:  (a: any) => _indicesOfImpl(a, /*zeroBased=*/ false),
  indicesof0: (a: any) => _indicesOfImpl(a, /*zeroBased=*/ true),
  maximum: (a: any) => {
    if (a && a.__table__ === true) return _tableReduce(a, (ARITH_OPS as any).maximum);
    const arr = _arrLike(a);
    let m = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    return m;
  },
  minimum: (a: any) => {
    if (a && a.__table__ === true) return _tableReduce(a, (ARITH_OPS as any).minimum);
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
    if (a && a.__table__ === true) return _tableReduce(a, (ARITH_OPS as any).var);
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
    if (a && a.__table__ === true) return _tableReduce(a, (ARITH_OPS as any).std);
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

/**
 * indicesof / indicesof0 implementation (spec §07). Produces axis
 * indices for arrays / vectors / tables, with a 0-based variant.
 *
 *   - 1-D vector / typed array         → rank-1 Value of length N
 *   - Multi-axis Value (rank ≥ 2)      → tuple of per-axis index
 *                                        Values (one per axis)
 *   - Nested JS array                  → tuple of per-axis indices
 *   - Record-of-columns (table)        → row indices (rank-1 Value)
 *   - Scalar / unknown shape           → empty rank-1 Value
 *
 * Engines may emit integer ranges per spec; the JS engine has no
 * built-in lazy range, so we materialise a Float64Array.
 */
function _indicesOfImpl(a: any, zeroBased: boolean): any {
  const offset = zeroBased ? 0 : 1;
  function _packRange(n: number): any {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = i + offset;
    return { shape: [n], data: out };
  }
  function _multiAxisTuple(dims: number[]): any {
    if (dims.length === 1) return _packRange(dims[0]);
    return dims.map(_packRange);
  }
  if (valueLib.isValue(a)) {
    if (a.shape.length === 0) return _packRange(0);
    if (a.shape.length === 1) return _packRange(a.shape[0]);
    return _multiAxisTuple(a.shape);
  }
  if (a == null) return _packRange(0);
  if (typeof a === 'number') return _packRange(0);
  if (a.BYTES_PER_ELEMENT !== undefined && typeof a.length === 'number') {
    return _packRange(a.length);
  }
  if (Array.isArray(a)) {
    const dims: number[] = [];
    let cur: any = a;
    while (Array.isArray(cur)) {
      dims.push(cur.length);
      cur = cur[0];
    }
    return _multiAxisTuple(dims);
  }
  // Record-of-columns ⇒ table. Row count = first column's length.
  if (typeof a === 'object') {
    for (const k in a) {
      if (!Object.prototype.hasOwnProperty.call(a, k)) continue;
      const col = a[k];
      if (valueLib.isValue(col)) return _packRange(col.shape[0]);
      if (col && typeof col.length === 'number') return _packRange(col.length);
    }
  }
  return _packRange(0);
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
// Populate ARITH_OPS_N from ARITH_OPS via sampler-eval-batched.
// =====================================================================
// One-shot init — must run AFTER ARITH_OPS is fully defined (above) and
// BEFORE anything calls evaluateExprN. The dispatcher functions
// (evaluateExprN / _evalN / _batchedApproximation / _perAtomFallback)
// were destructured from sampler-eval-batched at the top of this file;
// they close over the same ARITH_OPS_N object this call populates.
_evalBatched.initARITHOPSN(ARITH_OPS);

// =====================================================================
// Linear-algebra helpers (textbook algorithms; small-matrix sized)
// =====================================================================
// Extracted to sampler-linalg.ts (sampler split; module map in ARCHITECTURE.md).

const _linalg = require('./sampler-linalg.ts');
const _broadcastShape = require('./broadcast-shape.ts');
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
// Extracted to sampler-aggregate.ts (sampler split; module map in ARCHITECTURE.md).

const _aggregate = require('./sampler-aggregate.ts');
const { _evalAggregate } = _aggregate;


// Dispatch a `(call target=({ns: <module-alias>, name: X}) args,
// kwargs)` to the standard-module registry's JS impl. See
// `evaluateCall`'s cross-module-call paragraph for context.
//
// `env.__moduleRegistry[<alias>]` provides the resolved (stdName,
// stdCompat) for the module binding. The registry lookup
// (`standard-modules.ts`) then yields the BindingDescriptor whose
// `sig.inputs` is the ordered parameter list. Call arguments
// (kwargs + positional, post-alias-resolution) get matched against
// `sig.inputs` by name / position, evaluated via `evaluateExpr`,
// then passed positionally to `desc.impl`.
//
// Errors are raised at the failing layer (no registry entry, no
// binding, missing param, …) with sources pointing at the call IR's
// `loc`.
function _evaluateStandardModuleCall(ir: any, env: any): any {
  const modAlias = ir.target.ns;
  const bindingName = ir.target.name;
  const reg = env && env.__moduleRegistry;
  if (!reg || !reg[modAlias]) {
    throw new Error("evaluateCall: cross-module ref '"
      + modAlias + "." + bindingName
      + "' has no entry in env.__moduleRegistry — host did not "
      + "thread the module registry into the worker session env");
  }
  const modInfo = reg[modAlias];
  if (modInfo.kind !== 'standard') {
    throw new Error("evaluateCall: cross-module ref '"
      + modAlias + "." + bindingName + "' resolves to a "
      + modInfo.kind + " module — only standard_module is currently "
      + "dispatchable at runtime (load_module wiring is a follow-up)");
  }
  const stdMods = require('./standard-modules.ts');
  const entry = stdMods.lookupStandardModule(modInfo.stdName, modInfo.stdCompat);
  if (!entry) {
    throw new Error("evaluateCall: standard module '"
      + modInfo.stdName + "@" + modInfo.stdCompat
      + "' is not provided by this engine");
  }
  const desc = entry.bindings.get(bindingName);
  if (!desc) {
    throw new Error("evaluateCall: standard module '"
      + modInfo.stdName + "@" + modInfo.stdCompat
      + "' has no binding '" + bindingName + "'");
  }
  if (typeof desc.impl !== 'function') {
    throw new Error("evaluateCall: standard module binding '"
      + modInfo.stdName + "." + bindingName
      + "' has no runtime impl (it may be a value-only descriptor)");
  }
  const inputs = (desc.sig && desc.sig.inputs) || [];
  const kwargs = ir.kwargs || {};
  const positional = ir.args || [];
  const callArgs: any[] = [];
  for (let i = 0; i < inputs.length; i++) {
    const paramName = inputs[i].name;
    let argIR: any;
    if (paramName in kwargs)          argIR = kwargs[paramName];
    else if (i < positional.length)   argIR = positional[i];
    else throw new Error("evaluateCall: '" + modAlias + "."
      + bindingName + "' missing argument '" + paramName + "'");
    callArgs.push(evaluateExpr(argIR, env));
  }
  return desc.impl.apply(null, callArgs);
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
  // Module-aliased ref `(%ref <modAlias> <name>)` — the alias-
  // resolution pass canonicalised an `arr.f = mod.foo; broadcast(arr.f,
  // …)` callsite into a direct module-namespaced callable. Synthesise
  // a wrapper functionof on the fly: parameter list comes from the
  // standard-module descriptor's `sig.inputs`; body is a single call
  // to the module's binding with each param bound as a %local ref.
  // The body's call then dispatches through evaluateCall's cross-
  // module-call path (`_evaluateStandardModuleCall`).
  if (fnIR.kind === 'ref' && fnIR.ns && fnIR.ns !== '%local'
      && env && env.__moduleRegistry && env.__moduleRegistry[fnIR.ns]) {
    return _synthStdModuleFn(fnIR, env);
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

// Build a synthetic functionof-shaped wrapper for a module-aliased
// callable ref. Each sig.input becomes a %local param; the body
// is a single call to `(%ref <modAlias> <name>)` with positional args
// bound to the params in declaration order. Used by `_resolveFn` so
// higher-order ops (broadcast, reduce, scan, filter) can iterate a
// std-module callable element-wise without bespoke per-op support.
function _synthStdModuleFn(refIR: any, env: any): any {
  const modInfo = env.__moduleRegistry[refIR.ns];
  if (!modInfo || modInfo.kind !== 'standard') return null;
  const stdMods = require('./standard-modules.ts');
  const entry = stdMods.lookupStandardModule(modInfo.stdName, modInfo.stdCompat);
  if (!entry) return null;
  const desc = entry.bindings.get(refIR.name);
  if (!desc || !desc.sig || !Array.isArray(desc.sig.inputs)) return null;
  // Use the descriptor's actual param names. They map kwarg-by-name
  // when the call passes kwargs; positionally when args are
  // positional. The synthetic body refs each by name as a `%local`.
  const params = desc.sig.inputs.map((i: any) => i.name);
  const paramKwargs = params.slice();
  const args = params.map((p: string) => ({ kind: 'ref', ns: '%local', name: p }));
  const body = {
    kind: 'call',
    target: { ns: refIR.ns, name: refIR.name },
    args,
  };
  return {
    body,
    params,
    paramKwargs,
    paramName: params[0],
  };
}

// broadcast(callable, inputs…) per spec §04 + commit 87c9be1, extended
// for nested-vector arguments (see test/nested-broadcast.test.ts for
// the conceptual model).
//
//   - Collection inputs are iterated. There are two kinds of collection:
//
//       * "flat" collection — a Value of rank ≥ 1. Every axis is a
//         loop axis; per cell the body sees a scalar.
//
//       * "nested-vector" collection — a JS array whose elements are
//         themselves Values (or further nested arrays). The OUTER JS-
//         array layer is one loop axis; per cell the body sees the
//         inner element WHOLE. This is the Ref-wrap idiom: `[C]` (a
//         length-1 nested-vector wrapping the rank-1 C) holds C
//         constant under broadcast while keeping the outer axis count
//         in agreement with the spec's "same number of axes" rule.
//
//     Mixing flat and nested-vector collections is fine as long as
//     their outer axis counts agree (the spec rule). Nested-vector
//     contributes one outer axis (its JS-array length); flat
//     contributes shape.length outer axes.
//
//   - All collections must have the SAME number of (outer) axes — the
//     engine never inserts leading/trailing axes implicitly;
//     `addaxes(...)` is the explicit way to align ranks.
//
//   - Along each axis, sizes must be equal or 1; size-1 expands by
//     repetition (singleton broadcast).
//
//   - Non-collection inputs (scalars, Value rank-0, function / kernel
//     objects) are loop-invariant. The callable is loop-invariant in
//     exactly this sense; it is not special with respect to iteration.
//
//   - Records and tuples are not allowed as broadcast inputs.
//
//   - No collection inputs ⇒ a single call (everything constant).
//
// Result is either a shape-explicit Value (when every cell returns a
// finite scalar / boolean and there are no nested-vector loop axes —
// the engine-concepts §2.1 convention) or a nested JS array of the
// broadcast shape.
function _broadcastApply(fn: any, inputs: any, env: any): any {
  const P = fn.params.length;
  const slots = new Array(P);
  for (let i = 0; i < P; i++) {
    const v = inputs[i];
    slots[i] = _classifyBroadcastSlot(v);
  }

  const elemEnv = Object.assign({}, env);
  const colls: any[] = [];
  for (let i = 0; i < P; i++) if (slots[i].coll) colls.push(slots[i]);

  if (colls.length === 0) {
    // No collection arguments → a single call (spec).
    for (let p = 0; p < P; p++) elemEnv[fn.params[p]] = slots[p].val;
    return evaluateExpr(fn.body, elemEnv);
  }

  // Same number of OUTER axes across all collections. For flat slots
  // that's shape.length; for nested-vector slots it's the JS-array
  // nesting depth (any depth — `test/nested-broadcast.test.ts` covers
  // the multi-level "v2" cases, e.g. `[[C]]` over a 2-D collection).
  const rank = colls[0].outerRank;
  for (let c = 1; c < colls.length; c++) {
    if (colls[c].outerRank !== rank) {
      throw new Error('broadcast: all collection arguments must have the '
        + 'same number of axes (got outer ranks ' + rank + ' and '
        + colls[c].outerRank + '); the engine does not insert axes '
        + 'implicitly — use addaxes(...) to align them (spec §04)');
    }
  }

  // Per-axis broadcast size: each collection's size must be that size
  // or 1 (singleton, expanded by repetition).
  const bshape = new Array(rank);
  for (let a = 0; a < rank; a++) {
    let sz = 1;
    for (let c = 0; c < colls.length; c++) {
      const s = colls[c].outerShape[a];
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

  const collOf = new Array(P);
  for (let p = 0, k = 0; p < P; p++) collOf[p] = slots[p].coll ? k++ : -1;

  const idx = new Array(rank).fill(0);
  const total = bshape.reduce((a: number, b: number) => a * b, 1);
  // Fast-path packing: if NO nested-vector slot is present AND every
  // cell returns a finite scalar / boolean, pack into a shape-explicit
  // Value. With a nested-vector slot the per-cell result need not be
  // scalar (often a Value itself), so we go straight to the stacking
  // path that builds a Value-of-stacked-cells when possible, or a
  // nested JS array otherwise.
  const anyNested = colls.some((c: any) => c.kind === 'nested');
  const flat = anyNested ? null : new Float64Array(total);
  let allNumeric = !anyNested;
  let pos = 0;
  // Cell results stashed when at least one slot is a nested-vector
  // (post-hoc stacking).
  const cellResults: any[] = anyNested ? new Array(total) : (null as any);
  function recur(axis: any): any {
    if (axis === rank) {
      for (let p = 0; p < P; p++) {
        if (collOf[p] < 0) { elemEnv[fn.params[p]] = slots[p].val; continue; }
        elemEnv[fn.params[p]] = colls[collOf[p]].getCell(idx);
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
        if (anyNested) {
          cellResults[pos++] = child;
        } else if (allNumeric) {
          const t = typeof child;
          if (t === 'number') (flat as any)[pos] = child;
          else if (t === 'boolean') (flat as any)[pos] = child ? 1 : 0;
          else allNumeric = false;
          pos++;
        } else {
          pos++;
        }
      }
    }
    return out;
  }
  const nested = recur(0);
  if (!anyNested && allNumeric && total === pos) {
    return { shape: bshape.slice(), data: flat };
  }
  if (anyNested) {
    // Try to stack cell results into a shape-explicit Value. Works
    // when every cell is either a scalar number / boolean OR a Value
    // with the SAME inner shape across cells (gufunc-style stacking).
    const stacked = _tryStackBroadcastCells(cellResults, bshape);
    if (stacked) return stacked;
    return cellResults;
  }
  return nested;
}

// Broadcast-shape classification and per-cell stacking live in
// `broadcast-shape.ts` so they can be re-used by future broadcast
// consumers (kernel-broadcast lowering, atom-batched fast paths,
// aggregate's outer-axis classification at `get(arr, .i, .j)`
// sites) without dragging the whole sampler with them. The two
// helpers below are thin local aliases.
const _classifyBroadcastSlot = _broadcastShape.classifyAxisStructure;
const _tryStackBroadcastCells = _broadcastShape.tryStackBroadcastCells;

function evaluateCall(ir: any, env: any): any {
  // Cross-module call dispatch — `(call target=({ns: <module-alias>,
  // name: X}) args, kwargs)` where <module-alias> is a load_module /
  // standard_module binding. Resolves through the env's module
  // registry to the standard-modules.ts BindingDescriptor and
  // invokes the JS impl with the resolved args.
  //
  // The registry is threaded via `env.__moduleRegistry` (a map
  // `<alias> → {kind: 'standard', stdName, stdCompat}`); the host
  // (viewer / test setup / materialiser) pushes it via setEnv
  // alongside fixedValues. Cross-module callsites that bypass the
  // registry (e.g. the alias-resolution pass didn't get to canonicalise
  // because the binding was injected post-hoc) surface a clear error
  // here rather than failing silently.
  if (ir && ir.target && ir.target.ns && ir.target.ns !== 'self') {
    return _evaluateStandardModuleCall(ir, env);
  }
  const op = ir.op;
  // aggregate migrated to OpDecl as kind='higher-order' (engine-
  // concepts §18.9 Phase 5c). The OpDecl's logical delegates to
  // `_evalAggregate` from sampler-aggregate.ts — same dispatch as
  // before, routed through the unified op-decl framework for
  // consistency.
  if (op === 'aggregate') {
    return opsModule.dispatchHigherOrder(op, ir, {
      env,
      evaluateExpr,
      resolveFn: _resolveFn,
    });
  }
  if (op in ARITH_OPS) {
    const args = (ir.args || []).map((a: any) => evaluateExpr(a, env));
    // Phase 2: declared ops dispatch through ops.ts (engine-concepts
    // §18). Both paths return the same result for the declared ops;
    // conformance suite pins this. ARITH_OPS entries that haven't
    // migrated still take the legacy path below.
    if (opsModule.isDeclared(op)) {
      return opsModule.dispatch(op, args);
    }
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
    // Table column access (spec §03): a __table__-marked value's
    // fields are in `obj.columns`, not directly on `obj`.
    if (obj.__table__ === true && obj.columns) {
      return obj.columns[key];
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
      // Table dispatch (spec §03). A table is a `__table__`-marked
      // object holding a column-name → array map plus a row count.
      // - String selector → column array.
      // - Integer selector → record-of-column-values at that row.
      // - ALL → iterate rows, returning an array of row-records.
      if (c && c.__table__ === true && c.columns) {
        if (typeof s === 'string') {
          return applyGet(c.columns[s], rest);
        }
        if (typeof s === 'number' || typeof s === 'boolean') {
          const idx = oneBased ? ((s as number) | 0) - 1 : ((s as number) | 0);
          if (idx < 0 || idx >= c.nrows) {
            throw new Error(`evaluateExpr: ${op} row index ${s} out of `
              + `bounds for table with ${c.nrows} rows`);
          }
          const row: Record<string, any> = {};
          for (const k in c.columns) {
            const col = c.columns[k];
            row[k] = valueLib.isValue(col) ? col.data[idx] : col[idx];
          }
          return applyGet(row, rest);
        }
        if (s === ALL) {
          const rows: any[] = [];
          for (let i = 0; i < c.nrows; i++) {
            const row: Record<string, any> = {};
            for (const k in c.columns) {
              const col = c.columns[k];
              row[k] = valueLib.isValue(col) ? col.data[i] : col[i];
            }
            rows.push(applyGet(row, rest));
          }
          return rows;
        }
      }
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
  // Higher-order value-domain ops (engine-concepts §18.1 Phase 5c):
  // broadcast / reduce / scan / filter migrated to ops-declarations.ts
  // as kind='higher-order'. Their `logical(ir, ctx)` does its own
  // callable resolution and iteration using ctx.evaluateExpr +
  // ctx.resolveFn. aggregate also migrated — it delegates to
  // `_evalAggregate` from sampler-aggregate.ts via its OpDecl.
  if (op === 'broadcast' || op === 'reduce' || op === 'scan' || op === 'filter') {
    return opsModule.dispatchHigherOrder(op, ir, {
      env,
      evaluateExpr,
      resolveFn: _resolveFn,
    });
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
  // table(col1=..., col2=...) per spec §03 §11. At runtime, a table
  // is a `__table__` marker object: { __table__: true, columns: {col→array} }.
  // The marker distinguishes a table value from a plain record (which
  // is just a JS object with field values). Downstream consumers
  // (get, lengthof, reductions, broadcast) branch on the marker.
  if (op === 'table' && Array.isArray(ir.fields)) {
    const columns: Record<string, any> = {};
    let nrows: number | null = null;
    for (const f of ir.fields) {
      const colV = evaluateExpr(f.value, env);
      const arr = valueLib.isValue(colV) ? colV
                : (colV && colV.BYTES_PER_ELEMENT !== undefined) ? colV
                : Array.isArray(colV) ? colV
                : null;
      if (arr === null) {
        throw new Error(`table: column '${f.name}' must be an array (vector), got ${typeof colV}`);
      }
      const len = valueLib.isValue(arr) ? arr.shape[0] : arr.length;
      if (nrows === null) nrows = len;
      else if (nrows !== len) {
        throw new Error(`table: column '${f.name}' has length ${len}, but earlier columns have length ${nrows} (spec §03: all columns must have equal length)`);
      }
      columns[f.name] = arr;
    }
    return { __table__: true, columns, nrows: nrows || 0 };
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
  // evaluated as a value — it's passed verbatim to the in-module measure
  // walker (`walk`, below) which handles iid / joint / record /
  // leaf-distribution recursion and threads the rng state. Refused for
  // measures rand can't sample (weighted, logweighted, bayesupdate,
  // multivariate truncation) by the walker's dispatch — those throw with
  // a clear message.
  if (op === 'rand') {
    return evaluateRand(ir, env);
  }
  // rand_succ(state) — the value-domain successor rngstate for a COMPOSITE
  // `rand` (engine-concepts §11). Synthesised by the lift rand_succ
  // rewrite for the state half `tuple_get(rand(state, iid(<composite>,n)), 1)`;
  // never surface syntax. The composite draw (matRandSample) takes split
  // lane 0 of the parent key; the successor is split lane 1 — so this
  // depends ONLY on the parent state and never walks the (unwalkable)
  // composite measure. Composite succession chains by SPLIT key-derivation,
  // orthogonal to a leaf rand's successor (the threaded counter-advanced
  // state from sampleLeafN). Returns a fresh rngstate, evaluable
  // synchronously so a chained `rand` consumes it via fixedValues /
  // evaluateExpr. The lane assignment is the single-source-of-truth
  // randSplitLanes helper shared with matRandSample.
  //
  // The successor is derived from state.key ONLY — state.counter is read as
  // a type guard but not used (the SPLIT paradigm: a fresh key per lane,
  // not a counter advance). This is sound because composite state halves
  // always originate from rnginit or a prior rand_succ, both counter-zero;
  // two composite chains that share a key would need to share an rnginit
  // seed, which already makes them the same stream by design.
  if (op === 'rand_succ') {
    const args = ir.args || [];
    if (args.length !== 1) {
      throw new Error(`evaluateExpr: rand_succ expects 1 arg (state), got ${args.length}`);
    }
    const state: any = evaluateExpr(args[0], env);
    if (!state || typeof state !== 'object' || !state.key || !state.counter) {
      throw new Error(`evaluateExpr: rand_succ's arg must be an rngstate (got ${typeof state})`);
    }
    const { succKey } = rng.randSplitLanes(state.key);
    return rng.stateFromKey(succKey[0], succKey[1]);
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
  // route through the in-module measure walker, exactly the path `rand`
  // takes — same RNG-state threading, same per-distribution math.
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
    const r: any = walk(state, mIR, env, opts);
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
  // binding (e.g. `m_alias`), the walker needs an IR for that binding.
  // The orchestrator supplies a closure when calling us; the bare-
  // sampler path resolves only literal measure calls inline. If a ref
  // shows up without resolveMeasureRef, the walker throws a clear error.
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
  const r: any = walk(state, args[1], env, opts);
  return [r.value, r.state];
}

// Build a synthetic measure IR `kernel(...kernelInputKwargs)` from a
// kernel name + the kernel_input IR + optional trailing dim IRs. Used
// by `builtin_sample(rngstate, kernel, kernel_input, n, m, ...)` to
// route through the in-module `walk` — same path `rand(state, m)` takes.
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


// =====================================================================
// Measure walker (was traceeval.ts) — value-position sampling primitive
// =====================================================================
//
// The single per-draw sample-side walker for FlatPPL measure expressions.
// Given a measure IR and an env mapping value-position refs to numerics,
// draw a value from that measure, threading rng state through any
// stochastic ancestors that env doesn't already carry. It is the engine's
// recursive composite / lazy-ancestor sampling primitive — the leaf base
// case delegates to the single batched leaf endpoint (`sampleLeafN`), and
// composite measures (joint / record / iid-of-composite / weighted / lawof)
// recurse here. Scoring lives in density.ts (the single density impl); this
// is sample-side only. Folded in from the former traceeval.ts so the
// sampler↔traceeval require cycle disappears and there is one sampling
// model in one module (engine-concepts §11).
//
// Public entry: walk(state, ir, env, opts) → { value, state }
//   opts.resolveMeasureRef - `(name) → measureIR` for measure self-refs.
//   opts.resolveValueRef   - `(name, state) → [value, newState]` lazy
//                            resolution of value-position self-refs (e.g.
//                            `rand(state, M)` where M's params reference
//                            stochastic ancestors); threads state and is
//                            expected to cache through env so two refs to
//                            one name share a draw.

function walk(state: any, ir: IRNode, env: any, opts: any) {
  opts = opts || {};
  const ctx = {
    resolveRef: opts.resolveMeasureRef || null,
    resolveValueRef: opts.resolveValueRef || null,
  };
  return walkInner(state, ir, env, ctx);
}

function walkInner(state: any, ir: IRNode, env: any, ctx: any): any {
  // Self-ref to another measure binding — dereference via the supplied
  // resolver so the binding map isn't baked into the walker.
  if (ir && ir.kind === 'ref' && ir.ns === 'self') {
    if (!ctx.resolveRef) {
      // This is where a BARE un-destructured composite `rand` surfaces: a
      // `r = rand(state, iid(<composite>, n))` reaches the worker's bare
      // evaluate path (no resolver), and walking its inner measure ref hits
      // here. The supported surface is to DESTRUCTURE — the draw half then
      // materialises (matRandSample) and the state half threads via
      // rand_succ; the bare 2-tuple (display-array draw + rngstate
      // successor) has no value representation yet (deferred). Fail loud
      // with that guidance rather than a wrong value.
      throw new Error(
        `sampler.walk: encountered measure ref '${ir.name}' but no ` +
        `resolveMeasureRef was supplied. If this is a bare ` +
        `\`rand(state, <measure>)\`, destructure it — \`draw, succ = rand(...)\` ` +
        `— so the draw materialises and the state half threads via rand_succ ` +
        `(a bare un-destructured composite rand is not yet supported). ` +
        `Otherwise inline the measure or pass opts.resolveMeasureRef.`
      );
    }
    const inner = ctx.resolveRef(ir.name);
    if (!inner) {
      throw new Error(`sampler.walk: resolveMeasureRef returned no IR for '${ir.name}'`);
    }
    return walkInner(state, inner, env, ctx);
  }

  if (!ir || ir.kind !== 'call') {
    throw new Error(
      `sampler.walk: expected a measure call IR (or self-ref), got ` +
      `kind=${ir && ir.kind}`
    );
  }
  const op = ir.op;

  // Leaf distribution — base case.
  if (isKnownDistribution(op)) {
    return walkLeaf(state, ir, env, ctx);
  }

  // Dispatch through MEASURE_OP_WALKERS (composite escape-hatch). Adding a
  // new measure-algebra walker is one entry there plus the handler.
  const handler: any = op != null ? (MEASURE_OP_WALKERS as any)[op] : null;
  if (handler) return handler(state, ir, env, ctx);
  // A forward-composite value-op in measure position (broadcast / aggregate
  // / scan / reduce / filter, typically under a `lawof`) is a COMPOSITE
  // rand whose DRAW half the materialiser samples (matRandSample) and whose
  // STATE half the lift rand_succ rewrite handles — neither walks here. A
  // bare un-destructured composite rand surfaces earlier, at the
  // measure-ref guard above (its inner measure is a hoisted ref). So this
  // throw is the generic "not a sampleable measure" wall.
  throw new Error(
    `sampler.walk: op '${op}' is not a measure expression we can ` +
    `sample. Known: leaf distributions, ` +
    Object.keys(MEASURE_OP_WALKERS).join(', ') + '.'
  );
}

function walkLeaf(state: any, ir: IRNode, env: any, ctx: any) {
  // Pre-fill env with any value-position refs in the kwargs that aren't
  // already known (the leaf's params may reference stochastic ancestors
  // the caller hasn't materialised yet). fillEnvFromRefs threads state
  // through any recursive sampling the resolver does.
  state = fillEnvFromRefs(state, ir, env, ctx);
  // Single leaf endpoint: a scalar draw is sampleLeafN at dims=[] (a batch
  // of 1). Collapses the former scalar realisation (a per-draw stdlib
  // randFn.factory — ziggurat for Normal/LogNormal) onto the one batched
  // endpoint, so `rand(state, <leaf>)` and `rand(state, iid(<leaf>, 1))[0]`
  // are now bit-for-bit identical with no second leaf realisation in the
  // value-position rand / builtin_sample path (engine-concepts §11
  // stage 4). The worker's per-binding materialisation samplers (rand /
  // makeSampler / makeParametricSampler) are a separate path; folding them
  // onto sampleLeafN too is a follow-up (TODO RNG).
  return sampleLeafN(state, ir, env, []);
}

function walkJoint(state: any, ir: IRNode, env: any, ctx: any) {
  // kwarg-joint / record → record keyed by field name; positional joint →
  // array of per-component samples. Sequential per-component recursion
  // threading state.
  if (Array.isArray(ir.fields)) {
    const out: Record<string, any> = {};
    let st = state;
    for (let i = 0; i < ir.fields.length; i++) {
      const f = ir.fields[i];
      const r = walkInner(st, f.value, env, ctx);
      out[f.name] = r.value;
      st = r.state;
    }
    return { value: out, state: st };
  }
  if (Array.isArray(ir.args)) {
    const components = ir.args;
    const out = new Array(components.length);
    let st = state;
    for (let i = 0; i < components.length; i++) {
      const r = walkInner(st, components[i], env, ctx);
      out[i] = r.value;
      st = r.state;
    }
    return { value: out, state: st };
  }
  throw new Error('sampler.walk: joint with neither fields nor args');
}

function walkIid(state: any, ir: IRNode, env: any, ctx: any) {
  // iid(M, size): `size` is a positive integer or a vector of positive
  // integers (multi-axis shape). Draw prod(size) iid samples of M sharing
  // params (that's what makes it iid vs a vectorised per-index call).
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`sampler.walk: iid expected 2 args (measure, size), got ${args.length}`);
  }
  const M = args[0];
  // size may reference fixed-phase value bindings — pre-fill before eval.
  state = fillEnvFromRefs(state, args[1], env, ctx);
  const sizeVal: any = evaluateExpr(args[1], env);
  const dims = _sizeAsDims(sizeVal);

  // Leaf-distribution inner: batch all prod(dims) draws in ONE
  // makeBulkSampler call (the single batched leaf endpoint, sampleLeafN)
  // instead of looping the per-draw walker. iid semantics: the leaf's
  // params are resolved ONCE here and shared across every draw. Both
  // builtin_sample and rand(state, iid(<leaf>, dims)) route through here,
  // so they stay bit-for-bit equal (engine-concepts §11). Composite inner
  // measures keep the per-draw recursion below. The lift pass hoists an
  // inline inner measure to an anon, so M can arrive as `(ref self <anon>)`
  // — resolve a measure-ref to its IR first so both shapes reach the leaf
  // check.
  let leafM = M;
  if (leafM && leafM.kind === 'ref' && leafM.ns === 'self' && ctx.resolveRef) {
    const resolved = ctx.resolveRef(leafM.name);
    if (resolved) leafM = resolved;
  }
  if (leafM && leafM.kind === 'call' && leafM.op && isKnownDistribution(leafM.op)) {
    state = fillEnvFromRefs(state, leafM, env, ctx);   // resolve leaf params once
    return sampleLeafN(state, leafM, env, dims);
  }

  const total = dims.reduce((p: number, n: number) => p * n, 1);
  // Sequential composite draws: total `total` samples threading state.
  const flat = new Array(total);
  let st = state;
  let allNumeric = true;
  for (let j = 0; j < total; j++) {
    const r = walkInner(st, M, env, ctx);
    flat[j] = r.value;
    const t = typeof r.value;
    if (t !== 'number' && t !== 'boolean') allNumeric = false;
    st = r.state;
  }
  if (dims.length <= 1) return { value: flat, state: st };
  // Multi-axis scalar atoms → shape-explicit rank-≥2 Value (spec §03:
  // vectors of vectors are not matrices; downstream value-ops need the tag).
  if (allNumeric) {
    const data = new Float64Array(total);
    for (let j = 0; j < total; j++) data[j] = +flat[j];
    return { value: { shape: dims.slice(), data }, state: st };
  }
  // Heterogeneous / structured atoms: keep the nested-array v0.1 form.
  return { value: _reshapeNested(flat, dims), state: st };
}

// Reshape a flat array of length prod(dims) into a nested JS array of
// shape `dims` (row-major), used by walkIid for heterogeneous atoms.
function _reshapeNested(flat: any[], dims: number[]): any {
  if (dims.length === 0) return flat[0];
  if (dims.length === 1) return flat;
  const n = dims[0];
  const innerDims = dims.slice(1);
  const innerSize = innerDims.reduce((p, n) => p * n, 1);
  const out: any[] = new Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = _reshapeNested(flat.slice(i * innerSize, (i + 1) * innerSize), innerDims);
  }
  return out;
}

// weighted / logweighted: sampling is a pure pass-through (the weight only
// affects density, which lives in density.ts).
function walkWeightedPassThrough(state: any, ir: IRNode, env: any, ctx: any): any {
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`sampler.walk: weighted/logweighted expected 2 args, got ${args.length}`);
  }
  return walkInner(state, args[1], env, ctx);
}

// lawof(M) / draw(M): pass-through wrappers per `lawof(draw(M)) ≡ M`.
function walkUnwrap(state: any, ir: IRNode, env: any, ctx: any): any {
  const args = ir.args || [];
  if (args.length !== 1) {
    throw new Error(`sampler.walk: ${ir.op} expected 1 arg, got ${args.length}`);
  }
  return walkInner(state, args[0], env, ctx);
}

const MEASURE_OP_WALKERS = {
  joint:       walkJoint,
  record:      walkJoint,
  iid:         walkIid,
  weighted:    walkWeightedPassThrough,
  logweighted: walkWeightedPassThrough,
  lawof:       walkUnwrap,
  draw:        walkUnwrap,
};

// Walk a value-position IR collecting every `(ref self <name>)` env doesn't
// know; resolve each via ctx.resolveValueRef (threading state), mutating
// env so a subsequent evaluateExpr sees a populated environment. Value
// position only — measure-position children are handled by the walker's
// own recursion / ctx.resolveRef. Returns the (possibly advanced) state.
function fillEnvFromRefs(state: any, ir: any, env: any, ctx: any) {
  if (!ctx || !ctx.resolveValueRef) return state;
  const refs = new Set<string>();
  collectValueRefs(ir, refs);
  for (const name of refs) {
    if (env && env[name] !== undefined) continue;
    const r = ctx.resolveValueRef(name, state);
    if (!r) continue;
    env[name] = r[0];
    state = r[1];
  }
  return state;
}

function collectValueRefs(ir: any, out: Set<string>) {
  if (ir == null || typeof ir !== 'object') return;
  if (ir.kind === 'ref' && ir.ns === 'self') { out.add(ir.name); return; }
  if (Array.isArray(ir.args)) for (const a of ir.args) collectValueRefs(a, out);
  if (ir.kwargs) for (const k in ir.kwargs) collectValueRefs(ir.kwargs[k], out);
  // Don't descend into `fields` / `body` — measure / scope boundaries
  // handled by the surrounding walker's recursion.
}


module.exports = {
  // Primary API
  rand,
  walk,
  makeSampler,
  makeParametricSampler,
  makeBulkSampler,
  sampleLeafN,
  density,
  makeAnalytical,
  evaluateExpr,
  evaluateExprN,
  isBatch,

  // Used by the in-module measure walker + the fixed-values value-ref
  // resolver — both dispatch on the distribution registry and resolve
  // param expressions against an env.
  lookupDistribution,
  resolveParams,
  makePhiloxPrngAdapter,
  makeBulkUniformPrngAdapter,

  // Introspection
  isKnownDistribution,
  listDistributions,
  hasRandNFn,

  // Internal — exported for tests + the sampler-eval-batched cycle
  // (`resolveConst` is read lazily by _evalN's `const` case) +
  // the broadcast OpDecl which delegates per-element iteration to
  // the engine's `_broadcastApply`.
  _internal: {
    REGISTRY,
    ARITH_OPS,
    ARITH_OPS_N,
    makePhiloxPrngAdapter,
    resolveParams,
    lookupDistribution,
    resolveConst,
    _broadcastApply,
  },
};
