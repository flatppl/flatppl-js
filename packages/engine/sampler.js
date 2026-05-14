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

const rng = require('./rng');

// Math special functions from stdlib for gamma / loggamma / erf-based
// probit / invprobit. The straight-JS variants (logit, invlogit, min,
// max) don't need stdlib backing.
const stdlibGamma    = require('@stdlib/math-base-special-gamma');
const stdlibGammaln  = require('@stdlib/math-base-special-gammaln');
const stdlibErfc     = require('@stdlib/math-base-special-erfc');
const stdlibErfcinv  = require('@stdlib/math-base-special-erfcinv');

// Distribution constructors (analytical PDF/CDF/quantile/mean/etc.)
const Normal      = require('@stdlib/stats-base-dists-normal-ctor');
const Exponential = require('@stdlib/stats-base-dists-exponential-ctor');
const Uniform     = require('@stdlib/stats-base-dists-uniform-ctor');
const LogNormal   = require('@stdlib/stats-base-dists-lognormal-ctor');
const Beta        = require('@stdlib/stats-base-dists-beta-ctor');
const Gamma       = require('@stdlib/stats-base-dists-gamma-ctor');
const Cauchy      = require('@stdlib/stats-base-dists-cauchy-ctor');
const StudentT    = require('@stdlib/stats-base-dists-t-ctor');
const Bernoulli   = require('@stdlib/stats-base-dists-bernoulli-ctor');
const Binomial    = require('@stdlib/stats-base-dists-binomial-ctor');
const Poisson     = require('@stdlib/stats-base-dists-poisson-ctor');

// Random-sample factories. Each exposes `.factory(...params, opts)` that
// returns a closure `() → sample`. opts.prng is a [0,1) closure we plug
// our Philox adapter into.
const randNormal      = require('@stdlib/random-base-normal');
const randExponential = require('@stdlib/random-base-exponential');
const randUniform     = require('@stdlib/random-base-uniform');
const randLogNormal   = require('@stdlib/random-base-lognormal');
const randBeta        = require('@stdlib/random-base-beta');
const randGamma       = require('@stdlib/random-base-gamma');
const randCauchy      = require('@stdlib/random-base-cauchy');
const randT           = require('@stdlib/random-base-t');
const randBernoulli   = require('@stdlib/random-base-bernoulli');
const randBinomial    = require('@stdlib/random-base-binomial');
const randPoisson     = require('@stdlib/random-base-poisson');

// Log-density / log-mass functions. Continuous distributions get a
// direct -logpdf module (Lebesgue reference); discrete distributions
// get either a direct -logpmf (counting reference) or a -pmf that we
// wrap with Math.log. Either way the function signature is uniform —
// `(x, ...params) → log p(x)` — so the trace evaluator (traceeval.js)
// can dispatch on it generically.
const logpdfNormal      = require('@stdlib/stats-base-dists-normal-logpdf');
const logpdfExponential = require('@stdlib/stats-base-dists-exponential-logpdf');
const logpdfUniform     = require('@stdlib/stats-base-dists-uniform-logpdf');
const logpdfLogNormal   = require('@stdlib/stats-base-dists-lognormal-logpdf');
const logpdfBeta        = require('@stdlib/stats-base-dists-beta-logpdf');
const logpdfGamma       = require('@stdlib/stats-base-dists-gamma-logpdf');
const logpdfCauchy      = require('@stdlib/stats-base-dists-cauchy-logpdf');
const logpdfT           = require('@stdlib/stats-base-dists-t-logpdf');
const pmfBernoulli      = require('@stdlib/stats-base-dists-bernoulli-pmf');
const logpmfBinomial    = require('@stdlib/stats-base-dists-binomial-logpmf');
const logpmfPoisson     = require('@stdlib/stats-base-dists-poisson-logpmf');

// Bernoulli ships pmf only — wrap with Math.log. For two atoms this is
// numerically fine; if stdlib adds -logpmf-bernoulli in the future we
// can switch over without touching callers.
function logpmfBernoulli(x, p) {
  return Math.log(pmfBernoulli(x, p));
}

// Dirac point-mass measure. Spec §sec:measure-algebra: `Dirac(value=v)`
// concentrates probability 1 at v. There's no stdlib distribution to
// wrap, so we synthesise the same shape (Ctor + factory + logpdf) the
// rest of the registry uses, so makeSampler / makeParametricSampler /
// makeAnalytical / density all dispatch generically without
// special-casing Dirac at the call sites.
//
// Identity rewrite for `draw(Dirac(value=e))` happens earlier in
// orchestrator.classifyForChain — that path never reaches the sampler.
// What does reach here: `m = Dirac(value = v)` measure-alias bindings
// and any nested Dirac inside a measure-algebra expression we haven't
// rewritten yet (e.g. weighted, joint).
//
// Restriction: scalar-value Diracs only. Sampling produces a
// Float64Array, so the value coerces to a number. Record / array /
// tuple-valued Diracs would need a structured EmpiricalMeasure path
// analogous to joint / iid; deferred.
const randDirac = {
  // stdlib's dual-mode factory:
  //   factory(p, opts)  → returns no-arg closure (params baked in)
  //   factory(opts)     → returns closure that takes (p) per call
  // Detect parametric form by the single object-with-prng arg shape;
  // anything else is the static form.
  factory: function() {
    const args = Array.prototype.slice.call(arguments);
    if (args.length === 1 && args[0] && typeof args[0] === 'object'
        && ('prng' in args[0] || 'seed' in args[0])) {
      // Parametric: per-call (value).
      return function parametricDiracSampler(value) { return +value; };
    }
    // Static: factory(value, opts) — closure returns value.
    const value = +args[0];
    return function staticDiracSampler() { return value; };
  },
};

// Synthetic stdlib-shaped Ctor. Only the methods the engine's
// makeAnalytical / density / quantile paths actually call are
// implemented. A Dirac has no Lebesgue density (singular w.r.t.
// Lebesgue); pdf returns 1/0 indicator so density-overlay calls
// don't crash, but viewer-side fixed-Dirac rendering bypasses this
// and renders the surface form as text.
function DiracCtor(value) {
  this.value = +value;
  this.mean = +value;
  this.variance = 0;
  this.stdev = 0;
  this.support = [+value, +value];
}
DiracCtor.prototype.pdf      = function(x) { return x === this.value ? 1 : 0; };
DiracCtor.prototype.logpdf   = function(x) { return x === this.value ? 0 : -Infinity; };
DiracCtor.prototype.cdf      = function(x) { return x < this.value ? 0 : 1; };
DiracCtor.prototype.quantile = function(_p) { return this.value; };

function logpdfDirac(x, value) {
  return x === value ? 0 : -Infinity;
}

// Per-distribution metadata, including the param translation between
// FlatPPL spec names (used in surface code and in IR kwargs) and stdlib's
// constructor argument order.
//
// `params`:    Ordered list of FlatPPL spec parameter names. The order
//              determines the positional argument order to stdlib's ctor
//              and rand factory. (stdlib's APIs are positional; we
//              translate from FlatPPL kwargs at the call boundary.)
// `aliases`:   FlatPPL-name → other-name map for params that have
//              alternate spec names. We accept both.
// `discrete`:  Whether the distribution's reference measure is counting
//              (true) or Lebesgue (false). Used by density(...) to decide
//              the plot shape.
// `Ctor`:      stdlib constructor (analytical methods).
// `randFn`:    stdlib random sampler module (has `.factory(...)`).
// `logpdfFn`:  log-density / log-mass at a point — `(x, ...params) → number`.
//              Continuous: log p.d.f. w.r.t. Lebesgue. Discrete: log p.m.f.
//              w.r.t. counting. Same positional param order as randFn.
//              Used by traceeval.js to score clamped values at leaf sites.
const REGISTRY = {
  Normal: {
    params:   ['mu', 'sigma'],
    aliases:  {},
    discrete: false,
    Ctor:     Normal,
    randFn:   randNormal,
    logpdfFn: logpdfNormal,
  },
  Exponential: {
    // Spec: Exponential(rate). stdlib's positional ctor takes lambda
    // (which is the same rate parameter, just renamed).
    params:   ['rate'],
    aliases:  {},
    discrete: false,
    Ctor:     Exponential,
    randFn:   randExponential,
    logpdfFn: logpdfExponential,
  },
  Uniform: {
    // Spec form: Uniform(support = S) where S is a set (typically an
    // interval). stdlib's positional ctor takes (a, b). We extract the
    // numeric bounds from the support arg via regionBoundsFromIR,
    // which understands literal interval(lo, hi) and named real sets.
    // Unbounded supports (e.g. `reals`) make the Uniform improper —
    // sampling would never terminate — so we reject them at param-
    // resolution time with a clear error rather than passing through
    // an infinity to stdlib.
    params:   ['support'],   // surface name only; bounds get derived
    aliases:  {},
    discrete: false,
    Ctor:     Uniform,
    randFn:   randUniform,
    logpdfFn: logpdfUniform,
    customResolveParams: function (measureIR, env) {
      const kwargs = measureIR.kwargs || {};
      const positional = measureIR.args || [];
      const supportIR = ('support' in kwargs) ? kwargs.support
                      : (positional.length > 0 ? positional[0] : null);
      if (!supportIR) {
        throw new Error('sampler: Uniform missing support argument');
      }
      const [lo, hi] = regionBoundsFromIR(supportIR, env);
      if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
        throw new Error('sampler: Uniform requires a bounded support, got ['
          + lo + ', ' + hi + ']');
      }
      return [lo, hi];
    },
  },
  LogNormal: {
    params:   ['mu', 'sigma'],
    aliases:  {},
    discrete: false,
    Ctor:     LogNormal,
    randFn:   randLogNormal,
    logpdfFn: logpdfLogNormal,
  },
  Beta: {
    params:   ['alpha', 'beta'],
    aliases:  {},
    discrete: false,
    Ctor:     Beta,
    randFn:   randBeta,
    logpdfFn: logpdfBeta,
  },
  Gamma: {
    // Spec: Gamma(shape, rate). stdlib calls them alpha, beta but they
    // mean shape and rate respectively (per stdlib's docs). We keep the
    // spec names.
    params:   ['shape', 'rate'],
    aliases:  {},
    discrete: false,
    Ctor:     Gamma,
    randFn:   randGamma,
    logpdfFn: logpdfGamma,
  },
  Cauchy: {
    // Spec: Cauchy(location, scale). stdlib uses x0, gamma — same things.
    params:   ['location', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     Cauchy,
    randFn:   randCauchy,
    logpdfFn: logpdfCauchy,
  },
  StudentT: {
    // Spec: StudentT(nu). stdlib calls it T with parameter v.
    params:   ['nu'],
    aliases:  {},
    discrete: false,
    Ctor:     StudentT,
    randFn:   randT,
    logpdfFn: logpdfT,
  },
  Bernoulli: {
    params:   ['p'],
    aliases:  {},
    discrete: true,
    Ctor:     Bernoulli,
    randFn:   randBernoulli,
    logpdfFn: logpmfBernoulli,
  },
  Binomial: {
    params:   ['n', 'p'],
    aliases:  {},
    discrete: true,
    Ctor:     Binomial,
    randFn:   randBinomial,
    logpdfFn: logpmfBinomial,
  },
  Poisson: {
    // Spec: Poisson(rate). stdlib calls it lambda.
    params:   ['rate'],
    aliases:  {},
    discrete: true,
    Ctor:     Poisson,
    randFn:   randPoisson,
    logpdfFn: logpmfPoisson,
  },
  Dirac: {
    // Degenerate point-mass measure. The 'value' kwarg may be of any
    // scalar type (real / integer / bool); 'discrete' is left false
    // because the engine has no consistent way to tell from the IR
    // alone. Affects only density-curve overlay, which the viewer
    // skips for fixed Diracs anyway.
    params:   ['value'],
    aliases:  {},
    discrete: false,
    Ctor:     DiracCtor,
    randFn:   randDirac,
    logpdfFn: logpdfDirac,
  },
};

// =====================================================================
// Public API
// =====================================================================

/**
 * Whether this distribution name is recognized as a built-in samplable
 * measure. Used by orchestrators to decide whether they can produce
 * samples directly via `rand` or need a higher-level path (e.g.,
 * importance sampling for posterior measures).
 */
function isKnownDistribution(name) {
  return Object.prototype.hasOwnProperty.call(REGISTRY, name);
}

function listDistributions() {
  return Object.keys(REGISTRY);
}

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
function rand(state, measureIR, env) {
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
function makeSampler(state, measureIR, env) {
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
function makeParametricSampler(state, measureIR) {
  const entry = lookupDistribution(measureIR);
  const prng = makePhiloxPrngAdapter(state);
  const sampler = entry.randFn.factory({ prng });
  return {
    drawWith(env) {
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
function makeAnalytical(measureIR, env) {
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
function density(measureIR, env, opts) {
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
function evaluateExpr(ir, env) {
  switch (ir.kind) {
    case 'lit':
      return ir.value;
    case 'const':
      return resolveConst(ir.name);
    case 'ref':
      return resolveRef(ir, env);
    case 'call':
      return evaluateCall(ir, env);
    default:
      throw new Error(`evaluateExpr: unsupported IR node kind '${ir.kind}'`);
  }
}

function resolveConst(name) {
  switch (name) {
    case 'pi':  return Math.PI;
    case 'e':   return Math.E;
    case 'inf': return Infinity;
    case '-inf': return -Infinity;
    default:
      throw new Error(`evaluateExpr: unknown constant '${name}'`);
  }
}

function resolveRef(ir, env) {
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

const ARITH_OPS = {
  add: (a, b) => a + b,
  sub: (a, b) => a - b,
  mul: (a, b) => a * b,
  div: (a, b) => a / b,
  mod: (a, b) => a % b,
  neg: a => -a,
  pos: a => +a,
  // Common unary maths — extend EVALUABLE_OPS in orchestrator.js
  // alongside any addition here so the static gate matches.
  abs:   a => Math.abs(a),
  abs2:  a => a * a,
  exp:   a => Math.exp(a),
  log:   a => Math.log(a),
  log10: a => Math.log10(a),
  sqrt:  a => Math.sqrt(a),
  sin:   a => Math.sin(a),
  cos:   a => Math.cos(a),
  floor: a => Math.floor(a),
  ceil:  a => Math.ceil(a),
  round: a => Math.round(a),
  pow:   (a, b) => Math.pow(a, b),
  // Binary scalar reductions (spec §07). Distinct from the variadic
  // `maximum` / `minimum` reductions, which take an array argument.
  min:   (a, b) => Math.min(a, b),
  max:   (a, b) => Math.max(a, b),
  // Special functions: gamma function family and link functions (spec
  // §07 "Elementary functions"). Domain checks are minimal here —
  // stdlib returns NaN / Infinity for out-of-domain inputs which the
  // caller can detect via isfinite / isnan.
  gamma:     a => stdlibGamma(a),
  loggamma:  a => stdlibGammaln(a),
  // Link functions:
  //   logit(p)    = log(p / (1−p))                            on (0,1)
  //   invlogit(x) = 1 / (1 + exp(−x))                         on ℝ
  //   probit(p)   = Φ⁻¹(p)  via the erfcinv identity          on (0,1)
  //   invprobit(x)= Φ(x)    via the erfc identity             on ℝ
  // probit and invprobit reach ±∞ at the endpoints (spec §07).
  logit:     p => Math.log(p / (1 - p)),
  invlogit:  x => 1 / (1 + Math.exp(-x)),
  probit:    p => -Math.SQRT2 * stdlibErfcinv(2 * p),
  invprobit: x => 0.5 * stdlibErfc(-x / Math.SQRT2),
  // Comparison ops produce booleans (per spec §07). Spec preserves
  // strict typing — no implicit numeric→boolean cast — so the
  // operands are treated as reals and the result is JS boolean.
  lt:      (a, b) => a < b,
  le:      (a, b) => a <= b,
  gt:      (a, b) => a > b,
  ge:      (a, b) => a >= b,
  equal:   (a, b) => a === b,
  unequal: (a, b) => a !== b,
  // Predicates over reals.
  isfinite: a => Number.isFinite(a),
  isinf:    a => !Number.isNaN(a) && !Number.isFinite(a),
  isnan:    a => Number.isNaN(a),
  iszero:   a => a === 0,
  // Logic / conditionals (spec §07). FlatPPL booleans are strict — we
  // don't coerce truthy values, the typeinfer pass already requires
  // boolean operands. lxor is exclusive-or; ifelse is the conditional
  // expression returning the first or second branch by cond.
  land:    (a, b) => a && b,
  lor:     (a, b) => a || b,
  lxor:    (a, b) => a !== b,
  lnot:    a => !a,
  ifelse:  (c, a, b) => c ? a : b,
  // Vector constructor — turns positional args into a JS array.
  // Lower than the typed Float64Array path used for materialised
  // measures: this is for inline `[a, b, c]` literals that surface
  // as (call vector (lit …) (lit …) …) IR. Reductions below take
  // these arrays directly.
  vector: (...xs) => xs,
  // Reductions over an array. Operate on JS arrays / TypedArrays
  // alike (both expose .length and indexed access). Spec semantics:
  //   sum     = Σ x[i]
  //   mean    = sum / N
  //   prod    = Π x[i]
  //   length  = N
  //   maximum = max x[i]   (Math.max would .apply-blow-stack at length 1e6)
  //   minimum = min x[i]
  //   var     = mean( (x - mean)² )  — population variance, divisor N
  sum:     reduce((acc, v) => acc + v, 0),
  mean:    arr => reduce((acc, v) => acc + v, 0)(arr) / arr.length,
  prod:    reduce((acc, v) => acc * v, 1),
  length:  arr => arr.length,
  maximum: arr => {
    let m = -Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] > m) m = arr[i];
    return m;
  },
  minimum: arr => {
    let m = Infinity;
    for (let i = 0; i < arr.length; i++) if (arr[i] < m) m = arr[i];
    return m;
  },
  var: arr => {
    const n = arr.length;
    if (n === 0) return 0;
    let s = 0;
    for (let i = 0; i < n; i++) s += arr[i];
    const mu = s / n;
    let v = 0;
    for (let i = 0; i < n; i++) { const d = arr[i] - mu; v += d * d; }
    return v / n;
  },
};

// Helper: build a one-pass reducer that loops over array indices.
// Avoids repeated Array.prototype.reduce overhead for tight inner
// loops on large literal arrays.
function reduce(step, init) {
  return arr => {
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
function regionBoundsFromIR(ir, env) {
  if (!ir) throw new Error('regionBoundsFromIR: missing IR');
  if (ir.kind === 'call' && ir.op === 'interval'
      && Array.isArray(ir.args) && ir.args.length === 2) {
    return [evaluateExpr(ir.args[0], env), evaluateExpr(ir.args[1], env)];
  }
  if (ir.kind === 'const') {
    switch (ir.name) {
      case 'reals':       return [-Infinity, Infinity];
      case 'posreals':    return [0, Infinity];
      case 'nonnegreals': return [0, Infinity];
      case 'unitinterval':return [0, 1];
    }
  }
  throw new Error('regionBoundsFromIR: unsupported region shape (kind='
    + ir.kind + (ir.op ? ', op=' + ir.op : '') + ')');
}

function evaluateCall(ir, env) {
  const op = ir.op;
  if (op in ARITH_OPS) {
    const args = (ir.args || []).map(a => evaluateExpr(a, env));
    return ARITH_OPS[op](...args);
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
    const t = evaluateExpr(args[0], env);
    const i = evaluateExpr(args[1], env) | 0;
    if (!Array.isArray(t)) {
      throw new Error(`evaluateExpr: tuple_get target is not an array (got ${typeof t})`);
    }
    return t[i];
  }
  // tuple(...) — JS array of evaluated args. Used for surface
  // `(a, b, ...)` literals and as an intermediate value when downstream
  // code projects via tuple_get.
  if (op === 'tuple') {
    return (ir.args || []).map(a => evaluateExpr(a, env));
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
    const obj = evaluateExpr(args[0], env);
    const key = args[1] && args[1].kind === 'lit' ? args[1].value : evaluateExpr(args[1], env);
    if (obj == null || typeof obj !== 'object') {
      throw new Error(`evaluateExpr: get_field target is not a record (got ${typeof obj})`);
    }
    return obj[key];
  }
  // Shape functions (spec §07 Approximation functions). All three take
  // kwargs so they don't pass through ARITH_OPS; dispatch explicitly.
  // The kwargs are `coefficients` / `values` (a fixed-phase array) and
  // `x` (the evaluation point), plus `edges` for stepwise.
  if (op === 'polynomial') {
    const kw = ir.kwargs || {};
    const coeffs = kw.coefficients != null ? evaluateExpr(kw.coefficients, env)
                                           : evaluateExpr(ir.args[0], env);
    const x = kw.x != null ? evaluateExpr(kw.x, env)
                           : evaluateExpr(ir.args[1], env);
    // Σ a_i · x^i, evaluated Horner-style for numerical stability.
    let acc = 0;
    for (let i = coeffs.length - 1; i >= 0; i--) acc = acc * x + coeffs[i];
    return acc;
  }
  if (op === 'bernstein') {
    // Bernstein basis on [0, 1]: f(x) = Σ_{k=0..n} a_k · C(n, k) · x^k · (1-x)^{n-k}
    // where n = length(coefficients) - 1. Numerically stable for x ∈ [0,1].
    const kw = ir.kwargs || {};
    const coeffs = kw.coefficients != null ? evaluateExpr(kw.coefficients, env)
                                           : evaluateExpr(ir.args[0], env);
    const x = kw.x != null ? evaluateExpr(kw.x, env)
                           : evaluateExpr(ir.args[1], env);
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
    const edges    = kw.edges    != null ? evaluateExpr(kw.edges,    env)
                                         : evaluateExpr(ir.args[0],  env);
    const regionIR = kw.region   != null ? kw.region                 : ir.args[1];
    const counts   = kw.counts   != null ? evaluateExpr(kw.counts,   env)
                                         : evaluateExpr(ir.args[2],  env);
    const [lo, hi] = regionBoundsFromIR(regionIR, env);
    const n = counts.length;
    if (edges.length !== n + 1) {
      throw new Error('selectbins: edges length must equal counts length + 1');
    }
    const out = [];
    for (let i = 0; i < n; i++) {
      // Bin [edges[i], edges[i+1]] intersects [lo, hi] iff
      //   edges[i] ≤ hi  AND  edges[i+1] ≥ lo
      if (edges[i] <= hi && edges[i + 1] >= lo) {
        out.push(counts[i]);
      }
    }
    return out;
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
    const bins = kw.bins != null ? evaluateExpr(kw.bins, env)
                                 : evaluateExpr(ir.args[0], env);
    const data = kw.data != null ? evaluateExpr(kw.data, env)
                                 : evaluateExpr(ir.args[1], env);
    if (!bins || typeof bins.length !== 'number'
        || (bins.length > 0 && typeof bins[0] !== 'number')) {
      throw new Error('bincounts: multi-dimensional binning not yet supported');
    }
    const n = bins.length - 1;
    if (n < 0) throw new Error('bincounts: bins must have at least 1 edge');
    const counts = new Array(n).fill(0);
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
    return counts;
  }
  if (op === 'stepwise') {
    // Piecewise constant: edges has length n+1, values has length n.
    // For x in [edges[i], edges[i+1]) return values[i]; right edge
    // is closed for the last bin.
    const kw = ir.kwargs || {};
    const edges  = kw.edges  != null ? evaluateExpr(kw.edges,  env)
                                     : evaluateExpr(ir.args[0], env);
    const values = kw.values != null ? evaluateExpr(kw.values, env)
                                     : evaluateExpr(ir.args[1], env);
    const x      = kw.x      != null ? evaluateExpr(kw.x,      env)
                                     : evaluateExpr(ir.args[2], env);
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
    const out = {};
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
    const seed = evaluateExpr(args[0], env);
    if (!isByteVector(seed)) {
      throw new Error(`evaluateExpr: rnginit seed must be a byte vector (array of integers in 0..255)`);
    }
    return rng.seedFromBytes(seed);
  }
  // rngstate(<bytes>) — round-trip an externally-serialized state.
  // Round-trip semantics: rngstate(bytesFromState(s)) ≡ s.
  if (op === 'rngstate') {
    const args = ir.args || [];
    if (args.length !== 1) {
      throw new Error(`evaluateExpr: rngstate expects 1 arg, got ${args.length}`);
    }
    const bytes = evaluateExpr(args[0], env);
    if (!isByteVector(bytes)) {
      throw new Error(`evaluateExpr: rngstate bytes must be a byte vector (array of integers in 0..255)`);
    }
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
// integers are also accepted.
function isByteVector(x) {
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
let _traceeval = null;
function getTraceeval() {
  if (!_traceeval) _traceeval = require('./traceeval');
  return _traceeval;
}

function evaluateRand(ir, env) {
  const args = ir.args || [];
  if (args.length !== 2) {
    throw new Error(`evaluateExpr: rand expects 2 args (state, measure), got ${args.length}`);
  }
  const state = evaluateExpr(args[0], env);
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
  const opts = { tally: 'none' };
  if (env && typeof env.__resolveMeasureRef === 'function') {
    opts.resolveMeasureRef = env.__resolveMeasureRef;
  }
  if (env && typeof env.__resolveValueRef === 'function') {
    opts.resolveValueRef = env.__resolveValueRef;
  }
  const r = getTraceeval().walk(state, args[1], env, undefined, opts);
  return [r.value, r.state];
}

function lookupDistribution(measureIR) {
  if (!measureIR || measureIR.kind !== 'call') {
    throw new Error(
      `sampler: expected a measure call IR, got kind=${measureIR?.kind}`
    );
  }
  const name = measureIR.op;
  const entry = REGISTRY[name];
  if (!entry) {
    throw new Error(
      `sampler: '${name}' is not a known distribution (or not yet implemented). ` +
      `Known: ${listDistributions().join(', ')}`
    );
  }
  return entry;
}

function resolveParams(measureIR, entry, env) {
  // Hook for distributions whose surface params aren't directly
  // evaluable as numerics (e.g. Uniform(support = interval(lo, hi)),
  // where the `support` arg is a set IR rather than a value). The
  // entry-supplied resolver returns the positional numeric list that
  // the stdlib factory / ctor / logpdf call expects.
  if (typeof entry.customResolveParams === 'function') {
    return entry.customResolveParams(measureIR, env);
  }
  const kwargs = measureIR.kwargs || {};
  const positional = measureIR.args || [];
  const out = [];
  for (let i = 0; i < entry.params.length; i++) {
    const paramName = entry.params[i];
    let exprIR;
    // Try kwargs first (idiomatic FlatPPL style).
    if (paramName in kwargs) {
      exprIR = kwargs[paramName];
    } else if (entry.aliases[paramName] && entry.aliases[paramName] in kwargs) {
      exprIR = kwargs[entry.aliases[paramName]];
    } else if (i < positional.length) {
      // Positional fallback (some FlatPPL forms allow this).
      exprIR = positional[i];
    } else {
      throw new Error(
        `sampler: '${measureIR.op}' missing parameter '${paramName}'`
      );
    }
    out.push(evaluateExpr(exprIR, env));
  }
  return out;
}

// =====================================================================
// Philox prng adapter
// =====================================================================
//
// stdlib's random factories take an `opts.prng` callback that returns a
// pseudorandom float in [0, 1). Our Philox is pure-functional —
// `nextUniform(state) → [value, new_state]`. We bridge by capturing the
// Philox state in a closure that mutates it on each call. The closure
// also exposes `getState()` so the caller can read out the trailing
// state when sampling completes — preserving end-to-end functional
// state-threading at the public API.

function makePhiloxPrngAdapter(initialState) {
  let state = initialState;
  function prng() {
    const [u, next] = rng.nextUniform(state);
    state = next;
    return u;
  }
  prng.getState = () => state;
  // Also expose a reset hook for tests.
  prng._setState = (s) => { state = s; };
  return prng;
}

// =====================================================================
// Misc helpers
// =====================================================================

function readSupport(dist) {
  // stdlib distributions expose `support` as either a getter or an array.
  // Some return objects like { lower, upper } — we normalize to [lo, hi].
  const s = dist.support;
  if (Array.isArray(s)) return [s[0], s[1]];
  if (s && typeof s === 'object') {
    if ('lower' in s && 'upper' in s) return [s.lower, s.upper];
  }
  // Fallback: use the full real line. (This is a degenerate case; if it
  // happens we'll have already used the quantile-based bounds.)
  return [-Infinity, Infinity];
}

module.exports = {
  // Primary API
  rand,
  makeSampler,
  makeParametricSampler,
  density,
  makeAnalytical,
  evaluateExpr,

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
    makePhiloxPrngAdapter,
    resolveParams,
    lookupDistribution,
  },
};
