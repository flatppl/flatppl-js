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

// ---------------------------------------------------------------------
// Synthetic Logistic and Weibull distributions
// ---------------------------------------------------------------------
//
// Neither has a corresponding @stdlib/random-base-* or stats-base-dists-
// *-ctor package installed. Both have simple closed-form inverse-CDFs,
// so we hand-roll factory + Ctor + logpdfFn in the same shape the rest
// of the REGISTRY uses. Inverse-CDF sampling: u ~ U(0,1), x = Q(u).
// Edge clipping at u → 0 / u → 1 avoids ±∞ output from log / log1p
// when the prng emits the very boundary; an ~eps-thin band is
// statistically harmless at any practical sample count.

function uClip(u) {
  if (u <= 0)            return Number.EPSILON;
  if (u >= 1 - 1e-16)    return 1 - Number.EPSILON;
  return u;
}

// Logistic(mu, s):  pdf = exp(-z) / (s · (1 + exp(-z))^2)  where z = (x − μ)/s
//                   cdf = 1 / (1 + exp(-z))
//                   Q(p) = μ + s · log(p / (1 − p))
const randLogistic = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      // Parametric: returned closure takes (mu, s) per call.
      return function parametricLogisticSampler(mu, s) {
        const u = uClip(prng());
        return +mu + (+s) * Math.log(u / (1 - u));
      };
    }
    const mu = +args[0], s = +args[1];
    return function staticLogisticSampler() {
      const u = uClip(prng());
      return mu + s * Math.log(u / (1 - u));
    };
  },
};

function LogisticCtor(mu, s) {
  this.mu = +mu; this.s = +s;
  this.mean = +mu;
  this.variance = (s * s) * (Math.PI * Math.PI) / 3;
  this.stdev = Math.sqrt(this.variance);
  this.support = [-Infinity, Infinity];
}
LogisticCtor.prototype.pdf = function (x) {
  const z = (x - this.mu) / this.s;
  const ez = Math.exp(-z);
  const denom = 1 + ez;
  return ez / (this.s * denom * denom);
};
LogisticCtor.prototype.logpdf = function (x) {
  const z = (x - this.mu) / this.s;
  return -z - Math.log(this.s) - 2 * Math.log(1 + Math.exp(-z));
};
LogisticCtor.prototype.cdf = function (x) {
  return 1 / (1 + Math.exp(-(x - this.mu) / this.s));
};
LogisticCtor.prototype.quantile = function (p) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  return this.mu + this.s * Math.log(p / (1 - p));
};
function logpdfLogistic(x, mu, s) {
  const z = (x - mu) / s;
  return -z - Math.log(s) - 2 * Math.log(1 + Math.exp(-z));
}

// Weibull(shape, scale):  pdf = (k/λ)·(x/λ)^{k−1}·exp(−(x/λ)^k) for x ≥ 0
//                         cdf = 1 − exp(−(x/λ)^k)
//                         Q(p) = λ · (−log(1−p))^{1/k}
const randWeibull = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      return function parametricWeibullSampler(k, lambda) {
        const u = uClip(prng());
        return (+lambda) * Math.pow(-Math.log(1 - u), 1 / (+k));
      };
    }
    const k = +args[0], lambda = +args[1];
    return function staticWeibullSampler() {
      const u = uClip(prng());
      return lambda * Math.pow(-Math.log(1 - u), 1 / k);
    };
  },
};

function WeibullCtor(shape, scale) {
  // FlatPPL spec names: shape (k), scale (λ).
  this.k = +shape; this.lambda = +scale;
  // Mean = λ · Γ(1 + 1/k); variance = λ² · (Γ(1 + 2/k) − Γ(1 + 1/k)²).
  // We don't have stdlib gamma in scope here without adding a require —
  // the analytical handle leaves these undefined; plot ranges fall
  // back to quantile-based bounds.
  this.support = [0, Infinity];
}
WeibullCtor.prototype.pdf = function (x) {
  if (x < 0) return 0;
  if (x === 0) return this.k === 1 ? 1 / this.lambda : (this.k > 1 ? 0 : Infinity);
  const z = x / this.lambda;
  return (this.k / this.lambda) * Math.pow(z, this.k - 1) * Math.exp(-Math.pow(z, this.k));
};
WeibullCtor.prototype.logpdf = function (x) {
  if (x < 0) return -Infinity;
  if (x === 0) {
    if (this.k === 1) return -Math.log(this.lambda);
    return this.k > 1 ? -Infinity : Infinity;
  }
  const z = x / this.lambda;
  return Math.log(this.k / this.lambda)
       + (this.k - 1) * Math.log(z)
       - Math.pow(z, this.k);
};
WeibullCtor.prototype.cdf = function (x) {
  if (x <= 0) return 0;
  return 1 - Math.exp(-Math.pow(x / this.lambda, this.k));
};
WeibullCtor.prototype.quantile = function (p) {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  return this.lambda * Math.pow(-Math.log(1 - p), 1 / this.k);
};
function logpdfWeibull(x, k, lambda) {
  if (x < 0) return -Infinity;
  if (x === 0) {
    if (k === 1) return -Math.log(lambda);
    return k > 1 ? -Infinity : Infinity;
  }
  const z = x / lambda;
  return Math.log(k / lambda) + (k - 1) * Math.log(z) - Math.pow(z, k);
}

// ---------------------------------------------------------------------
// Synthetic GeneralizedNormal (symmetric)
// ---------------------------------------------------------------------
//
// Spec §08: GeneralizedNormal(mean, alpha, beta).
//   pdf  =  β / (2α · Γ(1/β)) · exp( −(|x − μ| / α)^β )
//   β = 2  → reduces to Normal with σ = α/√2
//   β = 1  → reduces to Laplace
//
// Sampling: the canonical "scaled-gamma + Rademacher" construction.
//   Y ~ Gamma(shape = 1/β, rate = 1)
//   R = ±1  uniformly
//   X = μ + R · α · Y^(1/β)
// Derivation: Y^(1/β) follows the absolute-value of GeneralizedNormal
// up to scale; the Rademacher symmetrises around the mean. We use
// stdlib's randGamma factory bound to the same prng for both draws,
// and a separate prng() call for the sign bit.

const randGeneralizedNormal = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      const inner = randGamma.factory({ prng });
      return function parametricGenNormalSampler(mean, alpha, beta) {
        const y = inner(1 / beta, 1);        // Gamma(1/β, rate=1)
        const r = prng() < 0.5 ? -1 : 1;
        return mean + r * alpha * Math.pow(y, 1 / beta);
      };
    }
    const mean  = +args[0];
    const alpha = +args[1];
    const beta  = +args[2];
    const inner = randGamma.factory(1 / beta, 1, { prng });
    return function staticGenNormalSampler() {
      const y = inner();
      const r = prng() < 0.5 ? -1 : 1;
      return mean + r * alpha * Math.pow(y, 1 / beta);
    };
  },
};

function GeneralizedNormalCtor(mean, alpha, beta) {
  this.mean  = +mean;
  this.alpha = +alpha;
  this.beta  = +beta;
  this.support = [-Infinity, Infinity];
}
GeneralizedNormalCtor.prototype.pdf = function (x) {
  return Math.exp(this.logpdf(x));
};
GeneralizedNormalCtor.prototype.logpdf = function (x) {
  const a = this.alpha, b = this.beta;
  const z = Math.abs(x - this.mean) / a;
  return Math.log(b) - Math.log(2 * a) - stdlibGammaln(1 / b) - Math.pow(z, b);
};
function logpdfGeneralizedNormal(x, mean, alpha, beta) {
  const z = Math.abs(x - mean) / alpha;
  return Math.log(beta) - Math.log(2 * alpha)
       - stdlibGammaln(1 / beta) - Math.pow(z, beta);
}

// ---------------------------------------------------------------------
// Synthetic InverseGamma
// ---------------------------------------------------------------------
//
// Spec §08: InverseGamma(shape, scale). Mathematical equivalence with
// Gamma (spec line 200): if Y ~ Gamma(shape, rate = scale) then
// 1/Y ~ InverseGamma(shape, scale). We delegate sampling to stdlib's
// Gamma rand factory with rate = scale (= the InverseGamma scale
// parameter), then invert. logpdf has a closed form — no integration.

const randInverseGamma = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      // Parametric: returned closure takes (shape, scale) per call.
      // Build one parametric Gamma factory bound to prng; invoke
      // it per call with the right (shape, rate) pair.
      const inner = randGamma.factory({ prng });
      return function parametricInverseGammaSampler(shape, scale) {
        const y = inner(+shape, +scale);  // Gamma(shape, rate=scale)
        return 1 / y;
      };
    }
    const shape = +args[0], scale = +args[1];
    // Static: bake (shape, rate=scale) into a single Gamma sampler.
    const inner = randGamma.factory(shape, scale, { prng });
    return function staticInverseGammaSampler() { return 1 / inner(); };
  },
};

function InverseGammaCtor(shape, scale) {
  this.shape = +shape;
  this.scale = +scale;
  this.support = [0, Infinity];
}
InverseGammaCtor.prototype.pdf = function (x) {
  if (x <= 0) return 0;
  // β^α / Γ(α) · x^(-α-1) · exp(-β/x)
  return Math.exp(this.logpdf(x));
};
InverseGammaCtor.prototype.logpdf = function (x) {
  if (x <= 0) return -Infinity;
  const a = this.shape, b = this.scale;
  return a * Math.log(b) - stdlibGammaln(a) - (a + 1) * Math.log(x) - b / x;
};
function logpdfInverseGamma(x, shape, scale) {
  if (x <= 0) return -Infinity;
  return shape * Math.log(scale)
       - stdlibGammaln(shape)
       - (shape + 1) * Math.log(x)
       - scale / x;
}

// ---------------------------------------------------------------------
// Synthetic Categorical / Categorical0
// ---------------------------------------------------------------------
//
// Discrete distributions over {1, …, n} (Categorical, 1-based per
// FlatPPL convention) and {0, …, n−1} (Categorical0). Sampling: draw
// u ~ U(0,1), walk the cumulative sum of p and pick the first index
// where the running sum ≥ u. pmf is just p_k at the support point;
// outside the support the log-pmf is −∞.
//
// The p argument is a probability vector — already on the standard
// simplex by spec. We don't renormalize here; callers should use
// l1unit / softmax upstream when starting from logits or unnormalized
// weights.

function _catSample(p, prng, offset) {
  let u = prng();
  if (u <= 0) u = Number.EPSILON;
  if (u >= 1) u = 1 - Number.EPSILON;
  let cum = 0;
  for (let i = 0; i < p.length; i++) {
    cum += p[i];
    if (u <= cum) return i + offset;
  }
  // Numerical drift: total may be 1 − ε. Return the last index.
  return (p.length - 1) + offset;
}

function _catLogpmf(k, p, offset) {
  const idx = (k | 0) - offset;
  if (idx < 0 || idx >= p.length) return -Infinity;
  const pi = p[idx];
  return pi > 0 ? Math.log(pi) : -Infinity;
}

const randCategorical = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      return function parametricCategoricalSampler(p) {
        return _catSample(p, prng, 1);
      };
    }
    const p = args[0];
    return function staticCategoricalSampler() { return _catSample(p, prng, 1); };
  },
};

const randCategorical0 = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      return function parametricCategorical0Sampler(p) {
        return _catSample(p, prng, 0);
      };
    }
    const p = args[0];
    return function staticCategorical0Sampler() { return _catSample(p, prng, 0); };
  },
};

function CategoricalCtor(p) {
  this.p = p;
  this.support = [1, p.length];
}
CategoricalCtor.prototype.pmf = function (k) {
  const idx = (k | 0) - 1;
  return (idx < 0 || idx >= this.p.length) ? 0 : this.p[idx];
};
CategoricalCtor.prototype.logpmf = function (k) { return _catLogpmf(k, this.p, 1); };
// .pdf as alias so density() — which dispatches on `discrete` flag — works.
CategoricalCtor.prototype.pdf    = function (k) { return this.pmf(k); };
CategoricalCtor.prototype.logpdf = function (k) { return this.logpmf(k); };

function Categorical0Ctor(p) {
  this.p = p;
  this.support = [0, p.length - 1];
}
Categorical0Ctor.prototype.pmf = function (k) {
  const idx = k | 0;
  return (idx < 0 || idx >= this.p.length) ? 0 : this.p[idx];
};
Categorical0Ctor.prototype.logpmf = function (k) { return _catLogpmf(k, this.p, 0); };
Categorical0Ctor.prototype.pdf    = function (k) { return this.pmf(k); };
Categorical0Ctor.prototype.logpdf = function (k) { return this.logpmf(k); };

function logpdfCategorical(x, p)  { return _catLogpmf(x, p, 1); }
function logpdfCategorical0(x, p) { return _catLogpmf(x, p, 0); }

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
  Logistic: {
    // Spec: Logistic(mu, s) — location mu, scale s. Synthesised
    // because @stdlib/random-base-logistic isn't installed; cheap
    // inverse-CDF sampler, exact pdf/cdf/quantile.
    params:   ['mu', 's'],
    aliases:  {},
    discrete: false,
    Ctor:     LogisticCtor,
    randFn:   randLogistic,
    logpdfFn: logpdfLogistic,
  },
  Weibull: {
    // Spec: Weibull(shape, scale) — Weibull(1, 1/rate) ≡
    // Exponential(rate). Synthesised inverse-CDF sampler.
    params:   ['shape', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     WeibullCtor,
    randFn:   randWeibull,
    logpdfFn: logpdfWeibull,
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
  GeneralizedNormal: {
    // Spec: GeneralizedNormal(mean, alpha, beta). β = 2 → Normal,
    // β = 1 → Laplace. Synthesised via scaled-gamma + Rademacher.
    params:   ['mean', 'alpha', 'beta'],
    aliases:  {},
    discrete: false,
    Ctor:     GeneralizedNormalCtor,
    randFn:   randGeneralizedNormal,
    logpdfFn: logpdfGeneralizedNormal,
  },
  InverseGamma: {
    // Spec: InverseGamma(shape, scale). Synthesised via 1/Gamma(shape,
    // rate=scale) — the standard mathematical equivalence (spec §08
    // line 200).
    params:   ['shape', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     InverseGammaCtor,
    randFn:   randInverseGamma,
    logpdfFn: logpdfInverseGamma,
  },
  Categorical: {
    // Categorical(p): discrete uniform-or-not over {1, …, length(p)}.
    // p is a probability vector (already on the standard simplex per
    // spec). Synthesised cumulative-sum sampler; pmf at k is p[k−1].
    params:   ['p'],
    aliases:  {},
    discrete: true,
    Ctor:     CategoricalCtor,
    randFn:   randCategorical,
    logpdfFn: logpdfCategorical,
  },
  Categorical0: {
    // Zero-based variant: support {0, …, length(p)−1}.
    params:   ['p'],
    aliases:  {},
    discrete: true,
    Ctor:     Categorical0Ctor,
    randFn:   randCategorical0,
    logpdfFn: logpdfCategorical0,
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
  rowstack: vs => {
    if (!Array.isArray(vs) || vs.length === 0) return [];
    const n = vs[0].length;
    for (let i = 1; i < vs.length; i++) {
      if (vs[i].length !== n) {
        throw new Error('rowstack: row length mismatch at index ' + i);
      }
    }
    // Shallow copy each row so the caller's nested arrays are not aliased.
    const out = new Array(vs.length);
    for (let i = 0; i < vs.length; i++) out[i] = vs[i].slice();
    return out;
  },
  colstack: vs => {
    if (!Array.isArray(vs) || vs.length === 0) return [];
    const n = vs[0].length;
    for (let i = 1; i < vs.length; i++) {
      if (vs[i].length !== n) {
        throw new Error('colstack: column length mismatch at index ' + i);
      }
    }
    const out = new Array(n);
    for (let r = 0; r < n; r++) {
      const row = new Array(vs.length);
      for (let c = 0; c < vs.length; c++) row[c] = vs[c][r];
      out[r] = row;
    }
    return out;
  },
  // ---- Linear algebra (spec §07) ------------------------------------
  // Matrices are nested JS arrays (row-major): M[i][j] is row i, col j.
  // Vectors are flat JS arrays. We implement textbook algorithms
  // directly — no stdlib BLAS/LAPACK available — which is fine for the
  // problem sizes FlatPPL targets (HEP models with ~few × few matrices,
  // covariance matrices up to dozens of dimensions). Larger matrices
  // would warrant a native-backed engine; this implementation favours
  // clarity and self-containment.
  transpose: M => {
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
  // (Complex support deferred.)
  adjoint: M => ARITH_OPS.transpose(M),
  trace: M => {
    if (!Array.isArray(M)) throw new Error('trace: argument must be a matrix');
    const n = M.length;
    if (n === 0 || M[0].length !== n) {
      throw new Error('trace: argument must be a square matrix');
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += M[i][i];
    return s;
  },
  diagmat: v => {
    const n = v.length;
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array(n);
      for (let j = 0; j < n; j++) row[j] = (i === j) ? v[i] : 0;
      out[i] = row;
    }
    return out;
  },
  self_outer: v => {
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
  det: A => {
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('det: argument must be a non-empty square matrix');
    }
    return _detLU(A);
  },
  // logabsdet(A): log |det(A)|. Computed alongside det via the LU
  // decomposition to keep numerical stability on near-singular inputs.
  logabsdet: A => {
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('logabsdet: argument must be a non-empty square matrix');
    }
    return _logAbsDetLU(A);
  },
  // inv(A): matrix inverse via Gauss-Jordan with partial pivoting. O(n³).
  // Throws on singular matrices.
  inv: A => {
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('inv: argument must be a non-empty square matrix');
    }
    return _invGaussJordan(A);
  },
  // linsolve(A, b): solve A x = b for x. b may be a vector (returns
  // vector) or a matrix (returns matrix, solved column-by-column).
  linsolve: (A, b) => {
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('linsolve: A must be a non-empty square matrix');
    }
    return _linsolveLU(A, b);
  },
  // lower_cholesky(A): lower-triangular L with A = L L^T for symmetric
  // positive-definite A. Throws if A is not PD.
  lower_cholesky: A => {
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('lower_cholesky: argument must be a non-empty square matrix');
    }
    return _cholesky(A);
  },
  // row_gram(A) = A · A^T; col_gram(A) = A^T · A. Useful for LKJ ↔
  // LKJCholesky conversions and Gram-matrix priors.
  row_gram: A => _matmul(A, ARITH_OPS.transpose(A)),
  col_gram: A => _matmul(ARITH_OPS.transpose(A), A),
  // fill(x, n, m, ...) — n-D array of shape `n × m × ...` filled with x.
  // Returns nested JS arrays for 2-D+; flat array for 1-D. Spec §07.
  fill: (x, ...dims) => {
    if (dims.length === 0) return +x;
    function build(level) {
      const n = dims[level] | 0;
      const out = new Array(n);
      if (level === dims.length - 1) {
        for (let i = 0; i < n; i++) out[i] = x;
      } else {
        for (let i = 0; i < n; i++) out[i] = build(level + 1);
      }
      return out;
    }
    return build(0);
  },
  // zeros / ones — convenience wrappers around fill. Spec §07.
  zeros: (...dims) => {
    if (dims.length === 0) return 0;
    function build(level) {
      const n = dims[level] | 0;
      const out = new Array(n);
      if (level === dims.length - 1) {
        for (let i = 0; i < n; i++) out[i] = 0;
      } else {
        for (let i = 0; i < n; i++) out[i] = build(level + 1);
      }
      return out;
    }
    return build(0);
  },
  ones: (...dims) => {
    if (dims.length === 0) return 1;
    function build(level) {
      const n = dims[level] | 0;
      const out = new Array(n);
      if (level === dims.length - 1) {
        for (let i = 0; i < n; i++) out[i] = 1;
      } else {
        for (let i = 0; i < n; i++) out[i] = build(level + 1);
      }
      return out;
    }
    return build(0);
  },
  // eye(n) — n × n identity matrix. Spec §07.
  eye: n => {
    const k = n | 0;
    if (k <= 0) return [];
    const out = new Array(k);
    for (let i = 0; i < k; i++) {
      const row = new Array(k);
      for (let j = 0; j < k; j++) row[j] = i === j ? 1 : 0;
      out[i] = row;
    }
    return out;
  },
  // onehot(i, n) — length-n basis vector with 1 at position i (1-based
  // per FlatPPL convention). Spec §07.
  onehot: (i, n) => {
    const idx = i | 0;
    const k = n | 0;
    if (k <= 0) return [];
    if (idx < 1 || idx > k) {
      throw new Error('onehot: index ' + idx + ' out of range [1, ' + k + ']');
    }
    const out = new Array(k);
    for (let j = 0; j < k; j++) out[j] = (j === idx - 1) ? 1 : 0;
    return out;
  },
  // Scalar restrictors (spec §07). Identity at runtime; static typing
  // catches domain violations at type-check time when they're
  // discernible. The runtime versions check the obvious cases.
  boolean: x => {
    if (x === true || x === false) return x;
    if (x === 0) return false;
    if (x === 1) return true;
    throw new Error('boolean: value ' + x + ' is not a boolean');
  },
  integer: x => {
    if (Number.isInteger(x)) return x;
    throw new Error('integer: value ' + x + ' is not an integer');
  },
  // linspace(from, to, n) — endpoint-inclusive range of n real numbers
  // evenly spaced from `from` to `to`. n=1 returns [from]; both endpoints
  // are included exactly (not computed via accumulating step). Spec §07.
  linspace: (from, to, n) => {
    const k = n | 0;
    if (k <= 0) return [];
    if (k === 1) return [+from];
    const out = new Array(k);
    const lo = +from, hi = +to;
    for (let i = 0; i < k; i++) {
      // Use the parametric form (1−t)·lo + t·hi so the endpoints land
      // exactly on lo and hi without floating-point drift.
      const t = i / (k - 1);
      out[i] = (1 - t) * lo + t * hi;
    }
    return out;
  },
  // extlinspace(from, to, n) — like linspace but with -inf and +inf
  // prepended/appended. Useful for overflow-bin definitions in binned
  // analyses. Spec §07.
  extlinspace: (from, to, n) => {
    const k = n | 0;
    if (k <= 0) return [-Infinity, Infinity];
    const out = new Array(k + 2);
    out[0] = -Infinity;
    out[k + 1] = Infinity;
    if (k === 1) { out[1] = +from; return out; }
    const lo = +from, hi = +to;
    for (let i = 0; i < k; i++) {
      const t = i / (k - 1);
      out[i + 1] = (1 - t) * lo + t * hi;
    }
    return out;
  },
  // partition(xs, spec) — split a vector into groups. spec may be a
  // positive integer (equal-size groups) or a vector of positive
  // integers (custom group sizes). Spec §07. Returns a vector of
  // sub-vectors (JS arrays of JS arrays).
  partition: (xs, spec) => {
    const n = xs.length;
    if (typeof spec === 'number') {
      const k = spec | 0;
      if (k <= 0) throw new Error('partition: group size must be positive, got ' + k);
      if (n % k !== 0) {
        throw new Error('partition: length ' + n + ' not divisible by group size ' + k);
      }
      const groups = n / k;
      const out = new Array(groups);
      for (let g = 0; g < groups; g++) {
        const grp = new Array(k);
        for (let i = 0; i < k; i++) grp[i] = xs[g * k + i];
        out[g] = grp;
      }
      return out;
    }
    if (Array.isArray(spec) || (spec && spec.BYTES_PER_ELEMENT)) {
      let total = 0;
      for (let i = 0; i < spec.length; i++) total += (spec[i] | 0);
      if (total !== n) {
        throw new Error('partition: spec sums to ' + total + ' but vector length is ' + n);
      }
      const out = new Array(spec.length);
      let cursor = 0;
      for (let g = 0; g < spec.length; g++) {
        const sz = spec[g] | 0;
        const grp = new Array(sz);
        for (let i = 0; i < sz; i++) grp[i] = xs[cursor + i];
        out[g] = grp;
        cursor += sz;
      }
      return out;
    }
    throw new Error('partition: spec must be a positive integer or a vector of positive integers');
  },
  // reverse(xs) — reverse element order in a vector. Tables defer for
  // now (no canonical table runtime yet).
  reverse: xs => {
    if (xs && xs.BYTES_PER_ELEMENT) {
      // Typed array — slice and reverse to keep type, but Float64Array's
      // reverse is in-place. Make a copy first.
      const out = new Float64Array(xs.length);
      for (let i = 0; i < xs.length; i++) out[xs.length - 1 - i] = xs[i];
      return out;
    }
    if (Array.isArray(xs)) {
      const out = new Array(xs.length);
      for (let i = 0; i < xs.length; i++) out[xs.length - 1 - i] = xs[i];
      return out;
    }
    throw new Error('reverse: argument must be a vector');
  },
  cat: (...xs) => {
    if (xs.length === 0) return [];
    const first = xs[0];
    if (typeof first === 'number') {
      return xs.slice();
    }
    if (Array.isArray(first) || (first && first.BYTES_PER_ELEMENT)) {
      const out = [];
      for (let j = 0; j < xs.length; j++) {
        const v = xs[j];
        for (let i = 0; i < v.length; i++) out.push(v[i]);
      }
      return out;
    }
    if (first && typeof first === 'object') {
      const out = {};
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
  // Norms and normalization (spec §07). All take a single vector
  // argument. Numerically stable forms — logsumexp uses the standard
  // shift-by-max trick so exp doesn't overflow on large entries.
  l1norm: arr => {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i]);
    return s;
  },
  l2norm: arr => {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
    return Math.sqrt(s);
  },
  l1unit: arr => {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i]);
    if (s === 0) throw new Error('l1unit: zero-norm vector has no unit form');
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] / s;
    return out;
  },
  l2unit: arr => {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
    if (s === 0) throw new Error('l2unit: zero-norm vector has no unit form');
    const r = Math.sqrt(s);
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] / r;
    return out;
  },
  logsumexp: arr => {
    const n = arr.length;
    if (n === 0) return -Infinity;
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
    if (!Number.isFinite(m)) return m;   // all -Inf → result is -Inf
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.exp(arr[i] - m);
    return m + Math.log(s);
  },
  softmax: arr => {
    const n = arr.length;
    if (n === 0) return [];
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
    if (!Number.isFinite(m)) {
      // All -Inf: degenerate uniform on the simplex would be the
      // continuous limit, but emit zeros (mass shifts to nowhere).
      // Match the standard library behavior on all-zero exp inputs.
      throw new Error('softmax: all-(-Infinity) input is undefined');
    }
    const exps = new Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) { exps[i] = Math.exp(arr[i] - m); s += exps[i]; }
    for (let i = 0; i < n; i++) exps[i] /= s;
    return exps;
  },
  logsoftmax: arr => {
    const n = arr.length;
    if (n === 0) return [];
    let m = -Infinity;
    for (let i = 0; i < n; i++) if (arr[i] > m) m = arr[i];
    if (!Number.isFinite(m)) {
      throw new Error('logsoftmax: all-(-Infinity) input is undefined');
    }
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.exp(arr[i] - m);
    const lse = m + Math.log(s);
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = arr[i] - lse;
    return out;
  },
};

// =====================================================================
// Linear-algebra helpers (textbook algorithms; small-matrix sized)
// =====================================================================

// LU decomposition with partial pivoting. Returns
//   { LU: in-place factorized matrix, piv: row-permutation, sign: ±1 }
// where LU stores L (below diagonal, with implicit unit diagonal) and U
// (on and above diagonal) in a single n×n array. piv[i] holds the row
// at row i after permutation; sign tracks the parity of row swaps so
// the caller can recover det(A) = sign · prod(diag(U)).
function _luDecomp(A) {
  const n = A.length;
  // Deep-copy A so the caller's matrix isn't mutated.
  const LU = new Array(n);
  for (let i = 0; i < n; i++) LU[i] = A[i].slice();
  const piv = new Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  let sign = 1;
  for (let k = 0; k < n; k++) {
    // Partial pivot: find row with max |a[r][k]| for r ≥ k.
    let maxAbs = Math.abs(LU[k][k]);
    let maxRow = k;
    for (let r = k + 1; r < n; r++) {
      const v = Math.abs(LU[r][k]);
      if (v > maxAbs) { maxAbs = v; maxRow = r; }
    }
    if (maxAbs === 0) return { LU, piv, sign: 0 };  // singular
    if (maxRow !== k) {
      const tmp = LU[k]; LU[k] = LU[maxRow]; LU[maxRow] = tmp;
      const tp = piv[k]; piv[k] = piv[maxRow]; piv[maxRow] = tp;
      sign = -sign;
    }
    // Eliminate below the diagonal.
    const pivot = LU[k][k];
    for (let r = k + 1; r < n; r++) {
      const factor = LU[r][k] / pivot;
      LU[r][k] = factor;
      for (let c = k + 1; c < n; c++) {
        LU[r][c] -= factor * LU[k][c];
      }
    }
  }
  return { LU, piv, sign };
}

function _detLU(A) {
  const { LU, sign } = _luDecomp(A);
  if (sign === 0) return 0;
  const n = LU.length;
  let det = sign;
  for (let i = 0; i < n; i++) det *= LU[i][i];
  return det;
}

function _logAbsDetLU(A) {
  const { LU, sign } = _luDecomp(A);
  if (sign === 0) return -Infinity;
  const n = LU.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.log(Math.abs(LU[i][i]));
  return s;
}

// Solve A · x = b given the LU factorization. b may be a vector or a
// row-major matrix. Forward substitution (L) followed by backward
// substitution (U).
function _linsolveLU(A, b) {
  const { LU, piv, sign } = _luDecomp(A);
  if (sign === 0) throw new Error('linsolve: matrix is singular');
  const n = LU.length;
  const isMat = Array.isArray(b[0]);
  function solveOne(bvec) {
    // Apply permutation.
    const y = new Array(n);
    for (let i = 0; i < n; i++) y[i] = bvec[piv[i]];
    // Forward: L y = Pb (L has unit diagonal).
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) y[i] -= LU[i][j] * y[j];
    }
    // Backward: U x = y.
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let j = i + 1; j < n; j++) s -= LU[i][j] * x[j];
      x[i] = s / LU[i][i];
    }
    return x;
  }
  if (!isMat) {
    if (b.length !== n) throw new Error('linsolve: dimension mismatch');
    return solveOne(b);
  }
  // Matrix b: solve column by column.
  const ncols = b[0].length;
  const cols = new Array(ncols);
  for (let c = 0; c < ncols; c++) {
    const bc = new Array(n);
    for (let i = 0; i < n; i++) bc[i] = b[i][c];
    cols[c] = solveOne(bc);
  }
  // Stitch column-major back to row-major.
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(ncols);
    for (let c = 0; c < ncols; c++) row[c] = cols[c][i];
    out[i] = row;
  }
  return out;
}

// Gauss-Jordan inverse. Slightly less numerically stable than
// "linsolve(A, I)" for large matrices but fine at the FlatPPL target
// sizes. We use the LU-based form for symmetry with linsolve.
function _invGaussJordan(A) {
  const n = A.length;
  // Solve A X = I via the LU path.
  const I = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = (i === j) ? 1 : 0;
    I[i] = row;
  }
  return _linsolveLU(A, I);
}

// Cholesky factorization for symmetric positive-definite A. Returns
// lower-triangular L with A = L · L^T. Diagonal entries are positive.
// Standard recursion (Cholesky-Banachiewicz form).
function _cholesky(A) {
  const n = A.length;
  const L = new Array(n);
  for (let i = 0; i < n; i++) L[i] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) {
          throw new Error('lower_cholesky: matrix is not positive definite '
            + '(non-positive pivot at row ' + i + ')');
        }
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

// Matrix-matrix multiplication. Handles non-square shapes.
function _matmul(A, B) {
  const arows = A.length;
  if (arows === 0) return [];
  const acols = A[0].length;
  const brows = B.length;
  if (acols !== brows) {
    throw new Error('matmul: dimension mismatch (' + arows + '×' + acols
      + ' · ' + brows + '×?)');
  }
  const bcols = B[0].length;
  const out = new Array(arows);
  for (let i = 0; i < arows; i++) {
    const row = new Array(bcols);
    for (let j = 0; j < bcols; j++) {
      let s = 0;
      for (let k = 0; k < acols; k++) s += A[i][k] * B[k][j];
      row[j] = s;
    }
    out[i] = row;
  }
  return out;
}

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

// Resolve a function-positional argument used by higher-order ops
// (filter, reduce, scan) into { body, params } regardless of whether
// the IR is an inline `functionof(...)` call or a self-ref to a named
// fn / functionof / kernelof binding. The orchestrator's pre-eval
// attaches env.__resolveFnBody for the ref path; the inline path
// reads `body` / `params` directly off the functionof IR.
function _resolveFn(fnIR, env) {
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
    const arrs = sources.map(s => evaluateExpr(s, env));
    if (arrs.length === 0) return [];
    const n = arrs[0].length;
    for (let i = 1; i < arrs.length; i++) {
      if (arrs[i].length !== n) {
        throw new Error('broadcast: array length mismatch at position ' + i
          + ' (expected ' + n + ', got ' + arrs[i].length + ')');
      }
    }
    const elemEnv = Object.assign({}, env);
    const out = new Array(n);
    for (let r = 0; r < n; r++) {
      for (let p = 0; p < fn.params.length; p++) {
        elemEnv[fn.params[p]] = arrs[p][r];
      }
      out[r] = evaluateExpr(fn.body, elemEnv);
    }
    return out;
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
    const xs = evaluateExpr(args[1], env);
    if (!Array.isArray(xs) && !(xs && xs.BYTES_PER_ELEMENT)) {
      throw new Error('reduce: xs must be a vector');
    }
    if (xs.length === 0) {
      throw new Error('reduce: empty vector has no initial value');
    }
    const elemEnv = Object.assign({}, env);
    let acc = xs[0];
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
    const xs   = evaluateExpr(args[2], env);
    if (!Array.isArray(xs) && !(xs && xs.BYTES_PER_ELEMENT)) {
      throw new Error('scan: xs must be a vector');
    }
    const out = new Array(xs.length);
    const elemEnv = Object.assign({}, env);
    let acc = init;
    for (let i = 0; i < xs.length; i++) {
      elemEnv[fn.params[0]] = acc;
      elemEnv[fn.params[1]] = xs[i];
      acc = evaluateExpr(fn.body, elemEnv);
      out[i] = acc;
    }
    return out;
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
    const data = evaluateExpr(args[1], env);
    if (!Array.isArray(data) && !(data && data.BYTES_PER_ELEMENT)) {
      throw new Error('filter: data must be a vector (got '
        + (data === null ? 'null' : typeof data) + ')');
    }
    const elemEnv = Object.assign({}, env);
    const out = [];
    for (let i = 0; i < data.length; i++) {
      elemEnv[fn.params[0]] = data[i];
      const keep = evaluateExpr(fn.body, elemEnv);
      if (keep) out.push(data[i]);
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
