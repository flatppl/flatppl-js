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

const rng = require('./rng.ts');
const valueLib = require('./value.ts');
const valueOps = require('./value-ops.ts');

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
function logpmfBernoulli(x: any, p: any) {
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
      return function parametricDiracSampler(value: any) { return +value; };
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
function DiracCtor(this: any, value: any) {
  this.value = +value;
  this.mean = +value;
  this.variance = 0;
  this.stdev = 0;
  this.support = [+value, +value];
}
DiracCtor.prototype.pdf      = function(x: any) { return x === this.value ? 1 : 0; };
DiracCtor.prototype.logpdf   = function(x: any) { return x === this.value ? 0 : -Infinity; };
DiracCtor.prototype.cdf      = function(x: any) { return x < this.value ? 0 : 1; };
DiracCtor.prototype.quantile = function(this: any, _p: any) { return this.value; };

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

function uClip(u: any) {
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
      return function parametricLogisticSampler(mu: any, s: any) {
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

function LogisticCtor(this: any, mu: any, s: any) {
  this.mu = +mu; this.s = +s;
  this.mean = +mu;
  this.variance = (s * s) * (Math.PI * Math.PI) / 3;
  this.stdev = Math.sqrt(this.variance);
  this.support = [-Infinity, Infinity];
}
LogisticCtor.prototype.pdf = function(this: any, x: any) {
  const z = (x - this.mu) / this.s;
  const ez = Math.exp(-z);
  const denom = 1 + ez;
  return ez / (this.s * denom * denom);
};
LogisticCtor.prototype.logpdf = function(this: any, x: any) {
  const z = (x - this.mu) / this.s;
  return -z - Math.log(this.s) - 2 * Math.log(1 + Math.exp(-z));
};
LogisticCtor.prototype.cdf = function(this: any, x: any) {
  return 1 / (1 + Math.exp(-(x - this.mu) / this.s));
};
LogisticCtor.prototype.quantile = function(this: any, p: any) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  return this.mu + this.s * Math.log(p / (1 - p));
};
function logpdfLogistic(x: any, mu: any, s: any) {
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
      return function parametricWeibullSampler(k: any, lambda: any) {
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

function WeibullCtor(this: any, shape: any, scale: any) {
  // FlatPPL spec names: shape (k), scale (λ).
  this.k = +shape; this.lambda = +scale;
  // Mean = λ · Γ(1 + 1/k); variance = λ² · (Γ(1 + 2/k) − Γ(1 + 1/k)²).
  // We don't have stdlib gamma in scope here without adding a require —
  // the analytical handle leaves these undefined; plot ranges fall
  // back to quantile-based bounds.
  this.support = [0, Infinity];
}
WeibullCtor.prototype.pdf = function(this: any, x: any) {
  if (x < 0) return 0;
  if (x === 0) return this.k === 1 ? 1 / this.lambda : (this.k > 1 ? 0 : Infinity);
  const z = x / this.lambda;
  return (this.k / this.lambda) * Math.pow(z, this.k - 1) * Math.exp(-Math.pow(z, this.k));
};
WeibullCtor.prototype.logpdf = function(this: any, x: any) {
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
WeibullCtor.prototype.cdf = function(this: any, x: any) {
  if (x <= 0) return 0;
  return 1 - Math.exp(-Math.pow(x / this.lambda, this.k));
};
WeibullCtor.prototype.quantile = function(this: any, p: any) {
  if (p <= 0) return 0;
  if (p >= 1) return Infinity;
  return this.lambda * Math.pow(-Math.log(1 - p), 1 / this.k);
};
function logpdfWeibull(x: any, k: any, lambda: any) {
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
      return function parametricGenNormalSampler(mean: any, alpha: any, beta: any) {
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

function GeneralizedNormalCtor(this: any, mean: any, alpha: any, beta: any) {
  this.mean  = +mean;
  this.alpha = +alpha;
  this.beta  = +beta;
  this.support = [-Infinity, Infinity];
}
GeneralizedNormalCtor.prototype.pdf = function(this: any, x: any) {
  return Math.exp(this.logpdf(x));
};
GeneralizedNormalCtor.prototype.logpdf = function(this: any, x: any) {
  const a = this.alpha, b = this.beta;
  const z = Math.abs(x - this.mean) / a;
  return Math.log(b) - Math.log(2 * a) - stdlibGammaln(1 / b) - Math.pow(z, b);
};
function logpdfGeneralizedNormal(x: any, mean: any, alpha: any, beta: any) {
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
      return function parametricInverseGammaSampler(shape: any, scale: any) {
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

function InverseGammaCtor(this: any, shape: any, scale: any) {
  this.shape = +shape;
  this.scale = +scale;
  this.support = [0, Infinity];
}
InverseGammaCtor.prototype.pdf = function(this: any, x: any) {
  if (x <= 0) return 0;
  // β^α / Γ(α) · x^(-α-1) · exp(-β/x)
  return Math.exp(this.logpdf(x));
};
InverseGammaCtor.prototype.logpdf = function(this: any, x: any) {
  if (x <= 0) return -Infinity;
  const a = this.shape, b = this.scale;
  return a * Math.log(b) - stdlibGammaln(a) - (a + 1) * Math.log(x) - b / x;
};
function logpdfInverseGamma(x: any, shape: any, scale: any) {
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

function _catSample(p: any, prng: any, offset: any) {
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

function _catLogpmf(k: any, p: any, offset: any) {
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
      return function parametricCategoricalSampler(p: any) {
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
      return function parametricCategorical0Sampler(p: any) {
        return _catSample(p, prng, 0);
      };
    }
    const p = args[0];
    return function staticCategorical0Sampler() { return _catSample(p, prng, 0); };
  },
};

function CategoricalCtor(this: any, p: any) {
  this.p = p;
  this.support = [1, p.length];
}
CategoricalCtor.prototype.pmf = function(this: any, k: any) {
  const idx = (k | 0) - 1;
  return (idx < 0 || idx >= this.p.length) ? 0 : this.p[idx];
};
CategoricalCtor.prototype.logpmf = function(this: any, k: any) { return _catLogpmf(k, this.p, 1); };
// .pdf as alias so density() — which dispatches on `discrete` flag — works.
CategoricalCtor.prototype.pdf    = function (k: any) { return this.pmf(k); };
CategoricalCtor.prototype.logpdf = function(this: any, k: any) { return this.logpmf(k); };

function Categorical0Ctor(this: any, p: any) {
  this.p = p;
  this.support = [0, p.length - 1];
}
Categorical0Ctor.prototype.pmf = function(this: any, k: any) {
  const idx = k | 0;
  return (idx < 0 || idx >= this.p.length) ? 0 : this.p[idx];
};
Categorical0Ctor.prototype.logpmf = function(this: any, k: any) { return _catLogpmf(k, this.p, 0); };
Categorical0Ctor.prototype.pdf    = function (k: any) { return this.pmf(k); };
Categorical0Ctor.prototype.logpdf = function(this: any, k: any) { return this.logpmf(k); };

function logpdfCategorical(x: any, p: any)  { return _catLogpmf(x, p, 1); }
function logpdfCategorical0(x: any, p: any) { return _catLogpmf(x, p, 0); }

function logpdfDirac(x: any, value: any) {
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
    customResolveParams: function (measureIR: any, env: any) {
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
function isKnownDistribution(name: any) {
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
function evaluateExpr(ir: any, env: any): any {
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

function evaluateExprN(ir: any, refArrays: any, count: any, baseEnv: any, opts: any) {
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
      // Phase 7a: batched approximation functions. Each has the shape
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

// Phase 7a: batched approximation function dispatch (polynomial /
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
  // Preserve Phase 1 "same kind as inputs" semantics: if xVal was a
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
  // Pre-compute per-ref accessors. Phase 8: refArrays uniformly carry
  // Values internally (post unification); legacy Float64Array entries
  // are accepted as a back-compat path for tests that pass refArrays
  // directly. shape=[N] (scalar atoms) → `i => data[i]`; shape=[N,
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
  // Phase 7c: per-atom uniform-length numeric array outputs (e.g.
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
//
// Scalar complex representation: a plain JS object `{re, im}` of two
// numbers. Operations auto-promote real → complex when meeting a
// complex operand. Storage is plain-object rather than a class so
// JSON round-trips work and structured clone preserves shape across
// postMessage.
//
// For atom-batched complex storage (a per-atom complex value), the
// design is: Value with `dtype: 'complex'`, `data: Float64Array(N)` for
// real parts, `im: Float64Array(N)` for imaginary parts (separate
// buffers, matching TF.js's complex-tensor representation). That
// extension lands when a concrete use case demands it; for now scalar
// complex values are sufficient for the spec §07 operator set.

function _isComplex(v: any) {
  return v != null && typeof v === 'object'
    && typeof v.re === 'number' && typeof v.im === 'number';
}

function _toComplex(v: any) {
  if (_isComplex(v)) return v;
  if (typeof v === 'number') return { re: v, im: 0 };
  if (typeof v === 'boolean') return { re: v ? 1 : 0, im: 0 };
  throw new Error('cannot promote ' + typeof v + ' to complex');
}

function _cAdd(a: any, b: any) { return { re: a.re + b.re, im: a.im + b.im }; }
function _cSub(a: any, b: any) { return { re: a.re - b.re, im: a.im - b.im }; }
function _cNeg(a: any)    { return { re: -a.re,       im: -a.im }; }
function _cMul(a: any, b: any) {
  return { re: a.re * b.re - a.im * b.im,
           im: a.re * b.im + a.im * b.re };
}
function _cDiv(a: any, b: any) {
  const d = b.re * b.re + b.im * b.im;
  return { re: (a.re * b.re + a.im * b.im) / d,
           im: (a.im * b.re - a.re * b.im) / d };
}
function _cConj(a: any)   { return { re: a.re, im: -a.im }; }
function _cAbs(a: any)    { return Math.hypot(a.re, a.im); }
function _cAbs2(a: any)   { return a.re * a.re + a.im * a.im; }
function _cExp(z: any) {
  const r = Math.exp(z.re);
  return { re: r * Math.cos(z.im), im: r * Math.sin(z.im) };
}
function _cLog(z: any) {
  // Principal branch: arg ∈ (-π, π] per spec §07.
  return { re: Math.log(Math.hypot(z.re, z.im)),
           im: Math.atan2(z.im, z.re) };
}
function _cSqrt(z: any) {
  // Principal branch.
  if (z.im === 0 && z.re >= 0) return { re: Math.sqrt(z.re), im: 0 };
  const r = Math.hypot(z.re, z.im);
  // half-angle formulas avoiding catastrophic cancellation near the
  // negative real axis (preferred over re-deriving via _cExp(0.5·_cLog)).
  const reOut = Math.sqrt((r + z.re) / 2);
  const sign = z.im >= 0 ? 1 : -1;
  return { re: reOut, im: sign * Math.sqrt((r - z.re) / 2) };
}
function _cPow(base: any, exp: any) {
  // z^w = exp(w · log(z)), principal branch (spec §07).
  return _cExp(_cMul(exp, _cLog(base)));
}

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

// Phase 2c: shape-aware add/sub/neg dispatcher (used inline in the
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
  // Common unary maths — extend EVALUABLE_OPS in orchestrator.js
  // alongside any addition here so the static gate matches. Complex
  // arguments dispatch to the complex helpers (spec §07: exp / log /
  // sqrt / abs / abs2 extend naturally; sin/cos defer until needed).
  abs:   (a: any) => _isComplex(a) ? _cAbs(a) : Math.abs(a),
  abs2:  (a: any) => _isComplex(a) ? _cAbs2(a) : a * a,
  exp:   (a: any) => _isComplex(a) ? _cExp(a) : Math.exp(a),
  log:   (a: any) => _isComplex(a) ? _cLog(a) : Math.log(a),
  log10: (a: any) => Math.log10(a),
  sqrt:  (a: any) => _isComplex(a) ? _cSqrt(a) : Math.sqrt(a),
  sin:   (a: any) => Math.sin(a),
  cos:   (a: any) => Math.cos(a),
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
  vector: (...xs: any[]) => xs,
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
  rowstack: (vs: any) => {
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
  colstack: (vs: any) => {
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
  // Phase 3 of the shape-explicit refactor: ops dispatch on Value
  // inputs. Value path bridges through nested-array form for the
  // matrix-decomposition algorithms (atom-indep one-shot ops; the
  // realloc cost is negligible against the algorithm work). transpose
  // and adjoint use the free Klein-4 tag flip on Value input (no
  // data allocation).
  //
  // Legacy nested-JS-array inputs continue to work unchanged — the
  // existing linear-algebra test suite exercises them via the IR
  // pipeline. Once materialisers (Phase 4) produce Values, the
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
      return valueLib.scalar(ARITH_OPS.trace(valueOps._valueToNested(M)));
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
      const vec = v.shape.length === 1 ? v : valueLib.asValue(valueOps._valueToNested(v));
      return valueLib.diagMatrix(vec.data, vec.im);
    }
    return valueLib.diagMatrix(v instanceof Float64Array ? v : Float64Array.from(v));
  },
  self_outer: (v: any): any => {
    if (valueLib.isValue(v)) {
      return valueOps._nestedToValue(ARITH_OPS.self_outer(valueOps._valueToNested(v)));
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
      return valueLib.scalar(ARITH_OPS.det(valueOps._valueToNested(A)));
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
      return valueLib.scalar(ARITH_OPS.logabsdet(valueOps._valueToNested(A)));
    }
    if (!Array.isArray(A) || A.length === 0 || A[0].length !== A.length) {
      throw new Error('logabsdet: argument must be a non-empty square matrix');
    }
    return _logAbsDetLU(A);
  },
  // inv(A): matrix inverse via Gauss-Jordan with partial pivoting. O(n³).
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
      return valueOps._nestedToValue(ARITH_OPS.inv(valueOps._valueToNested(A)));
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
      const aN = valueLib.isValue(A) ? valueOps._valueToNested(A) : A;
      const bN = valueLib.isValue(b) ? valueOps._valueToNested(b) : b;
      return valueOps._nestedToValue(ARITH_OPS.linsolve(aN, bN));
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
      return valueOps._nestedToValue(
        ARITH_OPS.lower_cholesky(valueOps._valueToNested(A)));
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
    const n = size.length;
    if (dimorder.length !== n) {
      throw new Error('array: dimorder length ' + dimorder.length
        + ' must match size length ' + n);
    }
    let total = 1;
    for (let i = 0; i < n; i++) total *= (size[i] | 0);
    if (data.length !== total) {
      throw new Error('array: prod(size) = ' + total
        + ' does not match data length ' + data.length);
    }
    if (n === 0) return data.length > 0 ? data[0] : null;
    // Build nested array of the desired shape (using row-major
    // convention internally), then traverse `data` in dimorder.
    // For each linear index k into data, decode it via dimorder to
    // get the n-D coordinate.
    //   k = ∑ c[dimorder[i] - 1] · stride_in_dimorder
    // where the slowest-varying axis is dimorder[0].
    function makeShape(level: any) {
      if (level === n) return 0;
      const out = new Array(size[level]);
      for (let i = 0; i < size[level]; i++) out[i] = makeShape(level + 1);
      return out;
    }
    const result = makeShape(0);
    function setAt(coord: any, value: any) {
      let cur: any = result;
      for (let level = 0; level < n - 1; level++) cur = cur[coord[level]];
      cur[coord[n - 1]] = value;
    }
    // Compute stride per axis index in dimorder. Slowest-varying axis
    // has stride = prod of all faster axes' sizes.
    const stridesInDimorder = new Array(n);
    let strideAcc = 1;
    for (let i = n - 1; i >= 0; i--) {
      // dimorder[i] is the (1-based) axis index for position i in
      // the traversal; size[dimorder[i] - 1] is that axis's length.
      stridesInDimorder[i] = strideAcc;
      strideAcc *= size[(dimorder[i] | 0) - 1];
    }
    const coord = new Array(n);
    for (let k = 0; k < total; k++) {
      // Decode k into per-axis indices via dimorder + strides.
      let rem = k;
      for (let i = 0; i < n; i++) {
        const axis = (dimorder[i] | 0) - 1;
        const stride = stridesInDimorder[i];
        coord[axis] = Math.floor(rem / stride) % size[axis];
        rem %= stride;
      }
      setAt(coord, data[k]);
    }
    return result;
  },
  // fill(x, n, m, ...) — n-D array of shape `n × m × ...` filled with x.
  // Returns nested JS arrays for 2-D+; flat array for 1-D. Spec §07.
  fill: (x: any, ...dims: any[]) => {
    if (dims.length === 0) return +x;
    function build(level: any) {
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
  zeros: (...dims: any[]) => {
    if (dims.length === 0) return 0;
    function build(level: any) {
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
  ones: (...dims: any[]) => {
    if (dims.length === 0) return 1;
    function build(level: any) {
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
  // eye(n) → n×n identity, as a vector-backed diag of ones.
  eye: (n: any) => {
    const k = n | 0;
    if (k <= 0) return [];
    const d = new Float64Array(k);
    d.fill(1);
    return valueLib.diagMatrix(d);
  },
  // onehot(i, n) — length-n basis vector with 1 at position i (1-based
  // per FlatPPL convention). Spec §07.
  onehot: (i: any, n: any) => {
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
  linspace: (from: any, to: any, n: any) => {
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
  extlinspace: (from: any, to: any, n: any) => {
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
  partition: (xs: any, spec: any) => {
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
  reverse: (xs: any) => {
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
    if (xs.length === 0) return [];
    const first = xs[0];
    if (typeof first === 'number') {
      return xs.slice();
    }
    if (Array.isArray(first) || (first && first.BYTES_PER_ELEMENT)) {
      const out: any[] = [];
      for (let j = 0; j < xs.length; j++) {
        const v = xs[j];
        for (let i = 0; i < v.length; i++) out.push(v[i]);
      }
      return out;
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
  // Reductions over an array. Operate on JS arrays / TypedArrays /
  // Values (Phase 7b) alike. `_arrLike` unwraps the underlying
  // indexable form. Spec semantics:
  //   sum     = Σ x[i]
  //   mean    = sum / N
  //   prod    = Π x[i]
  //   length  = N
  //   maximum = max x[i]   (Math.max would .apply-blow-stack at length 1e6)
  //   minimum = min x[i]
  //   var     = mean( (x - mean)² )  — population variance, divisor N
  sum:     (a: any) => { const arr = _arrLike(a); let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i]; return s; },
  mean:    (a: any) => { const arr = _arrLike(a); let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i]; return s / arr.length; },
  prod:    (a: any) => { const arr = _arrLike(a); let p = 1;
    for (let i = 0; i < arr.length; i++) p *= arr[i]; return p; },
  length:  (a: any) => _arrLike(a).length,
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
  var: (a: any) => {
    const arr = _arrLike(a);
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
  l1unit: (a: any) => {
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += Math.abs(arr[i]);
    if (s === 0) throw new Error('l1unit: zero-norm vector has no unit form');
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] / s;
    return out;
  },
  l2unit: (a: any) => {
    const arr = _arrLike(a);
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
    if (s === 0) throw new Error('l2unit: zero-norm vector has no unit form');
    const r = Math.sqrt(s);
    const out = new Array(arr.length);
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] / r;
    return out;
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
  softmax: (a: any) => {
    const arr = _arrLike(a);
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
  logsoftmax: (a: any) => {
    const arr = _arrLike(a);
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

// _arrLike: unwrap Value to its underlying Float64Array data, or pass
// through Float64Array / JS array. Used by reductions (sum, mean,
// l1norm, etc.) so they handle Value inputs uniformly without
// branching on every loop iteration.
function _arrLike(v: any) {
  if (valueLib.isValue(v)) return v.data;
  return v;
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
  abs: 1, abs2: 1, exp: 1, log: 1, log10: 1, sqrt: 1,
  sin: 1, cos: 1, floor: 1, ceil: 1, round: 1,
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
// Polymorphic on Value inputs (Phase 1 of the shape-explicit refactor;
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

// Phase 2b/2c/2d: shape-aware mul / add / sub / neg take precedence over
// the broadcast{1,2}-based dispatch when any operand carries an
// intrinsic vector/matrix shape (Value with rank ≥ 1 whose leading
// dim isn't the atom count). Bare scalars and batched scalars
// (shape=[N]) stay on the scalar broadcast path.
//
// Phase 2d adds the atom-batched cross — when either operand has a
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
  if (r === 1 && v.shape[0] === N) return false;  // batched scalar (Phase 1)
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

// LU decomposition with partial pivoting. Returns
//   { LU: in-place factorized matrix, piv: row-permutation, sign: ±1 }
// where LU stores L (below diagonal, with implicit unit diagonal) and U
// (on and above diagonal) in a single n×n array. piv[i] holds the row
// at row i after permutation; sign tracks the parity of row swaps so
// the caller can recover det(A) = sign · prod(diag(U)).
function _luDecomp(A: any) {
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

function _detLU(A: any) {
  const { LU, sign } = _luDecomp(A);
  if (sign === 0) return 0;
  const n = LU.length;
  let det = sign;
  for (let i = 0; i < n; i++) det *= LU[i][i];
  return det;
}

function _logAbsDetLU(A: any) {
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
function _linsolveLU(A: any, b: any) {
  const { LU, piv, sign } = _luDecomp(A);
  if (sign === 0) throw new Error('linsolve: matrix is singular');
  const n = LU.length;
  const isMat = Array.isArray(b[0]);
  function solveOne(bvec: any) {
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
function _invGaussJordan(A: any) {
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
function _cholesky(A: any) {
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
function _matmul(A: any, B: any) {
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
function regionBoundsFromIR(ir: any, env: any) {
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
      out[i] = recur(axis + 1);
    }
    return out;
  }
  return recur(0);
}

function evaluateCall(ir: any, env: any): any {
  const op = ir.op;
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
    // `all` (axis slice) is a selector token, not a value — keep it as
    // a sentinel; every other selector evaluates to an index / key /
    // subset array.
    const ALL = Symbol('all');
    const sels = args.slice(1).map((a: any) =>
      (a && a.kind === 'const' && a.name === 'all')
        ? ALL : evaluateExpr(a, env));
    const isArrayLike = (c: any) => Array.isArray(c) || ArrayBuffer.isView(c);
    const applyGet = (c: any, ss: any): any => {
      if (ss.length === 0) return c;
      const s = ss[0], rest = ss.slice(1);
      if (s === ALL) {                       // whole axis: map over it
        if (!isArrayLike(c)) {
          throw new Error(`evaluateExpr: ${op} ':' axis-slice target is not an array`);
        }
        const ca = c as any;
        const out: any[] = [];
        for (let i = 0; i < ca.length; i++) out.push(applyGet(ca[i], rest));
        return out;
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
        const ca = c as any;
        const sn = s as number | boolean;
        const idx = (oneBased ? ((sn as number) | 0) - 1 : ((sn as number) | 0));
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
    const coeffs: any = kw.coefficients != null ? evaluateExpr(kw.coefficients, env)
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
    const xs: any = evaluateExpr(args[1], env);
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
    const xs: any   = evaluateExpr(args[2], env);
    if (!Array.isArray(xs) && !(xs && xs.BYTES_PER_ELEMENT)) {
      throw new Error('scan: xs must be a vector');
    }
    const out: any = new Array(xs.length);
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
    const out: any[] = [];
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
    const bins: any = kw.bins != null ? evaluateExpr(kw.bins, env)
                                 : evaluateExpr(ir.args[0], env);
    const data = kw.data != null ? evaluateExpr(kw.data, env)
                                 : evaluateExpr(ir.args[1], env);
    if (!bins || typeof bins.length !== 'number'
        || (bins.length > 0 && typeof bins[0] !== 'number')) {
      throw new Error('bincounts: multi-dimensional binning not yet supported');
    }
    const n: any = bins.length - 1;
    if (n < 0) throw new Error('bincounts: bins must have at least 1 edge');
    const counts: any = new Array(n).fill(0);
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
    const values: any = kw.values != null ? evaluateExpr(kw.values, env)
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
    const seed: any = evaluateExpr(args[0], env);
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
    const bytes: any = evaluateExpr(args[0], env);
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
function isByteVector(x: any) {
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

function lookupDistribution(measureIR: any) {
  if (!measureIR || measureIR.kind !== 'call') {
    throw new Error(
      `sampler: expected a measure call IR, got kind=${measureIR?.kind}`
    );
  }
  const name = measureIR.op;
  const entry = (REGISTRY as any)[name];
  if (!entry) {
    throw new Error(
      `sampler: '${name}' is not a known distribution (or not yet implemented). ` +
      `Known: ${listDistributions().join(', ')}`
    );
  }
  return entry;
}

function resolveParams(measureIR: any, entry: any, env: any) {
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
  const out: any[] = [];
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

function makePhiloxPrngAdapter(initialState: any) {
  let state = initialState;
  function prng() {
    const [u, next] = rng.nextUniform(state);
    state = next;
    return u;
  }
  prng.getState = () => state;
  // Also expose a reset hook for tests.
  prng._setState = (s: any) => { state = s; };
  return prng;
}

// =====================================================================
// Misc helpers
// =====================================================================

function readSupport(dist: any) {
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
