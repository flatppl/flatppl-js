'use strict';

// Sampler distribution registry — all built-in measure kernels plus
// the per-call dispatch helpers (lookupDistribution / resolveParams),
// the Philox PRNG bridge, and the structural-set → numeric-bounds
// helper used by `Uniform`'s `customResolveParams`.
//
// Each REGISTRY entry holds:
//   - `params`:   ordered FlatPPL parameter names (spec)
//   - `aliases`:  spec-name → alternate-name map (empty for v1)
//   - `discrete`: true for counting-reference measures (Bernoulli /
//                 Binomial / Poisson / Geometric / NegativeBinomial /
//                 NegativeBinomial2 / Categorical / Categorical0)
//   - `Ctor`:     analytical constructor (.pdf / .cdf / .quantile / …)
//   - `randFn`:   `.factory(...params, opts)` returning a sampler closure
//   - `logpdfFn`: positional `(x, ...params) → log p(x)` density
//   - `customResolveParams`: optional hook for non-numeric surface params
//                            (e.g. `Uniform(support = interval(lo, hi))`)
//
// Standard modules add distributions by pushing into this REGISTRY
// (see TODO §09). The split here keeps that growth-prone surface
// isolated from the evaluator core in sampler.ts.

const rng      = require('./rng.ts');
const valueLib = require('./value.ts');
const transforms = require('./transforms.ts');

// Math special functions for gamma / loggamma / erf-based math.
const stdlibGamma       = require('@stdlib/math-base-special-gamma');
const stdlibGammaln     = require('@stdlib/math-base-special-gammaln');
// @stdlib's own `ln` (not native Math.log). The scalar Exponential path
// goes through @stdlib/random-base-exponential, whose transform is
// `-ln(1 - u) / lambda` using THIS ln; the batched randNExponential must
// use the same primitive to stay bit-exact (the two ln impls differ in
// the last ULP on ~1.6% of inputs).
const stdlibLn          = require('@stdlib/math-base-special-ln');
const stdlibErfc        = require('@stdlib/math-base-special-erfc');
const stdlibErfcinv     = require('@stdlib/math-base-special-erfcinv');
// Regularized lower incomplete gamma P(s, x): `gammainc(x, s)` (x = upper
// limit of integration, s = shape). Used for the Argus normalization.
const stdlibGammainc    = require('@stdlib/math-base-special-gammainc');

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

// Random-sample factories.
// Normal / LogNormal sample via the spec-canonical inverse-CDF transport
// (`probit`, Φ⁻¹) defined below — NOT stdlib's ziggurat normal — so the
// scalar randFn loop and the batched randNFn kernel consume the same
// Philox uniform stream and agree byte-for-byte (see randNormal /
// randNNormal below). The other dists keep their stdlib samplers.
const randExponential = require('@stdlib/random-base-exponential');
const randUniform     = require('@stdlib/random-base-uniform');
const randBeta        = require('@stdlib/random-base-beta');
const randGamma       = require('@stdlib/random-base-gamma');
const randCauchy      = require('@stdlib/random-base-cauchy');
const randT           = require('@stdlib/random-base-t');
const randBernoulli   = require('@stdlib/random-base-bernoulli');
const randBinomial    = require('@stdlib/random-base-binomial');
const randPoisson     = require('@stdlib/random-base-poisson');

// Log-density / log-mass functions (Lebesgue or counting reference).
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

// =====================================================================
// Synthetic distributions (no stdlib backing)
// =====================================================================

// Dirac point-mass measure. There's no stdlib distribution to wrap, so
// we synthesise the same shape (Ctor + factory + logpdf) the rest of
// the registry uses, so makeSampler / makeAnalytical / density all
// dispatch generically without special-casing Dirac at the call sites.
const randDirac = {
  factory: function() {
    const args = Array.prototype.slice.call(arguments);
    if (args.length === 1 && args[0] && typeof args[0] === 'object'
        && ('prng' in args[0] || 'seed' in args[0])) {
      return function parametricDiracSampler(value: any) { return +value; };
    }
    const value = +args[0];
    return function staticDiracSampler() { return value; };
  },
};

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

// Synthetic Logistic and Weibull — no stdlib packages installed.
// Both have simple closed-form inverse-CDFs.
function uClip(u: any) {
  if (u <= 0)            return Number.EPSILON;
  if (u >= 1 - 1e-16)    return 1 - Number.EPSILON;
  return u;
}

// Standard-normal quantile Φ⁻¹ (spec §07 `probit`): Φ⁻¹(p) = −√2·erfc⁻¹(2p).
// This is the one normal-sampling primitive shared by the scalar randFn
// factories and the batched randNNormal kernel — driving both off the
// same Φ⁻¹ over the same Philox uniform stream is what makes the two
// paths bit-exact (the same role stdlibLn plays for Exponential). The
// uniform is uClip'd first: nextUniform ∈ [0, 1−2⁻³²] can return exactly
// 0, and Φ⁻¹(0) = −∞; uClip maps that to the smallest positive double so
// a draw stays finite. Both paths clip identically, so they still agree.
function _probit(u: number): number {
  return -Math.SQRT2 * stdlibErfcinv(2 * uClip(u));
}

// Inverse-CDF Normal / LogNormal sample factories — drop-in replacements
// for stdlib's ziggurat `random-base-normal` / `-lognormal`, matching the
// dual factory API the sampler relies on (sampler.ts):
//
//   factory(mu, sigma, { prng })  → () => draw          (rand / makeSampler /
//                                                         bulk-uniform adapter)
//   factory({ prng })             → (mu, sigma) => draw  (makeParametricSampler,
//                                                         per-draw params)
//
// One uniform per draw → x = μ + σ·Φ⁻¹(u); LogNormal exponentiates. The
// per-draw transform is byte-identical to randNNormal's vectorised form.
function _makeInvCdfNormalFactory(xform: (z: number, mu: number, sigma: number) => number) {
  return {
    factory: function () {
      const args = Array.prototype.slice.call(arguments);
      const prng = args[args.length - 1].prng;
      if (args.length === 1) {
        // Parametric form: params supplied at each draw.
        return (mu: number, sigma: number) => xform(_probit(prng()), mu, sigma);
      }
      // Baked-param form.
      const mu = args[0];
      const sigma = args[1];
      return () => xform(_probit(prng()), mu, sigma);
    },
  };
}

const randNormal    = _makeInvCdfNormalFactory((z, mu, sigma) => mu + sigma * z);
const randLogNormal = _makeInvCdfNormalFactory((z, mu, sigma) => Math.exp(mu + sigma * z));

// Logistic(mu, s): pdf = exp(-z)/(s·(1+exp(-z))²), Q(p) = μ + s·log(p/(1−p))
const randLogistic = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
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

// Weibull(shape, scale): pdf = (k/λ)·(x/λ)^{k−1}·exp(−(x/λ)^k) for x ≥ 0
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
  this.k = +shape; this.lambda = +scale;
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

// Pareto(shape α, scale θ): pdf = α·θ^α / x^(α+1) for x ≥ θ, else 0.
// Inverse-CDF sampler: x = θ·(1−u)^(−1/α). Matches Distributions.jl Pareto(α, θ).
const randPareto = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      return function parametricParetoSampler(alpha: any, scale: any) {
        const u = uClip(prng());
        return (+scale) * Math.pow(1 - u, -1 / (+alpha));
      };
    }
    const alpha = +args[0], scale = +args[1];
    return function staticParetoSampler() {
      const u = uClip(prng());
      return scale * Math.pow(1 - u, -1 / alpha);
    };
  },
};
function ParetoCtor(this: any, shape: any, scale: any) {
  this.alpha = +shape; this.xm = +scale;
  this.support = [+scale, Infinity];
}
ParetoCtor.prototype.pdf = function(this: any, x: any) {
  if (x < this.xm) return 0;
  return (this.alpha / x) * Math.pow(this.xm / x, this.alpha);
};
ParetoCtor.prototype.logpdf = function(this: any, x: any) {
  if (x < this.xm) return -Infinity;
  return Math.log(this.alpha) + this.alpha * Math.log(this.xm)
       - (this.alpha + 1) * Math.log(x);
};
ParetoCtor.prototype.cdf = function(this: any, x: any) {
  if (x <= this.xm) return 0;
  return 1 - Math.pow(this.xm / x, this.alpha);
};
ParetoCtor.prototype.quantile = function(this: any, p: any) {
  if (p <= 0) return this.xm;
  if (p >= 1) return Infinity;
  return this.xm * Math.pow(1 - p, -1 / this.alpha);
};
function logpdfPareto(x: any, alpha: any, scale: any) {
  if (x < scale) return -Infinity;
  return Math.log(alpha) + alpha * Math.log(scale) - (alpha + 1) * Math.log(x);
}

// GeneralizedNormal(mean, alpha, beta) — canonical scaled-gamma +
// Rademacher construction:  Y ~ Gamma(1/β, 1), R = ±1, X = μ + R·α·Y^(1/β)
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
        const y = inner(1 / beta, 1);
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

// InverseGamma(shape, scale): 1/Y where Y ~ Gamma(shape, rate=scale).
const randInverseGamma = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      const inner = randGamma.factory({ prng });
      return function parametricInverseGammaSampler(shape: any, scale: any) {
        const y = inner(+shape, +scale);
        return 1 / y;
      };
    }
    const shape = +args[0], scale = +args[1];
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

// ChiSquared(k) ≡ Gamma(k/2, rate=0.5).
const randChiSquared = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      const inner = randGamma.factory({ prng });
      return function parametricChiSquaredSampler(k: any) {
        return inner(k / 2, 0.5);
      };
    }
    const k = +args[0];
    const inner = randGamma.factory(k / 2, 0.5, { prng });
    return function staticChiSquaredSampler() { return inner(); };
  },
};

function ChiSquaredCtor(this: any, k: any) {
  this.k = +k;
  this.mean = +k;
  this.variance = 2 * (+k);
  this.stdev = Math.sqrt(this.variance);
  this.support = [0, Infinity];
}
ChiSquaredCtor.prototype.pdf = function(this: any, x: any) {
  return Math.exp(this.logpdf(x));
};
ChiSquaredCtor.prototype.logpdf = function(this: any, x: any) {
  if (x <= 0) return -Infinity;
  const k = this.k;
  return -(k / 2) * Math.LN2 - stdlibGammaln(k / 2)
       + (k / 2 - 1) * Math.log(x) - x / 2;
};
function logpdfChiSquared(x: any, k: any) {
  if (x <= 0) return -Infinity;
  return -(k / 2) * Math.LN2 - stdlibGammaln(k / 2)
       + (k / 2 - 1) * Math.log(x) - x / 2;
}

// VonMises(mu, kappa) — Best & Fisher (1979) rejection sampler.
// I_0(κ) computed via Chebyshev approximation (A&S 9.8.1/9.8.2).
function logBesselI0(x: any) {
  const ax = Math.abs(+x);
  if (ax < 3.75) {
    const t = (ax / 3.75); const t2 = t * t;
    const I0 = 1
      + t2 * (3.5156229
      + t2 * (3.0899424
      + t2 * (1.2067492
      + t2 * (0.2659732
      + t2 * (0.0360768
      + t2 * 0.0045813)))));
    return Math.log(I0);
  }
  const y = 3.75 / ax;
  const poly = 0.39894228
    + y * (0.01328592
    + y * (0.00225319
    + y * (-0.00157565
    + y * (0.00916281
    + y * (-0.02057706
    + y * (0.02635537
    + y * (-0.01647633
    + y * 0.00392377)))))));
  return ax - 0.5 * Math.log(ax) + Math.log(poly);
}

function _vmDraw(mu: any, kappa: any, prng: any) {
  const k = +kappa;
  const m = +mu;
  if (k < 1e-8) {
    return m + (prng() - 0.5) * 2 * Math.PI;
  }
  const a = 1 + Math.sqrt(1 + 4 * k * k);
  const b = (a - Math.sqrt(2 * a)) / (2 * k);
  const r = (1 + b * b) / (2 * b);
  for (let i = 0; i < 1000; i++) {
    const u1 = prng();
    const z = Math.cos(Math.PI * u1);
    const f = (1 + r * z) / (r + z);
    const c = k * (r - f);
    const u2 = prng();
    if (u2 < c * (2 - c) || u2 <= c * Math.exp(1 - c)) {
      const u3 = prng();
      const sign = (u3 - 0.5) < 0 ? -1 : 1;
      return m + sign * Math.acos(f);
    }
  }
  return m;
}

const randVonMises = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    if (args.length === 1 && args[0] === opts) {
      return function parametricVonMisesSampler(mu: any, kappa: any) {
        return _vmDraw(mu, kappa, prng);
      };
    }
    const mu = +args[0], kappa = +args[1];
    return function staticVonMisesSampler() { return _vmDraw(mu, kappa, prng); };
  },
};

function VonMisesCtor(this: any, mu: any, kappa: any) {
  this.mu = +mu;
  this.kappa = +kappa;
  this.mean = +mu;
  this.support = [+mu - Math.PI, +mu + Math.PI];
}
VonMisesCtor.prototype.pdf = function(this: any, x: any) {
  return Math.exp(this.logpdf(x));
};
VonMisesCtor.prototype.logpdf = function(this: any, x: any) {
  const k = this.kappa;
  return k * Math.cos(x - this.mu)
       - Math.log(2 * Math.PI) - logBesselI0(k);
};
function logpdfVonMises(x: any, mu: any, kappa: any) {
  return kappa * Math.cos(x - mu)
       - Math.log(2 * Math.PI) - logBesselI0(kappa);
}

// Laplace(location, scale): inverse-CDF sampler.
const randLaplace = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    function draw(mu: any, b: any) {
      const u = prng() - 0.5;
      const s = u < 0 ? -1 : 1;
      const a = 1 - 2 * Math.abs(u);
      const ac = a < Number.EPSILON ? Number.EPSILON : a;
      return (+mu) - (+b) * s * Math.log(ac);
    }
    if (args.length === 1 && args[0] === opts) {
      return function parametricLaplaceSampler(mu: any, b: any) { return draw(mu, b); };
    }
    const mu = +args[0], b = +args[1];
    return function staticLaplaceSampler() { return draw(mu, b); };
  },
};

function LaplaceCtor(this: any, location: any, scale: any) {
  this.mu = +location;
  this.b = +scale;
  this.mean = +location;
  this.variance = 2 * (+scale) * (+scale);
  this.stdev = Math.sqrt(this.variance);
  this.support = [-Infinity, Infinity];
}
LaplaceCtor.prototype.pdf = function(this: any, x: any) {
  return Math.exp(-Math.abs(x - this.mu) / this.b) / (2 * this.b);
};
LaplaceCtor.prototype.logpdf = function(this: any, x: any) {
  return -Math.log(2 * this.b) - Math.abs(x - this.mu) / this.b;
};
LaplaceCtor.prototype.cdf = function(this: any, x: any) {
  const z = (x - this.mu) / this.b;
  return z < 0 ? 0.5 * Math.exp(z) : 1 - 0.5 * Math.exp(-z);
};
LaplaceCtor.prototype.quantile = function(this: any, p: any) {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const u = p - 0.5;
  const s = u < 0 ? -1 : 1;
  return this.mu - this.b * s * Math.log(1 - 2 * Math.abs(u));
};
function logpdfLaplace(x: any, location: any, scale: any) {
  return -Math.log(2 * scale) - Math.abs(x - location) / scale;
}

// Geometric(p): number of failures until first success.
const randGeometric = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    function draw(p: any) {
      const pp = +p;
      if (pp <= 0) throw new Error('Geometric: p must be > 0');
      if (pp >= 1) return 0;
      const u = uClip(prng());
      return Math.floor(Math.log(1 - u) / Math.log(1 - pp));
    }
    if (args.length === 1 && args[0] === opts) {
      return function parametricGeometricSampler(p: any) { return draw(p); };
    }
    const p = +args[0];
    return function staticGeometricSampler() { return draw(p); };
  },
};

function GeometricCtor(this: any, p: any) {
  this.p = +p;
  this.mean = (1 - this.p) / this.p;
  this.variance = (1 - this.p) / (this.p * this.p);
  this.stdev = Math.sqrt(this.variance);
  this.support = [0, Infinity];
}
GeometricCtor.prototype.pmf = function(this: any, k: any) {
  const ki = k | 0;
  if (ki < 0 || ki !== +k) return 0;
  return this.p * Math.pow(1 - this.p, ki);
};
GeometricCtor.prototype.logpmf = function(this: any, k: any) {
  const ki = k | 0;
  if (ki < 0 || ki !== +k) return -Infinity;
  return Math.log(this.p) + ki * Math.log1p(-this.p);
};
GeometricCtor.prototype.pdf    = function(k: any) { return this.pmf(k); };
GeometricCtor.prototype.logpdf = function(this: any, k: any) { return this.logpmf(k); };
function logpdfGeometric(x: any, p: any) {
  const ki = x | 0;
  if (ki < 0 || ki !== +x) return -Infinity;
  return Math.log(p) + ki * Math.log1p(-p);
}

// NegativeBinomial(alpha, beta) and NegativeBinomial2(mu, psi) —
// Gamma-Poisson mixture sampling.
const randNegativeBinomial = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    const innerGamma   = randGamma.factory({ prng });
    const innerPoisson = randPoisson.factory({ prng });
    function draw(alpha: any, beta: any) {
      const lambda = innerGamma(+alpha, +beta);
      return innerPoisson(lambda);
    }
    if (args.length === 1 && args[0] === opts) {
      return function parametricNBSampler(alpha: any, beta: any) { return draw(alpha, beta); };
    }
    const alpha = +args[0], beta = +args[1];
    return function staticNBSampler() { return draw(alpha, beta); };
  },
};

function NegativeBinomialCtor(this: any, alpha: any, beta: any) {
  this.alpha = +alpha;
  this.beta = +beta;
  this.mean = this.alpha / this.beta;
  this.variance = this.alpha * (this.beta + 1) / (this.beta * this.beta);
  this.stdev = Math.sqrt(this.variance);
  this.support = [0, Infinity];
}
NegativeBinomialCtor.prototype.pmf = function(this: any, k: any) {
  return Math.exp(this.logpmf(k));
};
NegativeBinomialCtor.prototype.logpmf = function(this: any, k: any) {
  return _logpmfNegativeBinomial(k, this.alpha, this.beta);
};
NegativeBinomialCtor.prototype.pdf    = function(k: any) { return this.pmf(k); };
NegativeBinomialCtor.prototype.logpdf = function(this: any, k: any) { return this.logpmf(k); };
function _logpmfNegativeBinomial(k: any, alpha: any, beta: any) {
  const ki = k | 0;
  if (ki < 0 || ki !== +k) return -Infinity;
  // Generalized binomial coefficient C(k+alpha-1, k) via gammaln. alpha is posreals
  // (spec), so it may be non-integer; stdlibBinomcoefln returns NaN for non-integer
  // args. gammaln agrees with binomcoefln on integer alpha.
  return (stdlibGammaln(ki + alpha) - stdlibGammaln(alpha) - stdlibGammaln(ki + 1))
       + alpha * (Math.log(beta) - Math.log(beta + 1))
       - ki * Math.log(beta + 1);
}
function logpdfNegativeBinomial(x: any, alpha: any, beta: any) {
  return _logpmfNegativeBinomial(x, alpha, beta);
}

const randNegativeBinomial2 = {
  factory: function () {
    const args = Array.prototype.slice.call(arguments);
    const lastIdx = args.length - 1;
    const opts = (args.length > 0 && args[lastIdx]
                  && typeof args[lastIdx] === 'object'
                  && ('prng' in args[lastIdx])) ? args[lastIdx] : {};
    const prng = opts.prng || Math.random;
    const innerGamma   = randGamma.factory({ prng });
    const innerPoisson = randPoisson.factory({ prng });
    function draw(mu: any, psi: any) {
      const lambda = innerGamma(+psi, (+psi) / (+mu));
      return innerPoisson(lambda);
    }
    if (args.length === 1 && args[0] === opts) {
      return function parametricNB2Sampler(mu: any, psi: any) { return draw(mu, psi); };
    }
    const mu = +args[0], psi = +args[1];
    return function staticNB2Sampler() { return draw(mu, psi); };
  },
};

function NegativeBinomial2Ctor(this: any, mu: any, psi: any) {
  this.mu = +mu;
  this.psi = +psi;
  this.mean = +mu;
  this.variance = (+mu) + (+mu) * (+mu) / (+psi);
  this.stdev = Math.sqrt(this.variance);
  this.support = [0, Infinity];
}
NegativeBinomial2Ctor.prototype.pmf = function(this: any, k: any) {
  return Math.exp(this.logpmf(k));
};
NegativeBinomial2Ctor.prototype.logpmf = function(this: any, k: any) {
  return _logpmfNegativeBinomial2(k, this.mu, this.psi);
};
NegativeBinomial2Ctor.prototype.pdf    = function(k: any) { return this.pmf(k); };
NegativeBinomial2Ctor.prototype.logpdf = function(this: any, k: any) { return this.logpmf(k); };
function _logpmfNegativeBinomial2(k: any, mu: any, psi: any) {
  const ki = k | 0;
  if (ki < 0 || ki !== +k) return -Infinity;
  // Generalized binomial coefficient C(k+psi-1, k) via gammaln. psi is posreals
  // (spec), so it may be non-integer; stdlibBinomcoefln returns NaN for non-integer
  // args. gammaln agrees with binomcoefln on integer psi.
  return (stdlibGammaln(ki + psi) - stdlibGammaln(ki + 1) - stdlibGammaln(psi))
       + ki * (Math.log(mu) - Math.log(mu + psi))
       + psi * (Math.log(psi) - Math.log(mu + psi));
}
function logpdfNegativeBinomial2(x: any, mu: any, psi: any) {
  return _logpmfNegativeBinomial2(x, mu, psi);
}

// Categorical / Categorical0 — synthesised cumulative-sum samplers.
function _catSample(p: any, prng: any, offset: any) {
  const arr = valueLib.isValue(p) ? p.data : p;
  let u = prng();
  if (u <= 0) u = Number.EPSILON;
  if (u >= 1) u = 1 - Number.EPSILON;
  let cum = 0;
  for (let i = 0; i < arr.length; i++) {
    cum += arr[i];
    if (u <= cum) return i + offset;
  }
  return (arr.length - 1) + offset;
}

function _catLogpmf(k: any, p: any, offset: any) {
  const arr = valueLib.isValue(p) ? p.data : p;
  const idx = (k | 0) - offset;
  if (idx < 0 || idx >= arr.length) return -Infinity;
  const pi = arr[idx];
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

// =====================================================================
// REGISTRY
// =====================================================================

// Explicit batched samplers (commit 3, hybrid Option C). Each
// `randNFn(state, params, n, out?) → { state, out }` produces n iid
// draws in one bulk call — `rng.philoxN*` does the cipher amortization,
// and the per-atom transform is a vectorised loop over the pre-filled
// uniforms (no per-atom JS function dispatch). When a REGISTRY entry
// has a randNFn, the worker's static-params sampleN path picks it over
// the scalar `randFn` loop; absence falls back to the scalar path
// (optionally fed by `makeBulkUniformPrngAdapter` for cipher
// amortization alone).
//
// Normal (and LogNormal, which exponentiates Normal) draw one uniform per
// sample and map it through Φ⁻¹ (`_probit`), so uniform `i` produces
// normal `i` with no pairing or reshuffling — the same one-uniform-per-draw
// ordering as Uniform / Exponential, which is what keeps the batched and
// scalar paths bit-exact.

// Per-element parameter read for the randNFn transforms. A param column
// is either a plain number (the STATIC case — one value shared by every
// atom, the degenerate stride-0 broadcast) or a Float64Array of length n
// (the PER-I case — one value per atom, used by the materialiser's
// per-binding sampling when a distribution parameter references an
// upstream draw). Both callers run the SAME transform loop body, so there
// is exactly one leaf-math realisation per distribution — not a static
// kernel plus a separate per-i kernel (engine-concepts §11, Q2 fold).
//
// Bit-exactness corollary: with a scalar param, `_p(c, i)` returns the
// identical value on every iteration, so the loop is arithmetically (and
// byte-for-byte) identical to the old hoisted-scalar form. The static
// Uniform/Exponential bit-exact gate (sampler-batched.test.ts) pins this.
function _p(c: any, i: number): number {
  // A param COLUMN is a typed array (Float64Array) — read it per atom.
  // Anything else is a SCALAR: coerce with unary `+`, matching the
  // pre-fold `+params[k]` exactly. So a numeric scalar is byte-for-byte
  // identical to the old hoisted-scalar form, and a non-numeric scalar
  // (a boolean/string from a malformed param) coerces the WHOLE value
  // rather than mis-indexing it character-by-character. Keying on
  // ArrayBuffer.isView (not `typeof === 'number'`) keeps the static and
  // per-i paths agreeing on every input — one realisation, not two.
  return ArrayBuffer.isView(c) ? (c as any)[i] : +c;
}

function randNNormal(state: any, params: any, n: number, out?: Float64Array) {
  // params = [mu, sigma]; each is a scalar (static) or a Float64Array(n)
  // column (per-i). Inverse-CDF transport: x = mu + sigma·Φ⁻¹(u), one
  // uniform per draw. Bit-exact equivalent to the scalar randFn loop under
  // the same Philox state because (a) philoxNUniform is bit-exact
  // equivalent to n scalar nextUniform calls (same uniform stream + order),
  // (b) per-i changes only the affine params, and (c) both paths apply the
  // SAME `_probit` primitive (`uClip` + stdlib erfcinv) — exactly the
  // discipline that makes Exponential bit-exact via the shared `ln`.
  const mu = params[0];
  const sigma = params[1];
  const dest = out ?? new Float64Array(n);
  const r = rng.philoxNUniform(state, n, dest);
  for (let i = 0; i < n; i++) r.out[i] = _p(mu, i) + _p(sigma, i) * _probit(r.out[i]);
  return r;
}

function randNExponential(state: any, params: any, n: number, out?: Float64Array) {
  // params = [rate (= lambda)]; scalar (static) or Float64Array(n) (per-i).
  // Transform per stdlib: x = -ln(1 - u) / lambda. One uniform per draw.
  // Bit-exact equivalent to scalar randExponential under the same Philox
  // state because (a) philoxNUniform is bit-exact equivalent to n scalar
  // nextUniform calls (uniform stream + order identical), (b) per-i
  // changes only the divisor, and (c) we use @stdlib's `ln` here — the
  // SAME primitive scalar randExponential uses — not native Math.log.
  // Math.log differs from @stdlib ln in the last ULP on ~1.6% of inputs,
  // which broke the bit-exact gate before this used stdlibLn.
  const lambda = params[0];
  const dest = out ?? new Float64Array(n);
  const r = rng.philoxNUniform(state, n, dest);
  for (let i = 0; i < n; i++) r.out[i] = -stdlibLn(1 - r.out[i]) / _p(lambda, i);
  return r;
}

function randNUniform(state: any, params: any, n: number, out?: Float64Array) {
  // params = [a, b]  (lo, hi); each scalar (static) or Float64Array(n)
  // (per-i). One uniform per draw.
  // Transform per stdlib uniform.js: x = b*u + (1-u)*a — numerically
  // stable as written (vs. naive (b-a)*u + a) and bit-exact equivalent
  // to the scalar-loop path. The a/b reads live INSIDE the loop now (via
  // _p) so the per-i column case works; for scalar params each iteration
  // reads the same value, so the bytes are unchanged.
  const a = params[0];
  const b = params[1];
  const dest = out ?? new Float64Array(n);
  const r = rng.philoxNUniform(state, n, dest);
  for (let i = 0; i < n; i++) {
    const u = r.out[i];
    r.out[i] = _p(b, i) * u + (1 - u) * _p(a, i);
  }
  return r;
}

function randNLogNormal(state: any, params: any, n: number, out?: Float64Array) {
  // params = [mu, sigma]; x = exp(mu + sigma * z), z = Φ⁻¹(u) ~ Normal(0,1).
  // Reuses the inverse-CDF Normal kernel then exp-in-place.
  const r = randNNormal(state, params, n, out);
  for (let i = 0; i < n; i++) r.out[i] = Math.exp(r.out[i]);
  return r;
}

// ContinuedPoisson is density-only (spec §09): the continuous extension of
// Poisson to the reals is not a probability measure and has no generator, so
// every sampling entry point throws rather than returning a bogus draw.
function _continuedPoissonNonGenerative(): never {
  throw new Error('ContinuedPoisson is density-only (spec §09): the continuous '
    + 'Poisson extension is not a probability measure and has no sampler — '
    + 'rand/draw(ContinuedPoisson) is undefined.');
}

// =====================================================================
// particle-physics distribution densities (spec §09). DENSITY-ONLY for now
// (logpdf landed; samplers deferred) — all are absent from
// SAMPLEABLE_DISTRIBUTIONS and carry `densityOnly: true`, so a draw throws.
// Each `logpdf` is the spec's normalized density; normalizations are
// closed-form (verified against scipy / numerical quadrature to ≲1e-12).
// =====================================================================

const _SQRT_HALF_PI = Math.sqrt(Math.PI / 2);  // ∫ e^{-z²/2} half-Gaussian scale
const _SQRT_2 = Math.SQRT2;
const _erf = (x: number): number => 1 - stdlibErfc(x);

function _hepDensityOnly(name: string) {
  return function (): never {
    throw new Error(`${name}: density-only in flatppl-js (spec §09) — the `
      + `logpdf is implemented but the sampler is not yet; draw/rand is unavailable.`);
  };
}

// One-sided Crystal Ball log-density (left power-law tail). Normalization
// N = σ·(C + D): C the tail integral (needs n > 1), D the Gaussian-core
// integral. Matches scipy.stats.crystalball to ~1e-12.
function _crystalBallLogpdf(x: number, m0: number, sigma: number, alpha: number, n: number): number {
  const aAbs = Math.abs(alpha);
  const A = Math.pow(n / aAbs, n) * Math.exp(-aAbs * aAbs / 2);
  const B = n / aAbs - aAbs;
  const C = (n / aAbs) * (1 / (n - 1)) * Math.exp(-aAbs * aAbs / 2);
  const D = _SQRT_HALF_PI * (1 + _erf(aAbs / _SQRT_2));
  const logN = Math.log(sigma * (C + D));
  const z = (x - m0) / sigma;
  const core = z < -aAbs
    ? Math.log(A) - n * Math.log(B - z)
    : -0.5 * z * z;
  return core - logN;
}

// Double-sided Crystal Ball: independent power-law tails on both sides,
// each with its own σ. Normalization sums the two tail integrals and the
// two half-Gaussian core integrals.
function _doubleSidedCrystalBallLogpdf(
  x: number, m0: number, sigmaL: number, sigmaR: number,
  alphaL: number, alphaR: number, nL: number, nR: number,
): number {
  const aL = Math.abs(alphaL), aR = Math.abs(alphaR);
  const AL = Math.pow(nL / aL, nL) * Math.exp(-aL * aL / 2), BL = nL / aL - aL;
  const AR = Math.pow(nR / aR, nR) * Math.exp(-aR * aR / 2), BR = nR / aR - aR;
  const tailL = sigmaL * (nL / aL) * (1 / (nL - 1)) * Math.exp(-aL * aL / 2);
  const coreL = sigmaL * _SQRT_HALF_PI * _erf(aL / _SQRT_2);
  const coreR = sigmaR * _SQRT_HALF_PI * _erf(aR / _SQRT_2);
  const tailR = sigmaR * (nR / aR) * (1 / (nR - 1)) * Math.exp(-aR * aR / 2);
  const logN = Math.log(tailL + coreL + coreR + tailR);
  const zL = (x - m0) / sigmaL, zR = (x - m0) / sigmaR;
  let core: number;
  if (zL < -aL) core = Math.log(AL) - nL * Math.log(BL - zL);
  else if (zL <= 0) core = -0.5 * zL * zL;
  else if (zR <= aR) core = -0.5 * zR * zR;
  else core = Math.log(AR) - nR * Math.log(BR + zR);
  return core - logN;
}

// ARGUS log-density on (0, m0). Normalization (slope c < 0): with a = −c,
// M = (m0²/2)·a^{−(p+1)}·γ(p+1, a), γ the lower incomplete gamma =
// P(p+1, a)·Γ(p+1). Outside (0, m0) the density is 0 → −∞.
function _argusLogpdf(x: number, resonance: number, slope: number, power: number): number {
  if (x <= 0 || x >= resonance) return -Infinity;
  const m0 = resonance, c = slope, p = power;
  const a = -c;
  const gLower = stdlibGammainc(a, p + 1) * stdlibGamma(p + 1);  // γ(p+1, a)
  const logM = Math.log(m0 * m0 / 2) - (p + 1) * Math.log(a) + Math.log(gLower);
  const u = 1 - (x / m0) * (x / m0);
  return Math.log(x) + p * Math.log(u) + c * u - logM;
}

// Relativistic Breit–Wigner log-density on x > 0. Normalization M is
// closed-form (spec §09): M = π√(m²+γ)/(2√2·m·Γ·γ), γ = √(m²(m²+Γ²)).
function _relativisticBreitWignerLogpdf(x: number, mean: number, width: number): number {
  if (x <= 0) return -Infinity;
  const m = mean, G = width;
  const gamma = Math.sqrt(m * m * (m * m + G * G));
  const logM = Math.log(Math.PI * Math.sqrt(m * m + gamma) / (2 * _SQRT_2 * m * G * gamma));
  return -Math.log((x * x - m * m) ** 2 + m * m * G * G) - logM;
}

// Split (bifurcated) normal log-density — already normalized.
function _bifurcatedNormalLogpdf(x: number, mean: number, sigmaL: number, sigmaR: number): number {
  const s = x < mean ? sigmaL : sigmaR;
  return 0.5 * Math.log(2 / Math.PI) - Math.log(sigmaL + sigmaR)
    - (x - mean) * (x - mean) / (2 * s * s);
}

// Landau density φ((x−x0)/xi)/xi. CERN G110 DENLAN rational approximation
// (Kölbig & Schorr), the algorithm behind ROOT's TMath::Landau — matched to it
// to machine epsilon (≤5e-16) on the conformance points. φ has no elementary
// closed form (it is an inverse-Laplace integral); the standard density (xi=1,
// x0=0) peaks at φ(−0.22278) and equals 0.1788541609 at 0.
const _LANDAU_P1 = [0.4259894875, -0.1249762550, 0.03984243700, -0.006298287635, 0.001511162253];
const _LANDAU_Q1 = [1.0, -0.3388260629, 0.09594393323, -0.01608042283, 0.003778942063];
const _LANDAU_P2 = [0.1788541609, 0.1173957403, 0.01488850518, -0.001394989411, 0.0001283617211];
const _LANDAU_Q2 = [1.0, 0.7428795082, 0.3153932961, 0.06694219548, 0.008790609714];
const _LANDAU_P3 = [0.1788544503, 0.09359161662, 0.006325387654, 0.00006611667319, -0.000002031049101];
const _LANDAU_Q3 = [1.0, 0.6097809921, 0.2560616665, 0.04746722384, 0.006957301675];
const _LANDAU_P4 = [0.9874054407, 118.6723273, 849.2794360, -743.7792444, 427.0262186];
const _LANDAU_Q4 = [1.0, 106.8615961, 337.6496214, 2016.712389, 1597.063511];
const _LANDAU_P5 = [1.003675074, 167.5702434, 4789.711289, 21217.86767, -22324.94910];
const _LANDAU_Q5 = [1.0, 156.9424537, 3745.310488, 9834.698876, 66924.28357];
const _LANDAU_P6 = [1.000827619, 664.9143136, 62972.92665, 475554.6998, -5743609.109];
const _LANDAU_Q6 = [1.0, 651.4101098, 56974.73333, 165917.4725, -2815759.939];
const _LANDAU_A1 = [0.04166666667, -0.01996527778, 0.02709538966];
const _LANDAU_A2 = [-1.845568670, -4.284640743];

// Horner evaluation of degree-4 ratio p(t)/q(t) with the DENLAN coefficient
// layout (constant term first).
function _landauRatio(p: number[], q: number[], t: number): number {
  const num = p[0] + (p[1] + (p[2] + (p[3] + p[4] * t) * t) * t) * t;
  const den = q[0] + (q[1] + (q[2] + (q[3] + q[4] * t) * t) * t) * t;
  return num / den;
}

function _landauPdf(x: number, xi: number, x0: number): number {
  if (xi <= 0) return 0;
  const v = (x - x0) / xi;
  let den: number;
  if (v < -5.5) {
    const u = Math.exp(v + 1.0);
    if (u < 1e-10) return 0;
    den = 0.3989422803 * (Math.exp(-1 / u) / Math.sqrt(u))
      * (1 + (_LANDAU_A1[0] + (_LANDAU_A1[1] + _LANDAU_A1[2] * u) * u) * u);
  } else if (v < -1) {
    const u = Math.exp(-v - 1);
    den = Math.exp(-u) * Math.sqrt(u) * _landauRatio(_LANDAU_P1, _LANDAU_Q1, v);
  } else if (v < 1) {
    den = _landauRatio(_LANDAU_P2, _LANDAU_Q2, v);
  } else if (v < 5) {
    den = _landauRatio(_LANDAU_P3, _LANDAU_Q3, v);
  } else if (v < 12) {
    const u = 1 / v;
    den = u * u * _landauRatio(_LANDAU_P4, _LANDAU_Q4, u);
  } else if (v < 50) {
    const u = 1 / v;
    den = u * u * _landauRatio(_LANDAU_P5, _LANDAU_Q5, u);
  } else if (v < 300) {
    const u = 1 / v;
    den = u * u * _landauRatio(_LANDAU_P6, _LANDAU_Q6, u);
  } else {
    const u = 1 / (v - v * Math.log(v) / (v + 1));
    den = u * u * (1 + (_LANDAU_A2[0] + _LANDAU_A2[1] * u) * u);
  }
  return den / xi;
}

// Landau log-density: location-scale family over the standard Landau density,
// (1/s)φ((x−ℓ)/s). `loc`/`scale` are the location/scale arguments (the true mean
// and variance are undefined — heavy 1/x² tail), so `loc` is named for what it
// is, not `mean`. Spec §09.
function _landauLogpdf(x: number, loc: number, scale: number): number {
  const den = _landauPdf(x, scale, loc);
  return den > 0 ? Math.log(den) : -Infinity;
}

// Faddeeva function w(z) = e^{−z²}·erfc(−iz), real part only, for z in the
// upper half-plane (Im z > 0 — always true for a Voigt argument with Γ,σ > 0).
// Weideman (1994) rational approximation; coefficients precomputed for N=32
// (max |Re error| vs scipy.wofz ≈ 2.4e-14 over the relevant region).
const _FADDEEVA_L = 4.756828460010884;
const _FADDEEVA_A = [
  -1.3031797863050087e-12, 3.741077554722698e-12, 8.03033538228436e-12,
  -2.154353731437607e-11, -5.5442356114348654e-11, 1.1658254682023323e-10,
  4.153742716146919e-10, -5.231021437249246e-10, -3.208015140788758e-09,
  8.124888965682105e-10, 2.3797556759384297e-08, 2.2930439119439287e-08,
  -1.4813078912439695e-07, -4.1840763705308403e-07, 4.255833137922926e-07,
  4.401531731590925e-06, 6.821031944024978e-06, -2.1409619201869843e-05,
  -0.00013075449254623973, -0.00024532980270020207, 0.00039259136070070517,
  0.004519541105349235, 0.019006155784845408, 0.05730440352983719,
  0.1406071622689377, 0.29544451071508726, 0.5460139720639342,
  0.9019254893648, 1.345544169234545, 1.8256696296324815,
  2.2635372999002676, 2.5722534081245696,
];

function _faddeevaWRe(zr: number, zi: number): number {
  const L = _FADDEEVA_L;
  // L + i·z  and  L − i·z  (with i·z = −zi + i·zr).
  const numR = L - zi, numI = zr;     // L + i z
  const denR = L + zi, denI = -zr;    // L − i z
  const denMag = denR * denR + denI * denI;
  // Z = (L + i z)/(L − i z)
  const Zr = (numR * denR + numI * denI) / denMag;
  const Zi = (numI * denR - numR * denI) / denMag;
  // p = polyval(A, Z) — A[0] highest degree, Horner.
  let pr = _FADDEEVA_A[0], pi = 0;
  for (let k = 1; k < _FADDEEVA_A.length; k++) {
    const nr = pr * Zr - pi * Zi + _FADDEEVA_A[k];
    const ni = pr * Zi + pi * Zr;
    pr = nr; pi = ni;
  }
  // (L − i z)²
  const sqR = denR * denR - denI * denI, sqI = 2 * denR * denI;
  const sqMag = sqR * sqR + sqI * sqI;
  // 2p/(L − i z)²  (real part)
  const t1r = (2 * pr * sqR + 2 * pi * sqI) / sqMag;
  // (1/√π)/(L − i z)  (real part)
  const c = 1 / Math.sqrt(Math.PI);
  const t2r = c * denR / denMag;
  return t1r + t2r;
}

// Voigt profile log-density: Re(w((x−μ+iΓ/2)/(σ√2)))/(σ√(2π)). Already
// normalized (the Faddeeva form is the pdf). Matches scipy.special.voigt_profile.
function _voigtianLogpdf(x: number, mean: number, width: number, sigma: number): number {
  const denom = sigma * _SQRT_2;
  const zr = (x - mean) / denom;
  const zi = (width / 2) / denom;
  const val = _faddeevaWRe(zr, zi) / (sigma * Math.sqrt(2 * Math.PI));
  return Math.log(val);
}

const REGISTRY = {
  Normal: {
    params:   ['mu', 'sigma'],
    aliases:  {},
    discrete: false,
    Ctor:     Normal,
    randFn:   randNormal,
    // Explicit batched path: x = mu + sigma·Φ⁻¹(u), one uniform per draw.
    // Bit-exact equivalent to the scalar randFn under the same initial
    // Philox state — both consume the same uniform stream (philoxNUniform
    // contract) and apply the same `_probit`, exactly as Exponential does
    // with its shared `ln`.
    randNFn:  randNNormal,
    logpdfFn: logpdfNormal,
  },
  Exponential: {
    params:   ['rate'],
    aliases:  {},
    discrete: false,
    Ctor:     Exponential,
    randFn:   randExponential,
    // Explicit batched path: -ln(1-u)/lambda, one uniform per draw.
    // Bit-exact equivalent to scalar randExponential under the same
    // initial Philox state (philoxNUniform contract guarantees this).
    randNFn:  randNExponential,
    logpdfFn: logpdfExponential,
  },
  Uniform: {
    params:   ['support'],
    aliases:  {},
    discrete: false,
    Ctor:     Uniform,
    randFn:   randUniform,
    // Explicit batched path: b*u + (1-u)*a (stdlib's stable form).
    // Bit-exact equivalent to scalar randUniform.
    randNFn:  randNUniform,
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
    params:   ['mu', 's'],
    aliases:  {},
    discrete: false,
    Ctor:     LogisticCtor,
    randFn:   randLogistic,
    logpdfFn: logpdfLogistic,
  },
  Weibull: {
    params:   ['shape', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     WeibullCtor,
    randFn:   randWeibull,
    logpdfFn: logpdfWeibull,
  },
  Pareto: {
    params:   ['shape', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     ParetoCtor,
    randFn:   randPareto,
    logpdfFn: logpdfPareto,
  },
  LogNormal: {
    params:   ['mu', 'sigma'],
    aliases:  {},
    discrete: false,
    Ctor:     LogNormal,
    randFn:   randLogNormal,
    // Explicit batched path: exp(mu + sigma·Φ⁻¹(u)). Bit-exact equivalent
    // to the scalar randFn — inherits Normal's shared-uniform, shared-Φ⁻¹
    // guarantee, then exponentiates in place.
    randNFn:  randNLogNormal,
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
    params:   ['shape', 'rate'],
    aliases:  {},
    discrete: false,
    Ctor:     Gamma,
    randFn:   randGamma,
    logpdfFn: logpdfGamma,
  },
  Cauchy: {
    params:   ['location', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     Cauchy,
    randFn:   randCauchy,
    logpdfFn: logpdfCauchy,
  },
  StudentT: {
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
    params:   ['mean', 'alpha', 'beta'],
    aliases:  {},
    discrete: false,
    Ctor:     GeneralizedNormalCtor,
    randFn:   randGeneralizedNormal,
    logpdfFn: logpdfGeneralizedNormal,
  },
  InverseGamma: {
    params:   ['shape', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     InverseGammaCtor,
    randFn:   randInverseGamma,
    logpdfFn: logpdfInverseGamma,
  },
  ChiSquared: {
    params:   ['k'],
    aliases:  {},
    discrete: false,
    Ctor:     ChiSquaredCtor,
    randFn:   randChiSquared,
    logpdfFn: logpdfChiSquared,
  },
  VonMises: {
    params:   ['mu', 'kappa'],
    aliases:  {},
    discrete: false,
    Ctor:     VonMisesCtor,
    randFn:   randVonMises,
    logpdfFn: logpdfVonMises,
  },
  Laplace: {
    params:   ['location', 'scale'],
    aliases:  {},
    discrete: false,
    Ctor:     LaplaceCtor,
    randFn:   randLaplace,
    logpdfFn: logpdfLaplace,
  },
  Geometric: {
    params:   ['p'],
    aliases:  {},
    discrete: true,
    Ctor:     GeometricCtor,
    randFn:   randGeometric,
    logpdfFn: logpdfGeometric,
  },
  NegativeBinomial: {
    params:   ['alpha', 'beta'],
    aliases:  {},
    discrete: true,
    Ctor:     NegativeBinomialCtor,
    randFn:   randNegativeBinomial,
    logpdfFn: logpdfNegativeBinomial,
  },
  NegativeBinomial2: {
    params:   ['mu', 'psi'],
    aliases:  {},
    discrete: true,
    Ctor:     NegativeBinomial2Ctor,
    randFn:   randNegativeBinomial2,
    logpdfFn: logpdfNegativeBinomial2,
  },
  Categorical: {
    params:   ['p'],
    aliases:  {},
    discrete: true,
    Ctor:     CategoricalCtor,
    randFn:   randCategorical,
    logpdfFn: logpdfCategorical,
  },
  Categorical0: {
    params:   ['p'],
    aliases:  {},
    discrete: true,
    Ctor:     Categorical0Ctor,
    randFn:   randCategorical0,
    logpdfFn: logpdfCategorical0,
  },
  Poisson: {
    params:   ['rate'],
    aliases:  {},
    discrete: true,
    Ctor:     Poisson,
    randFn:   randPoisson,
    logpdfFn: logpmfPoisson,
  },
  ContinuedPoisson: {
    params:   ['rate'],
    aliases:  {},
    discrete: false,
    // Density-only: no sampler, so exempt from the REGISTRY ⊆
    // SAMPLEABLE_DISTRIBUTIONS invariant (it is deliberately absent there).
    densityOnly: true,
    // Density-only (spec §09): logpdf(x; rate) = x·ln(rate) − rate − ln Γ(x+1),
    // the continuous extension that agrees with Poisson's logpmf on integer x.
    // Not normalized, not generative — the sampler stubs throw (see above).
    Ctor:     _continuedPoissonNonGenerative,
    randFn:   { factory: _continuedPoissonNonGenerative },
    logpdfFn: (x: number, rate: number) =>
      x * Math.log(rate) - rate - stdlibGammaln(x + 1),
  },
  CrystalBall: {
    params:   ['m0', 'sigma', 'alpha', 'n'],
    aliases:  {},
    discrete: false,
    densityOnly: true,
    Ctor:     _hepDensityOnly('CrystalBall'),
    randFn:   { factory: _hepDensityOnly('CrystalBall') },
    logpdfFn: _crystalBallLogpdf,
  },
  DoubleSidedCrystalBall: {
    params:   ['m0', 'sigmaL', 'sigmaR', 'alphaL', 'alphaR', 'nL', 'nR'],
    aliases:  {},
    discrete: false,
    densityOnly: true,
    Ctor:     _hepDensityOnly('DoubleSidedCrystalBall'),
    randFn:   { factory: _hepDensityOnly('DoubleSidedCrystalBall') },
    logpdfFn: _doubleSidedCrystalBallLogpdf,
  },
  Argus: {
    params:   ['resonance', 'slope', 'power'],
    aliases:  {},
    discrete: false,
    densityOnly: true,
    Ctor:     _hepDensityOnly('Argus'),
    randFn:   { factory: _hepDensityOnly('Argus') },
    logpdfFn: _argusLogpdf,
  },
  RelativisticBreitWigner: {
    params:   ['mean', 'width'],
    aliases:  {},
    discrete: false,
    densityOnly: true,
    Ctor:     _hepDensityOnly('RelativisticBreitWigner'),
    randFn:   { factory: _hepDensityOnly('RelativisticBreitWigner') },
    logpdfFn: _relativisticBreitWignerLogpdf,
  },
  Voigtian: {
    params:   ['mean', 'width', 'sigma'],
    aliases:  {},
    discrete: false,
    densityOnly: true,
    Ctor:     _hepDensityOnly('Voigtian'),
    randFn:   { factory: _hepDensityOnly('Voigtian') },
    logpdfFn: _voigtianLogpdf,
  },
  Landau: {
    params:   ['loc', 'scale'],
    aliases:  {},
    discrete: false,
    densityOnly: true,
    Ctor:     _hepDensityOnly('Landau'),
    randFn:   { factory: _hepDensityOnly('Landau') },
    logpdfFn: _landauLogpdf,
  },
  BifurcatedNormal: {
    params:   ['mean', 'sigmaL', 'sigmaR'],
    aliases:  {},
    discrete: false,
    densityOnly: true,
    Ctor:     _hepDensityOnly('BifurcatedNormal'),
    randFn:   { factory: _hepDensityOnly('BifurcatedNormal') },
    logpdfFn: _bifurcatedNormalLogpdf,
  },
  Dirac: {
    params:   ['value'],
    aliases:  {},
    discrete: false,
    Ctor:     DiracCtor,
    randFn:   randDirac,
    logpdfFn: logpdfDirac,
  },
};

// =====================================================================
// Dispatch helpers
// =====================================================================

function isKnownDistribution(name: any) {
  return Object.prototype.hasOwnProperty.call(REGISTRY, name);
}

function listDistributions() {
  return Object.keys(REGISTRY);
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

// resolveParams + regionBoundsFromIR both need the deterministic
// value evaluator. The evaluator lives in sampler.ts (which also
// holds ARITH_OPS, evaluateCall, complex helpers, linalg, ...). The
// dependency is one-way at module load (sampler-registry has no
// top-level evaluator usage) and resolved per-call via lazy require
// below — same pattern used elsewhere in the engine (materialiser
// lazy-requires sampler in mat-multivariate's worker dispatch).

function resolveParams(measureIR: any, entry: any, env: any) {
  if (typeof entry.customResolveParams === 'function') {
    return entry.customResolveParams(measureIR, env);
  }
  const evaluateExpr = require('./sampler.ts').evaluateExpr;
  let kwargs = measureIR.kwargs || {};
  let positional = measureIR.args || [];

  // Auto-splatting (spec §04 "Calling conventions"): a single positional
  // record argument is equivalent to passing each field as a keyword arg
  // — `Dist(record(a = x, b = y))` ≡ `Dist(a = x, b = y)`. Fires only
  // when there are no explicit kwargs and the lone positional arg is a
  // record whose field names cover every parameter (or its alias). A
  // field/parameter-name mismatch is left untouched so the per-param
  // loop below raises the existing "missing parameter" — matching the
  // spec's "names that do not match the callable's argument names is a
  // static error". Splatting is shallow.
  if (Object.keys(kwargs).length === 0 && positional.length === 1) {
    const a0 = positional[0];
    if (a0 && a0.kind === 'call' && a0.op === 'record'
        && Array.isArray(a0.fields)) {
      const byName: Record<string, any> = Object.create(null);
      for (const f of a0.fields) byName[f.name] = f.value;
      let covers = true;
      for (const p of entry.params) {
        if (p in byName) continue;
        if (entry.aliases[p] && entry.aliases[p] in byName) continue;
        covers = false; break;
      }
      if (covers) { kwargs = byName; positional = []; }
    }
  }

  const out: any[] = [];
  for (let i = 0; i < entry.params.length; i++) {
    const paramName = entry.params[i];
    let exprIR;
    if (paramName in kwargs) {
      exprIR = kwargs[paramName];
    } else if (entry.aliases[paramName] && entry.aliases[paramName] in kwargs) {
      exprIR = kwargs[entry.aliases[paramName]];
    } else if (i < positional.length) {
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

function regionBoundsFromIR(ir: any, env: any) {
  if (!ir) throw new Error('regionBoundsFromIR: missing IR');
  const evaluateExpr = require('./sampler.ts').evaluateExpr;
  // Recognised region shapes live in transforms.regionBounds (shared with the
  // MCMC constraining transform); endpoints evaluate in the sampler's env.
  const bounds = transforms.regionBounds(ir, (a: any) => evaluateExpr(a, env));
  if (!bounds) {
    throw new Error('regionBoundsFromIR: unsupported region shape (kind='
      + ir.kind + (ir.op ? ', op=' + ir.op : '') + ')');
  }
  return bounds;
}

// True iff `measureIR` is a registry distribution that has an explicit
// batched kernel (randNFn). The worker's per-i materialisation path uses
// this to dispatch: randNFn dists fold onto the single batched leaf math
// (makeBulkSampler perI mode); everything else keeps the per-draw scalar
// sampler. Never throws — a non-call / unknown op is simply not a
// randNFn dist.
function hasRandNFn(measureIR: any) {
  if (!measureIR || measureIR.kind !== 'call') return false;
  const entry = (REGISTRY as any)[measureIR.op];
  return !!entry && typeof entry.randNFn === 'function';
}

// Normalise one evaluateExprN result into a randNFn param column:
//   number / boolean       → a scalar (atom-independent stride-0 read)
//   Float64Array(count)    → a per-atom column
//   Value shape=[count]    → its .data (atom-batched scalar)
//   anything else          → null (signal: cannot batch-resolve, the
//                            caller should fall back to the scalar path)
//
// A vector-atom result (Value shape=[count, k]) is NOT a hard error: it
// falls back like everything else. The per-draw scalar path (the canonical
// handler — e.g. a nested-chain param that batch-evaluates to a per-atom
// vector but resolves to a scalar per atom) takes over, and if a parameter
// is genuinely a vector into a scalar distribution it fails loud THERE, at
// the same site it always has. Falling back is never silently wrong;
// throwing on this hot dispatch would regress paths the scalar sampler
// handles correctly.
function _normalizeParamColumn(result: any, count: number, op: string): any {
  if (typeof result === 'number') return result;
  if (typeof result === 'boolean') return +result;
  if (result && result.BYTES_PER_ELEMENT !== undefined && result.length === count) {
    return result;   // Float64Array(count) — per-atom scalar column
  }
  if (result && Array.isArray(result.shape) && result.data
      && result.data.BYTES_PER_ELEMENT !== undefined
      && result.shape.length === 1 && result.shape[0] === count
      && result.data.length === count) {
    return result.data;   // atom-batched scalar
  }
  return null;   // scalar-incompatible (vector-atom / unexpected) — fall back
}

function _columnAllFinite(col: any): boolean {
  if (typeof col === 'number') return Number.isFinite(col);
  for (let i = 0; i < col.length; i++) if (!Number.isFinite(col[i])) return false;
  return true;
}

// Per-i parameter resolution — the per-atom analogue of resolveParams.
// Each declared parameter expression is evaluated OVER THE BATCH via
// evaluateExprN, yielding a column that is either a plain number (the
// parameter is atom-independent, read as a stride-0 broadcast) or a
// Float64Array(count) (the parameter varies per atom, e.g. it references
// an upstream draw). Single-sources the kwargs>aliases>positional
// precedence and the Uniform custom-support handling with resolveParams,
// so the per-i caller never re-derives them.
//
// Returns null — a "cannot batch-resolve, fall back" SIGNAL, not an error
// — when a parameter can't be resolved as a scalar/column: a Uniform
// whose support is not interval(lo, hi), a non-finite Uniform bound, or
// any unexpected result shape. The caller (makeBulkSampler perI mode)
// turns null into { unhandled: true } and the worker keeps its per-draw
// scalar fallback (which reproduces the canonical error path for genuinely
// invalid params, and correctly handles cases like nested-chain params
// that batch-evaluate to a vector-atom but resolve to a scalar per atom).
// resolveParamsN never throws for a scalar-incompatible column — it falls
// back, so it can't regress a path the scalar sampler handles.
function resolveParamsN(
  measureIR: any, entry: any, refArrays: any, count: number, baseEnv: any,
) {
  const evaluateExprN = require('./sampler.ts').evaluateExprN;
  const kwargs = measureIR.kwargs || {};
  const positional = measureIR.args || [];

  // Custom-resolve dists (Uniform): only the interval(lo, hi) support
  // shape can be batch-resolved; const-region / ref-region supports fall
  // back. Mirrors customResolveParams' kwargs.support|positional[0] pick.
  if (typeof entry.customResolveParams === 'function') {
    const supportIR = ('support' in kwargs) ? kwargs.support
                    : (positional.length > 0 ? positional[0] : null);
    if (!supportIR || supportIR.kind !== 'call' || supportIR.op !== 'interval'
        || !Array.isArray(supportIR.args) || supportIR.args.length !== 2) {
      return null;   // non-interval support — fall back to the scalar path
    }
    const lo = _normalizeParamColumn(
      evaluateExprN(supportIR.args[0], refArrays, count, baseEnv), count, measureIR.op);
    const hi = _normalizeParamColumn(
      evaluateExprN(supportIR.args[1], refArrays, count, baseEnv), count, measureIR.op);
    if (lo === null || hi === null) return null;
    // Bounded support required (matches customResolveParams' finiteness
    // guard); a non-finite bound falls back so the scalar path raises the
    // canonical "requires a bounded support" error.
    if (!_columnAllFinite(lo) || !_columnAllFinite(hi)) return null;
    return [lo, hi];
  }

  // Generic: iterate declared params with resolveParams' precedence.
  const out: any[] = [];
  for (let i = 0; i < entry.params.length; i++) {
    const paramName = entry.params[i];
    let exprIR;
    if (paramName in kwargs) {
      exprIR = kwargs[paramName];
    } else if (entry.aliases[paramName] && entry.aliases[paramName] in kwargs) {
      exprIR = kwargs[entry.aliases[paramName]];
    } else if (i < positional.length) {
      exprIR = positional[i];
    } else {
      throw new Error(
        `sampler: '${measureIR.op}' missing parameter '${paramName}'`
      );
    }
    const col = _normalizeParamColumn(
      evaluateExprN(exprIR, refArrays, count, baseEnv), count, measureIR.op);
    if (col === null) return null;
    out.push(col);
  }
  return out;
}

// =====================================================================
// Philox PRNG adapter — bridges pure-functional Philox to stdlib's
// stateful `opts.prng` callback contract.
// =====================================================================

function makePhiloxPrngAdapter(initialState: any) {
  let state = initialState;
  function prng() {
    const [u, next] = rng.nextUniform(state);
    state = next;
    return u;
  }
  prng.getState = () => state;
  prng._setState = (s: any) => { state = s; };
  return prng;
}

// =====================================================================
// Bulk-uniform PRNG adapter (commit 3, hybrid Option C).
//
// Pre-fills a Float64Array of `N * factor` uniforms via the bit-exact
// `rng.philoxNUniform` bulk path, then serves them via a per-call .prng
// closure that simply pops the buffer at i++. When the buffer drains
// (rejection-method dists consume variable uniforms per draw and may
// exhaust the pre-fill estimate), the adapter transparently refills
// from `rng.philoxNUniform` at the current state — bit-exact equivalent
// to a sequence of scalar `nextUniform` calls, so any stdlib transform
// closure built on top of this adapter produces samples identical to
// the scalar-loop path (no stream-byte drift for the bulk-adapter case).
//
// `factor` is the over-sampling multiple. 1 is tight (only safe for
// known-1U-per-draw dists); 4 is the audit's recommended default and
// matches the typical worst case for ziggurat-style rejection samplers.
// The cost of over-sampling is small: each extra uniform is ~one
// Float64 read + one cipher amortization, far below the per-call
// closure cost we're eliminating.
//
// Returns a prng closure with the same surface as
// `makePhiloxPrngAdapter` (.getState / ._setState).
//
// On state retrieval (.getState()), the adapter "rewinds" the unused
// tail of the pre-fill buffer: it returns the state that corresponds
// to the residue of uniforms not yet handed out, so the caller's
// next operation resumes seamlessly. This is achieved by running a
// SECOND `philoxNUniform` of size `consumedSoFar` from the initial
// state — bit-exact equivalence guarantees this lands on the same
// state that scalar-loop execution would have reached after the same
// number of `nextUniform` calls.

function makeBulkUniformPrngAdapter(initialState: any, n: number, factor?: number) {
  const f = factor != null ? (factor | 0) : 4;
  const _factor = f > 0 ? f : 4;

  // `baseState` is the cumulative "starting state" for the current
  // buffer. Each refill advances baseState to the state AFTER the
  // newly-filled buffer (returned by philoxNUniform), so a partial
  // consumption of the current buffer can be unwound by re-running a
  // smaller philoxNUniform from the PREVIOUS baseState.
  let prevBaseState: any = initialState;
  let baseState: any = initialState;
  let bufLen = 0;
  let cursor = 0;
  let buf: Float64Array | null = null;

  function _refill(requested: number) {
    const size = Math.max(requested, 1);
    prevBaseState = baseState;
    const r = rng.philoxNUniform(baseState, size);
    buf = r.out;
    bufLen = size;
    cursor = 0;
    baseState = r.state;
  }

  // Initial pre-fill — size = max(n * factor, 1).
  _refill(Math.max((n | 0) * _factor, 1));

  function prng() {
    if (cursor >= bufLen) {
      // Refill on demand; use the original (over-)sampling factor as a
      // rough heuristic for how much we still need.
      _refill(Math.max((n | 0) * _factor, 4));
    }
    return (buf as Float64Array)[cursor++];
  }

  // Materialise the state corresponding to "exactly `cursor` uniforms
  // consumed from `prevBaseState`". For cursor == bufLen this is just
  // baseState. For cursor < bufLen we re-run philoxNUniform at the
  // previous base for `cursor` outputs — bit-exact-equivalent to having
  // called scalar `nextUniform` exactly `cursor` times.
  prng.getState = function () {
    if (cursor === bufLen) return baseState;
    if (cursor === 0)       return prevBaseState;
    const r = rng.philoxNUniform(prevBaseState, cursor);
    return r.state;
  };
  prng._setState = function (s: any) {
    // Reset: a downstream caller is overriding our state. Discard the
    // buffer and reset both bases to the new state.
    prevBaseState = s;
    baseState = s;
    buf = null;
    bufLen = 0;
    cursor = 0;
  };
  return prng;
}

// =====================================================================
// stdlib distribution support normalization
// =====================================================================

function readSupport(dist: any) {
  const s = dist.support;
  if (Array.isArray(s)) return [s[0], s[1]];
  if (s && typeof s === 'object') {
    if ('lower' in s && 'upper' in s) return [s.lower, s.upper];
  }
  return [-Infinity, Infinity];
}

module.exports = {
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
};
