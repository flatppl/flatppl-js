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
const stdlibBinomcoefln = require('@stdlib/math-base-special-binomcoefln');

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
  return stdlibBinomcoefln(ki + alpha - 1, alpha - 1)
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
  return stdlibBinomcoefln(ki + psi - 1, ki)
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
// Pairing convention for Normal (and LogNormal, which exponentiates
// Normal): philoxNNormal already documents that pair `i` of two
// uniforms produces normal `i` = r*cos(theta) (first output) and
// `i+1` = r*sin(theta) (second output). We inherit that ordering
// without further reshuffling.

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
  // column (per-i). The raw Box-Muller draw is unchanged — only the
  // affine transform reads params per element.
  const mu = params[0];
  const sigma = params[1];
  const dest = out ?? new Float64Array(n);
  const r = rng.philoxNNormal(state, n, dest);
  // Vectorised affine: x = mu + sigma * z. Inplace.
  for (let i = 0; i < n; i++) r.out[i] = _p(mu, i) + _p(sigma, i) * r.out[i];
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
  // params = [mu, sigma]; x = exp(mu + sigma * z), z ~ Normal(0,1).
  // Reuses the Box-Muller path via randNNormal then exp-in-place.
  const r = randNNormal(state, params, n, out);
  for (let i = 0; i < n; i++) r.out[i] = Math.exp(r.out[i]);
  return r;
}

const REGISTRY = {
  Normal: {
    params:   ['mu', 'sigma'],
    aliases:  {},
    discrete: false,
    Ctor:     Normal,
    randFn:   randNormal,
    // Explicit batched path: Box-Muller via philoxNNormal. NOT
    // bit-equivalent to scalar randNormal (which uses ziggurat) —
    // produces different sample bytes from the same key. The
    // engine's seed→sample mapping was already broken in commit 2
    // (PhiloxKey rework); commit 3 documents that ziggurat→Box-Muller
    // is the additional reason scalar and batched paths differ.
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
  LogNormal: {
    params:   ['mu', 'sigma'],
    aliases:  {},
    discrete: false,
    Ctor:     LogNormal,
    randFn:   randLogNormal,
    // Explicit batched path: exp(Normal(mu, sigma)) via Box-Muller.
    // Same bit-equivalence caveat as Normal (Box-Muller ≠ scalar
    // ziggurat-based randLogNormal).
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
  const kwargs = measureIR.kwargs || {};
  const positional = measureIR.args || [];
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
  if (ir.kind === 'call' && ir.op === 'interval'
      && Array.isArray(ir.args) && ir.args.length === 2) {
    const evaluateExpr = require('./sampler.ts').evaluateExpr;
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
