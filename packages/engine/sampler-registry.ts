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
    params:   ['rate'],
    aliases:  {},
    discrete: false,
    Ctor:     Exponential,
    randFn:   randExponential,
    logpdfFn: logpdfExponential,
  },
  Uniform: {
    params:   ['support'],
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
  regionBoundsFromIR,
  makePhiloxPrngAdapter,
  readSupport,
};
