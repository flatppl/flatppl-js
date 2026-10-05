// packages/engine/forward-cdf.ts
'use strict';
// Forward CDF ladder — companion to inverse-cdf.ts. Needed by the nested prior
// transform for TRUNCATED priors: a truncate(D, [lo,hi]) latent maps a cube
// coord u through F_D^{-1}(F_D(lo) + u·(F_D(hi)-F_D(lo))), which needs F_D.
const stdlibErfc     = require('@stdlib/math-base-special-erfc');
const stdlibGammainc = require('@stdlib/math-base-special-gammainc');   // regularized P(x, s)
const stdlibBetainc  = require('@stdlib/math-base-special-betainc');    // regularized I_x(a,b)
const stdlibGammaincinv = require('@stdlib/math-base-special-gammaincinv');
const stdlibBetaincinv = require('@stdlib/math-base-special-betaincinv');
const { QUANTILE, probit, numericalQuantile } = require('./inverse-cdf.ts');

// Φ(x) = ½ erfc(−x/√2).
function normCdf(x: number): number { return 0.5 * stdlibErfc(-x / Math.SQRT2); }

const CDF: Record<string, (x: number, q: any) => number> = {
  Normal:      (x, q) => normCdf((x - q.mu) / q.sigma),
  LogNormal:   (x, q) => x <= 0 ? 0 : normCdf((Math.log(x) - q.mu) / q.sigma),
  Exponential: (x, q) => x <= 0 ? 0 : -Math.expm1(-q.rate * x),
  Uniform:     (x, q) => x <= q.lo ? 0 : (x >= q.hi ? 1 : (x - q.lo) / (q.hi - q.lo)),
  Beta:        (x, q) => x <= 0 ? 0 : (x >= 1 ? 1 : stdlibBetainc(x, q.alpha, q.beta)),
  Gamma:       (x, q) => x <= 0 ? 0 : stdlibGammainc(q.rate * x, q.shape),   // P(shape, rate·x)
  Cauchy:      (x, q) => Math.atan2(q.scale, q.location - x) / Math.PI,
  HalfCauchy:  (x, q) => x <= 0 ? 0 : 2 * Math.atan(x / q.scale) / Math.PI,
  HalfNormal:  (x, q) => x <= 0 ? 0 : 2 * normCdf(x / q.sigma) - 1,
  Logistic:    (x, q) => 1 / (1 + Math.exp(-(x - q.mu) / q.s)),
  Weibull:     (x, q) => x <= 0 ? 0 : -Math.expm1(-Math.pow(x / q.scale, q.shape)),
  Pareto:      (x, q) => x < q.scale ? 0 : 1 - Math.pow(q.scale / x, q.shape),
  Laplace:     (x, q) => { const z = (x - q.location) / q.scale; return z < 0 ? 0.5 * Math.exp(z) : 1 - 0.5 * Math.exp(-z); },
  // X~IG(shape,scale) ⇔ scale/X ~ Gamma(shape, rate=1) ⇒ F_X(x)=1-P(shape,scale/x).
  InverseGamma: (x, q) => x <= 0 ? 0 : stdlibGammainc(q.scale / x, q.shape, true, true),
  // ChiSquared(k) ≡ Gamma(shape=k/2, rate=1/2): F(x) = P(k/2, x/2). Verified vs
  // scipy.stats.chi2(df=k).cdf.
  ChiSquared: (x, q) => x <= 0 ? 0 : stdlibGammainc(x / 2, q.k / 2),
  // Standard StudentT(nu): F(x) = 1 - ½·I_z(nu/2,½) for x>0, ½·I_z(nu/2,½) for
  // x≤0, with z = nu/(nu+x²) (symmetric in x so the same z serves both
  // tails). Verified vs scipy.stats.t(df=nu).cdf across nu∈{3,10} incl. tails.
  StudentT: (x, q) => {
    const z = q.nu / (q.nu + x * x);
    const half = 0.5 * stdlibBetainc(z, q.nu / 2, 0.5);
    return x > 0 ? 1 - half : half;
  },
};

// Complementary CDFs must be evaluated directly: 1-F(x) loses an entire
// representable tail once F(x) rounds to one (spec §06 support restriction).
const SURVIVAL: Record<string, (x: number, q: any) => number> = {
  Normal: (x, q) => normCdf((q.mu - x) / q.sigma),
  LogNormal: (x, q) => x <= 0 ? 1 : normCdf((q.mu - Math.log(x)) / q.sigma),
  Exponential: (x, q) => x <= 0 ? 1 : Math.exp(-q.rate * x),
  Uniform: (x, q) => x <= q.lo ? 1 : x >= q.hi ? 0 : (q.hi - x) / (q.hi - q.lo),
  Beta: (x, q) => x <= 0 ? 1 : x >= 1 ? 0 : stdlibBetainc(x, q.alpha, q.beta, true, true),
  Gamma: (x, q) => x <= 0 ? 1 : stdlibGammainc(q.rate * x, q.shape, true, true),
  Cauchy: (x, q) => Math.atan2(q.scale, x - q.location) / Math.PI,
  HalfCauchy: (x, q) => x <= 0 ? 1 : 2 * Math.atan2(q.scale, x) / Math.PI,
  HalfNormal: (x, q) => x <= 0 ? 1 : stdlibErfc(x / (q.sigma * Math.SQRT2)),
  Logistic: (x, q) => 1 / (1 + Math.exp((x - q.mu) / q.s)),
  Weibull: (x, q) => x <= 0 ? 1 : Math.exp(-Math.pow(x / q.scale, q.shape)),
  Pareto: (x, q) => x < q.scale ? 1 : Math.pow(q.scale / x, q.shape),
  Laplace: (x, q) => { const z = (x - q.location) / q.scale; return z > 0 ? 0.5 * Math.exp(-z) : 1 - 0.5 * Math.exp(z); },
  InverseGamma: (x, q) => x <= 0 ? 1 : stdlibGammainc(q.scale / x, q.shape),
  ChiSquared: (x, q) => x <= 0 ? 1 : stdlibGammainc(x / 2, q.k / 2, true, true),
  StudentT: (x, q) => CDF.StudentT(-x, q),
};

// Invert the survival probability directly, without rounding 1-p to one.
const UPPER_QUANTILE: Record<string, (p: number, q: any) => number> = {
  Normal: (p, q) => q.mu - q.sigma * probit(p),
  LogNormal: (p, q) => Math.exp(q.mu - q.sigma * probit(p)),
  Exponential: (p, q) => -Math.log(p) / q.rate,
  Uniform: (p, q) => q.hi - p * (q.hi - q.lo),
  Beta: (p, q) => stdlibBetaincinv(p, q.alpha, q.beta, true),
  Gamma: (p, q) => stdlibGammaincinv(p, q.shape, true) / q.rate,
  Cauchy: (p, q) => q.location + q.scale / Math.tan(Math.PI * p),
  HalfCauchy: (p, q) => q.scale / Math.tan(Math.PI * p / 2),
  HalfNormal: (p, q) => -q.sigma * probit(p / 2),
  Logistic: (p, q) => q.mu + q.s * (Math.log1p(-p) - Math.log(p)),
  Weibull: (p, q) => q.scale * Math.pow(-Math.log(p), 1 / q.shape),
  Pareto: (p, q) => q.scale / Math.pow(p, 1 / q.shape),
  Laplace: (p, q) => p < 0.5 ? q.location - q.scale * Math.log(2 * p)
    : q.location + q.scale * Math.log(2 * (1 - p)),
  InverseGamma: (p, q) => q.scale / stdlibGammaincinv(p, q.shape),
  ChiSquared: (p, q) => 2 * stdlibGammaincinv(p, q.k / 2, true),
  StudentT: (p, q) => -QUANTILE.StudentT(p, q),
};
function hasCdf(distOp: string): boolean { return Object.prototype.hasOwnProperty.call(CDF, distOp); }
function cdf(distOp: string, x: number, params: any): number {
  const f = CDF[distOp];
  if (!f) throw new Error(`forward-cdf: no CDF for '${distOp}'`);
  return f(x, params);
}

/** Mass of a scalar interval under a registered probability law (§06 truncate). */
function intervalProbability(distOp: string, params: any, lo: number, hi: number): number {
  if (lo >= hi) return 0;
  const Flo = cdf(distOp, lo, params);
  return Flo >= 0.5
    ? SURVIVAL[distOp](lo, params) - SURVIVAL[distOp](hi, params)
    : cdf(distOp, hi, params) - Flo;
}

// F⁻¹(F(lo) + u·(F(hi)-F(lo))) — the quantile of D restricted to [lo,hi].
// Prefers inverse-cdf.ts's closed-form/library quantile (rungs 1-2); falls back
// to its numericalQuantile (rung 3, bracketed on [lo,hi]) for a distOp — e.g.
// plain Cauchy — that has a forward CDF here but no registered inverse there.
function truncatedQuantile(distOp: string, u: number, params: any, lo: number, hi: number): number {
  // Clip the conditional cube coordinate, never its tiny absolute tail mass.
  u = Math.max(1e-15, Math.min(1 - 1e-15, u));
  const Flo = cdf(distOp, lo, params);
  const mass = intervalProbability(distOp, params, lo, hi);
  if (!(mass > 0)) throw new Error('truncatedQuantile: interval has no representable probability mass');
  let result: number;
  if (Flo >= 0.5) {
    const Slo = SURVIVAL[distOp](lo, params), Shi = SURVIVAL[distOp](hi, params);
    result = UPPER_QUANTILE[distOp]((1 - u) * Slo + u * Shi, params);
  } else if (QUANTILE[distOp]) {
    result = QUANTILE[distOp](Flo + u * mass, params);
  } else {
    result = numericalQuantile((x: number) => (cdf(distOp, x, params) - Flo) / mass, u, lo, hi);
  }
  // An inverse can land a few ulps across an endpoint at extreme cube values.
  return Math.max(lo, Math.min(hi, result));
}
module.exports = { cdf, hasCdf, intervalProbability, truncatedQuantile, CDF };
