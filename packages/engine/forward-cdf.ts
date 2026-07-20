// packages/engine/forward-cdf.ts
'use strict';
// Forward CDF ladder — companion to inverse-cdf.ts. Needed by the nested prior
// transform for TRUNCATED priors: a truncate(D, [lo,hi]) latent maps a cube
// coord u through F_D^{-1}(F_D(lo) + u·(F_D(hi)-F_D(lo))), which needs F_D.
const stdlibErfc     = require('@stdlib/math-base-special-erfc');
const stdlibGammainc = require('@stdlib/math-base-special-gammainc');   // regularized P(x, s)
const stdlibBetainc  = require('@stdlib/math-base-special-betainc');    // regularized I_x(a,b)
const { quantile, hasQuantile, numericalQuantile } = require('./inverse-cdf.ts');

// Φ(x) = ½ erfc(−x/√2).
function normCdf(x: number): number { return 0.5 * stdlibErfc(-x / Math.SQRT2); }

const CDF: Record<string, (x: number, q: any) => number> = {
  Normal:      (x, q) => normCdf((x - q.mu) / q.sigma),
  LogNormal:   (x, q) => x <= 0 ? 0 : normCdf((Math.log(x) - q.mu) / q.sigma),
  Exponential: (x, q) => x <= 0 ? 0 : -Math.expm1(-q.rate * x),
  Uniform:     (x, q) => x <= q.lo ? 0 : (x >= q.hi ? 1 : (x - q.lo) / (q.hi - q.lo)),
  Beta:        (x, q) => x <= 0 ? 0 : (x >= 1 ? 1 : stdlibBetainc(x, q.alpha, q.beta)),
  Gamma:       (x, q) => x <= 0 ? 0 : stdlibGammainc(q.rate * x, q.shape),   // P(shape, rate·x)
  Cauchy:      (x, q) => 0.5 + Math.atan((x - q.location) / q.scale) / Math.PI,
  HalfCauchy:  (x, q) => x <= 0 ? 0 : 2 * Math.atan(x / q.scale) / Math.PI,
  HalfNormal:  (x, q) => x <= 0 ? 0 : 2 * normCdf(x / q.sigma) - 1,
  Logistic:    (x, q) => 1 / (1 + Math.exp(-(x - q.mu) / q.s)),
  Weibull:     (x, q) => x <= 0 ? 0 : -Math.expm1(-Math.pow(x / q.scale, q.shape)),
  Pareto:      (x, q) => x < q.scale ? 0 : 1 - Math.pow(q.scale / x, q.shape),
  Laplace:     (x, q) => { const z = (x - q.location) / q.scale; return z < 0 ? 0.5 * Math.exp(z) : 1 - 0.5 * Math.exp(-z); },
  // X~IG(shape,scale) ⇔ scale/X ~ Gamma(shape, rate=1) ⇒ F_X(x)=1-P(shape,scale/x).
  InverseGamma: (x, q) => x <= 0 ? 0 : 1 - stdlibGammainc(q.scale / x, q.shape),
};
function hasCdf(distOp: string): boolean { return Object.prototype.hasOwnProperty.call(CDF, distOp); }
function cdf(distOp: string, x: number, params: any): number {
  const f = CDF[distOp];
  if (!f) throw new Error(`forward-cdf: no CDF for '${distOp}'`);
  return f(x, params);
}

// F⁻¹(F(lo) + u·(F(hi)-F(lo))) — the quantile of D restricted to [lo,hi].
// Prefers inverse-cdf.ts's closed-form/library quantile (rungs 1-2); falls back
// to its numericalQuantile (rung 3, bracketed on [lo,hi]) for a distOp — e.g.
// plain Cauchy — that has a forward CDF here but no registered inverse there.
function truncatedQuantile(distOp: string, u: number, params: any, lo: number, hi: number): number {
  const Flo = Number.isFinite(lo) ? cdf(distOp, lo, params) : 0;
  const Fhi = Number.isFinite(hi) ? cdf(distOp, hi, params) : 1;
  const target = Flo + u * (Fhi - Flo);
  if (hasQuantile(distOp)) return quantile(distOp, target, params);
  return numericalQuantile((x: number) => cdf(distOp, x, params), target, lo, hi);
}
module.exports = { cdf, hasCdf, truncatedQuantile, CDF };
