// packages/engine/inverse-cdf.ts
'use strict';
//
// Inverse-CDF ("quantile") ladder for the prior distributions a model can use —
// the building block of nested sampling's prior transform T:[0,1]^d → θ (each
// latent's cube coordinate maps through its conditional F⁻¹). Three rungs, by cost:
//
//   1. Closed form              — Normal/LogNormal/Exponential/Uniform/HalfNormal/
//                                 HalfCauchy/Logistic/Weibull/Pareto/Laplace/Dirac.
//   2. Library inverse          — Beta (betaincinv) and Gamma (gammaincinv): the
//                                 regularized-incomplete inverses @stdlib ships,
//                                 accurate to scipy's ppf (verified) — no
//                                 hand-rolled Newton needed.
//   3. Numerical inversion      — `numericalQuantile`: monotone bracketed bisection
//                                 on any user-supplied cdf, the last-resort path for
//                                 a distribution with a cdf but no faster route.
//
// Params are the same resolved objects the prior log-density table consumes
// (model-view PRIOR_LOGPDF): { mu, sigma } / { rate } / { alpha, beta } /
// { shape, rate } / { lo, hi } / { scale } — so the transform passes a latent's
// conditionally-resolved params straight through.

const stdlibErfcinv    = require('@stdlib/math-base-special-erfcinv');
const stdlibGammaincinv = require('@stdlib/math-base-special-gammaincinv');
const stdlibBetaincinv  = require('@stdlib/math-base-special-betaincinv');

// Standard-normal quantile Φ⁻¹(p) = −√2·erfc⁻¹(2p) (spec §07 `probit`).
function probit(p: number): number { return -Math.SQRT2 * stdlibErfcinv(2 * p); }

// Clamp p into the open unit interval so the tail inverses stay finite.
const EPS = 1e-15;
function clip(p: number): number { return p < EPS ? EPS : (p > 1 - EPS ? 1 - EPS : p); }

const QUANTILE: Record<string, (p: number, q: any) => number> = {
  Normal:      (p, q) => q.mu + q.sigma * probit(p),
  LogNormal:   (p, q) => Math.exp(q.mu + q.sigma * probit(p)),
  Exponential: (p, q) => -Math.log1p(-p) / q.rate,
  Uniform:     (p, q) => q.lo + p * (q.hi - q.lo),
  Beta:        (p, q) => stdlibBetaincinv(p, q.alpha, q.beta),
  Gamma:       (p, q) => stdlibGammaincinv(p, q.shape) / q.rate,   // shape/rate; gammaincinv is scale-1
  HalfNormal:  (p, q) => q.sigma * probit(0.5 * (1 + p)),          // |N(0,σ)|: Φ⁻¹((1+p)/2)·σ
  HalfCauchy:  (p, q) => q.scale * Math.tan(0.5 * Math.PI * p),
  Logistic:    (p, q) => q.mu + q.s * Math.log(p / (1 - p)),
  Weibull:     (p, q) => q.scale * Math.pow(-Math.log1p(-p), 1 / q.shape),
  Pareto:      (p, q) => q.scale / Math.pow(1 - p, 1 / q.shape),   // scale = x_m, shape = α
  Laplace:     (p, q) => { const x = p - 0.5; return q.location - q.scale * Math.sign(x) * Math.log1p(-2 * Math.abs(x)); },
  Dirac:       (_p, q) => q.value,
  // X~IG(shape,scale) ⇔ scale/X ~ Gamma(shape, rate=1) ⇒ F_X(x)=1-P(shape,scale/x);
  // invert: x = scale / gammaincinv(1-p, shape).
  InverseGamma: (p, q) => q.scale / stdlibGammaincinv(1 - p, q.shape),
  Cauchy: (p, q) => q.location + q.scale * Math.tan(Math.PI * (p - 0.5)),
  // ChiSquared(k) ≡ Gamma(shape=k/2, rate=1/2); gammaincinv is scale-1 (rate-1),
  // so divide by rate=1/2 ⇔ multiply by 2. Verified vs scipy.stats.chi2(df=k).ppf.
  ChiSquared: (p, q) => 2 * stdlibGammaincinv(p, q.k / 2),
  // Standard StudentT(nu) (zero mean, unit scale, spec §08). Inverse via the
  // regularized-incomplete-beta identity: for T~t(nu), P(|T|>t) = I_z(nu/2,½)
  // with z = nu/(nu+t²); invert for t given the tail probability, then place
  // the sign by which side of the median p falls on. Verified vs
  // scipy.stats.t(df=nu).ppf across nu∈{3,10} and p incl. tails (see
  // test/inverse-cdf.test.ts) — the naive z=betaincinv(...) ↔ t mapping is
  // easy to get backwards/inverted; this form matches scipy exactly to
  // float64 precision, not just approximately.
  StudentT: (p, q) => {
    const p2 = p < 0.5 ? 2 * p : 2 * (1 - p);
    const z = stdlibBetaincinv(p2, q.nu / 2, 0.5);
    const t = Math.sqrt(q.nu * (1 - z) / z);
    return p < 0.5 ? -t : (p > 0.5 ? t : 0);
  },
};

// True iff a distOp has a registered closed-form / library quantile.
function hasQuantile(distOp: string): boolean { return Object.prototype.hasOwnProperty.call(QUANTILE, distOp); }

// F⁻¹(p) for a distOp with a registered quantile. `p` is clamped to (0,1).
function quantile(distOp: string, p: number, params: any): number {
  const f = QUANTILE[distOp];
  if (!f) throw new Error(`inverse-cdf: no quantile for '${distOp}' (rung 3: pass a cdf to numericalQuantile)`);
  return f(clip(p), params);
}

// Rung 3 — monotone bracketed bisection of an arbitrary continuous cdf on
// [lo, hi] (either bound may be ±Infinity: the bracket grows geometrically from a
// finite seed until it straddles p). `cdf` must be non-decreasing. Accurate to
// `tol` in x or `maxIter` steps. Use only when no closed-form/library inverse
// exists — it is the slow lane.
function numericalQuantile(cdf: (x: number) => number, p: number, lo: number, hi: number, tol = 1e-12, maxIter = 200): number {
  p = clip(p);
  let a = lo, b = hi;
  // Expand infinite bounds to a finite bracket straddling p.
  if (!Number.isFinite(a)) { let s = -1; while (cdf(s) > p) s *= 2; a = s; }
  if (!Number.isFinite(b)) { let s = 1; while (cdf(s) < p) s *= 2; b = s; }
  for (let it = 0; it < maxIter; it++) {
    const m = 0.5 * (a + b);
    if (b - a < tol) return m;
    if (cdf(m) < p) a = m; else b = m;
  }
  return 0.5 * (a + b);
}

module.exports = { quantile, hasQuantile, numericalQuantile, probit, QUANTILE };
