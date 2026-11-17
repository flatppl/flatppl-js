'use strict';

// §09 normalized particle-physics laws. Every draw consumes only its supplied
// uniform stream. Crystal Ball draws mix analytic core/tail laws; ARGUS uses
// bounded-proposal rejection or a truncated gamma transform. Landau uses the
// Chambers–Mallows–Stuck α=β=1 stable transform in the spec's φ convention.
const { integral } = require('./scalar-quadrature.ts');
const { cdf, truncatedQuantile } = require('./forward-cdf.ts');
const { probit } = require('./inverse-cdf.ts');
const PI = Math.PI, SQRT2PI = Math.sqrt(2 * PI), HALF = Math.sqrt(PI / 2);
type Uniform = () => number;
const open = (u: Uniform): number => Math.max(Number.EPSILON, Math.min(1 - Number.EPSILON, u()));
const normalCdf = (x: number): number => cdf('Normal', x, { mu: 0, sigma: 1 });

/** §09 Crystal Ball normalization. Log tail masses retain large exponents. */
function crystalParameters(alpha: number, n: number): { logTail: number; core: number } {
  if (!(n > 1)) throw new Error('CrystalBall: n must be greater than one');
  return { logTail: Math.log(n) - Math.log(alpha) - Math.log(n - 1) - alpha * alpha / 2,
    core: SQRT2PI * normalCdf(alpha) };
}

function logadd(a: number, b: number): number {
  const high = Math.max(a, b);
  return high + Math.log1p(Math.exp(Math.min(a, b) - high));
}

/** §09 CrystalBall: invert the normalized Gaussian core / Pareto-tail mixture. */
function crystalBall(u: Uniform, m0: number, sigma: number, alpha: number, n: number): number {
  const q = crystalParameters(alpha, n), logTotal = logadd(q.logTail, Math.log(q.core));
  const draw = open(u), logFraction = q.logTail - logTotal;
  const z = Math.log(draw) < logFraction
    ? -alpha - n / alpha * Math.expm1((logFraction - Math.log(draw)) / (n - 1))
    : -probit(Math.exp(Math.log1p(-draw) + logTotal - Math.log(SQRT2PI)));
  return m0 + sigma * z;
}

function crystalCdf(x: number, m0: number, sigma: number, alpha: number, n: number): number {
  const q = crystalParameters(alpha, n), total = logadd(q.logTail, Math.log(q.core)), z = (x - m0) / sigma;
  if (z < -alpha) return Math.exp(q.logTail - total
    + (1 - n) * Math.log1p((-alpha - z) * alpha / n));
  if (z > 0) return 1 - normalCdf(-z) * Math.exp(Math.log(SQRT2PI) - total);
  return Math.exp(q.logTail - total)
    + (normalCdf(z) - normalCdf(-alpha)) * Math.exp(Math.log(SQRT2PI) - total);
}

function crystalLogpdf(x: number, m0: number, sigma: number, alpha: number, n: number): number {
  const q = crystalParameters(alpha, n), z = (x - m0) / sigma;
  const kernel = z < -alpha ? -alpha * alpha / 2 - n * Math.log1p((-alpha - z) * alpha / n)
    : -z * z / 2;
  return kernel - Math.log(sigma) - logadd(q.logTail, Math.log(q.core));
}

/** §09 double-sided normalization, shared by the score, sampler and exact CDF. */
function doubleParameters(sigmaL: number, sigmaR: number, alphaL: number, alphaR: number,
    nL: number, nR: number): { logs: number[]; total: number } {
  const left = crystalParameters(alphaL, nL), right = crystalParameters(alphaR, nR);
  const logs = [Math.log(sigmaL) + left.logTail,
    Math.log(sigmaL) + Math.log(HALF * (1 - 2 * normalCdf(-alphaL))),
    Math.log(sigmaR) + Math.log(HALF * (1 - 2 * normalCdf(-alphaR))),
    Math.log(sigmaR) + right.logTail];
  return { logs, total: logadd(logadd(logs[0], logs[1]), logadd(logs[2], logs[3])) };
}

/** §09 DoubleSidedCrystalBall: choose one of four normalized core/tail pieces. */
function doubleCrystalBall(u: Uniform, m0: number, sigmaL: number, sigmaR: number,
    alphaL: number, alphaR: number, nL: number, nR: number): number {
  const { logs: logWeights } = doubleParameters(sigmaL, sigmaR, alphaL, alphaR, nL, nR);
  const coreL = HALF * (1 - 2 * normalCdf(-alphaL));
  const coreR = HALF * (1 - 2 * normalCdf(-alphaR));
  const high = Math.max(...logWeights), weights = logWeights.map(w => Math.exp(w - high));
  const choice = open(u) * weights.reduce((a, b) => a + b), v = open(u);
  if (choice < weights[0]) return m0 - sigmaL * (alphaL + nL / alphaL * Math.expm1(-Math.log(v) / (nL - 1)));
  if (choice < weights[0] + weights[1]) return m0 + sigmaL * probit(0.5 - v * coreL / SQRT2PI);
  if (choice < weights[0] + weights[1] + weights[2]) return m0 - sigmaR * probit(0.5 - v * coreR / SQRT2PI);
  return m0 + sigmaR * (alphaR + nR / alphaR * Math.expm1(-Math.log(v) / (nR - 1)));
}

function doubleCrystalLogpdf(x: number, m0: number, sigmaL: number, sigmaR: number,
    alphaL: number, alphaR: number, nL: number, nR: number): number {
  const z = (x - m0) / (x < m0 ? sigmaL : sigmaR);
  const alpha = x < m0 ? alphaL : alphaR, n = x < m0 ? nL : nR;
  const kernel = Math.abs(z) > alpha
    ? -alpha * alpha / 2 - n * Math.log1p((Math.abs(z) - alpha) * alpha / n) : -z * z / 2;
  return kernel - doubleParameters(sigmaL, sigmaR, alphaL, alphaR, nL, nR).total;
}

function doubleCrystalCdf(x: number, m0: number, sigmaL: number, sigmaR: number,
    alphaL: number, alphaR: number, nL: number, nR: number): number {
  const { logs, total } = doubleParameters(sigmaL, sigmaR, alphaL, alphaR, nL, nR);
  const z = (x - m0) / (x < m0 ? sigmaL : sigmaR);
  if (z < -alphaL) return Math.exp(logs[0] - total
    + (1 - nL) * Math.log1p((-alphaL - z) * alphaL / nL));
  if (z > alphaR) return -Math.expm1(logs[3] - total
    + (1 - nR) * Math.log1p((z - alphaR) * alphaR / nR));
  if (z <= 0) return Math.exp(logs[0] - total)
    + (normalCdf(z) - normalCdf(-alphaL)) * Math.exp(Math.log(sigmaL * SQRT2PI) - total);
  return Math.exp(logs[0] - total) + Math.exp(logs[1] - total)
    + (normalCdf(z) - 0.5) * Math.exp(Math.log(sigmaR * SQRT2PI) - total);
}

/** ∫₀¹ u^power exp(slope·u)du, evaluated without cancelling gamma factors. */
function argusLogIntegral(slope: number, power: number): number {
  if (slope === 0) return -Math.log(power + 1);
  if (slope < 0 && power < -slope) {
    const gamma = require('./value-ops.ts').loggammaElem(require('./value.ts').scalar(power + 1)).data[0];
    return gamma + Math.log(cdf('Gamma', 1, { shape: power + 1, rate: -slope }))
      - (power + 1) * Math.log(-slope);
  }
  const rate = power + 1 + slope;
  const logKernel = (s: number) => -(power + 1) * s / rate + slope * Math.expm1(-s / rate);
  let upper = 1;
  while (logKernel(upper) > -45) upper *= 2;
  return slope - Math.log(rate) + Math.log(integral((s: number) => Math.exp(logKernel(s)), 0, upper, Math.ceil(upper / 4)));
}

/** §09 ARGUS: under t=1-(x/m)² the density is t^power exp(slope*t).
 * The proposal follows the dominant power or exponential, avoiding rejection
 * rates that vanish for either large positive or large negative slopes. */
function argus(u: Uniform, resonance: number, slope: number, power: number): number {
  if (slope < 0 && power < -slope) {
    const t = truncatedQuantile('Gamma', open(u), { shape: power + 1, rate: -slope }, 0, 1);
    return resonance * Math.sqrt(1 - t);
  }
  for (;;) {
    let t: number, logAccept: number;
    if (slope > power + 1) {
      const v = -Math.log1p(-open(u) * -Math.expm1(-slope)) / slope;
      t = 1 - v; logAccept = power * Math.log1p(-v);
    } else if (slope < 0) {
      t = Math.exp(Math.log(open(u)) / (power + slope + 1));
      logAccept = -slope * (Math.log(t) + 1 - t);
    } else {
      t = Math.exp(Math.log(open(u)) / (power + 1));
      logAccept = slope * (t - 1);
    }
    if (Math.log(open(u)) <= logAccept) return resonance * Math.sqrt(1 - t);
  }
}

/** The quartic splits into two real quadratics. atan2 retains its [0,π] branch. */
function breitWignerCdf(x: number, mean: number, width: number): number {
  if (x <= 0) return 0;
  if (x === Infinity) return 1;
  const r = mean * Math.hypot(mean, width), a = Math.sqrt(2 * (r + mean * mean));
  const d = mean * width / a, minus = (x - a / 2) ** 2 + d * d;
  const logarithm = Math.log1p(2 * a * x / minus);
  return (Math.atan2(2 * x * d, r - x * x) + d / a * logarithm) / PI;
}

/** Bracketed monotone inversion; stop at adjacent floating-point endpoints. */
function inverse(fn: (x: number) => number, p: number, lo: number, hi: number): number {
  while (fn(lo) > p) lo = lo < 0 ? lo * 2 : -1;
  while (fn(hi) < p) hi = hi > 0 ? hi * 2 : 1;
  for (let k = 0; k < 80; k++) {
    const mid = lo / 2 + hi / 2;
    if (mid === lo || mid === hi) return mid;
    if (fn(mid) < p) lo = mid; else hi = mid;
  }
  return lo / 2 + hi / 2;
}

function breitWigner(u: Uniform, mean: number, width: number): number {
  return inverse(x => breitWignerCdf(x, mean, width), open(u), 0, mean + width);
}

/** §09 Voigtian is the convolution of a normal and a width/2 Cauchy law. */
function voigtian(u: Uniform, mean: number, width: number, sigma: number): number {
  return mean + sigma * probit(open(u)) + width / 2 * Math.tan(PI * (open(u) - 0.5));
}

/** §09 Landau/CERN convention, using the α=β=1 CMS stable transform. */
function landau(u: Uniform, loc: number, scale: number): number {
  const angle = PI * (open(u) - 0.5), a = PI / 2 + angle, exponential = -Math.log(open(u));
  return loc + scale * (a * Math.tan(angle) - Math.log(exponential * Math.cos(angle) / a));
}

/** §09 split normal: side mass is proportional to its Gaussian width. */
function bifurcated(u: Uniform, mean: number, sigmaL: number, sigmaR: number): number {
  const side = open(u) * (sigmaL + sigmaR) < sigmaL ? -sigmaL : sigmaR;
  return mean + side * -probit(open(u) / 2);
}

function bifurcatedCdf(x: number, mean: number, sigmaL: number, sigmaR: number): number {
  return x < mean ? 2 * sigmaL / (sigmaL + sigmaR) * normalCdf((x - mean) / sigmaL)
    : 1 - 2 * sigmaR / (sigmaL + sigmaR) * normalCdf((mean - x) / sigmaR);
}

/** Analytical plot helpers share the same declared density. Sampling does not
 * use this numerical CDF. The Cauchy coordinate includes both infinite tails. */
function analytical(logpdf: any, params: number[], support: number[],
    center: number, scale: number, exactCdf?: (...args: number[]) => number): any {
  const lower = support[0], upper = support[1];
  const score = logpdf.factory ? logpdf.factory(...params) : (x: number) => logpdf(x, ...params);
  const density = (x: number) => Math.exp(score(x));
  const distribution: any = { support, pdf: density, logpdf: score };
  distribution.cdf = (x: number) => {
    if (x <= lower) return 0;
    if (x >= upper) return 1;
    if (exactCdf) return exactCdf(x, ...params);
    const a = Number.isFinite(lower) ? Math.atan((lower - center) / scale) : -PI / 2;
    const b = Math.atan((x - center) / scale);
    return Math.max(0, Math.min(1, integral((t: number) => {
      const c = Math.cos(t), z = center + scale * Math.tan(t);
      return density(z) * scale / (c * c);
    }, a, b, 8, 1e-8, 1e-10)));
  };
  distribution.quantile = (p: number) => p <= 0 ? lower : p >= 1 ? upper
    : inverse(distribution.cdf, p, Number.isFinite(lower) ? lower : center - scale,
      Number.isFinite(upper) ? upper : center + scale);
  return distribution;
}

module.exports = { crystalBall, crystalCdf, crystalLogpdf, doubleCrystalBall, doubleCrystalCdf,
  doubleCrystalLogpdf, argus, argusLogIntegral, breitWigner,
  breitWignerCdf, voigtian, landau, bifurcated, bifurcatedCdf, analytical };
