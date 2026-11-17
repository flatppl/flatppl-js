'use strict';

// Regularized gamma P/Q, DLMF 8.7.1, 8.9.2 and 8.12.3–10. The central
// prefactor uses Stirling's remainder instead of subtracting two O(a log a)
// values. Temme's uniform expansion handles shapes too large for a series.
const V = require('./value.ts');
const logGamma = (a: number) => require('./value-ops.ts').loggammaElem(V.scalar(a)).data[0];
const erfc = (x: number) => 2 * require('./forward-cdf.ts').CDF.Normal(-x * Math.SQRT2, { mu: 0, sigma: 1 });

function log1pmx(x: number): number {
  if (Math.abs(x) >= 0.125) return Math.log1p(x) - x;
  let term = -x * x, sum = term / 2;
  for (let n = 3; n < 64; n++) {
    term *= -x;
    const add = term / n; sum += add;
    if (Math.abs(add) <= Number.EPSILON * Math.abs(sum)) break;
  }
  return sum;
}

function prefactor(a: number, x: number): number {
  if (a < 8) return a * Math.log(x) - x - logGamma(a);
  const delta = (x - a) / a;
  // Far below the mean delta can round to -1 while x is still positive.
  // Direct log ratios retain that lower tail; log1pmx handles cancellation
  // near the mean where the two O(a) terms nearly cancel.
  const exponent = Math.abs(delta) < 0.5 ? a * log1pmx(delta)
    : a * (Math.log(x) - Math.log(a)) + a - x;
  return exponent + (Math.log(a) - Math.log(2 * Math.PI)) / 2 - stirlingRemainder(a);
}

function stirlingRemainder(a: number): number {
  const r = 1 / a, square = r * r;
  const coefficients = [1 / 12, -1 / 360, 1 / 1260, -1 / 1680, 1 / 1188,
    -691 / 360360, 1 / 156, -3617 / 122400];
  let correction = coefficients[7];
  for (let n = 6; n >= 0; n--) correction = coefficients[n] + square * correction;
  return r * correction;
}

function polynomial(coefficients: number[], x: number): number {
  let value = coefficients[coefficients.length - 1];
  for (let i = coefficients.length - 2; i >= 0; i--) value = value * x + coefficients[i];
  return value;
}

/** DLMF 5.7.3 retains log Γ(1+a) when adding a to one would lose bits. */
function logGamma1p(a: number): number {
  if (a >= 0.125) return logGamma(1 + a);
  const zeta = [0.8224670334241132, -0.40068563438653143, 0.27058080842778455,
    -0.20738555102867399, 0.1695571769974082, -0.14404989676884612,
    0.12550966952474304, -0.11133426586956469, 0.10009945751278181,
    -0.09095401714582904, 0.083353840546109, -0.0769325164113522,
    0.07143294629536134, -0.06666870588242047, 0.06250095514121304];
  return a * (-0.5772156649015329 + a * polynomial(zeta, a));
}

function temme(a: number, x: number, upper: boolean): number {
  const delta = (x - a) / a;
  const eta = Math.sign(delta) * Math.sqrt(-2 * log1pmx(delta));
  const c0 = Math.abs(delta) < 0.01 ? polynomial([-1 / 3, 1 / 12, -23 / 540,
    353 / 12960, -589 / 30240, 81083 / 5443200, -7783 / 653184,
    514303 / 52254720, -646245559 / 77598259200], delta) : 1 / delta - 1 / eta;
  const c1 = Math.abs(delta) < 0.01 ? polynomial([-1 / 540, -1 / 288, 23 / 6048,
    -3733 / 1088640, 3253 / 1088640, -135719 / 52254720, 176215213 / 77598259200], delta)
    : 1 / eta ** 3 - 1 / delta ** 3 - 1 / delta ** 2 - 1 / (12 * delta);
  const correction = Math.exp(-a * eta * eta / 2 - (Math.log(2 * Math.PI) + Math.log(a)) / 2) * (c0 + c1 / a);
  const argument = eta * Math.sqrt(a / 2);
  return upper ? erfc(argument) / 2 + correction : erfc(-argument) / 2 - correction;
}

/** The upper continued fraction keeps tiny survival probabilities in log scale. */
function regularizedGamma(a: number, x: number, upper = false): number {
  if (!(a > 0) || !(x >= 0)) return NaN;
  if (x === 0) return upper ? 1 : 0;
  if (x === Infinity) return a === Infinity ? NaN : upper ? 0 : 1;
  if (a === Infinity) return upper ? 1 : 0;
  if (a >= 1e7) return temme(a, x, upper);
  if (upper && a < 0.5 && x < 1) {
    // Complement the alternating lower-gamma series before its leading one
    // rounds away a small upper tail. This is stable even as a tends to zero.
    const log = a * Math.log(x) - logGamma1p(a);
    let term = -x, sum = term / (a + 1);
    for (let n = 2; n < 1000; n++) {
      term *= -x / n;
      const add = term / (a + n); sum += add;
      if (Math.abs(add) <= Number.EPSILON * Math.abs(sum)) break;
    }
    return -Math.expm1(log) - a * Math.exp(log) * sum;
  }
  const logScale = prefactor(a, x);
  if (x < a + 1) {
    let term = 1 / a, sum = term;
    for (let n = 1; n < 1000000; n++) {
      term *= x / (a + n); sum += term;
      if (Math.abs(term) <= Number.EPSILON * Math.abs(sum)) {
        const logP = logScale + Math.log(sum);
        return upper ? -Math.expm1(logP) : Math.exp(logP);
      }
    }
  } else {
    const tiny = 1e-300;
    let b = x + 1 - a, c = 1 / tiny, d = 1 / b, h = d;
    for (let n = 1; n < 1000000; n++) {
      const an = -n * (n - a); b += 2;
      d = an * d + b; if (Math.abs(d) < tiny) d = Math.sign(d || 1) * tiny;
      c = b + an / c; if (Math.abs(c) < tiny) c = Math.sign(c || 1) * tiny;
      d = 1 / d;
      const delta = d * c; h *= delta;
      if (Math.abs(delta - 1) <= 4 * Number.EPSILON) {
        const logQ = logScale + Math.log(h);
        return upper ? Math.exp(logQ) : -Math.expm1(logQ);
      }
    }
  }
  throw new Error('Regularized gamma did not converge');
}

/** Invert P/Q with a library seed and safeguarded Newton refinement. The
 * smaller probability tail is evaluated directly, including upper tails far
 * below epsilon. Refinement uses the same stable prefactor as the CDF. */
function inverseRegularizedGamma(a: number, p: number, upper = false): number {
  if (!(a > 0) || !(p >= 0 && p <= 1)) return NaN;
  if (p === 0) return upper ? Infinity : 0;
  if (p === 1) return upper ? 0 : Infinity;
  if (p > 0.5) { p = 1 - p; upper = !upper; }
  let x = require('@stdlib/math-base-special-gammaincinv')(p, a, upper);
  if (x === 0) return 0; // the inverse is below the representable range
  if (!(x > 0) || !Number.isFinite(x)) x = Math.max(a, 1);
  let lo = 0, hi = Math.max(x, a, 1);
  const below = (v: number) => upper ? v > p : v < p;
  while (below(regularizedGamma(a, hi, upper))) hi *= 2;
  for (let i = 0; i < 256; i++) {
    const residual = regularizedGamma(a, x, upper) - p;
    if (Math.abs(residual) <= 8 * Number.EPSILON * p) return x;
    if (upper ? residual > 0 : residual < 0) lo = x; else hi = x;
    const density = Math.exp(prefactor(a, x) - Math.log(x));
    const next = x - residual / (upper ? -density : density);
    // In a far tail one ulp of a large x can change the probability by much
    // more than one ulp of p. Stop at x precision before rejecting the step.
    if (Math.abs(next - x) <= 4 * Number.EPSILON * x) return next;
    const candidate = next > lo && next < hi ? next : lo / 2 + hi / 2;
    if (candidate === x || candidate === lo || candidate === hi) return candidate;
    x = candidate;
  }
  throw new Error('Inverse regularized gamma did not converge');
}

module.exports = { regularizedGamma, inverseRegularizedGamma, log1pmx, stirlingRemainder };
