'use strict';

// Regularized beta, DLMF 8.17.22. The continued fraction selects the nearer
// endpoint. Shift small shapes before the Stirling prefactor so subtraction
// of large log-gamma values does not bias central probabilities.
const { log1pmx, log1mexp, logGamma1pOverA, stirlingRemainder } = require('./incomplete-gamma.ts');
const digamma = require('@stdlib/math-base-special-digamma');
const polygamma = require('@stdlib/math-base-special-polygamma');

function log1pRatio(numerator: number, denominator: number): number {
  return numerator <= denominator ? Math.log1p(numerator / denominator)
    : Math.log(numerator) - Math.log(denominator) + Math.log1p(denominator / numerator);
}

/** Positive-argument beta via a shifted Stirling expansion (DLMF 5.11). */
function logBeta(a: number, b: number): number {
  if (!(a > 0) || !(b > 0)) return NaN;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return -Infinity;
  let aa = Math.min(a, b), bb = Math.max(a, b), shifts = 0;
  while (aa < 8) { shifts += log1pRatio(bb, aa); aa += 1; }
  while (bb < 8) { shifts += log1pRatio(aa, bb); bb += 1; }
  const ratio = aa / bb, logSum = Math.log(bb) + Math.log1p(ratio);
  return aa * (Math.log(ratio) - Math.log1p(ratio)) - bb * Math.log1p(ratio)
    + (Math.log(2 * Math.PI) + logSum - Math.log(aa) - Math.log(bb)) / 2
    + stirlingRemainder(aa) + stirlingRemainder(bb) - stirlingRemainder(aa + bb) + shifts;
}

function logPrefactor(a: number, b: number, x: number): number {
  let aa = a, bb = b, shifts = 0;
  const logx = Math.log(x), logy = Math.log1p(-x);
  while (aa < 8) { shifts += Math.log(aa + bb) - Math.log(aa) + logx; aa += 1; }
  while (bb < 8) { shifts += Math.log(aa + bb) - Math.log(bb) + logy; bb += 1; }
  const sum = aa + bb, mean = aa / sum, deviation = x - mean;
  const central = Math.abs(deviation / mean) < 0.125 && Math.abs(deviation / (1 - mean)) < 0.125;
  const powers = central ? aa * log1pmx(deviation / mean) + bb * log1pmx(-deviation / (1 - mean))
    : aa * (logx - Math.log(mean)) + bb * (logy - Math.log1p(-mean));
  return powers
    + (Math.log(aa) + Math.log(bb) - Math.log(sum) - Math.log(2 * Math.PI)) / 2
    - stirlingRemainder(aa) - stirlingRemainder(bb) + stirlingRemainder(sum) - shifts;
}

function fraction(a: number, b: number, x: number): number {
  const tiny = 1e-300, sum = a + b;
  let c = 1, d = (a + 1 - sum * x) / (a + 1);
  if (Math.abs(d) < tiny) d = Math.sign(d || 1) * tiny;
  d = 1 / d;
  let h = d;
  for (let m = 1; m < 1000000; m++) {
    const even = (m / (a + 2 * m - 1)) * ((b - m) / (a + 2 * m)) * x;
    d = 1 + even * d; if (Math.abs(d) < tiny) d = Math.sign(d || 1) * tiny;
    c = 1 + even / c; if (Math.abs(c) < tiny) c = Math.sign(c || 1) * tiny;
    d = 1 / d; h *= d * c;
    const odd = -((a + m) / (a + 2 * m)) * ((sum + m) / (a + 2 * m + 1)) * x;
    d = 1 + odd * d; if (Math.abs(d) < tiny) d = Math.sign(d || 1) * tiny;
    c = 1 + odd / c; if (Math.abs(c) < tiny) c = Math.sign(c || 1) * tiny;
    d = 1 / d;
    const delta = d * c; h *= delta;
    if (Math.abs(delta - 1) <= Number.EPSILON) return h;
  }
  throw new Error('Regularized beta did not converge');
}

/** DLMF 8.17.7, with the leading one removed before summation. For tiny a,
 * log P is O(a): subtracting O(log a) prefactors loses a small upper tail.
 * Taylor-expand log Γ(b+a)-log Γ(b) through sixth order. Shifting b<1
 * first avoids overflowing polygamma near zero; a<0.01 bounds
 * the omitted log-gamma term below 2e-15. */
function smallShapeLogTail(a: number, b: number, x: number, upper: boolean): number {
  const shifted = b < 1 ? b + 1 : b;
  const difference = digamma(shifted) + a * (polygamma(1, shifted) / 2
    + a * (polygamma(2, shifted) / 6 + a * (polygamma(3, shifted) / 24
      + a * (polygamma(4, shifted) / 120 + a * polygamma(5, shifted) / 720))));
  const logRatio = b >= 1 ? 0 : log1pRatio(a, b);
  const normalization = logGamma1pOverA(a) - difference;
  let term = 1, sum = 0;
  for (let n = 1; n < 10000; n++) {
    term *= ((n - b) * x) / n;
    const add = term / (a + n); sum += add;
    if (Math.abs(add) <= Number.EPSILON * Math.abs(sum) || term === 0) {
      const z = a * sum;
      const base = Math.log(x) - normalization + sum * (z === 0 ? 1 : Math.log1p(z) / z);
      const logP = a * base - logRatio;
      if (!upper) return logP;
      // Form Q/a before exponentiation, so its log survives subnormal a.
      const coefficient = base - logRatio / a;
      return logP < -Math.LN2 || !Number.isFinite(coefficient) ? log1mexp(logP)
        : Math.log(a) + Math.log(-coefficient) + Math.log(logP === 0 ? 1 : Math.expm1(logP) / logP);
    }
  }
  throw new Error('Small-shape regularized beta did not converge');
}

function logRegularizedBeta(a: number, b: number, x: number, upper = false): number {
  if (!(a > 0) || !(b > 0) || !(x >= 0 && x <= 1)) return NaN;
  if (x === 0) return upper ? 0 : -Infinity;
  if (x === 1) return upper ? -Infinity : 0;
  const reflected = x > (a + 1) / (a + b + 2);
  const aa = reflected ? b : a, bb = reflected ? a : b, xx = reflected ? 1 - x : x;
  if (aa < 0.01 && xx * bb <= 1) {
    return smallShapeLogTail(aa, bb, xx, upper !== reflected);
  }
  const logValue = logPrefactor(aa, bb, xx) + Math.log(fraction(aa, bb, xx)) - Math.log(aa);
  return upper === reflected ? logValue : log1mexp(logValue);
}

function regularizedBeta(a: number, b: number, x: number, upper = false): number {
  return Math.exp(logRegularizedBeta(a, b, x, upper));
}

module.exports = { regularizedBeta, logRegularizedBeta, logBeta };
