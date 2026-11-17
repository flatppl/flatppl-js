'use strict';

// Regularized beta, DLMF 8.17.22. The continued fraction selects the nearer
// endpoint. Shift small shapes before the Stirling prefactor so subtraction
// of large log-gamma values does not bias central probabilities.
const { log1pmx, stirlingRemainder } = require('./incomplete-gamma.ts');

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

function regularizedBeta(a: number, b: number, x: number, upper = false): number {
  if (!(a > 0) || !(b > 0) || !(x >= 0 && x <= 1)) return NaN;
  if (x === 0) return upper ? 1 : 0;
  if (x === 1) return upper ? 0 : 1;
  const reflected = x > (a + 1) / (a + b + 2);
  const aa = reflected ? b : a, bb = reflected ? a : b, xx = reflected ? 1 - x : x;
  const logValue = logPrefactor(aa, bb, xx) + Math.log(fraction(aa, bb, xx)) - Math.log(aa);
  return upper === reflected ? Math.exp(logValue) : -Math.expm1(logValue);
}

module.exports = { regularizedBeta };
