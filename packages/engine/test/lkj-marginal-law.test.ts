'use strict';
// EACH OFF-DIAGONAL CORRELATION OF LKJ(n, eta) FOLLOWS A SCALED BETA LAW.
//
// For R ~ LKJ(n, eta) every off-diagonal entry has the same marginal:
// (R_ij + 1) / 2 ~ Beta(p, p) with p = eta - 1 + n / 2 (Lewandowski, Kurowicka
// and Joe 2009, eq. 16). So R_ij has mean 0, variance 1 / (2p + 1) and CDF
// F(x) = I_{(x+1)/2}(p, p).
//
// This pins the sampler against that closed form. A check on the sampler's own
// row-radius law cannot catch a wrong radius law, because it reuses the Beta
// shapes the sampler draws from. The marginal does not depend on them.
//
// Two statistics per entry, each with a bound that a draw from the true law
// passes with overwhelming probability at the fixed seed:
//   - Kolmogorov-Smirnov distance against the closed-form CDF, bounded by
//     1.95 / sqrt(N) (the 0.1 % critical value).
//   - Sample variance against 1 / (2p + 1), bounded by six standard errors.
//     Beta(p, p) has excess kurtosis -6 / (2p + 3), which fixes the fourth
//     moment and so the standard error of the variance.
// A sampler that draws the row radii from Beta(eta + (n-i-1)/2, i/2) instead of
// Beta(i/2, eta + (n-i-1)/2) moves both statistics far outside these bounds.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 100000;

// ln Γ, Lanczos approximation (g = 5, n = 6), accurate to about 2e-10.
function lnGamma(x: number): number {
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let y = x;
  let t = x + 5.5;
  t -= (x + 0.5) * Math.log(t);
  let s = 1.000000000190015;
  for (const ci of c) s += ci / ++y;
  return -t + Math.log(2.5066282746310005 * s / x);
}

// Regularised incomplete beta I_x(a, b), Lentz continued fraction.
function incBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const front = Math.exp(lnGamma(a + b) - lnGamma(a) - lnGamma(b)
    + a * Math.log(x) + b * Math.log1p(-x));
  const cf = (aa: number, bb: number, xx: number) => {
    const tiny = 1e-300;
    let c = 1;
    let d = 1 - (aa + bb) * xx / (aa + 1);
    if (Math.abs(d) < tiny) d = tiny;
    d = 1 / d;
    let h = d;
    for (let m = 1; m < 400; m++) {
      const m2 = 2 * m;
      let k = m * (bb - m) * xx / ((aa + m2 - 1) * (aa + m2));
      d = 1 + k * d; if (Math.abs(d) < tiny) d = tiny;
      c = 1 + k / c; if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d; h *= d * c;
      k = -(aa + m) * (aa + bb + m) * xx / ((aa + m2) * (aa + m2 + 1));
      d = 1 + k * d; if (Math.abs(d) < tiny) d = tiny;
      c = 1 + k / c; if (Math.abs(c) < tiny) c = tiny;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < 3e-16) break;
    }
    return h;
  };
  return x < (a + 1) / (a + b + 2)
    ? front * cf(a, b, x) / a
    : 1 - front * cf(b, a, 1 - x) / b;
}

test('the incomplete beta helper matches known values', () => {
  // Beta(1, 1) is uniform; Beta(a, b) is symmetric about 1/2 when a = b;
  // I_x(1, 2) = 1 - (1 - x)^2 and I_x(2, 1) = x^2.
  assert.ok(Math.abs(incBeta(0.3, 1, 1) - 0.3) < 1e-12);
  assert.ok(Math.abs(incBeta(0.5, 2.5, 2.5) - 0.5) < 1e-12);
  assert.ok(Math.abs(incBeta(0.3, 1, 2) - (1 - 0.7 ** 2)) < 1e-12);
  assert.ok(Math.abs(incBeta(0.3, 2, 1) - 0.09) < 1e-12);
});

function checkMarginals(R: (atom: number, i: number, j: number) => number,
                        n: number, eta: number, label: string) {
  const p = eta - 1 + n / 2;
  const variance = 1 / (2 * p + 1);
  const kurtosis = 3 - 6 / (2 * p + 3);
  const seVariance = Math.sqrt((kurtosis - 1) * variance * variance / N);
  const ksBound = 1.95 / Math.sqrt(N);
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < i; j++) {
      const x = new Float64Array(N);
      let mean = 0;
      for (let a = 0; a < N; a++) { x[a] = R(a, i, j); mean += x[a]; }
      mean /= N;
      let ss = 0;
      for (let a = 0; a < N; a++) ss += (x[a] - mean) ** 2;
      const sampleVariance = ss / (N - 1);
      const sorted = Float64Array.from(x).sort();
      let ks = 0;
      for (let a = 0; a < N; a++) {
        const F = incBeta((sorted[a] + 1) / 2, p, p);
        ks = Math.max(ks, Math.abs(F - a / N), Math.abs(F - (a + 1) / N));
      }
      const where = `${label} n=${n} eta=${eta} entry (${i},${j})`;
      assert.ok(ks < ksBound, `${where}: KS ${ks} exceeds ${ksBound}`);
      assert.ok(Math.abs(sampleVariance - variance) < 6 * seVariance,
        `${where}: variance ${sampleVariance}, expected ${variance} ± ${6 * seVariance}`);
      assert.ok(Math.abs(mean) < 6 * Math.sqrt(variance / N),
        `${where}: mean ${mean}, expected 0`);
    }
  }
}

const CASES: Array<[number, number]> = [[2, 1], [3, 2], [4, 1.5], [5, 0.7]];

for (const [n, eta] of CASES) {
  test(`LKJ(${n}, ${eta}): off-diagonal correlations follow the scaled Beta marginal`, async () => {
    const { ctx } = ctxFor(`m = LKJ(${n}, ${eta})`, N);
    const s = (await ctx.getMeasure('m')).samples;
    checkMarginals((a, i, j) => s[a * n * n + i * n + j], n, eta, 'LKJ');
  });
}

test('LKJCholesky(4, 1.5): L Lᵀ has the same scaled Beta marginals', async () => {
  const n = 4;
  const eta = 1.5;
  const { ctx } = ctxFor(`m = LKJCholesky(${n}, ${eta})`, N);
  const L = (await ctx.getMeasure('m')).samples;
  const R = (a: number, i: number, j: number) => {
    let s = 0;
    for (let k = 0; k <= j; k++) s += L[a * n * n + i * n + k] * L[a * n * n + j * n + k];
    return s;
  };
  checkMarginals(R, n, eta, 'LKJCholesky');
});
