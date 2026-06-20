'use strict';
// elliptical-slice.test.ts — the elliptical-slice kernel (via runMcmc) recovers
// a known correlated 2-D Gaussian under BOTH reference modes: an exact Gaussian
// reference (mv.gaussianPrior set) and a fitted one (null → fit from the prior
// pool). The ellipse runs against the reference; the slice on logπ − log g
// corrects to the target either way.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const { runMcmc } = require('../mcmc-driver.ts');
const { makeEllipticalSliceKernel } = require('../elliptical-slice-kernel.ts');

const muStar = [3, -2];
const S = [[2, 1.2], [1.2, 1]];
const det = S[0][0] * S[1][1] - S[0][1] * S[1][0];
const Pm = [[S[1][1] / det, -S[0][1] / det], [-S[1][0] / det, S[0][0] / det]];
const logdet = Math.log(det);
function logTarget(y: Float64Array): number {
  const d0 = y[0] - muStar[0], d1 = y[1] - muStar[1];
  const q = d0 * (Pm[0][0] * d0 + Pm[0][1] * d1) + d1 * (Pm[1][0] * d0 + Pm[1][1] * d1);
  return -0.5 * (2 * Math.log(2 * Math.PI) + logdet + q);
}
const s0 = 8;
function makeMv(withGaussianPrior: boolean): any {
  return {
    dim: 2, names: ['a', 'b'],
    logPosterior: (y: Float64Array) => logTarget(y),
    logPosteriorBatch: (ys: Float64Array[]) => { const o = new Float64Array(ys.length); for (let i = 0; i < ys.length; i++) o[i] = logTarget(ys[i]); return o; },
    constrainAll: (y: Float64Array) => ({ a: y[0], b: y[1] }),
    initFromPrior: (n: number, prng: () => number) => {
      const out: Float64Array[] = [];
      for (let i = 0; i < n; i++) { const y = new Float64Array(2); y[0] = s0 * Math.sqrt(-2 * Math.log(Math.max(prng(), 1e-300))) * Math.cos(2 * Math.PI * prng()); y[1] = s0 * Math.sqrt(-2 * Math.log(Math.max(prng(), 1e-300))) * Math.cos(2 * Math.PI * prng()); out.push(y); }
      return out;
    },
    gaussianPrior: withGaussianPrior ? { mu: new Float64Array([0, 0]), sigma: new Float64Array([s0, s0]) } : null,
  };
}

for (const exact of [true, false]) {
  test(`elliptical slice recovers a correlated Gaussian (${exact ? 'exact' : 'fitted'} reference)`, () => {
    const mv = makeMv(exact);
    const post = runMcmc(mv, makeEllipticalSliceKernel(), { nWalkers: 8, warmup: 600, draws: 1200, seed: 4 });
    const a = post.drawsByName['a'], b = post.drawsByName['b']; const n = a.length;
    let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
    let va = 0, vb = 0, c = 0; for (let i = 0; i < n; i++) { va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; c += (a[i] - ma) * (b[i] - mb); } va /= n; vb /= n; c /= n;
    assert.ok(Math.abs(ma - 3) < 0.25, `mean[a] ${ma} ≈ 3`);
    assert.ok(Math.abs(mb + 2) < 0.25, `mean[b] ${mb} ≈ -2`);
    assert.ok(Math.abs(va - 2) < 0.5, `var[a] ${va} ≈ 2`);
    assert.ok(Math.abs(vb - 1) < 0.35, `var[b] ${vb} ≈ 1`);
    assert.ok(Math.abs(c - 1.2) < 0.4, `cov ${c} ≈ 1.2`);
  });
}
