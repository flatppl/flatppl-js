'use strict';
// smc-sample.test.ts — the SMC driver recovers a known correlated 2-D Gaussian
// AND its log normalizing constant. Target N(μ*, Σ*); prior' = a broad N(0,10²I)
// (β=0, directly sampleable); lik = logTarget − prior' so that prior'+lik =
// logTarget (a normalised density). The adaptive ladder must reach β=1, the
// final equal-weight particles must recover μ*/Σ*, and logZ must be ≈ 0 (since
// ∫ target = 1) — validating the evidence accumulation analytically.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const { smcSample } = require('../smc-sample.ts');

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
const s0 = 10;
function logPrior(y: Float64Array): number {
  return -2 * Math.log(s0) - Math.log(2 * Math.PI) - 0.5 * ((y[0] * y[0] + y[1] * y[1]) / (s0 * s0));
}
const mv: any = {
  dim: 2, names: ['a', 'b'],
  logPriorLikBatch(ys: Float64Array[]) {
    const n = ys.length; const prior = new Float64Array(n); const lik = new Float64Array(n);
    for (let i = 0; i < n; i++) { prior[i] = logPrior(ys[i]); lik[i] = logTarget(ys[i]) - prior[i]; }
    return { prior, lik };
  },
  constrainAll(y: Float64Array) { return { a: y[0], b: y[1] }; },
  initFromPrior(n: number, prng: () => number) {
    const out: Float64Array[] = [];
    for (let i = 0; i < n; i++) {
      const y = new Float64Array(2);
      y[0] = s0 * Math.sqrt(-2 * Math.log(Math.max(prng(), 1e-300))) * Math.cos(2 * Math.PI * prng());
      y[1] = s0 * Math.sqrt(-2 * Math.log(Math.max(prng(), 1e-300))) * Math.cos(2 * Math.PI * prng());
      out.push(y);
    }
    return out;
  },
};

test('SMC recovers a correlated Gaussian and its evidence (logZ≈0)', () => {
  const res = smcSample(mv, { smcParticles: 2000, seed: 5 });
  assert.ok(res.betas[res.betas.length - 1] === 1, 'ladder reaches β=1');
  assert.ok(res.rungs >= 2, `multiple rungs (got ${res.rungs})`);

  const X = res.samples; const n = X.length;
  const m = [0, 0]; for (const x of X) { m[0] += x[0]; m[1] += x[1]; } m[0] /= n; m[1] /= n;
  let v0 = 0, v1 = 0, c = 0;
  for (const x of X) { v0 += (x[0] - m[0]) ** 2; v1 += (x[1] - m[1]) ** 2; c += (x[0] - m[0]) * (x[1] - m[1]); }
  v0 /= n; v1 /= n; c /= n;

  assert.ok(Math.abs(m[0] - 3) < 0.25, `mean[0] ${m[0]} ≈ 3`);
  assert.ok(Math.abs(m[1] + 2) < 0.25, `mean[1] ${m[1]} ≈ -2`);
  assert.ok(Math.abs(v0 - 2) < 0.5, `var[0] ${v0} ≈ 2`);
  assert.ok(Math.abs(v1 - 1) < 0.35, `var[1] ${v1} ≈ 1`);
  assert.ok(Math.abs(c - 1.2) < 0.4, `cov ${c} ≈ 1.2`);
  assert.ok(Math.abs(res.logZ) < 0.3, `logZ ${res.logZ} ≈ 0 (∫target=1)`);
});
