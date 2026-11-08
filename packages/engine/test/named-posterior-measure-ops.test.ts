// `weighted(2.0, post)` and `normalize(post)` over a NAMED posterior
// (`post = bayesupdate(L, prior)`). The analyzer's isMeasureExpr knew
// `lawof` and measure-typed `call` bindings only, so these classified as
// nothing and failed "no derivation for 'M'". Oracle: the conjugate
// posterior N(0.25, 1/2) for a N(0, 1) prior and one N(mu, 1) observation
// at 0.5, and the evidence Z = φ(0.5; 0, √2).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const BASE = 'mu ~ Normal(0.0, 1.0)\ny ~ Normal(mu, 1.0)\nprior = lawof(record(mu = mu))\n'
  + 'K = kernelof(record(y = y), mu = mu)\nL = likelihoodof(K, record(y = 0.5))\n'
  + 'post = bayesupdate(L, prior)\n';
const N = 40000;

async function moments(expr: string) {
  const { ctx } = makeMatCtx(BASE + `M = ${expr}\n`, { sampleCount: N });
  const m = await ctx.getMeasure('M');
  const s = m.fields.mu.samples, lw = m.logWeights;
  let W = 0, S = 0, Q = 0;
  for (let i = 0; i < s.length; i++) { const w = Math.exp(lw[i]); W += w; S += w * s[i]; Q += w * s[i] * s[i]; }
  return { mean: S / W, variance: Q / W - (S / W) ** 2, logTotalmass: m.logTotalmass };
}

test('measure operations over a named bayesupdate binding', async () => {
  const logZ = -0.5 * Math.log(4 * Math.PI) - 0.25 ** 2;   // log φ(0.5; 0, √2)
  const w = await moments('weighted(2.0, post)');
  const n = await moments('normalize(post)');
  for (const r of [w, n]) {
    // Importance weights over 40000 prior draws: ESS ~ 0.7 N, so SE(mean) ~ 0.004.
    assert.ok(Math.abs(r.mean - 0.25) < 0.03, `mean ${r.mean}`);
    assert.ok(Math.abs(r.variance - 0.5) < 0.04, `variance ${r.variance}`);
  }
  assert.ok(Math.abs(n.logTotalmass) < 1e-12, 'normalize has unit mass');
  assert.ok(Math.abs(w.logTotalmass - (Math.log(2) + logZ)) < 0.03, `log mass ${w.logTotalmass}`);
});
