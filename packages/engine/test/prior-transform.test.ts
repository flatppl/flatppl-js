'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildPriorTransform } = require('../prior-transform.ts');

// Independent prior: mu ~ Normal(1,2), rate ~ Exponential(0.5). No likelihood
// dependence needed for the prior transform, but the model needs a posterior
// derivation to enumerate latents — use a trivial observed likelihood.
const SRC = `
flatppl_compat = "0.1"
mu ~ Normal(1.0, 2.0)
lam ~ Exponential(0.5)
prior = lawof(record(mu = mu, lam = lam))
y ~ Normal.(mu, lam)
K = kernelof(record(y = y), mu = mu, lam = lam)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;

function ks(a: number[], b: number[]): number {           // two-sample KS statistic
  const A = a.slice().sort((x, y) => x - y), B = b.slice().sort((x, y) => x - y);
  const all = A.concat(B).sort((x, y) => x - y);
  let d = 0;
  for (const v of all) {
    const fa = A.filter((x) => x <= v).length / A.length;
    const fb = B.filter((x) => x <= v).length / B.length;
    d = Math.max(d, Math.abs(fa - fb));
  }
  return d;
}

test('prior transform: scalar independent latents match forward draws (KS)', () => {
  const { ctx } = ctxFor(SRC, 100);
  const d = ctx.derivations['posterior'];
  const pt = buildPriorTransform(ctx, d);
  assert.equal(pt.dim, 2);
  assert.deepEqual(pt.latentNames, ['mu', 'lam']);
  // Deterministic quasi-forward draws through T over a stratified grid on [0,1]^2.
  const N = 400;
  const muT: number[] = [], lamT: number[] = [];
  for (let i = 0; i < N; i++) {
    const u = new Float64Array([ (i + 0.5) / N, ((i * 7 + 3) % N + 0.5) / N ]);
    const rec = pt.transform(u);
    muT.push(rec.mu); lamT.push(rec.lam);
  }
  // Analytic forward draws via the same quantile ladder from independent U — the
  // oracle here is the closed-form prior CDF, realised as a reference sample.
  const muRef: number[] = [], lamRef: number[] = [];
  for (let i = 0; i < N; i++) {
    const p = (i + 0.5) / N;
    muRef.push(1.0 + 2.0 * Math.SQRT2 * require('@stdlib/math-base-special-erfinv')(2 * p - 1)); // Normal(1,2) ppf
    lamRef.push(-Math.log1p(-p) / 0.5);                                                          // Exp(0.5) ppf
  }
  assert.ok(ks(muT, muRef) < 0.1, `mu KS ${ks(muT, muRef)}`);
  assert.ok(ks(lamT, lamRef) < 0.1, `lam KS ${ks(lamT, lamRef)}`);
});
