'use strict';
const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');

function postDeriv(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') return v;
  }
  return null;
}

// G=2, N=2 matrix beta-binomial (matches the litter model's matrix-latent shape).
const SRC = `
a_plus_b ~ iid(Gamma(2, 1), 2)
mu ~ iid(Beta(1, 1), 2)
a = mu .* a_plus_b
b = (1 .- mu) .* a_plus_b
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), 2)
p ~ beta_row_K.(a, b)
binom_row = (n_row, p_row) -> Binomial.(n_row, p_row)
r ~ binom_row.([[10, 10],[10, 10]], p)
prior = lawof(record(a_plus_b = a_plus_b, mu = mu, p = p))
forward_kernel = kernelof(record(r = r), a_plus_b = a_plus_b, mu = mu, p = p)
L = likelihoodof(forward_kernel, record(r = [[7, 5],[8, 6]]))
posterior = bayesupdate(L, prior)
`;

test('logPiBatch equals scalar logPi for a [G,N] matrix latent (batched path keeps rank)', async () => {
  const { ctx } = ctxFor(SRC, 1);
  const { logPi, logPiBatch } = await buildLogPi(ctx, postDeriv(ctx));
  // Two distinct in-support points; p as a [2,2] Value (matrix-latent scorer form).
  const pts = [
    { a_plus_b: Float64Array.from([5, 5]), mu: Float64Array.from([0.5, 0.5]),
      p: { shape: [2, 2], data: Float64Array.from([0.7, 0.5, 0.8, 0.6]) } },
    { a_plus_b: Float64Array.from([3, 8]), mu: Float64Array.from([0.4, 0.7]),
      p: { shape: [2, 2], data: Float64Array.from([0.6, 0.55, 0.9, 0.5]) } },
  ];
  const scalar = pts.map((p) => logPi(p));
  const batched = logPiBatch(pts);
  for (let i = 0; i < pts.length; i++) {
    assert.ok(Number.isFinite(scalar[i]), `scalar logPi[${i}] finite: ${scalar[i]}`);
    assert.ok(Math.abs(batched[i] - scalar[i]) <= 1e-9,
      `batched[${i}]=${batched[i]} vs scalar=${scalar[i]}`);
  }
});
