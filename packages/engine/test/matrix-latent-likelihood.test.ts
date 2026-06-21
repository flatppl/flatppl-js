'use strict';
// matrix-latent-likelihood.test.ts — a [G,N] MATRIX latent (e.g. the litter
// Beta-Binomial's `p ~ beta_row_K.(a,b)`) must reach the likelihood density
// with its full 2-axis shape. The scorer's pointToRefArrays turned a Value
// {shape:[G,N],data} into the scalar `+Value = NaN` fallback → a 1-axis NaN
// refArray → the likelihood's `broadcast(Binomial, n[G,N], p)` raised
// "all collection arguments must have the same number of axes; arg 'p' has 1,
// expected 2" → swallowed to −∞ → logπ ≡ −∞ (samplers wander the prior; SMC's
// 0·(−∞)=NaN at β=0 wipes all weights). pointToRefArrays now routes a rank≥2
// Value through the env (shape preserved), matching buildEnv.
//
// Oracle (Distributions.jl, independent): the matrix Binomial likelihood at
// p=[[0.7,0.5],[0.8,0.6]], n=[[10,10],[10,10]], r=[[7,5],[8,6]] is
// sum logpdf(Binomial(n,p), r) = -5.303564880841569.

const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const modelSpec      = require('../model-spec.ts');

const TOL = 1e-9;

function postDeriv(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') return v;
  }
  return null;
}

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

test('a [G,N] matrix latent scores its likelihood to the Distributions.jl oracle (1e-9)', async () => {
  const { ctx } = ctxFor(SRC, 1);
  const { likOf, logPi } = await buildLogPi(ctx, postDeriv(ctx));
  const pt = {
    a_plus_b: Float64Array.from([5, 5]),
    mu: Float64Array.from([0.5, 0.5]),
    p: { shape: [2, 2], data: Float64Array.from([0.7, 0.5, 0.8, 0.6]) },
  };
  const lik = likOf(pt);
  assert.ok(Math.abs(lik - (-5.303564880841569)) <= TOL, `matrix binomial lik: got ${lik}, oracle -5.303564880841569`);
  assert.ok(Number.isFinite(logPi(pt)), `logPi finite: ${logPi(pt)}`);
});

test('a pushfwd latent gets the image support (Pareto [0.1, ∞) → greaterThan), not real', () => {
  const { ctx } = ctxFor(`
pareto = pushfwd(fn(0.1 * exp(_)), Exponential(1.5))
a_plus_b ~ iid(pareto, 2)
mu ~ iid(Beta(1, 1), 2)
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), 3)
p ~ beta_row_K.(a_plus_b, mu)
binom_row = (n_row, p_row) -> Binomial.(n_row, p_row)
r ~ binom_row.([[5,5,5],[5,5,5]], p)
prior = lawof(record(a_plus_b = a_plus_b, mu = mu, p = p))
forward_kernel = kernelof(record(r = r), a_plus_b = a_plus_b, mu = mu, p = p)
L = likelihoodof(forward_kernel, record(r = [[3,2,4],[1,5,0]]))
posterior = bayesupdate(L, prior)
`, 1);
  const latents = modelSpec.enumerateLatents(postDeriv(ctx), ctx);
  const apb = latents.find((l: any) => l.name === 'a_plus_b');
  assert.equal(apb.support.kind, 'greaterThan', `support: ${JSON.stringify(apb.support)}`);
  assert.ok(Math.abs(apb.support.lo - 0.1) < 1e-12, `lower bound 0.1: ${apb.support.lo}`);
});
