'use strict';
// mixture-normalize-scorer.test.ts — the MCMC/SMC scorer must resolve a
// `normalize(...)` in the LIKELIHOOD body, exactly as the IS path (matScore)
// does. A Gaussian-mixture likelihood
//   mix = normalize(superpose(weighted(theta, Normal(mu[1],1)),
//                             weighted(1-theta, Normal(mu[2],1))))
//   y ~ iid(mix, N)
// lowers to a normalize node carrying massFrom {ref:'__anon…'}. buildLogPi
// resolved massFrom only on the PRIOR draw measures, never on likBodyIR, so the
// node reached walkNormalize unresolved → "normalize with unresolved totalmass"
// → swallowed to −∞. Symptom: every score-the-prior sampler (mh/emcee/smc/amis/
// ess) sees logπ = −∞ everywhere → stuck at init; SMC dies at β=0 because the
// tempered target prior + β·lik computes 0·(−∞) = NaN → "all weights vanished".
// The IS default forward-samples, so it was unaffected.
//
// Oracle (Distributions.jl, independent): at theta=0.5, mu=[1.5,-0.5] the
// two-component mixture has weights summing to 1 ⇒ Z=1 (normalize is a no-op),
// so lik = sum(logpdf(MixtureModel([N(1.5,1),N(-0.5,1)],[.5,.5]), y_data))
//        = -37.69815469151789.

const { test }       = require('node:test');
const assert         = require('node:assert/strict');
const { ctxFor }     = require('./density/regression-baseline.test.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const TOL = 1e-9;
const SRC = `
N = 20
y_data = [1.31, 2.25, -1.79, 2.30, 2.53, -1.49, 1.87, 1.55, -1.34, 2.18, 1.28, -3.34, 0.56, 1.37, 0.43, -0.84, -0.54, 1.50, 1.71, 1.33]
theta ~ Beta(1, 1)
mu ~ iid(Normal(0, 5), 2)
mix = normalize(superpose(weighted(theta, Normal(mu[1], 1)), weighted(1 - theta, Normal(mu[2], 1))))
y ~ iid(mix, N)
prior = lawof(record(theta = theta, mu = mu))
forward_kernel = kernelof(record(y = y), theta = theta, mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

function postDeriv(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations as Record<string, any>)) {
    if (v && (v as any).kind === 'bayesupdate') return v;
  }
  return null;
}

test('scorer resolves a normalize() in the likelihood body to the Distributions.jl oracle (1e-9)', async () => {
  const { ctx } = ctxFor(SRC, 1);
  const d = postDeriv(ctx);
  const { logPi, priorOf, likOf } = await buildLogPi(ctx, d);
  const pt = { theta: 0.5, mu: Float64Array.from([1.5, -0.5]) };
  const lik = likOf(pt);
  assert.ok(Math.abs(lik - (-37.69815469151789)) <= TOL, `lik: got ${lik}, oracle -37.69815469151789`);
  assert.ok(Number.isFinite(priorOf(pt)), `prior finite: ${priorOf(pt)}`);
  assert.ok(Number.isFinite(logPi(pt)), `logPi finite: ${logPi(pt)}`);
});

// The normalize totalmass must be evaluated at the SCORED θ, not baked as a
// constant. Weights w1=w2=theta do NOT sum to 1, so Z(θ)=2θ and the normalized
// density is (N1+N2)/2 — INDEPENDENT of θ. A constant-Z bake leaves a spurious
// +N·log θ; the per-θ −log Z(θ) expression cancels it.
// Oracle (Distributions.jl): lik = sum logpdf(MixtureModel([N(.5,1),N(-.5,1)],
// [.5,.5]), [0.1,-0.2,0.3,1.0,-1.0]) = -6.032014419862955, ∀θ.
const SRC_ZDEP = `
N = 5
y_data = [0.1, -0.2, 0.3, 1.0, -1.0]
theta ~ Beta(2, 2)
mu ~ iid(Normal(0, 5), 2)
mix = normalize(superpose(weighted(theta, Normal(mu[1], 1)), weighted(theta, Normal(mu[2], 1))))
y ~ iid(mix, N)
prior = lawof(record(theta = theta, mu = mu))
forward_kernel = kernelof(record(y = y), theta = theta, mu = mu)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

test('normalize totalmass is evaluated per-θ (Z=2θ mixture is θ-invariant, matches oracle)', async () => {
  const { ctx } = ctxFor(SRC_ZDEP, 1);
  const { likOf } = await buildLogPi(ctx, postDeriv(ctx));
  const mu = Float64Array.from([0.5, -0.5]);
  const liks = [0.2, 0.5, 0.8].map((theta) => likOf({ theta, mu }));
  for (const lik of liks) {
    assert.ok(Math.abs(lik - (-6.032014419862955)) <= TOL, `lik: got ${lik}, oracle -6.032014419862955`);
  }
  // θ-invariance: all three identical (the constant-Z bug made them differ by N·log(θ ratio)).
  assert.ok(Math.abs(liks[0] - liks[2]) <= TOL, `θ-invariant: ${liks.join(', ')}`);
});
