// logdensityof of a measure built over a NAMED posterior: `weighted(2.0,
// post)`, `logweighted(c, post)`, `superpose(post, prior)`. Only a bare
// `logdensityof(post, θ)` scored (the posterior_density derivation); these
// failed "no derivation" (or, once classified, "density: unsupported measure op
// 'bayesupdate'"). Spec §06: weighted(w, M) has density w·p_M, logweighted(g, M)
// has exp(g)·p_M, superpose(M₁, M₂) has p_M₁ + p_M₂, and bayesupdate(L, prior)
// has L·p_prior.
// Oracle: the conjugate closed form. Prior N(0, 1), one N(mu, 1) observation
// at 0.5: L(θ)·p(θ) = Z·N(θ; 0.25, 1/2) with evidence Z = φ(0.5; 0, √2).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lnN = (x: number, m: number, v: number) => -0.5 * Math.log(2 * Math.PI * v) - (x - m) ** 2 / (2 * v);
const BASE = 'mu ~ Normal(0.0, 1.0)\ny ~ Normal(mu, 1.0)\nprior = lawof(record(mu = mu))\n'
  + 'K = kernelof(record(y = y), mu = mu)\nL = likelihoodof(K, record(y = 0.5))\n'
  + 'post = bayesupdate(L, prior)\n';
const X = 0.3;
const LOG_POST = lnN(0.5, 0, 2) + lnN(X, 0.25, 0.5);   // log Z + log N(θ; 0.25, 1/2)

async function score(measure: string) {
  const { ctx } = makeMatCtx(BASE + `M = ${measure}\nlp = logdensityof(M, record(mu = ${X}))\n`, { sampleCount: 4 });
  const m = await ctx.getMeasure('lp');
  return m.value ? m.value.data[0] : m.samples[0];
}

test('measures over a named posterior score in closed form', async () => {
  const cases: Array<[string, number]> = [
    ['weighted(2.0, post)', Math.log(2) + LOG_POST],
    ['logweighted(-0.7, post)', -0.7 + LOG_POST],
    ['superpose(post, post)', Math.log(2) + LOG_POST],
    ['superpose(post, prior)', Math.log(Math.exp(LOG_POST) + Math.exp(lnN(X, 0, 1)))],
    ['weighted(3.0, superpose(post, weighted(0.5, prior)))',
      Math.log(3) + Math.log(Math.exp(LOG_POST) + 0.5 * Math.exp(lnN(X, 0, 1)))],
  ];
  for (const [measure, want] of cases) {
    const got = await score(measure);
    assert.ok(Math.abs(got - want) < 1e-12, `${measure}: ${got} vs ${want}`);
  }
  // The bare posterior still scores as before.
  assert.ok(Math.abs((await score('post')) - LOG_POST) < 1e-12);
});
