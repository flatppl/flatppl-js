'use strict';
// #309: the native score path can score `logdensityof(bayesupdate(L, prior), θ)`.
// Previously it tried to feed the prior MEASURE as a value into the posterior
// kernel's boundary inputs, which threw `feedInputs: measure for "prior" has
// neither .value nor .samples`. Per spec §06 `bayesupdate(L, prior) ≡
// logweighted(fn(logdensityof(L, _)), prior)`, so the posterior density at θ is
// `logdensityof(L, θ) + logdensityof(prior, θ)` — the two are scored separately
// and summed. Oracles are independently derived (scipy), NOT read back from the
// engine.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function score(src: string, point: string): Promise<number> {
  const { ctx } = ctxFor(src + `\n__score__ = logdensityof(posterior, ${point})\n`, 1);
  const mm = await ctx.getMeasure('__score__');
  return mm.value ? mm.value.data[0] : mm.samples[0];
}
const H = 'flatppl_compat = "0.1"\n';

// Model A — single-parameter posterior. mu ~ Normal(0,3); y ~ iid(Normal(mu,1),2)
// against y_data=[1,2]; scored at mu=0.5.
//   scipy: norm.logpdf(0.5,0,3) + norm.logpdf(1,0.5,1) + norm.logpdf(2,0.5,1)
const MODEL_A = H + `
y_data = [1.0, 2.0]
mu ~ Normal(0.0, 3.0)
prior = lawof(record(mu = mu))
y ~ iid(Normal(mu, 1.0), 2)
K = kernelof(record(y = y), mu = mu)
L = likelihoodof(K, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

// Model B — two-parameter posterior (mu AND sigma latent). mu ~ Normal(0,5);
// sigma ~ Gamma(shape=4, rate=2); y ~ iid(Normal(mu,sigma),3) against
// y_data=[0.5,1.5,2.5]; scored at record(mu=1, sigma=1.5).
const MODEL_B = H + `
y_data = [0.5, 1.5, 2.5]
mu ~ Normal(0.0, 5.0)
sigma ~ Gamma(shape = 4.0, rate = 2.0)
prior = lawof(record(mu = mu, sigma = sigma))
y ~ iid(Normal(mu, sigma), 3)
K = kernelof(record(y = y), mu = mu, sigma = sigma)
L = likelihoodof(K, record(y = y_data))
posterior = bayesupdate(L, prior)
`;

test('#309: single-param bayesupdate posterior scores at θ (== scipy oracle)', async () => {
  const v = await score(MODEL_A, 'record(mu = 0.5)');
  assert.ok(Math.abs(v - (-5.119316777171017)) < 1e-9, `posterior@mu=0.5 got ${v}`);
});

test('#309: two-param bayesupdate posterior scores at θ (== scipy oracle)', async () => {
  const v = await score(MODEL_B, 'record(mu = 1.0, sigma = 1.5)');
  assert.ok(Math.abs(v - (-7.935473903352175)) < 1e-9, `posterior@(1,1.5) got ${v}`);
});

// The posterior density is exactly logdensityof(L, θ) + logdensityof(prior, θ):
// the two sub-scores must still work standalone and sum to the posterior.
test('#309: posterior density == logdensityof(L, θ) + logdensityof(prior, θ)', async () => {
  const mk = (extra: string, name: string) => {
    const { ctx } = ctxFor(MODEL_B + extra, 1);
    return ctx.getMeasure(name).then((m: any) => (m.value ? m.value.data[0] : m.samples[0]));
  };
  const pt = 'record(mu = 1.0, sigma = 1.5)';
  const lL = await mk(`\n__L__ = logdensityof(L, ${pt})\n`, '__L__');
  const lP = await mk(`\n__P__ = logdensityof(prior, ${pt})\n`, '__P__');
  const lPost = await score(MODEL_B, pt);
  assert.ok(Math.abs(lPost - (lL + lP)) < 1e-12, `sum ${lL + lP} vs posterior ${lPost}`);
});
