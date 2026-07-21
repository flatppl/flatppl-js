'use strict';
// End-to-end test: backend:'demcz' posterior path through the materialiser.
// Conjugate Normal-Normal: mu ~ Normal(0,10); y ~ Normal(mu,1) observed at 5.
// Analytic posterior: mu | y ~ Normal(postMean, postVar),
//   postVar  = 1/(1/100 + 1/1) = 0.990099…,  postMean = postVar*5 = 4.950495…
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const MODEL = `
mu = draw(Normal(mu = 0.0, sigma = 10.0))
prior = lawof(record(mu = mu))
obs_dist = joint(y = Normal(mu = mu, sigma = 1.0))
K = functionof(obs_dist, mu = mu)
L = likelihoodof(K, record(y = 5.0))
posterior = bayesupdate(L, prior)
`;

const stats = (arr: Float64Array) => {
  let m = 0; for (const v of arr) m += v; m /= arr.length;
  let v = 0; for (const d of arr) v += (d - m) ** 2; v /= arr.length - 1;
  return { m, v };
};

test('backend:demcz recovers the conjugate Normal-Normal posterior', async () => {
  const { proc, ctx } = ctxFor(MODEL, 2000);
  const errs = (proc.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, `parse errors: ${errs.map((e: any) => e.message).join('; ')}`);
  const m = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'demcz', chains: 8, warmup: 1000, draws: 2000, seed: 1,
  });
  const { m: mean, v: variance } = stats(m.fields.mu.samples);
  const postVar = 1 / (1 + 1 / 100), postMean = 5 * postVar;
  assert.ok(Math.abs(mean - postMean) < 0.1, `mean ${mean} vs ${postMean}`);
  assert.ok(Math.abs(variance - postVar) < 0.1, `var ${variance} vs ${postVar}`);
  const { acceptRate, perParam } = m.diagnostics;
  assert.ok(acceptRate > 0.1 && acceptRate < 0.9, `acceptRate ${acceptRate}`);
  assert.ok(Number.isFinite(perParam.mu.rHat) && perParam.mu.rHat < 1.1, `split-R̂ ${perParam.mu.rHat}`);
});

test('backend:demcz is deterministic for a fixed seed', async () => {
  const run = async () => {
    const { ctx } = ctxFor(MODEL, 2000);
    const m = await materialiser.materialiseMeasure('posterior', ctx, { backend: 'demcz', chains: 8, warmup: 300, draws: 300, seed: 42 });
    return Array.from(m.fields.mu.samples);
  };
  const a = await run(), b = await run();
  assert.deepEqual(a, b);
});
