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

const ZERO_SUM_MODEL = `
x ~ tilt(iid(Normal(0, 1), 6), Lebesgue(zerosum(6)))
prior = lawof(record(x = x))
y ~ Normal(x[1], 1)
L = likelihoodof(kernelof(y, x = x), 0)
posterior = bayesupdate(L, prior)
`;

async function runShort(source: string, counts: { walkers?: number | null; chains?: number }) {
  const { ctx } = ctxFor(source, 100);
  return materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'demcz', warmup: 20, draws: 10, seed: 1, ...counts,
  });
}

for (const [name, source, field, walkers] of [
  ['scalar', MODEL, 'mu', 8],
  // Six output coordinates have five free coordinates: ten walkers, not twelve.
  ['zero-sum vector', ZERO_SUM_MODEL, 'x', 10],
] as const) {
  test(`DEMCz automatic walkers use the free dimension of a ${name}`, async () => {
    const auto = await runShort(source, { chains: 4, walkers: null });
    const absent = await runShort(source, {});
    const explicit = await runShort(source, { walkers });
    assert.equal(auto.diagnostics.nSamples, walkers * 10);
    assert.equal(absent.diagnostics.nSamples, walkers * 10);
    assert.deepEqual(auto.fields[field].samples, explicit.fields[field].samples);
    assert.deepEqual(absent.fields[field].samples, explicit.fields[field].samples);
  });
}

test('DEMCz preserves explicit walkers and the absent-walkers chains alias', async () => {
  // Archive-backed DEMCz permits odd ensembles and counts below twice dim.
  const explicit = await runShort(ZERO_SUM_MODEL, { walkers: 5, chains: 20 });
  const alias = await runShort(ZERO_SUM_MODEL, { chains: 5 });
  const minimum = await runShort(ZERO_SUM_MODEL, { walkers: 4 });
  assert.equal(explicit.diagnostics.nSamples, 50);
  assert.equal(alias.diagnostics.nSamples, 50);
  assert.equal(minimum.diagnostics.nSamples, 40);
  assert.deepEqual(alias.fields.x.samples, explicit.fields.x.samples);
});

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
