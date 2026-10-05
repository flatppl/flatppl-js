'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser, clm, orchestrator } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

// §04 deterministic record fields keep their dependence on the fed variate.
// y[1] ~ Normal(mu, 1), y[2] ~ Normal(2*mu, 1), observed at [1,2].
const SOURCE = `
mu ~ Normal(0, 1)
step(x) = record(mean = x + mu)
s1 = step(0)
s2 = step(s1.mean)
y ~ Normal.([s1.mean, s2.mean], 1)
prior = lawof(record(mu = mu))
K = kernelof(record(y = y), mu = mu)
L = likelihoodof(K, record(y = [1, 2]))
posterior = bayesupdate(L, prior)
`;

function expectedLikelihood(mu: number): number {
  return -Math.log(2 * Math.PI) - 2.5 * (1 - mu) ** 2;
}

test('record recurrence likelihood keeps the scored parameter in scalar and batch paths', async () => {
  const { ctx, built } = makeMatCtx(SOURCE, { sampleCount: 32, rootSeed: 123 });
  const scorer = await buildLogPi(ctx, built.derivations.posterior);
  const points = [{ mu: -0.5 }, { mu: 0.25 }, { mu: 1.5 }];
  const batch = scorer.priorLikBatch(points);
  for (let i = 0; i < points.length; i++) {
    const mu = points[i].mu;
    const expected = expectedLikelihood(mu);
    assert.ok(Math.abs(scorer.likOf(points[i]) - expected) < 1e-12);
    assert.ok(Math.abs(batch.lik[i] - expected) < 1e-12);
    const prior = -0.5 * Math.log(2 * Math.PI) - 0.5 * mu * mu;
    assert.ok(Math.abs(scorer.logPi(points[i]) - prior - expected) < 1e-12);
  }
});

test('record recurrence log likelihood profile matches the closed-form curve', async () => {
  const { ctx } = makeMatCtx(SOURCE);
  // Use the viewer's canonical lowering and worker profile dispatch.
  const signature = orchestrator.signatureOf('L', ctx.bindings);
  const lowered = clm.lowerMeasure(signature.body, ctx, { boundaries: {}, freeInputs: ['mu'] });
  assert.ok(lowered);
  const reply = await ctx.sendWorker({
    type: 'profileN', ir: lowered.body, sweepName: 'mu',
    range: [-0.5, 1.5], count: 3, mode: 'logdensity',
    observed: { y: [1, 2] }, tally: 'clamped',
  });
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(reply.samples[i] - expectedLikelihood(-0.5 + i)) < 1e-12);
  }
});

test('record recurrence posterior sampling matches the conjugate Normal posterior', async () => {
  const { ctx } = makeMatCtx(SOURCE, { sampleCount: 4000, rootSeed: 123 });
  const result = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'mh', chains: 2, warmup: 500, draws: 2000, seed: 1,
  });
  const samples = result.fields.mu.samples;
  const mean = samples.reduce((sum: number, x: number) => sum + x, 0) / samples.length;
  const variance = samples.reduce((sum: number, x: number) => sum + (x - mean) ** 2, 0) / samples.length;
  assert.ok(Math.abs(mean - 5 / 6) < 0.06, `mean ${mean} vs 5/6`);
  assert.ok(Math.abs(variance - 1 / 6) < 0.04, `variance ${variance} vs 1/6`);
});
