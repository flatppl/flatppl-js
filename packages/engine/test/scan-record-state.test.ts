'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('..');
const { toJS } = require('./_value-helpers.ts');
const { createWorkerHandler } = require('../worker.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

// §04 scan calls f(acc, next), with independently typed state and input.
test('scan carries a record accumulator through scalar inputs without numeric coercion', () => {
  const source = `
step(state, input) = record(total = state.total + input, count = state.count + 1)
states = scan(step, record(total = 0.0, count = 0), [2.0, -1.0, 4.0])
total(state) = state.total
totals = total.(states)
`;
  const lifted = processSource(source);
  assert.deepEqual(lifted.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(lifted.linkedBindings);
  assert.deepEqual(toJS(built.fixedValues.get('states')), [
    { total: 2, count: 1 }, { total: 1, count: 2 }, { total: 5, count: 3 },
  ]);
  assert.deepEqual(toJS(built.fixedValues.get('totals')), [2, 1, 5]);
});

const MODEL = `
mu ~ Normal(0, 1)
step(state, input) = record(mean = state.mean + mu * input)
states = scan(step, record(mean = 0.0), [1.0, 1.0])
state_mean(state) = state.mean
y ~ Normal.(state_mean.(states), 1)
prior = lawof(record(mu = mu))
K = kernelof(record(y = y), mu = mu)
L = likelihoodof(K, record(y = [1, 2]))
posterior = bayesupdate(L, prior)
`;

for (const backend of ['mh', 'ram', 'amis']) {
  test(`scan record state scores finitely in the ${backend} worker sampler`, async () => {
    const worker = createWorkerHandler();
    const reply = await worker.handle({
      type: 'mcmcRun', source: MODEL, name: 'posterior', sampleCount: 32, seed: 1,
      inferenceOpts: { backend, chains: 1, warmup: 50, draws: 64, seed: 1 },
    });
    assert.equal(reply.type, 'mcmcResult', reply.message);
    const { ctx, built } = makeMatCtx(MODEL);
    const scorer = await buildLogPi(ctx, built.derivations.posterior);
    for (const mu of reply.measure.fields.mu.samples) {
      assert.ok(Number.isFinite(mu));
      // Independent Normal product: means [mu, 2mu], data [1, 2].
      const likelihood = -Math.log(2 * Math.PI) - 2.5 * (1 - mu) ** 2;
      const prior = -0.5 * Math.log(2 * Math.PI) - 0.5 * mu ** 2;
      assert.ok(Math.abs(scorer.likOf({ mu }) - likelihood) < 1e-12);
      assert.ok(Math.abs(scorer.logPi({ mu }) - likelihood - prior) < 1e-12);
    }
  });
}
