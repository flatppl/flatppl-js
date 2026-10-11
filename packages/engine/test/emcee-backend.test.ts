'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

test('emcee automatic walkers ignore the viewer chain count and preserve explicit counts', async () => {
  const names = ['a', 'b', 'c', 'd', 'e'];
  const fields = names.map((name) => `${name} = ${name}`).join(', ');
  const source = `
${names.map((name) => `${name} ~ Normal(0, 1)`).join('\n')}
prior = lawof(record(${fields}))
y ~ Normal(a, 1)
K = kernelof(y, ${fields})
L = likelihoodof(K, 0)
posterior = bayesupdate(L, prior)
`;
  const run = async (counts: any) => {
    const { ctx } = ctxFor(source, 120);
    return materialiser.materialiseMeasure('posterior', ctx, {
      backend: 'emcee', warmup: 20, draws: 10, seed: 1, ...counts,
    });
  };
  // These are the viewer's unchanged count options when selecting emcee.
  const auto = await run({ chains: 4, walkers: null });
  const explicit = await run({ walkers: 12 });
  assert.equal(auto.diagnostics.nSamples, 120);
  for (const name of names) assert.deepEqual(auto.fields[name].samples, explicit.fields[name].samples);
  // An absent walkers option retains the explicit chains alias. Ten is the
  // minimum accepted count for five dimensions; it must not become twelve.
  const minimum = await run({ walkers: 10 });
  const alias = await run({ chains: 10 });
  assert.equal(minimum.diagnostics.nSamples, 100);
  for (const name of names) assert.deepEqual(alias.fields[name].samples, minimum.fields[name].samples);
  await assert.rejects(run({ walkers: 4 }), /even and >= 10.*dim=5/);
});
