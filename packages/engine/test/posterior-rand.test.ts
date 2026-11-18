'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser, processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const model = `
p = elementof(interval(0.0, 1.0))
L = likelihoodof(functionof(Bernoulli(p), p = p), true)
prior = joint(p = Beta(1.0, 1.0))
posterior = bayesupdate(L, prior)
state = rnginit(42)
`;

test('rand rejects raw posteriors during inference, including a successor-only query', () => {
  for (const source of [
    'alias = posterior\nresult, next = rand(alias, state)',
    'result, next = rand(iid(posterior, 4), state)',
    'product = joint(theta = posterior)\nresult, next = rand(product, state)',
    'M = pushfwd(fn(_), posterior)\nresult, next = rand(M, state)',
    '_, next = rand(bayesupdate(L, prior), state)\noutputs = next',
  ]) {
    const { diagnostics } = processSource(model + source);
    assert.ok(diagnostics.some((d: any) => d.severity === 'error' &&
      d.message.startsWith('rand requires a probability measure')), source);
  }
});

test('rand refuses empirical posterior draws inside normalized composite measures', async () => {
  for (const measure of [
    'normalize(posterior)',
    'joint(theta = normalize(posterior))',
    'pushfwd(fn(_), normalize(posterior))',
    'iid(normalize(posterior), 4)',
    'kchain(normalize(posterior), r -> joint(y = Normal(r.p, 1)))',
  ]) {
    const { ctx } = makeMatCtx(model + 'M = ' + measure + '\nresult, next = rand(M, state)');
    for (const backend of ['is', 'mh', 'nuts']) {
      await assert.rejects(materialiser.materialiseMeasure('result', ctx, { backend }),
        /posterior sampler|normalize.*cannot be sampled/, measure + ' with ' + backend);
    }
  }
});

test('a deterministic function of a posterior draw still needs an exact sampler', async () => {
  const { ctx } = makeMatCtx(model + 'x ~ normalize(posterior)\ny = x.p + 1\nM = lawof(y)\nresult, next = rand(iid(M, 16), state)');
  await assert.rejects(ctx.getMeasure('result'), /posterior sampler/);
});

test('an explicit kernel input replaces the draw at its boundary', async () => {
  const { ctx } = makeMatCtx(model + 'x ~ normalize(posterior)\nK = functionof(Normal(x.p, 1), x=x)\nM = joint(s=K(record(p=0.7)))\nresult, next = rand(M, state)');
  const result = await ctx.getMeasure('result');
  assert.ok(Number.isFinite(result.fields.s.samples[0]));
});

test('a posterior mass can supply a scalar parameter of a sampleable measure', async () => {
  const { ctx } = makeMatCtx(model + 'score = totalmass(posterior)\nM = joint(s = Dirac(score))\nresult, next = rand(M, state)');
  const result = await ctx.getMeasure('result');
  assert.ok(result.fields.s.samples[0] > 0 && result.fields.s.samples[0] < 1);
});

test('the posterior itself retains the likelihood weights', async () => {
  const { ctx } = makeMatCtx(model, { sampleCount: 16 });
  const measure = await ctx.getMeasure('posterior');
  const p = measure.fields.p.samples;
  const weights = measure.logWeights;
  assert.equal(p.length, 16);
  for (let i = 1; i < p.length; ++i) {
    assert.ok(Math.abs((weights[i] - weights[0]) - Math.log(p[i] / p[0])) < 1e-12);
  }
});
