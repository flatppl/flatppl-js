'use strict';

// A distribution parameter outside its §08 value set (`param-domains.ts`) at an
// MCMC proposal rejects that proposal and is counted in the run diagnostics. At
// an initial position it is fatal.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { ctxFor } = require('./density/regression-baseline.test.ts');

const OPTS = { backend: 'mh', chains: 4, walkers: null, warmup: 40, draws: 80, seed: 1 };

test('a proposal that puts a parameter outside its set is rejected and counted', async () => {
  // MH saturates `mu ~ Beta(1, 1)` at exactly 1.0, so `b = (1 - mu) * a_plus_b`
  // is 0 and `Beta(a, 0)` has no density there.
  const src = fs.readFileSync(path.join(__dirname, 'fixtures/baseline/surgical-failures.flatppl'), 'utf8');
  const { ctx } = ctxFor(src, 60);
  ctx.inferenceOpts = OPTS;
  const m = await ctx.getMeasure('posterior');
  const rejections = m.diagnostics?.parameterRejections;
  assert.ok(rejections && rejections.count > 0, JSON.stringify(m.diagnostics));
  assert.match(rejections.first, /parameter '[a-z]+' = .* is outside its declared value set posreals/);
  for (const field of Object.values(m.fields) as any[]) {
    const samples = field.samples ?? field.value?.data;
    assert.ok(samples && samples.length > 0);
  }
});

test('an initial position outside a parameter set aborts the run', async () => {
  const { ctx } = ctxFor(`
s ~ Normal(-5, 0.1)
y ~ Normal(0, s)
prior = lawof(record(s = s))
L = likelihoodof(kernelof(record(y = y), s = s), record(y = 0.3))
posterior = bayesupdate(L, prior)
`, 60);
  ctx.inferenceOpts = OPTS;
  await assert.rejects(() => ctx.getMeasure('posterior'),
    /Normal: parameter 'sigma' = -\d.* is outside its declared value set posreals/);
});
