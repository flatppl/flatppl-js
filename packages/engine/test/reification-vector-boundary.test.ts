'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('explicit vector boundaries retain broadcast shape in functions and kernels', async () => {
  const { ctx } = makeMatCtx(`
cuts = elementof(cartpow(reals, 2))
shifted = cuts .- 1.0
joined = cat(shifted, [1.0])
f = functionof(joined, points = cuts)
value = f(points = [2.0, 4.0])
y ~ Normal.(mu = joined, sigma = 1.0)
K = kernelof(y, points = cuts)
L = likelihoodof(K, [1.0, 3.0, 1.0])
score = logdensityof(L, record(points = [2.0, 4.0]))
`, { sampleCount: 4 });

  assert.deepEqual(Array.from((await ctx.getMeasure('value')).value.data), [1, 3, 1]);
  // Three independent standard-Normal residuals, each exactly zero.
  const expected = -1.5 * Math.log(2 * Math.PI);
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0] - expected) < 1e-12);
});
