'use strict';

// §08 Normal's mean is real. Both calling conventions must stop before a
// real-only sampling or density kernel coerces the complex mean to NaN.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

test('complex Normal parameters refuse drawing and scoring through either calling convention', async () => {
  for (const args of ['mu = complex(1.0, 2.0), sigma = 1.0', 'complex(1.0, 2.0), 1.0']) {
    const { ctx } = ctxFor(`x ~ Normal(${args})\nlp = logdensityof(Normal(${args}), 1.0)\n`, 2);
    for (const name of ['x', 'lp']) await assert.rejects(ctx.getMeasure(name), /statically invalid.*Normal:.*expects real, got complex/);
  }
  const { ctx } = ctxFor('lp = logdensityof(Normal(mu = 1.0, sigma = 2.0), 1.0)\n', 1);
  const m = await ctx.getMeasure('lp');
  assert.ok(Math.abs(m.samples[0] + Math.log(2 * Math.sqrt(2 * Math.PI))) < 1e-14);
});
