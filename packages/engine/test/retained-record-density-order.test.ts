'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('a retained record law scores observed sources independently of field order and labels', async () => {
  const { ctx } = makeMatCtx(`
u ~ Normal(0, 1)
v ~ Normal(u, 2)
forward = lawof(record(u = u, v = v))
reverse = lawof(record(v = v, u = u))
renamed = lawof(record(v = u, u = v))
a = logdensityof(forward, record(u = 0.3, v = -0.2))
b = logdensityof(reverse, record(v = -0.2, u = 0.3))
c = logdensityof(renamed, record(v = 0.3, u = -0.2))
`, { sampleCount: 4 });
  // log N(0.3;0,1) + log N(-0.2;0.3,2).
  const expected = -Math.log(2 * Math.PI) - Math.log(2) - 0.3 ** 2 / 2 - 0.5 ** 2 / 8;
  for (const name of ['a', 'b', 'c']) {
    const score = await ctx.getMeasure(name);
    for (const value of score.samples) assert.ok(Math.abs(value - expected) < 1e-12,
      `${name}: ${value} != ${expected}`);
  }
});
