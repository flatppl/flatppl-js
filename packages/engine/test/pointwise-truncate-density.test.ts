'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('pointwise density callbacks resolve named sets and each live interval', async () => {
  const { ctx } = makeMatCtx(`
half(mu, x) = logdensityof(normalize(truncate(Normal(mu,1),posreals)),x)
bounded(lo, hi, x) = logdensityof(truncate(Normal(lo,1),interval(lo,hi)),x)
a = half.([0,0], [.5,1.5])
b = bounded.([0,1], [1,2], [.5,1.5])
c = bounded.([0,2], [1,3], [-1,1.5])
`, { sampleCount: 1 });
  const logNormal = (x: number) => -(Math.log(2*Math.PI)+x*x)/2;
  for (const [name, expected] of [
    ['a', [.5,1.5].map(x => logNormal(x)+Math.log(2))],
    ['b', [.5,.5].map(logNormal)],
    ['c', [-Infinity,-Infinity]],
  ] as Array<[string, number[]]>) {
    const actual = (await ctx.getMeasure(name)).samples;
    assert.equal(actual.length, expected.length);
    for (let i = 0; i < expected.length; i++) {
      assert.ok(actual[i] === expected[i] || Math.abs(actual[i]-expected[i]) < 1e-12);
    }
  }
});
