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

test('Gamma priors and live half-line normalizers retain their analytic mass', async () => {
  const { ctx } = makeMatCtx(`
p = logdensityof(normalize(truncate(Gamma(.01,.01),interval(0,inf))),2.3)
score(rate) = logdensityof(normalize(truncate(Gamma(2,rate),interval(5,inf))),7)
rows = score.([.1,1])
`, { sampleCount: 1 });
  // Independent base-R dgamma(2.3, shape=.01, rate=.01, log=TRUE).
  assert.ok(Math.abs((await ctx.getMeasure('p')).samples[0] - (-5.493111611607655)) < 1e-12);
  const actual = (await ctx.getMeasure('rows')).samples;
  [.1,1].forEach((rate, i) => {
    // Erlang density rate^2*x*exp(-rate*x), divided by P(X>5).
    const expected = 2*Math.log(rate) + Math.log(7) - 2*rate - Math.log1p(5*rate);
    assert.ok(Math.abs(actual[i] - expected) < 1e-12);
  });
});
