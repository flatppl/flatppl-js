'use strict';

// Spec §03 "Note on infinities": `reals` admits `inf` and `-inf` as legal values,
// but "the reference measure is the ordinary Lebesgue measure on the finite-real
// part; the points ±∞ carry zero Lebesgue mass". So `±inf` lies outside the
// support of every univariate density, the density tends to 0 there, and the
// log-density is `-inf`. Formulas such as `(k - 1) log(x) - (x / lambda)^k` gave
// the IEEE artifact `inf - inf = NaN` (Weibull and ChiSquared at `inf`).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function score(source: string): Promise<number> {
  const { ctx } = makeMatCtx(source, { sampleCount: 1 });
  const m = await ctx.getMeasure('score');
  return (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
}

const DENSITIES: Array<[string, string]> = [
  ['Gamma(shape = 2.5, rate = 1.5)', 'inf'],
  ['InverseGamma(shape = 2.5, scale = 1.5)', 'inf'],
  ['Weibull(shape = 2.5, scale = 1.5)', 'inf'],
  ['Rayleigh(scale = 1.5)', 'inf'],
  ['ChiSquared(k = 2.5)', 'inf'],
  ['Exponential(rate = 2.5)', 'inf'],
  ['LogNormal(mu = 0.0, sigma = 2.5)', 'inf'],
  ['Pareto(shape = 2.5, scale = 1.5)', 'inf'],
  ['Logistic(mu = 0.0, s = 2.5)', 'inf'],
  ['Logistic(mu = 0.0, s = 2.5)', '-inf'],
  ['Normal(mu = 0.0, sigma = 2.5)', 'inf'],
  ['Normal(mu = 0.0, sigma = 2.5)', '-inf'],
];

test('an infinite variate scores -inf for every univariate density', async () => {
  for (const [density, x] of DENSITIES) {
    assert.equal(await score(`score = logdensityof(${density}, ${x})\n`), -Infinity, `${density} at ${x}`);
  }
});

test('finite points keep their closed-form log-density', async () => {
  // Weibull(k = 2.5, lambda = 1.5) at 3: log(k/lambda) + (k-1) log(x/lambda) - (x/lambda)^k.
  const k = 2.5, lambda = 1.5, x = 3.0;
  const want = Math.log(k / lambda) + (k - 1) * Math.log(x / lambda) - (x / lambda) ** k;
  const got = await score('score = logdensityof(Weibull(shape = 2.5, scale = 1.5), 3.0)\n');
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
});

test('a plate with one infinite entry scores -inf through the prepared density', async () => {
  // A shared Normal scale scores the plate through the prepared (factory) path.
  assert.equal(await score('score = logdensityof(iid(Normal(0.0, 1.5), 3), [0.5, inf, -0.5])\n'), -Infinity);
  assert.equal(await score('score = logdensityof(iid(Weibull(2.5, 1.5), 2), [1.0, inf])\n'), -Infinity);
});

test('a NaN density with no derivation names the NaN', async () => {
  const { ctx } = makeMatCtx('z = 0.0\nscore = builtin_logdensityof(Normal, record(mu = 0.0, sigma = 1.0), z / z)\n',
    { sampleCount: 1 });
  await assert.rejects(() => ctx.getMeasure('score'),
    /fixed binding 'score' evaluated to NaN, and no derivation computes it/);
});
