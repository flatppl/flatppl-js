'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('location-scale density callbacks retain their own inputs', async () => {
  const { ctx } = makeMatCtx(`
normal(mu, scale) = logdensityof(locscale(Normal(0, 1), mu, scale), 0)
student(mu, scale) = logdensityof(locscale(StudentT(5), mu, scale), 0)
mu = elementof(reals)
scale = elementof(reals)
explicit = functionof(logdensityof(locscale(StudentT(5), mu, scale), 0), mu = mu, scale = scale)
n = normal(1, -2)
t = student(1, 2)
r = student.([1, 2], [-2, 3])
e = explicit(1, 2)
`, { sampleCount: 1 });
  const n = (await ctx.getMeasure('n')).samples[0];
  const t = (await ctx.getMeasure('t')).samples[0];
  assert.ok(Math.abs(n - (-0.5 * Math.log(2 * Math.PI) - 0.125 - Math.log(2))) < 1e-12);
  const student = Math.log(8 / (3 * Math.PI * Math.sqrt(5))) - 3 * Math.log1p(0.05) - Math.log(2);
  assert.ok(Math.abs(t - student) < 1e-12);
  assert.ok(Math.abs((await ctx.getMeasure('e')).samples[0] - student) < 1e-12);
  const r = (await ctx.getMeasure('r')).samples;
  assert.equal(r.length, 2);
  assert.ok(Math.abs(r[0] - student) < 1e-12);
  const second = Math.log(8 / (3 * Math.PI * Math.sqrt(5))) - 3 * Math.log1p(4 / 45) - Math.log(3);
  assert.ok(Math.abs(r[1] - second) < 1e-12);
});

test('location-scale callbacks preserve discrete masses and mapped support', async () => {
  const { ctx } = makeMatCtx(`
score(mu, scale, y) = logdensityof(locscale(Bernoulli(0.3), mu, scale), y)
a = score(0.25, -0.5, -0.25)
b = score(0.25, -0.5, 0.25)
c = score(0.25, -0.5, 0)
r = score.([0.25, 0.5], [-0.5, 2], [-0.25, 0.5])
`, { sampleCount: 1 });
  assert.ok(Math.abs((await ctx.getMeasure('a')).samples[0] - Math.log(0.3)) < 1e-12);
  assert.ok(Math.abs((await ctx.getMeasure('b')).samples[0] - Math.log(0.7)) < 1e-12);
  assert.equal((await ctx.getMeasure('c')).samples[0], -Infinity);
  const r = (await ctx.getMeasure('r')).samples;
  assert.equal(r.length, 2);
  assert.ok(Math.abs(r[0] - Math.log(0.3)) < 1e-12);
  assert.ok(Math.abs(r[1] - Math.log(0.7)) < 1e-12);
});

test('the affine inverse cannot shadow a captured user binding', async () => {
  const { ctx } = makeMatCtx(`
__locscale_input = 3
score(s, y) = logdensityof(locscale(Normal(0, 1), __locscale_input, s), y)
r = score.([2, 3], [1, 1])
`, { sampleCount: 1 });
  const r = (await ctx.getMeasure('r')).samples;
  assert.equal(r.length, 2);
  for (let i = 0; i < 2; i++) {
    const scale = i + 2;
    const expected = -0.5 * Math.log(2 * Math.PI) - 2 / (scale * scale) - Math.log(scale);
    assert.ok(Math.abs(r[i] - expected) < 1e-12);
  }
});

test('an applied location-scale measure factory retains its sampling law', async () => {
  const { ctx } = makeMatCtx(`
factory(mu, scale) = locscale(Uniform(interval(0, 1)), mu, scale)
M = factory(2, -1)
x ~ M
`, { sampleCount: 256, rootSeed: 117 });
  const samples = (await ctx.getMeasure('x')).samples;
  assert.equal(samples.length, 256);
  let sum = 0;
  for (const x of samples) {
    assert.ok(x >= 1 && x <= 2);
    sum += x;
  }
  assert.ok(Math.abs(sum / samples.length - 1.5) < 0.1);
});
