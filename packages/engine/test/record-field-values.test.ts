'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// §03 records hold values and §04 gives an inline expression the same
// meaning as a named binding. Each sampled a therefore yields [0,a,0].
for (const [name, value, expected] of [
  ['mixed array', '[0,a,0]', (x: number) => [0,x,0]],
  ['scalar product', 'a * [0,1,0]', (x: number) => [0*x,x,0*x]],
] as const) {
  test(`record ${name} preserves each sampled vector`, async () => {
    const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
r = record(v = ${value})
result = r.v
`, { sampleCount: 4, rootSeed: 132 });
    const a: Float64Array = (await ctx.getMeasure('a')).samples;
    const result = (await ctx.getMeasure('result')).value;
    assert.deepEqual(result.shape, [4,3]);
    assert.deepEqual(Array.from(result.data), Array.from(a).flatMap(expected));
  });
}

test('an interpolator keeps fixed record fields whole in either field order', async () => {
  for (const [sampleCount, fields] of [
    [3, 'x = .5, knots = [0,1,2], values = [0,a,0]'],
    [4, 'knots = [0,1,2], values = [0,a,0], x = .5'],
  ] as const) {
    const { ctx } = makeMatCtx(`
sp = standard_module("splines", "0.1")
a ~ Normal(0,1)
r = record(${fields})
result = sp.interp_cubic(r)
`, { sampleCount, rootSeed: 132 });
    const a: Float64Array = (await ctx.getMeasure('a')).samples;
    const result = (await ctx.getMeasure('result')).samples;
    assert.equal(result.length, sampleCount);
    // The natural interpolant on [0,1] is a(3x-x^3)/2.
    for (let i = 0; i < a.length; i++) {
      assert.ok(Math.abs(result[i] - (11/16)*a[i]) < 2e-12);
    }
  }
});

test('a fixed vector field does not change a weighted record law atom count', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
r = record(v = [1,2], a = a)
w = logweighted(r -> r.a, lawof(r))
result = pushfwd(r -> sum(r.v)+r.a, w)
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = await ctx.getMeasure('result');
  assert.equal((await ctx.getMeasure('r')).n_eff, 4);
  assert.equal(result.samples.length, 4);
  assert.equal(result.logWeights.length, 4);
  for (let i = 0; i < a.length; i++) {
    assert.equal(result.samples[i], 3+a[i]);
    assert.ok(Math.abs(result.logWeights[i]-result.logWeights[0]-(a[i]-a[0])) < 2e-12);
  }
});

test('a projected fixed vector remains whole through weighting and a later pushforward', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
r = record(v = [1,2], a = a)
w = logweighted(r -> r.a, lawof(r))
p = pushfwd(r -> r.v, w)
u = logweighted(v -> sum(v), p)
result = pushfwd(v -> 2*v, u)
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = await ctx.getMeasure('result');
  assert.deepEqual(result.value.shape, [4,2]);
  assert.deepEqual(Array.from(result.samples), [2,4,2,4,2,4,2,4]);
  assert.equal(result.logWeights.length, 4);
  const p = await ctx.getMeasure('p');
  assert.ok(Math.abs(result.logTotalmass-p.logTotalmass-3) < 2e-12);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(result.logWeights[i]-result.logWeights[0]-(a[i]-a[0])) < 2e-12);
  }
});

test('pushforward name splatting preserves fixed and sampled vector fields', async () => {
  for (const [value, expected] of [
    ['[1,2]', (a: number) => a+3],
    ['[a,a+1]', (a: number) => 3*a+1],
  ] as const) {
    const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
r = record(v = ${value}, a = a)
result = pushfwd((a,v) -> a+sum(v), lawof(r))
`, { sampleCount: 4, rootSeed: 132 });
    const a: Float64Array = (await ctx.getMeasure('a')).samples;
    const result = (await ctx.getMeasure('result')).samples;
    for (let i = 0; i < 4; i++) assert.ok(Math.abs(result[i]-expected(a[i])) < 2e-12);
  }
});

test('named record arguments retain fixed real and complex scalar fields', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
r = record(a = a, b = 3, c = 2+im)
result = pushfwd((a,b,c) -> (a+b)*c, lawof(r))
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [4]);
  for (let i = 0; i < 4; i++) {
    assert.equal(result.data[i], 2*(a[i]+3));
    assert.equal(result.im[i], a[i]+3);
  }
});

test('the law of a fixed vector supplies whole vectors through constant weighting', async () => {
  const { ctx } = makeMatCtx(`
a = [1,2]
result = pushfwd(x -> 2*x, weighted(2, lawof(a)))
`, { sampleCount: 4, rootSeed: 132 });
  const measure = await ctx.getMeasure('result');
  const result = measure.value;
  assert.deepEqual(result.shape, [4,2]);
  assert.deepEqual(Array.from(result.data), [2,4,2,4,2,4,2,4]);
  assert.equal(measure.logWeights.length, 4);
  assert.ok(Math.abs(measure.logTotalmass-Math.log(2)) < 2e-12);
});
