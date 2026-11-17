'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('a shared vector is not a scalar batch when its length equals the sample count', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
result = [0,1,0] * a
`, { sampleCount: 3, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [3,3]);
  assert.deepEqual(Array.from(result.data), Array.from(a).flatMap((x: number) => [0*x,x,0*x]));
});

test('sampled scalar products preserve complex coordinates and adjoints', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
result = (a+im) * [1,im]
v = complex.([1,0],[0,1])
norm = (a * adjoint(v)) * v
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = (await ctx.getMeasure('result')).value;
  const norm = (await ctx.getMeasure('norm')).value;
  assert.deepEqual(result.shape, [4,2]);
  for (let i = 0; i < a.length; i++) {
    assert.equal(result.data[i*2], a[i]);
    assert.equal(result.data[i*2+1], -1);
    assert.equal(result.im[i*2], 1);
    assert.equal(result.im[i*2+1], a[i]);
    assert.equal(norm.data[i], 2*a[i]);
    assert.equal(norm.im[i], 0);
  }
});

test('a sampled scalar scales a shared diagonal matrix as one matrix per atom', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
result = a * diagmat([1,2])
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [4,2,2]);
  assert.deepEqual(Array.from(result.data), Array.from(a).flatMap((x: number) => [x,0,0,2*x]));
});

test('a named sampled row vector keeps its orientation through a weighted record argument', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
b = a * transpose([1,2])
r = record(v = b, a = a)
w = weighted(2, lawof(r))
result = pushfwd(r -> r.a+r.v*[3,4], w)
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = await ctx.getMeasure('result');
  assert.equal(result.logWeights.length, 4);
  assert.ok(Math.abs(result.logTotalmass-Math.log(2)) < 2e-12);
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(result.samples[i]-12*a[i]) < 2e-12);
});
