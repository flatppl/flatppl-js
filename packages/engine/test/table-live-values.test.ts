'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('live table rows bind by name and preserve each sampled column', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
f(x,y) = x-y
rows = table(y = [1,2], x = [a,a])
result = f.(rows)
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [4,2]);
  assert.deepEqual(Array.from(result.data), Array.from(a).flatMap((x: number) => [x-1,x-2]));
});

test('a table produced by broadcasting remains one table per atom', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
rows = (x -> record(x = x)).(a .+ [1,2])
result = (r -> r.x).(rows)
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [4,2]);
  assert.deepEqual(Array.from(result.data), Array.from(a).flatMap((x: number) => [x+1,x+2]));
});

test('a spline table broadcast retains vector-valued rows', async () => {
  const { ctx } = makeMatCtx(`
sp = standard_module("splines", "0.1")
x ~ Uniform(interval(0,1))
t = [0,0,0,1,1,1]
rows = table(x = [x,x], knots = [t,t], degree = [2,2], order = [0,0])
result = sp.bspline_derivative.(rows)
`, { sampleCount: 4, rootSeed: 132 });
  const x = (await ctx.getMeasure('x')).samples;
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [4,2,3]);
  for (let i = 0; i < x.length; i++) {
    const expected = [(1-x[i])**2, 2*x[i]*(1-x[i]), x[i]**2];
    for (let j = 0; j < 6; j++) {
      assert.ok(Math.abs(result.data[i*6+j] - expected[j%3]) < 2e-12);
    }
  }
});

test('a table-valued conditional law retains both rows in each sampled table', async () => {
  const { ctx } = makeMatCtx(`
K(x) = lawof(table(v = [x,x+1]))
M = kchain(Normal(0,1),K)
t ~ M
result = (r -> r.v).(t)
`, { sampleCount: 4, rootSeed: 132 });
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [4,2]);
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(result.data[i*2+1]-result.data[i*2]-1) < 2e-12);
});

test('table broadcasts preserve nested vector levels through indexing', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
rows = table(x = [a,a+1])
u = (r -> [[r.x,2*r.x],[3*r.x,4*r.x]]).(rows)
z = u[1]
result = sum.(z)
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const result = (await ctx.getMeasure('result')).value;
  assert.deepEqual(result.shape, [4,2]);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(result.data[i*2]-3*a[i]) < 2e-12);
    assert.ok(Math.abs(result.data[i*2+1]-7*a[i]) < 2e-12);
  }
});

test('table broadcasts retain real and imaginary parts of scalar and vector rows', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0,1)
rows = table(x = [a,a+1])
scalars = (r -> r.x+im).(rows)
vectors = (r -> [r.x,r.x+im]).(rows)
`, { sampleCount: 4, rootSeed: 132 });
  const a: Float64Array = (await ctx.getMeasure('a')).samples;
  const scalars = (await ctx.getMeasure('scalars')).value;
  const vectors = (await ctx.getMeasure('vectors')).value;
  assert.deepEqual(scalars.shape, [4,2]);
  assert.deepEqual(vectors.shape, [4,2,2]);
  for (let i = 0; i < 4; i++) for (let row = 0; row < 2; row++) {
    assert.equal(scalars.data[i*2+row], a[i]+row);
    assert.equal(scalars.im[i*2+row], 1);
    assert.equal(vectors.data[i*4+row*2], a[i]+row);
    assert.equal(vectors.data[i*4+row*2+1], a[i]+row);
    assert.equal(vectors.im[i*4+row*2], 0);
    assert.equal(vectors.im[i*4+row*2+1], 1);
  }
});
