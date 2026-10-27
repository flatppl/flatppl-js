'use strict';

// §03 transposed-vector orientation survives per-atom arithmetic (§07).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('a shared vector stays whole when its length equals the atom count', async () => {
  const { ctx } = makeMatCtx('x~iid(Normal(0,1),3)\nw=[1.,-2.,3.]\ny=transpose(x)*w',
    { sampleCount: 3, rootSeed: 117 });
  const x = (await ctx.getMeasure('x')).samples;
  const y = (await ctx.getMeasure('y')).samples;
  for (let k = 0; k < 3; k++) {
    assert.equal(y[k], x[3*k] - 2*x[3*k+1] + 3*x[3*k+2]);
  }
});

test('packed vector references still supply their corresponding atom at equal lengths', async () => {
  const { ctx } = makeMatCtx(`
x~iid(Normal(0,1),2)
z~iid(Normal(0,1),2)
y=transpose(x)*z
`, { sampleCount: 2, rootSeed: 117 });
  const x = (await ctx.getMeasure('x')).samples;
  const z = (await ctx.getMeasure('z')).samples;
  const y = (await ctx.getMeasure('y')).samples;
  for (let k = 0; k < 2; k++) {
    const want = x[2*k]*z[2*k] + x[2*k+1]*z[2*k+1];
    assert.equal(y[k], want);
  }
});

test('an oriented per-atom right operand forms the outer product', async () => {
  const { ctx } = makeMatCtx('x~iid(Normal(0,1),2)\nz~iid(Normal(0,1),3)\ny=x*transpose(z)',
    { sampleCount: 5, rootSeed: 117 });
  const x = (await ctx.getMeasure('x')).samples;
  const z = (await ctx.getMeasure('z')).samples;
  const y = (await ctx.getMeasure('y')).value;
  assert.deepEqual(y.shape, [5,2,3]);
  for (let k = 0; k < 5; k++) for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++) {
    assert.equal(y.data[6*k+3*i+j], x[2*k+i]*z[3*k+j]);
  }
});

test('an adjoint inner product retains its imaginary scalar result', async () => {
  const { ctx } = makeMatCtx('a~iid(Normal(0,1),2)\nb~iid(Normal(0,1),2)\ny=adjoint(complex.(a,b))*[1.,2.]',
    { sampleCount: 5, rootSeed: 117 });
  const a = (await ctx.getMeasure('a')).samples;
  const b = (await ctx.getMeasure('b')).samples;
  const y = (await ctx.getMeasure('y')).value;
  for (let k = 0; k < 5; k++) {
    assert.equal(y.data[k], a[2*k] + 2*a[2*k+1]);
    assert.equal(y.im?.[k], -b[2*k] - 2*b[2*k+1]);
  }
});
