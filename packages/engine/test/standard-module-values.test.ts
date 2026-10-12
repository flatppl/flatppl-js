'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('qualified and nested standard functions evaluate at each sampled input', async () => {
  const { ctx } = makeMatCtx(`
poly = standard_module("polynomials", "0.1")
x ~ Normal(0, 1)
f = poly.legendre
y = f(x = x, n = 3)
z = abs(poly.legendre(3, x))
`, { sampleCount: 32, rootSeed: 117 });
  const x = (await ctx.getMeasure('x')).samples;
  const y = (await ctx.getMeasure('y')).samples;
  const z = (await ctx.getMeasure('z')).samples;
  for (let i = 0; i < x.length; i++) {
    const expected = (5 * x[i] ** 3 - 3 * x[i]) / 2;
    assert.ok(Math.abs(y[i] - expected) < 1e-12);
    assert.ok(Math.abs(z[i] - Math.abs(expected)) < 1e-12);
  }
});

test('standard functions preserve complete array arguments and result shapes per atom', async () => {
  const { ctx } = makeMatCtx(`
pyhf = standard_module("pyhf-helpers", "0.1")
a ~ iid(Normal(0, 1), [2, 2])
y = pyhf.expected_counts(a)
`, { sampleCount: 32, rootSeed: 117 });
  const a = (await ctx.getMeasure('a')).value;
  const y = (await ctx.getMeasure('y')).value;
  assert.deepEqual(y.shape, [32, 2]);
  for (let i = 0; i < 32; i++) {
    assert.equal(y.data[2 * i], a.data[4 * i] + a.data[4 * i + 2]);
    assert.equal(y.data[2 * i + 1], a.data[4 * i + 1] + a.data[4 * i + 3]);
  }
});
