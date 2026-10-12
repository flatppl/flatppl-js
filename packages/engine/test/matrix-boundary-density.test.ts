'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

for (const [point, mean, count] of [
  ['rowstack([[1.,2.],[3.,4.]])', 5, 3],
  ['transpose(rowstack([[1.,2.],[3.,4.]]))', 7, 1],
  ['diagmat([1.,4.])', 1, 3],
] as const) {
  test(`matrix kernel boundary preserves logical entries at ${count} atoms: ${point}`, async () => {
    const { proc, ctx } = ctxFor(`
      z = elementof(cartpow(reals,[2,2]))
      y ~ Normal((z * [1.,2.])[1],1)
      K = kernelof(y,z=z)
      L = likelihoodof(K,8.)
      score = logdensityof(L,${point})
    `, count);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const result = await ctx.getMeasure('score');
    const values = result.value ? result.value.data : result.samples;
    const expected = -.5 * Math.log(2 * Math.PI) - (8 - mean) ** 2 / 2;
    assert.equal(values.length, count);
    for (const value of values) assert.ok(Math.abs(value - expected) < 1e-12);
  });
}
