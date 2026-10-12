'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function score(source: string): Promise<number> {
  const { proc, ctx } = ctxFor(source, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const result = await ctx.getMeasure('score');
  return result.value ? result.value.data[0] : result.samples[0];
}

for (const measure of ['iid(Normal(0,1), [2,2])', 'lawof(z)']) {
  test(`multi-axis iid density consumes every scalar: ${measure}`, async () => {
    const actual = await score(`
      z ~ iid(Normal(0,1), [2,2])
      score = logdensityof(${measure}, rowstack([[.1,-.3],[.4,.2]]))
    `);
    // Four independent standard Normals: -2 log(2 pi) - sum(z^2)/2.
    const expected = -2 * Math.log(2 * Math.PI) - .3 / 2;
    assert.ok(Math.abs(actual - expected) < 1e-12);
  });
}

test('multi-axis iid refuses a different observation shape with equal size', async () => {
  await assert.rejects(() => score(`
    score = logdensityof(iid(Normal(0,1), [2,2]), rowstack([[.1,-.3,.4,.2]]))
  `));
});
