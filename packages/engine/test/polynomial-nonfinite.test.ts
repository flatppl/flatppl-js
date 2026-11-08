'use strict';
// The scalar polynomial evaluator seeded Horner with 0, so its first step was
// 0 · x = NaN at x = ±inf. Oracle: IEEE evaluation of a0 + a1 x + a2 x^2,
// which for (1, 2, 3) is +inf at both +inf and -inf, NaN at NaN, and 17 at 2.
// The batched evaluator already seeded with the top coefficient.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function scalar(x: string): Promise<number> {
  const { ctx } = makeMatCtx(`p = polynomial(coefficients = [1.0, 2.0, 3.0], x = ${x})\n`, { sampleCount: 1 });
  return (await ctx.getMeasure('p')).value.data[0];
}

test('the scalar polynomial follows IEEE at non-finite x', async () => {
  assert.equal(await scalar('inf'), Infinity);
  assert.equal(await scalar('-inf'), Infinity);
  assert.ok(Number.isNaN(await scalar('0.0 * inf')));
  assert.equal(await scalar('2.0'), 17);
});

test('an odd top degree keeps the sign of x at -inf', async () => {
  const { ctx } = makeMatCtx('p = polynomial(coefficients = [1.0, 0.0, 0.0, 2.0], x = -inf)\n', { sampleCount: 1 });
  assert.equal((await ctx.getMeasure('p')).value.data[0], -Infinity);
});

test('the batched polynomial agrees with the scalar at an infinite per-atom x', async () => {
  const { ctx } = makeMatCtx(
    'u = draw(Normal(0.0, 1.0))\nx = ifelse(u > 100.0, 0.0, -inf)\np = polynomial(coefficients = [1.0, 2.0, 3.0], x = x)\n',
    { sampleCount: 4 });
  assert.deepEqual(Array.from((await ctx.getMeasure('p')).value.data), [Infinity, Infinity, Infinity, Infinity]);
});
