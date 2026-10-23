'use strict';

// Spec §03 (flatppl-design #210): `reals` is the extended real line, "±∞ admitted";
// `posreals` is (0, +∞] and `nonnegreals` is [0, +∞], both "including +∞". So `x in S`
// is true at an infinite endpoint the set contains and false only outside it.
// ORACLE: the set definitions themselves, endpoint by endpoint.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function member(expr: string): Promise<boolean> {
  const { ctx } = ctxFor(`r = ${expr}\n`, 1);
  return (await ctx.getMeasure('r')).value.data[0] === 1;
}

test('reals contains both infinities', async () => {
  assert.equal(await member('inf in reals'), true);
  assert.equal(await member('-inf in reals'), true);
  assert.equal(await member('0.0 in reals'), true);
});

test('posreals is (0, +inf]', async () => {
  assert.equal(await member('inf in posreals'), true);
  assert.equal(await member('1.0 in posreals'), true);
  assert.equal(await member('0.0 in posreals'), false);
  assert.equal(await member('-inf in posreals'), false);
});

test('nonnegreals is [0, +inf]', async () => {
  assert.equal(await member('inf in nonnegreals'), true);
  assert.equal(await member('0.0 in nonnegreals'), true);
  assert.equal(await member('-1.0 in nonnegreals'), false);
  assert.equal(await member('-inf in nonnegreals'), false);
});

test('a Cartesian power takes the same cells', async () => {
  assert.equal(await member('[inf, 1.0] in cartpow(posreals, 2)'), true);
  assert.equal(await member('[1.0, -2.0] in cartpow(posreals, 2)'), false);
});

test('the bounded sets stay bounded', async () => {
  assert.equal(await member('inf in unitinterval'), false);
  assert.equal(await member('inf in interval(0.0, 5.0)'), false);
});
