'use strict';

// Spec §07 `rnginit`: a byte-vector seed holds integers in {0, ..., 255} and "any non-empty
// vector is accepted"; an integer seed lies in {0, ..., 2^64 - 1}. The seed is read when the
// state is used, so each case draws once.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function draws(seed: string) {
  const { ctx } = ctxFor(`s = rnginit(${seed})\nout = rand(Normal(0, 1), s)\nr = get(out, 1)\n`, 1);
  return (await ctx.getMeasure('r')).value.data[0];
}

test('a non-empty byte vector and an integer seed are accepted', async () => {
  assert.ok(Number.isFinite(await draws('[1, 2, 255]')));
  assert.ok(Number.isFinite(await draws('[0]')));
  assert.ok(Number.isFinite(await draws('42')));
});

test('an empty seed, an out-of-range byte and a negative integer are refused', async () => {
  await assert.rejects(draws('fill(0, 0)'), /non-empty byte vector/);
  await assert.rejects(draws('[1, 2, 300]'), /byte vector/);
  await assert.rejects(draws('[1, -2]'), /byte vector/);
  await assert.rejects(draws('-7'), /0\.\.2\^64-1/);
});
