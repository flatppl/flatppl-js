'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function evaluate(source: string) {
  const { ctx } = makeMatCtx(source, { sampleCount: 1 });
  return Array.from<number>((await ctx.getMeasure('score')).samples);
}

test('ifelse joins integer and real branches in either order under a live condition', async () => {
  assert.deepEqual(await evaluate(`
choose(flag) = ifelse(flag, 2, 1.5)
choose_reversed(flag) = ifelse(flag, 1.5, 2)
score = cat(choose.([true,false]), choose_reversed.([true,false]))`), [2,1.5,1.5,2]);
});

test('ifelse preserves complex components when selecting narrower scalar branches', async () => {
  assert.deepEqual(await evaluate(`
choose(flag) = ifelse(flag, 2, complex(3,4))
choose_reversed(flag) = ifelse(flag, complex(3,4), 1.5)
score = [real(choose(true)), imag(choose(true)), real(choose(false)),
  imag(choose(false)), real(choose_reversed(false)), imag(choose_reversed(true))]`), [2,0,3,4,1.5,4]);
});

test('a count-density callback accepts a real support witness without changing its score', async () => {
  const actual = await evaluate(`
score = (y -> builtin_logdensityof(Poisson, record(rate = .7),
  ifelse(y in interval(1.0,inf), y, 1.0))).([0,1,2,3])`);
  const expected = [1,1,2,3].map(k => k*Math.log(.7) - .7
    - Math.log(k === 3 ? 6 : k === 2 ? 2 : 1));
  actual.forEach((value, i) =>
    assert.ok(Math.abs(value - expected[i]) < 1e-12));
});

test('ifelse retains its Boolean, arity, and structural branch checks', () => {
  for (const source of [
    'score = ifelse(1, 2, 1.5)',
    'score = ifelse(true, 2)',
    'score = ifelse(true, 2, 1.5, 3)',
    'score = ifelse(true, 2, record(value = 1))',
    'score = ifelse(true, 2, [1,2])',
  ]) assert.throws(() => makeMatCtx(source), /parse\/analyze errors/);
});
