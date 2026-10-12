'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('an inline weight binds derived reification inputs at the scored point', async () => {
  const { ctx } = makeMatCtx(`
a ~ Normal(0, 1)
d = 2 * a
f = functionof(d, a = a)
M = logweighted(p -> f(a = p), Normal(0, 1))
score = logdensityof(M, 2)
`, { sampleCount: 1 });
  const score = (await ctx.getMeasure('score')).samples[0];
  const expected = -0.5 * Math.log(2 * Math.PI) - 2 + 4;
  assert.ok(Math.abs(score - expected) < 1e-12, `${score} != ${expected}`);
});

test('a unary log weight receives the whole record variate', async () => {
  const { ctx } = makeMatCtx(`
M = logweighted(p -> p.a, joint(a = Normal(0, 1)))
score = logdensityof(M, record(a = 2))
`, { sampleCount: 1 });
  const score = (await ctx.getMeasure('score')).samples[0];
  const expected = -0.5 * Math.log(2 * Math.PI) - 2 + 2;
  assert.ok(Math.abs(score - expected) < 1e-12, `${score} != ${expected}`);
});

test('a Normal density broadcast weights ordered Lebesgue at the scored vector', async () => {
  const { ctx } = makeMatCtx(`
M = logweighted(fn(sum(logdensityof.(Normal(0, 5), _))), Lebesgue(ordered(8)))
score = logdensityof(M, [-4, -3, -2, -1, 0, 1, 2, 3])
`, { sampleCount: 4 });
  const score = (await ctx.getMeasure('score')).samples[0];
  const expected = -8 * Math.log(5 * Math.sqrt(2 * Math.PI)) - 44 / 50;
  assert.ok(Math.abs(score - expected) < 1e-12, `${score} != ${expected}`);
});

test('a held Normal in a zero-sum weight resolves the supplied scale', async () => {
  const { ctx } = makeMatCtx(`
s = elementof(posreals)
N = Normal(0, s)
M = logweighted(fn(sum(logdensityof.(N, _))), Lebesgue(zerosum(3)))
score = logdensityof(M, [-1, 0, 1])
f = functionof(score, scale = s)
a = f(2)
b = f(3)
`, { sampleCount: 4 });
  const a = (await ctx.getMeasure('a')).samples[0];
  const b = (await ctx.getMeasure('b')).samples[0];
  const expected = (s: number) => -3 * Math.log(s * Math.sqrt(2 * Math.PI)) - 1 / s ** 2;
  assert.ok(Math.abs(a - expected(2)) < 1e-12, `${a} != ${expected(2)}`);
  assert.ok(Math.abs(b - expected(3)) < 1e-12, `${b} != ${expected(3)}`);
});
