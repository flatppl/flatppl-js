'use strict';

// §06 truncate(M, S) restricts M to S. S may be a record set,
// cartprod(name = set, ...), which holds each field in its own set.
//
// ORACLES: Lebesgue has density 1 on its support; truncation keeps it inside
// S and gives -Infinity outside. The log weight adds -sum(s).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function value(src: string, N = 1): Promise<number[]> {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples);
}

const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

test('a record Lebesgue truncated by a record set, under a log weight', async () => {
  const src = 'log_prior(p) = -sum(p.s)\n'
    + 'M = logweighted(log_prior, truncate(Lebesgue(cartprod(s = ordered(2))), cartprod(s = cartpow(posreals, 2))))\n';
  for (const N of [1, 3]) {
    assert.deepEqual(await value(src + 'lp = logdensityof(M, record(s = [0.5, 1.5]))\n', N), new Array(N).fill(-2));
    assert.deepEqual(await value(src + 'lp = logdensityof(M, record(s = [-0.5, 1.5]))\n', N), new Array(N).fill(-Infinity));
  }
});

test('a record joint truncated by a record set', async () => {
  const head = 'M = truncate(joint(a = Normal(0, 1), b = Normal(0, 1)), cartprod(a = posreals, b = reals))\n';
  const inside = await value(head + 'lp = logdensityof(M, record(a = 0.5, b = -0.3))\n');
  assert.ok(Math.abs(inside[0] - (logN(0.5) + logN(-0.3))) < 1e-12, String(inside[0]));
  assert.deepEqual(await value(head + 'lp = logdensityof(M, record(a = -0.5, b = -0.3))\n'), [-Infinity]);
});

test('sampling a truncation to a record set refuses as an engine limitation', async () => {
  const { ctx } = ctxFor('M = truncate(joint(a = Normal(0, 1), b = Normal(0, 1)), cartprod(a = posreals, b = reals))\n', 4);
  await assert.rejects(ctx.getMeasure('M'), /engine limitation: a truncation to a record set/);
});
