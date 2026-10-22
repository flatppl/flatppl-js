'use strict';

// `truncate(M, cartpow(S, n))` restricts an n-vector variate to the Cartesian
// power of S: EVERY coordinate must lie in S (spec §03 "Cartesian power",
// §06 truncate). `parseSetIR` unwraps `cartpow(S, n)` to its element set S, and
// `walkTruncate` then gated only the first coordinate, so
// `logdensityof(truncate(iid(Normal(0, 1), 2), cartpow(interval(0, 5), 2)), [1, 9])`
// scored -42.84 instead of -Infinity while `[9, 1]` was -Infinity.
//
// ORACLE: closed form. Inside the box the density is the base density,
// Σ_i log N(x_i; 0, 1), and outside it is -Infinity. Constants computed by hand:
//   [0.5, 1.0]       -2.4628770664093453
//   [1.0, 1.0]       -2.8378770664093453
//   [1.0, 2.0, 0.5]  -5.3818155996140185

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const density = require('../density.ts');
const values = require('../value.ts');
const { parseSetIR } = require('../ir-shared.ts');
const { processSource } = require('../index.ts');

const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

async function perPoint(measure: string, pts: number[][]) {
  const src = `M = ${measure}\n` + pts.map((p, i) => `s${i} = logdensityof(M, [${p.join(', ')}])\n`).join('');
  const { ctx } = ctxFor(src, 1);
  const out: number[] = [];
  for (let i = 0; i < pts.length; i++) out.push((await ctx.getMeasure('s' + i)).samples[0]);
  return out;
}

const BOX = 'truncate(iid(Normal(0, 1), 2), cartpow(interval(0, 5), 2))';
const POINTS = [[0.5, 1.0], [1.0, 9.0], [9.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0], [0.0, 5.0]];
const INSIDE = [true, false, false, false, false, true, true];

test('a coordinate outside the box sends the point to -Infinity, whichever coordinate it is', async () => {
  const got = await perPoint(BOX, POINTS);
  POINTS.forEach((p, i) => {
    const want = INSIDE[i] ? logN(p[0]) + logN(p[1]) : -Infinity;
    assert.equal(got[i], want, `${JSON.stringify(p)}: got ${got[i]}, want ${want}`);
  });
  assert.ok(Math.abs(got[0] - -2.4628770664093453) < 1e-12);
  assert.ok(Math.abs(got[5] - -2.8378770664093453) < 1e-12);
});

test('a named element set gates every coordinate too', async () => {
  const got = await perPoint('truncate(iid(Normal(0, 1), 3), cartpow(posreals, 3))',
    [[1, 2, 0.5], [-1, 2, 0.5], [1, -2, 0.5], [1, 2, -0.5], [1, 2, 0]]);
  assert.ok(Math.abs(got[0] - -5.3818155996140185) < 1e-12);
  assert.deepEqual(got.slice(1), [-Infinity, -Infinity, -Infinity, -Infinity]);
});

test('a truncated block inside a joint gates its own coordinates only', async () => {
  const { ctx } = ctxFor(`
M = joint(truncate(iid(Normal(0, 1), 2), cartpow(posreals, 2)), Normal(0, 1))
inside = logdensityof(M, [1.0, 2.0, 0.5])
second = logdensityof(M, [1.0, -2.0, 0.5])
first = logdensityof(M, [-1.0, 2.0, 0.5])
last = logdensityof(M, [1.0, 2.0, -0.5])
`, 1);
  const at = async (n: string) => (await ctx.getMeasure(n)).samples[0];
  assert.ok(Math.abs(await at('inside') - -5.3818155996140185) < 1e-12);
  assert.equal(await at('second'), -Infinity);
  assert.equal(await at('first'), -Infinity);
  // The trailing Normal is not part of the truncated block, so a negative value
  // there is in the support.
  assert.ok(Math.abs(await at('last') - (logN(1) + logN(2) + logN(-0.5))) < 1e-12);
});

test('points-batched scoring of the box gates each point on all its coordinates', async () => {
  // Through broadcast(logdensityof, M, pts): the batched path, plain-array points.
  const src = `M = ${BOX}\nlp = broadcast(logdensityof, M, [${POINTS.map((p) => '[' + p.join(', ') + ']').join(', ')}])\n`;
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('lp');
  const got = Array.from((m.value?.data ?? m.samples) as ArrayLike<number>);
  assert.deepEqual(got, POINTS.map((p, i) => (INSIDE[i] ? logN(p[0]) + logN(p[1]) : -Infinity)));
});

test('points held as a Value are gated per point on every coordinate', () => {
  // Lebesgue(R^2) has log density 0, so the score is 0 in the box and -Infinity outside.
  const ir = processSource('M = truncate(Lebesgue(support = cartpow(reals, 2)), cartpow(interval(0, 5), 2))\n')
    .loweredModule.bindings.get('M').rhs;
  const opts = { parseSet: (s: any) => parseSetIR(s, null), pointsBatched: true };
  const want = POINTS.map((_p, i) => (INSIDE[i] ? 0 : -Infinity));
  assert.deepEqual(Array.from(density.logDensityN(ir, values.asValue(POINTS), {}, POINTS.length, opts)), want);
  assert.deepEqual(Array.from(density.logDensityN(ir, POINTS, {}, POINTS.length, opts)), want);
});
