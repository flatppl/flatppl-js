'use strict';

// `broadcast(logdensityof, M, pts)` scores all points in one worker pass when M
// needs a normaliser resolved first. That pass gives the points axis to the
// walker's N-axis, which only scalar variates have a layout for: an array point
// (one row of an iid plate) was consumed as iid copies instead. The result was
// the points in swapped order for two points, and a refusal ("value has
// unconsumed leftover") for three. Array points now take the per-point route.
//
// ORACLE: closed form. Each coordinate is Normal(0, 1) truncated to [0, 5],
// so with Z = Phi(5) - Phi(0) = 0.4999997133484282 the density of a point
// (a, b) is log N(a) + log N(b) - 2 log Z:
//   [1, 2]     -2.9515815586828387
//   [0.5, 1]   -1.0765815586828387
//   [3, 0.1]   -4.9565815586828395

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const M = 'iid(normalize(truncate(Normal(0, 1), interval(0, 5))), 2)';

async function score(pts: string) {
  const { ctx } = ctxFor(`M = ${M}\nlp = broadcast(logdensityof, M, ${pts})\n`, 1);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples) as number[];
}

function near(got: number, want: number) {
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
}

test('two array points keep their order', async () => {
  const got = await score('[[1.0, 2.0], [0.5, 1.0]]');
  near(got[0], -2.9515815586828387);
  near(got[1], -1.0765815586828387);
});

test('three array points score instead of refusing', async () => {
  const got = await score('[[1.0, 2.0], [0.5, 1.0], [3.0, 0.1]]');
  near(got[0], -2.9515815586828387);
  near(got[1], -1.0765815586828387);
  near(got[2], -4.9565815586828395);
});
