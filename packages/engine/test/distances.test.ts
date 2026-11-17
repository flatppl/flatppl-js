'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const prefix = 'd = standard_module("distances", "0.1")\n';
const close = (a: number, b: number, tol = 2e-14) =>
  assert.ok(Math.abs(a - b) <= tol * Math.max(1, Math.abs(b)), `${a} != ${b}`);

test('distance metrics match independent norms and KL values through qualified calls', async () => {
  const { ctx } = makeMatCtx(prefix + `
norms = [d.euclidean([0, 0], [3, 4]), d.squared_euclidean([0, 0], [3, 4]),
  d.manhattan([0, 0], [3, 4]), d.chebyshev([0, 0], [3, 4]),
  d.minkowski(p = 3, u = [0, 0], v = [3, 4]), d.cosine([1, 0], [0, 1]),
  d.jensenshannon([1, 0], [0, 1]), d.jensenshannon([1, 0, 0], [0, 1, 0])]
large = d.euclidean([1e200, 0], [0, 1e200])
near = d.jensenshannon([0.50000001, 0.49999999], [0.5, 0.5])
`, { sampleCount: 1 });
  const expected = [5, 25, 7, 4, Math.cbrt(91), 1, Math.sqrt(Math.log(2)), Math.sqrt(Math.log(2))];
  Array.from<number>((await ctx.getMeasure('norms')).samples).forEach((v, i) => close(v, expected[i]));
  close((await ctx.getMeasure('large')).samples[0] / 1e200, Math.sqrt(2));
  close((await ctx.getMeasure('near')).samples[0] / 1e-8, 1 / Math.sqrt(2), 1e-8);
});

test('pairwise callbacks retain positional order, asymmetry, diagonal and captured values', async () => {
  const { ctx } = makeMatCtx(prefix + `
offset = 2.0
directed(a, b) = sum(a) - sum(b) + offset
f = directed
P = d.pairwise_distance(distance = f, x = [[0, 0], [3, 4]])
C = d.cross_distance(d.euclidean, [[0, 0], [3, 4]], [[0, 4]])
`, { sampleCount: 1 });
  const p = await ctx.getMeasure('P'), c = await ctx.getMeasure('C');
  assert.deepEqual(p.intrinsicShape, [2, 2]);
  assert.deepEqual(Array.from(p.samples), [2, -5, 9, 2]);
  assert.deepEqual(c.intrinsicShape, [2, 1]);
  assert.deepEqual(Array.from(c.samples), [4, 3]);
});

test('distance calls and higher-order callbacks retain each sampled vector', async () => {
  const { ctx } = makeMatCtx(prefix + `
u ~ iid(Normal(0, 1), 2)
metric = d.euclidean
y = metric(u, [0, 0])
P = d.cross_distance(metric, [u], [[3, 4]])
B = d.euclidean.([[0, 0], [3, 4]], [[0, 4]])
`, { sampleCount: 24, rootSeed: 117 });
  const u = (await ctx.getMeasure('u')).value;
  const y = (await ctx.getMeasure('y')).samples, p = (await ctx.getMeasure('P')).samples;
  for (let i = 0; i < 24; i++) {
    close(y[i], Math.hypot(u.data[2 * i], u.data[2 * i + 1]));
    close(p[i], Math.hypot(u.data[2 * i] - 3, u.data[2 * i + 1] - 4));
  }
  assert.deepEqual(Array.from((await ctx.getMeasure('B')).samples), [4, 3]);
});

test('empty collections, zero norms and tuple-held callbacks preserve their domains', async () => {
  const { ctx } = makeMatCtx(prefix + `
empty = fill([0.0, 0.0], 0)
P = d.pairwise_distance(d.euclidean, empty)
C = d.cross_distance(d.euclidean, [[1.0, 2.0]], empty)
metrics = [d.euclidean(zeros(0), zeros(0)), d.manhattan(zeros(0), zeros(0)),
  d.chebyshev(zeros(0), zeros(0)), d.minkowski([0, 0], [1, 4], 0.5),
  d.cosine([1e200, 0], [1e200, 0])]
callbacks = (d.euclidean, 2)
Q = d.pairwise_distance(callbacks[1], [[0, 0], [3, 4]])
`, { sampleCount: 1 });
  const p = await ctx.getMeasure('P'), c = await ctx.getMeasure('C');
  assert.deepEqual(p.intrinsicShape, [0, 0]);
  assert.deepEqual(c.intrinsicShape, [1, 0]);
  assert.equal(p.samples.length, 0);
  assert.equal(c.samples.length, 0);
  assert.deepEqual(Array.from((await ctx.getMeasure('metrics')).samples), [0, 0, 0, 9, 0]);
  assert.deepEqual(Array.from((await ctx.getMeasure('Q')).samples), [0, 5, 5, 0]);
});
