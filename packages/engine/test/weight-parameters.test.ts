'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('explicit weight parameters retain the whole vector variate', async () => {
  const { ctx } = makeMatCtx(`
w(x, p) = p[1] + x[1]^2 + x[2]^2
M = weighted(w, iid(Normal(0, 1), 2), [2, 3])
lp = logdensityof(M, [0.5, -1.0])
`);
  const expected = Math.log(3.25) - Math.log(2 * Math.PI) - 0.625;
  assert.ok(Math.abs((await ctx.getMeasure('lp')).value.data[0] - expected) < 1e-12);
});

test('record weight parameters retain the polynomial density', async () => {
  const { ctx } = makeMatCtx(`
w(x, p) = p.offset + p.slope * x
M = weighted(w, Uniform(unitinterval), (offset = 1.0, slope = 2.0))
lp = logdensityof(M, 0.25)
`);
  assert.ok(Math.abs((await ctx.getMeasure('lp')).value.data[0] - Math.log(1.5)) < 1e-12);
});

test('explicit log-weight parameters retain the Normal tilt closed form', async () => {
  const { ctx } = makeMatCtx(`
theta ~ Normal(0, 1)
M = normalize(logweighted((x, p) -> p * x, Normal(0, 1), theta))
score = functionof(logdensityof(M, 0.5), theta)
a = score(-0.75)
b = score(1.25)
c = score(-0.75)
`);
  for (const [name, theta] of [['a', -0.75], ['b', 1.25], ['c', -0.75]] as const) {
    const expected = -0.5 * Math.log(2 * Math.PI) - 0.5 * (0.5 - theta) ** 2;
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-12);
  }
});

test('explicit weight parameters apply a builtin function', async () => {
  const { ctx } = makeMatCtx(`
M = weighted(pow, Uniform(unitinterval), 2)
lp = logdensityof(M, 0.5)
`);
  assert.equal((await ctx.getMeasure('lp')).value.data[0], Math.log(0.25));
});

test('explicit weight parameters reject invalid callbacks and captured draws', () => {
  for (const source of [
    'M = weighted(2, Normal(0, 1), 3)',
    'M = logweighted(x -> x, Normal(0, 1), 3)',
    'M = weighted((x, y, z) -> x + y + z, Normal(0, 1), 3)',
    'M = weighted((x, p) -> x * p, iid(Normal(0, 1), 2), 3)',
    'f = lt\nM = weighted(f, Uniform(unitinterval), 0.5)',
    'M = weighted(sin, Uniform(unitinterval), 2)',
    'theta ~ Normal(0, 1)\nM = logweighted((x, p) -> theta * x + p, Normal(0, 1), 3)',
  ]) {
    const errors = processSource(source).diagnostics.filter((d: any) => d.severity === 'error');
    assert.ok(errors.length > 0, source);
  }
});
