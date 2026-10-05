'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

test('normal link functions agree across scalar and broadcast tails', () => {
  const { proc, ctx } = ctxFor(`
ps = [1e-20, 0.001, 0.2]
z = probit.(ps)
p = invprobit.(z)
scalar = probit(1e-20)
`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const z = ctx.fixedValues.get('z').data;
  const p = ctx.fixedValues.get('p').data;
  assert.ok(Array.from(z).every(Number.isFinite));
  assert.ok(Math.abs(z[0] - ctx.fixedValues.get('scalar')) < 1e-12);
  for (const [i, expected] of [1e-20, 0.001, 0.2].entries()) {
    assert.ok(Math.abs(p[i] / expected - 1) < 1e-12);
  }
});

test('round uses half-to-even in scalar, broadcast, aggregate, and sampled expressions', async () => {
  const { proc, ctx } = ctxFor(`
a = round(2.5)
b = round(-1.5)
v = [0.5, 1.5, 2.5, -0.5, -1.5, -2.5]
r = round.(v)
s = aggregate(sum, [], round(v[.i]))
x ~ Dirac(value=2.5)
y = round(x)
`, 4);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal(ctx.fixedValues.get('a'), 2);
  assert.equal(ctx.fixedValues.get('b'), -2);
  assert.deepEqual(Array.from(ctx.fixedValues.get('r').data), [0, 2, 2, -0, -2, -2]);
  assert.equal(ctx.fixedValues.get('s'), 0);
  assert.deepEqual(Array.from((await ctx.getMeasure('y')).samples), [2, 2, 2, 2]);
});
