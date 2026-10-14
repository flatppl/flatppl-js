'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { lowerMeasure, feedInputs } = require('../clm.ts');
const { logDensityN } = require('../density.ts');

const normalAtZero = (mu: number) => -.5 * Math.log(2 * Math.PI) - .5 * mu ** 2;

async function score(ctx: any, boundaries: any) {
  const node = lowerMeasure('y', ctx, { boundaries });
  const fed = await feedInputs(node, ctx);
  const value = logDensityN(node.body, 0, fed.refArrays, 1, { baseEnv: fed.fixedEnv })[0];
  return { value, fixed: fed.fixedEnv };
}

const MATRIX = `
d = external(cartpow(reals, 2))
X = rowstack([d, d .+ 1])
b = elementof(cartpow(reals, 2))
y ~ Normal(sum(X * b), 1)
`;

test('scoring reuses the prepared matrix and keeps parameter inputs live', async () => {
  const { ctx } = ctxFor(MATRIX, 1);
  ctx.fixedValues.set('d', [1, 2]);
  for (const [b, mu] of [[[.2, .3], 2.1], [[.4, -.1], .7]] as const) {
    const result = await score(ctx, { b });
    assert.strictEqual(result.fixed.X, ctx.fixedValues.get('X'));
    assert.ok(Math.abs(result.value - normalAtZero(mu)) < 1e-12);
  }
});

test('fixed matrices belong to their initialized context', async () => {
  const parsed = processSource(MATRIX);
  const a = orchestrator.buildDerivations(parsed.linkedBindings);
  const b = orchestrator.buildDerivations(parsed.linkedBindings);
  a.sampleCount = b.sampleCount = 1;
  a.fixedValues.set('d', [1, 2]);
  b.fixedValues.set('d', [3, 4]);
  for (const [ctx, mu] of [[a, 2.1], [b, 4.1], [a, 2.1]] as const) {
    const result = await score(ctx, { b: [.2, .3] });
    assert.ok(Math.abs(result.value - normalAtZero(mu)) < 1e-12);
  }
});

test('boundary substitution reaches fixed arrays through named callables', async () => {
  for (const d of ['2', 'elementof(reals)']) {
    const { ctx } = ctxFor(`
d = ${d}
e = d + 1
f(x) = x * e
X = f.([1, 2])
y ~ Normal(sum(X), 1)
`, 1);
    const result = await score(ctx, { d: 4 });
    assert.ok(Math.abs(result.value - normalAtZero(15)) < 1e-12);
  }
});

test('reified arguments remain separate from their original binding', async () => {
  const { ctx } = ctxFor(`
d = elementof(reals)
e = d + 1
f = functionof(e, d = d)
X = f(2)
y ~ Normal(X + d, 1)
`, 1);
  const result = await score(ctx, { d: 5 });
  assert.ok(Math.abs(result.value - normalAtZero(8)) < 1e-12);
});

test('named scalar bounds retain exact weighted Lebesgue normalization', async () => {
  for (const bounds of ['lo = 0\nhi = 1', 'bounds = [0, 1]\nlo = bounds[1]\nhi = get0(bounds, 1)']) {
    const { ctx } = ctxFor(`
${bounds}
S = interval(lo, hi)
B = Lebesgue(support = S)
w(x) = x
M = normalize(weighted(w, B))
s = logdensityof(M, .75)
`, 1);
    const result = await ctx.getMeasure('s');
    // Integral of x over [0,1] is 1/2, hence density(.75) = 1.5.
    assert.ok(Math.abs((result.value?.data || result.samples)[0] - Math.log(1.5)) < 1e-10);
  }
});
