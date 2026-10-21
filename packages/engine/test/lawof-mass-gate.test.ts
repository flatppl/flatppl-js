'use strict';

// §04 reifies value nodes. Measures and kernels are objects, so lawof cannot
// act as a normalization shortcut, regardless of their mass class.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function errorsOf(src: string): string[] {
  return processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

test('lawof of drawn scalar and record values preserves their probability density', async () => {
  const { ctx } = makeMatCtx(`
x ~ Normal(0, 1)
y ~ Beta(1, 1)
scalar = lawof(x)
recordlaw = lawof((a = x, b = y))
a = logdensityof(scalar, 1)
b = logdensityof(recordlaw, (a = 1, b = 0.25))
`, { sampleCount: 1 });
  const expected = -(Math.log(2 * Math.PI) + 1) / 2;
  for (const name of ['a', 'b']) {
    assert.ok(Math.abs((await ctx.getMeasure(name)).value.data[0] - expected) < 1e-14);
  }
});

test('lawof refuses every measure object including normalized and deferred measures', () => {
  for (const source of [
    'm = Normal(0, 1)',
    'm = (1, 2)',
    'm = superpose(weighted(0.3, Normal(0, 1)), weighted(0.7, Normal(1, 1)))',
    'm = relabel(joint(Normal(0, 1), Beta(1, 1)), ["a", "b"])',
    'm = normalize(truncate(Normal(0, 1), interval(0, 1)))',
    'm = truncate(Normal(0, 1), interval(0, 1))',
    'm = weighted(2, Normal(0, 1))',
    'm = Lebesgue(reals)',
    'K(x) = Normal(x, 1)\nL = likelihoodof(K, 0.5)\nm = bayesupdate(L, Normal(0, 1))',
  ]) {
    assert.ok(errorsOf(source + '\nlaw = lawof(m)')
      .some((message) => /lawof expects a value-typed argument/.test(message)), source);
  }
});

test('lawof and kernelof refuse named and builtin kernels as arguments', () => {
  for (const op of ['lawof', 'kernelof']) {
    for (const body of ['Normal', 'K']) {
      const errors = errorsOf(`K(x) = Normal(x, 1)\nlaw = ${op}(${body})`);
      assert.ok(errors.some((message) => /value.typed argument|reifies value nodes/.test(message)),
        `${op}(${body}): ${errors}`);
    }
  }
});

test('functionof reifies an unnormalized measure without changing its mass', async () => {
  const { ctx } = makeMatCtx(`
mu = elementof(reals)
KT = functionof(truncate(Normal(mu, 1), interval(0, 1)), mu)
lp = logdensityof(KT(0), 0.5)
`, { sampleCount: 1 });
  assert.ok(Math.abs((await ctx.getMeasure('lp')).value.data[0]
    + Math.log(2 * Math.PI) / 2 + 0.125) < 1e-14);
  assert.ok(errorsOf('mu = elementof(reals)\nK = kernelof(Normal(mu, 1), mu)')
    .some((message) => /reifies value nodes/.test(message)));
});

test('lawof cannot bypass the draw normalization requirement', async () => {
  const base = 'm = truncate(Normal(0, 1), interval(0, 1))\n';
  assert.ok(errorsOf(base + 'x ~ m').some((message) => /draw requires/.test(message)));
  assert.ok(errorsOf(base + 'x ~ lawof(m)')
    .some((message) => /lawof expects a value-typed argument/.test(message)));
  const { ctx } = makeMatCtx(`
m = normalize(truncate(Normal(0, 1), interval(0, 1)))
x ~ m
law = lawof(x)
lp = logdensityof(law, 0.5)
`, { sampleCount: 1 });
  // Phi(1) - Phi(0), independently tabulated Normal probability.
  const expected = -Math.log(2 * Math.PI) / 2 - 0.125 - Math.log(0.3413447460685429);
  assert.ok(Math.abs((await ctx.getMeasure('lp')).value.data[0] - expected) < 1e-12);
});
