'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('every callable construction must cut or absorb its stochastic ancestors', () => {
  const prefix = 'z ~ Normal(0, 1)\n';
  for (const body of [
    'f = functionof(2*z)',
    'f(x) = x + z',
    'f = fn(_ + z)',
    'f = x -> x + self.z',
    'out = broadcast(x -> x+z, [1, 2])',
    'f = x -> lawof((y -> y+z)(x))',
    'f = functionof(draw(Normal(0, 1)))',
    'f = x -> Normal(x, exp(z))',
  ]) {
    const { diagnostics } = processSource(prefix + body);
    assert.ok(diagnostics.some((d: any) =>
      d.severity === 'error' && d.message.includes('cannot capture')), body);
  }
});

test('explicit boundaries and nested fixed callables evaluate their inputs', async () => {
  const src = `
z ~ Normal(0, 1)
f = functionof(2*self.z, z=z)
g = x -> (functionof(2*z, z=z))(x)
a = f(3)
b = g(4)
`;
  const { ctx, lifted } = makeMatCtx(src, { sampleCount: 1 });
  assert.equal(lifted.bindings.get('f').phase, 'fixed');
  assert.equal(lifted.bindings.get('g').phase, 'fixed');
  assert.equal((await ctx.getMeasure('a')).samples[0], 6);
  assert.equal((await ctx.getMeasure('b')).samples[0], 8);
});

test('law and kernel reification absorb internal randomness', async () => {
  const src = `
mu = elementof(reals)
x ~ Normal(mu, 1)
F = functionof(lawof(x))
K = kernelof(x, mu=mu)
a = logdensityof(F(mu=0), 1)
b = logdensityof(K(0), 1)
`;
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const expected = -0.5 * (Math.log(2 * Math.PI) + 1);
  assert.ok(Math.abs((await ctx.getMeasure('a')).samples[0] - expected) < 1e-14);
  assert.ok(Math.abs((await ctx.getMeasure('b')).samples[0] - expected) < 1e-14);
});

test('fixed functions can be applied to stochastic arguments', async () => {
  const { ctx, lifted } = makeMatCtx(`
z ~ Normal(0, 1)
f = x -> x+1
y = f(z)
`, { sampleCount: 32, rootSeed: 42 });
  assert.equal(lifted.bindings.get('f').phase, 'fixed');
  assert.equal(lifted.bindings.get('y').phase, 'stochastic');
  const z = (await ctx.getMeasure('z')).samples;
  const y = (await ctx.getMeasure('y')).samples;
  assert.deepEqual(Array.from(y), Array.from(z, (v: number) => v + 1));
});
