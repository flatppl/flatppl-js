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
    'r = rnginit(1)\nf(x) = rand(Normal(z+x, 1), r)',
    's, r = rand(Normal(z, 1), rnginit(1))\nf(x) = x+s',
    's, r = rand(Normal(self.z, 1), rnginit(1))\nf(x) = x+s',
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

test('explicit boundaries cut descendant nodes before tracing parameter leaves', async () => {
  const source = `
a = elementof(reals)
b = elementof(reals)
offset = 2
c = a + b
y ~ Normal(c + offset, 1)
K = kernelof(y, c)
lp = logdensityof(K(3), 4)
`;
  const { ctx } = makeMatCtx(source, { sampleCount: 1 });
  assert.ok(Math.abs((await ctx.getMeasure('lp')).value.data[0]
    + (Math.log(2 * Math.PI) + 1) / 2) < 1e-14);
  const errors = processSource(source.replace('kernelof(y, c)', 'kernelof(y, a)'))
    .diagnostics.filter((d: any) => d.severity === 'error');
  assert.ok(errors.length > 0, 'b must not survive as a hidden parameter');
  assert.ok(processSource(`
s, r = rand(Normal(elementof(reals), 1), rnginit(1))
f(x) = x+s
`).diagnostics.some((d: any) => d.severity === 'error'));
});

test('reifying a fixed kernel application preserves its conditional law', async () => {
  const { ctx } = makeMatCtx(`
a = elementof(reals)
K(x) = Normal(2*x, 3)
y ~ K(a)
R = kernelof(y, a)
lp = logdensityof(R(2), 1)
`, { sampleCount: 1 });
  const expected = -Math.log(3) - (Math.log(2 * Math.PI) + 1) / 2;
  assert.ok(Math.abs((await ctx.getMeasure('lp')).value.data[0] - expected) < 1e-14);
});

test('a fixed draw from a closed law remains available to a callable', async () => {
  for (const source of [`
z ~ Normal(0, 1)
M = lawof(self.z)
s, unused = rand(M, rnginit(1))
`, `
Normal ~ base.Normal(0, 1)
exp ~ base.Normal(0, 1)
s, unused = rand(base.Normal(base.exp(0), 1), rnginit(1))
`]) {
    const { ctx } = makeMatCtx(source + `
f(x) = x + s
out = f(2)
`, { sampleCount: 1 });
    assert.equal((await ctx.getMeasure('out')).value.data[0],
      2 + (await ctx.getMeasure('s')).value.data[0]);
  }
});
