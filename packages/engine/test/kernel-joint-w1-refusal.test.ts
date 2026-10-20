'use strict';

// §06 fan-out applies each component at its declared boundary. A sibling law
// reads its own ambient parameters; reusing a source does not merge law scopes.
const test = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const scoreOf = async (src: string) => {
  const { ctx } = ctxFor(src, 1);
  return (await ctx.getMeasure('ld')).samples[0];
};
const TOL = 1e-13;
const CENTERED_PRODUCT = -Math.log(2 * Math.PI);

for (const fields of ['a = K, b = M', 'b = M, a = K']) {
  test(`kernel input and sibling ambient parameter stay separate: ${fields}`, async () => {
    const got = await scoreOf(`
z = elementof(reals)
u ~ Normal(z, 1.0)
K = kernelof(u, input = z)
M = lawof(u)
P = joint(${fields})
score = logdensityof(P(input = 3.0), record(a = 3.0, b = 7.0))
F = functionof(score, ambient = z)
ld = F(ambient = 7.0)
`);
    assert.ok(Math.abs(got - CENTERED_PRODUCT) < TOL, `got ${got}`);
  });
}

test('components may bind one source node under distinct input names', async () => {
  const got = await scoreOf(`
z = elementof(reals)
u ~ Normal(z, 1.0)
K1 = kernelof(u, left = z)
K2 = kernelof(u, right = z)
P = joint(a = K1, b = K2)
ld = logdensityof(P(left = -2.0, right = 3.0), record(a = -2.0, b = 3.0))
`);
  assert.ok(Math.abs(got - CENTERED_PRODUCT) < TOL);
});

test('self.K aliases retain independent kernel component boundaries', async () => {
  const source = `
z = elementof(reals)
u ~ Normal(z, 1.0)
K = kernelof(u, z = z)
P = joint(a = self.K, b = self.K)
M = P(z = 1.0)
ld = logdensityof(M, record(a = 1.0, b = 1.0))
`;
  assert.ok(Math.abs(await scoreOf(source) - CENTERED_PRODUCT) < TOL);
  const { ctx } = makeMatCtx(source, { sampleCount: 4096, rootSeed: 391 });
  const { a, b } = (await ctx.getMeasure('M')).fields;
  const centered = (xs: Float64Array) => xs.reduce((sum, x) => sum + x, 0) / xs.length;
  const ma = centered(a.samples), mb = centered(b.samples);
  const cov = a.samples.reduce((sum: number, x: number, i: number) =>
    sum + (x - ma) * (b.samples[i] - mb), 0) / a.samples.length;
  assert.ok(Math.abs(ma - 1) < 0.1);
  assert.ok(Math.abs(mb - 1) < 0.1);
  assert.ok(Math.abs(cov) < 0.1, `covariance ${cov}, oracle 0`);
});

test('an IID law component keeps its ambient input outside a sibling boundary', async () => {
  const got = await scoreOf(`
z = elementof(reals)
u ~ Normal(z, 1.0)
K = kernelof(u, input = z)
M = iid(lawof(u), 2)
P = joint(a = K, b = M)
score = logdensityof(P(input = 3.0), record(a = 3.0, b = [7.0, 7.0]))
F = functionof(score, ambient = z)
ld = F(ambient = 7.0)
`);
  assert.ok(Math.abs(got + 1.5 * Math.log(2 * Math.PI)) < TOL);
});
