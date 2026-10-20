'use strict';

// Inline kernelof/functionof components retain their input boundaries and
// contribute independent measures, just like named kernel components (§06).
const test = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const scoreOf = async (src: string) => {
  const { ctx } = ctxFor(src, 1);
  return (await ctx.getMeasure('ld')).samples[0];
};
const TOL = 1e-13;
// N(0,√2) at 1 times N(0,1) at .5.
const PRODUCT_ORACLE = -Math.log(2 * Math.PI) - 0.5 * Math.log(2) - 0.375;
const MODEL = `
z = elementof(reals)
u ~ Normal(mu = z, sigma = 1.0)
a1 ~ Normal(mu = u, sigma = 1.0)
K1 = kernelof(a1, z = z)
`;

for (const component of ['kernelof(u, z = z)', 'functionof(lawof(u), z = z)']) {
  test(`inline ${component} scores the independent product at each input`, async () => {
    for (const z of [0, 5]) {
      const got = await scoreOf(MODEL + `
KJ = joint(p = K1, q = ${component})
ld = logdensityof(KJ(z = ${z}.0), record(p = ${z + 1}.0, q = ${z + 0.5}))
`);
      assert.ok(Math.abs(got - PRODUCT_ORACLE) < TOL, `got ${got}, oracle ${PRODUCT_ORACLE}`);
    }
  });
}

test('an inline positional kernel component contributes its own density', async () => {
  const got = await scoreOf(MODEL + `
KJ = joint(K1, kernelof(u, z = z))
ld = logdensityof(KJ(z = 0.0), [1.0, 0.5])
`);
  assert.ok(Math.abs(got - PRODUCT_ORACLE) < TOL);
});

test('a closed inline law component ignores the fanned input', async () => {
  const got = await scoreOf(`
z = elementof(reals)
u ~ Normal(0.0, 1.0)
a1 ~ Normal(u + z, 1.0)
K1 = kernelof(a1, z = z)
KJ = joint(p = K1, q = lawof(u))
ld = logdensityof(KJ(z = 5.0), record(p = 6.0, q = 5.5))
`);
  const oracle = -Math.log(2 * Math.PI) - 0.5 * Math.log(2) - 0.25 - 5.5 ** 2 / 2;
  assert.ok(Math.abs(got - oracle) < TOL);
});

test('an inline law reads its ambient input independently of a kernel boundary', async () => {
  const got = await scoreOf(MODEL + `
KJ = joint(p = K1, q = lawof(u))
score = logdensityof(KJ(z = 3.0), record(p = 3.0, q = 7.0))
F = functionof(score, ambient = z)
ld = F(ambient = 7.0)
`);
  const oracle = -Math.log(2 * Math.PI) - 0.5 * Math.log(2);
  assert.ok(Math.abs(got - oracle) < TOL, `got ${got}, oracle ${oracle}`);
});

test('inline components may bind one source under different input names', async () => {
  const got = await scoreOf(MODEL + `
KJ = joint(p = K1, q = kernelof(u, other = z))
ld = logdensityof(KJ(z = 5.0, other = -2.0), record(p = 6.0, q = -1.5))
`);
  assert.ok(Math.abs(got - PRODUCT_ORACLE) < TOL);
});
