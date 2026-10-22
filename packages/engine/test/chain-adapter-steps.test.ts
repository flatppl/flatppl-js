'use strict';

// A lambda adapter `pars -> K(x = pars.x)` is a legal `jointchain` / `kchain`
// step (spec §04 "Calling conventions" recommends it for a one-input kernel fed a
// record; spec §06 "Dependent composition" feeds a step the `cat` of every variate
// to its left). Each oracle is the closed form of the chain at the point.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const lnNormal = (x: number, mu: number, sigma: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(sigma) - ((x - mu) / sigma) ** 2 / 2;
const lnExponential = (x: number, rate: number) => Math.log(rate) - rate * x;
// Gamma(shape = 2, rate): Γ(2) = 1.
const lnGamma2 = (x: number, rate: number) => 2 * Math.log(rate) + Math.log(x) - rate * x;

async function score(src: string) {
  const errs = processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), []);
  const { ctx } = ctxFor(src, 4);
  return (await ctx.getMeasure('s')).samples[0];
}
const close = (got: number, want: number) => assert.ok(Math.abs(got - want) < 1e-12, `${got} vs ${want}`);

const RECORD_CHAIN = `
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(b = draw(Exponential(rate = z))), z = z)
jc = jointchain(lawof(record(a = z)), pars -> k(z = pars.a))
`;
const SCALAR_CHAIN = `
a = draw(Normal(mu = 0.0, sigma = 1.0))
b = draw(Exponential(rate = a))
kb = kernelof(b, a = a)
kc = kernelof(draw(Gamma(shape = 2.0, rate = b)), b = b)
jc = jointchain(lawof(a), kb, pars -> kc(b = pars[2]))
`;

test('a record jointchain adapter step scores each component at its slot', async () => {
  const got = await score(RECORD_CHAIN + 's = logdensityof(jc, record(a = 0.3, b = 0.5))\n');
  close(got, lnNormal(0.3, 0, 1) + lnExponential(0.5, 0.3));
});

test('a scalar-cat jointchain adapter step reads its slot of the fed cat', async () => {
  const got = await score(SCALAR_CHAIN + 's = logdensityof(jc, [0.3, 0.5, 0.7])\n');
  close(got, lnNormal(0.3, 0, 1) + lnExponential(0.5, 0.3) + lnGamma2(0.7, 0.5));
});

test('a scalar-cat jointchain prefix keep drops an adapter step', async () => {
  const got = await score(
    SCALAR_CHAIN + 's = logdensityof(pushfwd(fn(get(_, [1, 2])), jc), [0.3, 0.5])\n');
  close(got, lnNormal(0.3, 0, 1) + lnExponential(0.5, 0.3));
});

test('a kchain adapter step marginalizes the closed form', async () => {
  // z ~ N(0, 1), y | z ~ N(z, 2): y ~ N(0, sqrt(1 + 4)).
  const got = await score(`
z = draw(Normal(mu = 0.0, sigma = 1.0))
k = kernelof(record(y = draw(Normal(mu = z, sigma = 2.0))), z = z)
pp = kchain(lawof(record(z = z)), pars -> k(z = pars.z))
s = logdensityof(pp, record(y = 1.5))
`);
  close(got, lnNormal(1.5, 0, Math.sqrt(5)));
});
