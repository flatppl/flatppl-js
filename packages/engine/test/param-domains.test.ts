'use strict';

// §08/§09 parameter value sets: a parameter outside its set raises an error that
// names the distribution and the parameter. NaN propagates, and a variate
// outside the support still scores -Infinity.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const densityPrims = require('../density-prims.ts');
const sampler = require('../sampler.ts');
const paramDomains = require('../param-domains.ts');

const outside = (kernel: string, name: string) =>
  new RegExp(`${kernel}: parameter '${name}' = .* is outside its declared value set`);

test('every scalar set rejects its boundary violations through builtin_logdensityof', () => {
  const cases: Array<[string, Record<string, any>, any, string]> = [
    ['Normal', { mu: 0, sigma: 0 }, 0.5, 'sigma'],
    ['Normal', { mu: 0, sigma: -1 }, 0.5, 'sigma'],
    ['LogNormal', { mu: 0, sigma: -2 }, 1, 'sigma'],
    ['Exponential', { rate: 0 }, 1, 'rate'],
    ['Gamma', { shape: -1, rate: 1 }, 1, 'shape'],
    ['Beta', { alpha: 2, beta: 0 }, 0.5, 'beta'],
    ['StudentT', { nu: -3 }, 0, 'nu'],
    ['VonMises', { mu: 0, kappa: -1 }, 0, 'kappa'],
    ['Poisson', { rate: -0.5 }, 0, 'rate'],
    ['Bernoulli', { p: 1.5 }, 1, 'p'],
    ['Bernoulli', { p: -0.1 }, 0, 'p'],
    ['Binomial', { n: 0, p: 0.5 }, 0, 'n'],
    ['Binomial', { n: 2.5, p: 0.5 }, 1, 'n'],
    ['Geometric', { p: 2 }, 1, 'p'],
    ['NegativeBinomial2', { mu: 0, psi: 1 }, 1, 'mu'],
    ['CrystalBall', { m0: 0, sigma: -1, alpha: 1, n: 2 }, 0, 'sigma'],
  ];
  for (const [kernel, params, x, name] of cases) {
    assert.throws(() => densityPrims.builtinLogdensityof(kernel, params, x), outside(kernel, name),
      `${kernel}(${JSON.stringify(params)})`);
  }
});

test('vector and matrix sets reject a non-simplex, a negative cell and a non-positive-definite matrix', () => {
  assert.throws(() => densityPrims.builtinLogdensityof('Categorical', { p: [0.5, 0.6] }, 1),
    outside('Categorical', 'p'));
  assert.throws(() => densityPrims.builtinLogdensityof('Categorical0', { p: [-0.5, 1.5] }, 0),
    outside('Categorical0', 'p'));
  assert.throws(() => densityPrims.builtinLogdensityof('Multinomial', { n: 2, p: [0.2, 0.2] }, [1, 1]),
    outside('Multinomial', 'p'));
  assert.throws(() => densityPrims.builtinLogdensityof('Dirichlet', { alpha: [1, -1] }, [0.5, 0.5]),
    outside('Dirichlet', 'alpha'));
  const notPd = { shape: [2, 2], data: new Float64Array([1, 2, 2, 1]) };
  assert.throws(() => densityPrims.builtinLogdensityof('MvNormal', { mu: [0, 0], cov: notPd }, [0, 0]),
    outside('MvNormal', 'cov'));
  assert.throws(() => densityPrims.builtinLogdensityof('Wishart', { nu: 3, scale: notPd },
    { shape: [2, 2], data: new Float64Array([1, 0, 0, 1]) }), outside('Wishart', 'scale'));
});

test('a simplex tolerates the rounding of its sum, 8 n eps of one', () => {
  // 0.1 + 0.2 + 0.7 rounds above one in binary64.
  assert.doesNotThrow(() => paramDomains.checkParam('Categorical', 'p', [0.1, 0.2, 0.7]));
  const n = 4, tolerance = paramDomains.simplexTolerance(n);
  assert.equal(tolerance, 8 * n * Number.EPSILON);
  assert.doesNotThrow(() => paramDomains.checkParam('Categorical', 'p', [0.25, 0.25, 0.25, 0.25 + tolerance / 2]));
  assert.throws(() => paramDomains.checkParam('Categorical', 'p', [0.25, 0.25, 0.25, 0.25 + 2 * tolerance]),
    outside('Categorical', 'p'));
  assert.throws(() => paramDomains.checkParam('Categorical', 'p', [Infinity, 0]), outside('Categorical', 'p'));
});

test('NaN propagates without an error, and a variate outside the support still scores -Infinity', () => {
  assert.ok(Number.isNaN(densityPrims.builtinLogdensityof('Normal', { mu: 0, sigma: NaN }, 0.5)));
  assert.ok(Number.isNaN(densityPrims.builtinLogdensityof('Poisson', { rate: NaN }, 1)));
  assert.equal(densityPrims.builtinLogdensityof('Exponential', { rate: 2 }, -1), -Infinity);
  assert.equal(densityPrims.builtinLogdensityof('Beta', { alpha: 2, beta: 2 }, 1.5), -Infinity);
  assert.equal(densityPrims.builtinLogdensityof('Poisson', { rate: 2 }, -1), -Infinity);
});

test('closed-set boundaries and infinities stay valid where §03 admits them', () => {
  // Poisson's rate is nonnegreals: zero is a point mass at zero.
  assert.equal(densityPrims.builtinLogdensityof('Poisson', { rate: 0 }, 0), 0);
  // unitinterval is closed.
  assert.equal(densityPrims.builtinLogdensityof('Bernoulli', { p: 1 }, 1), 0);
  // posreals admits +inf (§03 "Note on infinities").
  assert.doesNotThrow(() => densityPrims.builtinLogdensityof('Exponential', { rate: Infinity }, 1));
  // No integer set holds an infinity.
  assert.throws(() => densityPrims.builtinLogdensityof('Binomial', { n: Infinity, p: 0.5 }, 1),
    outside('Binomial', 'n'));
});

test('valid parameters keep their closed-form values', () => {
  const close = (a: number, b: number) => assert.ok(Math.abs(a - b) < 1e-12, `${a} != ${b}`);
  // log N(0.7; 0.3, 1.2) = -log(1.2) - log(2 pi)/2 - ((0.7 - 0.3)/1.2)^2/2
  close(densityPrims.builtinLogdensityof('Normal', { mu: 0.3, sigma: 1.2 }, 0.7),
    -Math.log(1.2) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((0.7 - 0.3) / 1.2) ** 2);
  // log Poisson(3; 2.5) = 3 log 2.5 - 2.5 - log 6
  close(densityPrims.builtinLogdensityof('Poisson', { rate: 2.5 }, 3), 3 * Math.log(2.5) - 2.5 - Math.log(6));
  // log Categorical(2; [0.2, 0.3, 0.5]) = log 0.3
  close(densityPrims.builtinLogdensityof('Categorical', { p: [0.2, 0.3, 0.5] }, 2), Math.log(0.3));
  // log MvNormal(x; 0, diag(1, 4)) at x = [1, 2]: -log(2 pi) - log(2) - (1 + 1)/2
  close(densityPrims.builtinLogdensityof('MvNormal',
    { mu: [0, 0], cov: { shape: [2, 2], data: new Float64Array([1, 0, 0, 4]) } }, [1, 2]),
  -Math.log(2 * Math.PI) - Math.log(2) - 1);
});

test('the walker, the sampler and the prepared factory raise the same error', async () => {
  const { ctx } = makeMatCtx('s = 1.0 - 3.0\nr = logdensityof(Normal(0, s), 0.5)', { sampleCount: 1 });
  await assert.rejects(() => ctx.getMeasure('r'), outside('Normal', 'sigma'));
  const { ctx: drawn } = makeMatCtx('x ~ Gamma(2, -1)', { sampleCount: 4 });
  await assert.rejects(() => drawn.getMeasure('x'), outside('Gamma', 'rate'));
  const entry = sampler._internal.REGISTRY.Normal;
  assert.throws(() => entry.logpdfFn.factory(0, -1), outside('Normal', 'sigma'));
  assert.throws(() => new entry.Ctor(0, -1), outside('Normal', 'sigma'));
  assert.throws(() => entry.randFn.factory(0, -1, { prng: Math.random }), outside('Normal', 'sigma'));
  const parametric = entry.randFn.factory({ prng: Math.random });
  assert.throws(() => parametric(0, -1), outside('Normal', 'sigma'));
});
