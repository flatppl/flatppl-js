'use strict';

// §06 truncate intersects support, and normalize divides by that integral.
// These public likelihood queries exercise the resolver and the product gate.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const box = (lo: number, hi: number, d = 2) => `cartprod(${Array(d).fill(`interval(${lo}.0, ${hi}.0)`).join(', ')})`;
const unit = box(0, 1);
function model(weight: string, region = unit, base = region, point = '[0.9, 0.9]') {
  return 'theta ~ Uniform(interval(0.5, 2.0))\n'
    + `M = normalize(truncate(weighted((v, t) -> ${weight}, Lebesgue(${base}), theta), ${region}))\n`
    + `y ~ M\nL = likelihoodof(kernelof(y, theta = theta), ${point})\nlp = logdensityof(L, record(theta = 1.0))\n`;
}
async function score(source: string) {
  const { ctx } = ctxFor(source, 1);
  return (await ctx.getMeasure('lp')).samples[0];
}
function close(got: number, want: number) { assert.ok(Math.abs(got - want) < 2e-12, `${got} vs ${want}`); }

test('axis cuts retain a narrow slab and a narrow rectangle in a truncated box', async () => {
  // Extra mass is 100*1e-4, or 1e6*1e-4*2e-4, respectively.
  close(await score(model('t + 100.0 * ifelse(v[1] in interval(0.5, 0.5001), 1.0, 0.0)')), -Math.log1p(0.01));
  close(await score(model('t + 1.0e6 * ifelse(v[1] in interval(0.2, 0.2001), 1.0, 0.0) '
    + '* ifelse(v[2] in interval(0.7, 0.7002), 1.0, 0.0)')), -Math.log1p(0.02));
});

test('polynomial and affine-exponential box integrals retain axis order', async () => {
  // ∫[0,1]^3 (t+x^9+2yz)=1+1/10+1/2=1.6.
  close(await score(model('t + v[1]^9 + 2.0*v[2]*v[3]', box(0, 1, 3), undefined, '[0.2, 0.3, 0.4]')),
    Math.log((1 + 0.2 ** 9 + 2 * 0.3 * 0.4) / 1.6));
  // ∫[0,1]^2 exp(x+2y)=expm1(1)*expm1(2)/2.
  close(await score(model('exp(t*v[1] + 2.0*v[2])')),
    2.7 - Math.log(Math.expm1(1) * Math.expm1(2) / 2));
});

test('step values are constant between their solved axis cuts', async () => {
  // Four strips have weights 1,2,3,4, so Z=2.5; queried weight is 4.
  close(await score(model('t + floor(4.0*v[2])')), Math.log(4 / 2.5));
});

test('the truncation integral respects the base support intersection', async () => {
  const base = 'cartprod(interval(0.0, 0.5), interval(0.0, 0.5))';
  close(await score(model('t + 0.0*v[1]', unit, base, '[0.2, 0.2]')), Math.log(4));
  assert.equal(await score(model('t + 0.0*v[1]', unit, base, '[0.8, 0.2]')), -Infinity);
  assert.equal(await score(model('t + 0.0*v[1]', unit, base, '[-0.1, 0.2]')), -Infinity);
});

test('separable Gaussian box masses retain the established CDF oracle', async () => {
  // O1: product of standard-normal interval masses on [-2,3]×[-1,2].
  const region = 'cartprod(interval(-2.0, 3.0), interval(-1.0, 2.0))';
  close(await score(model('t*exp(-0.5*(v[1]^2 + v[2]^2))/(2.0*pi)', region, region, '[0.0, 0.0]')),
    -Math.log(2 * Math.PI) + 0.2245614818793499);
  // A Gaussian slab uses an interval-width factor on the absent second axis.
  close(await score(model('t*exp(-0.5*((v[1]-0.3)/0.001)^2)', unit, unit, '[0.3, 0.9]')),
    -Math.log(0.001 * Math.sqrt(2 * Math.PI)));
  // Completing squares: exp(-((x-1)^2+(y+2)^2)/2) has mass π/2 on x≥1,y≤-2.
  const tail = 'cartprod(interval(1.0, inf), interval(-inf, -2.0))';
  close(await score(model('t*exp(-0.5*((v[1]-1.0)^2 + (v[2]+2.0)^2))', tail, tail, '[2.0, -3.0]')),
    -1 - Math.log(Math.PI / 2));
  // Independent erf Taylor series at 1/√2; the omitted term is below 1e-26.
  let term = Math.SQRT1_2, series = 0;
  for (let n = 0; n < 20; n++) { series += term / (2 * n + 1); term *= -0.5 / (n + 1); }
  const centralMass = 2 * series / Math.sqrt(Math.PI);
  const shifted = 'cartprod(interval(99999999.0, 100000001.0), interval(-100000001.0, -99999999.0))';
  close(await score(model('t*exp(-0.5*((v[1]-1.0e8)^2 + (v[2]+1.0e8)^2) - 0.1)', shifted, shifted, '[1.0e8, -1.0e8]')),
    -Math.log(2 * Math.PI) - 2 * Math.log(centralMass));
});

test('an unproved smooth peak refuses instead of disappearing between grid nodes', async () => {
  // Exact Z=1+0.1√(2π), up to tails beyond 300 standard deviations. The
  // former seed grid can see only weight 1, which incorrectly gives log p=0.
  const source = model('t + 100.0*exp(-0.5*((v[1]-0.3)/0.001)^2)');
  await assert.rejects(() => score(source), { code: 'ENGINE_LIMITATION' });
});

test('a generic unbounded Cauchy weight requires a recognized integral', async () => {
  const source = model('t/(pi*5.0*(1.0+(v/5.0)^2))', 'interval(0.0, inf)', undefined, '5.0');
  await assert.rejects(() => score(source), { code: 'ENGINE_LIMITATION' });
});

test('constant pieces require representable interior points and a bounded piece count', async () => {
  // The first true integral is 4, but rounding the midpoint to the lower
  // endpoint would instead integrate the boundary value and produce 2.
  const narrow = 'cartprod(interval(1.0e16, 10000000000000002.0), interval(0.0, 1.0))';
  await assert.rejects(() => score(model('t + ifelse(v[1] > 1.0e16, 1.0, 0.0)', narrow, narrow, '[1.0e16, 0.5]')),
    { code: 'ENGINE_LIMITATION' });
  await assert.rejects(() => score(model('t + floor(500.0*v[1]) + floor(500.0*v[2])')),
    { code: 'ENGINE_LIMITATION' });
});
