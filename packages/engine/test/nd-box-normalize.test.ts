'use strict';

// `normalize(weighted(w, Lebesgue(cartprod(interval, …))))` with a θ-independent
// weight over an ARRAY variate took Z from the materialise fallback, an
// importance-sampling estimate over the sampled atoms, so the score moved with
// the sample count: `v[1] + v[2]` over [0, 1]² gave log-density 0.402 / 0.604 /
// 0.583 at N = 1 / 100 / 4000 against log 1.8 = 0.588. Z is now a deterministic
// cubature, seeded at about 4096 cells and split at axis-aligned discontinuities;
// a jump that is not axis-aligned refuses.
//
// Oracles: closed forms, and scipy dblquad / tplquad at epsabs = epsrel = 1e-14
// (1e-13 for tplquad), split by hand at the step.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const unit2 = 'cartprod(interval(0.0, 1.0), interval(0.0, 1.0))';
async function score(w: string, box: string, point: string, N = 1): Promise<number> {
  const { ctx } = ctxFor(`M = normalize(weighted(${w}, Lebesgue(${box})))\nlp = logdensityof(M, ${point})\n`, N);
  const m = await ctx.getMeasure('lp');
  return m.value ? m.value.data[0] : m.samples[0];
}

const CASES: Array<[string, string, string, string, number]> = [
  ['a smooth weight, any sample count', 'v -> v[1] + v[2]', unit2, '[0.9, 0.9]', Math.log(1.8)],
  ['a step on one axis', 'v -> 1.0 + 9.0 * ifelse(v[1] > 0.5, 1.0, 0.0)', unit2, '[0.9, 0.9]', Math.log(10 / 5.5)],
  ['a narrow square spike', 'v -> 1.0 + 100.0 * ifelse(v[1] in interval(0.5, 0.5001), 1.0, 0.0) '
    + '* ifelse(v[2] in interval(0.5, 0.5001), 1.0, 0.0)', unit2, '[0.9, 0.9]', -Math.log(1 + 100 * 1e-8)],
  ['a non-unit box (scipy dblquad)', 'v -> exp(v[1]) * (1.0 + v[2] * v[2])',
    'cartprod(interval(0.0, 1.0), interval(0.0, 2.0))', '[0.9, 1.5]', -0.0031148992184206747],
  ['a step on the second axis (scipy, split)', 'v -> 1.0 + 4.0 * ifelse(v[2] in interval(0.5, 1.25), 1.0, 0.0)',
    'cartprod(interval(0.0, 1.0), interval(0.0, 2.0))', '[0.9, 0.9]', 0],
  ['three axes (scipy tplquad)', 'v -> v[1] + v[2] + v[3]',
    'cartprod(interval(0.0, 1.0), interval(0.0, 1.0), interval(0.0, 1.0))', '[0.9, 0.9, 0.9]', 0.5877866649021191],
  // Closed form -log(1 + 100·2π·sd²·Φ-terms), mpmath.
  ['a smooth bump of sd 0.01', 'v -> 1.0 + 100.0 * exp(-0.5 * ((v[1] - 0.3)^2 + (v[2] - 0.6)^2) / 0.01^2)',
    unit2, '[0.9, 0.1]', -0.06093690535372161],
  ['an affine window on one axis', 'v -> 1.0 + 100.0 * ifelse(abs(2.0 * v[1] - 1.0) < 1.0e-4, 1.0, 0.0)',
    unit2, '[0.9, 0.9]', -Math.log1p(0.01)],
  ['floor of one axis', 'v -> 1.0 + floor(4.0 * v[2])', unit2, '[0.9, 0.1]', -Math.log(2.5)],
];
for (const [label, w, box, point, want] of CASES) {
  test(`N-D box normalizer, ${label}`, async () => {
    for (const N of [1, 100]) {
      const got = await score(w, box, point, N);
      assert.ok(Math.abs(got - want) < 1e-11, `N = ${N}: ${got} vs ${want}`);
    }
  });
}

for (const [label, w, msg] of [
  ['a sum of coordinates', 'v[1] + v[2] < 1.0', /couples coordinates of the variate, which is not axis-aligned/],
  ['two coordinates compared', 'v[1] < v[2]', /couples coordinates of the variate, which is not axis-aligned/],
  ['a test on mod of a coordinate', 'mod(v[1], 0.1) < 0.05', /`lt` test on an expression that is not affine/],
] as Array<[string, string, RegExp]>) {
  test(`a jump that is not axis-aligned refuses: ${label}`, async () => {
    const { ctx } = ctxFor(`M = normalize(weighted(v -> 1.0 + 9.0 * ifelse(${w}, 1.0, 0.0), `
      + `Lebesgue(${unit2})))\nlp = logdensityof(M, [0.9, 0.9])\n`, 1);
    await assert.rejects(ctx.getMeasure('lp'), msg);
  });
}

for (const [label, w, msg] of [
  ['an everywhere-zero weight', 'v -> 0.0 * v[1]', /normalizer Z = ∫ w dv is 0/],
  ['a weight the cubature cannot resolve', 'v -> 1.0 + sin(1.0e7 * v[1])', /adaptive quadrature did not converge/],
] as Array<[string, string, RegExp]>) {
  test(`N-D box normalizer refuses ${label}`, async () => {
    const { ctx } = ctxFor(`M = normalize(weighted(${w}, Lebesgue(${unit2})))\nlp = logdensityof(M, [0.9, 0.9])\n`, 1);
    await assert.rejects(ctx.getMeasure('lp'), msg);
  });
}
