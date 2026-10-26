'use strict';

// `normalize(weighted(w, Lebesgue(S)))` with a θ-independent weight resolves Z by
// deterministic quadrature. A fixed 8192-node midpoint rule could not see a
// narrow feature: the spike `100 · [x ∈ (0.5, 0.5001)]` over [0, 1] scored
// -0.012133 against -0.009950 (testsuite corpora/stablehlo/
// integration_narrow_interval), and on the truncate spelling the seeded cubature
// grid missed it entirely (log-density 0). The integral is now split at the
// weight's discontinuities and each piece integrated adaptively.
//
// Oracle: scipy.integrate.quad with points=[lo, hi], epsabs = epsrel = 1e-14,
// and the closed form -log(1 + 100 (hi - lo)) (over [0, 2]: -log(2 + 100 (hi - lo))).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const CASES: Array<[number, number, number, number]> = [
  // lo, hi, log-density at a point of weight 1 over [0, 1], and over [0, 2]
  [0.26, 0.27, -0.6931471805599457, -1.09861228866811],
  [0.5, 0.5001, -0.009950330853166992, -0.6981347220709838],
  [0.1, 0.1001, -0.009950330853166992, -0.6981347220709838],
];

async function score(src: string, name: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure(name);
  return m.value ? m.value.data[0] : m.samples[0];
}

for (const [lo, hi, unit, wide] of CASES) {
  const w = `fn(1.0 + 100.0 * ifelse(_ in interval(${lo}, ${hi}), 1.0, 0.0))`;
  const spellings: Array<[string, string, number]> = [
    ['weighted Lebesgue', `M = normalize(weighted(${w}, Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.8)\n`, unit],
    ['a record-parameter weight', `weight(x, p) = 1.0 + 100.0 * ifelse(in(x, interval(p.lo, p.hi)), 1.0, 0.0)\n`
      + `M = normalize(weighted(weight, Lebesgue(interval(0.0, 1.0)), record(lo = ${lo}, hi = ${hi})))\n`
      + `lp = logdensityof(M, 0.8)\n`, unit],
    ['comparisons', `M = normalize(weighted(x -> 1.0 + 100.0 * ifelse(lt(x, ${hi}), 1.0, 0.0) * ifelse(gt(x, ${lo}), 1.0, 0.0), `
      + `Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.8)\n`, unit],
    ['weighted Uniform', `M = normalize(weighted(${w}, Uniform(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.8)\n`, unit],
    ['truncate', `M = normalize(truncate(weighted(${w}, Lebesgue(reals)), interval(0.0, 1.0)))\nlp = logdensityof(M, 0.8)\n`, unit],
    ['a wider support', `M = normalize(weighted(${w}, Lebesgue(interval(0.0, 2.0))))\nlp = logdensityof(M, 1.8)\n`, wide],
  ];
  for (const [label, src, want] of spellings) {
    test(`a spike on [${lo}, ${hi}], ${label}: the normalizer matches the scipy oracle`, async () => {
      const got = await score(src, 'lp');
      assert.ok(Math.abs(got - want) < 1e-12, `${got} vs ${want}`);
    });
  }
}

test('a weight that is zero off an interval integrates over that interval only', async () => {
  // Z = 2 * (0.7 - 0.2) = 1, so the log-density inside is log 2.
  const got = await score('M = normalize(weighted(x -> ifelse(x in interval(0.2, 0.7), 2.0, 0.0), '
    + 'Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.5)\n', 'lp');
  assert.ok(Math.abs(got - Math.LN2) < 1e-12, `${got}`);
});

test('a weight the adaptive rule cannot resolve refuses instead of scoring', async () => {
  const { ctx } = ctxFor('M = normalize(weighted(x -> 1.0 + sin(1.0e7 * x), Lebesgue(interval(0.0, 1.0))))\n'
    + 'lp = logdensityof(M, 0.5)\n', 1);
  await assert.rejects(ctx.getMeasure('lp'), /adaptive quadrature did not converge/);
});
