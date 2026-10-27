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

// A feature narrower than the cubature's seed grid, or a jump the breakpoint
// walk could not see, scored as absent (log-density 0). Each row's oracle is
// scipy.integrate.quad split at the jumps (epsrel 1e-13), or for the Gaussian
// bumps the closed form -log(1 + 100·sd·√(2π)·(Φ((1-0.3)/sd) - Φ(-0.3/sd))),
// which scipy misses for sd = 1e-4 without a split at the peak.
const IND = (c: string) => `ifelse(${c}, 1.0, 0.0)`;
const ROWS: Array<[string, string, number, number]> = [
  ['a smooth bump of sd 1e-3', '1.0 + 100.0 * exp(-0.5 * ((x - 0.3) / 0.001)^2)', 0.8, -0.22367367274549067],
  ['a smooth bump of sd 1e-4', '1.0 + 100.0 * exp(-0.5 * ((x - 0.3) / 0.0001)^2)', 0.8, -0.02475727659357076],
  ['abs(x - 0.5) < 0.005', '1.0 + 100.0 * ' + IND('abs(x - 0.5) < 0.005'), 0.8, -0.6931471805599457],
  ['abs(x - 0.5) < 5e-5', '1.0 + 100.0 * ' + IND('abs(x - 0.5) < 5.0e-5'), 0.8, -0.00995033085316697],
  ['abs(x - 0.4) < 5e-7', '1.0 + 100.0 * ' + IND('abs(x - 0.4) < 5.0e-7'), 0.8, -9.999500033621182e-05],
  ['(2x) in interval(0.5, 0.5002)', '1.0 + 100.0 * ' + IND('(2.0 * x) in interval(0.5, 0.5002)'), 0.8, -0.00995033085316697],
  ['1 - x > 0.9999', '1.0 + 100.0 * ' + IND('1.0 - x > 0.9999'), 0.8, -0.009950330853168092],
  ['1 + floor(10x)', '1.0 + floor(10.0 * x)', 0.05, -1.7047480922384253],
  ['floor(1000x) == 500', '1.0 + 100.0 * ' + IND('floor(1000.0 * x) == 500.0'), 0.8, -0.0953101798043249],
  ['round(x / 0.001) == 250', '1.0 + 100.0 * ' + IND('round(x / 0.001) == 250.0'), 0.8, -0.0953101798043249],
  ['1 + ceil(x / 0.25)', '1.0 + ceil(x / 0.25)', 0.1, -0.5596157879354228],
  // A window of width 1e-4, Z = 1.01, spelled through each affine form.
  ['-x > -1e-4', '1.0 + 100.0 * ' + IND('-x > -1.0e-4'), 0.8, -Math.log1p(0.01)],
  ['x * 2 < 2e-4', '1.0 + 100.0 * ' + IND('x * 2.0 < 2.0e-4'), 0.8, -Math.log1p(0.01)],
  ['x + x < 2e-4', '1.0 + 100.0 * ' + IND('x + x < 2.0e-4'), 0.8, -Math.log1p(0.01)],
  ['1e-4 > x', '1.0 + 100.0 * ' + IND('1.0e-4 > x'), 0.8, -Math.log1p(0.01)],
  ['x / 2 in interval(0.25, 0.25005)', '1.0 + 100.0 * ' + IND('x / 2.0 in interval(0.25, 0.25005)'), 0.8, -Math.log1p(0.01)],
  ['5e-5 > abs(x - 0.5)', '1.0 + 100.0 * ' + IND('5.0e-5 > abs(x - 0.5)'), 0.8, -Math.log1p(0.01)],
  // A broadcast of a continuous function: w = 2 + x and w = 1 + 2x.
  ['a broadcast operator', '1.0 + sum([x] .+ 1.0)', 0.05, Math.log(2.05 / 2.5)],
  ['a broadcast fn', '1.0 + sum(broadcast(fn(_ * 2.0), [x]))', 0.05, Math.log(1.1 / 2)],
];
for (const [label, w, pt, want] of ROWS) {
  test(`${label}: the normalizer matches the oracle`, async () => {
    const got = await score(`M = normalize(weighted(x -> ${w}, Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, ${pt})\n`, 'lp');
    assert.ok(Math.abs(got - want) < 1e-11, `${got} vs ${want}`);
  });
}

for (const [label, w, msg] of [
  ['x * x < 1e-8', '1.0 + 100.0 * ' + IND('x * x < 1.0e-8'), /`lt` test on an expression that is not affine in the variate at line 1, column 50/],
  ['mod(x, 0.1)', '1.0 + mod(x, 0.1)', /`mod` of the variate, which may jump/],
  ['x in a set that is not an interval', '1.0 + ' + IND('x in integers'), /`in` test/],
  ['abs of a non-affine expression', '1.0 + 100.0 * ' + IND('abs(x * x - 0.25) < 1.0e-4'), /`lt` test on an expression that is not affine/],
  ['floor of a non-affine expression', '1.0 + floor(x * x)', /`floor` of an expression that is not affine/],
  ['a broadcast comparison', '1.0 + 100.0 * sum(ifelse.([x] .< 1.0e-4, 1.0, 0.0))', /`broadcast` of the variate through a function that may jump/],
  ['a broadcast floor', '1.0 + sum(floor.([10.0 * x]))', /`broadcast` of the variate through a function that may jump/],
  ['floor with more steps than it splits at', '1.0 + floor(1.0e5 * x)', /`floor` of the variate with 100001 steps over the support \(more than 4096\)/],
] as Array<[string, string, RegExp]>) {
  test(`a jump the normalizer cannot split at refuses: ${label}`, async () => {
    const { ctx } = ctxFor(`M = normalize(weighted(x -> ${w}, Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.8)\n`, 1);
    await assert.rejects(ctx.getMeasure('lp'), msg);
  });
}

test('a window scaled by a named constant: the normalizer matches the oracle', async () => {
  const got = await score('k = 2.0\nM = normalize(weighted(x -> 1.0 + 100.0 * ifelse(x * k < 2.0e-4, 1.0, 0.0), '
    + 'Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.8)\n', 'lp');
  assert.ok(Math.abs(got - -Math.log1p(0.01)) < 1e-11, `${got}`);
});

test('a broadcast of a named function: the normalizer matches the closed form', async () => {
  // w = 1 + 2x, Z = 2.
  const got = await score('g(t) = 2.0 * t\nM = normalize(weighted(x -> 1.0 + sum(broadcast(g, [x])), '
    + 'Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.8)\n', 'lp');
  assert.ok(Math.abs(got - Math.log(2.6 / 2)) < 1e-11, `${got}`);
});

// §03 named intervals supply finite cuts just like interval(...). Direct
// integration: Z = 3 + length(S ∩ [-1,2]); endpoint inclusion affects density
// at zero, but changes no Lebesgue mass.
for (const [set, mass, atZero] of [
  ['posreals', 5, 1], ['nonnegreals', 5, 2], ['unitinterval', 4, 2],
] as Array<[string, number, number]>) {
  test(`membership in ${set} splits the integral and retains its endpoint`, async () => {
    const source = `M = normalize(weighted(x -> 1.0 + ifelse(x in ${set}, 1.0, 0.0), Lebesgue(interval(-1.0, 2.0))))\n`;
    const outside = await score(source + 'lp = logdensityof(M, -0.5)\n', 'lp');
    const zero = await score(source + 'lp = logdensityof(M, 0.0)\n', 'lp');
    assert.ok(Math.abs(outside + Math.log(mass)) < 1e-12);
    assert.ok(Math.abs(zero - Math.log(atZero / mass)) < 1e-12);
  });
}

test('an affine narrow window through a named set splits at both endpoints', async () => {
  // (x-0.3)/1e-4 ∈ [0,1] has width 1e-4; Z = 1 + 100*1e-4.
  const got = await score('M = normalize(weighted(x -> 1.0 + 100.0 * ifelse((x - 0.3) / 1.0e-4 in unitinterval, 1.0, 0.0), '
    + 'Lebesgue(interval(0.0, 1.0))))\nlp = logdensityof(M, 0.8)\n', 'lp');
  assert.ok(Math.abs(got + Math.log1p(0.01)) < 1e-11);
});
