'use strict';
// packages/engine/test/inverse-cdf.test.ts
// Oracle test for the prior inverse-CDF ladder. Every quantile is checked against
// scipy.stats .ppf reference values (computed independently, hardcoded here) at a
// grid including the tails, plus a monotonicity sweep. The numerical-inversion
// last resort is checked against a closed-form quantile on the same distribution.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { quantile, numericalQuantile, probit } = require('../inverse-cdf.ts');

// distOp, params, [p...], scipy ppf reference [x...] — 6-dp from scipy.stats.
const CASES: [string, any, number[], number[]][] = [
  ['Normal',      { mu: 0, sigma: 1 },     [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999], [-3.090232, -1.644854, -0.674490, 0, 0.674490, 1.644854, 3.090232]],
  ['LogNormal',   { mu: 0, sigma: 1 },     [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999], [0.045491, 0.193041, 0.509416, 1, 1.963031, 5.180252, 21.982184]],
  ['Exponential', { rate: 2 },             [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999], [0.000500, 0.025647, 0.143841, 0.346574, 0.693147, 1.497866, 3.453878]],
  ['Beta',        { alpha: 2, beta: 2 },   [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999], [0.018370, 0.135350, 0.326352, 0.5, 0.673648, 0.864650, 0.981630]],
  ['Gamma',       { shape: 2, rate: 1 },   [0.001, 0.05, 0.25, 0.5, 0.75, 0.95, 0.999], [0.045402, 0.355362, 0.961279, 1.678347, 2.692635, 4.743865, 9.233413]],
  ['Weibull',     { shape: 1.5, scale: 2 },[0.05, 0.5, 0.95],                            [0.276103, 1.566440, 4.156221]],
  ['Pareto',      { shape: 2.5, scale: 1 },[0.05, 0.5, 0.95],                            [1.020729, 1.319508, 3.314454]],
  ['Logistic',    { mu: 0, s: 1 },         [0.05, 0.5, 0.95],                            [-2.944439, 0, 2.944439]],
  ['Laplace',     { location: 0, scale: 1 },[0.05, 0.5, 0.95],                           [-2.302585, 0, 2.302585]],
  ['HalfNormal',  { sigma: 2 },            [0.05, 0.5, 0.95],                            [0.125414, 1.348980, 3.919928]],
  ['HalfCauchy',  { scale: 1.5 },          [0.05, 0.5, 0.95],                            [0.118053, 1.5, 19.059307]],
  // scipy.stats.invgamma(a=5, scale=5).ppf([0.01,0.1,0.5,0.9,0.99]) — full precision.
  ['InverseGamma', { shape: 5, scale: 5 }, [0.01, 0.1, 0.5, 0.9, 0.99],
    [0.4308626733155887, 0.6255012152142634, 1.070455477822771, 2.055421542970347, 3.908979933575256]],
];

for (const [distOp, params, ps, refs] of CASES) {
  test(`inverse-cdf ${distOp} matches scipy ppf across the grid + tails`, () => {
    for (let i = 0; i < ps.length; i++) {
      const got = quantile(distOp, ps[i], params);
      const ref = refs[i];
      const tol = 1e-4 * (1 + Math.abs(ref));
      assert.ok(Math.abs(got - ref) < tol, `${distOp} q(${ps[i]}) = ${got} vs scipy ${ref}`);
    }
  });
}

test('inverse-cdf Uniform + Dirac', () => {
  assert.equal(quantile('Uniform', 0.25, { lo: -3, hi: 5 }), -1);
  assert.equal(quantile('Uniform', 0.0, { lo: -3, hi: 5 }) < -2.999, true);
  assert.equal(quantile('Dirac', 0.42, { value: 7 }), 7);
});

test('every quantile is monotone non-decreasing in p', () => {
  for (const [distOp, params] of CASES) {
    let prev = -Infinity;
    for (let k = 1; k < 200; k++) {
      const p = k / 200;
      const x = quantile(distOp, p, params);
      assert.ok(x >= prev - 1e-9, `${distOp} not monotone at p=${p}: ${x} < ${prev}`);
      prev = x;
    }
  }
});

test('numericalQuantile (rung 3) matches a closed-form on the same cdf', () => {
  // Standard-normal cdf Φ(x) = ½ erfc(−x/√2); invert numerically, compare to probit.
  const erfc = require('@stdlib/math-base-special-erfc');
  const cdf = (x: number) => 0.5 * erfc(-x / Math.SQRT2);
  for (const p of [0.01, 0.2, 0.5, 0.8, 0.99]) {
    const num = numericalQuantile(cdf, p, -Infinity, Infinity);
    const closed = probit(p);
    assert.ok(Math.abs(num - closed) < 1e-8, `numerical Φ⁻¹(${p}) = ${num} vs probit ${closed}`);
  }
});
