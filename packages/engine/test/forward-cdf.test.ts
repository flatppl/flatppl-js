'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { cdf, truncatedQuantile } = require('../forward-cdf.ts');
const { quantile } = require('../inverse-cdf.ts');

// distOp, params, [x...], scipy .cdf reference [F...] — fill from scipy.stats.
const CASES: [string, any, number[], number[]][] = [
  ['Normal',      { mu: 0, sigma: 1 },   [-2, -0.5, 0, 1, 3],   [0.022750, 0.308538, 0.5, 0.841345, 0.998650]],
  ['Exponential', { rate: 2 },           [0.05, 0.35, 1.0],     [0.095163, 0.503415, 0.864665]],
  ['Cauchy',      { location: 0, scale: 5 },  [-5, 0, 5],            [0.25, 0.5, 0.75]],
  ['Beta',        { alpha: 2, beta: 2 }, [0.1, 0.5, 0.9],       [0.028000, 0.5, 0.972000]],
  ['Gamma',       { shape: 2, rate: 1 }, [0.5, 1.678347, 4.0],  [0.090204, 0.5, 0.908422]],
];
for (const [distOp, params, xs, refs] of CASES) {
  test(`forward-cdf ${distOp} matches scipy cdf`, () => {
    for (let i = 0; i < xs.length; i++) {
      const got = cdf(distOp, xs[i], params);
      assert.ok(Math.abs(got - refs[i]) < 1e-5, `${distOp} F(${xs[i]}) = ${got} vs ${refs[i]}`);
    }
  });
}
test('cdf∘quantile round-trips to identity', () => {
  for (const p of [0.05, 0.3, 0.7, 0.95]) {
    const x = quantile('Gamma', p, { shape: 2, rate: 1 });
    assert.ok(Math.abs(cdf('Gamma', x, { shape: 2, rate: 1 }) - p) < 1e-6);
  }
});
test('truncatedQuantile inverts the truncated CDF (HalfCauchy = Cauchy on [0,inf))', () => {
  // truncate Cauchy(0,5) to [0,inf): median of the half is scale·tan(π/4)=5.
  const med = truncatedQuantile('Cauchy', 0.5, { location: 0, scale: 5 }, 0, Infinity);
  assert.ok(Math.abs(med - 5) < 1e-6, `got ${med}`);
  // endpoints: u→0 gives lo, u→1 gives large.
  assert.ok(truncatedQuantile('Cauchy', 1e-9, { location: 0, scale: 5 }, 0, Infinity) >= 0);
});
