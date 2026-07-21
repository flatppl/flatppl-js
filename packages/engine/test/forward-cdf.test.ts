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
  // scipy.stats.invgamma(a=5, scale=5).cdf(x) at x = that dist's own ppf([0.01,0.1,0.5,0.9,0.99])
  // (independently computed via scipy — see inverse-cdf.test.ts for the matching ppf values).
  ['InverseGamma', { shape: 5, scale: 5 },
    [0.4308626733155887, 0.6255012152142634, 1.070455477822771, 2.055421542970347, 3.908979933575256],
    [0.01, 0.1, 0.5, 0.9, 0.99]],
  // scipy.stats.t(df=3).cdf(x) — full precision.
  ['StudentT', { nu: 3 }, [-5, -2, -0.5, 0, 1, 3, 5],
    [0.007696219036651147, 0.06966298427942152, 0.3257239824240755, 0.5, 0.8044988905221148, 0.9711655571887814, 0.9923037809633488]],
  // scipy.stats.t(df=10).cdf(x) — full precision.
  ['StudentT', { nu: 10 }, [-5, -2, -0.5, 0, 1, 3, 5],
    [0.0002686668013782264, 0.03669401738537018, 0.3139468028714865, 0.5, 0.8295534338489701, 0.9933281724887152, 0.9997313331986217]],
  // scipy.stats.chi2(df=2).cdf(x) — full precision.
  ['ChiSquared', { k: 2 }, [0.5, 1.386294361119891, 3, 5, 10],
    [0.22119921692859512, 0.5, 0.7768698398515702, 0.9179150013761012, 0.9932620530009145]],
  // scipy.stats.chi2(df=5).cdf(x) — full precision.
  ['ChiSquared', { k: 5 }, [0.5, 1.386294361119891, 3, 5, 10],
    [0.007876706767370404, 0.0741933688528055, 0.3000141641213724, 0.5841198130044919, 0.9247647538534879]],
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
test('InverseGamma cdf∘quantile round-trips to identity', () => {
  for (const p of [0.01, 0.1, 0.5, 0.9, 0.99]) {
    const x = quantile('InverseGamma', p, { shape: 5, scale: 5 });
    assert.ok(Math.abs(cdf('InverseGamma', x, { shape: 5, scale: 5 }) - p) < 1e-9, `p=${p} x=${x}`);
  }
});
test('StudentT cdf∘quantile round-trips to identity', () => {
  for (const nu of [3, 10]) {
    for (const p of [0.01, 0.1, 0.5, 0.9, 0.99]) {
      const x = quantile('StudentT', p, { nu });
      assert.ok(Math.abs(cdf('StudentT', x, { nu }) - p) < 1e-9, `nu=${nu} p=${p} x=${x}`);
    }
  }
});
test('ChiSquared cdf∘quantile round-trips to identity', () => {
  for (const k of [2, 5]) {
    for (const p of [0.01, 0.1, 0.5, 0.9, 0.99]) {
      const x = quantile('ChiSquared', p, { k });
      assert.ok(Math.abs(cdf('ChiSquared', x, { k }) - p) < 1e-9, `k=${k} p=${p} x=${x}`);
    }
  }
});
test('truncatedQuantile inverts the truncated CDF (HalfCauchy = Cauchy on [0,inf))', () => {
  // truncate Cauchy(0,5) to [0,inf): median of the half is scale·tan(π/4)=5.
  const med = truncatedQuantile('Cauchy', 0.5, { location: 0, scale: 5 }, 0, Infinity);
  assert.ok(Math.abs(med - 5) < 1e-6, `got ${med}`);
  // endpoints: u→0 gives lo, u→1 gives large.
  assert.ok(truncatedQuantile('Cauchy', 1e-9, { location: 0, scale: 5 }, 0, Infinity) >= 0);
});
