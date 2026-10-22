'use strict';

// A truncate of a scalar law samples by inverting the law's CDF, and its mass
// is a CDF difference. These laws had neither method on their constructor, so
// truncating one crashed with "dist.cdf is not a function". Oracles: scipy.stats
// (vonmises at kappa = 1, gennorm, geom with loc = -1, nbinom) and, for
// VonMises at large kappa where scipy switches to a normal approximation,
// mpmath quadrature of the density at 40 digits.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { REGISTRY } = require('../sampler-registry.ts');

async function value(src: string, sampleCount = 1): Promise<number[]> {
  const { ctx } = makeMatCtx(src, { sampleCount });
  const m = await ctx.getMeasure('v');
  return Array.from(m.samples ?? m.value.data);
}

const close = (actual: number, expected: number, label: string, tol = 1e-12) =>
  assert.ok(Math.abs(actual - expected) < tol, `${label}: ${actual} vs ${expected}`);

const CASES = [
  // [base, set, scored point, mass over the set, log density of the base at the point, lo, hi, discrete]
  ['VonMises(0.0, 1.0)', 'interval(-1.0, 2.0)', 0.5, 0.7285962633246666, -1.1962088630261514, -1, 2, false],
  ['GeneralizedNormal(0.0, 1.0, 1.5)', 'interval(-1.0, 2.0)', 0.5, 0.8734716238126818, -0.9443857381925784, -1, 2, false],
  ['Geometric(0.3)', 'interval(1, 4)', 1, 0.53193, -1.5606477482646686, 1, 4, true],
  ['NegativeBinomial(2.0, 3.0)', 'interval(1, 4)', 1, 0.432861328125, -1.268511325463507, 1, 4, true],
  ['NegativeBinomial2(2.0, 3.0)', 'interval(1, 4)', 1, 0.687744, -1.3501553145040175, 1, 4, true],
  ['Categorical([0.2, 0.3, 0.5])', 'interval(1, 2)', 1, 0.5, Math.log(0.2), 1, 2, true],
  ['Categorical0([0.2, 0.3, 0.5])', 'interval(1, 2)', 1, 0.8, Math.log(0.3), 1, 2, true],
] as const;

for (const [base, set, x, mass, logDensity, lo, hi, discrete] of CASES) {
  test(`truncate(${base}, ${set}) has the oracle mass, density and support`, async () => {
    const T = `truncate(${base}, ${set})`;
    close((await value(`v = totalmass(${T})`))[0], mass, 'totalmass');
    close((await value(`v = logdensityof(normalize(${T}), ${x})`))[0], logDensity - Math.log(mass), 'logdensityof');
    const xs = await value(`v ~ normalize(${T})`, 2000);
    assert.ok(xs.every((s) => s >= lo && s <= hi), `samples leave [${lo}, ${hi}]`);
    if (discrete) assert.ok(xs.includes(lo) && xs.includes(hi), 'both closed ends are drawn');
  });
}

test('the VonMises CDF matches quadrature where scipy approximates', () => {
  const xs = [-3.0, -1.0, -0.3, -0.01, 0.5, 2.5];
  const oracle: Record<number, number[]> = {
    100: [1.142599799397869e-87, 5.10167823261133e-22, 0.0014180426759421573, 0.4602221673293832, 0.9999996127136057, 1],
    1000: [0, 3.3833693475802726e-202, 1.692503014559374e-21, 0.3759303243203694, 1, 1],
    100000: [0, 0, 0, 0.000782747173991507, 1, 1],
  };
  for (const [kappa, expected] of Object.entries(oracle)) {
    const d = new REGISTRY.VonMises.Ctor(0, +kappa);
    xs.forEach((x, i) => close(d.cdf(x), expected[i], `kappa ${kappa} at ${x}`, 1e-14));
  }
});

test('each quantile inverts its CDF', () => {
  const vm = new REGISTRY.VonMises.Ctor(0.3, 2.0);
  const gn = new REGISTRY.GeneralizedNormal.Ctor(0.5, 2.0, 1.5);
  for (const x of [-2.5, 0.1, 1.7]) {
    close(vm.quantile(vm.cdf(x)), x, `VonMises at ${x}`, 1e-9);
    close(gn.quantile(gn.cdf(x)), x, `GeneralizedNormal at ${x}`, 1e-9);
  }
  // A discrete quantile is the smallest atom whose CDF reaches p.
  for (const d of [new REGISTRY.Geometric.Ctor(0.3), new REGISTRY.NegativeBinomial.Ctor(2, 3),
    new REGISTRY.NegativeBinomial2.Ctor(2, 3), new REGISTRY.Categorical0.Ctor([0.2, 0.3, 0.5])]) {
    for (const k of [0, 1]) {
      assert.equal(d.quantile(d.cdf(k)), k);
      assert.equal(d.quantile(d.cdf(k) + 1e-9), k + 1);
    }
  }
});
