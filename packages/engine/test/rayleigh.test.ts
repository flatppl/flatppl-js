'use strict';

// Rayleigh(scale = σ), spec §08: density x/σ² · exp(−x²/(2σ²)) for x ≥ 0.
// Every pinned log-density is Distributions.jl 0.25.134 `logpdf(Rayleigh(σ), x)`,
// cross-checked against `logpdf(Weibull(2, σ√2), x)` (agreement within 4.4e-16).
// Moments: mean σ√(π/2), variance (4 − π)σ²/2.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const sampler = require('../sampler.ts');
const rng = require('../rng.ts');

async function score(source: string): Promise<number> {
  const { ctx } = makeMatCtx(source, { sampleCount: 1 });
  const m = await ctx.getMeasure('score');
  return (m.value ? Array.from(m.value.data) : m.samples)[0] as number;
}

// [x, σ, Distributions.jl logpdf]
const ORACLE: [number, number, number][] = [
  [1.5, 2.0, -1.2620792530117262],
  [0.7, 0.5, 0.049619417181158254],
  [3.0, 1.3, -2.0888381337579967],
  [2.5, 1.0, -2.208709268125845],
];

test('Rayleigh log-density matches the Distributions.jl oracle', async () => {
  for (const [x, sigma, want] of ORACLE) {
    const got = await score(`score = logdensityof(Rayleigh(scale = ${sigma}), ${x})\n`);
    assert.ok(Math.abs(got - want) < 1e-13, `Rayleigh(${sigma}) at ${x}: got ${got}, want ${want}`);
    const positional = await score(`score = logdensityof(Rayleigh(${sigma}), ${x})\n`);
    assert.equal(positional, got, `positional Rayleigh(${sigma}) at ${x}`);
  }
});

test('Rayleigh(σ) scores like Weibull(2, σ√2)', async () => {
  for (const [x, sigma, want] of ORACLE) {
    const got = await score(`score = logdensityof(Weibull(shape = 2.0, scale = ${sigma * Math.SQRT2}), ${x})\n`);
    assert.ok(Math.abs(got - want) < 1e-13, `Weibull(2, ${sigma}√2) at ${x}: got ${got}, want ${want}`);
  }
});

test('Rayleigh density is 0 at x = 0 and off the support', async () => {
  assert.equal(await score('score = logdensityof(Rayleigh(scale = 2.0), 0.0)\n'), -Infinity);
  assert.equal(await score('score = logdensityof(Rayleigh(scale = 2.0), -1.0)\n'), -Infinity);
});

test('Rayleigh density integrates to 1 and totalmass is 1', async () => {
  // Composite Simpson over [0, 20σ]; the tail beyond carries exp(−200) mass.
  const logpdf = sampler._internal.REGISTRY.Rayleigh.logpdfFn;
  for (const sigma of [0.5, 2.0]) {
    const n = 20000, hi = 20 * sigma, h = hi / n;
    let s = 0;
    for (let i = 0; i <= n; i++) {
      const w = i === 0 || i === n ? 1 : (i % 2 ? 4 : 2);
      s += w * Math.exp(logpdf(i * h, sigma));
    }
    const mass = s * h / 3;
    assert.ok(Math.abs(mass - 1) < 1e-10, `∫ Rayleigh(${sigma}) density = ${mass}`);
  }
  assert.equal(await score('score = totalmass(Rayleigh(scale = 2.0))\n'), 1);
});

test('Rayleigh interval masses and truncated quantiles match the oracle', () => {
  // Distributions.jl: cdf/ccdf differences and quantile(truncated(Rayleigh(2), lo, hi), u).
  const { intervalProbability, truncatedQuantile } = require('../forward-cdf.ts');
  const q = { scale: 2 };
  const near = (got: number, want: number, what: string) =>
    assert.ok(Math.abs(got - want) < 1e-12, `${what}: got ${got}, want ${want}`);
  near(intervalProbability('Rayleigh', q, 1.5, 4.0), 0.6195043187523946, 'mass [1.5, 4]');
  near(intervalProbability('Rayleigh', q, 4.0, Infinity), 0.1353352832366127, 'mass [4, inf)');
  near(truncatedQuantile('Rayleigh', 0.5, q, 4.0, Infinity), 4.641678300408114, 'upper-tail median');
  near(truncatedQuantile('Rayleigh', 0.3, q, 0.5, 1.5), 0.8940429627588782, 'lower quantile');
});

const N = 200_000;
const TOL_SIGMA = 5;

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}

function draws(sigma: number, seed: number) {
  const ir = {
    kind: 'call',
    op: 'Rayleigh',
    kwargs: { scale: { kind: 'lit', value: sigma, loc: synthLoc() } },
    loc: synthLoc(),
  };
  const s = sampler.makeSampler(rng.seedFromBytes([seed, 23, 0]), ir, {});
  const xs = new Float64Array(N);
  for (let i = 0; i < N; i++) xs[i] = s.draw();
  return xs;
}

for (const sigma of [0.5, 2.0]) {
  test(`Rayleigh(${sigma}) draws match the closed-form mean and variance within ${TOL_SIGMA}σ`, () => {
    const mean = sigma * Math.sqrt(Math.PI / 2);
    const variance = (4 - Math.PI) / 2 * sigma * sigma;
    // Excess kurtosis −(6π² − 24π + 16)/(4 − π)² gives the sample variance's SE.
    const kurt = 3 - (6 * Math.PI ** 2 - 24 * Math.PI + 16) / (4 - Math.PI) ** 2;
    const seMean = Math.sqrt(variance / N);
    const seVar = Math.sqrt((kurt - 1) * variance * variance / N);
    for (const seed of [1, 2]) {
      const xs = draws(sigma, seed);
      let sum = 0;
      for (let i = 0; i < N; i++) {
        assert.ok(xs[i] >= 0, `draw ${xs[i]} is negative`);
        sum += xs[i];
      }
      const m = sum / N;
      let c2 = 0;
      for (let i = 0; i < N; i++) c2 += (xs[i] - m) ** 2;
      const v = c2 / N;
      assert.ok(Math.abs(m - mean) / seMean < TOL_SIGMA, `seed ${seed}: mean ${m} vs ${mean}`);
      assert.ok(Math.abs(v - variance) / seVar < TOL_SIGMA, `seed ${seed}: var ${v} vs ${variance}`);
    }
  });
}
