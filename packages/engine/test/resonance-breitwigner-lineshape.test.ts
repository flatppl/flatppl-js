'use strict';

// Spec §09 `resonance_breitwigner`:
//   BW(σ) = 1 / (m² − σ − i m Γ(σ)),
//   Γ(σ) = Γ (m/√σ) (p(σ)/p₀) (F_ℓ(p(σ)) / F_ℓ(p₀))²,
// with F_ℓ = √(z^ℓ / χ_ℓ(z)), z = (d p)². The squared ratio therefore carries the
// numerator (p/p₀)^{2ℓ}, so Γ(σ) ∝ (p/p₀)^{2ℓ+1}, as in the PDG review "Resonances"
// (Navas et al. 2024, §50). The engine used the 1/χ_ℓ part alone, so its width was
// wrong for every ℓ ≥ 1; every existing pin was at ℓ = 0 with massless daughters,
// where the ratio is 1.
//
// ORACLE: a hand-written Python script, independent of the engine's polynomial
// table. χ_ℓ(z) = z^{ℓ+1}(j_ℓ(√z)² + y_ℓ(√z)²) from the spherical Bessel functions
// in closed form, p = √λ(σ, ma², mb²)/(2√σ), then the formula above. At ℓ = 1 and
// ℓ = 2 it agrees with the polynomial table to double precision, which also
// cross-checks the table. Parameters are (σ, m, Γ, ma, mb, ℓ, d).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { _internal } = require('../standard-modules.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const CASES: Array<{ args: number[]; re: number; im: number; width: number }> = [
  // ℓ = 1, above the pole and below it.
  { args: [1.44, 1.0, 0.1, 0.1, 0.1, 1, 1.5], re: -2.099370433464692, im: 0.6032745832427832, width: 0.1264382942598438 },
  { args: [0.64, 1.0, 0.1, 0.1, 0.1, 1, 1.5], re: 2.6734543820743015, im: 0.5281134720837285, width: 0.0711143048577585 },
  // ℓ = 2, two radii and daughter masses.
  { args: [1.44, 1.0, 0.1, 0.1, 0.1, 2, 1.5], re: -1.8999646479521162, im: 0.8415674715378668, width: 0.19489293544265648 },
  { args: [2.25, 1.2, 0.15, 0.14, 0.14, 2, 3.0], re: -1.112448238324963, im: 0.3685807969341706, width: 0.22364369807010226 },
  // ℓ = 0 with massless daughters: Γ(σ) = Γ.
  { args: [1.44, 1.0, 0.1, 0.0, 0.0, 0, 1.5], re: -2.1611001964636545, im: 0.49115913555992163, width: 0.1 },
];

for (const { args, re, im, width } of CASES) {
  test(`resonance_breitwigner(${args.join(', ')}) matches the spec lineshape`, () => {
    const v = _internal._resonanceBreitwigner(...args);
    assert.ok(Math.abs(v.re - re) < 1e-12 * Math.max(1, Math.abs(re)), `re ${v.re} vs ${re}`);
    assert.ok(Math.abs(v.im - im) < 1e-12 * Math.max(1, Math.abs(im)), `im ${v.im} vs ${im}`);
    // Γ(σ) is recoverable from the amplitude: BW⁻¹ = (m² − σ) − i m Γ(σ).
    const [sigma, m] = args;
    const inv = { re: v.re / (v.re * v.re + v.im * v.im), im: -v.im / (v.re * v.re + v.im * v.im) };
    assert.ok(Math.abs(-inv.im / m - width) < 1e-12, `Γ(σ) ${-inv.im / m} vs ${width}`);
    assert.ok(Math.abs(inv.re - (m * m - sigma)) < 1e-12);
  });
}

test('the width grows as (p/p0)^(2l+1): the l = 1 and l = 2 widths differ by (p/p0)^2 times the barrier ratio', () => {
  // Γ(σ)(ℓ=2) / Γ(σ)(ℓ=1) = (p/p₀)² · [χ₂-ratio / χ₁-ratio], using the table values.
  const [sigma, m, G, ma, mb, , d] = [1.44, 1.0, 0.1, 0.1, 0.1, 0, 1.5];
  const p = _internal._breakupMomentum(sigma, ma, mb);
  const p0 = _internal._breakupMomentum(m * m, ma, mb);
  const chi1 = (z: number) => 1 + z;
  const chi2 = (z: number) => 9 + 3 * z + z * z;
  const z = (d * p) ** 2, z0 = (d * p0) ** 2;
  const want = (p / p0) ** 2 * (chi2(z0) / chi2(z)) / (chi1(z0) / chi1(z));
  const gamma = (l: number) => {
    const v = _internal._resonanceBreitwigner(sigma, m, G, ma, mb, l, d);
    const n = v.re * v.re + v.im * v.im;
    return (v.im / n) / m;
  };
  assert.ok(Math.abs(gamma(2) / gamma(1) - want) < 1e-12);
});

test('the module function is the same lineshape when called from a model', async () => {
  const { ctx } = makeMatCtx(`
hep = standard_module("particle-physics", "0.1")
bw = hep.resonance_breitwigner(1.44, 1.0, 0.1, 0.1, 0.1, 1, 1.5)
re = real(bw)
im = imag(bw)
`, { sampleCount: 2, rootSeed: 1 });
  const re = (await ctx.getMeasure('re')).value?.data?.[0] ?? (await ctx.getMeasure('re')).samples?.[0];
  assert.ok(Math.abs(re - -2.099370433464692) < 1e-12, `re ${re}`);
});
