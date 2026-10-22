'use strict';

// Spec §09 `blatt_weisskopf` defines the barrier factor for 0 <= ℓ <= 7:
//   F_ℓ = √(z^ℓ / χ_ℓ(z)), z = (d p)², χ_ℓ(z) = z^{ℓ+1}(j_ℓ(√z)² + y_ℓ(√z)²).
// The engine's table stopped at ℓ = 4 and threw for ℓ = 5..7, both from
// `blatt_weisskopf` and from `resonance_breitwigner`.
//
// ORACLE: a Python script that builds χ_ℓ EXACTLY, in rational arithmetic, from
// the closed form x²(j_ℓ² + y_ℓ²) = Σ_k (2ℓ−k)!(2ℓ−2k)! / (k!((ℓ−k)!)²) (2x)^{2k−2ℓ},
// independent of the polynomial table the engine holds. For ℓ = 5, 6, 7 the
// exact polynomial equals the spec's table, which the first test also checks
// at a rational point.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { _internal } = require('../standard-modules.ts');

// F_ℓ(p; d) from the exact χ_ℓ.
const F_CASES: Array<[number, number, number, number]> = [
  [5, 1.3, 1.5, 0.023978050011807868],
  [6, 0.8, 2.0, 0.0014344125171668822],
  [7, 2.0, 1.1, 0.0015271897388544273],
  [7, 0.05, 3.0, 1.2632668683858745e-11],
];

for (const [l, p, d, want] of F_CASES) {
  test(`blatt_weisskopf(${l}, ${p}, ${d}) matches the exact barrier factor`, () => {
    const got = _internal._blattWeisskopf(l, p, d);
    assert.ok(Math.abs(got - want) <= 1e-13 * Math.abs(want), `got ${got}, want ${want}`);
  });
}

// (σ, m, Γ, ma, mb, ℓ, d) -> BW and Γ(σ), with the barrier ratio from the exact χ_ℓ.
const BW_CASES: Array<{ args: number[]; re: number; im: number; width: number }> = [
  { args: [1.44, 1.0, 0.1, 0.1, 0.1, 5, 1.5], re: -0.7208515452851786, im: 1.0576729249712287, width: 0.6455921334027681 },
  { args: [1.44, 1.0, 0.1, 0.1, 0.1, 6, 1.5], re: -0.40397751767256174, im: 0.8688687399131183, width: 0.9463453505143861 },
  { args: [2.25, 1.2, 0.15, 0.14, 0.14, 7, 3.0], re: -0.04569264496164381, im: 0.23307263886728966, width: 3.4430931141649705 },
  { args: [0.64, 1.0, 0.1, 0.1, 0.1, 7, 1.5], re: 2.7774781116153173, im: 0.028849890936123043, width: 0.003739349265641579 },
];

for (const { args, re, im, width } of BW_CASES) {
  test(`resonance_breitwigner(${args.join(', ')}) matches the exact lineshape`, () => {
    const v = _internal._resonanceBreitwigner(...args);
    assert.ok(Math.abs(v.re - re) <= 1e-12 * Math.max(1, Math.abs(re)), `re ${v.re} vs ${re}`);
    assert.ok(Math.abs(v.im - im) <= 1e-12 * Math.max(1, Math.abs(im)), `im ${v.im} vs ${im}`);
    const n = v.re * v.re + v.im * v.im;
    assert.ok(Math.abs((v.im / n) / args[1] - width) <= 1e-12 * Math.max(1, width), `Γ(σ) ${(v.im / n) / args[1]} vs ${width}`);
  });
}

test('the ℓ = 0..7 barrier polynomials are the exact closed form at a rational point', () => {
  // The spec's tabulated χ_ℓ for ℓ = 5..7, written out here from §09. A Python
  // script checked each row equal to the exact Σ_k closed form in rational
  // arithmetic (at z = 7/5), so the table, the spec and the closed form agree.
  const chiAt: Record<number, number> = {
    5: 893025 + 99225 * 1.4 + 6300 * 1.4 ** 2 + 315 * 1.4 ** 3 + 15 * 1.4 ** 4 + 1.4 ** 5,
    6: 108056025 + 9823275 * 1.4 + 496125 * 1.4 ** 2 + 18900 * 1.4 ** 3 + 630 * 1.4 ** 4 + 21 * 1.4 ** 5 + 1.4 ** 6,
    7: 18261468225 + 1404728325 * 1.4 + 58939650 * 1.4 ** 2 + 1819125 * 1.4 ** 3 + 47250 * 1.4 ** 4 + 1134 * 1.4 ** 5 + 28 * 1.4 ** 6 + 1.4 ** 7,
  };
  for (const l of [5, 6, 7]) {
    const got = 1 / _internal._blattWeisskopfSquared(l, 1.4);
    assert.ok(Math.abs(got - chiAt[l]) <= 1e-13 * chiAt[l], `χ_${l}(1.4) ${got} vs ${chiAt[l]}`);
  }
});

test('ℓ above 7 is outside the spec and refuses', () => {
  assert.throws(() => _internal._resonanceBreitwigner(1.44, 1, 0.1, 0.1, 0.1, 8, 1.5),
    /Blatt-Weisskopf factor for l=8 is not defined \(spec §09 defines 0\.\.7\)/);
});
