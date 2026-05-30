'use strict';

// =====================================================================
// complex-trig.test.ts — complex trig / hyperbolic / inverse trig
// =====================================================================
//
// Pins the closed-form scalar impls in sampler-complex.ts that
// extend complex unary coverage in value-ops elementwise (sin / cos /
// tan / sinh / cosh / tanh / asin / acos / atan / asinh / acosh /
// atanh). Each impl is validated against known closed-form values
// or against the spec's defining identities.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const cx = require('../sampler-complex.ts');

function approx(got: any, wantRe: number, wantIm: number, eps = 1e-10) {
  assert.ok(Math.abs(got.re - wantRe) < eps,
    `re: got ${got.re}, want ${wantRe}`);
  assert.ok(Math.abs(got.im - wantIm) < eps,
    `im: got ${got.im}, want ${wantIm}`);
}

// =====================================================================
// 1. sin / cos / tan — closed-form via sin(re)cosh(im) etc.
// =====================================================================

test('_cSin: sin(i) = i·sinh(1)', () => {
  // sin(0 + i) = sin(0)cosh(1) + i·cos(0)sinh(1) = i·sinh(1) ≈ 1.1752i
  approx(cx._cSin({ re: 0, im: 1 }), 0, Math.sinh(1));
});

test('_cCos: cos(i) = cosh(1)', () => {
  // cos(0 + i) = cos(0)cosh(1) − i·sin(0)sinh(1) = cosh(1)
  approx(cx._cCos({ re: 0, im: 1 }), Math.cosh(1), 0);
});

test('_cTan: tan(i) = i·tanh(1)', () => {
  approx(cx._cTan({ re: 0, im: 1 }), 0, Math.tanh(1));
});

test('_cSin / _cCos: sin² + cos² = 1 (Pythagorean identity)', () => {
  const z = { re: 0.7, im: 0.3 };
  const s = cx._cSin(z), c = cx._cCos(z);
  const s2 = cx._cMul(s, s), c2 = cx._cMul(c, c);
  const sum = cx._cAdd(s2, c2);
  approx(sum, 1, 0, 1e-12);
});

// =====================================================================
// 2. sinh / cosh / tanh — closed-form
// =====================================================================

test('_cSinh: sinh(i·π/2) = i (Euler-style)', () => {
  // sinh(i·θ) = i·sin(θ); sinh(i·π/2) = i·1 = i
  approx(cx._cSinh({ re: 0, im: Math.PI / 2 }), 0, 1, 1e-12);
});

test('_cCosh: cosh(i·π) = -1', () => {
  // cosh(i·θ) = cos(θ); cosh(i·π) = cos(π) = -1
  approx(cx._cCosh({ re: 0, im: Math.PI }), -1, 0, 1e-12);
});

test('_cTanh: tanh(real) = real Math.tanh', () => {
  // For real z, tanh(z) is purely real with value Math.tanh(re).
  approx(cx._cTanh({ re: 1.5, im: 0 }), Math.tanh(1.5), 0, 1e-12);
});

test('_cSinh / _cCosh: cosh² - sinh² = 1', () => {
  const z = { re: 0.6, im: 0.4 };
  const s = cx._cSinh(z), c = cx._cCosh(z);
  const s2 = cx._cMul(s, s), c2 = cx._cMul(c, c);
  const diff = cx._cSub(c2, s2);
  approx(diff, 1, 0, 1e-12);
});

// =====================================================================
// 3. Inverse trig — sin(asin(z)) = z (identity check)
// =====================================================================

test('_cAsin / _cSin: sin(asin(z)) = z', () => {
  const z = { re: 0.3, im: 0.4 };
  const a = cx._cAsin(z);
  const back = cx._cSin(a);
  approx(back, z.re, z.im, 1e-12);
});

test('_cAcos / _cCos: cos(acos(z)) = z', () => {
  const z = { re: 0.3, im: 0.4 };
  const a = cx._cAcos(z);
  const back = cx._cCos(a);
  approx(back, z.re, z.im, 1e-12);
});

test('_cAtan / _cTan: tan(atan(z)) = z', () => {
  const z = { re: 0.7, im: 0.2 };
  const a = cx._cAtan(z);
  const back = cx._cTan(a);
  approx(back, z.re, z.im, 1e-10);
});

// =====================================================================
// 4. Inverse hyperbolic — round-trip identity
// =====================================================================

test('_cAsinh / _cSinh: sinh(asinh(z)) = z', () => {
  const z = { re: 0.5, im: 0.7 };
  const a = cx._cAsinh(z);
  const back = cx._cSinh(a);
  approx(back, z.re, z.im, 1e-12);
});

test('_cAtanh / _cTanh: tanh(atanh(z)) = z', () => {
  const z = { re: 0.3, im: 0.2 };
  const a = cx._cAtanh(z);
  const back = cx._cTanh(a);
  approx(back, z.re, z.im, 1e-12);
});

// =====================================================================
// 5. Real-input → real-output consistency
// =====================================================================

test('all complex unary scalars agree with Math.<fn> on real input', () => {
  const xs = [-1.2, -0.3, 0, 0.5, 1.7];
  for (const x of xs) {
    approx(cx._cSin({ re: x, im: 0 }),   Math.sin(x),   0, 1e-12);
    approx(cx._cCos({ re: x, im: 0 }),   Math.cos(x),   0, 1e-12);
    approx(cx._cSinh({ re: x, im: 0 }),  Math.sinh(x),  0, 1e-12);
    approx(cx._cCosh({ re: x, im: 0 }),  Math.cosh(x),  0, 1e-12);
    approx(cx._cTanh({ re: x, im: 0 }),  Math.tanh(x),  0, 1e-12);
  }
});
