'use strict';

// =====================================================================
// DoubleSidedCrystalBall REGISTRY param order — regression (spec §09).
// =====================================================================
//
// The REGISTRY entry's `params` list feeds the record/kwarg dispatch
// path (`builtin_logdensityof('DoubleSidedCrystalBall', {…}, x)`): it
// maps kwarg names onto the positional slots `_doubleSidedCrystalBallLogpdf`
// expects. Spec §09 groups the tail params `(alphaL, alphaR, nL, nR)`,
// matching the logpdf signature — `params` must use the same order, or
// the record path silently swaps `alphaR` and `nL` into each other's
// slots while the positional `.()` path (which never consults `params`
// names) stays correct. Pin record == positional at asymmetric,
// tail-sensitive observation points so a reintroduced interleave fails
// loudly here instead of only on the record path in the field.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const densityPrims = require('../density-prims.ts');
const sampler = require('../sampler.ts');

const REGISTRY = sampler._internal.REGISTRY;

// Asymmetric so alphaL≠alphaR≠nL≠nR — a swap of alphaR/nL changes the number.
const M0 = 0, SIGMA_L = 1.0, SIGMA_R = 1.5;
const ALPHA_L = 1.2, ALPHA_R = 2.0, N_L = 3.0, N_R = 4.0;

function referenceLogpdf(x: number): number {
  // The logpdf itself is grouped-correct; call it positionally with the
  // spec §09 order to get the ground truth for this x.
  return REGISTRY.DoubleSidedCrystalBall.logpdfFn(
    x, M0, SIGMA_L, SIGMA_R, ALPHA_L, ALPHA_R, N_L, N_R);
}

function recordLogpdf(x: number): number {
  return densityPrims.builtinLogdensityof('DoubleSidedCrystalBall', {
    m0: M0, sigmaL: SIGMA_L, sigmaR: SIGMA_R,
    alphaL: ALPHA_L, alphaR: ALPHA_R, nL: N_L, nR: N_R,
  }, x);
}

test('DoubleSidedCrystalBall: record/kwarg density matches positional at left tail (x=-2.0)', () => {
  const got = recordLogpdf(-2.0);
  const want = referenceLogpdf(-2.0);
  assert.ok(Math.abs(got - want) < 1e-12,
    `left tail: got ${got}, want ${want} (Δ=${Math.abs(got - want)})`);
});

test('DoubleSidedCrystalBall: record/kwarg density matches positional at right tail (x=4.0)', () => {
  const got = recordLogpdf(4.0);
  const want = referenceLogpdf(4.0);
  assert.ok(Math.abs(got - want) < 1e-12,
    `right tail: got ${got}, want ${want} (Δ=${Math.abs(got - want)})`);
});
