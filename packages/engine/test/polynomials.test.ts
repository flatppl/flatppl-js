'use strict';

// `polynomials` standard module (spec §09) — legendre / hermite /
// laguerre, mirroring the landed chebyshev. Each is a three-term
// recurrence; the pinned values are an INDEPENDENT oracle (explicit
// closed-form expansions P_n/H_n/L_n, cross-checked in Julia to machine
// precision — NOT the engine's own output):
//   legendre  P_2=(3x²−1)/2, P_3=(5x³−3x)/2, P_4=(35x⁴−30x²+3)/8
//   hermite   H_2=4x²−2,     H_3=8x³−12x,    H_4=16x⁴−48x²+12  (physicist's)
//   laguerre  L_2=(x²−4x+2)/2, L_3=(−x³+9x²−18x+6)/6

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const sampler = require('../sampler.ts');
const sm = require('../standard-modules.ts');

const { _legendre, _hermite, _laguerre } = sm._internal;
const near = (a: any, b: any) => Math.abs(a - b) < 1e-9;

// =====================================================================
// Unit: recurrence impl vs the closed-form oracle
// =====================================================================

test('legendre: matches the closed-form oracle', () => {
  assert.ok(near(_legendre(0, 0.5), 1));
  assert.ok(near(_legendre(1, 0.5), 0.5));
  assert.ok(near(_legendre(2, 0.5), -0.125));
  assert.ok(near(_legendre(3, 0.5), -0.4375));
  assert.ok(near(_legendre(4, 0.5), -0.2890625));
  assert.ok(near(_legendre(4, -0.7), -0.4120625));
  // P_n(1) = 1 for all n.
  assert.ok(near(_legendre(3, 1.0), 1.0));
  assert.ok(near(_legendre(7, 1.0), 1.0));
});

test('hermite: matches the physicist closed-form oracle', () => {
  assert.ok(near(_hermite(0, 0.5), 1));
  assert.ok(near(_hermite(1, 0.5), 1.0));   // H_1 = 2x
  assert.ok(near(_hermite(2, 0.5), -1.0));
  assert.ok(near(_hermite(3, 0.5), -5.0));
  assert.ok(near(_hermite(4, 0.5), 1.0));
  assert.ok(near(_hermite(3, 1.0), -4.0));
  assert.ok(near(_hermite(4, -0.7), -7.6784));
});

test('laguerre: matches the closed-form oracle', () => {
  assert.ok(near(_laguerre(0, 0.5), 1));
  assert.ok(near(_laguerre(1, 0.5), 0.5));   // L_1 = 1 − x
  assert.ok(near(_laguerre(2, 0.5), 0.125));
  assert.ok(near(_laguerre(3, 0.5), -0.14583333333333334));
  assert.ok(near(_laguerre(4, 0.5), -0.3307291666666667));
  assert.ok(near(_laguerre(3, 1.0), -2 / 3));
  assert.ok(near(_laguerre(4, -0.7), 5.508670833333333));
});

test('polynomials: degree 0 is the constant 1; negative degree throws', () => {
  for (const f of [_legendre, _hermite, _laguerre]) {
    assert.equal(f(0, 3.14), 1);
    assert.throws(() => f(-1, 0.5), /non-negative integer/);
    assert.throws(() => f(2.5, 0.5), /non-negative integer/);
  }
});

// =====================================================================
// End-to-end through the engine (alias form, spec §09 loading)
// =====================================================================

test('polynomials: legendre/hermite/laguerre evaluate end-to-end', () => {
  const src = `poly = standard_module("polynomials", "0.1")
legendre = poly.legendre
hermite = poly.hermite
laguerre = poly.laguerre
a = legendre(3, 0.5)
b = hermite(4, 0.5)
c = laguerre(3, 1.0)
`;
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, JSON.stringify(errs));
  const built = orchestrator.buildDerivations(r.bindings);
  const env: any = { __moduleRegistry: r.loweredModule.moduleRegistry };
  const evalName = (nm: any) =>
    sampler.evaluateExpr(r.loweredModule.bindings.get(nm).rhs, env);
  assert.ok(near(evalName('a'), -0.4375));
  assert.ok(near(evalName('b'), 1.0));
  assert.ok(near(evalName('c'), -2 / 3));
});
