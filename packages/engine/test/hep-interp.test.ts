'use strict';

// Unit tests for the spec §09 systematic-interpolation functions
// (hepphys.interp_*), registered in the particle-physics standard module.
// Each matches the corresponding pyhf interpcode (verified against pyhf's
// interpolators to machine precision): pwlin→code0, pwexp→code1,
// poly2_lin→code2, poly6_lin→code4p, poly6_exp→code4.
//
// These call the registered `impl` directly (the deterministic function math),
// independent of the measure-density path.

const test = require('node:test');
const assert = require('node:assert');
require('..');  // ensure builtin standard modules are registered
const sm = require('../standard-modules.ts');

const ALPHAS = [-2.0, -0.5, 0.7, 1.5];
const TOL = 1e-12;

function impl(name: string) {
  const mod = sm.lookupStandardModule('particle-physics', '0.1');
  assert.ok(mod, 'particle-physics module registered');
  const b = mod.bindings.get(name);
  assert.ok(b && typeof b.impl === 'function', `${name} has an impl`);
  return b.impl;
}

function check(name: string, args: [number, number, number], expected: number[]) {
  const fn = impl(name);
  ALPHAS.forEach((a, i) => {
    const got = fn(args[0], args[1], args[2], a);
    assert.ok(Math.abs(got - expected[i]) < TOL,
      `${name}(${args}, ${a}): got ${got}, want ${expected[i]}`);
  });
}

test('interp_pwlin (code0)', () =>
  check('interp_pwlin', [0.9, 1.0, 1.2], [0.8, 0.95, 1.14, 1.2999999999999998]));

test('interp_pwexp (code1)', () =>
  check('interp_pwexp', [0.9, 1.0, 1.2], [0.81, 0.9486832980505138, 1.1361269771988887, 1.3145341380123987]));

test('interp_poly2_lin (code2)', () =>
  check('interp_poly2_lin', [47.0, 50.0, 53.0], [44.0, 48.5, 52.1, 54.5]));

test('interp_poly6_exp (code4 — normsys default)', () =>
  check('interp_poly6_exp', [0.88, 1.0, 1.15], [0.7744, 0.937444710705783, 1.102564077845851, 1.2332376088978148]));

test('interp_poly6_lin (code4p — histosys default)', () =>
  check('interp_poly6_lin', [47.0, 50.0, 53.0], [44.0, 48.5, 52.1, 54.5]));
