'use strict';

// Spec §09 `particle-physics` Wigner rotation functions: the small d-function
// d^j_{m1 m2}(β) = <j m1| exp(-iβJ_y) |j m2> of cos β, the D-function
// D = exp(-i(m1 α + m2 γ)) d, and their `_doublearg` forms, which take 2j, 2m1
// and 2m2 so half-integer spin stays integral. Conventions follow PDG.
//
// ORACLE: SymPy 1.14 `sympy.physics.quantum.spin.Rotation.d(j, m1, m2, acos(c))`
// and `Rotation.D(j, m1, m2, α, acos(c), γ)`, evaluated to 30 digits through
// the python MCP and cross-checked against `sympy.physics.wigner.wigner_d_small`
// (its matrices index m from -j). `Rotation.d(1, 1, 0, β) = -sin(β)/√2` and
// `Rotation.d(1/2, 1/2, -1/2, β) = -sin(β/2)` match the PDG d-function table.
// No value comes from the engine.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const HEAD = 'hepphys = standard_module("particle-physics", "0.1")\n';

function values(lines: string[]) {
  const r = processSource(HEAD + lines.join('\n') + '\n');
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), []);
  return orchestrator.buildDerivations(r.linkedBindings, { moduleRegistry: r.linkedModuleRegistry }).fixedValues;
}

// [2j, 2m1, 2m2, cos β, d^j_{m1 m2}(β)]
const D_SMALL: [number, number, number, number, number][] = [
  [0, 0, 0, 0.3, 1.0],
  [2, 2, 0, 0.3, -0.6745368781616021],
  [2, 0, 2, 0.3, 0.6745368781616021],
  [2, 2, 2, -0.7, 0.15],
  [2, -2, 2, -0.7, 0.85],
  [2, 2, -2, 0.3, 0.35],
  [2, 0, 0, -0.7, -0.7],
  [4, 4, 2, 0.3, -0.6200604809210146],
  [4, -4, -2, 0.3, 0.6200604809210146],
  [4, 2, -4, -0.7, -0.6070214164261423],
  [4, 0, 0, 0.3, -0.365],
  [4, -2, 2, 1.0, 0.0],
  [4, 4, 4, -1.0, 0.0],
  [4, 4, -4, -1.0, 1.0],
  [4, 0, 0, 0.0, -0.5],
  [6, 6, -6, 0.3, 0.042875],
  [6, -2, 4, -0.7, -0.527881822712243],
  [6, 0, 0, 0.3, -0.3825],
  [6, 4, -2, 0.9, -0.06375122547841727],
  [1, 1, -1, 0.3, -0.5916079783099616],
  [1, -1, 1, 0.3, 0.5916079783099616],
  [1, 1, 1, -0.7, 0.3872983346207417],
  [1, 1, -1, -1.0, -1.0],
  [1, -1, 1, -1.0, 1.0],
  [1, 1, 1, 1.0, 1.0],
  [3, 3, 1, 0.3, -0.6660517997873739],
  [3, -3, 1, 0.3, 0.48874840153191296],
  [3, 1, -1, -0.7, 0.5070749451511088],
  [3, -1, -3, 0.3, -0.6660517997873739],
  [3, 3, -3, -1.0, -1.0],
  [3, -3, 3, -1.0, 1.0],
  [3, 3, 3, 1.0, 1.0],
  [5, 5, -5, 0.3, -0.0724719773429703],
  [5, -3, 1, -0.7, -0.5819552173492389],
  [5, 1, 1, 0.3, -0.4635798205271666],
  [5, 5, 5, -1.0, 0.0],
  [5, 3, -1, 1.0, 0.0],
];

const close = (got: number, want: number, what: string) =>
  assert.ok(Math.abs(got - want) < 1e-12, `${what}: got ${got}, want ${want}`);

test('wignerd_doublearg matches SymPy for integer and half-integer j, negative m and m = ±j', () => {
  const lines = D_SMALL.map(([tj, a, b, c], i) => `d${i} = hepphys.wignerd_doublearg(${tj}, ${a}, ${b}, ${c})`);
  const fv = values(lines);
  D_SMALL.forEach(([tj, a, b, c, want], i) => close(fv.get(`d${i}`), want, `d(${tj}/2, ${a}/2, ${b}/2, ${c})`));
});

test('wignerd takes j, m1, m2 directly and equals the doubled form for integer spin', () => {
  const integral = D_SMALL.filter(([tj]) => tj % 2 === 0);
  const lines = integral.map(([tj, a, b, c], i) => `d${i} = hepphys.wignerd(${tj / 2}, ${a / 2}, ${b / 2}, ${c})`);
  const fv = values(lines);
  integral.forEach(([tj, a, b, c, want], i) => close(fv.get(`d${i}`), want, `d(${tj / 2}, ${a / 2}, ${b / 2}, ${c})`));
});

// High spin, where Wigner's alternating sum cancels (relative error 2e-4 at
// 2j = 80). SymPy Rotation.d at 40 digits, each value cross-checked against
// the exact sum evaluated in 80-digit mpmath.
const D_HIGH: [number, number, number, number, number][] = [
  [40, 0, 0, 0.3, 0.18028715947998047],
  [40, 40, -40, -0.7, 0.038759531084514356],
  [40, 12, -8, 0.9, 0.13808156062513945],
  [41, 1, -1, -0.05, -0.16761816679314271],
  [41, 41, 1, 0.3, 0.15532840634418830],
  [60, 20, -30, 0.3, -0.16642183987876062],
  [60, 0, 2, -0.7, 0.17053972275542101],
  [79, -79, 3, 0.9, 1.8196076741662578e-16],
  [79, 5, -7, 0.3, -0.12897775219950516],
  [80, 0, 0, -0.7, 0.14834359558654803],
  [80, 10, -6, 0.3, 0.094812606137472230],
  [80, 80, 80, 0.3, 3.2849916177340806e-8],
  [80, 40, -40, 0.0, -0.020852148314588703],
  [80, -24, 36, -0.9, 0.015838313078202240],
  [80, 2, 0, 0.99, 0.32080768085896322],
  [80, 78, -80, -0.3, -2.1560389535525159e-7],
];

test('wignerd_doublearg keeps relative precision up to 2j = 80', () => {
  const lines = D_HIGH.map(([tj, a, b, c], i) => `d${i} = hepphys.wignerd_doublearg(${tj}, ${a}, ${b}, ${c})`);
  const fv = values(lines);
  D_HIGH.forEach(([tj, a, b, c, want], i) => {
    const got = fv.get(`d${i}`);
    assert.ok(Math.abs(got - want) <= 1e-12 * Math.abs(want),
      `d(${tj}/2, ${a}/2, ${b}/2, ${c}): got ${got}, want ${want}`);
  });
});

// [2j, 2m1, 2m2, α, cos β, γ, Re D, Im D]
const D_BIG: [number, number, number, number, number, number, number, number][] = [
  [2, 2, 0, 0.4, 0.3, -1.1, -0.6212896074911282, 0.2626770329197432],
  [3, -1, 3, 0.4, -0.7, 2.0, -0.5372526669317915, -0.1910093502284051],
  [4, -2, 4, -0.9, 0.3, 0.25, 0.0567484121978815, -0.32902069496160774],
];

test('wignerD and wignerD_doublearg match SymPy Rotation.D', () => {
  const lines: string[] = [];
  D_BIG.forEach(([tj, a, b, al, c, g], i) => {
    lines.push(`w${i} = hepphys.wignerD_doublearg(${tj}, ${a}, ${b}, ${al}, ${c}, ${g})`);
    if (tj % 2 === 0) lines.push(`v${i} = hepphys.wignerD(${tj / 2}, ${a / 2}, ${b / 2}, ${al}, ${c}, ${g})`);
  });
  const fv = values(lines);
  D_BIG.forEach(([tj, , , , , , re, im], i) => {
    for (const k of tj % 2 === 0 ? [`w${i}`, `v${i}`] : [`w${i}`]) {
      close(fv.get(k).re, re, `${k} re`);
      close(fv.get(k).im, im, `${k} im`);
    }
  });
});

test('wignerd broadcasts over an array of cos β', () => {
  const fv = values(['d = hepphys.wignerd.(1, 1, 0, [0.3, -0.7])']);
  close(fv.get('d').data[0], -0.6745368781616021, 'cos β = 0.3');
  // Rotation.d(1, 1, 0, acos(-0.7)) = -sin(β)/√2 = -sqrt(0.51/2)
  close(fv.get('d').data[1], -0.5049752469181039, 'cos β = -0.7');
});

for (const [call, msg] of [
  ['hepphys.wignerd(1, 2, 0, 0.3)', /wignerd: j = 1, m1 = 2, m2 = 0 is not an angular momentum/],
  ['hepphys.wignerd_doublearg(2, 1, 0, 0.3)', /wignerd_doublearg: two_j = 2, two_m1 = 1, two_m2 = 0 is not an angular momentum/],
  ['hepphys.wignerD(1, 0, -2, 0.0, 0.3, 0.0)', /wignerD: j = 1, m1 = 0, m2 = -2 is not an angular momentum/],
] as const) {
  test(`an invalid angular momentum is refused: ${call}`, async () => {
    const { ctx } = ctxFor(`${HEAD}a = ${call}\n`, 1);
    await assert.rejects(ctx.getMeasure('a'), msg);
  });
}

test('cos β outside [-1, 1] gives NaN', () => {
  assert.ok(Number.isNaN(values(['a = hepphys.wignerd(1, 1, 0, 1.5)']).get('a')));
});
