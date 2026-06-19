'use strict';

// Spec §07 integer arithmetic: `div(a, b) = ⌊a/b⌋` (integer floor-division)
// and `mod(a, b) = a − b·⌊a/b⌋` (floor-modulo, sign of the divisor), both
// over the `integers` domain. These pin the two behaviours that JS-native
// operators get wrong for mixed-sign operands:
//
//   - JS `%` is truncated remainder (sign of the dividend): `-7 % 3 === -1`,
//     whereas spec floor-mod is `2`.
//   - JS `/` is real division; spec `div` floors.
//
// Oracle is the closed-form spec formula evaluated by hand — INDEPENDENT of
// the engine's own arithmetic (no `%` / `/` round-trip).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');
const valueOps = require('../value-ops.ts');
const value = require('../value.ts');
const R = orchestrator.resolveIRToValue;

function ctx() {
  const built = orchestrator.buildDerivations(processSource('z = 0.0\n').bindings);
  return [built.bindings, built.fixedValues];
}
const L = (v: any) => ({ kind: 'lit', value: v });
const C = (op: any, ...a: any[]) => ({ kind: 'call', op, args: a });

// (a, b) → [⌊a/b⌋, a − b·⌊a/b⌋], computed without JS % or truncating /.
function oracle(a: any, b: any) {
  const q = Math.floor(a / b);
  return [q, a - b * q];
}
const PAIRS = [[-7, 3], [7, -3], [7, 3], [-7, -3], [10, 4], [-10, 4], [0, 5]];

test('canonical value-ops.floorDiv / floorMod match the spec §07 closed form', () => {
  for (const [a, b] of PAIRS) {
    const [d, m] = oracle(a, b);
    assert.equal(valueOps.floorDiv(a, b), d, `floorDiv(${a},${b})`);
    assert.equal(valueOps.floorMod(a, b), m, `floorMod(${a},${b})`);
  }
  // The case JS `%` famously gets wrong, called out explicitly.
  assert.equal(valueOps.floorMod(-7, 3), 2, 'floor-mod sign-of-divisor');
  assert.notEqual(-7 % 3, 2, 'guards against a JS-`%` regression');
});

test('scalar div/mod through the deterministic evaluator floor per spec §07', () => {
  const [B, FV] = ctx();
  for (const [a, b] of PAIRS) {
    if (b === 0) continue;
    const [d, m] = oracle(a, b);
    assert.equal(R(C('div', L(a), L(b)), B, FV), d, `div(${a},${b})`);
    assert.equal(R(C('mod', L(a), L(b)), B, FV), m, `mod(${a},${b})`);
  }
});

test('elementwise div/mod (batched broadcast path) floor per spec §07', () => {
  const as = [-7, 7, -10, 10];
  const bs = [3, -3, 4, 4];
  const expD = as.map((a, i) => oracle(a, bs[i])[0]);
  const expM = as.map((a, i) => oracle(a, bs[i])[1]);
  assert.deepEqual(Array.from(valueOps.floorDivElem(value.vector(as), value.vector(bs)).data), expD);
  assert.deepEqual(Array.from(valueOps.modElem(value.vector(as), value.vector(bs)).data), expM);
});

test('div/mod are integer-domain: a real operand is a static error (spec §07)', () => {
  const errs = (src: any) => processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
  const dErr = errs('a = 7.0\nb = 2.0\nq = div(a, b)\n');
  assert.equal(dErr.length, 1, 'real div operand rejected');
  assert.match(dErr[0].message, /div: operands must be integer/);
  assert.match(dErr[0].message, /use `divide` for real division/);

  const mErr = errs('a = 7.0\nb = 2.0\nr = mod(a, b)\n');
  assert.equal(mErr.length, 1, 'real mod operand rejected');
  assert.match(mErr[0].message, /mod: operands must be integer/);

  // Integer operands pass; `divide` keeps its real (real, real) → real signature.
  assert.deepEqual(errs('a = 7\nb = 2\nq = div(a, b)\nr = mod(a, b)\n'), []);
  assert.deepEqual(errs('a = 7.0\nb = 2.0\nq = divide(a, b)\n'), []);
});
