'use strict';

// Conformance regressions for `mod` floor-modulo (spec §07):
//   mod(a, b) = a − b·floor(a/b),  domain integers, b ≠ 0.
//
// The scalar path (sampler ARITH_OPS) was already fixed and is pinned
// by test/arith-mod.test.ts. This file pins the remaining paths that
// still used JS `%` (truncated remainder) and the type-domain check:
//   H1 — the constant folder (materialiser._foldNumericIR) and the
//        shaped/elementwise primitive (value-ops.modElem).
//   M1 — the (Int,Int)→Int domain is enforced at typecheck time, so a
//        real operand to `mod` produces a diagnostic.
//   L1 — mod(a, 0) runtime behaviour is pinned (matches `div`'s
//        unguarded IEEE convention: produces NaN, no diagnostic).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const materialiser = require('../materialiser.ts');
const valueOps = require('../value-ops.ts');
const { processSource } = require('../index.ts');

const lit = (v: any) => ({ kind: 'lit', value: v });
const call = (op: any, ...args: any[]) => ({ kind: 'call', op, args });

// =====================================================================
// H1 — constant folder (materialiser._foldNumericIR)
// =====================================================================

test('H1: _foldNumericIR folds mod with floor-modulo, not JS `%`', () => {
  const fold = materialiser._foldNumericIR;
  assert.equal(fold(call('mod', lit(-7), lit(3))), 2);   // JS `%` would give -1
  assert.equal(fold(call('mod', lit(7), lit(3))), 1);
  assert.equal(fold(call('mod', lit(-7), lit(-3))), -1); // shares sign of divisor
  assert.equal(fold(call('mod', lit(7), lit(-3))), -2);  // JS `%` would give 1
  assert.equal(fold(call('mod', lit(8), lit(3))), 2);    // agrees with `%`
});

test('H1: _foldNumericIR folds nested mod-over-arith', () => {
  const fold = materialiser._foldNumericIR;
  // mod(sub(0, 7), 3) === mod(-7, 3) === 2
  assert.equal(fold(call('mod', call('sub', lit(0), lit(7)), lit(3))), 2);
});

// =====================================================================
// H1 — shaped/elementwise primitive (value-ops.modElem)
// =====================================================================

const floorMod = (a: number, b: number) => a - b * Math.floor(a / b);

test('H1: modElem applies floor-modulo elementwise over a vector', () => {
  const a = valueOps._nestedToValue([-7, 7, -7, 7, 8, 0]);
  const b = valueOps._nestedToValue([3, 3, -3, -3, 3, 5]);
  const out = valueOps.modElem(a, b);
  const got = valueOps._valueToNested(out);
  const expected = [-7, 7, -7, 7, 8, 0].map((x, i) =>
    floorMod(x, [3, 3, -3, -3, 3, 5][i]));
  assert.deepEqual(got, expected);
  assert.deepEqual(got, [2, 1, -1, -2, 2, 0]);
});

test('H1: modElem broadcasts a scalar divisor with floor-modulo', () => {
  const a = valueOps._nestedToValue([-7, 7, -7, 7]);
  const b = valueOps._nestedToValue(3);
  const got = valueOps._valueToNested(valueOps.modElem(a, b));
  assert.deepEqual(got, [2, 1, 2, 1]);
});

// =====================================================================
// M1 — integer-domain enforcement at typecheck time
// =====================================================================

test('M1: real operands to mod produce an integer-domain diagnostic', () => {
  const { diagnostics } = processSource('m = mod(2.5, 2.0)');
  const hit = diagnostics.find((d: any) =>
    d.severity === 'error' && /mod/.test(d.message) && /integer/i.test(d.message));
  assert.ok(hit,
    'expected a mod integer-domain diagnostic, got: '
    + JSON.stringify(diagnostics.map((d: any) => d.message)));
});

test('M1: integer-literal mod typechecks clean', () => {
  const { diagnostics } = processSource('m = mod(7, 3)');
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [],
    'mod(7,3) should be clean, got: '
    + JSON.stringify(errs.map((d: any) => d.message)));
});

// =====================================================================
// M4 — integer-domain enforcement drills through arrays (elemPrim)
// =====================================================================

test('M4: a real-element ARRAY operand to mod produces the integer-domain diagnostic', () => {
  const { diagnostics } = processSource('m = mod([1.5, 2.5], [1, 2])');
  const hit = diagnostics.find((d: any) =>
    d.severity === 'error' && /mod/.test(d.message) && /integer/i.test(d.message));
  assert.ok(hit,
    'expected a mod integer-domain diagnostic for a real-element array operand, got: '
    + JSON.stringify(diagnostics.map((d: any) => d.message)));
  assert.ok(/argument 1/.test(hit.message),
    `diagnostic should point at the offending (first) argument, got: ${hit.message}`);
});

test('M4: an integer-element ARRAY mod typechecks clean', () => {
  const { diagnostics } = processSource('m = mod([1, 2], [3, 4])');
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [],
    'mod over integer-element arrays should be clean, got: '
    + JSON.stringify(errs.map((d: any) => d.message)));
});

// =====================================================================
// L1 — mod(a, 0) runtime behaviour (matches `div`'s unguarded IEEE
// convention: no diagnostic, runtime NaN).
// =====================================================================

test('L1: mod(a, 0) is unguarded like div — no static diagnostic', () => {
  // `div` does not guard b=0; for consistency `mod` does not either.
  const modDiags = processSource('m = mod(7, 0)').diagnostics
    .filter((d: any) => d.severity === 'error');
  const divDiags = processSource('d = div(7, 0)').diagnostics
    .filter((d: any) => d.severity === 'error');
  assert.deepEqual(modDiags, divDiags,
    'mod(_,0) should be as (un)guarded as div(_,0)');
});

test('L1: floor-modulo by zero is NaN at the value layer', () => {
  // a - 0*floor(a/0) = a - 0*Inf = NaN. Pins the documented runtime.
  assert.ok(Number.isNaN(floorMod(7, 0)));
  const out = valueOps.modElem(valueOps._nestedToValue([7, -3]),
                               valueOps._nestedToValue([0, 0]));
  const got = valueOps._valueToNested(out);
  assert.ok(got.every((x: number) => Number.isNaN(x)),
    'modElem by zero should be NaN, got ' + JSON.stringify(got));
});
