'use strict';

// Conformance regressions for `div` integer floor-division (spec §07,
// line 419):
//   div(a, b) = floor(a/b),  domain integers, b ≠ 0.
//
// `div` is the MIRROR of the `mod` breaking-change (see
// test/mod-conformance.test.ts). The engine previously typed `div` as
// real division and evaluated `a / b` on every path, conflating it with
// the SEPARATE real/complex op `divide` (spec §07 line 449). This file
// pins the floor-division semantics and (Int,Int)→Int domain across all
// paths, and that `divide` is left untouched:
//   S1 — scalar path (sampler.evaluateExpr / ARITH_OPS).
//   S1 — scalar ≡ batched bit-identity (evaluateExprN).
//   H1 — the constant folder (materialiser._foldNumericIR) and the
//        shaped/elementwise primitive (value-ops.floorDivElem).
//   M1 — the (Int,Int)→Int domain is enforced at typecheck time, so a
//        real operand to `div` produces a diagnostic.
//   D1 — `divide` keeps its real-division semantics (NOT floored).
//   L1 — div(a, 0) runtime behaviour is pinned (matches `mod`'s
//        unguarded IEEE convention: no diagnostic, non-finite result —
//        floor(±Inf) = ±Inf for a ≠ 0, NaN for 0/0).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const materialiser = require('../materialiser.ts');
const valueOps = require('../value-ops.ts');
require('../ops-declarations.ts');   // registers the broadcasted(<op>) variants
const ops = require('../ops.ts');
const { processSource } = require('../index.ts');

const lit = (v: any) => ({ kind: 'lit', value: v });
const ref = (n: any) => ({ kind: 'ref', ns: 'self', name: n });
const call = (op: any, ...args: any[]) => ({ kind: 'call', op, args });

const floorDiv = (a: number, b: number) => Math.floor(a / b);
const divScalar = (a: number, b: number) =>
  sampler.evaluateExpr(call('div', lit(a), lit(b)), {});

// =====================================================================
// S1 — scalar floor-division values (spec §07)
// =====================================================================

test('S1: div is integer floor-division for mixed signs', () => {
  assert.equal(divScalar(-7, 3), -3);   // floor(-2.33) = -3, not -2
  assert.equal(divScalar(7, 3), 2);
  assert.equal(divScalar(-7, -3), 2);   // floor(2.33) = 2
  assert.equal(divScalar(7, -3), -3);   // floor(-2.33) = -3
});

test('S1: div agrees with truncation when it divides evenly', () => {
  assert.equal(divScalar(8, 4), 2);
  assert.equal(divScalar(9, 3), 3);
  assert.equal(divScalar(0, 5), 0);
});

// =====================================================================
// S1 — scalar ≡ batched bit-identity (ARITH_OPS_N derives from ARITH_OPS)
// =====================================================================

test('S1: batched evaluateExprN is bit-identical to scalar (scalar divisor)', () => {
  const a = new Float64Array([-7, 7, -7, 7, 8, 0]);
  const bConst = 3;
  const ir = call('div', ref('a'), lit(bConst));
  const batched = sampler.evaluateExprN(ir, { a }, a.length, {});
  assert.ok(batched.BYTES_PER_ELEMENT, 'expected Float64Array result');
  for (let i = 0; i < a.length; i++) {
    const scalar = divScalar(a[i], bConst);
    assert.equal(batched[i], scalar,
      `batched[${i}]=${batched[i]} != scalar div(${a[i]},${bConst})=${scalar}`);
  }
  assert.deepEqual(Array.from(batched), [-3, 2, -3, 2, 2, 0]);
});

test('S1: batched with both operands per-atom is bit-identical to scalar', () => {
  const a = new Float64Array([-7, 7, -7, 7]);
  const b = new Float64Array([3, 3, -3, -3]);
  const ir = call('div', ref('a'), ref('b'));
  const batched = sampler.evaluateExprN(ir, { a, b }, a.length, {});
  for (let i = 0; i < a.length; i++) {
    assert.equal(batched[i], divScalar(a[i], b[i]));
  }
  assert.deepEqual(Array.from(batched), [-3, 2, 2, -3]);
});

// =====================================================================
// H1 — constant folder (materialiser._foldNumericIR)
// =====================================================================

test('H1: _foldNumericIR folds div with integer floor-division', () => {
  const fold = materialiser._foldNumericIR;
  assert.equal(fold(call('div', lit(-7), lit(3))), -3);  // real div would give -2.33
  assert.equal(fold(call('div', lit(7), lit(3))), 2);
  assert.equal(fold(call('div', lit(-7), lit(-3))), 2);
  assert.equal(fold(call('div', lit(7), lit(-3))), -3);
  assert.equal(fold(call('div', lit(8), lit(4))), 2);
});

test('H1: _foldNumericIR folds nested div-over-arith', () => {
  const fold = materialiser._foldNumericIR;
  // div(sub(0, 7), 3) === div(-7, 3) === -3
  assert.equal(fold(call('div', call('sub', lit(0), lit(7)), lit(3))), -3);
});

// =====================================================================
// H1 — shaped/elementwise primitive (value-ops.floorDivElem)
// =====================================================================

test('H1: floorDivElem applies floor-division elementwise over a vector', () => {
  const a = valueOps._nestedToValue([-7, 7, -7, 7, 8, 0]);
  const b = valueOps._nestedToValue([3, 3, -3, -3, 4, 5]);
  const out = valueOps.floorDivElem(a, b);
  const got = valueOps._valueToNested(out);
  const expected = [-7, 7, -7, 7, 8, 0].map((x, i) =>
    floorDiv(x, [3, 3, -3, -3, 4, 5][i]));
  assert.deepEqual(got, expected);
  assert.deepEqual(got, [-3, 2, 2, -3, 2, 0]);
});

test('H1: floorDivElem broadcasts a scalar divisor with floor-division', () => {
  const a = valueOps._nestedToValue([-7, 7, -7, 7]);
  const b = valueOps._nestedToValue(3);
  const got = valueOps._valueToNested(valueOps.floorDivElem(a, b));
  assert.deepEqual(got, [-3, 2, -3, 2]);
});

// =====================================================================
// H2 — shaped/variant dispatch path (ops-declarations BCAST_TABLE)
// The `broadcasted(<op>)` variant for `div` must route to the floor
// primitive (floorDivElem), while `divide` stays on the real `divElem`.
// =====================================================================

test('H2: broadcasted(div) variant routes to integer floor-division', () => {
  const a = valueOps._nestedToValue([-7, 7, -7, 7, 8]);
  const b = valueOps._nestedToValue([3, 3, -3, -3, 4]);
  const out = ops.dispatchVariant('div', [a, b], { wrappingOp: 'broadcast' });
  assert.deepEqual(valueOps._valueToNested(out), [-3, 2, 2, -3, 2]);
});

test('H2: broadcasted(divide) variant stays real division (NOT floored)', () => {
  const a = valueOps._nestedToValue([-7, 7]);
  const b = valueOps._nestedToValue([3, 3]);
  const out = ops.dispatchVariant('divide', [a, b], { wrappingOp: 'broadcast' });
  assert.deepEqual(valueOps._valueToNested(out), [-7 / 3, 7 / 3]);
});

// =====================================================================
// M1 — integer-domain enforcement at typecheck time
// =====================================================================

test('M1: real operands to div produce an integer-domain diagnostic', () => {
  const { diagnostics } = processSource('q = div(2.5, 2.0)');
  const hit = diagnostics.find((d: any) =>
    d.severity === 'error' && /\bdiv\b/.test(d.message) && /integer/i.test(d.message));
  assert.ok(hit,
    'expected a div integer-domain diagnostic, got: '
    + JSON.stringify(diagnostics.map((d: any) => d.message)));
});

test('M1: integer-literal div typechecks clean', () => {
  const { diagnostics } = processSource('q = div(7, 3)');
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [],
    'div(7,3) should be clean, got: '
    + JSON.stringify(errs.map((d: any) => d.message)));
});

// =====================================================================
// M4 — integer-domain enforcement drills through arrays (elemPrim)
// =====================================================================

test('M4: a real-element ARRAY operand to div produces the integer-domain diagnostic', () => {
  const { diagnostics } = processSource('q = div([1.5, 2.5], [1, 2])');
  const hit = diagnostics.find((d: any) =>
    d.severity === 'error' && /\bdiv\b/.test(d.message) && /integer/i.test(d.message));
  assert.ok(hit,
    'expected a div integer-domain diagnostic for a real-element array operand, got: '
    + JSON.stringify(diagnostics.map((d: any) => d.message)));
  assert.ok(/argument 1/.test(hit.message),
    `diagnostic should point at the offending (first) argument, got: ${hit.message}`);
});

test('M4: an integer-element ARRAY div typechecks clean', () => {
  const { diagnostics } = processSource('q = div([1, 2], [3, 4])');
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [],
    'div over integer-element arrays should be clean, got: '
    + JSON.stringify(errs.map((d: any) => d.message)));
});

// =====================================================================
// D1 — `divide` is the SEPARATE real-division op (spec §07 line 449),
// left untouched: NOT floored, accepts reals.
// =====================================================================

test('D1: divide keeps real-division semantics (not floored)', () => {
  const fold = materialiser._foldNumericIR;
  assert.equal(fold(call('divide', lit(-7), lit(3))), -7 / 3);
  assert.equal(sampler.evaluateExpr(call('divide', lit(-7), lit(3)), {}), -7 / 3);
});

test('D1: divide on real operands typechecks clean (no integer-domain error)', () => {
  const { diagnostics } = processSource('q = divide(2.5, 2.0)');
  const errs = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [],
    'divide(2.5, 2.0) should be clean (real division), got: '
    + JSON.stringify(errs.map((d: any) => d.message)));
});

// =====================================================================
// L1 — div(a, 0) runtime behaviour (matches `mod`'s unguarded IEEE
// convention: no static diagnostic; non-finite IEEE result at runtime).
// =====================================================================

test('L1: div(a, 0) is unguarded like mod — no static diagnostic', () => {
  const divDiags = processSource('q = div(7, 0)').diagnostics
    .filter((d: any) => d.severity === 'error');
  const modDiags = processSource('m = mod(7, 0)').diagnostics
    .filter((d: any) => d.severity === 'error');
  assert.deepEqual(divDiags, modDiags,
    'div(_,0) should be as (un)guarded as mod(_,0)');
});

test('L1: floor-division by zero is non-finite at the value layer', () => {
  // No b≠0 guard (spec precondition): plain IEEE. floor(a/0) = floor(±Inf)
  // = ±Inf for a ≠ 0, and floor(0/0) = NaN. Pins the documented runtime
  // (unlike `mod`, which is always NaN at b=0 — div is ±Inf for a ≠ 0).
  assert.equal(floorDiv(7, 0), Infinity);
  assert.equal(floorDiv(-3, 0), -Infinity);
  assert.ok(Number.isNaN(floorDiv(0, 0)));
  assert.equal(divScalar(7, 0), Infinity);
  const out = valueOps.floorDivElem(valueOps._nestedToValue([7, -3, 0]),
                                    valueOps._nestedToValue([0, 0, 0]));
  const got = valueOps._valueToNested(out);
  assert.ok(got.every((x: number) => !Number.isFinite(x)),
    'floorDivElem by zero should be non-finite, got ' + JSON.stringify(got));
  assert.equal(got[0], Infinity);
  assert.equal(got[1], -Infinity);
  assert.ok(Number.isNaN(got[2]));
});
