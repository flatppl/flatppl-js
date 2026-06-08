'use strict';

// Regression for the `mod` floor-modulo fix (spec §07):
//   mod(a, b) = a − b·floor(a/b),  domain integers, b ≠ 0.
//
// The engine previously used JS `%` (truncated remainder, sign follows
// the dividend), which disagrees with the spec for mixed-sign operands:
// JS `-7 % 3 === -1`, but the spec's floor-modulo gives `2` (result
// shares the sign of the divisor). This file pins:
//   1. the corrected scalar values (evaluateExpr / ARITH_OPS),
//   2. scalar ≡ batched bit-identity (evaluateExprN derives mod from
//      the same scalar primitive via initARITHOPSN).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');

const lit = (v: any) => ({ kind: 'lit', value: v });
const ref = (n: any) => ({ kind: 'ref', ns: 'self', name: n });
const call = (op: any, ...args: any[]) => ({ kind: 'call', op, args });

const modScalar = (a: number, b: number) =>
  sampler.evaluateExpr(call('mod', lit(a), lit(b)), {});

// =====================================================================
// Scalar floor-modulo values (spec §07)
// =====================================================================

test('mod: floor-modulo matches spec for mixed signs', () => {
  // The canonical disagreement with JS `%`.
  assert.equal(modScalar(-7, 3), 2);   // JS `%` would give -1
  assert.equal(modScalar(7, 3), 1);
  assert.equal(modScalar(-7, -3), -1); // result shares sign of divisor
  assert.equal(modScalar(7, -3), -2);  // JS `%` would give 1
});

test('mod: agrees with JS `%` when signs match (no sign skew)', () => {
  assert.equal(modScalar(8, 3), 2);
  assert.equal(modScalar(9, 3), 0);
  assert.equal(modScalar(0, 5), 0);
});

// =====================================================================
// Scalar ≡ batched bit-identity (ARITH_OPS_N derives from ARITH_OPS)
// =====================================================================

test('mod: batched evaluateExprN is bit-identical to scalar', () => {
  const a = new Float64Array([-7, 7, -7, 7, 8, 0]);
  const bConst = 3;
  const ir = call('mod', ref('a'), lit(bConst));
  const batched = sampler.evaluateExprN(ir, { a }, a.length, {});
  assert.ok(batched.BYTES_PER_ELEMENT, 'expected Float64Array result');
  for (let i = 0; i < a.length; i++) {
    const scalar = modScalar(a[i], bConst);
    assert.equal(batched[i], scalar,
      `batched[${i}]=${batched[i]} != scalar mod(${a[i]},${bConst})=${scalar}`);
  }
  assert.deepEqual(Array.from(batched), [2, 1, 2, 1, 2, 0]);
});

test('mod: batched with both operands per-atom is bit-identical to scalar', () => {
  const a = new Float64Array([-7, 7, -7, 7]);
  const b = new Float64Array([3, 3, -3, -3]);
  const ir = call('mod', ref('a'), ref('b'));
  const batched = sampler.evaluateExprN(ir, { a, b }, a.length, {});
  for (let i = 0; i < a.length; i++) {
    assert.equal(batched[i], modScalar(a[i], b[i]));
  }
  assert.deepEqual(Array.from(batched), [2, 1, -1, -2]);
});
