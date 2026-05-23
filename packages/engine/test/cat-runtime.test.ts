'use strict';

// Spec §07: cat(...) — structural concatenation. Three shape classes:
//   * cat(scalar, scalar, ...)   → vector of those scalars
//   * cat(vector, vector, ...)   → concatenated vector
//   * cat(record, record, ...)   → merged record (disjoint fields)
// Mixing shape classes (e.g. scalar + vector) is a static error per
// spec; the runtime tolerates anything the static checker passed
// but throws on duplicate record fields and on unsupported shapes.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');

function lit(v: any)        { return { kind: 'lit', value: v }; }
function vec(...vs: any[])    { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function rec(...kvs: any[])   {
  return { kind: 'call', op: 'record',
    fields: kvs.map(([k, v]) => ({ name: k, value: lit(v) })) };
}
function call(op: any, ...args: any[]) { return { kind: 'call', op, args }; }
const ev = (ir: any) => sampler.evaluateExpr(ir, {});

// =====================================================================
// All-scalar form
// =====================================================================

test('cat(scalar, scalar, ...) ⇒ vector of those scalars', () => {
  assert.deepEqual(ev(call('cat', lit(1), lit(2), lit(3))), [1, 2, 3]);
});

test('cat of a single scalar produces a length-1 vector', () => {
  assert.deepEqual(ev(call('cat', lit(42))), [42]);
});

test('cat() with no arguments ⇒ empty vector', () => {
  assert.deepEqual(ev(call('cat')), []);
});

// =====================================================================
// All-vector form
// =====================================================================

test('cat(vector, vector) ⇒ concatenated vector', () => {
  assert.deepEqual(ev(call('cat', vec(1, 2), vec(3, 4, 5))), [1, 2, 3, 4, 5]);
});

test('cat of an empty vector preserves the rest', () => {
  assert.deepEqual(ev(call('cat', vec(), vec(1, 2, 3))), [1, 2, 3]);
});

// =====================================================================
// All-record form
// =====================================================================

test('cat(record, record) ⇒ merged record', () => {
  assert.deepEqual(
    ev(call('cat', rec(['a', 1], ['b', 2]), rec(['c', 3]))),
    { a: 1, b: 2, c: 3 });
});

test('cat of records with duplicate field names ⇒ runtime error', () => {
  assert.throws(
    () => ev(call('cat', rec(['a', 1]), rec(['a', 9]))),
    /duplicate field/);
});

// =====================================================================
// Field-order preservation
// =====================================================================

test('cat: field order follows argument order, then within each argument', () => {
  const r = ev(call('cat',
    rec(['z', 1], ['a', 2]),
    rec(['m', 3], ['b', 4])));
  // Keys in insertion order: z, a, m, b
  assert.deepEqual(Object.keys(r), ['z', 'a', 'm', 'b']);
});
