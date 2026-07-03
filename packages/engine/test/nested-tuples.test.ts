'use strict';

// Nested tuples — spec §04 (commit "Allow nested records and nested tuples").
//
//   - A tuple may appear inside another tuple (tuples nest); it still may
//     not appear inside an array, record, or table (see
//     containment-boundary.test.ts).
//   - Surface `t[i]` lowers to `get(t, i)` with a 1-BASED positive integer
//     literal index (spec §04). The engine-internal `tuple_get(t, slot)`
//     emitted by the multi-LHS decomposition rewriter is 0-BASED; the two
//     must not be conflated. Heterogeneous / nested tuples (whose elements
//     have genuinely different types) expose any off-by-one in the surface
//     `t[i]` type rule that a homogeneous tuple would hide.
//   - Positional decomposition `a, b = (...)` continues to bind per slot.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const T = require('../types.ts');

function infer(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  return { bindings: r.bindings, errors: errs };
}
function typeOf(bindings: any, name: string) {
  const b = bindings.get(name);
  return b && b.inferredType;
}
function valueOf(name: string, src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    throw new Error('source had errors: ' + errs.map((e: any) => e.message).join(' | '));
  }
  const ds = buildDerivations(r.bindings);
  return ds.fixedValues && ds.fixedValues.get(name);
}

// =====================================================================
// Surface t[i] is 1-based (spec §04) — the off-by-one regression guard
// =====================================================================

test('tuple t[i]: a heterogeneous flat tuple indexes 1-based by type', () => {
  // (10, 2.5) — element 1 is integer, element 2 is real. The off-by-one
  // bug typed t[1] as `real` (elem 2) and reported t[2] out of range.
  const { bindings, errors } = infer(`
    t = (10, 2.5)
    x = t[1]
    y = t[2]
  `);
  assert.equal(errors.length, 0, 'unexpected errors: ' + errors.map((e: any) => e.message).join(' | '));
  assert.ok(T.equal(typeOf(bindings, 'x'), T.INTEGER), 't[1] must be integer, got ' + T.show(typeOf(bindings, 'x')));
  assert.ok(T.equal(typeOf(bindings, 'y'), T.REAL), 't[2] must be real, got ' + T.show(typeOf(bindings, 'y')));
});

test('tuple t[i]: runtime value is 1-based and agrees with the type', () => {
  assert.equal(valueOf('x', `t = (10, 2.5)\nx = t[1]`), 10);
  assert.equal(valueOf('y', `t = (10, 2.5)\ny = t[2]`), 2.5);
});

test('tuple t[i]: indexing past the end is a 1-based out-of-range error', () => {
  const { errors } = infer(`
    t = (10, 2.5)
    z = t[3]
  `);
  assert.ok(errors.length >= 1, 'expected an out-of-range diagnostic for t[3]');
  assert.match(errors[0].message, /range/i);
});

// get0 is the ZERO-based variant of get (spec §07). Regression for the bug
// where get0 on a tuple was routed 1-based (like `get`), so get0(t, 0)
// resolved to index -1 and errored "tuple index 0 out of range". Exact
// scenario: the determiniser projects a builtin_sample (value, rngstate)
// tuple via get0(__sample, 0) / get0(__sample, 1).
test('tuple get0 is 0-based by type: get0(t,0) is the first element', () => {
  const { bindings, errors } = infer(`
    t = (10, 2.5)
    x = get0(t, 0)
    y = get0(t, 1)
  `);
  assert.equal(errors.length, 0, 'unexpected errors: ' + errors.map((e: any) => e.message).join(' | '));
  assert.ok(T.equal(typeOf(bindings, 'x'), T.INTEGER), 'get0(t,0) must be integer (first elem), got ' + T.show(typeOf(bindings, 'x')));
  assert.ok(T.equal(typeOf(bindings, 'y'), T.REAL), 'get0(t,1) must be real (second elem), got ' + T.show(typeOf(bindings, 'y')));
});

test('tuple get0: runtime value is 0-based and agrees with the type', () => {
  assert.equal(valueOf('x', `t = (10, 2.5)\nx = get0(t, 0)`), 10);
  assert.equal(valueOf('y', `t = (10, 2.5)\ny = get0(t, 1)`), 2.5);
});

// =====================================================================
// Nested tuples — literal, access, decomposition
// =====================================================================

test('nested tuple literal types recursively', () => {
  const { bindings, errors } = infer(`t = (1.0, (2.0, 3.0))`);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 't');
  assert.equal(t.kind, 'tuple');
  assert.equal(t.elems.length, 2);
  assert.ok(T.equal(t.elems[0], T.REAL));
  assert.equal(t.elems[1].kind, 'tuple');
  assert.equal(t.elems[1].elems.length, 2);
});

test('nested tuple: t[2] returns the inner tuple (type + value)', () => {
  const { bindings, errors } = infer(`
    t = (1.0, (2.0, 3.0))
    inner = t[2]
  `);
  assert.equal(errors.length, 0);
  const ty = typeOf(bindings, 'inner');
  assert.equal(ty.kind, 'tuple');
  assert.ok(T.equal(ty.elems[0], T.REAL) && T.equal(ty.elems[1], T.REAL));
  assert.deepEqual(valueOf('inner', `t = (1.0, (2.0, 3.0))\ninner = t[2]`), [2, 3]);
});

test('nested tuple: chained index t[2][1] reaches the inner scalar', () => {
  const { bindings, errors } = infer(`
    t = (1.0, (2.0, 3.0))
    x = t[2][1]
  `);
  assert.equal(errors.length, 0, errors.map((e: any) => e.message).join(' | '));
  assert.ok(T.equal(typeOf(bindings, 'x'), T.REAL));
  assert.equal(valueOf('x', `t = (1.0, (2.0, 3.0))\nx = t[2][1]`), 2);
});

test('nested tuple: positional decomposition binds per slot (0-based internal stays correct)', () => {
  const { bindings, errors } = infer(`
    a, inr = (10, (2.0, 3.0))
    b, c = inr
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'a'), T.INTEGER), 'a must be integer');
  assert.equal(typeOf(bindings, 'inr').kind, 'tuple');
  assert.ok(T.equal(typeOf(bindings, 'b'), T.REAL));
  assert.ok(T.equal(typeOf(bindings, 'c'), T.REAL));
  assert.equal(valueOf('a', `a, inr = (10, (2.0, 3.0))\nb, c = inr`), 10);
  assert.equal(valueOf('c', `a, inr = (10, (2.0, 3.0))\nb, c = inr`), 3);
});

test('nested tuple: a tuple may appear inside another tuple (no diagnostic)', () => {
  const { errors } = infer(`t = (1.0, (2.0, (3.0, 4.0)))`);
  assert.equal(errors.length, 0, 'tuple-in-tuple must be allowed (spec §04)');
});
