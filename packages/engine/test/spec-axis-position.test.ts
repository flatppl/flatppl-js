'use strict';

// =====================================================================
// spec-axis-position.test.ts — pin spec §05's bare-axis POSITION rule
// =====================================================================
//
// Spec §05 "Axis names and aggregation": "an axis name is legal only as
// an entry in `aggregate`'s `output_axes` axis list, as an index inside
// `[...]` within the body, or as a binder on the left-hand side of `:=`.
// Used anywhere else it is a static error."
//
// This is the sibling rule to the axis-LIST position rule pinned in
// spec-axislist-position.test.ts. That file covers where `[...]` itself
// may sit; this one covers where a bare `.name` may sit once inside a
// legal `aggregate(...)` / `metricsum(...)` call — in particular the "as
// an index inside `[...]` within the body" half, which is not just "any
// argument of the call": the indexed OBJECT of a body's `[...]` is a
// value position even though it sits inside the body, and the
// `f_reduction` / metric slot (argument 0) is a value position too.
//
// The matrix below mirrors flatppl-rust's `crates/infer/tests/
// axis_position.rs` (commit 63863d9) spelling for spelling, so both
// engines are pinned to the same spec text independently.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

const SETUP = 'g = eye(2)\nA = rowstack([[1.0, 0.0], [0.0, 1.0]])\nv = [1.0, 2.0]\n';

function errors(src: string): any[] {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

const AXIS = /may only appear inside aggregate\(\.\.\.\) or metricsum\(\.\.\.\)/;

function refused(src: string) {
  const errs = errors(src);
  assert.ok(errs.some((d: any) => AXIS.test(d.message)),
    `expected the axis-position refusal for ${JSON.stringify(src)}; got: `
    + `${errs.map((d: any) => d.message).join('; ')}`);
}

function accepted(src: string) {
  const errs = errors(src);
  assert.equal(errs.length, 0,
    `expected no diagnostic for ${JSON.stringify(src)}; got: `
    + `${errs.map((d: any) => d.message).join('; ')}`);
}

// ---- bare axis names out of position -------------------------------

test('§05: bare axis in arithmetic is refused', () => {
  refused('x = .i + 1\n');
});

test('§05: bare axis on a binding right-hand side is refused', () => {
  refused('x = .i\n');
});

test('§05: bare axis as a call argument is refused', () => {
  refused('x = sum(.i)\n');
});

test('§05: bare axis as a record field is refused', () => {
  refused('x = record(a = .i)\n');
});

test('§05: bare axis under a tilde-binding is refused', () => {
  refused('y ~ .i\n');
});

test('§05: a `[...]` index outside any aggregation is refused', () => {
  // An index is a legal axis position only "within the body" of an
  // aggregation — §05 scopes an axis name "to the enclosing aggregation",
  // and this index has no enclosing aggregation to be scoped to.
  refused(`${SETUP}x = A[.i]\n`);
});

test('§05: a `get(...)` index outside any aggregation is refused', () => {
  // Same rule through the `get` spelling §04 gives for the same construct.
  refused(`${SETUP}x = get(A, .i)\n`);
});

test('§05: a bare axis as the indexed OBJECT is refused even inside a body', () => {
  // The indexed object is not an index position, so an axis there is out
  // of position even inside a legal aggregation body.
  refused(`${SETUP}x = aggregate(sum, [.i], .i[1])\n`);
});

// ---- legal positions -------------------------------------------------

test('§05: aggregate output_axes and body indices are accepted', () => {
  accepted(`${SETUP}x = aggregate(sum, [.i], A[.i, .j])\n`);
});

test('§05: the aggregate binding binder is accepted', () => {
  accepted(`${SETUP}C[.i] := A[.i, .j]\n`);
});

test('§05: the metricsum binding binder is accepted', () => {
  accepted(`${SETUP}g: T[.mu^] := A[.mu^, .nu_] * v[.nu^]\n`);
});

test('§05: the get(...) index spelling is accepted', () => {
  // §04 spells both index forms out: "array indexing may contain axis
  // names, like `A[.i, 1, .j]` or `get(A, .i, 1, .j)`".
  accepted(`${SETUP}x = aggregate(sum, [.i], get(A, .i, .j))\n`);
});

test('§05: a nested aggregation is accepted', () => {
  accepted(`${SETUP}x = aggregate(sum, [.i], aggregate(sum, [], A[.i, .j]))\n`);
});

test('§05: a get(...) index nested under a plain call inside a body is accepted', () => {
  // The body flag, once set, persists through an intervening ordinary
  // call — it is not reset to "not in a body" by `identity(...)`.
  accepted(`${SETUP}x = aggregate(sum, [.i], identity(get(A, .i, .j)))\n`);
});

test('§05: a get(...) OBJECT slot is still refused inside a body', () => {
  // Only get's index arguments (position ≥ 1) are index positions; the
  // object (position 0) is a value position regardless of body scope.
  refused(`${SETUP}x = aggregate(sum, [.i], get(.i, 1))\n`);
});
