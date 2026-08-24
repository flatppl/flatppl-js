'use strict';

// Spec §07 boolean reductions: `lany(xs)` and `lall(xs)`.
//
//   lany — true if at least one element of xs is true
//   lall — true if every element of xs is true
//
// "`lany` is the `lor`-reduction of its input and `lall` the
// `land`-reduction. Both reduce a table column-wise" — and §04 lists both
// among the ten eligible aggregate reductions, which are order-invariant.
// The names avoid bare `all`, which collides with the axis-slicing
// keyword.
//
// Expected values are closed-form (the identity element for empty
// input, matching Julia's `any(Bool[]) == false` / `all(Bool[]) ==
// true`), not engine output.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const sampler = require('../sampler.ts');
const lowerMod = require('../lower.ts');

const ARITH_OPS = sampler._internal.ARITH_OPS;

function ev(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'unexpected errors: ' + JSON.stringify(errs));
  return buildDerivations(r.bindings).fixedValues;
}

function evalRHS(src: string, binding: string, env: any) {
  const ctx = processSource(src);
  const b = ctx.bindings.get(binding);
  assert.ok(b, `binding ${binding} not found`);
  return sampler.evaluateExpr(lowerMod.lowerExpr(b.node.value), env);
}

// =====================================================================
// Scalar results over a boolean vector
// =====================================================================

test('lany / lall over a mixed boolean vector', () => {
  const fv = ev(`
mask = [true, false, true]
a = lany(mask)
l = lall(mask)
`);
  assert.equal(fv.get('a'), true);
  assert.equal(fv.get('l'), false);
});

test('lany / lall over an all-true and an all-false vector', () => {
  const fv = ev(`
t = [true, true]
f = [false, false]
at = lany(t)
lt = lall(t)
af = lany(f)
lf = lall(f)
`);
  assert.equal(fv.get('at'), true);
  assert.equal(fv.get('lt'), true);
  assert.equal(fv.get('af'), false);
  assert.equal(fv.get('lf'), false);
});

test('lany / lall return a JS boolean, like lor / land', () => {
  // Not 1 / 0: a scalar boolean is a JS boolean throughout ARITH_OPS,
  // and `equal` compares with ===.
  const v = { shape: [2], data: Float64Array.from([1, 0]) };
  assert.strictEqual(ARITH_OPS.lany(v), true);
  assert.strictEqual(ARITH_OPS.lall(v), false);
});

test('lany / lall read booleans stored as 0 / 1 inside an array', () => {
  // `vector` packs boolean elements into a Float64Array as 1 / 0, so
  // the reductions must read truthiness, not identity against `true`.
  const fv = ev(`
mask = [false, false, true]
a = lany(mask)
`);
  assert.equal(fv.get('a'), true);
});

test('lany / lall over a boolean rank-2 array reduce every element', () => {
  const M = { shape: [2, 2], data: Float64Array.from([1, 0, 0, 0]) };
  assert.equal(ARITH_OPS.lany(M), true);
  assert.equal(ARITH_OPS.lall(M), false);
});

// =====================================================================
// Empty input — the identity of each reduction. RULED, per
// flatppl-dev/empty-arrays-ruling.md's §07 "Logic and conditionals" edit
// row (these are the forced lor-/land-reduction identities); pinned here
// as a regression, not because a change is expected. TODO-flatppl-js.md's
// entry now reads RULED and CONFORMED.
// =====================================================================

test('lany([]) = false, lall([]) = true (each reduction identity)', () => {
  const empty = { shape: [0], data: new Float64Array(0) };
  assert.strictEqual(ARITH_OPS.lany(empty), false);
  assert.strictEqual(ARITH_OPS.lall(empty), true);
});

// =====================================================================
// Table form — §07's "Table reductions" paragraph names lany and lall
// =====================================================================

test('lany / lall over a table reduce column-wise into a record', () => {
  const fv = ev(`
t = table(hit = [true, false, true], veto = [false, false, false])
a = lany(t)
l = lall(t)
`);
  const a: any = fv.get('a');
  const l: any = fv.get('l');
  assert.equal(a.hit, true);
  assert.equal(a.veto, false);
  assert.equal(l.hit, false);
  assert.equal(l.veto, false);
});

test('lany / lall of a table keep each column boolean in the type', () => {
  const r = processSource(`
t = table(hit = [true, false, true])
a = lany(t)
`);
  const t: any = r.bindings.get('a').inferredType;
  assert.equal(t.kind, 'record');
  assert.equal(t.fields.hit.prim, 'boolean');
});

// =====================================================================
// Aggregate eligibility — §04 §sec:aggregate lists lany and lall
// =====================================================================

test('aggregate(lany, [.i], …) reduces each row', () => {
  const M = { shape: [2, 2], data: Float64Array.from([1, 0, 0, 0]) };
  const out = evalRHS('r = aggregate(lany, [.i], B[.i, .j])', 'r', { B: M });
  // Row 1 holds a true, row 2 does not. Booleans inside the result
  // array are stored 1 / 0.
  assert.deepEqual(Array.from(out.data || out), [1, 0]);
});

test('aggregate(lall, [.j], …) reduces each column', () => {
  const M = { shape: [2, 2], data: Float64Array.from([1, 0, 0, 0]) };
  const out = evalRHS('r = aggregate(lall, [.j], B[.i, .j])', 'r', { B: M });
  assert.deepEqual(Array.from(out.data || out), [0, 0]);
});

test('aggregate(lany / lall, [], …) reduces to a scalar', () => {
  const M = { shape: [2, 2], data: Float64Array.from([1, 0, 0, 0]) };
  assert.equal(evalRHS('r = aggregate(lany, [], B[.i, .j])', 'r', { B: M }), 1);
  assert.equal(evalRHS('r = aggregate(lall, [], B[.i, .j])', 'r', { B: M }), 0);
});

test('aggregate(lany, …) types as boolean whatever the body was', () => {
  const r = processSource(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
s = aggregate(lany, [], A[.i, .j])
`);
  const t: any = r.bindings.get('s').inferredType;
  assert.equal(t.kind, 'scalar');
  assert.equal(t.prim, 'boolean');
});
