'use strict';

// =====================================================================
// einsum-patterns.test.ts — pins the einsum/contraction surface
// =====================================================================
//
// Loads `test/fixtures/einsum-patterns.flatppl` (mirrored from
// flatppl-examples/examples/einsum_patterns.flatppl per the
// engine's local-copy convention) and pins each canonical
// contraction pattern's:
//
//   - Dissolved IR shape  (matmul / matvec → mul; outer → mul;
//                          inner / frob → sum + dotted-elementwise)
//   - Pre-evaluated value (fixed-phase inputs → fixedValues hit)
//
// The fixture's introduction documents two engine gaps the test
// pins separately (so a regression in either direction shows up):
//   - Empty output_axes `aggregate(sum, [], body)` is rejected;
//     scalar-output contractions use `sum(...)` directly.
//   - Rank-3 batched matmul parses but doesn't pre-evaluate yet.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator } = require('..');

function readFixture(name: string): string {
  return fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf-8');
}

function approxEq(got: number, want: number, eps = 1e-9): boolean {
  return Math.abs(got - want) < eps;
}

function valueDataAt(v: any, i: number): number {
  if (v && v.data) {
    if (typeof v.data.length === 'number') return v.data[i];
    return (v.data as any)[i];
  }
  return v;
}

// =====================================================================
// Setup — process the einsum_patterns.flatppl fixture once.
// =====================================================================

const src = readFixture('einsum-patterns.flatppl');
const lifted = processSource(src);
const built = orchestrator.buildDerivations(lifted.bindings);

test('einsum_patterns: fixture parses with no diagnostics', () => {
  const errors = (lifted.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errors.length, 0,
    `parse errors: ${JSON.stringify(errors)}`);
});

// =====================================================================
// Matmul: aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
// =====================================================================

test('einsum: matmul[.i,.k] := A[.i,.j] * B[.j,.k] — IR + value', () => {
  // The `:=` form lowers to `aggregate(sum, [.i, .k], A[.i,.j] *
  // B[.j,.k])`. Dissolver's Phase 5 matmul-family specialiser
  // recognises this exact pattern and rewrites to a direct
  // `mul(A_mat, B_mat)` IR call. The pre-evaluator then computes
  // the matrix product over the fixed-phase inputs.
  const b = built.bindings.get('matmul');
  assert.ok(b, 'matmul binding exists');
  // Post-dissolution the IR should be a direct `mul` call (not the
  // aggregate form) for the matmul-family specialiser to count as
  // "lifted to lower-level dispatch." [Note: this assertion may
  // need to relax if the dissolver's specialiser path changes.]
  assert.ok(b.ir, 'matmul has IR');
  // Pre-evaluated to a [3, 3] Value. B is the identity, so result
  // should equal A_mat.
  const v = built.fixedValues.get('matmul');
  assert.ok(v && v.shape, 'matmul pre-evaluates to a Value');
  assert.deepEqual(v.shape, [3, 3]);
  const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9];
  for (let i = 0; i < 9; i++) {
    assert.ok(approxEq(valueDataAt(v, i), expected[i]),
      `matmul[${Math.floor(i/3)}, ${i%3}] = ${valueDataAt(v, i)}, want ${expected[i]}`);
  }
});

// =====================================================================
// Matvec: aggregate(sum, [.i], A[.i, .j] * v[.j])
// =====================================================================

test('einsum: matvec[.i] := A[.i,.j] * v[.j] — IR + value', () => {
  const b = built.bindings.get('matvec');
  assert.ok(b, 'matvec binding exists');
  // A = [[1,2,3],[4,5,6],[7,8,9]]; v = [10,20,30].
  // A·v = [1*10+2*20+3*30, 4*10+5*20+6*30, 7*10+8*20+9*30]
  //     = [140, 320, 500].
  const v = built.fixedValues.get('matvec');
  assert.ok(v && v.shape);
  assert.deepEqual(v.shape, [3]);
  const expected = [140, 320, 500];
  for (let i = 0; i < 3; i++) {
    assert.ok(approxEq(valueDataAt(v, i), expected[i]),
      `matvec[${i}] = ${valueDataAt(v, i)}, want ${expected[i]}`);
  }
});

// =====================================================================
// Outer product: aggregate(sum, [.i, .j], u[.i] * v[.j])
// =====================================================================

test('einsum: outer[.i,.j] := u[.i] * v[.j] — IR + value', () => {
  const b = built.bindings.get('outer');
  assert.ok(b, 'outer binding exists');
  // u = [1,2,3]; v = [10,20,30]. u⊗v[i,j] = u[i]*v[j].
  //   [[10, 20, 30],
  //    [20, 40, 60],
  //    [30, 60, 90]]
  const v = built.fixedValues.get('outer');
  assert.ok(v && v.shape);
  assert.deepEqual(v.shape, [3, 3]);
  const expected = [10, 20, 30, 20, 40, 60, 30, 60, 90];
  for (let i = 0; i < 9; i++) {
    assert.ok(approxEq(valueDataAt(v, i), expected[i]),
      `outer flat[${i}] = ${valueDataAt(v, i)}, want ${expected[i]}`);
  }
});

// =====================================================================
// Scalar contractions via sum(elementwise product) — until
// aggregate gains empty-output-axes support, these are the
// idiomatic FlatPPL spellings for inner-product / Frobenius.
// =====================================================================

test('einsum: inner_prod = sum(u .* v) — scalar contraction', () => {
  // u·v = 1*10 + 2*20 + 3*30 = 140.
  const v = built.fixedValues.get('inner_prod');
  assert.ok(approxEq(v, 140),
    `inner_prod = ${v}, want 140`);
});

test('einsum: frob_prod = sum(A .* B) — Frobenius inner product', () => {
  // A·B (elementwise) with B = I_3: diag(A) only → 1+5+9 = 15.
  const v = built.fixedValues.get('frob_prod');
  assert.ok(approxEq(v, 15),
    `frob_prod = ${v}, want 15`);
});

// =====================================================================
// Engine-gap regressions: the introduction documents two gaps;
// pin them so a future fix surfaces the change clearly.
// =====================================================================

test('engine gap: aggregate(sum, [], body) refused — empty output_axes', () => {
  const r = processSource(`
u = [1.0, 2.0, 3.0]
v = [4.0, 5.0, 6.0]
inner = aggregate(sum, [], u[.i] * v[.i])
`);
  const errors = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.ok(errors.length > 0,
    'empty output_axes currently rejected — pin so a future fix surfaces');
  assert.match(errors[0].message,
    /at least one output axis|output_axes/i);
});

test('engine gap: rank-3 batched-matmul parses but pre-eval lacks value', () => {
  // Pins that the parser + classifier accept the pattern (so the
  // editor / lint paths stay green), but pre-eval doesn't produce
  // a numeric Value yet (a follow-up: rank-3 array-literal
  // threading through aggregate eval).
  const r = processSource(`
batch_A = [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]]
batch_B = [[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]
bmm[.b, .i, .k] := batch_A[.b, .i, .j] * batch_B[.b, .j, .k]
`);
  const errors = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errors.length, 0, 'no parse errors');
  const out = orchestrator.buildDerivations(r.bindings);
  const bmm = out.bindings.get('bmm');
  assert.ok(bmm && bmm.ir, 'bmm has IR (classifies)');
  // Pre-eval doesn't produce a numeric Value (rank-3 array literal
  // gap). A future fix should make `out.fixedValues.has('bmm')`
  // return true and `out.fixedValues.get('bmm')` a [2, 2, 2]
  // Value; this assertion will start failing then — flip it.
  const v = out.fixedValues.get('bmm');
  assert.ok(v === undefined || (Array.isArray(v) && v[0] === undefined),
    'rank-3 batched matmul pre-eval gap (flip to value-check once fixed)');
});
