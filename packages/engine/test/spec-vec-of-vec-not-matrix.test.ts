'use strict';

// =====================================================================
// spec-vec-of-vec-not-matrix.test.ts — pin spec §03 enforcement
// =====================================================================
//
// Spec §03 (Arrays):
//   "One-dimensional arrays of scalars act as vectors for linear
//    algebra ... Vectors of vectors are not interpreted as matrices
//    implicitly, but can be turned into matrices explicitly using
//    rowstack or colstack."
//
// FlatPPL is row/column-major-agnostic. By writing a bare nested
// literal `[[a, b], [c, d]]` the user has NOT committed to a layout;
// only `rowstack`/`colstack` commit one. The engine uses flat row-
// major Float64Array storage as its INTERNAL convention (the
// ArrayOfSimilarArrays-style "vec-of-equal-sized-vecs backed by flat
// n-d array" pattern, cf. Julia ArraysOfArrays.jl), but the runtime
// `outerRank` tag preserves the SEMANTIC distinction and every
// matrix-input op refuses Values that carry it.
//
// This file pins the invariants that enforce this distinction so the
// regression can't silently return:
//
//   1. `asValue([[…]])` returns a Value with `outerRank` set; storage
//      stays a contiguous Float64Array (storage optimisation
//      preserved).
//   2. `isNestedVectorValue` is true on the result; `false` on a
//      rowstack-derived matrix.
//   3. `densify(v)` preserves `outerRank` across the structured-flag
//      path.
//   4. Per matrix-input op, source `A = [[…]]; B = op(A)` produces a
//      typeinfer diagnostic mentioning "vector-of-vectors" /
//      "rowstack". The `rowstack(...)`-wrapped companion typechecks
//      clean.
//   5. Metricsum-specific: bare-nested metric or variance-marked
//      container is rejected with a §03-citing diagnostic.
//   6. Runtime `valueLib.requireMatrix(v)` throws on a synthetic
//      Value carrying outerRank — the §03 message points at
//      `rowstack(...)`.
//   7. Viewer defense in depth: a nested-vector fixed Value does NOT
//      get `intrinsicShape` set on its EmpiricalMeasure (so the
//      heatmap renderer can't fire on it).
//   8. Storage-optimisation invariant: `asValue([[…]]).data` is a
//      single contiguous Float64Array — the flat backing is preserved.

const test = require('node:test');
const assert = require('node:assert');

const valueLib = require('../value.ts');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const ops = require('../ops.ts');
const materialiserShared = require('../materialiser-shared.ts');

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function errors(src: string): any[] {
  const ctx = processSource(src);
  return ctx.diagnostics.filter((d: any) => d.severity === 'error');
}

// =====================================================================
// 1. asValue / outerRank / storage invariants
// =====================================================================

test('asValue: nested literal [[…]] sets outerRank=1, preserves flat F64 storage', () => {
  const v = valueLib.asValue([[1.0, 2.0], [3.0, 4.0]]);
  assert.deepEqual(v.shape, [2, 2], 'shape captured');
  assert.equal(v.outerRank, 1, 'outerRank set to 1 (one outer loop axis)');
  assert.ok(v.data instanceof Float64Array, 'storage IS a Float64Array');
  assert.equal(v.data.length, 4, 'storage is contiguous (4 entries, not 2)');
  assert.equal(v.data[0], 1.0);
  assert.equal(v.data[3], 4.0);
});

test('asValue: triply-nested literal [[[…]]] sets outerRank=2', () => {
  const v = valueLib.asValue([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
  assert.deepEqual(v.shape, [2, 2, 2]);
  assert.equal(v.outerRank, 2, 'three JS-Array nesting levels → outerRank=2');
  assert.ok(v.data instanceof Float64Array);
});

test('asValue: engine-internal [Float64Array, Float64Array] is treated as matrix (no outerRank)', () => {
  // Broadcast-reduce default emits results in this shape; per spec,
  // aggregate output is a flat array of the declared output_axes shape,
  // NOT a vec-of-vec.
  const inner1 = new Float64Array([1, 2]);
  const inner2 = new Float64Array([3, 4]);
  const v = valueLib.asValue([inner1, inner2]);
  assert.deepEqual(v.shape, [2, 2]);
  assert.equal(v.outerRank, undefined, 'jsNestingDepth=1 → no outerRank tag');
});

test('asValue: flat 1D literal [1,2,3] has no outerRank', () => {
  const v = valueLib.asValue([1, 2, 3]);
  assert.deepEqual(v.shape, [3]);
  assert.equal(v.outerRank, undefined, 'flat vec → no outerRank tag');
});

test('isNestedVectorValue: true on asValue([[…]]), false on rowstack-derived matrix', () => {
  const nested = valueLib.asValue([[1, 2], [3, 4]]);
  assert.ok(valueLib.isNestedVectorValue(nested), 'nested literal IS a nested-vector value');

  const matrix = valueLib.promoteNestedToMatrix(nested);
  assert.ok(!valueLib.isNestedVectorValue(matrix), 'promoted matrix is NOT a nested-vector value');
  assert.equal(matrix.outerRank, undefined, 'matrix has no outerRank tag');
  assert.deepEqual(matrix.shape, [2, 2]);
  // Storage shared (no copy).
  assert.strictEqual(matrix.data, nested.data);
});

test('densify: preserves outerRank across structured-flag round-trip', () => {
  // Build a Value with outerRank=1 and a (no-op) struct flag, then
  // densify and verify the tag survived.
  const v: any = {
    shape: [2, 2],
    data: new Float64Array([1, 2, 3, 4]),
    outerRank: 1,
    struct: valueLib.ST_DENSE,
  };
  const d = valueLib.densify(v);
  assert.equal(d.outerRank, 1, 'densify preserves the nested-vector tag');
});

// =====================================================================
// 2. requireMatrix runtime guard
// =====================================================================

test('requireMatrix: throws on a Value with outerRank<shape.length', () => {
  const vov = valueLib.asValue([[1, 2], [3, 4]]);
  let threw = false;
  try { valueLib.requireMatrix(vov, 'TEST_OP'); }
  catch (e: any) {
    threw = true;
    assert.ok(/TEST_OP/.test(e.message), 'mentions op name');
    assert.ok(/vector-of-vectors/.test(e.message), 'mentions vec-of-vec');
    assert.ok(/§03/.test(e.message), 'cites spec §03');
    assert.ok(/rowstack/.test(e.message), 'points at rowstack');
  }
  assert.ok(threw, 'requireMatrix throws on nested-vector input');
});

test('requireMatrix: passes through a matrix-typed Value unchanged', () => {
  const m = valueLib.matrix([1, 0, 0, 1], 2, 2);
  const r = valueLib.requireMatrix(m, 'TEST_OP');
  assert.strictEqual(r, m, 'returns input identity');
});

// =====================================================================
// 3. Per matrix-input op: bare-[[…]] source produces a clear diagnostic
// =====================================================================

const MATRIX_OPS = ['inv', 'det', 'logabsdet', 'lower_cholesky', 'row_gram', 'col_gram'];

for (const op of MATRIX_OPS) {
  test(`${op}: bare nested-literal arg yields typeinfer diagnostic with rowstack hint`, () => {
    const ds = errors(`
A = [[1.0, 0.0], [0.0, 1.0]]
B = ${op}(A)
`);
    assert.ok(ds.some((d: any) => /vector-of-vectors/.test(d.message)),
      `${op}: expected vec-of-vec diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
    assert.ok(ds.some((d: any) => /rowstack/.test(d.message)),
      `${op}: expected rowstack hint`);
  });

  test(`${op}: rowstack-wrapped arg typechecks clean`, () => {
    const ds = errors(`
A = rowstack([[1.0, 0.0], [0.0, 1.0]])
B = ${op}(A)
`);
    assert.equal(ds.length, 0,
      `${op}: rowstack-wrapped should typecheck; got: ${ds.map((d: any) => d.message).join('; ')}`);
  });
}

// =====================================================================
// 4. Metricsum-specific spec §sec:metricsum "Expression restrictions"
// =====================================================================

test('metricsum: bare nested-literal metric is rejected with §03 hint', () => {
  const ds = errors(`
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
norm = metricsum(g, [], p[.mu^] * p[.mu_])
`);
  assert.ok(ds.some((d: any) => /vector-of-vectors/.test(d.message) && /§03/.test(d.message)),
    'expected metricsum metric vec-of-vec diagnostic');
});

test('metricsum: rowstack-wrapped metric typechecks clean', () => {
  const ds = errors(`
g = rowstack([[1.0, 0.0], [0.0, -1.0]])
p = [3.0, 2.0]
norm = metricsum(g, [], p[.mu^] * p[.mu_])
`);
  assert.equal(ds.length, 0,
    `expected clean parse; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: bare nested-literal variance-marked-axis container is rejected', () => {
  const ds = errors(`
g = rowstack([[1.0, 0.0], [0.0, -1.0]])
A = [[1.0, 2.0], [3.0, 4.0]]
p = [3.0, 2.0]
r = metricsum(g, [.mu^], A[.mu^, .nu_] * p[.nu^])
`);
  assert.ok(ds.some((d: any) => /vector-of-vectors/.test(d.message)),
    'expected vec-of-vec diagnostic for body-indexed container');
});

test('_ms_check_symmetric op: refuses a nested-vector Value at runtime', () => {
  // Synthetic Value with outerRank=1 mimicking what asValue produces
  // from a bare nested literal.
  const vov = valueLib.asValue([[1.0, 0.5], [0.5, -1.0]]);
  let threw = false;
  try { ops.dispatch('_ms_check_symmetric', [vov]); }
  catch (e: any) {
    threw = true;
    assert.ok(/vector-of-vectors/.test(e.message) || /metricsum/.test(e.message),
      `expected metricsum-attributed message; got: ${e.message}`);
  }
  assert.ok(threw, 'runtime _ms_check_symmetric refuses vec-of-vec');
});

// =====================================================================
// 5. Aggregate over vec-of-vec stays legal (spec §04 sec:aggregate)
// =====================================================================
//
// Spec §04: `A[.i, .j] ≡ get(A, .i, .j) ≡ A[.i][.j]`. The aggregate
// runtime indexes by axis-name and is well-defined on both flat
// matrices AND vectors-of-vectors. Pinning this so Steps 1-4 don't
// over-reach and break the spec-permitted nested-array body access.

test('aggregate: bare nested-literal body indexing typechecks AND evaluates', () => {
  const ds = errors(`
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
T = aggregate(sum, [], A[.i, .j])
`);
  assert.equal(ds.length, 0,
    `expected clean parse for spec-legal aggregate-over-nested; got: ${ds.map((d: any) => d.message).join('; ')}`);
  const ctx = processSource(`
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B = [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
T = aggregate(sum, [], A[.i, .j])
`);
  const built = orchestrator.buildDerivations(ctx.bindings);
  const T = built.fixedValues.get('T');
  // T = sum over all entries = 1+2+3+4+5+6 = 21
  assert.equal(T, 21, 'sum-over-all-entries works on vec-of-vec body');
  const C = built.fixedValues.get('C');
  // C = A @ B as if A, B were row-major matrices (engine internal
  // convention) — same as rowstack(A) @ rowstack(B).
  assert.ok(C && C.shape && C.shape[0] === 2 && C.shape[1] === 2);
  assert.equal(C.data[0], 58, 'C[1][1] = 58');
  assert.equal(C.data[3], 154, 'C[2][2] = 154');
});

// =====================================================================
// 6. Viewer defense in depth: nested-vector → no intrinsicShape
// =====================================================================

test('materialiser fixedValueToMeasure: nested-vector Value does NOT set intrinsicShape', () => {
  const vov = valueLib.asValue([[1.0, 2.0], [3.0, 4.0]]);
  const m: any = materialiserShared.fixedValueToMeasure(vov, 4);
  assert.equal(m.intrinsicShape, undefined,
    'vec-of-vec must not trigger matrix-mode heatmap rendering');
});

test('materialiser fixedValueToMeasure: matrix Value DOES set intrinsicShape', () => {
  const mat = valueLib.matrix([1, 0, 0, 1], 2, 2);
  const m: any = materialiserShared.fixedValueToMeasure(mat, 4);
  assert.deepEqual(m.intrinsicShape, [2, 2],
    'true matrix triggers matrix-mode rendering');
});

// =====================================================================
// 7. promoteNestedToMatrix round-trip (storage shared, semantics flipped)
// =====================================================================

test('promoteNestedToMatrix: storage shared, semantics flipped', () => {
  const vov = valueLib.asValue([[1, 2], [3, 4]]);
  const m = valueLib.promoteNestedToMatrix(vov);
  assert.strictEqual(m.data, vov.data,
    'storage is shared (no copy — the ArrayOfSimilarArrays optimisation)');
  assert.equal(m.outerRank, undefined, 'matrix carries no outerRank');
  assert.ok(!valueLib.isNestedVectorValue(m), 'promoted form is matrix-typed');
});

test('promoteNestedToMatrix: no-op on already-matrix Value', () => {
  const mat = valueLib.matrix([1, 0, 0, 1], 2, 2);
  const r = valueLib.promoteNestedToMatrix(mat);
  assert.strictEqual(r, mat, 'no outerRank → returned as-is');
});
