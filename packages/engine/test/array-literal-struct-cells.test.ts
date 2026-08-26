'use strict';

// =====================================================================
// array-literal-struct-cells.test.ts
// =====================================================================
//
// The array literal `[a, b]` packs its elements into one contiguous buffer.
// It did that with its own copy of the cell-packing loop, so it carried the
// defects a sibling copy had already been fixed for: a diag-stored cell
// (shape [n, n], only n entries in `data`) had its diagonal copied into the
// first n slots of an n*n block and the rest left zero, which put the
// intended diagonal in the block's first ROW.
//
// Both consumers now share `valueLib.packUniformCells`, so a third cannot
// re-introduce it. What that shared contract owns, pinned here from the
// literal's side:
//
//   - a structured cell is densified before packing;
//   - a cell still short after densify refuses instead of zero-filling;
//   - a Klein-4-tagged cell refuses (a stack of transposed vectors is not
//     a matrix — §03, §07 "Linear algebra");
//   - `im` is all-or-nothing.
//
// Every expected matrix below is the definition of `diagmat`, written out by
// hand element by element — §07 "Linear algebra": diagmat(v) is the square
// matrix whose diagonal is v and whose off-diagonal entries are zero.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const valueLib = require('../value.ts');

// Compile, assert clean, and return the fixed value of `y`.
function valueOfY(src: string): any {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'expected a clean compile, got: ' + JSON.stringify(errs));
  return buildDerivations(r.bindings).fixedValues.get('y');
}

test('an array literal over diagmat cells stacks the FULL matrices', () => {
  // diagmat([1, 2]) = [[1, 0], [0, 2]]; diagmat([3, 4]) = [[3, 0], [0, 4]].
  // Row-major over shape [2, 2, 2]: block 0 is (0,0)=1 (0,1)=0 (1,0)=0
  // (1,1)=2, block 1 is (0,0)=3 (0,1)=0 (1,0)=0 (1,1)=4.
  //
  // At BASE this was [1, 2, 0, 0, 3, 4, 0, 0] — each diagonal written across
  // the block's first row, the second row zero.
  const src = 'a = diagmat([1.0, 2.0])\nb = diagmat([3.0, 4.0])\n';
  const y = valueOfY(src + 'y = [a, b]\n');
  assert.deepEqual(y.shape, [2, 2, 2]);
  assert.deepEqual(Array.from(y.data), [
    1, 0, 0, 2,
    3, 0, 0, 4,
  ]);
  // The nested-vector tag survives: §03's "vectors of vectors are not
  // interpreted as matrices implicitly" — this is a length-2 vector of 2x2
  // matrices, not a rank-3 tensor.
  assert.equal(y.outerRank, 1);
  // Indexing the stack reads the same matrices back whole. The literal is
  // 1-indexed (spec §03).
  const first = valueOfY(src + 'z = [a, b]\ny = get(z, 1)\n');
  assert.deepEqual(first.shape, [2, 2]);
  assert.deepEqual(Array.from(first.data), [1, 0, 0, 2]);
  const second = valueOfY(src + 'z = [a, b]\ny = get(z, 2)\n');
  assert.deepEqual(second.shape, [2, 2]);
  assert.deepEqual(Array.from(second.data), [3, 0, 0, 4]);
  // Element-level: the trace is the sum of the diagonal, and the diagonal is
  // the vector that went in. A first-row diagonal gives trace 1 and 3.
  const traces = valueOfY(src + 'z = [a, b]\ny = trace.(z)\n');
  assert.deepEqual(Array.from(traces.data), [3, 7]);
});

test('a mixed literal — one structured cell, one dense — stacks both', () => {
  // `1.0 + diagmat([1.0, 2.0])` densifies (the implicit zeros become ones),
  // so this is a genuinely dense untagged [2, 2] cell next to a diag-stored
  // one: [[2, 1], [1, 3]].
  const y = valueOfY(
    'a = diagmat([1.0, 2.0])\n'
    + 'b = 1.0 + diagmat([1.0, 2.0])\n'
    + 'y = [a, b]\n');
  assert.deepEqual(y.shape, [2, 2, 2]);
  assert.deepEqual(Array.from(y.data), [
    1, 0, 0, 2,
    2, 1, 1, 3,
  ]);
});

test('a complex literal keeps its imaginary half, and its cells do too', () => {
  // The literal built `{shape, data}` and dropped everything else, so a
  // complex element was presented as its real parts — a wrong VALUE. Carded
  // in TODO-flatppl-js.md and closed by the shared packer.
  const src = 'a = complex.([1.0, 2.0], [3.0, 4.0])\n'
    + 'b = complex.([5.0, 6.0], [7.0, 8.0])\n'
    + 'zz = [a, b]\n';
  const zz = valueOfY(src + 'y = zz\n');
  assert.equal(zz.dtype, 'complex');
  assert.deepEqual(Array.from(zz.data), [1, 2, 5, 6]);
  assert.deepEqual(Array.from(zz.im), [3, 4, 7, 8]);
  // A broadcast over that operand hands each cell its own imaginary half.
  // `adjoint` is a lazy Klein-4 flip, so the canonical `im` is unchanged and
  // the 'A' tag carries the conjugation.
  const adj = valueOfY(src + 'y = adjoint.(zz)\n');
  assert.ok(Array.isArray(adj), 'a per-cell list of tagged vectors');
  assert.equal(adj[0].t, 'A');
  assert.equal(adj[1].t, 'A');
  assert.deepEqual(Array.from(adj[0].im), [3, 4]);
  assert.deepEqual(Array.from(adj[1].im), [7, 8]);
});

test('a literal over transposed vectors refuses to stack', () => {
  // §07 "Linear algebra": "The transpose of a vector is a transposed vector
  // (see arrays), not a single-row matrix." A stack of two transposed
  // 3-vectors is therefore not a [2, 3] Value, and a `t` tag on that stacked
  // Value would claim matrix transpose — a different claim. The per-cell list
  // keeps each cell's own tag.
  const y = valueOfY('a = transpose([1.0, 2.0, 3.0])\n'
    + 'b = transpose([4.0, 5.0, 6.0])\n'
    + 'y = [a, b]\n');
  assert.ok(Array.isArray(y), 'a per-cell list');
  assert.equal(y[0].t, 'T');
  assert.equal(y[1].t, 'T');
  assert.deepEqual(Array.from(y[0].data), [1, 2, 3]);
  assert.deepEqual(Array.from(y[1].data), [4, 5, 6]);
});

test('packUniformCells is the one implementation both consumers call', () => {
  // Exercised directly, because the literal reaches only some of its refusals
  // and the broadcast stacker reaches others. This is the contract itself.
  const { packUniformCells } = valueLib;
  const diag = (d: number[]) =>
    ({ shape: [3, 3], data: Float64Array.from(d), struct: 2 });
  const p = packUniformCells([diag([1, 2, 3]), diag([4, 5, 6])]);
  assert.deepEqual(p.innerShape, [3, 3]);
  assert.equal(p.innerLen, 9);
  assert.deepEqual(Array.from(p.data), [
    1, 0, 0, 0, 2, 0, 0, 0, 3,
    4, 0, 0, 0, 5, 0, 0, 0, 6,
  ]);
  // A buffer still short after densify is a form this cannot read. Copying it
  // into the full stride is exactly what put a diagonal in the first row.
  assert.equal(
    packUniformCells([{ shape: [2, 2], data: Float64Array.from([1, 2, 3]) },
                      { shape: [2, 2], data: Float64Array.from([1, 2, 3]) }]),
    null, 'a short buffer refuses');
  // Mixed real and complex are not one Value; the missing halves are not
  // guessed as zero.
  assert.equal(
    packUniformCells([
      { shape: [2], data: Float64Array.from([1, 2]) },
      { shape: [2], data: Float64Array.from([3, 4]),
        im: Float64Array.from([5, 6]), dtype: 'complex' }]),
    null, 'mixed real/complex refuses');
  // Disagreeing inner shapes refuse.
  assert.equal(
    packUniformCells([{ shape: [2], data: Float64Array.from([1, 2]) },
                      { shape: [3], data: Float64Array.from([1, 2, 3]) }]),
    null, 'disagreeing inner shapes refuse');
  // A non-Value cell refuses.
  assert.equal(packUniformCells([1, 2]), null, 'a scalar cell refuses');
});
