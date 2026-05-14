'use strict';

// Tests for engine/value.js — the shape-tagged Value type and its
// helpers. Phase 0 of the shape-explicit refactor (see
// TODO-flatppl-js.md). This module is a pure addition: nothing in
// the engine consumes it yet, so these tests target the public surface
// directly without going through processSource / materialiser.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const value = require('..').value;
const {
  asValue, asScalar, asBatch,
  scalar, batchedScalar, vector, batchedVector, matrix, batchedMatrix,
  withShape,
  getShape, getData, getDType, isBatched, numel,
} = value;

// =====================================================================
// numel: element count from shape
// =====================================================================

test('numel: scalar shape [] has one element', () => {
  assert.equal(numel([]), 1);
});

test('numel: vector / batched / matrix / batched-matrix counts', () => {
  assert.equal(numel([5]), 5);
  assert.equal(numel([3, 4]), 12);
  assert.equal(numel([2, 3, 4]), 24);
  assert.equal(numel([1024, 7]), 1024 * 7);
});

test('numel: empty inner dim → 0', () => {
  assert.equal(numel([0]), 0);
  assert.equal(numel([5, 0]), 0);
});

// =====================================================================
// scalar / batchedScalar / vector constructors
// =====================================================================

test('scalar: shape=[], Float64Array(1) storage', () => {
  const s = scalar(3.14);
  assert.deepEqual(s.shape, []);
  assert.ok(s.data instanceof Float64Array);
  assert.equal(s.data.length, 1);
  assert.equal(s.data[0], 3.14);
});

test('scalar: boolean / int coerce to number', () => {
  assert.equal(scalar(true).data[0], 1);
  assert.equal(scalar(false).data[0], 0);
  assert.equal(scalar(42).data[0], 42);
});

test('batchedScalar: shape=[N], borrows Float64Array if given', () => {
  const arr = new Float64Array([1, 2, 3, 4]);
  const b = batchedScalar(arr);
  assert.deepEqual(b.shape, [4]);
  assert.equal(b.data, arr, 'should borrow the Float64Array, not copy');
});

test('batchedScalar: from plain JS array copies into Float64Array', () => {
  const b = batchedScalar([0.5, 1.5, 2.5]);
  assert.deepEqual(b.shape, [3]);
  assert.ok(b.data instanceof Float64Array);
  assert.deepEqual(Array.from(b.data), [0.5, 1.5, 2.5]);
});

test('vector: atom-indep length-k vector', () => {
  const v = vector([1, 2, 3]);
  assert.deepEqual(v.shape, [3]);
  assert.ok(v.data instanceof Float64Array);
  assert.deepEqual(Array.from(v.data), [1, 2, 3]);
});

// =====================================================================
// batchedVector / matrix / batchedMatrix
// =====================================================================

test('batchedVector: shape=[N, k] atom-major', () => {
  // 3 atoms × 2 components: atoms [a0, a1], [b0, b1], [c0, c1]
  const flat = new Float64Array([10, 11, 20, 21, 30, 31]);
  const v = batchedVector(flat, 2);
  assert.deepEqual(v.shape, [3, 2]);
  assert.equal(v.data, flat);
  // Atom 1 (zero-indexed in storage; FlatPPL is 1-indexed at the
  // surface, but here we're testing internal layout): components
  // start at i*k = 2.
  assert.equal(v.data[1 * 2 + 0], 20);
  assert.equal(v.data[1 * 2 + 1], 21);
});

test('batchedVector: rejects non-divisible length', () => {
  assert.throws(() => batchedVector(new Float64Array(7), 2), /not divisible by k=2/);
});

test('matrix: shape=[m, n] row-major', () => {
  // 2×3 matrix: [[1,2,3],[4,5,6]]
  const M = matrix([1, 2, 3, 4, 5, 6], 2, 3);
  assert.deepEqual(M.shape, [2, 3]);
  // Row 1, col 2 → index 1*3 + 2 = 5
  assert.equal(M.data[1 * 3 + 2], 6);
});

test('matrix: rejects size mismatch', () => {
  assert.throws(() => matrix([1, 2, 3], 2, 2), /data length 3 != m\*n = 4/);
});

test('batchedMatrix: shape=[N, m, n]', () => {
  // 2 atoms × 2×2 matrices
  const flat = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const bm = batchedMatrix(flat, 2, 2);
  assert.deepEqual(bm.shape, [2, 2, 2]);
  // Atom 0 row 1 col 0 → index 0*4 + 1*2 + 0 = 2 → 3
  assert.equal(bm.data[0 * 4 + 1 * 2 + 0], 3);
  // Atom 1 row 0 col 1 → index 1*4 + 0*2 + 1 = 5 → 6
  assert.equal(bm.data[1 * 4 + 0 * 2 + 1], 6);
});

// =====================================================================
// withShape: generic / last-resort
// =====================================================================

test('withShape: validates numel match and clones shape array', () => {
  const flat = new Float64Array(12);
  const v = withShape(flat, [2, 3, 2]);
  assert.deepEqual(v.shape, [2, 3, 2]);
  assert.equal(v.data, flat);
  // Mutating the passed-in shape array shouldn't affect the Value.
  const shape = [3, 4];
  const v2 = withShape(new Float64Array(12), shape);
  shape.push(99);
  assert.deepEqual(v2.shape, [3, 4]);
});

test('withShape: rejects numel mismatch', () => {
  assert.throws(() => withShape(new Float64Array(5), [2, 3]), /!= numel/);
});

// =====================================================================
// asValue: coercion from JS primitives, typed arrays, nested arrays
// =====================================================================

test('asValue: JS number → shape=[]', () => {
  const v = asValue(3.14);
  assert.deepEqual(v.shape, []);
  assert.equal(v.data[0], 3.14);
});

test('asValue: JS boolean → shape=[] with 0/1', () => {
  assert.equal(asValue(true).data[0], 1);
  assert.equal(asValue(false).data[0], 0);
});

test('asValue: Float64Array → shape=[length], borrows storage', () => {
  const arr = new Float64Array([1, 2, 3]);
  const v = asValue(arr);
  assert.deepEqual(v.shape, [3]);
  assert.equal(v.data, arr);
});

test('asValue: other typed array → shape=[length], copies', () => {
  const i32 = new Int32Array([1, 2, 3]);
  const v = asValue(i32);
  assert.deepEqual(v.shape, [3]);
  assert.ok(v.data instanceof Float64Array);
  assert.deepEqual(Array.from(v.data), [1, 2, 3]);
});

test('asValue: nested JS Array → row-major', () => {
  const v = asValue([[1, 2, 3], [4, 5, 6]]);
  assert.deepEqual(v.shape, [2, 3]);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4, 5, 6]);
});

test('asValue: 3-D nested array', () => {
  const v = asValue([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
  assert.deepEqual(v.shape, [2, 2, 2]);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('asValue: ragged nested array throws', () => {
  assert.throws(() => asValue([[1, 2], [3]]), /ragged/);
  assert.throws(() => asValue([1, [2, 3]]), /mixes scalars and arrays/);
});

test('asValue: Value passes through unchanged', () => {
  const v = scalar(2.5);
  assert.equal(asValue(v), v);
});

test('asValue: null/undefined throw', () => {
  assert.throws(() => asValue(null), /null\/undefined/);
  assert.throws(() => asValue(undefined), /null\/undefined/);
});

// =====================================================================
// asScalar / asBatch: strict extractors
// =====================================================================

test('asScalar: returns JS number from shape=[]', () => {
  assert.equal(asScalar(scalar(7)), 7);
});

test('asScalar: throws on any non-empty shape (incl. [1])', () => {
  assert.throws(() => asScalar(batchedScalar([3])), /shape is \[1\]/);
  assert.throws(() => asScalar(vector([1, 2, 3])), /shape is \[3\]/);
  assert.throws(() => asScalar({}), /not a Value/);
});

test('asBatch: returns the Float64Array of shape=[N]', () => {
  const arr = new Float64Array([1, 2, 3, 4]);
  const b = batchedScalar(arr);
  assert.equal(asBatch(b, 4), arr);
});

test('asBatch: throws on shape mismatch', () => {
  assert.throws(() => asBatch(scalar(1), 4), /shape is \[\]/);
  assert.throws(() => asBatch(batchedScalar([1, 2, 3]), 4), /shape is \[3\]/);
  assert.throws(() => asBatch(batchedVector(new Float64Array(6), 2), 3),
    /shape is \[3,2\]/);
});

// =====================================================================
// Accessors
// =====================================================================

test('getShape / getData / getDType: trivial accessors', () => {
  const v = vector([1, 2]);
  assert.equal(getShape(v), v.shape);
  assert.equal(getData(v), v.data);
  assert.equal(getDType(v), 'f64');
});

test('getDType: explicit dtype slot is surfaced', () => {
  const v = { shape: [], data: new Float64Array(1), dtype: 'bool' };
  assert.equal(getDType(v), 'bool');
});

test('isBatched: leading axis match', () => {
  const N = 1024;
  assert.equal(isBatched(scalar(1), N), false);
  assert.equal(isBatched(batchedScalar(new Float64Array(N)), N), true);
  // Vector of length k != N is not batched along N
  assert.equal(isBatched(vector(new Float64Array(3)), N), false);
  // Atom-batched matrix shape=[N, m, n] is batched along N
  assert.equal(isBatched(batchedMatrix(new Float64Array(N * 4), 2, 2), N),
    true);
});

// =====================================================================
// Round-trip invariants
// =====================================================================

test('round-trip: asScalar(scalar(x)) === x', () => {
  for (const x of [-1.5, 0, 0.5, 3.14, 1e10, -1e-10]) {
    assert.equal(asScalar(scalar(x)), x);
  }
});

test('round-trip: asBatch(batchedScalar(arr), N) shares storage with arr', () => {
  const arr = new Float64Array([0.1, 0.2, 0.3]);
  assert.equal(asBatch(batchedScalar(arr), 3), arr);
});

test('storage invariant: every constructor returns Float64Array data', () => {
  const samples = [
    scalar(0),
    batchedScalar([1, 2]),
    vector([1, 2, 3]),
    batchedVector(new Float64Array(6), 2),
    matrix([1, 2, 3, 4], 2, 2),
    batchedMatrix(new Float64Array(8), 2, 2),
    withShape(new Float64Array(6), [1, 2, 3]),
    asValue(3.14),
    asValue([1, 2, 3]),
    asValue([[1, 2], [3, 4]]),
  ];
  for (const v of samples) {
    assert.ok(v.data instanceof Float64Array,
      'data is not Float64Array for shape=' + JSON.stringify(v.shape));
    assert.equal(v.data.length, numel(v.shape),
      'data length mismatch for shape=' + JSON.stringify(v.shape));
  }
});
