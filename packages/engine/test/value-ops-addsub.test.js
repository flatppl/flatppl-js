'use strict';

// Tests for value-ops.add / sub / neg — shape-dispatched elementwise
// arithmetic. Phase 2c of the shape-explicit refactor.
//
// Coverage:
//
//   1. scalar + scalar
//   2. scalar + vector / matrix (broadcast, tag preserved)
//   3. vector + vector, matrix + matrix (elementwise, shape + tag must
//      match)
//   4. Orientation mismatch rejection (column + row vectors of same
//      length must NOT silently elementwise-add)
//   5. Shape mismatch rejection
//   6. sub non-commutativity with scalar broadcast (`s - v` ≠ `v - s`)
//   7. neg: pointwise; tag and shape preserved; fresh allocation
//   8. Dispatch through ARITH_OPS.add/sub/neg and ARITH_OPS_N.add/sub/neg

const { test } = require('node:test');
const assert = require('node:assert/strict');

const valueLib = require('..').value;
const valueOps = require('../value-ops');
const sampler = require('../sampler');

const { scalar, vector, matrix, transpose, getTag } = valueLib;

// =====================================================================
// add
// =====================================================================

test('add: scalar + scalar → scalar', () => {
  const r = valueOps.add(scalar(2), scalar(3));
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], 5);
});

test('add: scalar + vector → vector (tag preserved)', () => {
  const r = valueOps.add(scalar(10), vector([1, 2, 3]));
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [11, 12, 13]);
  assert.equal(getTag(r), 'N');
});

test('add: vector + scalar (symmetric)', () => {
  const r = valueOps.add(vector([1, 2, 3]), scalar(10));
  assert.deepEqual(Array.from(r.data), [11, 12, 13]);
});

test('add: scalar + transpose(vector) → row vector (tag T preserved)', () => {
  const rv = transpose(vector([1, 2, 3]));
  const r = valueOps.add(scalar(1), rv);
  assert.deepEqual(r.shape, [3]);
  assert.equal(getTag(r), 'T');
});

test('add: vector + vector (elementwise)', () => {
  const r = valueOps.add(vector([1, 2, 3]), vector([10, 20, 30]));
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [11, 22, 33]);
});

test('add: matrix + matrix (elementwise)', () => {
  const A = matrix([1, 2, 3, 4], 2, 2);
  const B = matrix([10, 20, 30, 40], 2, 2);
  const r = valueOps.add(A, B);
  assert.deepEqual(r.shape, [2, 2]);
  assert.deepEqual(Array.from(r.data), [11, 22, 33, 44]);
});

test('add: column + row of same length REJECTED (orientation mismatch)', () => {
  // Both have shape=[3] but logically column + row vector is undefined.
  // The Klein-4 tag distinguishes them.
  assert.throws(
    () => valueOps.add(vector([1, 2, 3]), transpose(vector([4, 5, 6]))),
    /opposite orientation/);
});

test('add: shape mismatch rejected', () => {
  assert.throws(
    () => valueOps.add(vector([1, 2]), vector([1, 2, 3])),
    /shape mismatch/);
});

test('add: rank mismatch rejected', () => {
  assert.throws(
    () => valueOps.add(vector([1, 2, 3, 4]), matrix([1, 2, 3, 4], 2, 2)),
    /rank mismatch/);
});

test('add: transpose(matrix) + transpose(matrix) of same shape → ok', () => {
  // Both have logical shape=[3, 2] with t='T'; data underneath is in
  // [2, 3] layout. Elementwise add on the data buffers is well-defined
  // because the underlying-shape and swapped bits match.
  const A = transpose(matrix([1, 2, 3, 4, 5, 6], 2, 3));
  const B = transpose(matrix([10, 20, 30, 40, 50, 60], 2, 3));
  const r = valueOps.add(A, B);
  assert.deepEqual(r.shape, [3, 2]);
  assert.equal(getTag(r), 'T');
  assert.deepEqual(Array.from(r.data), [11, 22, 33, 44, 55, 66]);
});

// =====================================================================
// sub
// =====================================================================

test('sub: scalar - scalar', () => {
  assert.equal(valueOps.sub(scalar(7), scalar(3)).data[0], 4);
});

test('sub: scalar - vector (broadcast respects argument order)', () => {
  // 10 - [1,2,3] = [9, 8, 7]
  const r = valueOps.sub(scalar(10), vector([1, 2, 3]));
  assert.deepEqual(Array.from(r.data), [9, 8, 7]);
});

test('sub: vector - scalar (symmetric)', () => {
  // [1,2,3] - 10 = [-9, -8, -7]
  const r = valueOps.sub(vector([1, 2, 3]), scalar(10));
  assert.deepEqual(Array.from(r.data), [-9, -8, -7]);
});

test('sub: vector - vector', () => {
  const r = valueOps.sub(vector([10, 20, 30]), vector([1, 2, 3]));
  assert.deepEqual(Array.from(r.data), [9, 18, 27]);
});

// =====================================================================
// neg
// =====================================================================

test('neg: scalar', () => {
  const r = valueOps.neg(scalar(5));
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], -5);
});

test('neg: vector', () => {
  const r = valueOps.neg(vector([1, -2, 3]));
  assert.deepEqual(Array.from(r.data), [-1, 2, -3]);
});

test('neg: transpose(vector) preserves T tag', () => {
  const r = valueOps.neg(transpose(vector([1, 2, 3])));
  assert.equal(getTag(r), 'T');
  assert.deepEqual(Array.from(r.data), [-1, -2, -3]);
});

test('neg: matrix', () => {
  const r = valueOps.neg(matrix([1, -2, 3, -4], 2, 2));
  assert.deepEqual(Array.from(r.data), [-1, 2, -3, 4]);
});

test('neg: allocates fresh data (does not alias input)', () => {
  const v = vector([1, 2, 3]);
  const r = valueOps.neg(v);
  v.data[0] = 999;
  assert.equal(r.data[0], -1, 'neg result must not alias input data');
});

// =====================================================================
// Dispatch through ARITH_OPS and ARITH_OPS_N
// =====================================================================

const { ARITH_OPS, ARITH_OPS_N } = sampler._internal;

test('ARITH_OPS.add: bare scalars stay on JS fast path', () => {
  assert.equal(ARITH_OPS.add(2, 3), 5);
  assert.equal(typeof ARITH_OPS.add(2, 3), 'number');
});

test('ARITH_OPS.add: shape-rich Value routes to value-ops', () => {
  const r = ARITH_OPS.add(vector([1, 2, 3]), vector([10, 20, 30]));
  assert.deepEqual(Array.from(r.data), [11, 22, 33]);
});

test('ARITH_OPS.sub: shape-rich Value routes to value-ops', () => {
  const A = matrix([5, 6, 7, 8], 2, 2);
  const B = matrix([1, 2, 3, 4], 2, 2);
  const r = ARITH_OPS.sub(A, B);
  assert.deepEqual(Array.from(r.data), [4, 4, 4, 4]);
});

test('ARITH_OPS.neg: shape-rich Value routes to value-ops', () => {
  const r = ARITH_OPS.neg(vector([1, -2, 3]));
  assert.deepEqual(Array.from(r.data), [-1, 2, -3]);
});

test('ARITH_OPS_N.add: shape-rich Value bypasses broadcast2', () => {
  // N=4; matrix is intrinsic (shape=[2,2], leading dim ≠ N).
  const A = matrix([1, 2, 3, 4], 2, 2);
  const B = matrix([10, 20, 30, 40], 2, 2);
  const r = ARITH_OPS_N.add([A, B], 4);
  assert.deepEqual(r.shape, [2, 2]);
  assert.deepEqual(Array.from(r.data), [11, 22, 33, 44]);
});

test('ARITH_OPS_N.neg: shape-rich Value bypasses broadcast1', () => {
  const v = vector([1, 2, 3]);
  const r = ARITH_OPS_N.neg([v], 4);
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [-1, -2, -3]);
});

test('ARITH_OPS_N.add: bare-only path unchanged (regression)', () => {
  const a = new Float64Array([1, 2, 3, 4]);
  const r = ARITH_OPS_N.add([a, 10], 4);
  assert.ok(r instanceof Float64Array);
  assert.deepEqual(Array.from(r), [11, 12, 13, 14]);
});
