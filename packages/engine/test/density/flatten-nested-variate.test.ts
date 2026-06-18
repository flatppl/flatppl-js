'use strict';
// Contract test for density._internal.flattenNestedVariate — the helper that
// consumes a multi-axis broadcast variate (the observed value) into a flat
// row-major scalar buffer + residual `rest`, for every value representation
// the density walk can receive. logdensityof literals always arrive as nested
// JS arrays of numbers (one path); the other representations (rank-1 Value,
// Float64Array, rank-0/rank-1 Value leaves) and the loud ragged/shape errors
// arise from internally-produced values and are pinned here directly.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const density = require('../../density.ts');
const valueLib = require('../../value.ts');
const fnv = density._internal.flattenNestedVariate;

const arr = (x: any) => Array.from(x);

test('nested JS array, exact grid → row-major cells, null rest', () => {
  const { cells, rest } = fnv([[1, 2, 3], [4, 5, 6]], [2, 3]);
  assert.deepEqual(arr(cells), [1, 2, 3, 4, 5, 6]);
  assert.equal(rest, null);
});

test('nested JS array with extra outer rows → rest carries the surplus', () => {
  const { cells, rest } = fnv([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [2, 3]);
  assert.deepEqual(arr(cells), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(rest, [[7, 8, 9]]);
});

test('rank-1 Value, 1-D grid, exact → cells, null rest', () => {
  const v = valueLib.withShape(new Float64Array([0.1, 0.2]), [2]);
  const { cells, rest } = fnv(v, [2]);
  assert.deepEqual(arr(cells), [0.1, 0.2]);
  assert.equal(rest, null);
});

test('rank-1 Value, 1-D grid, surplus → Value rest', () => {
  const v = valueLib.withShape(new Float64Array([0.1, 0.2, 0.3]), [3]);
  const { cells, rest } = fnv(v, [2]);
  assert.deepEqual(arr(cells), [0.1, 0.2]);
  assert.equal(rest.shape[0], 1);
  assert.equal(rest.data[0], 0.3);
});

test('Float64Array, 1-D grid → cells + typed-array rest', () => {
  const exact = fnv(new Float64Array([5, 6]), [2]);
  assert.deepEqual(arr(exact.cells), [5, 6]);
  assert.equal(exact.rest, null);
  const surplus = fnv(new Float64Array([5, 6, 7]), [2]);
  assert.deepEqual(arr(surplus.cells), [5, 6]);
  assert.deepEqual(arr(surplus.rest), [7]);
});

test('rank-0 Value leaves inside a nested array', () => {
  const v = [valueLib.withShape(new Float64Array([1.5]), []),
             valueLib.withShape(new Float64Array([2.5]), [])];
  const { cells } = fnv(v, [2]);
  assert.deepEqual(arr(cells), [1.5, 2.5]);
});

test('rank-1 Value leaf (first element consumed)', () => {
  const v = [valueLib.withShape(new Float64Array([9, 8]), [2])];
  const { cells } = fnv(v, [1]);
  assert.deepEqual(arr(cells), [9]);
});

test('Float64Array leaves inside a nested array', () => {
  const { cells } = fnv([new Float64Array([7]), new Float64Array([8])], [2]);
  assert.deepEqual(arr(cells), [7, 8]);
});

test('ragged nested array (short inner row) throws a §04-shaped error', () => {
  assert.throws(() => fnv([[1, 2], [3, 4, 5]], [2, 3]),
    /axis 1 wants dim=3 but got Array of length 2/);
});

test('rank-1 Value too short throws', () => {
  assert.throws(() => fnv(valueLib.withShape(new Float64Array([1, 2]), [2]), [3]),
    /wants 3 entries, only 2/);
});

test('Float64Array too short throws', () => {
  assert.throws(() => fnv(new Float64Array([1, 2]), [3]), /wants 3 entries, only 2/);
});

test('unexpected leaf type throws', () => {
  assert.throws(() => fnv(['not-a-number'], [1]), /unexpected leaf type/);
});

test('non-array at an intermediate depth throws', () => {
  assert.throws(() => fnv(42, [2, 2]), /expected Array at depth 0/);
});
