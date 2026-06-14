'use strict';

// The ragged-per-atom value kind (engine-concepts §2.3) — the
// VectorOfVectors foundation for PoissonProcess and any genuinely-ragged
// per-atom output. Pins: construction + framing validation, atom slicing
// back to a uniform Value, the flatview round-trip, the load-bearing
// composition properties (length-preserving elementwise reuses the flat
// path with offsets carried; segmented reduce collapses to uniform [N]).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const R = require('../ragged.ts');

const arr = (a: any) => Float64Array.from(a);

// =====================================================================
// Construction + framing
// =====================================================================

test('ragged: build from a JS array of variable-length arrays', () => {
  const v = R.raggedFromArrays([[1, 2, 3], [4], [], [5, 6]]);
  assert.ok(R.isRagged(v));
  assert.equal(R.raggedCount(v), 4);
  assert.deepEqual(Array.from(v.offsets), [0, 3, 4, 4, 6]);
  assert.deepEqual(Array.from(v.data), [1, 2, 3, 4, 5, 6]);
  assert.equal(R.raggedSpan(v, 0), 3);
  assert.equal(R.raggedSpan(v, 2), 0);   // empty atom
});

test('ragged: direct constructor validates the offsets frame', () => {
  const data = arr([1, 2, 3, 4]);
  assert.ok(R.isRagged(R.ragged(data, Int32Array.from([0, 2, 4]))));
  // offsets must end at data.length
  assert.throws(() => R.ragged(data, Int32Array.from([0, 2, 3])), /run 0 … data.length/);
  // non-decreasing
  assert.throws(() => R.ragged(data, Int32Array.from([0, 3, 2, 4])), /non-decreasing/);
});

test('ragged: kernelShape requires segment lengths divisible by the stride', () => {
  // d=2 points: each atom is a [count_i, 2] block, so spans must be even.
  const v = R.ragged(arr([1, 2, 3, 4, 5, 6]), Int32Array.from([0, 4, 6]), [2]);
  assert.equal(R.raggedElemCount(v, 0), 2);   // 4 / 2
  assert.equal(R.raggedElemCount(v, 1), 1);   // 2 / 2
  assert.throws(() => R.ragged(arr([1, 2, 3]), Int32Array.from([0, 3]), [2]),
    /not a multiple of kernelShape stride/);
});

// =====================================================================
// Slicing back to a uniform Value
// =====================================================================

test('ragged: raggedElem yields a uniform Value view of atom i (scalar points)', () => {
  const v = R.raggedFromArrays([[1, 2, 3], [4], [5, 6]]);
  const e0 = R.raggedElem(v, 0);
  assert.deepEqual(e0.shape, [3]);
  assert.deepEqual(Array.from(e0.data), [1, 2, 3]);
  const e2 = R.raggedElem(v, 2);
  assert.deepEqual(e2.shape, [2]);
  assert.deepEqual(Array.from(e2.data), [5, 6]);
});

test('ragged: raggedElem of a d-dim kernel yields [count_i, d]', () => {
  const v = R.ragged(arr([1, 2, 3, 4, 5, 6]), Int32Array.from([0, 4, 6]), [2]);
  const e0 = R.raggedElem(v, 0);
  assert.deepEqual(e0.shape, [2, 2]);
  assert.deepEqual(Array.from(e0.data), [1, 2, 3, 4]);
});

test('ragged: raggedElem is a VIEW (subarray, no copy)', () => {
  const v = R.raggedFromArrays([[1, 2], [3, 4]]);
  const e1 = R.raggedElem(v, 1);
  e1.data[0] = 99;
  assert.equal(v.data[2], 99, 'writing the view mutates the backing buffer');
});

test('ragged: round-trips through nested JS arrays', () => {
  const src = [[1, 2, 3], [], [4], [5, 6]];
  const v = R.raggedFromArrays(src);
  const back = R.raggedToNested(v).map((a: any) => Array.from(a));
  assert.deepEqual(back, src);
});

// =====================================================================
// Composition (§2.3): the load-bearing properties
// =====================================================================

test('ragged: length-preserving elementwise reuses the flat path, offsets carried', () => {
  const v = R.raggedFromArrays([[1, 2, 3], [4], [5, 6]]);
  // Flat fn over the whole [6] buffer (stand-in for the broadcast fast path).
  const doubled = R.raggedMapFlat(v, (flat: any) => ({
    shape: flat.shape, data: flat.data.map((x: number) => x * 2),
  }));
  assert.ok(R.isRagged(doubled));
  assert.equal(doubled.offsets, v.offsets, 'offsets carried by reference (pure metadata)');
  assert.deepEqual(R.raggedToNested(doubled).map((a: any) => Array.from(a)),
    [[2, 4, 6], [8], [10, 12]]);
});

test('ragged: elementwise map rejects a length-changing flatFn', () => {
  const v = R.raggedFromArrays([[1, 2], [3]]);
  assert.throws(() => R.raggedMapFlat(v, (flat: any) => ({
    shape: [2], data: Float64Array.from([1, 2]),   // wrong length (3 → 2)
  })), /length-preserving/);
});

test('ragged: segmented reduce collapses to a uniform [N] Value', () => {
  const v = R.raggedFromArrays([[1, 2, 3], [4], [], [5, 6]]);
  const sums = R.raggedSegmentReduce(v, (acc: number, x: number) => acc + x, 0);
  assert.deepEqual(sums.shape, [4]);
  assert.deepEqual(Array.from(sums.data), [6, 4, 0, 11]);
  // counts via reduce(+1)
  const counts = R.raggedSegmentReduce(v, (acc: number) => acc + 1, 0);
  assert.deepEqual(Array.from(counts.data), [3, 1, 0, 2]);
});

test('ragged: same-structure check is offsets (+ kernel) equality', () => {
  const a = R.raggedFromArrays([[1, 2], [3]]);
  const b = R.raggedFromArrays([[9, 8], [7]]);     // same offsets
  const c = R.raggedFromArrays([[1], [2, 3]]);     // different offsets
  assert.ok(R.raggedSameStructure(a, b));
  assert.ok(!R.raggedSameStructure(a, c));
});

// =====================================================================
// Disjointness from the uniform Value kind
// =====================================================================

test('ragged: a ragged value is NOT a uniform Value (no shape)', () => {
  const valueLib = require('../value.ts');
  const v = R.raggedFromArrays([[1, 2], [3]]);
  assert.ok(!valueLib.isValue(v), 'ragged must not be mistaken for a uniform Value');
  // ...and a uniform Value is not ragged.
  assert.ok(!R.isRagged({ shape: [2], data: arr([1, 2]) }));
});
