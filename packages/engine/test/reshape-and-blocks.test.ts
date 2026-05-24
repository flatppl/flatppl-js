'use strict';

// Spec §07 reshape additions + matrix constructors / extractors.
//
//   tile(A, size)         — repeat along each axis
//   splitblocks(A, blk)   — nested array of equal-shape sub-arrays
//   joinblocks(A)         — inverse of splitblocks
//   blockdiagmat(mats)    — block-diagonal matrix from vector-of-matrices
//   bandedmat(v, rows)    — banded matrix with v shifted right per row
//   diag(A, k=0)          — extract k-th diagonal of a matrix

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const valueLib = require('../value.ts');
const { toJS } = require('./_value-helpers.ts');

function lit(v: any)        { return { kind: 'lit', value: v }; }
function vec(...vs: any[])  { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function call(op: any, ...args: any[]) { return { kind: 'call', op, args }; }
const rowstack = (rows: any[][]) =>
  call('rowstack', { kind: 'call', op: 'vector', args: rows.map(r => vec(...r)) });
const evRaw = (ir: any) => sampler.evaluateExpr(ir, {});
const ev = (ir: any) => toJS(evRaw(ir));

// =====================================================================
// tile
// =====================================================================

test('tile([1,2,3], 3) ⇒ [1,2,3,1,2,3,1,2,3]', () => {
  assert.deepEqual(ev(call('tile', vec(1, 2, 3), lit(3))),
    [1, 2, 3, 1, 2, 3, 1, 2, 3]);
});

test('tile(rowstack([[1,2,3]]), [2, 1]) ⇒ rows repeated', () => {
  // M is (1, 3); tile by [2, 1] repeats rows.
  const r = ev(call('tile', rowstack([[1, 2, 3]]), vec(2, 1)));
  assert.deepEqual(r, [[1, 2, 3], [1, 2, 3]]);
});

test('tile(rowstack([[1,2,3]]), [1, 2]) ⇒ columns repeated', () => {
  const r = ev(call('tile', rowstack([[1, 2, 3]]), vec(1, 2)));
  assert.deepEqual(r, [[1, 2, 3, 1, 2, 3]]);
});

test('tile: size length mismatch ⇒ error', () => {
  assert.throws(() => ev(call('tile', vec(1, 2, 3), vec(2, 2))),
    /must match A\.rank/);
});

// =====================================================================
// splitblocks
// =====================================================================

test('splitblocks([1,2,3,4,5,6], 2) ⇒ [[1,2],[3,4],[5,6]]', () => {
  const r = ev(call('splitblocks', vec(1, 2, 3, 4, 5, 6), lit(2)));
  assert.deepEqual(r, [[1, 2], [3, 4], [5, 6]]);
});

test('splitblocks: non-divisible size ⇒ error', () => {
  assert.throws(() => ev(call('splitblocks', vec(1, 2, 3, 4, 5), lit(2))),
    /not divisible by blocksize/);
});

test('splitblocks: rank-2 with blocksize [m, n]', () => {
  // 4×4 matrix split into 2×2 blocks ⇒ 2×2 outer of 2×2 inner.
  const M = rowstack([
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12],
    [13,14, 15, 16],
  ]);
  const r = ev(call('splitblocks', M, vec(2, 2)));
  assert.deepEqual(r, [
    [[[1, 2], [5, 6]], [[3, 4], [7, 8]]],
    [[[9, 10], [13, 14]], [[11, 12], [15, 16]]],
  ]);
});

// =====================================================================
// joinblocks — inverse of splitblocks
// =====================================================================

test('joinblocks(splitblocks(v, k)) ≡ v (rank-1)', () => {
  const v = vec(1, 2, 3, 4, 5, 6);
  const round = ev(call('joinblocks', call('splitblocks', v, lit(2))));
  assert.deepEqual(round, [1, 2, 3, 4, 5, 6]);
});

test('joinblocks(splitblocks(M, blk)) ≡ M (rank-2)', () => {
  const M = rowstack([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
  ]);
  const round = ev(call('joinblocks',
    call('splitblocks', M, vec(1, 2))));
  assert.deepEqual(round, [[1, 2, 3, 4], [5, 6, 7, 8]]);
});

// =====================================================================
// blockdiagmat
// =====================================================================

test('blockdiagmat([A, B]) ⇒ block-diagonal matrix', () => {
  const A = rowstack([[1, 2], [3, 4]]);
  const B = rowstack([[5, 6, 7], [8, 9, 10]]);
  const r = ev(call('blockdiagmat',
    { kind: 'call', op: 'vector', args: [A, B] }));
  assert.deepEqual(r, [
    [1, 2, 0, 0, 0],
    [3, 4, 0, 0, 0],
    [0, 0, 5, 6, 7],
    [0, 0, 8, 9, 10],
  ]);
});

test('blockdiagmat([]) ⇒ empty matrix', () => {
  const r = evRaw(call('blockdiagmat',
    { kind: 'call', op: 'vector', args: [] }));
  assert.deepEqual(r.shape, [0, 0]);
});

// =====================================================================
// bandedmat
// =====================================================================

test('bandedmat([1, 2, 3], 4) ⇒ 4×6 banded matrix', () => {
  const r = ev(call('bandedmat', vec(1, 2, 3), lit(4)));
  assert.deepEqual(r, [
    [1, 2, 3, 0, 0, 0],
    [0, 1, 2, 3, 0, 0],
    [0, 0, 1, 2, 3, 0],
    [0, 0, 0, 1, 2, 3],
  ]);
});

test('bandedmat: scalar k=1 produces a 1×n shape', () => {
  const r = evRaw(call('bandedmat', vec(7, 8), lit(1)));
  assert.deepEqual(r.shape, [1, 2]);
  assert.deepEqual(Array.from(r.data), [7, 8]);
});

// =====================================================================
// diag (extract)
// =====================================================================

test('diag(M) extracts the main diagonal', () => {
  const M = rowstack([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  assert.deepEqual(ev(call('diag', M)), [1, 5, 9]);
});

test('diag(M, 1) extracts the first super-diagonal', () => {
  const M = rowstack([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  assert.deepEqual(ev(call('diag', M, lit(1))), [2, 6]);
});

test('diag(M, -1) extracts the first sub-diagonal', () => {
  const M = rowstack([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  assert.deepEqual(ev(call('diag', M, lit(-1))), [4, 8]);
});

test('diag(diagmat(v)) ≡ v', () => {
  const v = vec(2, 5, 9);
  const round = ev(call('diag', call('diagmat', v)));
  assert.deepEqual(round, [2, 5, 9]);
});

test('diag: non-matrix input ⇒ error', () => {
  assert.throws(() => ev(call('diag', vec(1, 2, 3))),
    /argument must be a rank-2 matrix/);
});

// =====================================================================
// Spec identities (sanity)
// =====================================================================

test('joinblocks(splitblocks(A, blocksize)) ≡ A (spec identity)', () => {
  // 6-element rank-1 split by 3 → outer [2], inner [3].
  const A = vec(10, 20, 30, 40, 50, 60);
  const back = ev(call('joinblocks', call('splitblocks', A, lit(3))));
  assert.deepEqual(back, [10, 20, 30, 40, 50, 60]);
});
