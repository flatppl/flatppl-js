'use strict';

// Tests for density.js Value-aware consume helpers — Phase 5 of the
// shape-explicit refactor. consumeScalar now understands rank-0 and
// rank-1 Values (alongside the legacy JS scalar / Float64Array / JS
// array forms); consumeVector splits a value into a leading n-vector
// + rest, used by multivariate leaves (Phase 6 MvNormal, follow-ups).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const density = require('..').density;
const valueLib = require('..').value;
const { consumeScalar, consumeVector } = density._internal;

// =====================================================================
// consumeScalar — Value awareness
// =====================================================================

test('consumeScalar: rank-0 Value → head=number, rest=null', () => {
  const v = valueLib.scalar(3.14);
  const r = consumeScalar(v);
  assert.equal(r.head, 3.14);
  assert.equal(r.rest, null);
});

test('consumeScalar: rank-1 Value (len 1) → head=first, rest=null', () => {
  const v = { shape: [1], data: new Float64Array([42]) };
  const r = consumeScalar(v);
  assert.equal(r.head, 42);
  assert.equal(r.rest, null);
});

test('consumeScalar: rank-1 Value (len >1) → head + rest=Value', () => {
  const v = { shape: [3], data: new Float64Array([1, 2, 3]) };
  const r = consumeScalar(v);
  assert.equal(r.head, 1);
  assert.ok(valueLib.isValue(r.rest));
  assert.deepEqual(r.rest.shape, [2]);
  assert.deepEqual(Array.from(r.rest.data), [2, 3]);
});

test('consumeScalar: chained consumes drain a rank-1 Value', () => {
  let v = { shape: [3], data: new Float64Array([10, 20, 30]) };
  const heads = [];
  while (v != null) {
    const r = consumeScalar(v);
    heads.push(r.head);
    v = r.rest;
  }
  assert.deepEqual(heads, [10, 20, 30]);
});

test('consumeScalar: rank>1 Value rejected', () => {
  const M = { shape: [2, 2], data: new Float64Array([1, 2, 3, 4]) };
  assert.throws(() => consumeScalar(M), /rank 2/);
});

test('consumeScalar: empty rank-1 Value rejected', () => {
  const empty = { shape: [0], data: new Float64Array(0) };
  assert.throws(() => consumeScalar(empty), /vector exhausted/);
});

test('consumeScalar: legacy forms still work (Float64Array, JS array, number)', () => {
  // Legacy path unchanged — Phase 5 only ADDED the Value branch.
  assert.deepEqual(consumeScalar(2.5), { head: 2.5, rest: null });
  const fa = new Float64Array([1, 2]);
  const r = consumeScalar(fa);
  assert.equal(r.head, 1);
  assert.ok(r.rest.BYTES_PER_ELEMENT);
  assert.equal(r.rest[0], 2);
});

// =====================================================================
// consumeVector — new in Phase 5
// =====================================================================

test('consumeVector: rank-1 Value of exact length → head=subarray, rest=null', () => {
  const v = { shape: [3], data: new Float64Array([1, 2, 3]) };
  const r = consumeVector(v, 3);
  assert.ok(r.head instanceof Float64Array);
  assert.equal(r.head.length, 3);
  assert.deepEqual(Array.from(r.head), [1, 2, 3]);
  assert.equal(r.rest, null);
});

test('consumeVector: rank-1 Value longer than n → head=prefix, rest=Value tail', () => {
  const v = { shape: [5], data: new Float64Array([1, 2, 3, 4, 5]) };
  const r = consumeVector(v, 2);
  assert.deepEqual(Array.from(r.head), [1, 2]);
  assert.ok(valueLib.isValue(r.rest));
  assert.deepEqual(r.rest.shape, [3]);
  assert.deepEqual(Array.from(r.rest.data), [3, 4, 5]);
});

test('consumeVector: Float64Array (legacy) of exact length', () => {
  const fa = new Float64Array([10, 20, 30]);
  const r = consumeVector(fa, 3);
  assert.deepEqual(Array.from(r.head), [10, 20, 30]);
  assert.equal(r.rest, null);
});

test('consumeVector: Float64Array longer than n → head + rest both subarrays', () => {
  const fa = new Float64Array([10, 20, 30, 40]);
  const r = consumeVector(fa, 2);
  assert.deepEqual(Array.from(r.head), [10, 20]);
  assert.deepEqual(Array.from(r.rest), [30, 40]);
});

test('consumeVector: JS array consumption', () => {
  const r = consumeVector([1, 2, 3, 4], 2);
  assert.deepEqual(Array.from(r.head), [1, 2]);
  assert.deepEqual(r.rest, [3, 4]);
});

test('consumeVector: JS number can satisfy n=1', () => {
  const r = consumeVector(3.14, 1);
  assert.equal(r.head.length, 1);
  assert.equal(r.head[0], 3.14);
  assert.equal(r.rest, null);
});

test('consumeVector: JS number with n>1 rejected', () => {
  assert.throws(() => consumeVector(3.14, 2), /from scalar/);
});

test('consumeVector: too short rejected with explicit count', () => {
  const v = { shape: [2], data: new Float64Array([1, 2]) };
  assert.throws(() => consumeVector(v, 3), /wants 3 entries, only 2 available/);
});

test('consumeVector: rank>1 Value rejected', () => {
  const M = { shape: [2, 2], data: new Float64Array(4) };
  assert.throws(() => consumeVector(M, 2), /rank-1 Value/);
});

test('consumeVector: head subarray shares storage with input', () => {
  // The head is a subarray view, not a copy — important for the
  // multivariate leaf walker that uses head as input to matvec /
  // logpdf-mvn without re-allocating.
  const data = new Float64Array([1, 2, 3, 4, 5]);
  const r = consumeVector({ shape: [5], data }, 3);
  data[0] = 999;
  assert.equal(r.head[0], 999, 'head must share storage with input');
});
