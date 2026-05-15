'use strict';

// Spec §07 Array generation: linspace, extlinspace, partition, reverse.
// All pure value functions; tested through sampler.evaluateExpr.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler');

function lit(v)        { return { kind: 'lit', value: v }; }
function vec(...vs)    { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function call(op, ...args) { return { kind: 'call', op, args }; }
const ev = (ir) => sampler.evaluateExpr(ir, {});

// =====================================================================
// linspace
// =====================================================================

test('linspace(0, 10, 5) ⇒ [0, 2.5, 5, 7.5, 10]', () => {
  assert.deepEqual(ev(call('linspace', lit(0), lit(10), lit(5))),
    [0, 2.5, 5, 7.5, 10]);
});

test('linspace endpoints are exact (no floating-point drift)', () => {
  const r = ev(call('linspace', lit(1.0), lit(2.0), lit(100)));
  assert.equal(r[0], 1.0);
  assert.equal(r[99], 2.0);
});

test('linspace(n=1) ⇒ single-element vector at `from`', () => {
  assert.deepEqual(ev(call('linspace', lit(3.14), lit(99), lit(1))), [3.14]);
});

test('linspace(n=0) ⇒ empty vector', () => {
  assert.deepEqual(ev(call('linspace', lit(0), lit(10), lit(0))), []);
});

// =====================================================================
// extlinspace — linspace with ±∞ overflow edges
// =====================================================================

test('extlinspace(0, 10, 5) ⇒ [-∞, 0, 2.5, 5, 7.5, 10, ∞]', () => {
  assert.deepEqual(ev(call('extlinspace', lit(0), lit(10), lit(5))),
    [-Infinity, 0, 2.5, 5, 7.5, 10, Infinity]);
});

test('extlinspace(n=0) ⇒ [-∞, ∞] (just the overflow edges)', () => {
  assert.deepEqual(ev(call('extlinspace', lit(0), lit(10), lit(0))),
    [-Infinity, Infinity]);
});

// =====================================================================
// partition
// =====================================================================

test('partition(xs, n): equal-size groups when n divides length', () => {
  assert.deepEqual(ev(call('partition', vec(1, 2, 3, 4, 5, 6), lit(3))),
    [[1, 2, 3], [4, 5, 6]]);
});

test('partition(xs, [n1, n2, ...]): per-group sizes', () => {
  assert.deepEqual(ev(call('partition', vec(1, 2, 3, 4, 5), vec(2, 3))),
    [[1, 2], [3, 4, 5]]);
});

test('partition: equal-size with non-divisible length ⇒ error', () => {
  assert.throws(
    () => ev(call('partition', vec(1, 2, 3, 4, 5), lit(2))),
    /not divisible/);
});

test('partition: spec sum ≠ length ⇒ error', () => {
  assert.throws(
    () => ev(call('partition', vec(1, 2, 3, 4, 5), vec(2, 2))),
    /spec sums to 4 but vector length is 5/);
});

// =====================================================================
// reverse
// =====================================================================

test('reverse: vector → reversed vector', () => {
  assert.deepEqual(ev(call('reverse', vec(1, 2, 3, 4))), [4, 3, 2, 1]);
});

test('reverse: empty vector ⇒ empty vector', () => {
  assert.deepEqual(ev(call('reverse', vec())), []);
});

test('reverse: single-element vector ⇒ same single element', () => {
  assert.deepEqual(ev(call('reverse', vec(42))), [42]);
});

// =====================================================================
// fill / zeros / ones / eye / onehot
// =====================================================================

test('fill(x, n) ⇒ length-n array of x', () => {
  assert.deepEqual(ev(call('fill', lit(3), lit(4))), [3, 3, 3, 3]);
});

test('fill(x, n, m) ⇒ n × m nested array', () => {
  assert.deepEqual(ev(call('fill', lit(0), lit(2), lit(3))),
    [[0, 0, 0], [0, 0, 0]]);
});

test('zeros(n) ⇒ vector of zeros; ones(n) ⇒ vector of ones', () => {
  assert.deepEqual(ev(call('zeros', lit(3))), [0, 0, 0]);
  assert.deepEqual(ev(call('ones',  lit(3))), [1, 1, 1]);
});

test('zeros / ones: multi-dimensional', () => {
  assert.deepEqual(ev(call('zeros', lit(2), lit(2))), [[0, 0], [0, 0]]);
  assert.deepEqual(ev(call('ones',  lit(2), lit(2))), [[1, 1], [1, 1]]);
});

test('eye(n) ⇒ n × n identity matrix', () => {
  assert.deepEqual(ev(call('eye', lit(3))),
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
});

test('onehot(i, n) ⇒ length-n basis vector with 1 at position i (1-based)', () => {
  assert.deepEqual(ev(call('onehot', lit(2), lit(4))), [0, 1, 0, 0]);
  assert.deepEqual(ev(call('onehot', lit(1), lit(3))), [1, 0, 0]);
  assert.deepEqual(ev(call('onehot', lit(3), lit(3))), [0, 0, 1]);
});

test('onehot: out-of-range index ⇒ runtime error', () => {
  assert.throws(() => ev(call('onehot', lit(0), lit(3))), /out of range/);
  assert.throws(() => ev(call('onehot', lit(4), lit(3))), /out of range/);
});

// =====================================================================
// boolean / integer scalar restrictors
// =====================================================================

test('boolean: accepts true / false; coerces 0 / 1', () => {
  assert.equal(ev(call('boolean', lit(true))),  true);
  assert.equal(ev(call('boolean', lit(false))), false);
  assert.equal(ev(call('boolean', lit(0))),     false);
  assert.equal(ev(call('boolean', lit(1))),     true);
});

test('boolean: non-boolean numeric ⇒ runtime error', () => {
  assert.throws(() => ev(call('boolean', lit(3.14))), /not a boolean/);
});

test('integer: accepts integers verbatim', () => {
  assert.equal(ev(call('integer', lit(42))),  42);
  assert.equal(ev(call('integer', lit(-7))),  -7);
  assert.equal(ev(call('integer', lit(0))),   0);
});

test('integer: non-integer ⇒ runtime error', () => {
  assert.throws(() => ev(call('integer', lit(3.5))), /not an integer/);
});

// =====================================================================
// array(data, size, dimorder) — n-D from flat data
// =====================================================================

test('array: row-major 2×3 (dimorder = [1, 2])', () => {
  // Equivalent to rowstack([[1,2,3], [4,5,6]]).
  assert.deepEqual(
    ev(call('array', vec(1, 2, 3, 4, 5, 6), vec(2, 3), vec(1, 2))),
    [[1, 2, 3], [4, 5, 6]]);
});

test('array: column-major 2×3 (dimorder = [2, 1])', () => {
  // Equivalent to colstack([[1,2], [3,4], [5,6]]).
  assert.deepEqual(
    ev(call('array', vec(1, 2, 3, 4, 5, 6), vec(2, 3), vec(2, 1))),
    [[1, 3, 5], [2, 4, 6]]);
});

test('array: 1-D shape pass-through', () => {
  assert.deepEqual(
    ev(call('array', vec(1, 2, 3), vec(3), vec(1))),
    [1, 2, 3]);
});

test('array: data length mismatch ⇒ runtime error', () => {
  assert.throws(
    () => ev(call('array', vec(1, 2, 3), vec(2, 3), vec(1, 2))),
    /does not match data length/);
});

test('array: dimorder length mismatch ⇒ runtime error', () => {
  assert.throws(
    () => ev(call('array', vec(1, 2, 3, 4, 5, 6), vec(2, 3), vec(1))),
    /must match size length/);
});

test('array: 3-D shape (2×2×2, row-major)', () => {
  // [1..8] with dimorder [1, 2, 3] (slowest = axis 1, fastest = axis 3)
  // Layout: [k=0: [1,2; 3,4],  k=1: [5,6; 7,8]]
  const r = ev(call('array', vec(1, 2, 3, 4, 5, 6, 7, 8),
                    vec(2, 2, 2), vec(1, 2, 3)));
  assert.deepEqual(r, [
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]],
  ]);
});

// =====================================================================
// rowstack / colstack
// =====================================================================

function vov(...rows) {
  return { kind: 'call', op: 'vector', args: rows.map(r => vec(...r)) };
}

test('rowstack: input vectors become rows', () => {
  assert.deepEqual(ev(call('rowstack', vov([1, 2, 3], [4, 5, 6]))),
    [[1, 2, 3], [4, 5, 6]]);
});

test('colstack: input vectors become columns', () => {
  assert.deepEqual(ev(call('colstack', vov([1, 2, 3], [4, 5, 6]))),
    [[1, 4], [2, 5], [3, 6]]);
});

test('rowstack: mismatched row lengths ⇒ runtime error', () => {
  assert.throws(() => ev(call('rowstack', vov([1, 2], [3, 4, 5]))),
    /length mismatch/);
});

test('colstack: mismatched column lengths ⇒ runtime error', () => {
  assert.throws(() => ev(call('colstack', vov([1, 2], [3, 4, 5]))),
    /length mismatch/);
});

// =====================================================================
// addaxes — explicit singular-axis insertion (spec §07; commit 87c9be1)
// =====================================================================

const valueLib = require('../value');

test('addaxes(A, 2, 1): inserts leading/trailing singular axes', () => {
  const r = ev(call('addaxes', vec(1, 2, 3), lit(2), lit(1)));
  assert.deepEqual(r.shape, [1, 1, 3, 1]);
  assert.deepEqual(Array.from(r.data), [1, 2, 3]);
  assert.equal(valueLib.numel(r.shape), r.data.length);
});

test('addaxes(A, 0, 0): identity shape, same content', () => {
  const r = ev(call('addaxes', vec(5, 6), lit(0), lit(0)));
  assert.deepEqual(r.shape, [2]);
  assert.deepEqual(Array.from(r.data), [5, 6]);
});

test('addaxes is zero-copy: result shares the input buffer', () => {
  // A as a Value flows through unchanged; the data buffer is reused.
  const A = valueLib.vector([7, 8, 9]);
  const r = sampler.evaluateExpr(
    { kind: 'call', op: 'addaxes',
      args: [{ kind: 'lit', value: A }, lit(1), lit(2)] }, {});
  assert.deepEqual(r.shape, [1, 3, 1, 1]);
  assert.equal(r.data, A.data, 'buffer shared (no copy)');
});

test('addaxes: NumPy- vs Julia-style alignment is explicit', () => {
  // A length-3 vector lifted to a row (1,3) vs a column (3,1) — the
  // engine never picks; the user states it via addaxes.
  const numpyRow = ev(call('addaxes', vec(1, 2, 3), lit(1), lit(0)));
  const juliaCol = ev(call('addaxes', vec(1, 2, 3), lit(0), lit(1)));
  assert.deepEqual(numpyRow.shape, [1, 3]);
  assert.deepEqual(juliaCol.shape, [3, 1]);
});

test('addaxes: negative axis count is rejected', () => {
  assert.throws(() => ev(call('addaxes', vec(1, 2), lit(-1), lit(0))),
    /non-negative integer/);
});

test('addaxes: complex Value carries its imaginary buffer through', () => {
  const z = valueLib.complexValue([1, 2], [3, 4], [2]);
  const r = sampler.evaluateExpr(
    { kind: 'call', op: 'addaxes',
      args: [{ kind: 'lit', value: z }, lit(1), lit(0)] }, {});
  assert.deepEqual(r.shape, [1, 2]);
  assert.equal(r.dtype, 'complex');
  assert.deepEqual(Array.from(r.im), [3, 4]);
});
