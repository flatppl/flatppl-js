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
