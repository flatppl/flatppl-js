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
