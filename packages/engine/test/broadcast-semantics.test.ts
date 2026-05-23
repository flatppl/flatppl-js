'use strict';

// broadcast realignment to spec §04 + flatppl-design commit 87c9be1:
// same-#-axes (no implicit axis insertion), singleton expansion,
// non-collection inputs loop-invariant (incl. the callable), records/
// tuples disallowed, no-collection ⇒ single call. Driven through
// sampler.evaluateExpr with a functionof IR (the value-evaluator path).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');

const ref = (n: any) => ({ kind: 'ref', ns: 'self', name: n });
const lit = (v: any) => ({ kind: 'lit', value: v });
const addaxes = (a: any, l: any, t: any) =>
  ({ kind: 'call', op: 'addaxes', args: [a, lit(l), lit(t)] });
const fnOf = (params: any, body: any) =>
  ({ kind: 'call', op: 'functionof', params: params, body: body });
const bc = (...a: any[]) => ({ kind: 'call', op: 'broadcast', args: a });
const ev = (ir: any) => sampler.evaluateExpr(ir, {});

const addAB = fnOf(['a', 'b'],
  { kind: 'call', op: 'add', args: [ref('a'), ref('b')] });
const dblA = fnOf(['a'],
  { kind: 'call', op: 'mul', args: [ref('a'), lit(2)] });

// ---- back-compat: equal-length 1-D ------------------------------------

test('broadcast: equal-length vectors (unchanged behaviour)', () => {
  assert.deepEqual(ev(bc(addAB, lit([1, 2, 3]), lit([10, 20, 30]))),
    [11, 22, 33]);
});

// ---- singleton expansion ---------------------------------------------

test('broadcast: size-1 axis expands by repetition', () => {
  assert.deepEqual(ev(bc(addAB, lit([1, 2, 3]), lit([100]))),
    [101, 102, 103]);
  // either side may be the singleton
  assert.deepEqual(ev(bc(addAB, lit([5]), lit([1, 2, 3, 4]))),
    [6, 7, 8, 9]);
});

// ---- non-collection inputs are loop-invariant ------------------------

test('broadcast: scalar input is held constant, not iterated', () => {
  assert.deepEqual(ev(bc(addAB, lit([1, 2, 3]), lit(5))), [6, 7, 8]);
});

test('broadcast: no collection arguments ⇒ a single call (scalar)', () => {
  assert.equal(ev(bc(dblA, lit(7))), 14);
});

// ---- no implicit axis insertion (same #axes required) ----------------

test('broadcast: differing ranks is an error (no implicit insertion)', () => {
  // matrix (1×3 via addaxes) vs bare vector (rank 1) — ranks differ.
  assert.throws(
    () => ev(bc(addAB, addaxes(lit([1, 2, 3]), 1, 0), lit([10, 20, 30]))),
    /same number of axes.*addaxes/s);
});

test('broadcast: NumPy- vs Julia-style alignment via addaxes', () => {
  // 2×3 matrix + a length-3 vector. NumPy-style: vector → (1,3),
  // expands down the rows. Julia-style: vector → (3,1) does NOT
  // conform here (3 vs 2 on axis 0) — that asymmetry is the whole
  // point: the engine never picks, the user states the alignment.
  const M = { kind: 'call', op: 'array',
    args: [lit([1, 2, 3, 4, 5, 6]), lit([2, 3]), lit([1, 2])] };
  const numpy = ev(bc(addAB, M, addaxes(lit([10, 20, 30]), 1, 0)));
  assert.deepEqual(numpy, [[11, 22, 33], [14, 25, 36]]);
  // Julia-style addaxes(v,0,1) → (3,1): axis0 3 vs matrix 2 ⇒ error.
  assert.throws(
    () => ev(bc(addAB, M, addaxes(lit([10, 20, 30]), 0, 1))),
    /incompatible sizes on axis 0/);
});

// ---- records / tuples disallowed -------------------------------------

test('broadcast: a record input is rejected (spec §04)', () => {
  assert.throws(() => ev(bc(dblA, lit({ x: 1, y: 2 }))),
    /records and tuples are not allowed/);
});

// ---- 2-D elementwise (matching ranks, no singletons) -----------------

test('broadcast: 2-D elementwise over two matrices', () => {
  const A = { kind: 'call', op: 'array',
    args: [lit([1, 2, 3, 4]), lit([2, 2]), lit([1, 2])] };
  const B = { kind: 'call', op: 'array',
    args: [lit([10, 20, 30, 40]), lit([2, 2]), lit([1, 2])] };
  assert.deepEqual(ev(bc(addAB, A, B)), [[11, 22], [33, 44]]);
});
