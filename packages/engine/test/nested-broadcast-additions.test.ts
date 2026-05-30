'use strict';

// =====================================================================
// nested-broadcast-additions.test.ts — three nested-broadcast fixes
// =====================================================================
//
// Pins three additions from the "what's incompatible with fast nested
// broadcasting" follow-up:
//
//   (1) transpose / adjoint / relabel / addaxes / identity now in
//       DISSOLVE_AT_ANY_RANK_OPS — dissolution can lower a broadcast
//       whose body uses these to a direct call. (Klein-4 tag flip for
//       transpose/adjoint is free; relabel/addaxes/identity are
//       value-level identity for the dissolver's purposes.)
//
//   (2) Singleton-axis broadcast in value-ops elementwise — NumPy-
//       style: `[3, 1] + [1, 5]` → `[3, 5]` via stride-0 reads.
//       Spec §04: "Size-one array axes are implicitly expanded by
//       repetition to match the size of the other collection
//       arguments along these axes."
//
//   (3) Multi-axis broadcast fusion in `_tryDissolveBroadcastReduction`:
//       previously emitted one output axis (.atom). Now emits K
//       output axes when the broadcast-over args have rank K.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const valueOps = require('../value-ops.ts');

// =====================================================================
// 1. DISSOLVE_AT_ANY_RANK_OPS — set membership
// =====================================================================

test('dissolver: transpose / adjoint / relabel / addaxes / identity in any-rank set', () => {
  const dissolver = require('../dissolver.ts');
  const safe = dissolver.DISSOLVE_SAFE_OPS || dissolver._internal && dissolver._internal.DISSOLVE_SAFE_OPS;
  // The dissolver exports the safe set; if both `transpose` and
  // `relabel` are present, the additions stuck.
  assert.ok(safe instanceof Set,
    'DISSOLVE_SAFE_OPS exported as a Set');
  assert.ok(safe.has('transpose'), 'transpose in DISSOLVE_SAFE_OPS');
  assert.ok(safe.has('adjoint'),   'adjoint in DISSOLVE_SAFE_OPS');
  assert.ok(safe.has('relabel'),   'relabel in DISSOLVE_SAFE_OPS');
  assert.ok(safe.has('addaxes'),   'addaxes in DISSOLVE_SAFE_OPS');
  assert.ok(safe.has('identity'),  'identity in DISSOLVE_SAFE_OPS');
});

// =====================================================================
// 2. Singleton-axis broadcast in elementwise
// =====================================================================

test('value-ops add: [3,1] + [1,5] → [3,5] via singleton-axis broadcast', () => {
  // a = [[1], [2], [3]]   shape=[3, 1]
  // b = [[10, 20, 30, 40, 50]]   shape=[1, 5]
  // a + b = [[11, 21, 31, 41, 51], [12, 22, 32, 42, 52], [13, 23, 33, 43, 53]]
  const a = { shape: [3, 1], data: new Float64Array([1, 2, 3]) };
  const b = { shape: [1, 5], data: new Float64Array([10, 20, 30, 40, 50]) };
  const r = valueOps.add(a, b);
  assert.deepEqual(r.shape, [3, 5]);
  assert.deepEqual(Array.from(r.data), [
    11, 21, 31, 41, 51,
    12, 22, 32, 42, 52,
    13, 23, 33, 43, 53,
  ]);
});

test('value-ops mulElem: [4, 1] * [1, 3] → [4, 3] via singleton-axis broadcast', () => {
  const a = { shape: [4, 1], data: new Float64Array([1, 2, 3, 4]) };
  const b = { shape: [1, 3], data: new Float64Array([10, 20, 30]) };
  const r = valueOps.mulElem(a, b);
  assert.deepEqual(r.shape, [4, 3]);
  assert.deepEqual(Array.from(r.data), [
    10, 20, 30,
    20, 40, 60,
    30, 60, 90,
    40, 80, 120,
  ]);
});

test('value-ops add: rank-3 singleton broadcast [2, 1, 3] + [2, 4, 1] → [2, 4, 3]', () => {
  const a = { shape: [2, 1, 3], data: new Float64Array([
    1, 2, 3,         // i=0, j=0, k=0..2
    4, 5, 6,         // i=1, j=0, k=0..2
  ])};
  const b = { shape: [2, 4, 1], data: new Float64Array([
    10, 20, 30, 40,  // i=0, j=0..3
    100, 200, 300, 400, // i=1, j=0..3
  ])};
  const r = valueOps.add(a, b);
  assert.deepEqual(r.shape, [2, 4, 3]);
  // i=0, j=0: a[0,0,*] + b[0,0,*] = [1,2,3] + 10 = [11,12,13]
  assert.deepEqual(Array.from(r.data.slice(0, 3)), [11, 12, 13]);
  // i=0, j=3: a[0,0,*] + b[0,3,*] = [1,2,3] + 40 = [41,42,43]
  assert.deepEqual(Array.from(r.data.slice(9, 12)), [41, 42, 43]);
  // i=1, j=2: a[1,0,*] + b[1,2,*] = [4,5,6] + 300 = [304,305,306]
  assert.deepEqual(Array.from(r.data.slice(18, 21)), [304, 305, 306]);
});

test('value-ops add: rank/shape mismatch with NO size-1 axis → still errors', () => {
  const a = { shape: [3, 2], data: new Float64Array([1, 2, 3, 4, 5, 6]) };
  const b = { shape: [3, 4], data: new Float64Array(12).fill(0) };
  assert.throws(() => valueOps.add(a, b), /shape mismatch/);
});

test('value-ops add: same shape — fast-path preserved (regression)', () => {
  const a = { shape: [4], data: new Float64Array([1, 2, 3, 4]) };
  const b = { shape: [4], data: new Float64Array([10, 20, 30, 40]) };
  const r = valueOps.add(a, b);
  assert.deepEqual(r.shape, [4]);
  assert.deepEqual(Array.from(r.data), [11, 22, 33, 44]);
});

// =====================================================================
// 3. Multi-axis broadcast fusion in _tryDissolveBroadcastReduction
// =====================================================================

test('_tryDissolveBroadcastReduction: rank-2 broadcast-over → 2 output axes', () => {
  const dissolver = require('../dissolver.ts');
  // Synthesise the IR for:
  //   polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
  //   Y = polyeval.([C], X)  where X is rank-2 (a matrix per atom).
  // After dissolution, Y's aggregate emits TWO atom axes.
  // Build a hand-rolled functionof binding instead of running the
  // full pipeline — keeps the test focused on the dissolver output.
  const bindings = new Map();
  // C: rank-1 array.
  bindings.set('C', {
    ir: { kind: 'call', op: 'vector',
          args: [{ kind: 'lit', value: 1 }, { kind: 'lit', value: 2 }] },
    inferredType: { kind: 'array', rank: 1, shape: [2], elem: { kind: 'scalar', prim: 'real' } },
  });
  // X: rank-2 array (will fuse to 2 output axes).
  bindings.set('X', {
    ir: { kind: 'call', op: 'vector', args: [] },
    inferredType: { kind: 'array', rank: 2, shape: [3, 4], elem: { kind: 'scalar', prim: 'real' } },
  });
  // user fn polyeval body: sum(coeffs * x^0 + …) — simplified to just
  // `sum(get(coeffs, .j))` to exercise the fusion shape.
  const polyevalBinding = {
    ir: {
      kind: 'call', op: 'functionof',
      params: ['coeffs', 'x'],
      paramKwargs: ['coeffs', 'x'],
      body: {
        kind: 'call', op: 'sum',
        args: [{ kind: 'ref', ns: '%local', name: 'coeffs' }],
      },
    },
  };
  bindings.set('polyeval', polyevalBinding);
  // The broadcast IR: polyeval.([C], X)  — C is Ref-wrap, X is rank-2
  // broadcast-over.
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      { kind: 'ref', ns: 'self', name: 'polyeval' },
      // [C] = vector(C) — Ref-wrap of C
      { kind: 'call', op: 'vector',
        args: [{ kind: 'ref', ns: 'self', name: 'C' }] },
      // X = rank-2
      { kind: 'ref', ns: 'self', name: 'X' },
    ],
  };
  const dissolved = dissolver._tryDissolveBroadcastReduction(bcIR, bindings);
  assert.ok(dissolved, 'fusion fires for rank-2 broadcast-over');
  assert.equal(dissolved.op, 'aggregate');
  // Output axes vector — should have TWO entries (one per X axis).
  const axesIR = dissolved.args[1];
  assert.equal(axesIR.op, 'vector');
  assert.equal(axesIR.args.length, 2,
    'aggregate emits 2 output axes for rank-2 broadcast-over');
  assert.equal(axesIR.args[0].kind, 'axis');
  assert.equal(axesIR.args[1].kind, 'axis');
  assert.notEqual(axesIR.args[0].name, axesIR.args[1].name,
    'axis names are distinct');
});
