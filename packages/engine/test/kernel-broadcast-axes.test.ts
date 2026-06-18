'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { resolveBroadcastAxes, coordToOffset, cellToCoord } = require('../kernel-broadcast-axes.ts');

test('equal rank-2 grids align to their shape', () => {
  const r = resolveBroadcastAxes([
    { name: 'n', collectionAxes: [2, 3] },
    { name: 'p', collectionAxes: [2, 3] },
  ]);
  assert.deepEqual(r.axes, [2, 3]);
  // row-major strides over [2,3]: axis0 stride 3, axis1 stride 1
  assert.deepEqual(r.strides.n, [3, 1]);
  assert.deepEqual(r.strides.p, [3, 1]);
});

test('singleton axis expands via stride 0', () => {
  const r = resolveBroadcastAxes([
    { name: 'n', collectionAxes: [2, 1] },
    { name: 'p', collectionAxes: [2, 3] },
  ]);
  assert.deepEqual(r.axes, [2, 3]);
  assert.deepEqual(r.strides.n, [1, 0]);   // axis1 singleton → stride 0
  assert.deepEqual(r.strides.p, [3, 1]);
});

test('a scalar arg (no collection axes) rides every cell via all-zero strides', () => {
  const r = resolveBroadcastAxes([
    { name: 'n', collectionAxes: [2, 3] },
    { name: 'c', collectionAxes: [] },
  ]);
  assert.deepEqual(r.axes, [2, 3]);
  assert.deepEqual(r.strides.c, [0, 0]);
});

test('differing axis-count is a §04 error', () => {
  assert.throws(() => resolveBroadcastAxes([
    { name: 'a', collectionAxes: [2, 3] },
    { name: 'b', collectionAxes: [3] },
  ]), /same number of axes/);
});

test('incompatible (non-equal, non-singular) axis size is an error', () => {
  assert.throws(() => resolveBroadcastAxes([
    { name: 'a', collectionAxes: [2, 3] },
    { name: 'b', collectionAxes: [2, 4] },
  ]), /axis 1/);
});

test('coordToOffset is the dot of coord and strides', () => {
  assert.equal(coordToOffset([1, 2], [3, 1]), 5);
  assert.equal(coordToOffset([1, 2], [1, 0]), 1);   // singleton axis ignored
});

test('cellToCoord is row-major flat → coordinate', () => {
  assert.deepEqual(cellToCoord(0, [2, 3]), [0, 0]);
  assert.deepEqual(cellToCoord(4, [2, 3]), [1, 1]);
  assert.deepEqual(cellToCoord(5, [2, 3]), [1, 2]);
});

// =====================================================================
// Zero-arg / scalar-only case: all args have no collection axes → axes=[]
// =====================================================================
test('all-scalar args (no collection axes on any arg) → axes=[] one implicit cell', () => {
  const r = resolveBroadcastAxes([
    { name: 'n', collectionAxes: [] },
    { name: 'p', collectionAxes: [] },
  ]);
  // ranked.length === 0 → early return with axes=[] and zero-length strides
  assert.deepEqual(r.axes, []);
  assert.deepEqual(r.strides.n, []);
  assert.deepEqual(r.strides.p, []);
});

test('single all-scalar arg → axes=[]', () => {
  const r = resolveBroadcastAxes([
    { name: 'mu', collectionAxes: [] },
  ]);
  assert.deepEqual(r.axes, []);
  assert.deepEqual(r.strides.mu, []);
});

// =====================================================================
// Rank-3 alignment and stride computation
// =====================================================================
test('rank-3 equal grids align to their shape with correct row-major strides', () => {
  // axes=[2,3,4]: row-major strides = [3*4, 4, 1] = [12, 4, 1]
  const r = resolveBroadcastAxes([
    { name: 'a', collectionAxes: [2, 3, 4] },
    { name: 'b', collectionAxes: [2, 3, 4] },
  ]);
  assert.deepEqual(r.axes, [2, 3, 4]);
  assert.deepEqual(r.strides.a, [12, 4, 1]);
  assert.deepEqual(r.strides.b, [12, 4, 1]);
});

test('rank-3 broadcast: singleton on axis 1 and scalar arg', () => {
  // a=[2,1,4], b=[2,3,4], c=[] (scalar)
  // output axes=[2,3,4]; strides.a=[4,0,1] (axis1 singleton→0, axis0 stride=1*4=4),
  // strides.b=[12,4,1], strides.c=[0,0,0]
  const r = resolveBroadcastAxes([
    { name: 'a', collectionAxes: [2, 1, 4] },
    { name: 'b', collectionAxes: [2, 3, 4] },
    { name: 'c', collectionAxes: [] },
  ]);
  assert.deepEqual(r.axes, [2, 3, 4]);
  // row-major strides for a over [2,1,4]: axis2 non-sing→acc=1→st[2]=1,acc→1*4=4;
  // axis1 sing→st[1]=0,acc stays 4; axis0 non-sing→st[0]=4,acc→4*2=8
  assert.deepEqual(r.strides.a, [4, 0, 1]);
  assert.deepEqual(r.strides.b, [12, 4, 1]);
  assert.deepEqual(r.strides.c, [0, 0, 0]);
});

test('cellToCoord rank-3 exhaustive', () => {
  // axes=[2,2,2]: 8 cells, row-major
  assert.deepEqual(cellToCoord(0, [2, 2, 2]), [0, 0, 0]);
  assert.deepEqual(cellToCoord(1, [2, 2, 2]), [0, 0, 1]);
  assert.deepEqual(cellToCoord(2, [2, 2, 2]), [0, 1, 0]);
  assert.deepEqual(cellToCoord(3, [2, 2, 2]), [0, 1, 1]);
  assert.deepEqual(cellToCoord(4, [2, 2, 2]), [1, 0, 0]);
  assert.deepEqual(cellToCoord(7, [2, 2, 2]), [1, 1, 1]);
});
