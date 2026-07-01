'use strict';

// shape-contract.ts — the single source of truth for structural shape
// derivation, decoupled from the three layers that project it (type /
// materialiser layout / density consume). Stage S0 (shape-contract refactor)
// covers the CAT-SHAPE FAMILY: `cat` (values, §07), positional `cartprod`
// (sets, §03), and positional `joint` (variates, §06) all share ONE rule —
// all scalars → a vector of the unified element type; all rank-1 vectors → a
// concatenated vector; all records → a merged record with distinct fields;
// mixing structural kinds is not permitted. Before S0 this rule lived as a
// nested `catShapeType` helper inside typeinfer, unreachable from the
// materialiser / density layers (which must not import from typeinfer). These
// tests pin `catShape` (the owner → ShapeSpec) and `typeOfShape` (the type
// projection) so the extraction is behaviour-preserving and the later
// layout/consume projections have a stable spec to build on.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const SC = require('../shape-contract.ts');
const T = require('../types.ts');

// --- catShape: all-scalar → a vector (cat-array) ----------------------------

test('catShape: all scalars → catArray of length n, footprint-1 components', () => {
  const spec = SC.catShape([T.REAL, T.REAL, T.REAL]);
  assert.equal(spec.kind, 'catArray');
  assert.equal(spec.length, 3);
  assert.ok(T.equal(spec.elem, T.REAL), 'unified element type is real');
  assert.deepEqual(spec.components, [{ footprint: 1 }, { footprint: 1 }, { footprint: 1 }]);
});

test('catShape: a single scalar → catArray of length 1', () => {
  const spec = SC.catShape([T.REAL]);
  assert.equal(spec.kind, 'catArray');
  assert.equal(spec.length, 1);
  assert.deepEqual(spec.components, [{ footprint: 1 }]);
});

// --- catShape: all-vector → a concatenated vector ---------------------------

test('catShape: all rank-1 vectors → catArray of summed length, per-vector footprints', () => {
  const spec = SC.catShape([T.array(1, [3], T.REAL), T.array(1, [2], T.REAL)]);
  assert.equal(spec.kind, 'catArray');
  assert.equal(spec.length, 5);
  assert.deepEqual(spec.components, [{ footprint: 3 }, { footprint: 2 }]);
});

test('catShape: a %dynamic vector dim makes total length and that footprint %dynamic', () => {
  const spec = SC.catShape([T.array(1, ['%dynamic'], T.REAL), T.array(1, [2], T.REAL)]);
  assert.equal(spec.kind, 'catArray');
  assert.equal(spec.length, '%dynamic');
  assert.deepEqual(spec.components, [{ footprint: '%dynamic' }, { footprint: 2 }]);
});

// --- catShape: all-record → a merged record ---------------------------------

test('catShape: all records → mergedRecord, fields concatenated in order', () => {
  const spec = SC.catShape([T.record({ a: T.REAL, b: T.REAL }), T.record({ c: T.REAL })]);
  assert.equal(spec.kind, 'mergedRecord');
  assert.deepEqual(spec.fields.map((f: any) => f[0]), ['a', 'b', 'c']);
});

test('catShape: duplicate record field across components → null (caller diagnoses)', () => {
  const spec = SC.catShape([T.record({ a: T.REAL }), T.record({ a: T.REAL })]);
  assert.equal(spec, null);
});

// --- catShape: mixing / under-resolution / empty ----------------------------

test('catShape: mixing structural kinds (scalar + vector) → null', () => {
  assert.equal(SC.catShape([T.REAL, T.array(1, [2], T.REAL)]), null);
});

test('catShape: a rank-2 array component is unsupported → null', () => {
  assert.equal(SC.catShape([T.array(2, [2, 2], T.REAL), T.array(2, [2, 2], T.REAL)]), null);
});

test('catShape: a deferred/any component defers the whole shape', () => {
  const spec = SC.catShape([T.REAL, T.deferred()]);
  assert.equal(spec.kind, 'deferred');
  const spec2 = SC.catShape([T.any(), T.REAL]);
  assert.equal(spec2.kind, 'deferred');
});

test('catShape: empty / null input → null', () => {
  assert.equal(SC.catShape([]), null);
  assert.equal(SC.catShape(null), null);
});

// --- typeOfShape: project a ShapeSpec to a structural Type -------------------

test('typeOfShape: catArray → rank-1 array of the unified element type', () => {
  const t = SC.typeOfShape(SC.catShape([T.REAL, T.REAL]));
  assert.ok(T.equal(t, T.array(1, [2], T.REAL)), 'got ' + T.show(t));
});

test('typeOfShape: mergedRecord → record with fields in merge order', () => {
  const t = SC.typeOfShape(SC.catShape([T.record({ a: T.REAL }), T.record({ b: T.REAL })]));
  assert.equal(t.kind, 'record');
  assert.deepEqual(Object.keys(t.fields), ['a', 'b']);
});

test('typeOfShape: deferred spec → deferred type; null spec → null', () => {
  assert.equal(SC.typeOfShape({ kind: 'deferred' }).kind, 'deferred');
  assert.equal(SC.typeOfShape(null), null);
});

// --- The seam is behaviour-preserving: typeOfShape ∘ catShape reproduces the
//     exact type the old nested `catShapeType` returned for each branch. -----

test('seam: typeOfShape(catShape(...)) matches the legacy catShapeType outputs', () => {
  // all-scalar → array(1,[n],real)
  assert.ok(T.equal(SC.typeOfShape(SC.catShape([T.REAL, T.REAL, T.REAL])),
    T.array(1, [3], T.REAL)));
  // all-vector → array(1,[Σ],real)
  assert.ok(T.equal(SC.typeOfShape(SC.catShape([T.array(1, [3], T.REAL), T.array(1, [2], T.REAL)])),
    T.array(1, [5], T.REAL)));
  // mixed → null (caller diagnoses)
  assert.equal(SC.catShape([T.REAL, T.array(1, [2], T.REAL)]), null);
});
