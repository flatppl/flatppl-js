'use strict';

// =====================================================================
// complex-elementwise.test.ts — Phase 3.1 + 3.2
// =====================================================================
//
// Pins the value-ops complex constructor / accessor primitives
// (complexElem / realElem / imagElem / conjElem / cisElem) and their
// dispatch through `BCAST_TABLE` so `complex.(re, im)` /
// `real.(z)` / `imag.(z)` / `conj.(z)` / `cis.(theta)` route through
// the variant registry as `broadcasted(<op>)` entries.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const vo = require('../value-ops.ts');
const vl = require('../value.ts');
const ops = require('../ops.ts');
require('../ops-declarations.ts');  // ensure BCAST_TABLE registered

// ---------------------------------------------------------------------
// Direct value-ops calls (Phase 3.1)
// ---------------------------------------------------------------------

test('complexElem: rank-1 real → rank-1 complex with planar buffers', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5, 6]);
  const c = vo.complexElem(re, im);
  assert.deepEqual(c.shape, [3]);
  assert.equal(c.dtype, 'complex');
  assert.deepEqual(Array.from(c.data), [1, 2, 3]);
  assert.deepEqual(Array.from(c.im), [4, 5, 6]);
});

test('complexElem: matrix shape preserved', () => {
  const re = { shape: [2, 3], data: new Float64Array([1, 2, 3, 4, 5, 6]) };
  const im = { shape: [2, 3], data: new Float64Array([7, 8, 9, 10, 11, 12]) };
  const c = vo.complexElem(re, im);
  assert.deepEqual(c.shape, [2, 3]);
  assert.equal(c.dtype, 'complex');
  assert.equal(c.data.length, 6);
  assert.equal(c.im.length, 6);
});

test('complexElem: rejects complex inputs', () => {
  const re = vl.vector([1, 2]);
  const im = vl.vector([3, 4]);
  const c = vo.complexElem(re, im);
  assert.throws(() => vo.complexElem(c, im), /must be real Values/);
});

test('complexElem: shape mismatch throws', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5]);
  assert.throws(() => vo.complexElem(re, im), /shape mismatch|rank mismatch/);
});

test('complexElem: rank-0 im broadcasts over rank-1 re (complex.(arr, 0.0))', () => {
  // The `complex.(values, 0.0)` idiom: a scalar imaginary part
  // broadcasts over the real array (shared `_broadcastOutShape`
  // rule — rank-0 expands trivially).
  const re = vl.vector([1, 2, 3]);
  const c = vo.complexElem(re, vl.scalar(0));
  assert.deepEqual(c.shape, [3]);
  assert.equal(c.dtype, 'complex');
  assert.deepEqual(Array.from(c.data), [1, 2, 3]);
  assert.deepEqual(Array.from(c.im), [0, 0, 0]);
});

test('complexElem: singleton-axis im expands (spec §04)', () => {
  // [2,3] re with a [1,3] im — the size-1 leading axis repeats per
  // row (stride-0 gather).
  const re = { shape: [2, 3], data: new Float64Array([1, 2, 3, 4, 5, 6]) };
  const im = { shape: [1, 3], data: new Float64Array([9, 8, 7]) };
  const c = vo.complexElem(re, im);
  assert.deepEqual(c.shape, [2, 3]);
  assert.deepEqual(Array.from(c.data), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(Array.from(c.im), [9, 8, 7, 9, 8, 7]);
});

test('realElem: identity on real Values', () => {
  const x = vl.vector([1, 2, 3]);
  const r = vo.realElem(x);
  assert.strictEqual(r, x, 'real Value passes through unchanged');
});

test('realElem: complex Value → real Value with re buffer', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5, 6]);
  const c = vo.complexElem(re, im);
  const r = vo.realElem(c);
  assert.equal(r.dtype, undefined);
  assert.equal(r.im, undefined);
  assert.deepEqual(Array.from(r.data), [1, 2, 3]);
});

test('imagElem: real Value → zero Value of same shape', () => {
  const x = vl.vector([1, 2, 3]);
  const i = vo.imagElem(x);
  assert.deepEqual(i.shape, [3]);
  assert.deepEqual(Array.from(i.data), [0, 0, 0]);
});

test('imagElem: complex Value → real Value with im buffer', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5, 6]);
  const c = vo.complexElem(re, im);
  const i = vo.imagElem(c);
  assert.deepEqual(Array.from(i.data), [4, 5, 6]);
});

test('imagElem: complex with conj-tag flips signs', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5, 6]);
  const c = vo.complexElem(re, im);
  const cConj = vo.conjElem(c);
  const i = vo.imagElem(cConj);
  assert.deepEqual(Array.from(i.data), [-4, -5, -6],
    'conj toggles the sign of the extracted imaginary part');
});

test('conjElem: real Value tag flip (logical no-op)', () => {
  const x = vl.vector([1, 2, 3]);
  const c = vo.conjElem(x);
  // conjugate toggles N ↔ C; data buffer unchanged.
  assert.equal(c.t, 'C');
  assert.deepEqual(Array.from(c.data), [1, 2, 3]);
});

test('conjElem: complex Value toggles conj bit', () => {
  const re = vl.vector([1, 2]);
  const im = vl.vector([3, 4]);
  const c = vo.complexElem(re, im);
  const cConj = vo.conjElem(c);
  // The data buffers stay the same; only the tag flips.
  assert.deepEqual(Array.from(cConj.data), [1, 2]);
  assert.deepEqual(Array.from(cConj.im), [3, 4]);
  assert.equal(cConj.t, 'C');
});

test('cisElem: matches cos+i*sin at canonical angles', () => {
  const theta = vl.vector([0, Math.PI / 2, Math.PI, 3 * Math.PI / 2]);
  const z = vo.cisElem(theta);
  assert.equal(z.dtype, 'complex');
  assert.deepEqual(z.shape, [4]);
  const eps = 1e-12;
  // cos: 1, 0, -1, 0
  assert.ok(Math.abs(z.data[0] - 1) < eps);
  assert.ok(Math.abs(z.data[1] - 0) < eps);
  assert.ok(Math.abs(z.data[2] - (-1)) < eps);
  assert.ok(Math.abs(z.data[3] - 0) < eps);
  // sin: 0, 1, 0, -1
  assert.ok(Math.abs(z.im[0] - 0) < eps);
  assert.ok(Math.abs(z.im[1] - 1) < eps);
  assert.ok(Math.abs(z.im[2] - 0) < eps);
  assert.ok(Math.abs(z.im[3] - (-1)) < eps);
});

test('cisElem: rejects complex input', () => {
  const re = vl.vector([1, 2]);
  const im = vl.vector([3, 4]);
  const c = vo.complexElem(re, im);
  assert.throws(() => vo.cisElem(c), /must be a real Value/);
});

// ---------------------------------------------------------------------
// BCAST_TABLE dispatch (Phase 3.2)
// ---------------------------------------------------------------------
//
// Verifies that `complex` / `real` / `imag` / `conj` / `cis` are
// registered with `wrappingOp: 'broadcast'` so `broadcast(<op>,
// args)` routes through the variant registry instead of falling to
// the runtime per-cell loop.

test('BCAST_TABLE: complex registered as broadcasted variant', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5, 6]);
  const r = ops.dispatchVariant('complex', [re, im], { wrappingOp: 'broadcast' });
  assert.ok(r, 'broadcasted(complex) variant must match for two real Values');
  assert.equal(r.dtype, 'complex');
  assert.deepEqual(Array.from(r.data), [1, 2, 3]);
  assert.deepEqual(Array.from(r.im), [4, 5, 6]);
});

test('BCAST_TABLE: real registered as broadcasted variant', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5, 6]);
  const c = vo.complexElem(re, im);
  const r = ops.dispatchVariant('real', [c], { wrappingOp: 'broadcast' });
  assert.ok(r, 'broadcasted(real) variant must match');
  assert.equal(r.dtype, undefined);
  assert.deepEqual(Array.from(r.data), [1, 2, 3]);
});

test('BCAST_TABLE: imag registered as broadcasted variant', () => {
  const re = vl.vector([1, 2, 3]);
  const im = vl.vector([4, 5, 6]);
  const c = vo.complexElem(re, im);
  const r = ops.dispatchVariant('imag', [c], { wrappingOp: 'broadcast' });
  assert.ok(r, 'broadcasted(imag) variant must match');
  assert.deepEqual(Array.from(r.data), [4, 5, 6]);
});

test('BCAST_TABLE: conj registered as broadcasted variant', () => {
  const re = vl.vector([1, 2]);
  const im = vl.vector([3, 4]);
  const c = vo.complexElem(re, im);
  const r = ops.dispatchVariant('conj', [c], { wrappingOp: 'broadcast' });
  assert.ok(r, 'broadcasted(conj) variant must match');
  assert.equal(r.t, 'C');
});

test('BCAST_TABLE: cis registered as broadcasted variant', () => {
  const theta = vl.vector([0, Math.PI]);
  const r = ops.dispatchVariant('cis', [theta], { wrappingOp: 'broadcast' });
  assert.ok(r, 'broadcasted(cis) variant must match');
  assert.equal(r.dtype, 'complex');
  assert.ok(Math.abs(r.data[0] - 1) < 1e-12);
  assert.ok(Math.abs(r.data[1] - (-1)) < 1e-12);
});
