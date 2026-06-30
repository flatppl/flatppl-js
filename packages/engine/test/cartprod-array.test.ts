'use strict';

// Positional `cartprod` is a set of ARRAYS, not a tuple — spec §03 (clarified):
// each member is the `cat` of one element per component, so the value is an
// ordinary array of the common element type (length = sum of component
// arities), with each component set refining its position. This mirrors
// `cat` on values and `cartpow` on a scalar set, and removes the tuple that
// leaked into `elementof` results, measure domains, and record fields (§04).
//
// (Per-position set membership belongs in the valueset/support layer, which is
// `%unknown` for cartprod today — a separate follow-up; this file pins the
// VALUE TYPE only. The keyword form stays a record.)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const T = require('../types.ts');

function typeOf(src: string, name: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  const b = r.bindings.get(name);
  return { t: b && b.inferredType, errs };
}

test('positional cartprod of scalar sets → array of the common element type', () => {
  const { t, errs } = typeOf('x = elementof(cartprod(reals, reals))', 'x');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [2]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('heterogeneous cartprod(reals, integers) → array of real (integers ⊆ reals)', () => {
  const { t, errs } = typeOf('x = elementof(cartprod(reals, integers))', 'x');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [2]);
  assert.ok(T.equal(t.elem, T.REAL), 'element type unifies to real, got ' + T.show(t.elem));
});

test('cartprod(reals, reals) and cartpow(reals, 2) infer the same type', () => {
  const a = typeOf('x = elementof(cartprod(reals, reals))', 'x').t;
  const b = typeOf('x = elementof(cartpow(reals, 2))', 'x').t;
  assert.ok(T.equal(a, b), 'cartprod ≡ cartpow for the homogeneous case: ' + T.show(a) + ' vs ' + T.show(b));
});

test('cartprod of vector components concatenates them (cat of vectors)', () => {
  // cat([..2..], [..3..]) → one 5-vector.
  const { t, errs } = typeOf('x = elementof(cartprod(cartpow(reals, 2), cartpow(reals, 3)))', 'x');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [5]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('cartprod mixing structural kinds (scalar + vector) is a static error (§07 cat)', () => {
  // §07: concatenating a mix of value kinds is not permitted, so cartprod of a
  // scalar set and a vector set is rejected (not silently flattened to a 4-array).
  const { errs } = typeOf('x = elementof(cartprod(reals, cartpow(reals, 3)))', 'x');
  assert.ok(errs.some((e: any) => /mixing structural kinds|all scalar sets, all vector/.test(e.message)),
    'expected a mixed-kind cartprod error; got ' + errs.map((e: any) => e.message).join(' | '));
});

test('single-component cartprod is the component itself', () => {
  const { t } = typeOf('x = elementof(cartprod(reals))', 'x');
  assert.ok(T.equal(t, T.REAL));
});

test('keyword cartprod stays a record (unchanged)', () => {
  const { t, errs } = typeOf('x = elementof(cartprod(a = reals, b = integers))', 'x');
  assert.equal(errs.length, 0);
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.a, T.REAL));
  assert.ok(T.equal(t.fields.b, T.INTEGER));
});

test('positional cartprod no longer leaks a tuple into a record field (§04)', () => {
  // The keyword form with a positional-cartprod component used to embed a
  // tuple (a §04 violation); now the field is an array, accepted cleanly.
  const { t, errs } = typeOf('x = elementof(cartprod(a = cartprod(reals, integers), b = reals))', 'x');
  assert.equal(errs.length, 0, 'no tuple-in-record §04 error: ' + errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'record');
  assert.equal(t.fields.a.kind, 'array');
  assert.deepEqual(t.fields.a.shape, [2]);
});

test('cartprod as a measure domain is an array domain, drawable (not a tuple)', () => {
  const { t, errs } = typeOf('m = Lebesgue(support = cartprod(reals, posreals))', 'm');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array', 'domain is an array, got ' + T.show(t.domain));
});
