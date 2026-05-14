'use strict';

// Tests for transposed_vector typeinfer — Phase A of "type inference
// + complex arithmetic". Adds the `tvector(length, elem)` type kind
// (spec §03 + FlatPIR §11 `%tvector`); transpose / adjoint dispatch on
// input shape per spec §07.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, types: T } = require('..');

function infer(src) {
  const { bindings, diagnostics } = processSource(src);
  const errors = diagnostics.filter(d => d.severity === 'error');
  return { bindings, errors };
}
function typeOf(bindings, name) {
  return bindings.get(name).inferredType;
}

// =====================================================================
// Type constructor
// =====================================================================

test('types.tvector: constructor + structural fields', () => {
  const t = T.tvector(3, T.REAL);
  assert.equal(t.kind, 'tvector');
  assert.equal(t.length, 3);
  assert.deepEqual(t.elem, T.REAL);
});

test('types.equal: tvector equality', () => {
  assert.ok(T.equal(T.tvector(3, T.REAL), T.tvector(3, T.REAL)));
  assert.ok(!T.equal(T.tvector(3, T.REAL), T.tvector(4, T.REAL)));
  // tvector vs array(1, ...) — distinct types per spec
  assert.ok(!T.equal(T.tvector(3, T.REAL), T.array(1, [3], T.REAL)));
});

test('types.show: tvector renders as "transposed vector of ..."', () => {
  assert.ok(T.show(T.tvector(3, T.REAL)).includes('transposed vector of real'));
  assert.ok(T.show(T.tvector('%dynamic', T.REAL)).includes('transposed vector of real'));
});

test('types.unify: tvector unification (%dynamic + length match)', () => {
  const s = new Map();
  assert.ok(T.unify(T.tvector(3, T.REAL), T.tvector(3, T.REAL), s) !== null);
  assert.ok(T.unify(T.tvector('%dynamic', T.REAL), T.tvector(5, T.REAL), s) !== null);
  assert.equal(T.unify(T.tvector(3, T.REAL), T.tvector(4, T.REAL), s), null);
});

test('types.isValue: tvector is a value type', () => {
  assert.ok(T.isValue(T.tvector(3, T.REAL)));
});

// =====================================================================
// transpose / adjoint inference
// =====================================================================

test('transpose(vector) → transposed_vector', () => {
  const { bindings, errors } = infer(`
v = [1.0, 2.0, 3.0]
t = transpose(v)
`);
  assert.deepEqual(errors, []);
  const tt = typeOf(bindings, 't');
  assert.equal(tt.kind, 'tvector', 'expected tvector, got ' + T.show(tt));
});

test('transpose(transpose(vector)) ≡ vector (involution)', () => {
  const { bindings, errors } = infer(`
v = [1.0, 2.0, 3.0]
tt = transpose(transpose(v))
`);
  assert.deepEqual(errors, []);
  const ttt = typeOf(bindings, 'tt');
  assert.equal(ttt.kind, 'array');
  assert.equal(ttt.rank, 1);
});

test('transpose(matrix) swaps dims', () => {
  const { bindings, errors } = infer(`
m = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
t = transpose(m)
`);
  assert.deepEqual(errors, []);
  const tt = typeOf(bindings, 't');
  assert.equal(tt.kind, 'array');
  assert.equal(tt.rank, 2);
});

test('adjoint(vector) → transposed_vector (same as transpose for real)', () => {
  const { bindings, errors } = infer(`
v = [1.0, 2.0]
a = adjoint(v)
`);
  assert.deepEqual(errors, []);
  const ta = typeOf(bindings, 'a');
  assert.equal(ta.kind, 'tvector');
});

test('adjoint(adjoint(vector)) ≡ vector (involution)', () => {
  const { bindings, errors } = infer(`
v = [1.0, 2.0]
aa = adjoint(adjoint(v))
`);
  assert.deepEqual(errors, []);
  const taa = typeOf(bindings, 'aa');
  assert.equal(taa.kind, 'array');
  assert.equal(taa.rank, 1);
});

// =====================================================================
// Substitute / unify round-trips (regression for the new branches)
// =====================================================================

test('substitute: tvector with type-variable elem', () => {
  const t = T.tvector(4, T.tvar('E'));
  const subst = new Map([['E', T.REAL]]);
  const r = T.substitute(t, subst);
  assert.equal(r.kind, 'tvector');
  assert.equal(r.length, 4);
  assert.deepEqual(r.elem, T.REAL);
});
