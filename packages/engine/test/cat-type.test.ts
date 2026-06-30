'use strict';

// `cat` type inference — spec §07: concatenating values of the SAME structural
// kind. All scalars → a vector of the unified scalar type; all vectors → a
// concatenated vector; all records → a merged record (distinct fields). Mixing
// kinds is not permitted (a static error). `cat` was typed `deferred`; it now
// shares the `catShapeType` rule with positional `cartprod`/`joint` so the
// three can't drift, and a mixed-kind `cat` is rejected statically (instead of
// the old `cat(scalar, vector)` runtime garbage).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const T = require('../types.ts');

function infer(src: string, name: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  const b = r.bindings.get(name);
  return { t: b && b.inferredType, errs };
}

test('cat of scalars → a vector of the unified scalar type', () => {
  const { t, errs } = infer('x = cat(1.0, 2.3)', 'x');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [2]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('cat of int and real scalars unifies to real', () => {
  const { t } = infer('x = cat(1, 2.3, 4)', 'x');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('cat of vectors concatenates lengths', () => {
  const { t, errs } = infer('x = cat([1.0, 2.0, 3.0], [4.0, 5.0])', 'x');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [5]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('cat of records merges field lists', () => {
  const { t, errs } = infer('x = cat(record(a = 1.0, b = 2.0), record(c = 3.0))', 'x');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  assert.equal(t.kind, 'record');
  assert.deepEqual(Object.keys(t.fields), ['a', 'b', 'c']);
});

test('cat mixing a scalar and a vector is a static error (§07)', () => {
  const { errs } = infer('x = cat(1.0, [2.0, 3.0])', 'x');
  assert.ok(errs.some((e: any) => /all scalars, all vectors, or all records|mix of value kinds|not permitted/.test(e.message)),
    'expected a mixed-kind cat error; got ' + errs.map((e: any) => e.message).join(' | '));
});

test('cat of records with a duplicate field is a static error', () => {
  const { errs } = infer('x = cat(record(a = 1.0), record(a = 2.0))', 'x');
  assert.ok(errs.length >= 1, 'expected a duplicate-field cat error');
});
