'use strict';

// The value/object containment boundary — spec §03 (arrays, records, tables)
// and §04 (tuples), as of commit "Allow nested records and nested tuples".
//
// What each container may hold:
//
//   array   : scalars or arrays                       (numeric; §03)
//   record  : scalars, arrays, or records             (§03 — NOT tables)
//   table   : vector or table columns                 (§03; see
//             table-nested-columns.test.ts)
//   tuple   : any object, including tuples             (§04 — tuples nest)
//
// This file pins what commit ee232b4 newly settles: nested records are
// allowed, and the OBJECT layer (measures / kernels / functions /
// likelihoods) plus TUPLES are forbidden in every value container — tuples
// everywhere except inside another tuple (§04).
//
// NOTE (out of scope here): §03 also says arrays are numeric (no
// record/table elements) and record fields are scalar/array/record (no
// tables). The engine does NOT yet enforce those — `[record(...), ...]` in
// particular is currently produced and consumed by the generative executor
// and the transport corpus. That pre-existing spec/engine tension is tracked
// separately and intentionally left as-is by this change.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

function infer(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  return { bindings: r.bindings, errors: errs };
}
function errText(errors: any[]) {
  return errors.map((e: any) => e.message).join(' | ');
}

// =====================================================================
// Allowed: the nesting the commit blesses
// =====================================================================

test('record may contain a record (nested records, §03)', () => {
  const { errors } = infer(`r = record(a = record(p = 1.0, q = 2.0), b = 3.0)`);
  assert.equal(errors.length, 0, 'nested record must be allowed: ' + errText(errors));
});

test('record may contain a scalar and an array', () => {
  const { errors } = infer(`r = record(a = 1.0, b = [1.0, 2.0, 3.0])`);
  assert.equal(errors.length, 0, errText(errors));
});

test('array may contain arrays (matrix literal)', () => {
  const { errors } = infer(`m = [[1.0, 2.0], [3.0, 4.0]]`);
  assert.equal(errors.length, 0, errText(errors));
});

test('tuple may contain a tuple (§04 — tuples nest)', () => {
  const { errors } = infer(`t = (1.0, (2.0, 3.0))`);
  assert.equal(errors.length, 0, errText(errors));
});

test('tuple may contain measures (a tuple bundles objects, §04)', () => {
  // §04: tuples "package such outputs as an ordered, fixed-length bundle of
  // FlatPPL objects" — e.g. a kernel and its base measure together.
  const { errors } = infer(`pair = (Normal(0.0, 1.0), Exponential(1.0))`);
  assert.equal(errors.length, 0, 'a tuple of measures must be allowed: ' + errText(errors));
});

// =====================================================================
// Rejected: tuples in value containers (§04)
// =====================================================================

test('a tuple inside a record is rejected (§04)', () => {
  const { errors } = infer(`r = record(t = (1.0, 2.0))`);
  assert.ok(errors.length >= 1, 'expected a §04 tuple-in-record error');
  assert.match(errors[0].message, /tuple may not appear inside a record/);
});

test('a tuple inside an array is rejected (§04)', () => {
  const { errors } = infer(`xs = [(1.0, 2.0), (3.0, 4.0)]`);
  assert.ok(errors.some((e: any) => /tuple may not appear inside an array/.test(e.message)),
    'expected a tuple-in-array error; got ' + errText(errors));
});

test('a tuple column inside a table is rejected (§04)', () => {
  const { errors } = infer(`t = table(c = [(1.0, 2.0), (3.0, 4.0)])`);
  assert.ok(errors.some((e: any) => /tuple may not appear inside/.test(e.message)),
    'expected a tuple-in-table error; got ' + errText(errors));
});

// =====================================================================
// Out of scope: arrays-of-records (a "vector of records") is currently
// PERMITTED by the engine even though §03 reserves that for tables. Pinned
// here to document the gap and to fail loudly if the policy ever changes
// without a deliberate decision (see the NOTE at the top of this file).
// =====================================================================

test('array-of-records is currently permitted (pre-existing §03 tension, not enforced)', () => {
  const { errors } = infer(`a = [record(p = 1.0), record(p = 2.0)]`);
  assert.equal(errors.length, 0,
    'array-of-records is intentionally still permitted by this change; '
    + 'unexpected error: ' + errText(errors));
});
