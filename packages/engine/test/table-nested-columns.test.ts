'use strict';

// Tables with table-valued columns — spec §03 (commit "Allow nested records
// and nested tuples"):
//
//   "Each column is a vector or a table ... Each row of a table is a record;
//    if some columns of the table are tables themselves, the corresponding
//    entries of the row records are records themselves."
//
// So a column may itself be a (sub-)table of the SAME row count. Then:
//   - `t.sub`   (column access) yields the sub-table (a table, nrows rows);
//   - `t[i]`    (row access) yields a record whose `sub` field is the row-i
//               record of the sub-table (a nested record, NOT a table);
//   - reductions (`sum`, `mean`, …) recurse column-wise into the sub-table,
//               producing a nested record of per-column reductions;
//   - broadcast traverses outer rows, each cell a record with a nested
//               record field.
// A higher-dimensional ARRAY column remains forbidden (no leading-axis
// convention); only vector or table columns are allowed.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const T = require('../types.ts');
const { toJS } = require('./_value-helpers.ts');

function infer(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  return { bindings: r.bindings, errors: errs };
}
function typeOf(bindings: any, name: string) {
  const b = bindings.get(name);
  return b && b.inferredType;
}
function valueOf(name: string, src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) {
    throw new Error('source had errors: ' + errs.map((e: any) => e.message).join(' | '));
  }
  const ds = buildDerivations(r.bindings);
  return ds.fixedValues && ds.fixedValues.get(name);
}

const NESTED = `
  inner = table(p = [1.0, 2.0, 3.0], q = [4.0, 5.0, 6.0])
  outer = table(x = [10.0, 20.0, 30.0], sub = inner)
`;

// =====================================================================
// Type inference
// =====================================================================

test('nested table: a table column infers a table-typed column', () => {
  const { bindings, errors } = infer(NESTED);
  assert.equal(errors.length, 0, errors.map((e: any) => e.message).join(' | '));
  const t = typeOf(bindings, 'outer');
  assert.equal(t.kind, 'table');
  assert.equal(t.nrows, 3);
  assert.ok(T.equal(t.columns.x, T.REAL));
  // The `sub` column is itself a table type of the same row count.
  assert.equal(t.columns.sub.kind, 'table');
  assert.equal(t.columns.sub.nrows, 3);
  assert.ok(T.equal(t.columns.sub.columns.p, T.REAL));
  assert.ok(T.equal(t.columns.sub.columns.q, T.REAL));
});

test('nested table: column access t.sub returns the sub-table', () => {
  const { bindings, errors } = infer(NESTED + `\n  c = outer.sub`);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'c');
  assert.equal(t.kind, 'table');
  assert.equal(t.nrows, 3);
  assert.ok(T.equal(t.columns.p, T.REAL));
});

test('nested table: a vector column still accesses as an array', () => {
  const { bindings, errors } = infer(NESTED + `\n  cx = outer.x`);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'cx');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('nested table: row access t[i] makes the table-column entry a record', () => {
  const { bindings, errors } = infer(NESTED + `\n  row = outer[1]`);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'row');
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.x, T.REAL));
  // The `sub` entry of the row is a RECORD (the sub-table's row), not a table.
  assert.equal(t.fields.sub.kind, 'record');
  assert.ok(T.equal(t.fields.sub.fields.p, T.REAL));
  assert.ok(T.equal(t.fields.sub.fields.q, T.REAL));
});

test('var/std over a vector-per-entry column: type is a real-leaf vector (reduce over rows)', () => {
  // var/std reduce over the ROW axis only, so a 3-vector-per-row column keeps
  // its cell shape with a real leaf — not a bare scalar real.
  const { bindings, errors } = infer(`
    t = table(p = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], w = [10.0, 20.0])
    v = var(t)
  `);
  assert.equal(errors.length, 0, errors.map((e: any) => e.message).join(' | '));
  const ty = typeOf(bindings, 'v');
  assert.equal(ty.kind, 'record');
  assert.equal(ty.fields.p.kind, 'array');
  assert.deepEqual(ty.fields.p.shape, [3]);
  assert.ok(T.equal(ty.fields.p.elem, T.REAL));
  assert.ok(T.equal(ty.fields.w, T.REAL));
});

test('all-rows access t[:] infers an array of the row record (not deferred)', () => {
  const { bindings, errors } = infer(`
    t = table(x = [1.0, 2.0, 3.0], y = [4.0, 5.0, 6.0])
    rows = t[:]
  `);
  assert.equal(errors.length, 0, errors.map((e: any) => e.message).join(' | '));
  const t = typeOf(bindings, 'rows');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.equal(t.elem.kind, 'record');
  assert.ok(T.equal(t.elem.fields.x, T.REAL));
});

test('all-rows access over a NESTED table infers array of nested row records', () => {
  const { bindings, errors } = infer(NESTED + `\n  rows = outer[:]`);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'rows');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.equal(t.elem.kind, 'record');
  // A table column becomes a nested record in each row record.
  assert.equal(t.elem.fields.sub.kind, 'record');
  assert.ok(T.equal(t.elem.fields.sub.fields.p, T.REAL));
});

test('nested table: chained access t.sub.p types as the inner column vector', () => {
  const { bindings, errors } = infer(NESTED + `\n  pcol = outer.sub.p`);
  assert.equal(errors.length, 0, errors.map((e: any) => e.message).join(' | '));
  const t = typeOf(bindings, 'pcol');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('nested table: column-wise sum recurses into a nested record', () => {
  const { bindings, errors } = infer(NESTED + `\n  s = sum(outer)`);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 's');
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.x, T.REAL));
  assert.equal(t.fields.sub.kind, 'record', 'sub reduction must be a nested record');
  assert.ok(T.equal(t.fields.sub.fields.p, T.REAL));
});

test('nested table: a sub-table column of a different row count is rejected', () => {
  const { errors } = infer(`
    inner = table(p = [1.0, 2.0])
    outer = table(x = [10.0, 20.0, 30.0], sub = inner)
  `);
  assert.ok(errors.length >= 1, 'expected a row-count mismatch diagnostic');
  assert.match(errors[0].message, /length|equal length|rows/i);
});

// =====================================================================
// Runtime values
// =====================================================================

test('nested table: runtime construction nests a __table__ column', () => {
  const t = valueOf('outer', NESTED);
  assert.ok(t && t.__table__ === true);
  assert.equal(t.nrows, 3);
  assert.ok(t.columns.sub && t.columns.sub.__table__ === true,
    'the sub column must itself be a __table__ value');
  assert.equal(t.columns.sub.nrows, 3);
});

test('nested table: runtime column access t.sub returns the sub-table', () => {
  const c = valueOf('c', NESTED + `\n  c = outer.sub`);
  assert.ok(c && c.__table__ === true);
  assert.equal(c.nrows, 3);
});

test('nested table: runtime row access nests the sub-table row as a record', () => {
  // Row 2 (1-based): x = 20, sub = { p: inner.p[2]=2, q: inner.q[2]=5 }.
  const row = valueOf('row', NESTED + `\n  row = outer[2]`);
  assert.deepEqual(row, { x: 20, sub: { p: 2, q: 5 } });
});

test('nested table: runtime broadcast sees nested record rows', () => {
  // row.sub.p + row.x over the 3 rows → [1+10, 2+20, 3+30].
  const out = valueOf('out', NESTED + `
  f = row -> row.sub.p + row.x
  out = f.(outer)`);
  assert.deepEqual(toJS(out), [11, 22, 33]);
});

test('nested table: runtime column-wise sum produces a nested record', () => {
  const s = valueOf('s', NESTED + `\n  s = sum(outer)`);
  assert.deepEqual(s, { x: 60, sub: { p: 6, q: 15 } });
});
