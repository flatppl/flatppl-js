'use strict';

// Tables — spec §03 §07 §11. Pins:
//
//   - `table(col=...)` constructs a first-class table value (a
//     `__table__`-marked object carrying the column-name → array
//     map and the explicit row count). Distinct from records,
//     which records are disallowed in broadcast.
//
//   - Indexing: `t[i]` returns a record over the column names
//     (row access); `t.col` returns the column as a vector
//     (column access).
//
//   - `lengthof(t)` returns the row count.
//
//   - Column-wise reductions per spec §07: `sum / mean / var /
//     std / prod / maximum / minimum` applied to a table return a
//     record whose fields are the column names.
//
//   - Broadcast over a table (spec §04): outer axis = nrows, per
//     cell the callable receives a record built from the column
//     values at that row.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const T = require('../types.ts');
const { toJS } = require('./_value-helpers.ts');

function infer(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  return { bindings: r.bindings, errors: errs, loweredModule: r.loweredModule };
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

// =====================================================================
// Type inference
// =====================================================================

test('table: literal construction infers table(columns, nrows)', () => {
  const { bindings, errors } = infer(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'events');
  assert.equal(t.kind, 'table');
  assert.equal(t.nrows, 3);
  assert.ok(T.equal(t.columns.mass, T.REAL));
  assert.ok(T.equal(t.columns.pt, T.REAL));
});

test('table: column-length mismatch is a typeinfer diagnostic', () => {
  const { errors } = infer(`
    bad = table(a = [1.0, 2.0], b = [3.0, 4.0, 5.0])
  `);
  assert.ok(errors.length >= 1,
    'expected a column-length-mismatch diagnostic; got ' + errors.length);
  assert.match(errors[0].message, /column.*length|equal length/i);
});

test('table: column access (t.col) returns array of column element type', () => {
  const { bindings, errors } = infer(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
    col = events.mass
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'col');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('table: row access (t[i]) returns record over column names', () => {
  const { bindings, errors } = infer(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
    row = events[1]
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'row');
  assert.equal(t.kind, 'record');
  assert.ok('mass' in t.fields);
  assert.ok('pt' in t.fields);
  assert.ok(T.equal(t.fields.mass, T.REAL));
});

test('table: lengthof(t) returns integer', () => {
  const { bindings, errors } = infer(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
    n = lengthof(events)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'n');
  assert.equal(t.kind, 'scalar');
  assert.equal(t.prim, 'integer');
});

// lengthof(t)/lengthof(<vec-of-vec>) must const-fold to the row/outer
// count in a SHAPE position (engine-concepts §17.1), so a dependent
// shape stays concrete instead of going %dynamic. The const-eval
// short-circuit reads array `shape[0]` and now table `nrows` uniformly
// (resolveIntegerShape + resolveIntegerVectorShape).
function shapeOf(src: string, name: string) {
  const r = processSource(src);
  const lb = r.loweredModule.bindings.get(name);
  return lb && lb.inferredType;
}

test('table: lengthof(t) const-folds the row count into a scalar shape position', () => {
  const t = shapeOf(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
    g = eye(lengthof(events))
  `, 'g');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3, 3]);
});

test('table: lengthof(t) const-folds into a vector shape position', () => {
  const t = shapeOf(`
    events = table(mass = [1.1, 1.2, 1.3, 1.4])
    z = zeros(lengthof(events))
  `, 'z');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [4]);
});

test('lengthof(<vec-of-vec>) const-folds to the OUTER length in a shape position', () => {
  // Spec §07: lengthof is the number of (outer) elements — a 2×3 matrix
  // literal has length 2, NOT 6 (the nested/flat distinction is kept).
  const t = shapeOf(`
    M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    z = zeros(lengthof(M))
  `, 'z');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [2]);
});

test('table: sum / mean return record of per-column reductions', () => {
  const { bindings, errors } = infer(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
    s = sum(events)
    m = mean(events)
  `);
  assert.equal(errors.length, 0);
  const ts = typeOf(bindings, 's'), tm = typeOf(bindings, 'm');
  assert.equal(ts.kind, 'record');
  assert.equal(tm.kind, 'record');
  assert.ok('mass' in ts.fields && 'pt' in ts.fields);
});

test('table: var / std return record-of-reals (Bessel correction makes the result real)', () => {
  const { bindings, errors } = infer(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
    v = var(events)
    s = std(events)
  `);
  assert.equal(errors.length, 0);
  for (const name of ['v', 's']) {
    const t = typeOf(bindings, name);
    assert.equal(t.kind, 'record');
    assert.ok(T.equal(t.fields.mass, T.REAL));
  }
});

test('table: broadcast over table — outer rank = nrows, cell = record', () => {
  // Spec §04: "When a table is passed to broadcast, it is traversed
  // row-wise and each row treated as a record passed to the function".
  // The fn here takes a record and computes mass + pt; broadcast over
  // a 3-row table produces a length-3 array of real.
  const { bindings, errors } = infer(`
    events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
    rowsum = row -> row.mass + row.pt
    out = rowsum.(events)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'out');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

// =====================================================================
// Runtime values
// =====================================================================

test('table: runtime — table(...) produces a marked table value', () => {
  const t = valueOf('events', `events = table(a = [1.0, 2.0], b = [3.0, 4.0])`);
  assert.ok(t && t.__table__ === true,
    'expected __table__-marked value; got ' + JSON.stringify(t));
  assert.equal(t.nrows, 2);
  assert.ok('a' in t.columns && 'b' in t.columns);
});

test('table: runtime — t.col returns the column', () => {
  const col = valueOf('col', `events = table(a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0])
col = events.a
`);
  // Column is whatever shape the construction value had — a Value
  // or a JS array. Either way, contents = [1, 2, 3].
  const arr = col && col.data ? Array.from(col.data) : col;
  assert.deepEqual(arr, [1, 2, 3]);
});

test('table: runtime — t[i] returns a record-of-column-values', () => {
  const row = valueOf('row', `events = table(a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0])
row = events[2]
`);
  assert.deepEqual(row, { a: 2, b: 5 });
});

test('table: runtime — lengthof returns the row count', () => {
  const n = valueOf('n', `events = table(a = [1.0, 2.0, 3.0])
n = lengthof(events)
`);
  assert.equal(n, 3);
});

test('table: runtime — sum applies column-wise → record', () => {
  const s = valueOf('s', `events = table(a = [1.0, 2.0, 3.0], b = [10.0, 20.0, 30.0])
s = sum(events)
`);
  assert.deepEqual(s, { a: 6, b: 60 });
});

test('table: runtime — mean / var / std applied per column', () => {
  const r = processSource(`events = table(a = [1.0, 2.0, 3.0, 4.0], b = [10.0, 10.0, 10.0, 10.0])
m = mean(events)
v = var(events)
s = std(events)
`);
  const ds = buildDerivations(r.bindings);
  const m = ds.fixedValues.get('m'), v = ds.fixedValues.get('v'), s = ds.fixedValues.get('s');
  assert.ok(Math.abs(m.a - 2.5) < 1e-9);
  assert.ok(Math.abs(m.b - 10.0) < 1e-9);
  // Bessel-corrected sample variance of [1, 2, 3, 4]: 5/3 ≈ 1.667.
  assert.ok(Math.abs(v.a - (5 / 3)) < 1e-9);
  assert.ok(Math.abs(v.b - 0.0) < 1e-9);
  assert.ok(Math.abs(s.a - Math.sqrt(5 / 3)) < 1e-9);
});

test('table: runtime — broadcast (rowsum) operates row-wise', () => {
  const r = processSource(`events = table(mass = [1.1, 1.2, 1.3], pt = [45.2, 32.1, 67.8])
rowsum = row -> row.mass + row.pt
out = rowsum.(events)
`);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    'expected clean parse; got ' + errs.map((e: any) => e.message).join(' | '));
  const ds = buildDerivations(r.bindings);
  const out = ds.fixedValues.get('out');
  const arr: any = toJS(out);
  assert.equal(arr.length, 3);
  assert.ok(Math.abs(arr[0] - (1.1 + 45.2)) < 1e-9);
  assert.ok(Math.abs(arr[1] - (1.2 + 32.1)) < 1e-9);
  assert.ok(Math.abs(arr[2] - (1.3 + 67.8)) < 1e-9);
});

// =====================================================================
// Spec §04 — records are NOT allowed in broadcast (tables are)
// =====================================================================

test('table: a plain record (not a table) is still rejected in broadcast (spec §04)', () => {
  // Records and tuples are explicitly disallowed as broadcast inputs.
  // The runtime classifyAxisStructure throws when it sees a record
  // input. Tables, marked distinctly, succeed.
  const r = processSource(`r = record(a = 1.0, b = 2.0)
f = (x) -> x + 1.0
out = f.(r)
`);
  // The error surfaces at the runtime broadcast call — record is
  // detected and rejected. The binding `out` doesn't make it to
  // fixedValues.
  const ds = buildDerivations(r.bindings);
  assert.ok(!ds.fixedValues.has('out'),
    'broadcast over a plain record must not produce a value; '
    + 'got ' + JSON.stringify(ds.fixedValues.get('out')));
});
