'use strict';

// Vector-per-entry table columns — spec §03: "a vector column's elements may
// themselves be arrays (e.g. a 3-vector per entry)". Such a column is typed as
// a per-row vector (t[i].col : array) and stored at runtime as a Value with a
// leading ROW axis and trailing CELL axes (e.g. [N, 3]). The bug under repair:
// row extraction (valueLib.tableRow) and column-wise reductions indexed/folded
// the FLAT buffer, so t[i].col returned a scalar (not the per-row vector) and
// sum/mean/... flattened the whole column to a scalar — both contradicting the
// (correct) inferred type. Row-access and column-access (t.col[i]) disagreed
// silently with no error.
//
//   t = table(p = [[1,2,3],[4,5,6]], w = [10,20])
//     t.p          -> Value [2,3]              (column-first; already correct)
//     t.p[1]       -> [1,2,3]                  (already correct)
//     t[1].p       -> [1,2,3]   (was: 1)       (row-first; the bug)
//     sum(t).p     -> [5,7,9]   (was: 21)      (element-wise over rows)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const { toJS } = require('./_value-helpers.ts');

function valueOf(name: string, src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  if (errs.length > 0) throw new Error('source had errors: ' + errs.map((e: any) => e.message).join(' | '));
  const ds = buildDerivations(r.bindings);
  return ds.fixedValues && ds.fixedValues.get(name);
}
// A row field may be a scalar or a Value (per-row vector). Normalise to JS.
function asJS(v: any) { return (v && v.data) ? Array.from(v.data) : v; }

const TBL = 't = table(p = [[1.0,2.0,3.0],[4.0,5.0,6.0]], w = [10.0, 20.0])\n';

// =====================================================================
// Row access — the per-row vector, not a flat scalar
// =====================================================================

test('vector column: t[i].col returns the per-row vector (not a scalar)', () => {
  assert.deepEqual(asJS(valueOf('x', TBL + 'x = t[1].p')), [1, 2, 3]);
  assert.deepEqual(asJS(valueOf('x', TBL + 'x = t[2].p')), [4, 5, 6]);
});

test('vector column: row-first and column-first access agree (t[i].col == t.col[i])', () => {
  const rowFirst = asJS(valueOf('x', TBL + 'x = t[1].p'));
  const colFirst = asJS(valueOf('y', TBL + 'y = t.p[1]'));
  assert.deepEqual(rowFirst, colFirst);
  assert.deepEqual(rowFirst, [1, 2, 3]);
});

test('vector column: the scalar sibling column is still a scalar per row', () => {
  assert.equal(asJS(valueOf('x', TBL + 'x = t[2].w')), 20);
});

// =====================================================================
// Broadcast over the table sees the per-row vector
// =====================================================================

test('vector column: broadcast row-wise sees the per-row vector', () => {
  // f(row) = sum(row.p) + row.w  ->  row1: (1+2+3)+10=16 ; row2: (4+5+6)+20=35
  const out = valueOf('out', TBL + 'f = row -> sum(row.p) + row.w\nout = f.(t)');
  assert.deepEqual(toJS(out), [16, 35]);
});

// =====================================================================
// Column-wise reductions reduce over the ROW axis, preserving cells
// =====================================================================

test('vector column: sum/mean reduce element-wise over rows -> a vector', () => {
  const s = valueOf('s', TBL + 's = sum(t)');
  assert.deepEqual(asJS(s.p), [5, 7, 9]);   // [1+4, 2+5, 3+6]
  assert.equal(asJS(s.w), 30);
  const m = valueOf('m', TBL + 'm = mean(t)');
  assert.deepEqual(asJS(m.p), [2.5, 3.5, 4.5]);
  assert.equal(asJS(m.w), 15);
});

test('vector column: maximum/minimum reduce element-wise over rows', () => {
  const mx = valueOf('mx', TBL + 'mx = maximum(t)');
  assert.deepEqual(asJS(mx.p), [4, 5, 6]);
  const mn = valueOf('mn', TBL + 'mn = minimum(t)');
  assert.deepEqual(asJS(mn.p), [1, 2, 3]);
});

test('vector column: var/std reduce element-wise over rows (Bessel)', () => {
  // per cell, 2 rows: var = ((x0-mu)^2+(x1-mu)^2)/(2-1). cell0: (1,4)->mu 2.5->4.5
  const v = valueOf('v', TBL + 'v = var(t)');
  assert.deepEqual(asJS(v.p), [4.5, 4.5, 4.5]);
  const sd = valueOf('sd', TBL + 'sd = std(t)');
  const sp = asJS(sd.p);
  sp.forEach((x: number) => assert.ok(Math.abs(x - Math.sqrt(4.5)) < 1e-12));
});

// =====================================================================
// Nested: a vector-per-entry column inside a sub-table column
// =====================================================================

test('vector column inside a sub-table: deep row extraction returns the vector', () => {
  const src = `
    inner = table(mom = [[1.0,2.0,3.0],[4.0,5.0,6.0]], e = [9.0, 8.0])
    outer = table(id = [100.0, 200.0], part = inner)
    row = outer[1]
  `;
  const row = valueOf('row', src);
  assert.equal(asJS(row.id), 100);
  assert.deepEqual(asJS(row.part.mom), [1, 2, 3]);
  assert.equal(asJS(row.part.e), 9);
});
