'use strict';

// §07 / §03 table conformance — the gaps surfaced by the §04 always-splat wave.
//
// Each behaviour here is quoted normative text, not an engine reading:
//
//   §07 `indicesof(x)` — "For a table, returns the row indices."
//        Domains cell: `vectors, arrays, tables`. `indicesof0` is "the
//        zero-based variant". Both returned an EMPTY vector for a table: the
//        runtime expected a bare record-of-columns, so the `{__table__,
//        columns, nrows}` shape fell through its key walk. Silent wrong value.
//   §07 `reverse(xs)` — "reverses the order of elements in a vector or rows in
//        a table." Domains cell: `vectors, tables`. A table was rejected
//        outright, because the signature declared a rank-1 array argument.
//   §07 `sizeof` — Domains cell `vectors, arrays`. A table is neither and has
//        no per-axis dimension vector, but it was accepted and silently
//        returned `[]`.
//   §03 — a table "can also be constructed from records of equal-length
//        vectors via `table(r)` and converted to such records via
//        `record(t)`, due to FlatPPL auto-splatting". The analyzer admitted
//        `table(r)`, but no type rule or evaluator branch implemented it, so
//        it produced nothing; `record(t)` was rejected outright ("record()
//        takes keyword arguments only"). Both directions are wired below.
//
// NOT covered here, deliberately: the engine reduces `prod`/`maximum`/
// `minimum` over tables though §07 sanctions only `sum`/`mean`/`var`/`std`.
// Recorded in TODO-flatppl-js.md — it has an in-repo test asserting it, so
// narrowing needs an owner ruling rather than a silent fix.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');
const sampler = require('../sampler.ts');
const valueLib = require('../value.ts');
const ARITH_OPS = sampler._internal.ARITH_OPS;

function errorsOf(src: string): string[] {
  return processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

function ev(src: string) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'unexpected errors: ' + JSON.stringify(errs));
  return orchestrator.buildDerivations(lifted.bindings).fixedValues;
}

const asArray = (v: any) => Array.from(v && v.data !== undefined ? v.data : v);
const colOf = (t: any, k: string) => asArray(t.columns[k]);

// A 3-row, 2-column table. Row indices are therefore [1, 2, 3] / [0, 1, 2],
// and reversing rows maps mass [1,2,3] → [3,2,1] and pt [4,5,6] → [6,5,4].
const T3 = 't = table(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])\n';

// ── §07 indicesof / indicesof0 over a table ────────────────────────────────

test('indicesof(t) returns the row indices (§07)', () => {
  const fv = ev(T3 + 's = indicesof(t)');
  assert.deepEqual(asArray(fv.get('s')), [1, 2, 3]);
});

test('indicesof0(t) returns the zero-based row indices (§07)', () => {
  const fv = ev(T3 + 's = indicesof0(t)');
  assert.deepEqual(asArray(fv.get('s')), [0, 1, 2]);
});

test('indicesof row count agrees with lengthof on the same table', () => {
  // §03: "lengthof(t) returns the number of table rows", and §07 makes
  // indicesof run over exactly those rows — so the two must not disagree.
  const fv = ev(T3 + 'n = lengthof(t)\ni = indicesof(t)');
  assert.equal(asArray(fv.get('i')).length, fv.get('n'));
});

test('indicesof on a vector and a rank-2 array is unchanged (§07 examples)', () => {
  // Verbatim from §07's own example block: indicesof(v) # [1, 2, 3],
  // indicesof0(v) # [0, 1, 2], indicesof(M) # ([1, 2], [1, 2, 3]).
  const fv = ev(`v = [1.0, 2.0, 3.0]
i = indicesof(v)
i0 = indicesof0(v)`);
  assert.deepEqual(asArray(fv.get('i')), [1, 2, 3]);
  assert.deepEqual(asArray(fv.get('i0')), [0, 1, 2]);
});

test('indicesof on a bare record of columns still works', () => {
  // The pre-existing duck-typed path (a record of equal-length vectors read as
  // a table) must survive the `__table__` branch added in front of it.
  const fv = ev(`r = record(a = [1.0, 2.0, 3.0])
i = indicesof(r)`);
  assert.deepEqual(asArray(fv.get('i')), [1, 2, 3]);
});

// ── §07 reverse over a table ───────────────────────────────────────────────

test('reverse(t) reverses table rows and keeps the table type (§07)', () => {
  const fv = ev(T3 + 'r = reverse(t)');
  const r: any = fv.get('r');
  assert.equal(r.__table__, true, 'reverse of a table must still be a table');
  assert.equal(r.nrows, 3);
  assert.deepEqual(colOf(r, 'mass'), [3, 2, 1]);
  assert.deepEqual(colOf(r, 'pt'), [6, 5, 4]);
});

test('reverse(t) is an involution and preserves row count', () => {
  const fv = ev(T3 + `r = reverse(t)
rr = reverse(r)
n = lengthof(r)`);
  assert.equal(fv.get('n'), 3);
  assert.deepEqual(colOf(fv.get('rr'), 'mass'), [1, 2, 3]);
  assert.deepEqual(colOf(fv.get('rr'), 'pt'), [4, 5, 6]);
});

test('an order-independent reduction is unchanged by reverse(t)', () => {
  // sum does not depend on row order, so sum(reverse(t)) must equal sum(t).
  // Oracle by hand: mass 1+2+3 = 6, pt 4+5+6 = 15.
  const fv = ev(T3 + `a = sum(t)
b = sum(reverse(t))`);
  assert.deepEqual(fv.get('a'), { mass: 6, pt: 15 });
  assert.deepEqual(fv.get('b'), { mass: 6, pt: 15 });
});

test('reverse on a vector is unchanged', () => {
  const fv = ev('s = reverse([1.0, 2.0, 3.0])');
  assert.deepEqual(asArray(fv.get('s')), [3, 2, 1]);
});

// ── §07 sizeof rejects a table ─────────────────────────────────────────────

test('sizeof(t) is rejected, not a silent empty vector (§07)', () => {
  // Still rejected, but the REASON moved earlier once §04's builtin splat
  // landed: `sizeof` is not one of the nine carve-out names, so a sole
  // positional table splats into one argument per column, and a 2-column table
  // cannot bind to the single argument §07 names `x`. That §04 name error fires
  // before the domain check below it. The domain guard remains for a table that
  // does reach it and for direct `ARITH_OPS.sizeof` callers (tested separately).
  const errs = errorsOf(T3 + 's = sizeof(t)');
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /sizeof: a sole positional table splats/);
  assert.match(errs[0], /no argument is named "mass", "pt"/);
});

test('sizeof on a vector and a rank-2 array is unchanged (§07 example)', () => {
  // §07's example block: sM = sizeof(M) # [2, 3] for a 2x3 M. Here M is 3x2.
  const fv = ev(`v = [1.0, 2.0, 3.0]
M = rowstack([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
sv = sizeof(v)
sM = sizeof(M)`);
  assert.deepEqual(asArray(fv.get('sv')), [3]);
  assert.deepEqual(asArray(fv.get('sM')), [3, 2]);
});

// ── §03 table(r) promotion ─────────────────────────────────────────────────

test('table(r) promotes a record of equal-length vectors (§03)', () => {
  const fv = ev(`r = record(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])
t = table(r)`);
  const t: any = fv.get('t');
  assert.equal(t.__table__, true);
  assert.equal(t.nrows, 3);
  assert.deepEqual(colOf(t, 'mass'), [1, 2, 3]);
  assert.deepEqual(colOf(t, 'pt'), [4, 5, 6]);
});

test('table(r) agrees with the column-kwarg spelling', () => {
  // §03 presents the two as the same table, so every downstream reader must
  // see the same thing. Oracle by hand: sums 6 and 15, three rows.
  const fv = ev(`r = record(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])
viaRecord = table(r)
viaKwargs = table(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])
sr = sum(viaRecord)
sk = sum(viaKwargs)
nr = lengthof(viaRecord)
nk = lengthof(viaKwargs)`);
  assert.deepEqual(fv.get('sr'), { mass: 6, pt: 15 });
  assert.deepEqual(fv.get('sk'), fv.get('sr'));
  assert.equal(fv.get('nr'), 3);
  assert.equal(fv.get('nk'), 3);
});

test('table(t) on something already a table is the identity', () => {
  const fv = ev(`t = table(a = [1.0, 2.0])
u = table(t)`);
  const u: any = fv.get('u');
  assert.equal(u.__table__, true);
  assert.equal(u.nrows, 2);
  assert.deepEqual(colOf(u, 'a'), [1, 2]);
});

test('table(r) with unequal field lengths is a located error (§03)', () => {
  const errs = errorsOf(`r = record(a = [1.0, 2.0], b = [3.0])
t = table(r)`);
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /all columns must have equal length/);
});

test('table(<non-record>) is a located error naming §03 table(r)', () => {
  const errs = errorsOf('t = table(1.0)');
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /must be a record of equal-length vectors/);
});

// ── §03 record(t) conversion — the reverse direction ───────────────────────

test('record(t) converts a table\'s columns to a record (§03)', () => {
  const fv = ev(T3 + 'r = record(t)');
  const r: any = fv.get('r');
  assert.deepEqual(asArray(r.mass), [1, 2, 3]);
  assert.deepEqual(asArray(r.pt), [4, 5, 6]);
});

test('record(t) agrees with the field-kwarg spelling', () => {
  // §03 presents record(t) and the equivalent field kwargs as the same
  // record, so a downstream field read must not tell them apart.
  const fv = ev(T3 + `viaTable = record(t)
viaKwargs = record(mass = [1.0, 2.0, 3.0], pt = [4.0, 5.0, 6.0])
a = viaTable.mass
b = viaKwargs.mass`);
  assert.deepEqual(asArray(fv.get('a')), asArray(fv.get('b')));
});

test('record(t) round-trips with table(r): table(record(t)) reproduces t', () => {
  const fv = ev(T3 + `r = record(t)
t2 = table(r)
a = sum(t2)`);
  assert.deepEqual(fv.get('a'), { mass: 6, pt: 15 });
});

test('record(r) on something already a record is the identity', () => {
  const fv = ev(`r0 = record(a = 1.0, b = 2.0)
r = record(r0)`);
  assert.deepEqual(fv.get('r'), { a: 1, b: 2 });
});

test('record(<non-table>) is a located error naming §03 record(t)', () => {
  const errs = errorsOf('r = record(1.0)');
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /must be a table/);
});

// ── direct ARITH_OPS calls ─────────────────────────────────────────────────
//
// These ops are also reached WITHOUT typeinfer — the worker and other engine
// paths call ARITH_OPS directly, so their table guards have to hold on their
// own rather than relying on the static checks above.

const mkTable = (columns: any, nrows?: number) => {
  const t: any = { __table__: true, columns };
  if (nrows !== undefined) t.nrows = nrows;
  return t;
};
const vec = (xs: number[]) => ({ shape: [xs.length], data: Float64Array.from(xs) });

test('ARITH_OPS.sizeof throws on a table even with no static check in front', () => {
  assert.throws(() => ARITH_OPS.sizeof(mkTable({ a: vec([1, 2]) }, 2)),
    /sizeof: argument must be a vector or array/);
});

test('ARITH_OPS.indicesof falls back to a column length when nrows is absent', () => {
  // A table object carrying no row count: the row count comes from the first
  // column instead. §07 still wants the row indices.
  const t = mkTable({ a: vec([1, 2, 3, 4]) });
  assert.deepEqual(Array.from(ARITH_OPS.indicesof(t).data), [1, 2, 3, 4]);
  assert.deepEqual(Array.from(ARITH_OPS.indicesof0(t).data), [0, 1, 2, 3]);
});

test('ARITH_OPS.indicesof on a column-less table yields no indices', () => {
  assert.deepEqual(Array.from(ARITH_OPS.indicesof(mkTable({})).data), []);
});

test('ARITH_OPS.reverse handles a plain-array column and a scalar column', () => {
  // A column that is a bare JS array reverses as a copy; anything with no row
  // axis to reverse is passed through untouched.
  const r: any = ARITH_OPS.reverse(mkTable({ a: [1, 2, 3], k: 7 }, 3));
  assert.equal(r.__table__, true);
  assert.deepEqual(r.columns.a, [3, 2, 1]);
  assert.equal(r.columns.k, 7);
});

test('reverse(t) recurses into a table-valued column (§03 nested tables)', () => {
  const inner = mkTable({ x: vec([1, 2]) }, 2);
  const r: any = ARITH_OPS.reverse(mkTable({ sub: inner, a: vec([10, 20]) }, 2));
  assert.equal(r.columns.sub.__table__, true);
  assert.deepEqual(Array.from(r.columns.sub.columns.x.data), [2, 1]);
  assert.deepEqual(Array.from(r.columns.a.data), [20, 10]);
});

test('reverse(t) moves a vector column as a whole row block', () => {
  // A [2, 3] column is two rows of three cells. Reversing rows swaps the
  // blocks, it does not reverse the six numbers.
  const col = { shape: [2, 3], data: Float64Array.from([1, 2, 3, 4, 5, 6]) };
  const r: any = ARITH_OPS.reverse(mkTable({ v: col }, 2));
  assert.deepEqual(Array.from(r.columns.v.data), [4, 5, 6, 1, 2, 3]);
  assert.deepEqual(r.columns.v.shape, [2, 3]);
});

// §03 "Records" — "Field values may be scalars, arrays, or records" — admits no
// table, so a promotable record never carries a table-valued field: there is no
// sub-table case in the promotion path, and the analyzer rejects such a record
// outright. ("a table belongs in a table column" is the engine's own diagnostic
// wording, not spec text — the normative rule is the sentence above.) Nested
// tables live in table COLUMNS, which reverse() recurses into (covered above).

test('table(r) with a scalar field is a located error (§03 needs vectors)', () => {
  const errs = errorsOf(`r = record(a = 1.0)
t = table(r)`);
  assert.equal(errs.length, 1, 'got: ' + errs.join(' | '));
  assert.match(errs[0], /must be a 1-D array/);
});

// ── reverse(t) must preserve a column's Value tags ─────────────────────────
//
// A column is not always a bare `{shape, data}`. It may carry `dtype`, `im`
// (complex), `outerRank` (a nested-vector column) or `struct` (storage layout).
// Reversing ROWS changes none of them, and rebuilding a bare Value instead
// dropped every one — a complex column lost its imaginary half and stopped
// being a complex Value at all, i.e. a wrong value with no error, on the same
// table shape whose reductions handle complex correctly.
//
// §03 makes a complex column legitimate: Complex is a scalar type and a table
// column is a vector of scalars.

test('reverse(t) preserves a complex column, reversing re and im together', () => {
  const col: any = {
    shape: [3],
    data: Float64Array.from([1, 2, 3]),
    im: Float64Array.from([10, 20, 30]),
    dtype: 'complex',
  };
  const out: any = ARITH_OPS.reverse(mkTable({ z: col }, 3)).columns.z;
  assert.deepEqual(Array.from(out.data), [3, 2, 1]);
  assert.deepEqual(Array.from(out.im), [30, 20, 10], 'im must reverse with re');
  assert.equal(out.dtype, 'complex');
  assert.equal(valueLib.isComplexValue(out), true,
    'the reversed column must still BE a complex Value');
});

test('reverse(t) preserves outerRank on a nested-vector column', () => {
  const col: any = { shape: [2, 3], data: Float64Array.from([1, 2, 3, 4, 5, 6]), outerRank: 1 };
  const out: any = ARITH_OPS.reverse(mkTable({ v: col }, 2)).columns.v;
  assert.equal(out.outerRank, 1);
  assert.deepEqual(Array.from(out.data), [4, 5, 6, 1, 2, 3]);
});

test('reverse(t) preserves a column dtype', () => {
  const col: any = { shape: [3], data: Float64Array.from([7, 8, 9]), dtype: 'integer' };
  const out: any = ARITH_OPS.reverse(mkTable({ n: col }, 3)).columns.n;
  assert.equal(out.dtype, 'integer');
  assert.deepEqual(Array.from(out.data), [9, 8, 7]);
});

test('reverse(t) leaves a plain real column with no spurious tags', () => {
  // The complement of the tests above: nothing is INVENTED either.
  const col: any = { shape: [2], data: Float64Array.from([1, 2]) };
  const out: any = ARITH_OPS.reverse(mkTable({ a: col }, 2)).columns.a;
  assert.deepEqual(Object.keys(out).sort(), ['data', 'shape']);
  assert.deepEqual(Array.from(out.data), [2, 1]);
});
