'use strict';

// Spec §07 load_data parsers — JSON / CSV / WSV.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const dataload = require('../dataload.ts');

// =====================================================================
// detectFormat
// =====================================================================

test('detectFormat: file extension determines parser', () => {
  assert.equal(dataload.detectFormat('foo.json'), 'json');
  assert.equal(dataload.detectFormat('foo.csv'), 'csv');
  assert.equal(dataload.detectFormat('foo.wsv'), 'wsv');
  assert.equal(dataload.detectFormat('FOO.JSON'), 'json');   // case-insensitive
  assert.equal(dataload.detectFormat('arrow.arrows'), null,  // not supported
    'Arrow deferred per TODO');
  assert.equal(dataload.detectFormat('no-ext'), null);
});

// =====================================================================
// JSON: array of scalars ⇒ vector
// =====================================================================

test('parseJSON: array of scalars ⇒ rank-1 Value', () => {
  const r = dataload.parseJSON('[1.0, 2.0, 3.0]');
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [1, 2, 3]);
});

test('parseJSON: empty array ⇒ empty vector', () => {
  const r = dataload.parseJSON('[]');
  assert.deepEqual(r.shape, [0]);
  assert.equal(r.data.length, 0);
});

// =====================================================================
// JSON: array of objects ⇒ table (struct of columns)
// =====================================================================

test('parseJSON: array of objects ⇒ table with named columns', () => {
  const r = dataload.parseJSON('[{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]');
  assert.ok(r.a && r.b, 'table has columns a and b');
  assert.deepEqual(Array.from(r.a.data), [1, 3, 5]);
  assert.deepEqual(Array.from(r.b.data), [2, 4, 6]);
});

// =====================================================================
// JSON: object of arrays ⇒ table
// =====================================================================

test('parseJSON: object of arrays ⇒ table', () => {
  const r = dataload.parseJSON('{"a": [1, 2, 3], "b": [4, 5, 6]}');
  assert.ok(r.a && r.b);
  assert.deepEqual(Array.from(r.a.data), [1, 2, 3]);
  assert.deepEqual(Array.from(r.b.data), [4, 5, 6]);
});

test('parseJSON: object-of-arrays with unequal columns → error', () => {
  assert.throws(() => dataload.parseJSON('{"a": [1, 2], "b": [3]}'),
    /unequal length/);
});

// =====================================================================
// CSV
// =====================================================================

test('parseCSV: 2-column table with header', () => {
  const r = dataload.parseCSV(`a,b
1,2
3,4
5,6
`);
  assert.deepEqual(Array.from(r.a.data), [1, 3, 5]);
  assert.deepEqual(Array.from(r.b.data), [2, 4, 6]);
});

test('parseCSV: single-column file ⇒ vector', () => {
  // Per spec §07 — single-column CSV can load as a vector.
  const r = dataload.parseCSV(`x
1.5
2.5
3.5
`);
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [1.5, 2.5, 3.5]);
});

test('parseCSV: blank and comment lines are skipped', () => {
  const r = dataload.parseCSV(`# a header comment
a,b
1,2

3,4
# trailing comment
5,6
`);
  assert.deepEqual(Array.from(r.a.data), [1, 3, 5]);
  assert.deepEqual(Array.from(r.b.data), [2, 4, 6]);
});

test('parseCSV: quoted cells with embedded quotes', () => {
  const r = dataload.parseCSV(`a,b
"1","2"
"3","4"
`);
  assert.deepEqual(Array.from(r.a.data), [1, 3]);
  assert.deepEqual(Array.from(r.b.data), [2, 4]);
});

// =====================================================================
// WSV (whitespace-separated)
// =====================================================================

test('parseWSV: whitespace-separated table', () => {
  const r = dataload.parseWSV(`a   b   c
1   2   3
4   5   6
`);
  assert.deepEqual(Array.from(r.a.data), [1, 4]);
  assert.deepEqual(Array.from(r.b.data), [2, 5]);
  assert.deepEqual(Array.from(r.c.data), [3, 6]);
});

test('parseWSV: tab-separated also works (any whitespace is a separator)', () => {
  const r = dataload.parseWSV(`x\ty\n1\t2\n3\t4\n`);
  assert.deepEqual(Array.from(r.x.data), [1, 3]);
  assert.deepEqual(Array.from(r.y.data), [2, 4]);
});

// =====================================================================
// parseByExtension dispatch
// =====================================================================

test('parseByExtension: dispatches by file ext', () => {
  const json = dataload.parseByExtension('data.json', '[1, 2, 3]');
  assert.deepEqual(Array.from(json.data), [1, 2, 3]);
  const csv = dataload.parseByExtension('data.csv', 'x\n1\n2\n3\n');
  assert.deepEqual(Array.from(csv.data), [1, 2, 3]);
});

test('parseByExtension: unsupported format ⇒ error', () => {
  assert.throws(() => dataload.parseByExtension('data.arrows', '...'),
    /unsupported file extension/);
});
