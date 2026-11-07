'use strict';

// The measure of a fixed table. §06 makes the law of a fixed value the point
// mass at it (`Dirac(value = v)` "for any variate type"), and §03 makes a table
// a value whose fields are its columns. The materialiser read the internal
// `{__table__, columns, nrows}` object as a record, so the measure's fields
// were `columns` and `nrows`. It is now the table measure an iid of a record
// measure already builds, with the table as its value and the columns as fields.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const BASE = 't = table(a = [1.0, 2.0, 3.0], b = [4.0, 5.0, 6.0])\n';

async function measure(src: string, name: string) {
  const { ctx } = makeMatCtx(BASE + src, { sampleCount: 4 });
  return ctx.getMeasure(name);
}

function assertTableMeasure(m: any) {
  assert.equal(m.shape, 'table');
  assert.equal(m.value.__table__, true);
  assert.equal(m.nrows, 3);
  assert.deepEqual(Object.keys(m.fields), ['a', 'b'], 'the fields are the columns');
  assert.deepEqual(Array.from(m.fields.a.value.data), [1, 2, 3]);
  assert.deepEqual(Array.from(m.fields.b.value.data), [4, 5, 6]);
}

for (const [label, src, name] of [
  ['a table binding', '', 't'],
  ['the law of a table', 'L = lawof(t)\n', 'L'],
  ['a draw from a point mass at a table', 's = rnginit(3)\nx, s2 = rand(Dirac(value = t), s)\n', 'x'],
] as [string, string, string][]) {
  test(`${label} materialises as a table measure`, async () => {
    assertTableMeasure(await measure(src, name));
  });
}

test('a nested table column is a table measure too', async () => {
  const m = await measure('u = table(a = [1.0, 2.0], s = table(x = [5.0, 6.0]))\n', 'u');
  assert.equal(m.shape, 'table');
  assert.deepEqual(Object.keys(m.fields), ['a', 's']);
  assert.equal(m.fields.s.shape, 'table');
  assert.deepEqual(Array.from(m.fields.s.fields.x.value.data), [5, 6]);
});

test('the point mass at a table still scores and maps', async () => {
  // log Dirac(t)(t) = 0, and sum(t.a) = 6 on every atom.
  assert.deepEqual(Array.from((await measure('lp = logdensityof(Dirac(value = t), t)\n', 'lp')).samples), [0, 0, 0, 0]);
  assert.deepEqual(Array.from((await measure('P = pushfwd(fn(sum(_.a)), Dirac(value = t))\n', 'P')).samples), [6, 6, 6, 6]);
});
