'use strict';

// measureToPerAtomRecords over a RECORD-OF-RECORDS measure — the density-side
// feeding contract for a reified kernel whose single parametric input is a
// whole record (clm.feedInputs whole-record param / matWeighted record base).
//
// The per-field accessors already recurse into a nested-record field, but the
// atom-count (N) scan only inspected `.samples` (scalar leaf) / `.value`
// (vector field). When EVERY top-level field is itself a record (a
// record-of-records, newly expressible now that nested records are blessed),
// no field exposes a length, N stays 0, and the function returned an empty
// array — feeding ZERO per-atom records and silently producing NaN/wrong
// likelihood downstream. N must also be taken from a nested-record field.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { measureToPerAtomRecords } = require('../materialiser-shared.ts');

const f64 = (a: number[]) => Float64Array.from(a);

test('record-of-records: per-atom records use the nested records’ atom count (not 0)', () => {
  // m.g = record(p, q), m.h = record(r) — every top-level field is a record.
  const m = {
    fields: {
      g: { fields: { p: { samples: f64([1, 2, 3, 4]) }, q: { samples: f64([10, 20, 30, 40]) } } },
      h: { fields: { r: { samples: f64([9, 8, 7, 6]) } } },
    },
  };
  const out = measureToPerAtomRecords(m, 'm', 'test');
  assert.equal(out.length, 4, 'atom count must come from the nested records, not stay 0');
  assert.deepEqual(out[0], { g: { p: 1, q: 10 }, h: { r: 9 } });
  assert.deepEqual(out[2], { g: { p: 3, q: 30 }, h: { r: 7 } });
});

test('record-of-records: a mixed record (one scalar field) keeps working', () => {
  const m = {
    fields: {
      a: { fields: { p: { samples: f64([1, 2, 3, 4]) } } },
      mu: { samples: f64([100, 200, 300, 400]) },
    },
  };
  const out = measureToPerAtomRecords(m, 'm', 'test');
  assert.equal(out.length, 4);
  assert.deepEqual(out[2], { a: { p: 3 }, mu: 300 });
});

test('record-of-records: three levels deep', () => {
  const m = {
    fields: {
      outer: { fields: { mid: { fields: { leaf: { samples: f64([5, 6, 7]) } } } } },
    },
  };
  const out = measureToPerAtomRecords(m, 'm', 'test');
  assert.equal(out.length, 3);
  assert.deepEqual(out[1], { outer: { mid: { leaf: 6 } } });
});
