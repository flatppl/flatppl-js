'use strict';

// Spec §07 "NaN inputs" (docs/07-functions.md): a NaN propagates through
// the order operations `max`, `min`, `maximum`, `minimum`, `cummax`,
// `cummin`, `linfnorm`, `median`, and `quantile`.
//
// Expected values are the numpy oracle (np.max/np.min/np.median/
// np.percentile(method='linear')/np.maximum.accumulate/np.minimum.accumulate),
// which propagates NaN by default — checked directly, not by intuition.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const sampler = require('../sampler.ts');

const ARITH_OPS = sampler._internal.ARITH_OPS;

function ev(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'unexpected errors: ' + JSON.stringify(errs));
  return buildDerivations(r.bindings).fixedValues;
}

function vecVal(xs: number[]) {
  return { shape: [xs.length], data: Float64Array.from(xs) };
}

// Position classes: NaN as the first element, the last, an interior
// element, every element, and the sole element of a length-1 array.
const POSITIONS: [string, number[]][] = [
  ['first',    [NaN, 1, 2]],
  ['last',     [1, 2, NaN]],
  ['interior', [1, NaN, 2]],
  ['all_nan',  [NaN, NaN]],
  ['scalar',   [NaN]],
];

// =====================================================================
// max / min — binary scalar ops, distinct from maximum / minimum
// =====================================================================

test('max/min: NaN in either argument propagates', () => {
  assert.ok(Number.isNaN(ARITH_OPS.max(1, NaN)));
  assert.ok(Number.isNaN(ARITH_OPS.max(NaN, 1)));
  assert.ok(Number.isNaN(ARITH_OPS.min(1, NaN)));
  assert.ok(Number.isNaN(ARITH_OPS.min(NaN, 1)));
});

test('max/min: NaN propagates through the full source path', () => {
  const fv = ev('a = max(1.0, 0.0 / 0.0)\nb = min(1.0, 0.0 / 0.0)');
  assert.ok(Number.isNaN(fv.get('a')));
  assert.ok(Number.isNaN(fv.get('b')));
});

// =====================================================================
// maximum / minimum — array reductions
// =====================================================================

for (const [label, xs] of POSITIONS) {
  test(`maximum([${xs}]) [${label}] = NaN`, () => {
    assert.ok(Number.isNaN(ARITH_OPS.maximum(vecVal(xs))));
  });
  test(`minimum([${xs}]) [${label}] = NaN`, () => {
    assert.ok(Number.isNaN(ARITH_OPS.minimum(vecVal(xs))));
  });
}

test('maximum / minimum over a table column propagate a NaN cell', () => {
  const fv = ev(`
t = table(x = [3.0, 0.0 / 0.0, 5.0])
mx = maximum(t)
mn = minimum(t)
`);
  const mx: any = fv.get('mx');
  const mn: any = fv.get('mn');
  assert.ok(Number.isNaN(mx.x));
  assert.ok(Number.isNaN(mn.x));
});

test('maximum / minimum over a vector-per-row table column propagate NaN', () => {
  // A vector-per-entry column ([N, cellLen]) routes through _reduceRowAxis,
  // a separate hand-rolled loop from the bare ARITH_OPS.maximum/minimum.
  const fv = ev(`
t = table(x = [[1.0, 2.0], [0.0 / 0.0, 4.0]])
mx = maximum(t)
mn = minimum(t)
`);
  const mx: any = fv.get('mx');
  const mn: any = fv.get('mn');
  assert.ok(Number.isNaN(mx.x.data[0]));
  assert.equal(mx.x.data[1], 4);
  assert.ok(Number.isNaN(mn.x.data[0]));
  assert.equal(mn.x.data[1], 2);
});

// =====================================================================
// cummax / cummin — running extrema, per-position expected VECTORS
// (numpy np.maximum.accumulate / np.minimum.accumulate oracle)
// =====================================================================

const CUMMAX_CASES: [string, number[], number[]][] = [
  ['first',    [NaN, 1, 2], [NaN, NaN, NaN]],
  ['last',     [1, 2, NaN], [1, 2, NaN]],
  ['interior', [1, NaN, 2], [1, NaN, NaN]],
  ['all_nan',  [NaN, NaN],  [NaN, NaN]],
  ['scalar',   [NaN],       [NaN]],
];
const CUMMIN_CASES: [string, number[], number[]][] = [
  ['first',    [NaN, 1, 2], [NaN, NaN, NaN]],
  ['last',     [1, 2, NaN], [1, 1, NaN]],
  ['interior', [1, NaN, 2], [1, NaN, NaN]],
  ['all_nan',  [NaN, NaN],  [NaN, NaN]],
  ['scalar',   [NaN],       [NaN]],
];

function assertArrNaNAware(got: ArrayLike<number>, want: number[]) {
  assert.equal(got.length, want.length);
  for (let i = 0; i < want.length; i++) {
    if (Number.isNaN(want[i])) assert.ok(Number.isNaN(got[i]), `index ${i}: expected NaN, got ${got[i]}`);
    else assert.equal(got[i], want[i]);
  }
}

for (const [label, xs, want] of CUMMAX_CASES) {
  test(`cummax([${xs}]) [${label}]`, () => {
    assertArrNaNAware(ARITH_OPS.cummax(vecVal(xs)).data, want);
  });
}
for (const [label, xs, want] of CUMMIN_CASES) {
  test(`cummin([${xs}]) [${label}]`, () => {
    assertArrNaNAware(ARITH_OPS.cummin(vecVal(xs)).data, want);
  });
}

// =====================================================================
// linfnorm — real and complex vectors
// =====================================================================

for (const [label, xs] of POSITIONS) {
  test(`linfnorm([${xs}]) [${label}] = NaN`, () => {
    assert.ok(Number.isNaN(ARITH_OPS.linfnorm(vecVal(xs))));
  });
}

test('linfnorm: a NaN real part in a complex vector propagates', () => {
  const z = [{ re: NaN, im: 0 }, { re: 1, im: 1 }];
  assert.ok(Number.isNaN(ARITH_OPS.linfnorm(z)));
});

// =====================================================================
// median — including the regression case: NaN interior to a sorted
// position that a post-sort scan (default sort pushes NaN to the end)
// would otherwise miss
// =====================================================================

for (const [label, xs] of POSITIONS) {
  test(`median([${xs}]) [${label}] = NaN`, () => {
    assert.ok(Number.isNaN(ARITH_OPS.median(vecVal(xs))));
  });
}

test('median: NaN not at the sorted-median index still propagates', () => {
  // Sorted (NaN pushed to the end): [1, 3, 4, NaN] → mean(s[1], s[2]) = 3.5
  // if the NaN check ran post-sort. Spec requires NaN regardless.
  assert.ok(Number.isNaN(ARITH_OPS.median(vecVal([1, NaN, 3, 4]))));
});

test('median over a table column propagates a NaN cell', () => {
  const fv = ev(`
t = table(x = [3.0, 0.0 / 0.0, 5.0, 9.0])
m = median(t)
`);
  assert.ok(Number.isNaN((fv.get('m') as any).x));
});

// =====================================================================
// quantile — array arm; p stays a plain probability argument
// =====================================================================

for (const [label, xs] of POSITIONS) {
  test(`quantile([${xs}], 0.5) [${label}] = NaN`, () => {
    assert.ok(Number.isNaN(ARITH_OPS.quantile(vecVal(xs), 0.5)));
  });
}

test('quantile: NaN not at the sorted-order-statistic index still propagates', () => {
  assert.ok(Number.isNaN(ARITH_OPS.quantile(vecVal([1, NaN, 3, 4]), 0.25)));
});

test('quantile: every p endpoint still sees the NaN (0 and 1 pin to min/max)', () => {
  assert.ok(Number.isNaN(ARITH_OPS.quantile(vecVal([1, NaN, 3]), 0.0)));
  assert.ok(Number.isNaN(ARITH_OPS.quantile(vecVal([1, NaN, 3]), 1.0)));
});
