'use strict';

// =====================================================================
// ops-variants.test.ts — shape-pattern variant dispatch conformance
// =====================================================================
//
// Pins the type-directed op dispatch registry (engine-concepts §18.11
// / P1 keystone). Three property layers:
//
//   1. PATTERN MATCHER — `_matchArgPattern` accepts / rejects args
//      against rank / shape / tag / struct / dtype constraints as
//      specified.
//
//   2. SPECIFICITY ORDERING — `_pickVariant` picks the most-specific
//      matching variant; ties resolve by registration order.
//
//   3. BROADCASTED-PRIMITIVES EQUIVALENCE — for each scalar primitive
//      registered with `wrappingOp: 'broadcast'`, `dispatchVariant(op,
//      args, {wrappingOp: 'broadcast'})` returns the same Value as
//      directly calling the corresponding `valueOps.*Elem` (or
//      `valueOps.add` / `sub` / `neg` for the spec-already-elementwise
//      ops). This is the P1 keystone in action: the legacy
//      `_BROADCASTED_PRIMS_CACHE` table is now registry entries, and
//      dispatch-through-registry must produce identical results.
//
// Property tests use fast-check generators over random shapes and
// data; unit tests cover edge cases (rank-0 broadcast, complex inputs,
// shape-mismatch rejection).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fc = require('fast-check');

const ops = require('../ops.ts');
const valueLib = require('../value.ts');
const valueOps = require('../value-ops.ts');

// Trigger op + variant registrations (side-effecting require).
require('../ops-declarations.ts');

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

const arbReal = fc.double({
  min: -1e3, max: 1e3, noNaN: true, noDefaultInfinity: true,
});

function vecValue(arr: number[]): any {
  const data = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) data[i] = arr[i];
  return { shape: [arr.length], data };
}

function matValue(rows: number[][]): any {
  const m = rows.length, n = rows[0].length;
  const data = new Float64Array(m * n);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) data[i * n + j] = rows[i][j];
  return { shape: [m, n], data };
}

function scalarValue(x: number): any {
  return { shape: [], data: new Float64Array([x]) };
}

function valuesEqual(a: any, b: any, tol = 1e-9): boolean {
  if (!a || !b) return a === b;
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  if (a.data.length !== b.data.length) return false;
  for (let i = 0; i < a.data.length; i++) {
    if (Math.abs(a.data[i] - b.data[i]) > tol
        && !(Number.isNaN(a.data[i]) && Number.isNaN(b.data[i]))) {
      return false;
    }
  }
  return true;
}

// =====================================================================
// 1. Pattern matcher
// =====================================================================

test('matchArgPattern: rank constraint', () => {
  const v = vecValue([1, 2, 3]);
  assert.equal(ops._matchArgPattern({ rank: 1 }, v), true);
  assert.equal(ops._matchArgPattern({ rank: 2 }, v), false);
  assert.equal(ops._matchArgPattern({}, v), true);     // no constraints
});

test('matchArgPattern: shape constraint with explicit + wildcard dims', () => {
  const m23 = matValue([[1, 2, 3], [4, 5, 6]]);
  assert.equal(ops._matchArgPattern({ shape: [2, 3] }, m23), true);
  assert.equal(ops._matchArgPattern({ shape: [2, '*'] }, m23), true);
  assert.equal(ops._matchArgPattern({ shape: ['*', 3] }, m23), true);
  assert.equal(ops._matchArgPattern({ shape: ['*', '*'] }, m23), true);
  assert.equal(ops._matchArgPattern({ shape: [2, 4] }, m23), false);
  assert.equal(ops._matchArgPattern({ shape: [3, 3] }, m23), false);
  assert.equal(ops._matchArgPattern({ shape: [2] }, m23), false);   // rank mismatch
});

test('matchArgPattern: Klein-4 tag constraint', () => {
  const v = vecValue([1, 2, 3]);
  const vT = { ...v, t: 'T' };
  assert.equal(ops._matchArgPattern({ tag: 'N' }, v), true);
  assert.equal(ops._matchArgPattern({ tag: 'T' }, v), false);
  assert.equal(ops._matchArgPattern({ tag: 'T' }, vT), true);
  assert.equal(ops._matchArgPattern({ tag: 'N' }, vT), false);
});

test('matchArgPattern: struct=diag picks diag-stored Values', () => {
  const dense = matValue([[1, 0], [0, 2]]);
  const diag = valueLib.diagMatrix(new Float64Array([1, 2]));
  assert.equal(ops._matchArgPattern({ struct: 'diag' }, dense), false);
  assert.equal(ops._matchArgPattern({ struct: 'diag' }, diag), true);
});

test('matchArgPattern: dtype constraint splits real / complex', () => {
  const real = vecValue([1, 2, 3]);
  const cx = valueLib.complexValue(
    new Float64Array([1, 2, 3]), new Float64Array([0.1, 0.2, 0.3]), [3]);
  assert.equal(ops._matchArgPattern({ dtype: 'real' }, real), true);
  assert.equal(ops._matchArgPattern({ dtype: 'complex' }, real), false);
  assert.equal(ops._matchArgPattern({ dtype: 'real' }, cx), false);
  assert.equal(ops._matchArgPattern({ dtype: 'complex' }, cx), true);
});

test('matchArgPattern: bare JS numbers / booleans match rank-0 only', () => {
  assert.equal(ops._matchArgPattern({ rank: 0 }, 3.14), true);
  assert.equal(ops._matchArgPattern({ rank: 1 }, 3.14), false);
  assert.equal(ops._matchArgPattern({}, true), true);
});

// =====================================================================
// 2. Specificity ordering
// =====================================================================

test('specificityScore: wrappingOp dominates argPattern fields', () => {
  const general = { argPatterns: [{}, {}], impl: () => null };
  const bcast   = { argPatterns: [{}, {}], wrappingOp: 'broadcast', impl: () => null };
  assert.ok(ops._specificityScore(bcast) > ops._specificityScore(general));
});

test('specificityScore: struct outweighs tag outweighs rank', () => {
  const r = { argPatterns: [{ rank: 2 }, { rank: 2 }], impl: () => null };
  const t = { argPatterns: [{ tag: 'N' }, { tag: 'N' }], impl: () => null };
  const s = { argPatterns: [{ struct: 'diag' }, { struct: 'diag' }], impl: () => null };
  assert.ok(ops._specificityScore(t) > ops._specificityScore(r));
  assert.ok(ops._specificityScore(s) > ops._specificityScore(t));
});

test('pickVariant: most-specific variant wins among matches', () => {
  // Two variants on a fictional op: a general one matching any rank,
  // and a specific one that adds tag=T constraint. For tag-T input,
  // the specific one should win.
  const general = { argPatterns: [{ rank: 1 }], impl: () => 'general' };
  const specific = { argPatterns: [{ rank: 1, tag: 'T' }], impl: () => 'specific' };
  const v = vecValue([1, 2, 3]);
  const vT = { ...v, t: 'T' };
  // Both match vT; specific should win.
  assert.equal(ops._pickVariant([general, specific], [vT], {}), specific);
  // Only general matches v (tag N).
  assert.equal(ops._pickVariant([general, specific], [v], {}), general);
});

test('pickVariant: wrappingOp constraint filters out incompatible variants', () => {
  const direct = { argPatterns: [{}, {}], wrappingOp: 'direct', impl: () => 'd' };
  const bcast  = { argPatterns: [{}, {}], wrappingOp: 'broadcast', impl: () => 'b' };
  const args = [vecValue([1]), vecValue([2])];
  assert.equal(ops._pickVariant([direct, bcast], args, { wrappingOp: 'broadcast' }), bcast);
  assert.equal(ops._pickVariant([direct, bcast], args, { wrappingOp: 'direct' }),    direct);
  // Default (no opts) is 'direct'.
  assert.equal(ops._pickVariant([direct, bcast], args, {}), direct);
});

test('pickVariant: no match returns null', () => {
  const onlyTag = { argPatterns: [{ tag: 'T' }], impl: () => 't' };
  const v = vecValue([1, 2, 3]);   // tag N
  assert.equal(ops._pickVariant([onlyTag], [v], {}), null);
});

// =====================================================================
// 3. Broadcasted-primitives equivalence (the P1 keystone proof)
// =====================================================================
//
// For each registered broadcast-wrapping variant, dispatching through
// the registry must produce identical results to the corresponding
// direct value-ops call. Drift here would mean the registry-routed
// fast path has diverged from the canonical impl.

// Binary primitives: (op, valueOps fn). Each maps the registered
// `(opName, wrappingOp:'broadcast')` variant to its expected impl.
const BCAST_BINARY: Array<[string, (a: any, b: any) => any]> = [
  ['add',    (a, b) => valueOps.add(a, b)],
  ['sub',    (a, b) => valueOps.sub(a, b)],
  ['mul',    (a, b) => valueOps.mulElem(a, b)],
  ['div',    (a, b) => valueOps.divElem(a, b)],
  ['divide', (a, b) => valueOps.divElem(a, b)],
  ['pow',    (a, b) => valueOps.powElem(a, b)],
  ['mod',    (a, b) => valueOps.modElem(a, b)],
];

// Unary primitives: (op, valueOps fn).
const BCAST_UNARY: Array<[string, (a: any) => any]> = [
  ['neg',    (a) => valueOps.neg(a)],
  ['exp',    (a) => valueOps.expElem(a)],
  ['log',    (a) => valueOps.logElem(a)],
  ['sqrt',   (a) => valueOps.sqrtElem(a)],
  ['sin',    (a) => valueOps.sinElem(a)],
  ['cos',    (a) => valueOps.cosElem(a)],
  ['tan',    (a) => valueOps.tanElem(a)],
  ['abs',    (a) => valueOps.absElem(a)],
  ['abs2',   (a) => valueOps.abs2Elem(a)],
  ['log10',  (a) => valueOps.log10Elem(a)],
  ['log1p',  (a) => valueOps.log1pElem(a)],
  ['expm1',  (a) => valueOps.expm1Elem(a)],
  ['floor',  (a) => valueOps.floorElem(a)],
  ['ceil',   (a) => valueOps.ceilElem(a)],
  ['round',  (a) => valueOps.roundElem(a)],
];

for (const [opName, refFn] of BCAST_BINARY) {
  test('broadcast variant: ' + opName + ' on length-N vectors ≡ valueOps direct', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 8 }),
      (n: number) => fc.assert(fc.property(
        fc.array(arbReal, { minLength: n, maxLength: n }),
        fc.array(arbReal, { minLength: n, maxLength: n }),
        (a: number[], b: number[]) => {
          // pow/mod: guard against domain pitfalls
          let A = a, B = b;
          if (opName === 'pow') {
            A = a.map((x: number) => Math.abs(x) + 0.5);  // positive bases
            B = b.map((x: number) => Math.max(-3, Math.min(3, x)));
          }
          if (opName === 'div' || opName === 'divide' || opName === 'mod') {
            B = b.map((x: number) => Math.abs(x) < 1e-6 ? 1 : x);  // non-zero divisor
          }
          const va = vecValue(A);
          const vb = vecValue(B);
          const r1 = ops.dispatchVariant(opName, [va, vb], { wrappingOp: 'broadcast' });
          const r2 = refFn(va, vb);
          assert.ok(valuesEqual(r1, r2),
            opName + ': dispatchVariant ≠ direct ' +
            'shape=' + JSON.stringify(r1 && r1.shape) +
            ' vs ' + JSON.stringify(r2 && r2.shape));
        }
      ), { numRuns: 5 })
    ), { numRuns: 4 });
  });
}

for (const [opName, refFn] of BCAST_UNARY) {
  test('broadcast variant: ' + opName + ' on length-N vectors ≡ valueOps direct', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 8 }),
      (n: number) => fc.assert(fc.property(
        fc.array(arbReal, { minLength: n, maxLength: n }),
        (a: number[]) => {
          // sqrt/log/log10/log1p domain
          let A = a;
          if (opName === 'sqrt' || opName === 'log' || opName === 'log10') {
            A = a.map((x: number) => Math.abs(x) + 1e-3);
          }
          if (opName === 'log1p') {
            A = a.map((x: number) => Math.max(-0.99, x));
          }
          const va = vecValue(A);
          const r1 = ops.dispatchVariant(opName, [va], { wrappingOp: 'broadcast' });
          const r2 = refFn(va);
          assert.ok(valuesEqual(r1, r2),
            opName + ': dispatchVariant ≠ direct ' +
            'shape=' + JSON.stringify(r1 && r1.shape) +
            ' vs ' + JSON.stringify(r2 && r2.shape));
        }
      ), { numRuns: 5 })
    ), { numRuns: 4 });
  });
}

// Higher-rank: matrix shapes go through the same elementwise impls.
test('broadcast variant: mul on [2,3] matrices ≡ mulElem direct', () => {
  const A = matValue([[1, 2, 3], [4, 5, 6]]);
  const B = matValue([[7, 8, 9], [10, 11, 12]]);
  const r1 = ops.dispatchVariant('mul', [A, B], { wrappingOp: 'broadcast' });
  const r2 = valueOps.mulElem(A, B);
  assert.ok(valuesEqual(r1, r2));
  // Sanity check: NOT equal to spec-mul (matrix product would need
  // shape match between cols(A) and rows(B); 3 != 2, so matrix product
  // would actually throw — confirming the broadcast variant is semantically
  // distinct from the direct call).
  assert.throws(() => valueOps.mul(A, B));
});

// Scalar broadcast: rank-0 × rank-N — value-ops elementwise factory
// handles this internally.
test('broadcast variant: rank-0 × rank-N scalar broadcast', () => {
  const s = scalarValue(3);
  const v = vecValue([1, 2, 3]);
  const r1 = ops.dispatchVariant('mul', [s, v], { wrappingOp: 'broadcast' });
  const r2 = valueOps.mulElem(s, v);
  assert.ok(valuesEqual(r1, r2));
});

// Without wrappingOp='broadcast' opt, the broadcast variant doesn't
// match: a plain `dispatchVariant('mul', [A, B])` returns null because
// there's no `direct`-wrapped variant for mul on rank-1 args (mul-direct
// still goes through valueOps.mul's shape switch, not the registry).
test('dispatchVariant: without wrappingOp, broadcast variant does NOT match', () => {
  const A = vecValue([1, 2, 3]);
  const B = vecValue([4, 5, 6]);
  const r = ops.dispatchVariant('mul', [A, B]);  // no opts
  assert.equal(r, null);
});

// Registry bookkeeping: every broadcasted-primitive op has a variant
// registered with wrappingOp='broadcast'. Drift between this test and
// the actual registration table signals a missing migration.
test('registry: all broadcasted primitives have wrappingOp=broadcast variants', () => {
  const expected = [
    'add', 'sub', 'mul', 'div', 'divide', 'pow', 'mod',
    'neg', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tan',
    'abs', 'abs2', 'log10', 'log1p', 'expm1',
    'floor', 'ceil', 'round',
  ];
  for (const name of expected) {
    const decl = ops.lookup(name);
    assert.ok(decl, 'no OpDecl for ' + name);
    assert.ok(decl.variants && decl.variants.length > 0,
      name + ' has no variants');
    const bcast = decl.variants.find((v: any) => v.wrappingOp === 'broadcast');
    assert.ok(bcast, name + ' has no wrappingOp=broadcast variant');
  }
});
