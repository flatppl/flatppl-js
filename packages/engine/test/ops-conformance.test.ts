'use strict';

// =====================================================================
// ops-conformance.test.ts — unified-declaration dispatcher conformance
// =====================================================================
//
// Pins THREE properties of the `ops.dispatch` model (engine-concepts
// §17.x, Phase 1):
//
//   1. ATOM-INDEP EQUIVALENCE
//      For each declared op f and random atom-indep inputs,
//      `ops.dispatch(name, args)` produces the SAME result as the
//      existing reference implementation (currently `ARITH_OPS[name]`).
//      Catches drift between the declaration's `logical` and the
//      production path.
//
//   2. ATOM-BATCHED ↔ PER-ATOM EQUIVALENCE
//      For each declared op f and random atom-batched inputs of size
//      N, `ops.dispatch(name, batchedArgs)` equals stacking the
//      per-atom results `[logical(args[i]) for i in 0..N]`. Catches
//      bugs in the dispatcher's batch-detection / sub-Value slicing /
//      output stitching.
//
//   3. SIGNATURE PRESENCE
//      Every registered op declares a signature; the registry's
//      bookkeeping is consistent.
//
// Phase 1 ships these properties against `cross` only. As ops migrate
// (self_outer / inv / linsolve / lower_cholesky / …), the generator
// table below grows; the test bodies stay the same.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fc = require('fast-check');

const ops = require('../ops.ts');
const valueLib = require('../value.ts');
const sampler = require('../sampler.ts');

// Trigger op registrations (side-effecting require).
require('../ops-declarations.ts');

const ARITH_OPS = sampler._internal.ARITH_OPS;

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------
//
// fc generators that produce per-op input tuples. Each entry returns
// EITHER atom-indep args (logical shape) OR atom-batched args
// (shape=[N, …logical]) depending on the `mode` flag.

// Small finite-non-pathological reals — avoid Infinity / NaN that
// would propagate trivially through both paths.
const arbReal = fc.double({
  min: -1e3, max: 1e3, noNaN: true, noDefaultInfinity: true,
});

function arbVec3(): any {
  return fc.tuple(arbReal, arbReal, arbReal);
}

// Build a Value with shape=[3] from a length-3 JS array.
function vec3Value(arr: [number, number, number]): any {
  const data = new Float64Array(3);
  data[0] = arr[0]; data[1] = arr[1]; data[2] = arr[2];
  return { shape: [3], data };
}

// Build a Value with shape=[N, 3] from a length-N array of length-3
// JS arrays. Atom-major layout (the §2.1 convention).
function batchedVec3Value(rows: Array<[number, number, number]>): any {
  const N = rows.length;
  const data = new Float64Array(N * 3);
  for (let i = 0; i < N; i++) {
    data[i * 3 + 0] = rows[i][0];
    data[i * 3 + 1] = rows[i][1];
    data[i * 3 + 2] = rows[i][2];
  }
  return { shape: [N, 3], data };
}

// Variable-length vector Value, shape=[n].
function vecValue(arr: number[]): any {
  const data = new Float64Array(arr.length);
  for (let i = 0; i < arr.length; i++) data[i] = arr[i];
  return { shape: [arr.length], data };
}

// Batched variable-length vectors, shape=[N, n].
function batchedVecValue(rows: number[][]): any {
  const N = rows.length;
  const n = rows[0].length;
  const data = new Float64Array(N * n);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < n; j++) data[i * n + j] = rows[i][j];
  }
  return { shape: [N, n], data };
}

// Convert a Value (or bare Float64Array) to a plain JS array for
// assert.deepEqual on the numeric content.
function dataArr(v: any): number[] {
  if (valueLib.isValue(v)) return Array.from(v.data as Float64Array);
  if (v instanceof Float64Array) return Array.from(v);
  if (Array.isArray(v)) return v.slice();
  throw new Error('dataArr: unexpected result type ' + typeof v);
}

// Per-op generator table. `mode: 'indep' | 'batched'` controls whether
// arguments are atom-indep (Values with logical shape) or atom-batched
// (shape=[N, …logical], with N drawn ≥ 1).
const GENERATORS: Record<string, {
  indep: () => any;
  batched: () => any;
}> = {
  cross: {
    indep: () => fc.tuple(arbVec3(), arbVec3()).map(([a, b]: any) => ({
      args: [vec3Value(a), vec3Value(b)],
      perAtomArgs: [[vec3Value(a), vec3Value(b)]],   // N=1
    })),
    batched: () => fc.tuple(
      fc.integer({ min: 1, max: 5 }),
    ).chain(([N]: any) => fc.tuple(
      fc.array(arbVec3(), { minLength: N, maxLength: N }),
      fc.array(arbVec3(), { minLength: N, maxLength: N }),
    ).map(([as, bs]: any) => ({
      N,
      args: [batchedVec3Value(as), batchedVec3Value(bs)],
      perAtomArgs: as.map((a: any, i: any) => [vec3Value(a), vec3Value(bs[i])]),
    }))),
  },
  self_outer: {
    indep: () => fc.integer({ min: 1, max: 6 }).chain((n: any) =>
      fc.array(arbReal, { minLength: n, maxLength: n })).map((v: any) => ({
      args: [vecValue(v)],
      perAtomArgs: [[vecValue(v)]],
    })),
    batched: () => fc.tuple(
      fc.integer({ min: 1, max: 4 }),
      fc.integer({ min: 1, max: 5 }),
    ).chain(([N, n]: any) =>
      fc.array(
        fc.array(arbReal, { minLength: n, maxLength: n }),
        { minLength: N, maxLength: N },
      ).map((rows: any) => ({
        N,
        args: [batchedVecValue(rows)],
        perAtomArgs: rows.map((r: any) => [vecValue(r)]),
      }))),
  },
};

// ---------------------------------------------------------------------
// Property 1: atom-indep ops.dispatch ≡ ARITH_OPS reference
// ---------------------------------------------------------------------

test('ops conformance: cross — atom-indep dispatch matches ARITH_OPS.cross', () => {
  fc.assert(fc.property(GENERATORS.cross.indep(), (sample: any) => {
    const viaDispatch = ops.dispatch('cross', sample.args);
    const viaArith    = ARITH_OPS.cross(sample.args[0], sample.args[1]);
    assert.deepEqual(dataArr(viaDispatch), dataArr(viaArith));
  }), { numRuns: 200 });
});

// ---------------------------------------------------------------------
// Property 2: atom-batched dispatch ≡ stacked per-atom logical calls
// ---------------------------------------------------------------------

test('ops conformance: cross — atom-batched dispatch matches per-atom stacking', () => {
  fc.assert(fc.property(GENERATORS.cross.batched(), (sample: any) => {
    const viaDispatch = ops.dispatch('cross', sample.args);
    assert.ok(valueLib.isValue(viaDispatch),
      'batched dispatch must return a Value');
    assert.deepEqual(viaDispatch.shape, [sample.N, 3],
      'batched dispatch must return shape=[N, 3]');
    // Stack per-atom results manually.
    const flat = new Float64Array(sample.N * 3);
    for (let i = 0; i < sample.N; i++) {
      const perAtom = ops.dispatch('cross', sample.perAtomArgs[i]);
      const pa = dataArr(perAtom);
      flat[i * 3 + 0] = pa[0];
      flat[i * 3 + 1] = pa[1];
      flat[i * 3 + 2] = pa[2];
    }
    assert.deepEqual(Array.from(viaDispatch.data), Array.from(flat));
  }), { numRuns: 200 });
});

// ---------------------------------------------------------------------
// self_outer — same properties at a higher rank (vec(n) → mat(n,n))
// ---------------------------------------------------------------------

test('ops conformance: self_outer — atom-indep dispatch matches ARITH_OPS.self_outer', () => {
  fc.assert(fc.property(GENERATORS.self_outer.indep(), (sample: any) => {
    const viaDispatch = ops.dispatch('self_outer', sample.args);
    const viaArith    = ARITH_OPS.self_outer(sample.args[0]);
    assert.deepEqual(dataArr(viaDispatch), dataArr(viaArith));
  }), { numRuns: 150 });
});

test('ops conformance: self_outer — atom-batched dispatch produces shape=[N, n, n]', () => {
  fc.assert(fc.property(GENERATORS.self_outer.batched(), (sample: any) => {
    const viaDispatch = ops.dispatch('self_outer', sample.args);
    assert.ok(valueLib.isValue(viaDispatch));
    const n = sample.args[0].shape[1];
    assert.deepEqual(viaDispatch.shape, [sample.N, n, n],
      'self_outer batched must lift logical [n,n] to [N, n, n]');
    // Verify per-atom equivalence: stack manually.
    const stride = n * n;
    for (let i = 0; i < sample.N; i++) {
      const perAtom = ops.dispatch('self_outer', sample.perAtomArgs[i]);
      const pa = dataArr(perAtom);
      const slice = Array.from(viaDispatch.data.slice(i * stride, (i + 1) * stride));
      assert.deepEqual(slice, pa,
        'atom ' + i + ': batched slice must equal per-atom result');
    }
  }), { numRuns: 150 });
});

// ---------------------------------------------------------------------
// Property 3: registry bookkeeping — every declared op has a signature
// ---------------------------------------------------------------------

test('ops conformance: every declared op has a signature + argRanks', () => {
  const declared = ops.listDeclared();
  assert.ok(declared.length > 0, 'at least one op must be declared');
  for (const name of declared) {
    const decl = ops.lookup(name);
    assert.ok(decl, 'lookup returns the registered declaration');
    assert.equal(typeof decl.name, 'string');
    assert.equal(decl.name, name, 'name field matches registry key');
    assert.ok(decl.signature, name + ': must declare a signature');
    assert.ok(Array.isArray(decl.argRanks),
      name + ': must declare argRanks');
    assert.equal(decl.argRanks.length, decl.signature.args.length,
      name + ': argRanks length must match signature args length');
    assert.equal(typeof decl.logical, 'function',
      name + ': must declare a logical impl');
  }
});

// ---------------------------------------------------------------------
// Targeted edge cases (in addition to the property-based runs)
// ---------------------------------------------------------------------

test('ops dispatch: cross — atom-indep × atom-batched broadcasts', () => {
  // Atom-indep e1 vs atom-batched [e2, e2, e2] should produce
  // [e3, e3, e3] (since e1 × e2 = e3).
  const e1 = vec3Value([1, 0, 0]);
  const e2_batched = batchedVec3Value([[0, 1, 0], [0, 1, 0], [0, 1, 0]]);
  const result = ops.dispatch('cross', [e1, e2_batched]);
  assert.ok(valueLib.isValue(result));
  assert.deepEqual(result.shape, [3, 3]);
  assert.deepEqual(Array.from(result.data),
    [0, 0, 1, 0, 0, 1, 0, 0, 1]);
});

test('ops dispatch: cross — atom-batch size mismatch surfaces clear error', () => {
  const a = batchedVec3Value([[1, 0, 0], [0, 1, 0]]);                 // N=2
  const b = batchedVec3Value([[0, 1, 0], [0, 0, 1], [1, 0, 0]]);      // N=3
  assert.throws(() => ops.dispatch('cross', [a, b]),
    /atom-batch size mismatch/);
});

test('ops dispatch: unknown op surfaces clear error', () => {
  assert.throws(() => ops.dispatch('nonexistent_op', []),
    /no declaration/);
});

test('ops dispatch: wrong arg count surfaces clear error', () => {
  assert.throws(() => ops.dispatch('cross', [vec3Value([1, 0, 0])]),
    /expects 2 args, got 1/);
});

test('ops dispatch: arg rank incompatible with logical rank surfaces clear error', () => {
  // shape=[3, 3] is rank 2; cross's logical rank is 1 (±1 = 1 or 2 for atom-batched).
  // shape=[3, 3] is exactly logical-rank+1, which the dispatcher treats as atom-batched
  // with N=3 — and each per-atom call with shape=[3] succeeds. So shape=[3,3]
  // ACTUALLY dispatches as N=3 batched. Use shape=[3, 3, 3] (rank 3) instead, which
  // is neither logical nor logical+1.
  const bad = { shape: [3, 3, 3], data: new Float64Array(27) };
  assert.throws(() => ops.dispatch('cross', [bad, vec3Value([1, 0, 0])]),
    /incompatible with logical rank/);
});
