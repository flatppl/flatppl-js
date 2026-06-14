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

// Square-matrix Value, shape=[n, n].
function matValue(rows: number[][]): any {
  const n = rows.length;
  const data = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) data[i * n + j] = rows[i][j];
  }
  return { shape: [n, n], data };
}

// Batched square matrices, shape=[N, n, n].
function batchedMatValue(matrices: number[][][]): any {
  const N = matrices.length;
  const n = matrices[0].length;
  const data = new Float64Array(N * n * n);
  for (let k = 0; k < N; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) data[k * n * n + i * n + j] = matrices[k][i][j];
    }
  }
  return { shape: [N, n, n], data };
}

// Random square matrix arbitrary, n × n.
function arbSquareMat(maxN = 5): any {
  return fc.integer({ min: 1, max: maxN }).chain((n: any) =>
    fc.array(fc.array(arbReal, { minLength: n, maxLength: n }),
      { minLength: n, maxLength: n }));
}

// Random SPD matrix: generate L lower-triangular with positive
// diagonal, return L · Lᵀ. Always positive definite by construction.
function arbSPD(maxN = 5): any {
  return fc.integer({ min: 1, max: maxN }).chain((n: any) =>
    fc.array(arbReal, { minLength: n * n, maxLength: n * n })
      .map((vals: any) => {
        // Build lower-triangular L with non-zero diagonal.
        const L: number[][] = [];
        let idx = 0;
        for (let i = 0; i < n; i++) {
          const row: number[] = new Array(n).fill(0);
          for (let j = 0; j <= i; j++) row[j] = vals[idx++];
          // Force a positive, non-tiny diagonal so det(L) is bounded
          // away from zero.
          row[i] = Math.abs(row[i]) + 1.0;
          L.push(row);
        }
        // A = L · Lᵀ.
        const A: number[][] = [];
        for (let i = 0; i < n; i++) {
          const row: number[] = new Array(n);
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let k = 0; k <= Math.min(i, j); k++) s += L[i][k] * L[j][k];
            row[j] = s;
          }
          A.push(row);
        }
        return A;
      }));
}

// Random non-singular matrix: A + n·I (diagonal-dominant).
function arbNonSingular(maxN = 5): any {
  return arbSquareMat(maxN).map((A: any) => {
    const n = A.length;
    const out = A.map((row: any) => row.slice());
    for (let i = 0; i < n; i++) out[i][i] += n + 1;  // diagonal dominance
    return out;
  });
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
// Phase 2 linalg family — trace / diagmat / det / logabsdet / inv /
// lower_cholesky / row_gram / col_gram. Each test runs ops.dispatch
// against the reference (ARITH_OPS) over random inputs, then checks
// the atom-batched stacking property.
//
// Helpers below close over `op` + an input generator (atom-indep) and
// run BOTH the legacy-reference equivalence and the
// batched ≡ per-atom property.
// ---------------------------------------------------------------------

// Compare two scalar / array / Value results structurally on numeric
// data, with a small tolerance for floating-point ops where one path
// might re-associate (det / inv / cholesky etc.).
function approxEqual(a: any, b: any, tol = 1e-9): boolean {
  const av = valueLib.isValue(a) ? a : valueLib.asValue(a);
  const bv = valueLib.isValue(b) ? b : valueLib.asValue(b);
  if (av.shape.length !== bv.shape.length) return false;
  for (let i = 0; i < av.shape.length; i++) {
    if (av.shape[i] !== bv.shape[i]) return false;
  }
  if (av.data.length !== bv.data.length) return false;
  for (let i = 0; i < av.data.length; i++) {
    const da = av.data[i], db = bv.data[i];
    if (!isFinite(da) && !isFinite(db)) continue;        // both ±inf / NaN — call equal
    if (Math.abs(da - db) > tol * (1 + Math.max(Math.abs(da), Math.abs(db)))) {
      return false;
    }
  }
  return true;
}

// Run the three load-bearing properties for an op declared via OpDecl
// against a generator of atom-indep Value-typed inputs.
function pinUnaryLinalgOp(opName: string, arbInputs: any) {
  // 1. atom-indep ops.dispatch ≡ ARITH_OPS.
  test('ops conformance: ' + opName + ' — atom-indep dispatch matches ARITH_OPS', () => {
    fc.assert(fc.property(arbInputs, (rows: any) => {
      const A = matValue(rows);
      const viaDispatch = ops.dispatch(opName, [A]);
      const viaArith    = ARITH_OPS[opName](A);
      assert.ok(approxEqual(viaDispatch, viaArith),
        opName + ' atom-indep mismatch (dispatch vs ARITH_OPS)');
    }), { numRuns: 100 });
  });
  // 2. atom-batched dispatch ≡ stacked per-atom logical results.
  test('ops conformance: ' + opName + ' — atom-batched dispatch matches per-atom stacking', () => {
    fc.assert(fc.property(
      fc.integer({ min: 1, max: 4 }),
      arbInputs,
      (N: any, rows: any) => {
        const matrices: number[][][] = [];
        for (let i = 0; i < N; i++) matrices.push(rows);
        const batched = batchedMatValue(matrices);
        const viaDispatch = ops.dispatch(opName, [batched]);
        // Each atom carries the SAME matrix, so per-atom results
        // must all equal the atom-indep result.
        const perAtom = ops.dispatch(opName, [matValue(rows)]);
        // viaDispatch must be a Value with leading dim N.
        assert.ok(valueLib.isValue(viaDispatch),
          opName + ' batched must return a Value');
        assert.equal(viaDispatch.shape[0], N,
          opName + ' batched: leading dim must be N');
        // Slice each atom out and compare to the atom-indep result.
        const tailShape = viaDispatch.shape.slice(1);
        const tailLen = tailShape.reduce((a: number, b: number) => a * b, 1);
        const expected = valueLib.isValue(perAtom)
          ? perAtom.data
          : (perAtom instanceof Float64Array ? perAtom : new Float64Array([perAtom]));
        for (let i = 0; i < N; i++) {
          const slice = viaDispatch.data.slice(i * tailLen, (i + 1) * tailLen);
          const sliceV = { shape: tailShape, data: slice };
          const expectV = { shape: tailShape, data: expected };
          assert.ok(approxEqual(sliceV, expectV),
            opName + ' batched atom ' + i + ' must equal atom-indep result');
        }
      },
    ), { numRuns: 50 });
  });
}

pinUnaryLinalgOp('trace', arbSquareMat());
pinUnaryLinalgOp('det', arbNonSingular());
pinUnaryLinalgOp('logabsdet', arbNonSingular());
pinUnaryLinalgOp('inv', arbNonSingular());
pinUnaryLinalgOp('lower_cholesky', arbSPD());
pinUnaryLinalgOp('row_gram', arbSquareMat());
pinUnaryLinalgOp('col_gram', arbSquareMat());

// ---------------------------------------------------------------------
// Higher-order ops (Phase 5c) — reduce / scan / filter. The
// dispatcher's `dispatchHigherOrder(name, irArgs, ctx)` threads the
// engine's env + evaluateExpr + resolveFn into the op's logical,
// which does its own callable resolution and iteration.
//
// These tests exercise the dispatch surface directly by building a
// `ctx` against `sampler.evaluateExpr` + a minimal resolveFn that
// recognises inline `functionof` IR. The engine's own evaluateCall
// already routes reduce/scan/filter through dispatchHigherOrder, so
// the integration tests in fold/reduce/scan/filter.test.ts cover
// the end-to-end path. Here we pin the conformance contract:
// `dispatchHigherOrder` produces the same result as the legacy
// inline impl in evaluateCall.
// ---------------------------------------------------------------------

// Minimal resolveFn for tests: recognises inline `functionof(params,
// body)` IR and returns { params, body }. Doesn't handle named-fn
// refs (those need orchestrator setup); the dispatch path's
// fallthrough behaviour matches the legacy impl on the inline form.
function _testResolveFn(fnIR: any, _env: any): any {
  if (fnIR && fnIR.kind === 'call' && fnIR.op === 'functionof'
      && Array.isArray(fnIR.params) && fnIR.body) {
    return { params: fnIR.params, body: fnIR.body };
  }
  return null;
}

function _testCtx(env: any): any {
  return {
    env: env || {},
    evaluateExpr: sampler.evaluateExpr,
    resolveFn: _testResolveFn,
  };
}

// Build `functionof((a, b), body)` IR for binary callables.
function _binFn(p0: string, p1: string, bodyIR: any): any {
  return { kind: 'call', op: 'functionof', params: [p0, p1], body: bodyIR };
}

// Build `functionof((a), body)` IR for unary callables.
function _unFn(p0: string, bodyIR: any): any {
  return { kind: 'call', op: 'functionof', params: [p0], body: bodyIR };
}

function _ref(name: string): any { return { kind: 'ref', ns: 'self', name }; }
function _lit(v: any): any { return { kind: 'lit', value: v }; }
function _binOp(op: string, a: any, b: any): any {
  return { kind: 'call', op, args: [a, b] };
}

// Build a call IR node for testing higher-order dispatch — the
// engine's evaluateCall passes the full `ir` node (so kwargs are
// visible to broadcast), and the OpDecl's `logical(ir, ctx)`
// extracts `ir.args` (and `ir.kwargs` when relevant).
function _callIR(op: string, args: any[], kwargs?: any): any {
  return kwargs ? { kind: 'call', op, args, kwargs }
                : { kind: 'call', op, args };
}

test('ops conformance: reduce — sum via dispatchHigherOrder matches legacy', () => {
  // reduce((a, b) -> a + b, [1, 2, 3, 4, 5]) ≡ 15
  const fn = _binFn('a', 'b', _binOp('add', _ref('a'), _ref('b')));
  const xs = { kind: 'call', op: 'vector',
    args: [1, 2, 3, 4, 5].map(_lit) };
  const r = ops.dispatchHigherOrder('reduce', _callIR('reduce', [fn, xs]), _testCtx({}));
  assert.equal(r, 15);
});

test('ops conformance: reduce — product via dispatchHigherOrder', () => {
  // reduce((a, b) -> a * b, [1, 2, 3, 4]) ≡ 24
  const fn = _binFn('a', 'b', _binOp('mul', _ref('a'), _ref('b')));
  const xs = { kind: 'call', op: 'vector',
    args: [1, 2, 3, 4].map(_lit) };
  const r = ops.dispatchHigherOrder('reduce', _callIR('reduce', [fn, xs]), _testCtx({}));
  assert.equal(r, 24);
});

test('ops conformance: scan — cumsum via dispatchHigherOrder', () => {
  // scan((a, b) -> a + b, 0, [1, 2, 3, 4]) ≡ [1, 3, 6, 10]
  const fn = _binFn('a', 'b', _binOp('add', _ref('a'), _ref('b')));
  const xs = { kind: 'call', op: 'vector',
    args: [1, 2, 3, 4].map(_lit) };
  const r = ops.dispatchHigherOrder('scan',
    _callIR('scan', [fn, _lit(0), xs]), _testCtx({}));
  assert.ok(valueLib.isValue(r));
  assert.deepEqual(r.shape, [4]);
  assert.deepEqual(Array.from(r.data), [1, 3, 6, 10]);
});

test('ops conformance: filter — keep evens via dispatchHigherOrder', () => {
  // filter((x) -> x mod 2 == 0, [1, 2, 3, 4, 5, 6]) ≡ [2, 4, 6]
  const pred = _unFn('x', { kind: 'call', op: 'equal',
    args: [{ kind: 'call', op: 'mod', args: [_ref('x'), _lit(2)] }, _lit(0)] });
  const xs = { kind: 'call', op: 'vector',
    args: [1, 2, 3, 4, 5, 6].map(_lit) };
  const r = ops.dispatchHigherOrder('filter',
    _callIR('filter', [pred, xs]), _testCtx({}));
  assert.ok(valueLib.isValue(r));
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [2, 4, 6]);
});

test('ops conformance: broadcast — positional dispatch matches legacy', () => {
  // broadcast((x) -> x * 2, [1, 2, 3, 4]) ≡ [2, 4, 6, 8]
  const fn = _unFn('x', _binOp('mul', _ref('x'), _lit(2)));
  const xs = { kind: 'call', op: 'vector',
    args: [1, 2, 3, 4].map(_lit) };
  const r = ops.dispatchHigherOrder('broadcast',
    _callIR('broadcast', [fn, xs]), _testCtx({}));
  // Broadcast result is either a JS array of element results or a
  // packed Value; both representations carry the same numeric data.
  const flat = valueLib.isValue(r) ? Array.from(r.data) : r;
  assert.deepEqual(flat, [2, 4, 6, 8]);
});

test('ops conformance: broadcast — two-arg binary dispatch', () => {
  // broadcast((x, y) -> x + y, [1, 2, 3], [10, 20, 30]) ≡ [11, 22, 33]
  const fn = _binFn('x', 'y', _binOp('add', _ref('x'), _ref('y')));
  const xs = { kind: 'call', op: 'vector', args: [1, 2, 3].map(_lit) };
  const ys = { kind: 'call', op: 'vector', args: [10, 20, 30].map(_lit) };
  const r = ops.dispatchHigherOrder('broadcast',
    _callIR('broadcast', [fn, xs, ys]), _testCtx({}));
  const flat = valueLib.isValue(r) ? Array.from(r.data) : r;
  assert.deepEqual(flat, [11, 22, 33]);
});

test('ops conformance: aggregate — matrix-vector via dispatchHigherOrder', () => {
  // aggregate(sum, [.i], A[.i, .j] * v[.j]) ≡ A · v
  // A = [[1, 2], [3, 4]]; v = [10, 20]; A · v = [50, 110]
  const A = { kind: 'call', op: 'rowstack', args: [
    { kind: 'call', op: 'vector', args: [
      { kind: 'call', op: 'vector', args: [_lit(1), _lit(2)] },
      { kind: 'call', op: 'vector', args: [_lit(3), _lit(4)] },
    ]},
  ]};
  const v = { kind: 'call', op: 'vector', args: [_lit(10), _lit(20)] };
  const env = { __resolveFnBody: () => null };
  // env needs the bindings inline-evaluated, so we wire them by
  // putting concrete values directly in env.
  // Body: A[.i, .j] * v[.j]
  const Av = sampler.evaluateExpr(A, env);
  const vv = sampler.evaluateExpr(v, env);
  const env2: any = { ...env, _A: Av, _v: vv };
  const body = _binOp('mul',
    { kind: 'call', op: 'get', args: [_ref('_A'),
      { kind: 'axis', name: 'i' }, { kind: 'axis', name: 'j' }] },
    { kind: 'call', op: 'get', args: [_ref('_v'),
      { kind: 'axis', name: 'j' }] });
  const aggIR = _callIR('aggregate', [
    _ref('sum'),                                        // reduction
    { kind: 'call', op: 'vector', args: [{ kind: 'axis', name: 'i' }] },
    body,
  ]);
  const r = ops.dispatchHigherOrder('aggregate', aggIR, _testCtx(env2));
  // Result should be [50, 110]
  const flat = valueLib.isValue(r) ? Array.from(r.data)
             : (r instanceof Float64Array ? Array.from(r) : r);
  assert.deepEqual(flat, [50, 110]);
});

test('ops dispatchHigherOrder: missing ctx surfaces clear error', () => {
  const fn = _binFn('a', 'b', _binOp('add', _ref('a'), _ref('b')));
  const xs = { kind: 'call', op: 'vector', args: [_lit(1), _lit(2)] };
  assert.throws(() => ops.dispatchHigherOrder('reduce',
    _callIR('reduce', [fn, xs]), null as any),
    /ctx must provide/);
});

test('ops dispatchHigherOrder: non-higher-order op surfaces clear error', () => {
  // `cross` is fixed-rank — should reject through dispatchHigherOrder.
  assert.throws(() => ops.dispatchHigherOrder('cross',
    _callIR('cross', []), _testCtx({})),
    /not kind=higher-order/);
});

test('ops dispatch: higher-order op surfaces clear error if routed through dispatch', () => {
  // `reduce` is higher-order — should reject through plain dispatch.
  assert.throws(() => ops.dispatch('reduce', [null, null]),
    /higher-order/);
});

// ---------------------------------------------------------------------
// Variadic ops (Phase 5b) — vector / cat. The dispatcher forwards
// all args to logical; no atom-batch detection.
// ---------------------------------------------------------------------

test('ops conformance: vector — variadic dispatch matches ARITH_OPS.vector', () => {
  fc.assert(fc.property(
    fc.array(arbReal, { minLength: 0, maxLength: 8 }),
    (xs: any) => {
      const viaDispatch = ops.dispatch('vector', xs);
      const viaArith    = ARITH_OPS.vector(...xs);
      if (valueLib.isValue(viaDispatch) && valueLib.isValue(viaArith)) {
        assert.deepEqual(viaDispatch.shape, viaArith.shape);
        assert.deepEqual(Array.from(viaDispatch.data),
                         Array.from(viaArith.data));
      } else {
        assert.deepEqual(viaDispatch, viaArith);
      }
    },
  ), { numRuns: 100 });
});

test('ops conformance: cat — variadic dispatch matches ARITH_OPS.cat (scalars)', () => {
  fc.assert(fc.property(
    fc.array(arbReal, { minLength: 1, maxLength: 6 }),
    (xs: any) => {
      const viaDispatch = ops.dispatch('cat', xs);
      const viaArith    = ARITH_OPS.cat(...xs);
      assert.deepEqual(viaDispatch.shape, viaArith.shape);
      assert.deepEqual(Array.from(viaDispatch.data),
                       Array.from(viaArith.data));
    },
  ), { numRuns: 100 });
});

test('ops conformance: cat — variadic dispatch matches ARITH_OPS.cat (vectors)', () => {
  fc.assert(fc.property(
    fc.array(
      fc.array(arbReal, { minLength: 1, maxLength: 4 })
        .map(vecValue),
      { minLength: 1, maxLength: 4 },
    ),
    (vs: any) => {
      const viaDispatch = ops.dispatch('cat', vs);
      const viaArith    = ARITH_OPS.cat(...vs);
      assert.deepEqual(viaDispatch.shape, viaArith.shape);
      assert.deepEqual(Array.from(viaDispatch.data),
                       Array.from(viaArith.data));
    },
  ), { numRuns: 100 });
});

test('ops dispatch: variadic ops accept zero args', () => {
  const empty = ops.dispatch('vector', []);
  assert.deepEqual(empty.shape, [0]);
  assert.equal(empty.data.length, 0);
});

// ---------------------------------------------------------------------
// Rank-polymorphic ops (Phase 5a) — transpose / adjoint / linsolve.
// The dispatcher does NOT auto-atom-batch these; the contract is to
// call `logical` with the input as-is. Conformance against ARITH_OPS
// reference still holds.
// ---------------------------------------------------------------------

test('ops conformance: transpose — matrix dispatch matches ARITH_OPS.transpose', () => {
  fc.assert(fc.property(
    fc.tuple(
      fc.integer({ min: 1, max: 5 }),
      fc.integer({ min: 1, max: 5 }),
    ).chain(([m, n]: any) =>
      fc.array(fc.array(arbReal, { minLength: n, maxLength: n }),
        { minLength: m, maxLength: m })),
    (rows: any) => {
      const m = rows.length, n = rows[0].length;
      const data = new Float64Array(m * n);
      for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) data[i * n + j] = rows[i][j];
      }
      const M = { shape: [m, n], data };
      const viaDispatch = ops.dispatch('transpose', [M]);
      const viaArith    = ARITH_OPS.transpose(M);
      // Both produce O(1)-tag-flip Values; compare via valueLib.densify.
      const dDisp = valueLib.densify(viaDispatch);
      const dArith = valueLib.densify(viaArith);
      assert.deepEqual(dDisp.shape, dArith.shape);
      assert.deepEqual(Array.from(dDisp.data), Array.from(dArith.data));
    },
  ), { numRuns: 100 });
});

test('ops conformance: transpose — vector (rank 1) tag-flips', () => {
  const v = vecValue([1, 2, 3]);
  const t = ops.dispatch('transpose', [v]);
  assert.ok(valueLib.isValue(t));
  // valueLib.transpose flips the Klein-4 tag; data unchanged.
  assert.deepEqual(Array.from(t.data), [1, 2, 3]);
});

test('ops conformance: adjoint matches transpose on real input', () => {
  const M = matValue([[1, 2], [3, 4]]);
  const T = ops.dispatch('transpose', [M]);
  const A = ops.dispatch('adjoint', [M]);
  // Real matrices: adjoint ≡ transpose (no conjugation effect).
  assert.deepEqual(Array.from(valueLib.densify(A).data),
                   Array.from(valueLib.densify(T).data));
});

test('ops conformance: linsolve(A, b) — vector b matches ARITH_OPS', () => {
  fc.assert(fc.property(
    arbNonSingular(),
    (rows: any) => {
      const n = rows.length;
      const A = matValue(rows);
      // b = (1, 2, …, n) for a deterministic right-hand side.
      const bArr: number[] = [];
      for (let i = 0; i < n; i++) bArr.push(i + 1);
      const b = vecValue(bArr);
      const viaDispatch = ops.dispatch('linsolve', [A, b]);
      const viaArith    = ARITH_OPS.linsolve(A, b);
      // Both produce vectors; compare data.
      const dDisp = valueLib.densify(viaDispatch);
      const dArith = valueLib.densify(viaArith);
      assert.deepEqual(dDisp.shape, dArith.shape);
      // Numerical tolerance for LU: compare via approxEqual.
      assert.ok(approxEqual(dDisp, dArith),
        'linsolve vector b mismatch (dispatch vs ARITH_OPS)');
    },
  ), { numRuns: 100 });
});

test('ops conformance: linsolve(A, b) — matrix b matches ARITH_OPS', () => {
  fc.assert(fc.property(
    arbNonSingular(),
    (rows: any) => {
      const n = rows.length;
      const A = matValue(rows);
      // b = n × n identity (so x should ≈ A⁻¹).
      const bArr: number[][] = [];
      for (let i = 0; i < n; i++) {
        const row: number[] = new Array(n).fill(0);
        row[i] = 1;
        bArr.push(row);
      }
      const b = matValue(bArr);
      const viaDispatch = ops.dispatch('linsolve', [A, b]);
      const viaArith    = ARITH_OPS.linsolve(A, b);
      const dDisp = valueLib.densify(viaDispatch);
      const dArith = valueLib.densify(viaArith);
      assert.deepEqual(dDisp.shape, dArith.shape);
      assert.ok(approxEqual(dDisp, dArith),
        'linsolve matrix b mismatch (dispatch vs ARITH_OPS)');
    },
  ), { numRuns: 100 });
});

// diagmat takes a VECTOR input, returns a structured diag matrix.
// The dispatcher's per-atom fallback for diagmat shape=[N, n] inputs
// would produce diag-structured per-atom results which then stack via
// `_stackPerAtom`. The structured-Value handling in _stackPerAtom
// would need to densify — exercise that explicitly here.
test('ops conformance: diagmat — atom-indep dispatch matches ARITH_OPS', () => {
  fc.assert(fc.property(
    fc.integer({ min: 1, max: 6 }).chain((n: any) =>
      fc.array(arbReal, { minLength: n, maxLength: n })),
    (v: any) => {
      const V = vecValue(v);
      const viaDispatch = ops.dispatch('diagmat', [V]);
      const viaArith    = ARITH_OPS.diagmat(V);
      // Both should be diag-structured Values with the same vector.
      assert.ok(valueLib.isValue(viaDispatch));
      assert.deepEqual(
        Array.from(viaDispatch.data),
        Array.from(viaArith.data),
        'diagmat must produce the same diag-stored data');
    },
  ), { numRuns: 100 });
});

// ---------------------------------------------------------------------
// Property 3: registry bookkeeping — every declared op has a signature
// ---------------------------------------------------------------------

test('ops conformance: every declared op has consistent bookkeeping', () => {
  const declared = ops.listDeclared();
  assert.ok(declared.length > 0, 'at least one op must be declared');
  for (const name of declared) {
    const decl = ops.lookup(name);
    assert.ok(decl, 'lookup returns the registered declaration');
    assert.equal(typeof decl.name, 'string');
    assert.equal(decl.name, name, 'name field matches registry key');
    assert.equal(typeof decl.logical, 'function',
      name + ': must declare a logical impl');
    const kind = decl.kind || 'fixed-rank';
    // kind='fixed-rank': signature + argRanks required, and they
    // must line up. Other kinds use loose signature handling.
    if (kind === 'fixed-rank') {
      assert.ok(decl.signature,
        name + ' (kind=fixed-rank): must declare a signature');
      assert.ok(Array.isArray(decl.argRanks),
        name + ' (kind=fixed-rank): must declare argRanks');
      assert.equal(decl.argRanks.length, decl.signature.args.length,
        name + ' (kind=fixed-rank): argRanks length must match ' +
        'signature args length');
    }
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

test('ops dispatchHigherOrder: higher-order op without a logical impl surfaces clear error', () => {
  // A higher-order op may be registered with only `variants` (no
  // `logical`) — register() permits that. dispatchHigherOrder has no
  // variant path, so it must reject the missing logical with a clear
  // error rather than throwing an opaque "not a function". Give the
  // decl a dummy variant so registration passes; dispatch never
  // consults it (it throws on the absent logical first).
  const name = '__test_higherorder_no_logical';
  ops.register({
    name,
    kind: 'higher-order',
    variants: [{ pattern: [], impl: () => null }],
  });
  assert.throws(
    () => ops.dispatchHigherOrder(name, _callIR(name, []), _testCtx({})),
    /has no logical impl/);
});

// ---------------------------------------------------------------------
// Scalar-primitive logical migration (engine-concepts §18). As each
// scalar family attaches a `logical` impl, its single-point eval routes
// through ops.dispatch; pin atom-indep dispatch ≡ ARITH_OPS over random
// scalars (the batched path is the broadcast variant, tested elsewhere).
//
// Family 1: pure-real unary elementary math.
// ---------------------------------------------------------------------

const REAL_UNARY_PRIMS = [
  'log10', 'log1p', 'expm1',
  'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
  'sinh', 'cosh', 'tanh', 'asinh', 'acosh', 'atanh',
  'floor', 'ceil', 'round',
];

for (const op of REAL_UNARY_PRIMS) {
  test('ops conformance: ' + op + ' (scalar logical) — dispatch matches ARITH_OPS', () => {
    assert.ok(ops.isDeclared(op), op + ' must be declared (logical attached)');
    fc.assert(fc.property(arbReal, (x: any) => {
      const viaDispatch = ops.dispatch(op, [x]);
      const viaArith = ARITH_OPS[op](x);
      // Object.is so NaN (out-of-domain, e.g. asin(2)) compares equal.
      return Object.is(viaDispatch, viaArith)
        || Math.abs(viaDispatch - viaArith) < 1e-15;
    }), { numRuns: 200 });
  });
}
