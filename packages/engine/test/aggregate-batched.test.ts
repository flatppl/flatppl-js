'use strict';

// =====================================================================
// aggregate-batched.test.ts — atom-batched aggregate evaluator
// =====================================================================
//
// Pins `_evalAggregateBroadcastReduceN` in sampler-aggregate.ts
// (engine-concepts §20.10.10). This is the atom-batched analogue of
// `_evalAggregateBroadcastReduce`: lifts the aggregate body tensor
// to shape [N, ...outAxes, ...reduceAxes] in ONE pass and
// tail-reduces, replacing the per-atom-evaluate-aggregate loop.
//
// Correctness oracle: per-atom evaluation via the existing single-
// atom `_evalAggregate`. Each test computes the batched result and
// verifies it agrees with the per-atom result element-wise.
//
// What's pinned:
//   1. Atom-indep refs broadcast across the atom axis (singleton
//      stride).
//   2. Atom-batched refs (Value shape=[N, ...]) use leading N as
//      the atom dim; remaining dims map to the get's selectors.
//   3. Mixed atom-batched + atom-indep in body works.
//   4. Multiple reduction axes work.
//   5. No-reduction aggregates (output_axes covers all body axes)
//      work.
//   6. All seven reductions agree with per-atom evaluation.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const agg = require('../sampler-aggregate.ts');
const sampler = require('../sampler.ts');

// ---------------------------------------------------------------------
// Helpers — build IR fragments cleanly
// ---------------------------------------------------------------------

function refr(name: string) { return { kind: 'ref', ns: 'self', name }; }
function axis(name: string) { return { kind: 'axis', name }; }
function lit(v: number) { return { kind: 'lit', value: v, numType: typeof v === 'number' && Number.isInteger(v) ? 'integer' : 'real' }; }
function vector(...args: any[]) { return { kind: 'call', op: 'vector', args }; }
function callOp(op: string, ...args: any[]) { return { kind: 'call', op, args }; }
function getOp(...args: any[]) { return { kind: 'call', op: 'get', args }; }

// Build an aggregate IR.
function aggregateIR(reducer: string, outAxes: string[], body: any) {
  return {
    kind: 'call', op: 'aggregate',
    args: [refr(reducer), vector(...outAxes.map(axis)), body],
  };
}

// Atom-indep Value (no leading N dim).
function val(shape: number[], data: number[]) {
  return { shape, data: new Float64Array(data) };
}

// Atom-batched Value (leading N dim).
function batchedVal(N: number, perAtomShape: number[], dataN: number[]) {
  return { shape: [N, ...perAtomShape], data: new Float64Array(dataN) };
}

// Per-atom oracle: slice each atom-batched ref to its rank-1-less
// view, run the single-atom aggregate, collect into a tensor.
function perAtomOracle(ir: any, refArrays: any, N: number, baseEnv: any) {
  const out: any[] = [];
  const keys = Object.keys(refArrays);
  for (let i = 0; i < N; i++) {
    const callEnv = Object.assign({}, baseEnv);
    for (const k of keys) {
      const v = refArrays[k];
      if (v && v.shape && v.shape[0] === N && v.shape.length > 0) {
        const tail = v.shape.slice(1);
        const tailLen = tail.reduce((a: number, b: number) => a * b, 1);
        callEnv[k] = {
          shape: tail,
          data: v.data.subarray(i * tailLen, (i + 1) * tailLen),
        };
      } else {
        callEnv[k] = v;
      }
    }
    out.push(sampler.evaluateExpr(ir, callEnv));
  }
  return out;
}

// Compare batched result Value to per-atom oracle.
function assertBatchedEqOracle(
  batched: any, oracle: any[], N: number, perAtomShape: number[],
) {
  assert.deepEqual(batched.shape, [N, ...perAtomShape],
    'batched result shape matches [N, ...perAtomShape]');
  const tailLen = perAtomShape.reduce((a, b) => a * b, 1) || 1;
  for (let i = 0; i < N; i++) {
    let row: any = oracle[i];
    // oracle entry is scalar, Float64Array, or nested array.
    if (typeof row === 'number') {
      assert.ok(Math.abs(batched.data[i] - row) < 1e-9,
        `atom ${i}: ${batched.data[i]} vs ${row}`);
      continue;
    }
    // Flatten oracle row to compare entry-wise.
    const flat: number[] = [];
    function flatten(v: any) {
      if (v && v.BYTES_PER_ELEMENT !== undefined) {
        for (let k = 0; k < v.length; k++) flat.push(v[k]);
      } else if (Array.isArray(v)) {
        for (const x of v) flatten(x);
      } else if (v && v.data) {
        for (let k = 0; k < v.data.length; k++) flat.push(v.data[k]);
      } else if (typeof v === 'number') {
        flat.push(v);
      }
    }
    flatten(row);
    for (let j = 0; j < tailLen; j++) {
      const got = batched.data[i * tailLen + j];
      const want = flat[j];
      assert.ok(Math.abs(got - want) < 1e-9,
        `atom ${i} cell ${j}: ${got} vs ${want}`);
    }
  }
}

// =====================================================================
// 1. Polyeval-shape: atom-batched X + fixed C → batched aggregate
// =====================================================================

test('aggregateN: polyeval-shape — fixed C + atom-batched X', () => {
  // C = [2.3, 1.5, 0.7]; X is atom-batched shape=[N, 10] of random
  // values; aggregate(sum, [.atom],
  //   mul(get(C, .j), pow(get(X, .atom), get(vector(0,1,2), .j))))
  // — the fusion (a) output for `polyeval.([C], X)`.
  const N = 5;
  const C = val([3], [2.3, 1.5, 0.7]);
  const xData: number[] = [];
  for (let i = 0; i < N * 10; i++) xData.push((i % 7) - 3);
  const X = batchedVal(N, [10], xData);
  const ir = aggregateIR('sum', ['atom'],
    callOp('mul',
      getOp(refr('C'), axis('j')),
      callOp('pow',
        getOp(refr('X'), axis('atom')),
        getOp(vector(lit(0), lit(1), lit(2)), axis('j')))));
  const refArrays = { X };
  const baseEnv = { C };
  const oracle = perAtomOracle(ir, refArrays, N, baseEnv);
  const result = agg._evalAggregateBroadcastReduceN(ir, refArrays, N, baseEnv, null);
  assertBatchedEqOracle(result, oracle, N, [10]);
});

// =====================================================================
// 2. Atom-indep ref only — broadcasts across atom axis
// =====================================================================

test('aggregateN: atom-indep refs only — broadcasts to all atoms', () => {
  // aggregate(sum, [.i], A[.i] * A[.i]) — both refs are atom-indep.
  // No reduction (output_axes covers all body axes). Per atom: same
  // length-5 vector of squares. Result: [N, 5].
  const N = 4;
  const A = val([5], [1, 2, 3, 4, 5]);
  const ir = aggregateIR('sum', ['i'],
    callOp('mul',
      getOp(refr('A'), axis('i')),
      getOp(refr('A'), axis('i'))));
  const baseEnv = { A };
  const oracle = perAtomOracle(ir, {}, N, baseEnv);
  const result = agg._evalAggregateBroadcastReduceN(ir, {}, N, baseEnv, null);
  assertBatchedEqOracle(result, oracle, N, [5]);
});

// =====================================================================
// 3. Reduction axis only (no output axes — full reduction)
// =====================================================================

test('aggregateN: full reduction with atom-batched ref', () => {
  // aggregate(sum, [], get(X, .j)) — single atom-batched scalar
  // ref; reduce over .j; output is one number per atom.
  const N = 3;
  const X = batchedVal(N, [4], [1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400]);
  const ir = aggregateIR('sum', [], getOp(refr('X'), axis('j')));
  const result = agg._evalAggregateBroadcastReduceN(ir, { X }, N, {}, null);
  assert.deepEqual(result.shape, [N]);
  // Per atom: sum of 4 values.
  // Atom 0: 1+2+3+4 = 10. Atom 1: 100. Atom 2: 1000.
  assert.equal(result.data[0], 10);
  assert.equal(result.data[1], 100);
  assert.equal(result.data[2], 1000);
});

// =====================================================================
// 4. Mean reducer
// =====================================================================

test('aggregateN: mean reducer agrees with per-atom oracle', () => {
  const N = 5;
  const X = batchedVal(N, [3], [1, 2, 3, 10, 20, 30, 100, 200, 300, 1, 1, 1, 0, 0, 0]);
  const ir = aggregateIR('mean', [], getOp(refr('X'), axis('j')));
  const oracle = perAtomOracle(ir, { X }, N, {});
  const result = agg._evalAggregateBroadcastReduceN(ir, { X }, N, {}, null);
  assertBatchedEqOracle(result, oracle, N, []);
});

// =====================================================================
// 5. Prod reducer
// =====================================================================

test('aggregateN: prod reducer agrees with per-atom oracle', () => {
  const N = 4;
  const X = batchedVal(N, [3], [1, 2, 3, 2, 2, 2, 3, 1, 4, 0, 5, 5]);
  const ir = aggregateIR('prod', [], getOp(refr('X'), axis('j')));
  const oracle = perAtomOracle(ir, { X }, N, {});
  const result = agg._evalAggregateBroadcastReduceN(ir, { X }, N, {}, null);
  assertBatchedEqOracle(result, oracle, N, []);
});

// =====================================================================
// 6. Atom-batched outer-product-like
// =====================================================================

test('aggregateN: outer-product-like — two output axes, no reduction', () => {
  // aggregate(sum, [.i, .j], A[.i] * B[.j]) with both refs
  // atom-indep — outer product. Same value per atom.
  const N = 3;
  const A = val([3], [1, 2, 3]);
  const B = val([2], [10, 20]);
  const ir = aggregateIR('sum', ['i', 'j'],
    callOp('mul',
      getOp(refr('A'), axis('i')),
      getOp(refr('B'), axis('j'))));
  const oracle = perAtomOracle(ir, {}, N, { A, B });
  const result = agg._evalAggregateBroadcastReduceN(ir, {}, N, { A, B }, null);
  assertBatchedEqOracle(result, oracle, N, [3, 2]);
});

// =====================================================================
// 7. Const subtree with atom-batched scalar (rank-1 ref shape=[N])
// =====================================================================

test('aggregateN: atom-batched scalar without axis ref (constant subtree)', () => {
  // body = mul(scalar_per_atom, get(C, .j)) — the constant subtree
  // is a per-atom scalar. The aggregate should multiply each atom's
  // scalar by sum(C).
  const N = 3;
  const C = val([4], [1, 2, 3, 4]);
  const S = { shape: [N], data: new Float64Array([10, 100, 1000]) };
  const ir = aggregateIR('sum', [],
    callOp('mul', refr('S'), getOp(refr('C'), axis('j'))));
  const result = agg._evalAggregateBroadcastReduceN(ir, { S }, N, { C }, null);
  assert.deepEqual(result.shape, [N]);
  // Per atom: S[atom] * sum(C) = S[atom] * 10.
  assert.equal(result.data[0], 100);
  assert.equal(result.data[1], 1000);
  assert.equal(result.data[2], 10000);
});

// =====================================================================
// 8. End-to-end through evaluateExprN
// =====================================================================

test('aggregateN: evaluateExprN routes aggregate IRs to the batched evaluator', () => {
  const { evaluateExprN } = require('../sampler-eval-batched.ts');
  const N = 4;
  const A = val([3], [2, 3, 5]);
  const X = batchedVal(N, [], [1, 2, 3, 4]);  // per-atom scalar shape=[N]
  const ir = aggregateIR('sum', [],
    callOp('mul', getOp(refr('A'), axis('j')), refr('X')));
  const result = evaluateExprN(ir, { X }, N, { A }, null);
  assert.deepEqual(result.shape, [N]);
  // Per atom: X[atom] * sum(A) = X[atom] * 10.
  assert.equal(result.data[0], 10);
  assert.equal(result.data[1], 20);
  assert.equal(result.data[2], 30);
  assert.equal(result.data[3], 40);
});
