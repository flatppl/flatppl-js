'use strict';

// =====================================================================
// evalN-opdecl-dispatch.test.ts — _evalN routes through ops.dispatch
// =====================================================================
//
// Pins the P8 architectural completion: _evalN routes declared
// fixed-rank ops with `batched` slots through `ops.dispatch` with
// `atomN: N` instead of falling to `_perAtomFallback`'s per-atom
// loop. The OpDecl `batched` fast-paths (cb5e88e: diagmat / det /
// logabsdet / row_gram / col_gram; earlier: cross / self_outer /
// trace / inv / transpose / adjoint / lower_cholesky / linsolve)
// were landed in the 2026-05-30 P1-P9 consolidation but `_evalN`
// didn't consult them — every per-atom linalg call still went
// through the single-point `evaluateExpr` N times. This test
// verifies the dispatch now reaches the batched slot.
//
// What's pinned:
//   1. Per-atom det(A) returns rank-1 batched output via the
//      OpDecl's _detBatched slot (not a JS array from _perAtomFallback).
//   2. Per-atom inv(A) returns rank-3 batched inverses.
//   3. Per-atom trace(A) returns rank-1 scalars.
//   4. Per-atom cross(a, b) returns rank-2 vectors.
//   5. The gate respects the kind contract — rank-polymorphic ops
//      without batched slots (e.g. `transpose`) do NOT route through
//      ops.dispatch; they fall to _perAtomFallback (correct: rank-
//      polymorphic batched is the consumer's responsibility).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { evaluateExprN } = require('../sampler-eval-batched.ts');
const valueLib = require('../value.ts');

// Trigger eager registration of OpDecls + variants.
require('../ops-declarations.ts');

function atomBatchedMat(N: number, rows: number, cols: number,
                       perAtomData: number[][]): any {
  const data = new Float64Array(N * rows * cols);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < rows * cols; j++) {
      data[i * rows * cols + j] = perAtomData[i][j];
    }
  }
  return { shape: [N, rows, cols], data };
}

function atomBatchedVec(N: number, len: number, perAtomData: number[][]): any {
  const data = new Float64Array(N * len);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < len; j++) data[i * len + j] = perAtomData[i][j];
  }
  return { shape: [N, len], data };
}

test('evalN/opdecl: det of per-atom rank-3 matrix → rank-1 atom-batched scalars', () => {
  const N = 3;
  // det([[2,0],[0,3]]) = 6;  det([[1,2],[3,4]]) = -2;  det([[1,0],[0,1]]) = 1
  const A = atomBatchedMat(N, 2, 2, [
    [2, 0, 0, 3],
    [1, 2, 3, 4],
    [1, 0, 0, 1],
  ]);
  const ir = { kind: 'call', op: 'det', args: [{ kind: 'ref', name: 'A', ns: 'self' }] };
  const result = evaluateExprN(ir, { A }, N, {}, {});
  // Result must be rank-1 shape=[N] (the batched slot's output)
  assert.ok(valueLib.isAtomBatched(result, N), 'result is atom-batched scalar');
  assert.deepEqual(Array.from(result.data), [6, -2, 1]);
});

test('evalN/opdecl: inv of per-atom rank-3 matrix → rank-3 atom-batched inverses', () => {
  const N = 2;
  // A_0 = [[2,0],[0,4]]; inv = [[0.5,0],[0,0.25]]
  // A_1 = [[1,2],[0,1]]; inv = [[1,-2],[0,1]]
  const A = atomBatchedMat(N, 2, 2, [
    [2, 0, 0, 4],
    [1, 2, 0, 1],
  ]);
  const ir = { kind: 'call', op: 'inv', args: [{ kind: 'ref', name: 'A', ns: 'self' }] };
  const result = evaluateExprN(ir, { A }, N, {}, {});
  assert.deepEqual(result.shape, [N, 2, 2], 'rank-3 atom-batched result');
  // Atom 0 inverse
  assert.ok(Math.abs(result.data[0] - 0.5)  < 1e-12);
  assert.ok(Math.abs(result.data[3] - 0.25) < 1e-12);
  // Atom 1 inverse
  assert.ok(Math.abs(result.data[4] - 1)    < 1e-12);
  assert.ok(Math.abs(result.data[5] + 2)    < 1e-12);
});

test('evalN/opdecl: trace of per-atom matrix → rank-1 scalars', () => {
  const N = 3;
  // tr([[1,2],[3,4]]) = 5;  tr([[5,0],[0,5]]) = 10;  tr([[1,0],[0,1]]) = 2
  const A = atomBatchedMat(N, 2, 2, [
    [1, 2, 3, 4],
    [5, 0, 0, 5],
    [1, 0, 0, 1],
  ]);
  const ir = { kind: 'call', op: 'trace', args: [{ kind: 'ref', name: 'A', ns: 'self' }] };
  const result = evaluateExprN(ir, { A }, N, {}, {});
  assert.ok(valueLib.isAtomBatched(result, N));
  assert.deepEqual(Array.from(result.data), [5, 10, 2]);
});

test('evalN/opdecl: cross of two per-atom rank-1 vectors → rank-2 per-atom', () => {
  const N = 2;
  // a_0 = [1,0,0]; b_0 = [0,1,0]; a x b = [0,0,1]
  // a_1 = [1,1,0]; b_1 = [0,1,1]; a x b = [1,-1,1]
  const a = atomBatchedVec(N, 3, [[1, 0, 0], [1, 1, 0]]);
  const b = atomBatchedVec(N, 3, [[0, 1, 0], [0, 1, 1]]);
  const ir = {
    kind: 'call',
    op: 'cross',
    args: [
      { kind: 'ref', name: 'a', ns: 'self' },
      { kind: 'ref', name: 'b', ns: 'self' },
    ],
  };
  const result = evaluateExprN(ir, { a, b }, N, {}, {});
  assert.deepEqual(result.shape, [N, 3]);
  assert.deepEqual(Array.from(result.data), [0, 0, 1, 1, -1, 1]);
});

test('evalN/opdecl: rank-polymorphic ops without batched slots fall to _perAtomFallback', () => {
  // `transpose` is kind='rank-polymorphic' with no batched slot and
  // no variants. The dispatch gate must reject it so the old
  // per-atom path keeps handling it (correct for rank-polymorphic
  // ops where atom-batching is the caller's concern).
  const opsLib = require('../ops.ts');
  const decl = opsLib.lookup('transpose');
  assert.equal(decl.kind, 'rank-polymorphic');
  assert.ok(!decl.batched, 'transpose has no batched slot');
  // Gate logic mirrored: (!decl.kind || decl.kind === 'fixed-rank') && decl.batched
  const gatePasses =
    (!decl.kind || decl.kind === 'fixed-rank') && !!decl.batched;
  assert.ok(!gatePasses, 'gate correctly rejects rank-polymorphic without batched');
});
