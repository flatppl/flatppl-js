'use strict';

// =====================================================================
// batched-matmul-aggregate.test.ts — Phase 2.2 specialiser dispatch
// =====================================================================
//
// Pins the AGGREGATE_PATTERNS → atom-batched matmul dispatch path
// (sampler-aggregate.ts → ops.mul atom-aware variant → _matBatchedMatMul).
//
// The conformance oracle is per-atom evaluation: for each atom i,
// evaluate `aggregate(sum, [.i, .k], A[i][.i, .j] * B[i][.j, .k])`
// and check the batched result equals the stack of per-atom results.
//
// Three input patterns reach the same vectorised matmul:
//
//   - A=[N, m, n] × B=[N, n, p]    → per-atom matmul, both per-atom
//   - A=[N, m, n] × B=[n, p]       → per-atom matmul, shared B
//   - A=[m, n]    × B=[N, n, p]    → per-atom matmul, shared A
//
// All three should produce shape [N, m, p].
//
// Without the Phase 2.2 atomN threading the AGGREGATE_PATTERNS
// matmul-family specialiser routed through `valueOps.mul(aV, bV)`
// (no atomN), which sees rank-3 × rank-3 operands and falls through
// (throws) — the test would then have hit the generic broadcast-
// reduce path, which is correct but slow. With the fix, the
// matmul-family specialiser routes through the atom-aware variant
// directly.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { evaluateExprN } = require('../sampler-eval-batched.ts');

// ---------------------------------------------------------------------
// Helpers — build IR fragments
// ---------------------------------------------------------------------

function refr(name: string) { return { kind: 'ref', ns: 'self', name }; }
function axis(name: string) { return { kind: 'axis', name }; }
function vector(...args: any[]) { return { kind: 'call', op: 'vector', args }; }
function callOp(op: string, ...args: any[]) { return { kind: 'call', op, args }; }
function getOp(...args: any[]) { return { kind: 'call', op: 'get', args }; }

// `aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])`
function matmulAggIR(aName: string, bName: string) {
  return {
    kind: 'call', op: 'aggregate',
    args: [
      refr('sum'),
      vector(axis('i'), axis('k')),
      callOp('mul',
        getOp(refr(aName), axis('i'), axis('j')),
        getOp(refr(bName), axis('j'), axis('k'))),
    ],
  };
}

function val(shape: number[], data: number[]) {
  return { shape, data: new Float64Array(data) };
}

function batchedVal(N: number, perAtomShape: number[], dataN: number[]) {
  return { shape: [N, ...perAtomShape], data: new Float64Array(dataN) };
}

// Reference matmul (m×n) · (n×p) → (m×p), reading row-major.
function refMatmul(
  A: Float64Array, aShape: number[],
  B: Float64Array, bShape: number[],
): Float64Array {
  const m = aShape[0], n = aShape[1], n2 = bShape[0], p = bShape[1];
  if (n !== n2) throw new Error('refMatmul: shape mismatch');
  const out = new Float64Array(m * p);
  for (let i = 0; i < m; i++) {
    for (let k = 0; k < p; k++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += A[i * n + j] * B[j * p + k];
      out[i * p + k] = s;
    }
  }
  return out;
}

// ---------------------------------------------------------------------
// Pattern 1: both atom-batched (A=[N, m, n] × B=[N, n, p])
// ---------------------------------------------------------------------

test('batched-matmul-aggregate: both factors atom-batched → [N, m, p]', () => {
  const N = 4, m = 3, n = 2, p = 5;
  const aData = new Float64Array(N * m * n);
  for (let i = 0; i < aData.length; i++) aData[i] = (i % 7) - 3;
  const bData = new Float64Array(N * n * p);
  for (let i = 0; i < bData.length; i++) bData[i] = ((i * 3) % 11) - 5;
  const A = batchedVal(N, [m, n], Array.from(aData));
  const B = batchedVal(N, [n, p], Array.from(bData));

  const ir = matmulAggIR('A', 'B');
  const result = evaluateExprN(ir, { A, B }, N, {}, null);

  assert.deepEqual(result.shape, [N, m, p],
    'output shape must be [N, m, p]');

  // Per-atom oracle.
  for (let i = 0; i < N; i++) {
    const aSlice = aData.subarray(i * m * n, (i + 1) * m * n);
    const bSlice = bData.subarray(i * n * p, (i + 1) * n * p);
    const expected = refMatmul(aSlice, [m, n], bSlice, [n, p]);
    for (let c = 0; c < m * p; c++) {
      const got = result.data[i * m * p + c];
      const want = expected[c];
      assert.ok(Math.abs(got - want) < 1e-9,
        `atom ${i} cell ${c}: got ${got}, want ${want}`);
    }
  }
});

// ---------------------------------------------------------------------
// Pattern 2: A atom-batched, B shared atom-indep (A=[N, m, n] × B=[n, p])
// ---------------------------------------------------------------------

test('batched-matmul-aggregate: A atom-batched, B shared → [N, m, p]', () => {
  const N = 3, m = 2, n = 4, p = 3;
  const aData = new Float64Array(N * m * n);
  for (let i = 0; i < aData.length; i++) aData[i] = (i % 5) - 2;
  const bData = new Float64Array(n * p);
  for (let i = 0; i < bData.length; i++) bData[i] = ((i * 7) % 13) - 6;
  const A = batchedVal(N, [m, n], Array.from(aData));
  const B = val([n, p], Array.from(bData));

  const ir = matmulAggIR('A', 'B');
  const result = evaluateExprN(ir, { A }, N, { B }, null);

  assert.deepEqual(result.shape, [N, m, p],
    'output shape must be [N, m, p]');

  for (let i = 0; i < N; i++) {
    const aSlice = aData.subarray(i * m * n, (i + 1) * m * n);
    const expected = refMatmul(aSlice, [m, n], bData, [n, p]);
    for (let c = 0; c < m * p; c++) {
      const got = result.data[i * m * p + c];
      const want = expected[c];
      assert.ok(Math.abs(got - want) < 1e-9,
        `atom ${i} cell ${c}: got ${got}, want ${want}`);
    }
  }
});

// ---------------------------------------------------------------------
// Pattern 3: A shared atom-indep, B atom-batched (A=[m, n] × B=[N, n, p])
// ---------------------------------------------------------------------

test('batched-matmul-aggregate: A shared, B atom-batched → [N, m, p]', () => {
  const N = 5, m = 3, n = 2, p = 4;
  const aData = new Float64Array(m * n);
  for (let i = 0; i < aData.length; i++) aData[i] = (i % 4) - 1;
  const bData = new Float64Array(N * n * p);
  for (let i = 0; i < bData.length; i++) bData[i] = ((i * 5) % 11) - 5;
  const A = val([m, n], Array.from(aData));
  const B = batchedVal(N, [n, p], Array.from(bData));

  const ir = matmulAggIR('A', 'B');
  const result = evaluateExprN(ir, { B }, N, { A }, null);

  assert.deepEqual(result.shape, [N, m, p],
    'output shape must be [N, m, p]');

  for (let i = 0; i < N; i++) {
    const bSlice = bData.subarray(i * n * p, (i + 1) * n * p);
    const expected = refMatmul(aData, [m, n], bSlice, [n, p]);
    for (let c = 0; c < m * p; c++) {
      const got = result.data[i * m * p + c];
      const want = expected[c];
      assert.ok(Math.abs(got - want) < 1e-9,
        `atom ${i} cell ${c}: got ${got}, want ${want}`);
    }
  }
});

// ---------------------------------------------------------------------
// Pattern 4: stability under random shapes — sanity probe
// ---------------------------------------------------------------------

test('batched-matmul-aggregate: varied shapes — all match per-atom oracle', () => {
  // Several (N, m, n, p) combinations exercise different argRank
  // dispatches in _matBatchedMatMul. (No randomness — fixed seeds
  // via deterministic index math.)
  const cases = [
    [2, 1, 1, 1],   // degenerate — single-element matrices
    [3, 5, 3, 2],   // tall A × short B
    [4, 2, 7, 3],   // wide A × wide B
    [6, 4, 4, 4],   // all-square matrices
  ];
  for (const [N, m, n, p] of cases) {
    const aData = new Float64Array(N * m * n);
    for (let i = 0; i < aData.length; i++) aData[i] = (i % 6) - 2;
    const bData = new Float64Array(N * n * p);
    for (let i = 0; i < bData.length; i++) bData[i] = ((i * 11) % 17) - 8;
    const A = batchedVal(N, [m, n], Array.from(aData));
    const B = batchedVal(N, [n, p], Array.from(bData));
    const ir = matmulAggIR('A', 'B');
    const result = evaluateExprN(ir, { A, B }, N, {}, null);
    assert.deepEqual(result.shape, [N, m, p],
      `case [N=${N}, m=${m}, n=${n}, p=${p}]: shape mismatch`);
    for (let i = 0; i < N; i++) {
      const aSlice = aData.subarray(i * m * n, (i + 1) * m * n);
      const bSlice = bData.subarray(i * n * p, (i + 1) * n * p);
      const expected = refMatmul(aSlice, [m, n], bSlice, [n, p]);
      for (let c = 0; c < m * p; c++) {
        const got = result.data[i * m * p + c];
        const want = expected[c];
        assert.ok(Math.abs(got - want) < 1e-9,
          `case [N=${N}, m=${m}, n=${n}, p=${p}] atom ${i} cell ${c}: got ${got}, want ${want}`);
      }
    }
  }
});
