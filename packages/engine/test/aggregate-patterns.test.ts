'use strict';

// =====================================================================
// aggregate-patterns.test.ts — P5 single pattern catalogue
// =====================================================================
//
// Pins the contract added by P5 of the broadcast/aggregate/batching
// consolidation (TODO-flatppl-js.md "In-flight P1-P9"):
//
//   - `aggregate-patterns.classifyMatmulBody(bodyIR, outAxes)` is
//     the SINGLE structural recogniser shared by the dissolver
//     (`dissolver._tryDissolveAggregate`) and the runtime
//     `AGGREGATE_PATTERNS` table.
//   - Recognises matmul (4 transpose variants), matvec (2 variants),
//     outer product.
//   - Returns null for non-matching shapes; consumers fall back to
//     their generic path.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const patterns = require('../aggregate-patterns.ts');

function axis(name: string) { return { kind: 'axis', name }; }
function refr(name: string) { return { kind: 'ref', ns: 'self', name }; }
function getOp(arr: any, ...sels: any[]) {
  return { kind: 'call', op: 'get', args: [arr, ...sels] };
}
function mul(a: any, b: any) {
  return { kind: 'call', op: 'mul', args: [a, b] };
}

// =====================================================================
// 1. Matmul recognition (4 transpose variants)
// =====================================================================

test('classifyMatmulBody: A[.i,.j] * B[.j,.k] → matmul, no transpose', () => {
  const body = mul(
    getOp(refr('A'), axis('i'), axis('j')),
    getOp(refr('B'), axis('j'), axis('k')),
  );
  const cls = patterns.classifyMatmulBody(body, [axis('i'), axis('k')]);
  assert.equal(cls.kind, 'matmul');
  assert.equal(cls.transA, false);
  assert.equal(cls.transB, false);
  assert.equal(cls.jName, 'j');
});

test('classifyMatmulBody: A[.j,.i] * B[.j,.k] → matmul, transposeA', () => {
  const body = mul(
    getOp(refr('A'), axis('j'), axis('i')),
    getOp(refr('B'), axis('j'), axis('k')),
  );
  const cls = patterns.classifyMatmulBody(body, [axis('i'), axis('k')]);
  assert.equal(cls.kind, 'matmul');
  assert.equal(cls.transA, true);
  assert.equal(cls.transB, false);
});

test('classifyMatmulBody: A[.i,.j] * B[.k,.j] → matmul, transposeB', () => {
  const body = mul(
    getOp(refr('A'), axis('i'), axis('j')),
    getOp(refr('B'), axis('k'), axis('j')),
  );
  const cls = patterns.classifyMatmulBody(body, [axis('i'), axis('k')]);
  assert.equal(cls.kind, 'matmul');
  assert.equal(cls.transA, false);
  assert.equal(cls.transB, true);
});

test('classifyMatmulBody: factor swap (B*A) still matches matmul', () => {
  // Scalar mul is commutative; classifier tries both orderings.
  const body = mul(
    getOp(refr('B'), axis('j'), axis('k')),
    getOp(refr('A'), axis('i'), axis('j')),
  );
  const cls = patterns.classifyMatmulBody(body, [axis('i'), axis('k')]);
  assert.equal(cls.kind, 'matmul');
});

// =====================================================================
// 2. Matvec recognition
// =====================================================================

test('classifyMatmulBody: A[.i,.j] * v[.j] → matvec', () => {
  const body = mul(
    getOp(refr('A'), axis('i'), axis('j')),
    getOp(refr('v'), axis('j')),
  );
  const cls = patterns.classifyMatmulBody(body, [axis('i')]);
  assert.equal(cls.kind, 'matvec');
  assert.equal(cls.transA, false);
  assert.equal(cls.jName, 'j');
});

test('classifyMatmulBody: A[.j,.i] * v[.j] → matvec, transposeA', () => {
  const body = mul(
    getOp(refr('A'), axis('j'), axis('i')),
    getOp(refr('v'), axis('j')),
  );
  const cls = patterns.classifyMatmulBody(body, [axis('i')]);
  assert.equal(cls.kind, 'matvec');
  assert.equal(cls.transA, true);
});

// =====================================================================
// 3. Outer product recognition
// =====================================================================

test('classifyMatmulBody: u[.i] * v[.j] → outer', () => {
  const body = mul(
    getOp(refr('u'), axis('i')),
    getOp(refr('v'), axis('j')),
  );
  const cls = patterns.classifyMatmulBody(body, [axis('i'), axis('j')]);
  assert.equal(cls.kind, 'outer');
});

// =====================================================================
// 4. Non-matches return null
// =====================================================================

test('classifyMatmulBody: non-mul body → null', () => {
  const body = { kind: 'call', op: 'add', args: [
    getOp(refr('A'), axis('i'), axis('j')),
    getOp(refr('B'), axis('j'), axis('k')),
  ]};
  assert.equal(patterns.classifyMatmulBody(body, [axis('i'), axis('k')]), null);
});

test('classifyMatmulBody: matmul body but outAxes have wrong shape → null', () => {
  // Body looks like matmul but the outer aggregate's output_axes is
  // [.i, .i] (duplicate) — caller should reject earlier; classifier
  // refuses too.
  const body = mul(
    getOp(refr('A'), axis('i'), axis('j')),
    getOp(refr('B'), axis('j'), axis('k')),
  );
  assert.equal(patterns.classifyMatmulBody(body, [axis('i'), axis('i')]), null);
});
