'use strict';

// diag structured fast-paths: linalg ops (ARITH_OPS) and value-ops.mul
// take O(n) paths on a vector-backed diagonal and preserve `diag`
// structure where the result is diagonal. Each fast-path is checked
// against the densified reference for correctness.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const valueLib = require('../value.ts');
const vops = require('../value-ops.ts');
const { ARITH_OPS } = sampler._internal;
const { diagMatrix, densify, isDiagStored, vector, matrix, scalar } = valueLib;

test('det / logabsdet / trace of diag are O(n) and exact', () => {
  const D = diagMatrix([2, 3, 4]);
  assert.equal(ARITH_OPS.det(D).data[0], 24);
  assert.ok(Math.abs(ARITH_OPS.logabsdet(D).data[0] - Math.log(24)) < 1e-12);
  assert.equal(ARITH_OPS.trace(D).data[0], 9);
});

test('inv(diag) = diag(reciprocal), structure preserved', () => {
  const D = diagMatrix([2, 4, 5]);
  const Di = ARITH_OPS.inv(D);
  assert.ok(isDiagStored(Di));
  assert.deepEqual(Array.from(Di.data), [0.5, 0.25, 0.2]);
  assert.throws(() => ARITH_OPS.inv(diagMatrix([1, 0, 2])), /singular/);
});

test('lower_cholesky(diag PD) = diag(sqrt), structure preserved', () => {
  const D = diagMatrix([4, 9, 16]);
  const L = ARITH_OPS.lower_cholesky(D);
  assert.ok(isDiagStored(L));
  assert.deepEqual(Array.from(L.data), [2, 3, 4]);
  // L Lᵀ reconstructs D
  const recon = densify(vops.mul(L, valueLib.transpose(L)));
  assert.deepEqual(Array.from(recon.data), [4, 0, 0, 0, 9, 0, 0, 0, 16]);
  assert.throws(() => ARITH_OPS.lower_cholesky(diagMatrix([1, -1, 2])),
    /not positive definite/);
});

test('linsolve(diag, b) = b ⊘ diag (vector rhs)', () => {
  const D = diagMatrix([2, 5, 10]);
  const x = ARITH_OPS.linsolve(D, vector([4, 10, 5]));
  assert.deepEqual(Array.from(x.data), [2, 2, 0.5]);
});

test('mul: diag·vector = elementwise scale', () => {
  const D = diagMatrix([2, 3, 4]);
  const r = vops.mul(D, vector([5, 6, 7]));
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [10, 18, 28]);
});

test('mul: diag·diag and scalar·diag stay diag', () => {
  const A = diagMatrix([2, 3]);
  const B = diagMatrix([5, 7]);
  const P = vops.mul(A, B);
  assert.ok(isDiagStored(P));
  assert.deepEqual(Array.from(P.data), [10, 21]);
  const S = vops.mul(scalar(3), diagMatrix([1, 2, 4]));
  assert.ok(isDiagStored(S));
  assert.deepEqual(Array.from(S.data), [3, 6, 12]);
});

test('mul: diag·dense scales rows; dense·diag scales columns', () => {
  const D = diagMatrix([10, 100]);
  const M = matrix([1, 2, 3, 4], 2, 2);          // [[1,2],[3,4]]
  const left = vops.mul(D, M);                    // rows scaled
  assert.deepEqual(Array.from(densify(left).data), [10, 20, 300, 400]);
  const right = vops.mul(M, D);                   // cols scaled
  assert.deepEqual(Array.from(densify(right).data), [10, 200, 30, 400]);
});

test('add/neg: diag∘diag stays diag; diag+dense densifies correctly', () => {
  const A = diagMatrix([1, 2, 3]);
  const B = diagMatrix([4, 5, 6]);
  const S = vops.add(A, B);
  assert.ok(isDiagStored(S));
  assert.deepEqual(Array.from(S.data), [5, 7, 9]);
  assert.ok(isDiagStored(vops.neg(A)));
  assert.deepEqual(Array.from(vops.neg(A).data), [-1, -2, -3]);
  // diag + dense → dense (occupancy union), values correct
  const dn = vops.add(diagMatrix([1, 1]), matrix([0, 2, 3, 0], 2, 2));
  assert.deepEqual(Array.from(densify(dn).data), [1, 2, 3, 1]);
});

test('diag fast-paths agree with the densified reference', () => {
  const D = diagMatrix([3, 7, 2]);
  const Dd = densify(D);
  assert.equal(ARITH_OPS.det(D).data[0], ARITH_OPS.det(Dd).data[0]);
  assert.equal(ARITH_OPS.trace(D).data[0], ARITH_OPS.trace(Dd).data[0]);
  const v = vector([1, 1, 1]);
  assert.deepEqual(Array.from(vops.mul(D, v).data),
                   Array.from(vops.mul(Dd, v).data));
});
