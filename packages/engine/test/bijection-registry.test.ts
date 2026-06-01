'use strict';

// =====================================================================
// bijection-registry.test.ts — Phase 5.1 Session 1
// =====================================================================
//
// Pins the open-bijection-registry surface added by Phase 5.1 of the
// broadcast/multivariate consolidation (engine-concepts §22). The
// registry is the single point of support for the bijection half of
// the universal multivariate decomposition
//   `M_multivariate = pushfwd(known_bijection, iid(scalar, D))`
// so multivariates lower compositionally instead of demanding per-dist
// engine codepaths.
//
// Session 1 lands:
//   - The registry CRUD surface (register / get / has /
//     registeredNames).
//   - The `affine` entry's sample-side atom-batched forward
//     (`y = L·z + b`) — same hot code matMvNormal already used inline.
//
// Density-side affine (`atomBatchedInverse` + `logDetJ`) lands in
// Phase 5.1 Session 2+ alongside matPushfwd vector-base support.

const { test } = require('node:test');
const assert = require('node:assert/strict');
require('../ops-declarations.ts');  // side-effect: register ops the
                                    // affine entry dispatches into
                                    // (mul + add atom-batched variants).
const bij = require('../bijection-registry.ts');
const valueLib = require('../value.ts');

// =====================================================================
// 1. Registry surface
// =====================================================================

test('registry: affine is registered after module load', () => {
  assert.equal(bij.hasBijection('affine'), true);
  const entry = bij.getBijection('affine');
  assert.ok(entry, 'getBijection returns the entry');
  assert.equal(entry.name, 'affine');
  assert.equal(typeof entry.atomBatchedForward, 'function');
  assert.deepEqual(entry.shapeContract, { inShape: '[D]', outShape: '[D]' });
});

test('registry: registeredNames includes affine', () => {
  const names = bij.registeredNames();
  assert.ok(names.includes('affine'),
    'affine appears in registeredNames(): ' + JSON.stringify(names));
});

test('registry: getBijection returns undefined for unknown names', () => {
  assert.equal(bij.getBijection('not_a_bijection'), undefined);
  assert.equal(bij.hasBijection('not_a_bijection'), false);
});

test('registry: duplicate registration throws', () => {
  assert.throws(
    () => bij.registerBijection({
      name: 'affine',          // already registered
      atomBatchedForward: () => null,
      shapeContract: { inShape: '[D]', outShape: '[D]' },
    }),
    /duplicate entry/);
});

test('registry: registration without a name throws', () => {
  assert.throws(
    () => bij.registerBijection({
      atomBatchedForward: () => null,
      shapeContract: { inShape: '[]', outShape: '[]' },
    } as any),
    /must have a name/);
});

// =====================================================================
// 2. affineAtomBatchedForward — y = L·z + b conformance
// =====================================================================

test('affine.atomBatchedForward: identity L + zero b acts as identity', () => {
  // L = I_3, b = 0_3. Output = input.
  const D = 3;
  const N = 4;
  const Lid = { shape: [D, D], data: new Float64Array(D * D) };
  for (let i = 0; i < D; i++) Lid.data[i * D + i] = 1;
  const b = { shape: [D], data: new Float64Array(D) };
  const zData = new Float64Array(N * D);
  for (let n = 0; n < N; n++) {
    for (let d = 0; d < D; d++) zData[n * D + d] = n * 10 + d + 1;
  }
  const z = { shape: [N, D], data: zData };
  const y = bij.affineAtomBatchedForward(z, { L: Lid, b: b }, N);
  assert.deepEqual(Array.from(y.shape), [N, D],
    'identity preserves shape [N, D]');
  for (let n = 0; n < N; n++) {
    for (let d = 0; d < D; d++) {
      assert.equal(y.data[n * D + d], zData[n * D + d],
        `identity L + zero b: y[${n}, ${d}] should equal z[${n}, ${d}]`);
    }
  }
});

test('affine.atomBatchedForward: dense L applied per atom + b shift', () => {
  // D=2, N=3, L = [[2, 0], [1, 3]], b = [10, 20].
  // z[0] = [1, 0] → L·z = [2, 1] + b = [12, 21]
  // z[1] = [0, 1] → L·z = [0, 3] + b = [10, 23]
  // z[2] = [1, 1] → L·z = [2, 4] + b = [12, 24]
  const D = 2;
  const N = 3;
  const L = { shape: [D, D], data: new Float64Array([2, 0, 1, 3]) };
  const b = { shape: [D], data: new Float64Array([10, 20]) };
  const z = { shape: [N, D], data: new Float64Array([1, 0,  0, 1,  1, 1]) };
  const y = bij.affineAtomBatchedForward(z, { L: L, b: b }, N);
  assert.deepEqual(Array.from(y.shape), [N, D]);
  const expected = [12, 21,  10, 23,  12, 24];
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(y.data[i] - expected[i]) < 1e-12,
      `y[${i}] = ${y.data[i]} expected ${expected[i]}`);
  }
});

test('affine.atomBatchedForward: throws when L is missing', () => {
  const z = { shape: [1, 2], data: new Float64Array([0, 0]) };
  const b = { shape: [2], data: new Float64Array([0, 0]) };
  assert.throws(
    () => bij.affineAtomBatchedForward(z, { b: b }, 1),
    /must include L and b/);
});

test('affine.atomBatchedForward: throws when b is missing', () => {
  const z = { shape: [1, 2], data: new Float64Array([0, 0]) };
  const L = { shape: [2, 2], data: new Float64Array([1, 0, 0, 1]) };
  assert.throws(
    () => bij.affineAtomBatchedForward(z, { L: L }, 1),
    /must include L and b/);
});

// =====================================================================
// 3. Equivalence with matMvNormal's pre-§22 inline code
// =====================================================================
//
// matMvNormal previously had this code path inline; Phase 5.1 Session 1
// extracted it into the registry's affine entry. The pin below verifies
// the registry produces results identical to a reference implementation
// of the same algorithm — guarding against future drift if either side
// is refactored.

test('affine.atomBatchedForward: matches a hand-rolled reference', () => {
  const D = 4;
  const N = 5;
  const L = { shape: [D, D], data: new Float64Array(D * D) };
  // Random-ish lower-triangular L.
  for (let i = 0; i < D; i++) {
    for (let j = 0; j <= i; j++) L.data[i * D + j] = 0.5 + 0.3 * (i + 1) + 0.1 * (j + 1);
  }
  const b = { shape: [D], data: new Float64Array([1, -2, 3, -4]) };
  const zData = new Float64Array(N * D);
  for (let n = 0; n < N; n++) {
    for (let d = 0; d < D; d++) zData[n * D + d] = Math.sin(n * D + d);
  }
  const z = { shape: [N, D], data: zData };
  const y = bij.affineAtomBatchedForward(z, { L: L, b: b }, N);
  // Reference: per atom n, y[n, :] = L·z[n, :] + b. L row-major.
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < D; i++) {
      let acc = b.data[i];
      for (let j = 0; j < D; j++) acc += L.data[i * D + j] * zData[n * D + j];
      assert.ok(Math.abs(y.data[n * D + i] - acc) < 1e-10,
        `mismatch at n=${n}, i=${i}: registry=${y.data[n * D + i]}, ref=${acc}`);
    }
  }
});
