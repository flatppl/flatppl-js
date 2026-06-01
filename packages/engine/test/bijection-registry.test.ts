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

// =====================================================================
// 4. affineAtomBatchedInverse — z = L⁻¹·(y - b) conformance
// =====================================================================
//
// Density-side affine inverse via forward-substitution against L.
// Pinned by the round-trip property: inverse(forward(z)) ≈ z up to
// floating-point error.

test('affine.atomBatchedInverse: round-trip identity (forward then inverse)', () => {
  const D = 3;
  const N = 5;
  // Random-ish lower-triangular L with positive diag.
  const L = { shape: [D, D], data: new Float64Array(D * D) };
  for (let i = 0; i < D; i++) {
    for (let j = 0; j <= i; j++) {
      L.data[i * D + j] = (i === j) ? 1.5 + 0.5 * i : 0.3 + 0.2 * (i - j);
    }
  }
  const b = { shape: [D], data: new Float64Array([0.7, -1.3, 2.1]) };
  const zData = new Float64Array(N * D);
  for (let n = 0; n < N; n++) {
    for (let d = 0; d < D; d++) zData[n * D + d] = Math.cos((n + 1) * (d + 1));
  }
  const z = { shape: [N, D], data: zData };
  const y = bij.affineAtomBatchedForward(z, { L: L, b: b }, N);
  const zRoundtrip = bij.affineAtomBatchedInverse(y, { L: L, b: b }, N);
  assert.deepEqual(Array.from(zRoundtrip.shape), [N, D]);
  for (let i = 0; i < N * D; i++) {
    assert.ok(Math.abs(zRoundtrip.data[i] - zData[i]) < 1e-10,
      `roundtrip mismatch at flat-index ${i}: z=${zData[i]} z'=${zRoundtrip.data[i]}`);
  }
});

test('affine.atomBatchedInverse: solves L·z = (y - b) exactly (hand-built)', () => {
  // D=2: L = [[2, 0], [1, 3]], b = [10, 20].
  // y = [12, 21] → y - b = [2, 1] → L·z = [2, 1]
  //   forward-sub: z[0] = 2/2 = 1; z[1] = (1 - 1·1)/3 = 0 → z = [1, 0] ✓
  // y = [10, 23] → y - b = [0, 3] → L·z = [0, 3]
  //   z[0] = 0; z[1] = (3 - 0)/3 = 1 → z = [0, 1] ✓
  const D = 2;
  const N = 2;
  const L = { shape: [D, D], data: new Float64Array([2, 0, 1, 3]) };
  const b = { shape: [D], data: new Float64Array([10, 20]) };
  const y = { shape: [N, D], data: new Float64Array([12, 21,  10, 23]) };
  const z = bij.affineAtomBatchedInverse(y, { L: L, b: b }, N);
  assert.deepEqual(Array.from(z.shape), [N, D]);
  const expected = [1, 0,  0, 1];
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(z.data[i] - expected[i]) < 1e-12,
      `z[${i}] = ${z.data[i]} expected ${expected[i]}`);
  }
});

test('affine.atomBatchedInverse: rejects shape-mismatched y', () => {
  const D = 2;
  const L = { shape: [D, D], data: new Float64Array([1, 0, 0, 1]) };
  const b = { shape: [D], data: new Float64Array([0, 0]) };
  // y has wrong leading atom size.
  const yBad = { shape: [3, D], data: new Float64Array(6) };
  assert.throws(
    () => bij.affineAtomBatchedInverse(yBad, { L: L, b: b }, /*N=*/2),
    /expects y of shape \[N, D\]/);
});

test('affine.atomBatchedInverse: rejects zero on the diagonal', () => {
  const D = 2;
  const Lzero = { shape: [D, D], data: new Float64Array([0, 0, 1, 1]) };
  const b = { shape: [D], data: new Float64Array([0, 0]) };
  const y = { shape: [1, D], data: new Float64Array([1, 1]) };
  assert.throws(
    () => bij.affineAtomBatchedInverse(y, { L: Lzero, b: b }, 1),
    /zero diagonal/);
});

// =====================================================================
// 5. affineLogDetJ — -sum(log|diag(L)|)
// =====================================================================

test('affine.logDetJ: dense L returns -sum(log|diag(L)|)', () => {
  // L = [[2, 0, 0], [1, 3, 0], [4, 5, 6]] → diag = [2, 3, 6]
  // logDetJ = -(log 2 + log 3 + log 6) = -log(36)
  const D = 3;
  const L = { shape: [D, D], data: new Float64Array([
    2, 0, 0,
    1, 3, 0,
    4, 5, 6,
  ]) };
  const result = bij.affineLogDetJ(null, { L: L }, /*N=*/10);
  const expected = -(Math.log(2) + Math.log(3) + Math.log(6));
  assert.equal(typeof result, 'number',
    'affine logDetJ is constant in y → scalar return per registry contract');
  assert.ok(Math.abs(result - expected) < 1e-12,
    `logDetJ = ${result} expected ${expected}`);
});

test('affine.logDetJ: abs handles negative diag entries', () => {
  // For Cholesky-positive L this never happens, but the registry
  // contract is defensive: -sum(log|diag(L)|) — abs guards the log.
  const D = 2;
  const L = { shape: [D, D], data: new Float64Array([-2, 0, 0, 3]) };
  const result = bij.affineLogDetJ(null, { L: L }, 1);
  const expected = -(Math.log(2) + Math.log(3));
  assert.ok(Math.abs(result - expected) < 1e-12);
});

test('affine.logDetJ: rejects zero diagonal', () => {
  const D = 2;
  const Lzero = { shape: [D, D], data: new Float64Array([0, 0, 1, 1]) };
  assert.throws(
    () => bij.affineLogDetJ(null, { L: Lzero }, 1),
    /zero diagonal/);
});

test('affine.logDetJ: rejects non-square L', () => {
  const L = { shape: [2, 3], data: new Float64Array(6) };
  assert.throws(
    () => bij.affineLogDetJ(null, { L: L }, 1),
    /must be square/);
});

// =====================================================================
// 6. Density equivalence: registry path matches MvNormal density-prim
// =====================================================================
//
// The whole point of engine-concepts §22 is that the bijection registry
// IS the canonical decomposition. For MvNormal that decomposition is
//   log p_MvNormal(x | mu, Sigma)
//     = log p_iidNormal(L⁻¹(x - mu), n) + log|det J_{f⁻¹}(x)|
//     = (-½ n log(2π) - ½ ‖L⁻¹(x - mu)‖²) + (-sum log|diag(L)|)
//     = -½ n log(2π) - ½ ‖z‖² - sum log|diag(L)|
// where the per-bijection pieces (`z = L⁻¹(x-mu)` and `logDetJ =
// -sum log|diag(L)|`) come from the registry's affine entry. The test
// reproduces MvNormal's density at a few observation points through the
// registry and confirms it matches density-prims' closed-form
// implementation — concrete proof the registry is a complete sample-
// and density-side replacement for matMvNormal / walkMvNormal's hot
// code (the integration into walkPushfwd lands in Session 3+).

const densityPrims = require('../density-prims.ts');

function logIidNormalScalar(z: Float64Array, n: number): number {
  let lp = -0.5 * n * Math.log(2 * Math.PI);
  for (let i = 0; i < n; i++) lp -= 0.5 * z[i] * z[i];
  return lp;
}

test('affine density: registry-composed MvNormal score equals density-prim closed form', () => {
  // 3-dim MvNormal with a non-trivial cov; score at 4 distinct points
  // and confirm the registry-composed score matches density-prim.
  const n = 3;
  const mu = [0.5, -1.2, 2.0];
  // SPD cov: Sigma = L · Lᵀ with hand-built lower-triangular L.
  const Llo = [
    [1.4, 0,   0  ],
    [0.3, 1.1, 0  ],
    [0.2, 0.4, 0.9],
  ];
  const Sigma = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k <= Math.min(i, j); k++) s += Llo[i][k] * Llo[j][k];
      Sigma[i * n + j] = s;
    }
  }
  const cov = { shape: [n, n], data: Sigma };
  // density-prim closed-form (recomputes Cholesky internally).
  const muVec = { shape: [n], data: new Float64Array(mu) };
  const observations = [
    [0.5, -1.2, 2.0],   // = mu → max-density point
    [1.0, -0.5, 1.5],
    [-0.4, -2.0, 3.5],
    [2.2,  1.0, 0.1],
  ];
  // Registry path: build L and b once.
  const Ldata = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) Ldata[i * n + j] = Llo[i][j];
  }
  const L = { shape: [n, n], data: Ldata };
  const b = muVec;
  const logDetJ = bij.affineLogDetJ(null, { L: L }, /*N=*/1);
  for (const xVec of observations) {
    const expected = densityPrims.MV_DENSITY_FNS.MvNormal(
      { shape: [n], data: new Float64Array(xVec) },
      { mu: mu, cov: cov });
    // Registry path: y = x, compute z = affine.inverse(y, {L, mu}).
    const yMat = { shape: [1, n], data: new Float64Array(xVec) };
    const z = bij.affineAtomBatchedInverse(yMat, { L: L, b: b }, /*N=*/1);
    const lpIid = logIidNormalScalar(z.data, n);
    const lpRegistry = lpIid + logDetJ;
    assert.ok(Math.abs(lpRegistry - expected) < 1e-10,
      `registry MvNormal density mismatch at x=${JSON.stringify(xVec)}: `
      + `registry=${lpRegistry}, density-prim=${expected}`);
  }
});

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
