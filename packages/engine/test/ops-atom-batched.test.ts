'use strict';

// =====================================================================
// ops-atom-batched.test.ts — P1 atom-batched variant fast-paths
// =====================================================================
//
// Pins the per-variant `batched` slot + atom-aware variant matching
// added to `ops.dispatch` (engine-concepts §18.2 / TODO-flatppl-js
// P1 follow-up). Variants registered with a `batched: (args, N, ctx)`
// impl are picked when the caller signals atom-batching via
// `opts.atomN`. The matcher tries each variant with the leading atom
// dim stripped from args; for variants with static pattern.rank, an
// arg of rank pattern.rank+1 with leading dim == atomN is treated
// as atom-batched.
//
// What's pinned:
//   1. add / sub atom-batched variants — asymmetric (mu_indep +
//      Lz_batched) routes through _atomBroadcastBinop.
//   2. mul atom-batched variant — matrix × atom-batched rank-1
//      routes through _matBatchedVecMul.
//   3. neg atom-batched variant — passthrough (neg works at any rank).
//   4. opts.atomN is REQUIRED for atom-aware routing — without it,
//      rank-2 atom-batched-vec args look like rank-2 matrices and
//      match the matmul variant (wrong impl).
//   5. The dispatcher prefers atom-aware over exact-rank when atomN
//      is set (shadowing protection for the rank-2-as-matrix case).
//   6. Mixed atom-batched + atom-indep with rank-mismatched atomN
//      refuses cleanly.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const ops      = require('../ops.ts');
const valueLib = require('../value.ts');
const valueOps = require('../value-ops.ts');

// Trigger eager registration of variants — the ops-declarations module
// runs _ensureAtomBatchedRegistered() at module load.
require('../ops-declarations.ts');

function vec(data: number[]): any {
  return { shape: [data.length], data: new Float64Array(data) };
}

function mat(rows: number, cols: number, data: number[]): any {
  return { shape: [rows, cols], data: new Float64Array(data) };
}

function arrEq(actual: any, expected: number[], eps = 1e-12) {
  assert.equal(actual.data.length, expected.length, 'data length matches');
  for (let i = 0; i < expected.length; i++) {
    const d = Math.abs(actual.data[i] - expected[i]);
    if (d > eps) {
      assert.fail(`element ${i}: got ${actual.data[i]}, want ${expected[i]} (Δ=${d})`);
    }
  }
}

// =====================================================================
// 1. add — atom-batched + atom-indep (mu + L·z shape)
// =====================================================================

test('add(rank-1, atom-batched rank-1): mu + atom_batched routes via _atomBroadcastBinop', () => {
  // mu shape=[3]; Lz shape=[2, 3] (N=2 atoms).
  // Expected: [[10+1, 20+2, 30+3], [40+1, 50+2, 60+3]] = [[11,22,33],[41,52,63]].
  const mu = vec([1, 2, 3]);
  const Lz = mat(2, 3, [10, 20, 30, 40, 50, 60]);
  const r = ops.dispatch('add', [mu, Lz], { atomN: 2 });
  assert.deepEqual(r.shape, [2, 3]);
  arrEq(r, [11, 22, 33, 41, 52, 63]);
});

test('add(atom-batched rank-1, rank-1): atom_batched + mu (swapped order)', () => {
  // Symmetric to the above with operand order flipped.
  const Lz = mat(2, 3, [10, 20, 30, 40, 50, 60]);
  const mu = vec([1, 2, 3]);
  const r = ops.dispatch('add', [Lz, mu], { atomN: 2 });
  assert.deepEqual(r.shape, [2, 3]);
  arrEq(r, [11, 22, 33, 41, 52, 63]);
});

test('add(atom-batched, atom-batched same shape): elementwise add', () => {
  // Both shape=[2, 3] with N=2. Expected: pairwise sum.
  const a = mat(2, 3, [1, 2, 3, 4, 5, 6]);
  const b = mat(2, 3, [10, 20, 30, 40, 50, 60]);
  const r = ops.dispatch('add', [a, b], { atomN: 2 });
  assert.deepEqual(r.shape, [2, 3]);
  arrEq(r, [11, 22, 33, 44, 55, 66]);
});

// =====================================================================
// 2. sub — atom-batched (non-commutative ordering)
// =====================================================================

test('sub(atom-batched, rank-1): batched - indep, order preserved', () => {
  // Lz shape=[2, 3]; mu shape=[3]. Expected: Lz - mu per atom.
  const Lz = mat(2, 3, [10, 20, 30, 40, 50, 60]);
  const mu = vec([1, 2, 3]);
  const r = ops.dispatch('sub', [Lz, mu], { atomN: 2 });
  arrEq(r, [9, 18, 27, 39, 48, 57]);
});

test('sub(rank-1, atom-batched): indep - batched, order preserved', () => {
  const mu = vec([100, 100, 100]);
  const Lz = mat(2, 3, [10, 20, 30, 40, 50, 60]);
  const r = ops.dispatch('sub', [mu, Lz], { atomN: 2 });
  arrEq(r, [90, 80, 70, 60, 50, 40]);
});

// =====================================================================
// 3. neg — atom-batched (rank-2 passthrough)
// =====================================================================

test('neg(atom-batched rank-1): negation works at any rank', () => {
  const Lz = mat(2, 3, [1, -2, 3, -4, 5, -6]);
  const r = ops.dispatch('neg', [Lz], { atomN: 2 });
  assert.deepEqual(r.shape, [2, 3]);
  arrEq(r, [-1, 2, -3, 4, -5, 6]);
});

// =====================================================================
// 4. mul — matrix × atom-batched rank-1 vec (MvNormal L·z hot path)
// =====================================================================

test('mul(rank-2 matrix, atom-batched rank-1 vec): routes via _matBatchedVecMul', () => {
  // L shape=[2, 2]; z shape=[3, 2] (N=3 atoms, length-2 per atom).
  // Each atom: L · z[atom].
  // L = [[1, 0], [2, 3]]; z[0]=[4,5], z[1]=[6,7], z[2]=[8,9].
  // L·z[0] = [1*4+0*5, 2*4+3*5] = [4, 23].
  // L·z[1] = [6, 33].
  // L·z[2] = [8, 43].
  const L = mat(2, 2, [1, 0, 2, 3]);
  const z = mat(3, 2, [4, 5, 6, 7, 8, 9]);
  const r = ops.dispatch('mul', [L, z], { atomN: 3 });
  assert.deepEqual(r.shape, [3, 2]);
  arrEq(r, [4, 23, 6, 33, 8, 43]);
});

// =====================================================================
// 5. opts.atomN gating — without it, atom-aware doesn't fire
// =====================================================================

test('without opts.atomN: rank-2 args match matmul variant (DIFFERENT semantics)', () => {
  // Same shapes as the mul test above, but WITHOUT opts.atomN. The
  // dispatcher routes to the matmul variant (rank-2 × rank-2),
  // computing L · z as matrix product (NOT atom-batched). This pins
  // that the atom-aware path is OPT-IN via opts.atomN.
  const L = mat(2, 2, [1, 0, 2, 3]);
  const z = mat(3, 2, [4, 5, 6, 7, 8, 9]);
  // L (2×2) × z (3×2) → matmul throws (dim mismatch). The variant
  // matcher should match the matmul variant first and the impl
  // throws the expected error.
  assert.throws(
    () => ops.dispatch('mul', [L, z]),
    (err: any) => /shape/i.test(err.message) || /dim/i.test(err.message));
});

// =====================================================================
// 6. add with opts.atomN but no atom-batched arg (all atom-indep)
// =====================================================================

test('opts.atomN with all atom-indep args: falls through to exact-rank match', () => {
  // Both args rank-1, opts.atomN set. atom-aware needs at least one
  // atom-batched arg (rank+1); none here → falls through to exact-
  // rank → matches the standard add(rank-1, rank-1) variant.
  const a = vec([1, 2, 3]);
  const b = vec([10, 20, 30]);
  const r = ops.dispatch('add', [a, b], { atomN: 5 });  // atomN=5 but neither is rank-2
  assert.deepEqual(r.shape, [3]);
  arrEq(r, [11, 22, 33]);
});

// =====================================================================
// 7. Mixed N error case — atom-batched args must share atomN
// =====================================================================

test('rank-2 arg with leading dim != opts.atomN: refuses (no match)', () => {
  // arg has shape=[5, 3] but opts.atomN=2. atom-aware can't match
  // (5 != 2); exact-rank can't match (add rank-1 variant expects
  // rank 1, got rank 2); other variants ([{}, {}] broadcasted)
  // don't apply for direct wrapping default. Should throw.
  const a = vec([1, 2, 3]);
  const b = mat(5, 3, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]);
  assert.throws(
    () => ops.dispatch('add', [a, b], { atomN: 2 }),
    /no variant matched|no declaration/i);
});

// =====================================================================
// 8. Equivalence with legacy value-ops mulN / addN / subN / negN
// =====================================================================

test('parity: ops.dispatch(mul, atomN) ≡ valueOps.mulN for the matBatchedVecMul case', () => {
  const L = mat(3, 3, [1, 0, 0, 2, 1, 0, 3, 2, 1]);
  const z = mat(4, 3, [
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
  ]);
  const viaDispatch = ops.dispatch('mul', [L, z], { atomN: 4 });
  const viaLegacy   = valueOps.mulN(L, z, 4);
  assert.deepEqual(viaDispatch.shape, viaLegacy.shape);
  arrEq(viaDispatch, Array.from(viaLegacy.data));
});

test('parity: ops.dispatch(add, atomN) ≡ valueOps.addN for the mu+Lz case', () => {
  const mu = vec([100, 200, 300]);
  const Lz = mat(4, 3, [
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    10, 11, 12,
  ]);
  const viaDispatch = ops.dispatch('add', [mu, Lz], { atomN: 4 });
  const viaLegacy   = valueOps.addN(mu, Lz, 4);
  assert.deepEqual(viaDispatch.shape, viaLegacy.shape);
  arrEq(viaDispatch, Array.from(viaLegacy.data));
});
