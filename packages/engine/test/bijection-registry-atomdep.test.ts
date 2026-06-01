'use strict';

// =====================================================================
// bijection-registry-atomdep.test.ts — Phase 5.1 Session 5f-2
// =====================================================================
//
// Pins the affine registry entry's ATOM-DEPENDENT param support added
// by Session 5f-2 (engine-concepts §22): the forward / inverse / logDetJ
// now accept atom-batched `b` ([N, D]) and `L` ([N, D, D]) in addition
// to the atom-independent `b` ([D]) / `L` ([D, D]) forms, so hierarchical
// MvNormal priors (mu and/or scale varying per atom) lower through the
// same `pushfwd(affine, iid(Normal, D))` decomposition.
//
// Risk register targets (from the 5f scout plan):
//   R1 — per-atom dense inverse forward-substitution indexing
//        (Ld[Ln+i*D+j] atom-strided L; out[baseN+j] atom-strided z).
//   R2 — atom-indep stride-0 reuse must be byte-identical to pre-5f-2.
//   R3 — logDetJ per-atom sign + Float64Array(N) shape.
//   R6 — forward per-atom matvec (the generic mul-atomN dispatch does
//        NOT handle rank-3 L; the registry does it explicitly).

const { test } = require('node:test');
const assert = require('node:assert/strict');
require('../ops-declarations.ts');  // side-effect: register mul/add ops

const bij = require('../bijection-registry.ts');

const entry = bij.getBijection('affine');

function val(shape: number[], data: number[]) {
  return { shape, data: Float64Array.from(data) };
}
function arrEq(a: any, b: number[], eps = 1e-12) {
  assert.equal(a.length, b.length, 'length');
  for (let i = 0; i < b.length; i++) {
    assert.ok(Math.abs(a[i] - b[i]) < eps,
      `idx ${i}: ${a[i]} vs ${b[i]}`);
  }
}

// =====================================================================
// R6 — atom-batched forward (per-atom matvec)
// =====================================================================

test('5f-2 forward: atom-batched L [N,D,D] + b [N,D] → per-atom y = L[n]·z[n] + b[n]', () => {
  const N = 3, D = 2;
  // atom 0: L=[[1,0],[0,2]], b=[0,0],   z=[1,2] → [1, 4]
  // atom 1: L=[[2,0],[0.5,3]], b=[10,-1], z=[3,4] → [2*3+10, 0.5*3+3*4-1] = [16, 12.5]
  // atom 2: L=[[3,0],[0,4]], b=[20,-2], z=[5,6] → [3*5+20, 4*6-2] = [35, 22]
  const L = val([N, D, D], [1, 0, 0, 2, 2, 0, 0.5, 3, 3, 0, 0, 4]);
  const b = val([N, D], [0, 0, 10, -1, 20, -2]);
  const z = { shape: [N, D], data: Float64Array.from([1, 2, 3, 4, 5, 6]), outerRank: 1 };
  const y = entry.atomBatchedForward(z, { L, b }, N);
  assert.deepEqual(Array.from(y.shape), [N, D]);
  arrEq(y.data, [1, 4, 16, 12.5, 35, 22]);
});

test('5f-2 forward: atom-indep L [D,D] + b [D] unchanged (regression)', () => {
  const N = 2, D = 2;
  // shared L=[[2,0],[0.5,3]], b=[1,-1]; atom0 z=[1,1]→[2+1, 0.5+3-1]=[3,2.5]
  //                                      atom1 z=[2,0]→[4+1, 1-1]=[5,0]
  const L = val([D, D], [2, 0, 0.5, 3]);
  const b = val([D], [1, -1]);
  const z = { shape: [N, D], data: Float64Array.from([1, 1, 2, 0]), outerRank: 1 };
  const y = entry.atomBatchedForward(z, { L, b }, N);
  arrEq(y.data, [3, 2.5, 5, 0]);
});

// =====================================================================
// R1 + R2 — atom-batched inverse + round-trip + atom-indep regression
// =====================================================================

test('5f-2 inverse: round-trip inverse(forward(z)) ≈ z for atom-batched L + b', () => {
  const N = 3, D = 2;
  const L = val([N, D, D], [1, 0, 0, 2, 2, 0, 0.5, 3, 3, 0, 0, 4]);
  const b = val([N, D], [0, 0, 10, -1, 20, -2]);
  const zOrig = [1, 2, 3, 4, 5, 6];
  const z = { shape: [N, D], data: Float64Array.from(zOrig), outerRank: 1 };
  const y = entry.atomBatchedForward(z, { L, b }, N);
  const zBack = entry.atomBatchedInverse({ shape: [N, D], data: y.data }, { L, b }, N);
  arrEq(zBack.data, zOrig, 1e-10);
});

test('5f-2 inverse: per-atom L with non-trivial off-diagonal (forward-sub indexing)', () => {
  // R1: distinct lower-triangular L per atom; verify L[n]·inverse = (y - b[n]).
  const N = 2, D = 3;
  // atom 0 L lower-tri
  const L = val([N, D, D], [
    2, 0, 0, 1, 3, 0, 0.5, -1, 4,    // atom 0
    1, 0, 0, 2, 2, 0, 1, 1, 5,       // atom 1
  ]);
  const b = val([N, D], [1, 2, 3, -1, 0, 1]);
  const y = val([N, D], [10, 20, 30, 5, 6, 7]);
  const z = entry.atomBatchedInverse(y, { L, b }, N);
  // Reconstruct y' = L[n]·z[n] + b[n], must equal y.
  const Ld = L.data, zd = z.data, bd = b.data, yd = y.data;
  for (let n = 0; n < N; n++) {
    for (let i = 0; i < D; i++) {
      let acc = bd[n * D + i];
      for (let j = 0; j < D; j++) acc += Ld[n * D * D + i * D + j] * zd[n * D + j];
      assert.ok(Math.abs(acc - yd[n * D + i]) < 1e-10,
        `reconstruct atom ${n} dim ${i}: ${acc} vs ${yd[n * D + i]}`);
    }
  }
});

test('5f-2 inverse: atom-indep L [D,D] + b [D] byte-identical to stride-0 reuse (regression)', () => {
  const N = 2, D = 2;
  const L = val([D, D], [2, 0, 0.5, 3]);
  const b = val([D], [1, -1]);
  const y = val([N, D], [5, 5, 6, 6]);
  const z = entry.atomBatchedInverse(y, { L, b }, N);
  // Hand: atom0 (y-b)=[4,6]; z0=4/2=2; z1=(6-0.5*2)/3=5/3.
  //        atom1 (y-b)=[5,7]; z0=5/2=2.5; z1=(7-0.5*2.5)/3=5.75/3.
  arrEq(z.data, [2, 5 / 3, 2.5, 5.75 / 3], 1e-12);
});

test('5f-2 inverse: atom-batched b [N,D] with atom-indep L [D,D]', () => {
  const N = 2, D = 2;
  const L = val([D, D], [2, 0, 0, 2]);
  const b = val([N, D], [1, 1, 10, 10]);   // per-atom mean
  const y = val([N, D], [5, 7, 14, 16]);
  const z = entry.atomBatchedInverse(y, { L, b }, N);
  // atom0 (y-b)=[4,6]/2=[2,3]; atom1 (y-b)=[4,6]/2=[2,3].
  arrEq(z.data, [2, 3, 2, 3], 1e-12);
});

// =====================================================================
// R3 — logDetJ sign + shape
// =====================================================================

test('5f-2 logDetJ: atom-batched L [N,D,D] → Float64Array(N) of -Σ log|diag L[n]|', () => {
  const N = 3, D = 2;
  // diag of atom n: (1,2), (3,4), (5,6)
  const L = val([N, D, D], [1, 0, 0, 2, 3, 0, 0, 4, 5, 0, 0, 6]);
  const j = entry.logDetJ(null, { L }, N);
  assert.ok(j instanceof Float64Array, 'returns Float64Array for atom-batched L');
  assert.equal(j.length, N);
  const expect = [
    -(Math.log(1) + Math.log(2)),
    -(Math.log(3) + Math.log(4)),
    -(Math.log(5) + Math.log(6)),
  ];
  arrEq(j, expect, 1e-12);
});

test('5f-2 logDetJ: atom-indep L [D,D] → scalar (regression)', () => {
  const D = 2;
  const L = val([D, D], [2, 0, 1, 4]);
  const j = entry.logDetJ(null, { L }, 1);
  assert.equal(typeof j, 'number');
  assert.ok(Math.abs(j - (-(Math.log(2) + Math.log(4)))) < 1e-12);
});

// =====================================================================
// Validation
// =====================================================================

test('5f-2 inverse: rejects b shape that is neither [D] nor [N,D]', () => {
  const N = 2, D = 2;
  const L = val([D, D], [1, 0, 0, 1]);
  const b = val([3], [1, 2, 3]);     // wrong D
  const y = val([N, D], [1, 1, 1, 1]);
  assert.throws(() => entry.atomBatchedInverse(y, { L, b }, N),
    /must be \[D\] or \[N, D\]/);
});

test('5f-2 inverse: rejects dense L shape that is neither [D,D] nor [N,D,D]', () => {
  const N = 2, D = 2;
  const L = val([N, D, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);  // [2,2,3] not square
  const b = val([D], [0, 0]);
  const y = val([N, D], [1, 1, 1, 1]);
  assert.throws(() => entry.atomBatchedInverse(y, { L, b }, N),
    /must be \[D, D\] or \[N, D, D\]/);
});
