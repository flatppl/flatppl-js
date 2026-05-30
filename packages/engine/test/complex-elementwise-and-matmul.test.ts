'use strict';

// =====================================================================
// complex-elementwise-and-matmul.test.ts — close two complex gaps
// =====================================================================
//
// Pins two complex-coverage additions:
//   (1) `_makeElementwiseUnop` now accepts a complex scalar fn —
//       complex unary maths (exp / log / sqrt / abs / abs2) no
//       longer reject complex input; they iterate planar re/im
//       directly. _cAbs / _cAbs2 produce REAL output (complex →
//       real); _cExp / _cLog / _cSqrt produce complex output.
//   (2) `_cxMatBatchedMatMul` — atom-batched complex rank-2 ×
//       rank-2 matmul (planar re/im). Three input patterns.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const valueOps = require('../value-ops.ts');

// =====================================================================
// 1. expElem / logElem on complex
// =====================================================================

test('expElem(complex): elementwise e^z over planar re/im', () => {
  const v = {
    shape: [3],
    data: new Float64Array([0, 0, 1]),
    im:   new Float64Array([0, Math.PI / 2, 0]),
    dtype: 'complex',
  };
  const r = valueOps.expElem(v);
  assert.equal(r.dtype, 'complex');
  assert.ok(Math.abs(r.data[0] - 1) < 1e-12, 'e^0 real = 1');
  assert.ok(Math.abs(r.im[0]) < 1e-12, 'e^0 imag = 0');
  assert.ok(Math.abs(r.data[1]) < 1e-12, 'e^(iπ/2) real = 0');
  assert.ok(Math.abs(r.im[1] - 1) < 1e-12, 'e^(iπ/2) imag = 1');
  assert.ok(Math.abs(r.data[2] - Math.E) < 1e-12, 'e^1 = e');
});

test('logElem(complex): log(z) elementwise principal branch', () => {
  const v = {
    shape: [2],
    data: new Float64Array([1, -1]),
    im:   new Float64Array([0, 0]),
    dtype: 'complex',
  };
  const r = valueOps.logElem(v);
  assert.equal(r.dtype, 'complex');
  assert.ok(Math.abs(r.data[0]) < 1e-12, 'log(1) re = 0');
  assert.ok(Math.abs(r.im[0]) < 1e-12, 'log(1) im = 0');
  assert.ok(Math.abs(r.data[1]) < 1e-12, 'log(-1) re = 0');
  assert.ok(Math.abs(r.im[1] - Math.PI) < 1e-12, 'log(-1) im = π');
});

test('absElem(complex): |z| → REAL Value', () => {
  const v = {
    shape: [2],
    data: new Float64Array([3, 5]),
    im:   new Float64Array([4, 12]),
    dtype: 'complex',
  };
  const r = valueOps.absElem(v);
  assert.ok(!r.dtype || r.dtype !== 'complex',
    '|complex z| yields a real Value (no dtype/im)');
  assert.ok(!r.im, 'no im buffer on real output');
  assert.equal(r.data[0], 5, '|3+4i| = 5');
  assert.equal(r.data[1], 13, '|5+12i| = 13');
});

test('abs2Elem(complex): |z|² → REAL Value', () => {
  const v = {
    shape: [2],
    data: new Float64Array([3, 5]),
    im:   new Float64Array([4, 12]),
    dtype: 'complex',
  };
  const r = valueOps.abs2Elem(v);
  assert.equal(r.data[0], 25);
  assert.equal(r.data[1], 169);
});

test('expElem(real): regression — real input still works', () => {
  const v = { shape: [3], data: new Float64Array([0, 1, 2]) };
  const r = valueOps.expElem(v);
  assert.ok(!r.dtype || r.dtype !== 'complex');
  assert.ok(Math.abs(r.data[0] - 1) < 1e-12);
  assert.ok(Math.abs(r.data[1] - Math.E) < 1e-12);
});

test('tanElem(complex): full complex coverage', () => {
  // P9 follow-up: tan/sin/cos plus inverse trig + hyperbolics are
  // now complex-extended via sampler-complex closed forms. Pin a
  // representative value to catch regressions.
  // tan(i) ≈ 0 + 0.76159i (= i·tanh(1))
  const v = {
    shape: [1],
    data: new Float64Array([0]),
    im:   new Float64Array([1]),
    dtype: 'complex',
  };
  const r = valueOps.tanElem(v);
  assert.equal(r.dtype, 'complex');
  assert.ok(Math.abs(r.data[0]) < 1e-12, 'tan(i) real part ≈ 0');
  assert.ok(Math.abs(r.im[0] - Math.tanh(1)) < 1e-12,
    'tan(i) imag part ≈ tanh(1)');
});

// =====================================================================
// 2. _cxMatBatchedMatMul — three input patterns
// =====================================================================

function cxMat(rows: Array<Array<[number, number]>>) {
  const r = rows.length, c = rows[0].length;
  const re = new Float64Array(r * c), im = new Float64Array(r * c);
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      re[i * c + j] = rows[i][j][0];
      im[i * c + j] = rows[i][j][1];
    }
  }
  return { shape: [r, c], data: re, im, dtype: 'complex' };
}

function cxBatch(atoms: Array<Array<Array<[number, number]>>>) {
  const N = atoms.length, m = atoms[0].length, n = atoms[0][0].length;
  const re = new Float64Array(N * m * n), im = new Float64Array(N * m * n);
  let k = 0;
  for (let a = 0; a < N; a++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        re[k] = atoms[a][i][j][0];
        im[k] = atoms[a][i][j][1];
        k++;
      }
    }
  }
  return { shape: [N, m, n], data: re, im, dtype: 'complex' };
}

test('_cxMatBatchedMatMul: shared real A × batched complex B', () => {
  const N = 2;
  // A = real identity 2x2 (re-only).
  const A = cxMat([[[1, 0], [0, 0]], [[0, 0], [1, 0]]]);
  // B per atom: atom 0 = [[2+0i, 0],[0, 3+0i]]; atom 1 = [[0+i, 0],[0, 0+2i]].
  const B = cxBatch([
    [[[2, 0], [0, 0]], [[0, 0], [3, 0]]],
    [[[0, 1], [0, 0]], [[0, 0], [0, 2]]],
  ]);
  const out = valueOps._cxMatBatchedMatMul(A, B, N);
  assert.deepEqual(out.shape, [N, 2, 2]);
  // I × B = B
  assert.equal(out.data[0], 2); assert.equal(out.im[0], 0);
  assert.equal(out.data[3], 3); assert.equal(out.im[3], 0);
  assert.equal(out.data[4], 0); assert.equal(out.im[4], 1);
  assert.equal(out.data[7], 0); assert.equal(out.im[7], 2);
});

test('complex singleton-axis broadcast: [3, 1] + [1, 2] → [3, 2]', () => {
  const a = {
    shape: [3, 1],
    data: new Float64Array([1, 2, 3]),     // re
    im:   new Float64Array([10, 20, 30]),  // im → 1+10i, 2+20i, 3+30i
    dtype: 'complex',
  };
  const b = {
    shape: [1, 2],
    data: new Float64Array([100, 200]),     // re → 100+0i, 200+0i
    im:   new Float64Array([0, 0]),
    dtype: 'complex',
  };
  const r = valueOps.add(a, b);
  assert.deepEqual(r.shape, [3, 2]);
  assert.equal(r.dtype, 'complex');
  // Row 0 (a=1+10i): [101+10i, 201+10i]
  assert.equal(r.data[0], 101); assert.equal(r.im[0], 10);
  assert.equal(r.data[1], 201); assert.equal(r.im[1], 10);
  // Row 1 (a=2+20i): [102+20i, 202+20i]
  assert.equal(r.data[2], 102); assert.equal(r.im[2], 20);
  // Row 2 (a=3+30i): [103+30i, 203+30i]
  assert.equal(r.data[5], 203); assert.equal(r.im[5], 30);
});

test('_cxMatBatchedMatMul: both per-atom — i·I × B[atom]', () => {
  const N = 2;
  // A: atom 0 = real I, atom 1 = i·I
  const A = cxBatch([
    [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
    [[[0, 1], [0, 0]], [[0, 0], [0, 1]]],
  ]);
  // B: same per atom = [[1, 0], [0, 1]] (real I)
  const B = cxBatch([
    [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
    [[[1, 0], [0, 0]], [[0, 0], [1, 0]]],
  ]);
  const out = valueOps._cxMatBatchedMatMul(A, B, N);
  // atom 0: I × I = I; atom 1: i·I × I = i·I.
  assert.equal(out.data[0], 1); assert.equal(out.im[0], 0);
  assert.equal(out.data[4], 0); assert.equal(out.im[4], 1);
  assert.equal(out.data[7], 0); assert.equal(out.im[7], 1);
});
