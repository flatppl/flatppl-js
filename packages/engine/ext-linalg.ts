'use strict';

// =====================================================================
// ext-linalg.ts — extra linear-algebra primitives for the
// `ext-linear-algebra` standard module (spec §09).
// =====================================================================
//
// Pure-algorithm leaf module — operates on row-major Float64Array data
// (the §2.1 shape contract). No engine-internal deps beyond
// sampler-linalg's existing LU primitive.
//
// Each function takes either flat data + shape or a Value, and returns
// a fresh Value (or record-of-Values for multi-output decompositions).
// Standard-modules.ts wires these into spec ops via the
// `_registerExtLinearAlgebra` function.

const sLinalg = require('./sampler-linalg.ts');

// Small helpers — Value constructors live in value.ts but we don't want
// the cross-cycle (value.ts → ops.ts → ... → ext-linalg.ts would be a
// risk). Use the canonical Value shape directly.
function _matValue(data: Float64Array, rows: number, cols: number): any {
  return { shape: [rows, cols], data };
}

function _vecValue(data: Float64Array, n: number): any {
  return { shape: [n], data };
}

// =====================================================================
// lu(A) — LU factorisation with partial pivoting (spec §09)
// =====================================================================
//
// Returns record(P, L, U) such that P · A = L · U, where:
//   * P is the n×n permutation matrix from partial pivoting.
//   * L is lower-triangular with unit diagonal.
//   * U is upper-triangular.
//
// Wraps the existing `_luDecompValue` in sampler-linalg.ts (used by
// det / logabsdet / linsolve). The factorisation is in-place in a
// fresh buffer; this function just unpacks it into the spec's
// canonical (P, L, U) record shape.

function _lu(A: any): any {
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('lu: A must be a square matrix');
  const { LU, piv, sign } = sLinalg._luDecompValue(A.data, n);
  if (sign === 0) throw new Error('lu: matrix is singular');
  // Unpack: L (unit-diagonal, below), U (on and above diagonal), P (from piv).
  const L = new Float64Array(n * n);
  const U = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    L[i * n + i] = 1;
    for (let j = 0; j < i; j++) L[i * n + j] = LU[i * n + j];
    for (let j = i; j < n; j++) U[i * n + j] = LU[i * n + j];
  }
  // piv[i] = original row index of the row now at row i. Build P such
  // that (P · A)[i, :] = A[piv[i], :].
  const P = new Float64Array(n * n);
  for (let i = 0; i < n; i++) P[i * n + piv[i]] = 1;
  return {
    shape: 'record',
    fields: {
      P: _matValue(P, n, n),
      L: _matValue(L, n, n),
      U: _matValue(U, n, n),
    },
  };
}

// =====================================================================
// kron(A, B) — Kronecker tensor product (spec §09)
// =====================================================================
//
// For A of shape (m, n) and B of shape (p, q), returns a matrix of
// shape (m*p, n*q) where block (i, j) is A[i, j] · B.
//
// Direct loop — for the size class FlatPPL targets (m, n, p, q in the
// 1..100 range), the constant-factor cost is dominated by the inner
// multiplication. A BLAS-style block algorithm would help for larger
// inputs; defer until profile demands.

function _kron(A: any, B: any): any {
  const m = A.shape[0], n = A.shape[1];
  const p = B.shape[0], q = B.shape[1];
  const outRows = m * p, outCols = n * q;
  const out = new Float64Array(outRows * outCols);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      const aij = A.data[i * n + j];
      if (aij === 0) continue;
      for (let r = 0; r < p; r++) {
        for (let c = 0; c < q; c++) {
          out[(i * p + r) * outCols + (j * q + c)] = aij * B.data[r * q + c];
        }
      }
    }
  }
  return _matValue(out, outRows, outCols);
}

// =====================================================================
// matexp(A) — matrix exponential via scaling + squaring with Padé(13)
// =====================================================================
//
// e^A is computed via the canonical Higham (2005) algorithm:
//   1. Choose s such that ||A / 2^s||_1 ≤ θ_13 ≈ 5.371920351148152.
//   2. Compute the [13/13] Padé approximant to e^(A/2^s).
//   3. Square the result s times.
//
// Numerically stable for matrices of modest size; sufficient for the
// Bayesian / particle-physics use cases the engine targets.

const _PADE13_B = [
  64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800,
  129060195264000, 10559470521600, 670442572800, 33522128640,
  1323241920, 40840800, 960960, 16380, 182, 1,
];

function _matmul(A: Float64Array, B: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < n; k++) {
      const aik = A[i * n + k];
      if (aik === 0) continue;
      for (let j = 0; j < n; j++) {
        out[i * n + j] += aik * B[k * n + j];
      }
    }
  }
  return out;
}

function _matAdd(A: Float64Array, B: Float64Array, n: number, s: number): Float64Array {
  // out = A + s·B
  const out = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) out[i] = A[i] + s * B[i];
  return out;
}

function _matScale(A: Float64Array, s: number): Float64Array {
  const out = new Float64Array(A.length);
  for (let i = 0; i < A.length; i++) out[i] = A[i] * s;
  return out;
}

function _identity(n: number): Float64Array {
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  return I;
}

function _norm1(A: Float64Array, n: number): number {
  // Max column sum of |A|.
  let max = 0;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.abs(A[i * n + j]);
    if (s > max) max = s;
  }
  return max;
}

function _matexp(A: any): any {
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('matexp: A must be a square matrix');
  // Scaling: pick s = max(0, ceil(log2(||A||_1 / θ_13))).
  const THETA_13 = 5.371920351148152;
  const norm = _norm1(A.data, n);
  let s = 0;
  if (norm > THETA_13) {
    s = Math.max(0, Math.ceil(Math.log2(norm / THETA_13)));
  }
  const scale = Math.pow(2, s);
  let M = _matScale(A.data, 1 / scale);
  // Padé(13): compute U, V using powers M^2, M^4, M^6.
  const M2 = _matmul(M, M, n);
  const M4 = _matmul(M2, M2, n);
  const M6 = _matmul(M4, M2, n);
  const I  = _identity(n);
  const B = _PADE13_B;
  // U = M · (M6 · (b13·M6 + b11·M4 + b9·M2) + b7·M6 + b5·M4 + b3·M2 + b1·I)
  // V =      M6 · (b12·M6 + b10·M4 + b8·M2) + b6·M6 + b4·M4 + b2·M2 + b0·I
  let tmp = _matAdd(_matScale(M6, B[13]), _matScale(M4, B[11]), n, 1);
  tmp = _matAdd(tmp, _matScale(M2, B[9]), n, 1);
  let Uinner = _matmul(M6, tmp, n);
  Uinner = _matAdd(Uinner, _matScale(M6, B[7]), n, 1);
  Uinner = _matAdd(Uinner, _matScale(M4, B[5]), n, 1);
  Uinner = _matAdd(Uinner, _matScale(M2, B[3]), n, 1);
  Uinner = _matAdd(Uinner, _matScale(I,  B[1]), n, 1);
  const U = _matmul(M, Uinner, n);

  tmp = _matAdd(_matScale(M6, B[12]), _matScale(M4, B[10]), n, 1);
  tmp = _matAdd(tmp, _matScale(M2, B[8]), n, 1);
  let V = _matmul(M6, tmp, n);
  V = _matAdd(V, _matScale(M6, B[6]), n, 1);
  V = _matAdd(V, _matScale(M4, B[4]), n, 1);
  V = _matAdd(V, _matScale(M2, B[2]), n, 1);
  V = _matAdd(V, _matScale(I,  B[0]), n, 1);

  // R = (V - U)^-1 · (V + U)
  const VminusU = _matAdd(V, U, n, -1);
  const VplusU  = _matAdd(V, U, n,  1);
  // Solve (V - U) · R = (V + U) column-wise via _linsolveLU.
  const { LU, piv, sign } = sLinalg._luDecompValue(VminusU, n);
  if (sign === 0) throw new Error('matexp: scaling-and-squaring solve was singular');
  let R = new Float64Array(n * n);
  const col = new Float64Array(n);
  for (let c = 0; c < n; c++) {
    for (let i = 0; i < n; i++) col[i] = VplusU[i * n + c];
    // Forward: solve L · y = P · col.
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) y[i] = col[piv[i]];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) y[i] -= LU[i * n + j] * y[j];
    }
    // Back: solve U · x = y.
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let acc = y[i];
      for (let j = i + 1; j < n; j++) acc -= LU[i * n + j] * x[j];
      x[i] = acc / LU[i * n + i];
    }
    for (let i = 0; i < n; i++) R[i * n + c] = x[i];
  }
  // Square s times.
  for (let k = 0; k < s; k++) R = _matmul(R, R, n);
  return _matValue(R, n, n);
}

module.exports = {
  _lu,
  _kron,
  _matexp,
  // Test-only / internal helpers
  _internal: {
    _matmul,
    _matAdd,
    _matScale,
    _identity,
    _norm1,
  },
};
