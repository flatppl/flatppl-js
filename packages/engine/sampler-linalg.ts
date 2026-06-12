// =====================================================================
// sampler-linalg.ts — linear-algebra helpers (textbook algorithms;
// small-matrix sized)
// =====================================================================
//
// Extracted from sampler.ts as part of the sampler split
// (engine-concepts §11). Pure leaf module — operates on plain
// Float64Array / nested-array data; no engine-internal deps.
//
// Two parallel surfaces:
// - nested-array form (`_luDecomp` / `_detLU` / `_logAbsDetLU` /
//   `_linsolveLU` / `_invGaussJordan` / `_cholesky` / `_matmul`) for
//   the original ARITH_OPS nested-array path.
// - Value-native form (`_luDecompValue` / `_detLUValue` /
//   `_logAbsDetLUValue` / `_linsolveLUValue` / `_invValue` /
//   `_choleskyValue`) for the row-major Float64Array path the §2.1
//   shape contract (engine-concepts) requires every numeric boundary
//   to flow through.

// LU decomposition with partial pivoting. Returns
//   { LU: in-place factorized matrix, piv: row-permutation, sign: ±1 }
// where LU stores L (below diagonal, with implicit unit diagonal) and U
// (on and above diagonal) in a single n×n array. piv[i] holds the row
// at row i after permutation; sign tracks the parity of row swaps so
// the caller can recover det(A) = sign · prod(diag(U)).
export function _luDecomp(A: any) {
  const n = A.length;
  // Deep-copy A so the caller's matrix isn't mutated.
  const LU = new Array(n);
  for (let i = 0; i < n; i++) LU[i] = A[i].slice();
  const piv = new Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  let sign = 1;
  for (let k = 0; k < n; k++) {
    // Partial pivot: find row with max |a[r][k]| for r ≥ k.
    let maxAbs = Math.abs(LU[k][k]);
    let maxRow = k;
    for (let r = k + 1; r < n; r++) {
      const v = Math.abs(LU[r][k]);
      if (v > maxAbs) { maxAbs = v; maxRow = r; }
    }
    if (maxAbs === 0) return { LU, piv, sign: 0 };  // singular
    if (maxRow !== k) {
      const tmp = LU[k]; LU[k] = LU[maxRow]; LU[maxRow] = tmp;
      const tp = piv[k]; piv[k] = piv[maxRow]; piv[maxRow] = tp;
      sign = -sign;
    }
    // Eliminate below the diagonal.
    const pivot = LU[k][k];
    for (let r = k + 1; r < n; r++) {
      const factor = LU[r][k] / pivot;
      LU[r][k] = factor;
      for (let c = k + 1; c < n; c++) {
        LU[r][c] -= factor * LU[k][c];
      }
    }
  }
  return { LU, piv, sign };
}

export function _detLU(A: any) {
  const { LU, sign } = _luDecomp(A);
  if (sign === 0) return 0;
  const n = LU.length;
  let det = sign;
  for (let i = 0; i < n; i++) det *= LU[i][i];
  return det;
}

// =====================================================================
// Value-native linear algebra (Float64Array, row-major, no nested-JS
// conversion). Replaces the _valueToNested/_nestedToValue bridges in
// ARITH_OPS.{trace, det, logabsdet, inv, linsolve, lower_cholesky,
// self_outer, diagmat} for Value inputs — the path the §2.1 contract
// (engine-concepts) requires every numeric boundary to flow through.
// =====================================================================

// In-place LU with partial pivoting on a row-major Float64Array of an
// n×n matrix. Returns { LU: Float64Array(n*n), piv: number[], sign: ±1 }.
// LU is a fresh copy of the input data (input is not mutated).
export function _luDecompValue(data: any, n: number) {
  const LU = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) LU[i] = data[i];
  const piv = new Array(n);
  for (let i = 0; i < n; i++) piv[i] = i;
  let sign = 1;
  for (let k = 0; k < n; k++) {
    let maxAbs = Math.abs(LU[k * n + k]);
    let maxRow = k;
    for (let r = k + 1; r < n; r++) {
      const v = Math.abs(LU[r * n + k]);
      if (v > maxAbs) { maxAbs = v; maxRow = r; }
    }
    if (maxAbs === 0) return { LU, piv, sign: 0 };
    if (maxRow !== k) {
      // swap rows k and maxRow in LU
      for (let c = 0; c < n; c++) {
        const t = LU[k * n + c]; LU[k * n + c] = LU[maxRow * n + c]; LU[maxRow * n + c] = t;
      }
      const tp = piv[k]; piv[k] = piv[maxRow]; piv[maxRow] = tp;
      sign = -sign;
    }
    const pivot = LU[k * n + k];
    for (let r = k + 1; r < n; r++) {
      const factor = LU[r * n + k] / pivot;
      LU[r * n + k] = factor;
      for (let c = k + 1; c < n; c++) {
        LU[r * n + c] -= factor * LU[k * n + c];
      }
    }
  }
  return { LU, piv, sign };
}

export function _detLUValue(V: any): number {
  const n = V.shape[0];
  if (n !== V.shape[1]) throw new Error('det: argument must be a square matrix');
  const { LU, sign } = _luDecompValue(V.data, n);
  if (sign === 0) return 0;
  let det = sign;
  for (let i = 0; i < n; i++) det *= LU[i * n + i];
  return det;
}

export function _logAbsDetLUValue(V: any): number {
  const n = V.shape[0];
  if (n !== V.shape[1]) throw new Error('logabsdet: argument must be a square matrix');
  const { LU, sign } = _luDecompValue(V.data, n);
  if (sign === 0) return -Infinity;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.log(Math.abs(LU[i * n + i]));
  return s;
}

// linsolve(A, b) where A is Value{shape:[n,n]} and b is Value vector
// {shape:[n]} or Value matrix {shape:[n, p]}. Returns the same shape
// as b. Forward + back substitution after one LU factorisation.
export function _linsolveLUValue(A: any, b: any): any {
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('linsolve: A must be a square matrix');
  const { LU, piv, sign } = _luDecompValue(A.data, n);
  if (sign === 0) throw new Error('linsolve: matrix is singular');
  function solveOne(bvec: any) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) y[i] = bvec[piv[i]];
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) y[i] -= LU[i * n + j] * y[j];
    }
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let j = i + 1; j < n; j++) s -= LU[i * n + j] * x[j];
      x[i] = s / LU[i * n + i];
    }
    return x;
  }
  if (b.shape.length === 1) {
    if (b.shape[0] !== n) throw new Error('linsolve: dimension mismatch');
    const x = solveOne(b.data);
    return { shape: [n], data: x };
  }
  if (b.shape.length === 2) {
    if (b.shape[0] !== n) throw new Error('linsolve: dimension mismatch');
    const p = b.shape[1];
    // Solve column by column — extract b's column j, solve, write back.
    const out = new Float64Array(n * p);
    const bcol = new Float64Array(n);
    for (let c = 0; c < p; c++) {
      for (let i = 0; i < n; i++) bcol[i] = b.data[i * p + c];
      const xcol = solveOne(bcol);
      for (let i = 0; i < n; i++) out[i * p + c] = xcol[i];
    }
    return { shape: [n, p], data: out };
  }
  throw new Error('linsolve: b must be a vector or matrix, got shape='
    + JSON.stringify(b.shape));
}

// Inverse via linsolve(A, I).
export function _invValue(A: any): any {
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('inv: argument must be a square matrix');
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  return _linsolveLUValue(A, { shape: [n, n], data: I });
}

// Cholesky-Banachiewicz on a row-major Float64Array. Returns Value
// shape=[n, n] with lower-triangular L (zeros above the diagonal).
export function _choleskyValue(A: any): any {
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('lower_cholesky: argument must be a square matrix');
  const data = A.data;
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = data[i * n + j];
      for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (s <= 0) {
          throw new Error('lower_cholesky: matrix is not positive definite '
            + '(non-positive pivot at row ' + i + ')');
        }
        L[i * n + j] = Math.sqrt(s);
      } else {
        L[i * n + j] = s / L[j * n + j];
      }
    }
  }
  return { shape: [n, n], data: L };
}

export function _logAbsDetLU(A: any) {
  const { LU, sign } = _luDecomp(A);
  if (sign === 0) return -Infinity;
  const n = LU.length;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.log(Math.abs(LU[i][i]));
  return s;
}

// Solve A · x = b given the LU factorization. b may be a vector or a
// row-major matrix. Forward substitution (L) followed by backward
// substitution (U).
export function _linsolveLU(A: any, b: any) {
  const { LU, piv, sign } = _luDecomp(A);
  if (sign === 0) throw new Error('linsolve: matrix is singular');
  const n = LU.length;
  const isMat = Array.isArray(b[0]);
  function solveOne(bvec: any) {
    // Apply permutation.
    const y = new Array(n);
    for (let i = 0; i < n; i++) y[i] = bvec[piv[i]];
    // Forward: L y = Pb (L has unit diagonal).
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) y[i] -= LU[i][j] * y[j];
    }
    // Backward: U x = y.
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let j = i + 1; j < n; j++) s -= LU[i][j] * x[j];
      x[i] = s / LU[i][i];
    }
    return x;
  }
  if (!isMat) {
    if (b.length !== n) throw new Error('linsolve: dimension mismatch');
    return solveOne(b);
  }
  // Matrix b: solve column by column.
  const ncols = b[0].length;
  const cols = new Array(ncols);
  for (let c = 0; c < ncols; c++) {
    const bc = new Array(n);
    for (let i = 0; i < n; i++) bc[i] = b[i][c];
    cols[c] = solveOne(bc);
  }
  // Stitch column-major back to row-major.
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(ncols);
    for (let c = 0; c < ncols; c++) row[c] = cols[c][i];
    out[i] = row;
  }
  return out;
}

// Gauss-Jordan inverse. Slightly less numerically stable than
// "linsolve(A, I)" for large matrices but fine at the FlatPPL target
// sizes. We use the LU-based form for symmetry with linsolve.
export function _invGaussJordan(A: any) {
  const n = A.length;
  // Solve A X = I via the LU path.
  const I = new Array(n);
  for (let i = 0; i < n; i++) {
    const row = new Array(n);
    for (let j = 0; j < n; j++) row[j] = (i === j) ? 1 : 0;
    I[i] = row;
  }
  return _linsolveLU(A, I);
}

// Cholesky factorization for symmetric positive-definite A. Returns
// lower-triangular L with A = L · L^T. Diagonal entries are positive.
// Standard recursion (Cholesky-Banachiewicz form).
export function _cholesky(A: any) {
  const n = A.length;
  const L = new Array(n);
  for (let i = 0; i < n; i++) L[i] = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) {
          throw new Error('lower_cholesky: matrix is not positive definite '
            + '(non-positive pivot at row ' + i + ')');
        }
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

// Matrix-matrix multiplication. Handles non-square shapes.
export function _matmul(A: any, B: any) {
  const arows = A.length;
  if (arows === 0) return [];
  const acols = A[0].length;
  const brows = B.length;
  if (acols !== brows) {
    throw new Error('matmul: dimension mismatch (' + arows + '×' + acols
      + ' · ' + brows + '×?)');
  }
  const bcols = B[0].length;
  const out = new Array(arows);
  for (let i = 0; i < arows; i++) {
    const row = new Array(bcols);
    for (let j = 0; j < bcols; j++) {
      let s = 0;
      for (let k = 0; k < acols; k++) s += A[i][k] * B[k][j];
      row[j] = s;
    }
    out[i] = row;
  }
  return out;
}
