'use strict';

// Triangular substitution for the ext-linear-algebra factor records (§09).
// Each right-hand side is solved against the supplied factors; none of these
// paths reconstructs A or performs another decomposition.
const V = require('./value.ts');

/** Resolve matrix views once, preserving the distinction from vectors of vectors. */
function matrix(input: any, name: string): any {
  V.requireMatrix(input, name);
  return V._logicalDense(input);
}

/** Right-hand-side columns share one substitution traversal and preserve rank. */
function rhs(input: any, n: number, name: string): any {
  const b = V._logicalDense(V.asValue(input));
  if ((b.shape.length !== 1 && b.shape.length !== 2) || b.shape[0] !== n || b.outerRank) {
    throw new Error(name + ': b must be a vector or matrix with ' + n + ' rows');
  }
  return b;
}

/** Forward/back substitution reads only the selected triangle. The internal
 * adjoint flag lets Cholesky reuse L without allocating its dense transpose. */
function solveTriangular(input: any, inputB: any, lower = true, unitDiagonal = false, adjoint = false): any {
  if (typeof lower !== 'boolean' || typeof unitDiagonal !== 'boolean') {
    throw new Error('solve_triangular: flags must be booleans');
  }
  const A = matrix(input, 'solve_triangular'), n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('solve_triangular: A must be square');
  const b = rhs(inputB, n, 'solve_triangular'), cols = b.shape.length === 1 ? 1 : b.shape[1];
  const re = new Float64Array(b.data), im = A.im || b.im ? new Float64Array(b.im || re.length) : null;
  const forward = adjoint ? !lower : lower;
  for (let step = 0; step < n; step++) {
    const i = forward ? step : n - 1 - step;
    for (let before = 0; before < step; before++) {
      const j = forward ? before : n - 1 - before;
      const index = adjoint ? j * n + i : i * n + j;
      const ar = A.data[index], ai = (A.im?.[index] ?? 0) * (adjoint ? -1 : 1);
      if (!Number.isFinite(ar) || !Number.isFinite(ai)) throw new Error('solve_triangular: triangle must be finite');
      for (let c = 0; c < cols; c++) {
        const to = i * cols + c, from = j * cols + c, xr = re[from], xi = im?.[from] ?? 0;
        re[to] -= ar * xr - ai * xi;
        if (im) im[to] -= ar * xi + ai * xr;
      }
    }
    if (!unitDiagonal) {
      const dr = A.data[i * n + i], di = (A.im?.[i * n + i] ?? 0) * (adjoint ? -1 : 1);
      const scale = Math.max(Math.abs(dr), Math.abs(di));
      if (scale === 0) throw new Error('solve_triangular: singular diagonal');
      if (!Number.isFinite(scale)) throw new Error('solve_triangular: diagonal must be finite');
      if (di === 0) {
        for (let c = 0; c < cols; c++) {
          re[i * cols + c] /= dr;
          if (im) im[i * cols + c] /= dr;
        }
        continue;
      }
      const pr = dr / scale, pi = di / scale, denominator = pr * pr + pi * pi;
      for (let c = 0; c < cols; c++) {
        const to = i * cols + c, xr = re[to] / scale, xi = (im?.[to] ?? 0) / scale;
        re[to] = (xr * pr + xi * pi) / denominator;
        if (im) im[to] = (xi * pr - xr * pi) / denominator;
      }
    }
  }
  return im ? V.complexValue(re, im, b.shape.slice()) : { shape: b.shape.slice(), data: re };
}

/** P*A=L*U implies U*x=L⁻¹*P*b. P stays an ordinary permutation matrix. */
function luSolve(factors: any, inputB: any): any {
  const P = matrix(factors.P, 'lu_solve'), L = matrix(factors.L, 'lu_solve');
  const U = matrix(factors.U, 'lu_solve'), n = L.shape[0];
  if ([P, L, U].some(f => f.shape[0] !== n || f.shape[1] !== n)) {
    throw new Error('lu_solve: P, L and U must be equally sized square matrices');
  }
  const seen = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let column = -1;
    for (let j = 0; j < n; j++) {
      const p = P.data[i * n + j];
      if ((P.im?.[i * n + j] ?? 0) !== 0 || (p !== 0 && p !== 1) || (p === 1 && column >= 0)) {
        throw new Error('lu_solve: P must be a permutation matrix');
      }
      if (p === 1) column = j;
    }
    if (column < 0 || seen[column]) throw new Error('lu_solve: P must be a permutation matrix');
    seen[column] = 1;
  }
  const b = rhs(inputB, n, 'lu_solve'), cols = b.shape.length === 1 ? 1 : b.shape[1];
  const re = new Float64Array(b.data.length), im = b.im ? new Float64Array(re.length) : null;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const p = P.data[i * n + j];
    if (p === 0) continue;
    for (let c = 0; c < cols; c++) {
      re[i * cols + c] += p * b.data[j * cols + c];
      if (im) im[i * cols + c] += p * b.im[j * cols + c];
    }
  }
  const permuted = im ? V.complexValue(re, im, b.shape.slice()) : { shape: b.shape.slice(), data: re };
  return solveTriangular(U, solveTriangular(L, permuted, true, true), false);
}

/** A=L*Lᴴ: solve twice with the same supplied lower Cholesky triangle. */
function choleskySolve(input: any, b: any): any {
  const L = matrix(input, 'cholesky_solve'), n = L.shape[0];
  if (n !== L.shape[1]) throw new Error('cholesky_solve: L must be square');
  for (let i = 0; i < n; i++) {
    if (!(L.data[i * n + i] > 0) || (L.im?.[i * n + i] ?? 0) !== 0) {
      throw new Error('cholesky_solve: L must have a positive real diagonal');
    }
  }
  return solveTriangular(L, solveTriangular(L, b, true), true, false, true);
}

module.exports = { solveTriangular, luSolve, choleskySolve };
