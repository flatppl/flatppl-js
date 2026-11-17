'use strict';

// =====================================================================
// ext-linalg.ts — extra linear-algebra primitives for the
// `ext-linear-algebra` standard module (spec §09).
// =====================================================================
//
// Numeric algorithms over row-major real and planar-complex Values (the
// §2.1 shape contract). value.ts resolves storage views at each boundary;
// the complex LU and general spectral paths have their own numeric leaves.
//
// Each function takes either flat data + shape or a Value, and returns
// a fresh Value (or record-of-Values for multi-output decompositions).
// Standard-modules.ts wires these into spec ops via the
// `_registerExtLinearAlgebra` function.

const sLinalg = require('./sampler-linalg.ts');
const valueLib = require('./value.ts');

// Small helpers for fresh, canonical dense output Values.
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
  A = valueLib._logicalDense(A);
  if (A.im) return require('./complex-lu.ts').lu(A);
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('lu: A must be a square matrix');
  const { LU, piv } = sLinalg._luDecompValue(A.data, n);
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
    P: _matValue(P, n, n),
    L: _matValue(L, n, n),
    U: _matValue(U, n, n),
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
  A = valueLib._logicalDense(A);
  B = valueLib._logicalDense(B);
  if (A.im || B.im) return _complexKron(A, B);
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

/** §09 complex Kronecker blocks use the same matrix layout as the real path.
 * Zero real parts do not imply zero blocks when the coefficient is imaginary. */
function _complexKron(A: any, B: any): any {
  const [m, n] = A.shape, [p, q] = B.shape, cols = n * q;
  const re = new Float64Array(m * p * cols), im = new Float64Array(re.length);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) {
    const ar = A.data[i * n + j], ai = A.im?.[i * n + j] || 0;
    if (ar === 0 && ai === 0) continue;
    for (let r = 0; r < p; r++) for (let c = 0; c < q; c++) {
      const br = B.data[r * q + c], bi = B.im?.[r * q + c] || 0;
      const index = (i * p + r) * cols + j * q + c;
      re[index] = ar * br - ai * bi; im[index] = ar * bi + ai * br;
    }
  }
  return valueLib.complexValue(re, im, [m * p, cols]);
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

function _norm1(A: Float64Array, n: number, divisor = 1): number {
  // Divide each entry before summation to keep the scaling norm finite.
  let max = 0;
  for (let j = 0; j < n; j++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += Math.abs(A[i * n + j] / divisor);
    if (s > max) max = s;
  }
  return max;
}

function _matexp(A: any): any {
  A = valueLib._logicalDense(A);
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('matexp: A must be a square matrix');
  if (A.im) {
    // Realification preserves multiplication and hence the exponential:
    // R(A)=[Re A,-Im A; Im A,Re A], exp(R(A))=R(exp(A)). Reuse the
    // real Padé solve and scaling decisions at this numerical boundary.
    const size = 2 * n, block = new Float64Array(size * size);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const k = i * n + j, top = i * size + j, bottom = (i + n) * size + j;
      block[top] = block[bottom + n] = A.data[k];
      block[top + n] = -A.im[k]; block[bottom] = A.im[k];
    }
    const exponential = _matexp(_matValue(block, size, size));
    const re = new Float64Array(n * n), im = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      re[i * n + j] = exponential.data[i * size + j];
      im[i * n + j] = exponential.data[(i + n) * size + j];
    }
    return valueLib.complexValue(re, im, [n, n]);
  }
  // Scaling: pick s = max(0, ceil(log2(||A||_1 / θ_13))).
  const THETA_13 = 5.371920351148152;
  let maximum = 0;
  for (const entry of A.data) {
    if (!Number.isFinite(entry)) {
      // In particular, infinity must never become an infinite squaring count.
      return _matValue(new Float64Array(n * n).fill(NaN), n, n);
    }
    maximum = Math.max(maximum, Math.abs(entry));
  }
  const safeMaximum = maximum || 1;
  const logNorm = Math.log2(safeMaximum) + Math.log2(Math.max(1, _norm1(A.data, n, safeMaximum)));
  const s = Math.max(0, Math.ceil(logNorm - Math.log2(THETA_13)));
  // Splitting the reciprocal avoids overflow or subnormal factors at large norms.
  let M = _matScale(A.data, Math.pow(2, -Math.floor(s / 2)));
  M = _matScale(M, Math.pow(2, -(s - Math.floor(s / 2))));
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
  let R: Float64Array = new Float64Array(n * n);
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

// =====================================================================
// qr(A) — QR factorisation via Householder reflections (spec §07 +
// §09; surfaces in the ext-linear-algebra standard module).
// =====================================================================
//
// For A of shape (m, n) with m ≥ n, returns record(Q, R) where Q is
// m×n with orthonormal columns and R is n×n upper-triangular such that
// A = Q · R. The algorithm:
//
//   1. For each column k in 0..n:
//        Compute the Householder reflector v_k that zeros A[k+1:m, k].
//        Apply H_k = I - beta_k v_k v_k^T to A[k:, k:] in-place.
//   2. R is the upper-triangular part of the modified A.
//   3. Apply the saved H_k in reverse to the first n identity columns,
//      forming Q without a full m×m workspace.
//
// Stable for m × n matrices in the size range FlatPPL targets.
// The base `qr` binding uses this same implementation (spec §07).

function _qr(A: any): any {
  A = valueLib._logicalDense(A);
  const m = A.shape[0], n = A.shape[1];
  if (m < n) throw new Error('qr: requires m >= n (got ' + m + 'x' + n + ')');
  // Work on a fresh copy of A; we'll modify it into R in-place.
  const R_full = new Float64Array(m * n);
  for (let i = 0; i < m * n; i++) R_full[i] = A.data[i];
  const reflectors = new Float64Array(m * n);
  const betas = new Float64Array(n);

  const v = new Float64Array(m);                 // reflector workspace
  for (let k = 0; k < n; k++) {
    // Extract column k from row k down: x = R_full[k:m, k].
    // Nothing below the diagonal: no reflector, as LAPACK dlarfg (and so
    // numpy) does, which keeps the sign of R[k, k].
    let belowScale = 0;
    for (let i = k + 1; i < m; i++) {
      belowScale = Math.max(belowScale, Math.abs(R_full[i * n + k]));
    }
    if (belowScale === 0) continue;
    const x0 = R_full[k * n + k];
    // Scale before squaring so finite columns do not overflow or underflow.
    const scale = Math.max(Math.abs(x0), belowScale);
    const pivot = x0 / scale;
    let norm2 = pivot * pivot;
    for (let i = k + 1; i < m; i++) {
      v[i] = R_full[i * n + k] / scale;
      norm2 += v[i] * v[i];
    }
    const normX = Math.sqrt(norm2);
    // alpha = -sign(x[0]) * normX (Householder convention; flip sign
    // to avoid catastrophic cancellation in v[0]).
    const alpha = (x0 >= 0 ? -1 : 1) * normX;
    // Keep v = x - alpha · e_1 at the scaled column's magnitude and use
    // beta = 2 / (v^T v), without forming squares of the original entries.
    v[k] = pivot - alpha;
    let vNorm2 = v[k] * v[k];
    for (let i = k + 1; i < m; i++) {
      vNorm2 += v[i] * v[i];
    }
    if (vNorm2 === 0) continue;
    // Apply H = I - (2 / vNorm2) v v^T to R_full[k:m, k:n] in-place.
    const beta = 2 / vNorm2;
    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += v[i] * R_full[i * n + j];
      const s = beta * dot;
      for (let i = k; i < m; i++) R_full[i * n + j] -= s * v[i];
    }
    betas[k] = beta;
    for (let i = k; i < m; i++) reflectors[i * n + k] = v[i];
  }
  // Apply reflectors in reverse to the first n columns of the identity.
  const Q = new Float64Array(m * n);
  for (let i = 0; i < n; i++) Q[i * n + i] = 1;
  for (let k = n - 1; k >= 0; k--) {
    const beta = betas[k];
    if (beta === 0) continue;
    for (let j = k; j < n; j++) {
      let dot = 0;
      for (let i = k; i < m; i++) dot += reflectors[i * n + k] * Q[i * n + j];
      const s = beta * dot;
      for (let i = k; i < m; i++) Q[i * n + j] -= s * reflectors[i * n + k];
    }
  }
  // R is the upper triangle: what the reflectors leave below it is rounding.
  const R = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) R[i * n + j] = R_full[i * n + j];
  }
  return {
    Q: _matValue(Q, m, n),
    R: _matValue(R, n, n),
  };
}

// =====================================================================
// lstsq(A, b) — least-squares solve for min_x ||A·x - b||₂ (spec §09)
// =====================================================================
//
// The minimum-norm least-squares solution for A of shape (m, k) and b of
// shape (m,) or (m,p): x = V · S⁺ · Uᴴ · b from the thin SVD.
// S⁺ inverts singular values strictly above max(atol,rtol*σ_max), with
// automatic rtol=max(m,k)*epsilon by default. Rank-deficient A
// gives the minimum-norm solution rather than dividing by a rounding-level
// pivot. The SVD boundary handles wide A by decomposing its adjoint and
// swapping U/V. Complex projection uses Uᴴ rather than a plain transpose.

/** Validate the shared absolute/relative cutoff used by rank, pinv and lstsq. */
function _cutoffs(A: any, rtol = -1, atol = 0): [number, number] {
  if (rtol === -1) rtol = Math.max(...A.shape) * Number.EPSILON;
  if (!Number.isFinite(rtol) || rtol < 0 || !Number.isFinite(atol) || atol < 0) {
    throw new Error('rtol and atol must be finite and nonnegative (rtol=-1 selects automatic tolerance)');
  }
  return [rtol, atol];
}

/** Multiplication is deliberate: rtol=0 must retain tiny positive singular
 * values even when their ratio to sigma_max would underflow. */
function _singularCutoff(S: Float64Array, relative: number, absolute: number): number {
  return Math.max(absolute, relative * (S[0] || 0));
}

function _lstsq(A: any, b: any, rtol = -1, atol = 0): any {
  valueLib.requireMatrix(A, 'lstsq');
  A = valueLib._logicalDense(A);
  b = valueLib._logicalDense(valueLib.asValue(b));
  const m = A.shape[0], k = A.shape[1];
  if ((b.shape.length !== 1 && b.shape.length !== 2) || b.shape[0] !== m || b.outerRank) {
    throw new Error('lstsq: b must be a vector or matrix with ' + m + ' rows');
  }
  const [relative, absolute] = _cutoffs(A, rtol, atol), { U: left, S, V: right } = _svd(A);
  const r = S.data.length, cols = b.shape.length === 1 ? 1 : b.shape[1];
  const cutoff = _singularCutoff(S.data, relative, absolute);
  const x = new Float64Array(k * cols), im = A.im || b.im ? new Float64Array(x.length) : null;
  for (let j = 0; j < r; j++) {
    const s = S.data[j];
    if (!(s > cutoff)) continue;
    for (let c = 0; c < cols; c++) {
      let cr = 0, ci = 0;
      for (let i = 0; i < m; i++) {
        const ur = left.data[i * r + j], ui = left.im?.[i * r + j] ?? 0;
        const br = b.data[i * cols + c], bi = b.im?.[i * cols + c] ?? 0;
        cr += ur * br + ui * bi; ci += ur * bi - ui * br;
      }
      cr /= s; ci /= s;
      for (let i = 0; i < k; i++) {
        const vr = right.data[i * r + j], vi = right.im?.[i * r + j] ?? 0;
        x[i * cols + c] += vr * cr - vi * ci;
        if (im) im[i * cols + c] += vr * ci + vi * cr;
      }
    }
  }
  const shape = b.shape.length === 1 ? [k] : [k, cols];
  return im ? valueLib.complexValue(x, im, shape) : { shape, data: x };
}

/** Moore-Penrose inverse from the existing economy SVD: V S⁺ Uᴴ. */
function _pinv(A: any, rtol = -1, atol = 0): any {
  valueLib.requireMatrix(A, 'pinv');
  A = valueLib._logicalDense(A);
  const [m, n] = A.shape, [relative, absolute] = _cutoffs(A, rtol, atol), { U, S, V } = _svd(A);
  const r = S.data.length, re = new Float64Array(n * m), im = A.im ? new Float64Array(n * m) : null;
  const cutoff = _singularCutoff(S.data, relative, absolute);
  for (let j = 0; j < r; j++) {
    const s = S.data[j];
    if (!(s > cutoff)) continue;
    for (let i = 0; i < n; i++) {
      const vr = V.data[i * r + j] / s, vi = (V.im?.[i * r + j] ?? 0) / s;
      for (let k = 0; k < m; k++) {
        const ur = U.data[k * r + j], ui = U.im?.[k * r + j] ?? 0;
        re[i * m + k] += vr * ur + vi * ui;
        if (im) im[i * m + k] += vi * ur - vr * ui;
      }
    }
  }
  return im ? valueLib.complexValue(re, im, [n, m]) : _matValue(re, n, m);
}

/** Spectral condition number uses the full singular spectrum, without truncation. */
function _cond(A: any): number {
  valueLib.requireMatrix(A, 'cond');
  const S = _svd(A).S.data;
  if (!S.length) throw new Error('cond: singular spectrum must be nonempty');
  return S[S.length - 1] > 0 ? S[0] / S[S.length - 1] : Infinity;
}

// =====================================================================
// Symmetric eigendecomposition via Jacobi rotations (spec §09)
// =====================================================================
//
// `_symmetricEigen(A)` returns record(values, vectors) for a real
// symmetric A: A · vectors = vectors · diag(values), with vectors'
// columns orthonormal. Algorithm: cyclic Jacobi sweeps.
//
//   Repeatedly pick the off-diagonal pivot (p, q) with the largest
//   |A[p, q]|; build the rotation R that zeros A[p, q] (and A[q, p]
//   by symmetry); apply A ← R^T · A · R + accumulate R into the
//   eigenvector matrix V (initialised to I). Each sweep visits all
//   (p, q) with p < q. Converges quadratically once near a fixed
//   point; for the size class FlatPPL targets (n in 2..50), 10-20
//   sweeps are sufficient.
//
// Pros: trivial to implement, numerically stable for symmetric input,
// produces highly accurate eigenpairs. Cons: O(n³) per sweep, slower
// than tridiagonal-QR for large n. Defer Householder-tridiag+QR until
// a profile shows the n > 50 case.
//
// The symmetric-real path remains the fast covariance-matrix case. General
// real and complex matrices use unitary Schur iteration in general-eigen.ts.

const _JACOBI_MAX_SWEEPS = 50;
const _JACOBI_TOL_REL    = 1e-12;

function _isSymmetric(A: any, n: number, tol: number): boolean {
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (Math.abs(A.data[i * n + j] - A.data[j * n + i]) > tol) return false;
    }
  }
  return true;
}

function _jacobiSymmetric(A: any, computeVectors = true, relativeTolerance = _JACOBI_TOL_REL): {
  values: Float64Array; vectors: Float64Array | null; n: number;
} {
  A = valueLib._logicalDense(A);
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('eigen: A must be a square matrix');
  // Scale norm and rotation-angle arithmetic to avoid overflow. Keep
  // the work entries in their original units: normalizing the entire
  // matrix could erase a tiny diagonal beside a much larger block.
  // Nonfinite inputs retain their existing propagation; zero needs no scaling.
  let scale = 0;
  for (let i = 0; i < n * n; i++) scale = Math.max(scale, Math.abs(A.data[i]));
  if (scale === 0 || !Number.isFinite(scale)) scale = 1;
  // The convergence tolerance uses the normalized Frobenius norm;
  // symmetry retains its relative tolerance and absolute floor in A's units.
  let frob2 = 0;
  for (let i = 0; i < n * n; i++) {
    const v = A.data[i] / scale;
    frob2 += v * v;
  }
  const tol = relativeTolerance * Math.sqrt(frob2);
  if (!_isSymmetric(A, n, Math.max(tol * scale, 1e-10))) {
    throw new Error('eigen: only symmetric matrices supported in this version '
      + '(non-symmetric requires complex-output path; deferred)');
  }
  // Work on a fresh copy; symmetrise to ensure exact symmetry under
  // the rotations.
  const M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const a = A.data[i * n + j], b = A.data[j * n + i];
      const sum = a + b;
      // Halve after adding when possible to preserve subnormal entries.
      M[i * n + j] = Number.isFinite(sum) ? 0.5 * sum : 0.5 * a + 0.5 * b;
    }
  }
  // V starts as identity; will hold eigenvectors as columns.
  const V = computeVectors ? new Float64Array(n * n) : null;
  if (V) for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let sweep = 0; sweep < _JACOBI_MAX_SWEEPS; sweep++) {
    // Off-diagonal sum-of-squares.
    let off = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const v = M[i * n + j] / scale;
        off += v * v;
      }
    }
    if (off < tol * tol || (off === 0 && tol === 0)) break;

    // Cyclic sweep: visit every (p, q) pair with p < q.
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = M[p * n + q];
        if (Math.abs(apq / scale) < tol) continue;
        const app = M[p * n + p];
        const aqq = M[q * n + q];
        // Compute c, s for the rotation that zeros apq using the
        // Numerical Recipes convention (consistent with the rotation
        // matrix R[p][q]=+s, R[q][p]=-s used in the off-diagonal
        // updates below):
        //   θ = (aqq - app) / (2·apq) = cot(2φ)
        //   t = sign(θ) / (|θ| + sqrt(1 + θ²))   — root of t² + 2θt - 1 = 0
        //   c = 1/sqrt(1 + t²),  s = t·c
        // Earlier draft had θ = (app - aqq) / (2·apq) — opposite sign
        // from NR — combined with the NR rotation convention this
        // corrupted off-diagonal updates on n ≥ 3 matrices (caught by
        // the 3×3 round-trip test before commit).
        const theta = (aqq / scale - app / scale) / (2 * (apq / scale));
        let t: number;
        if (Math.abs(theta) > 1e150) {
          // Avoid overflow when apq is tiny relative to (aqq - app).
          t = 0.5 / theta;
        } else {
          const sign = theta >= 0 ? 1 : -1;
          t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        }
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        // Update diagonal + zero off-diagonal.
        M[p * n + p] = app - t * apq;
        M[q * n + q] = aqq + t * apq;
        M[p * n + q] = 0;
        M[q * n + p] = 0;
        // Rotate the other entries of rows/columns p and q.
        for (let i = 0; i < n; i++) {
          if (i !== p && i !== q) {
            const aip = M[i * n + p], aiq = M[i * n + q];
            M[i * n + p] = c * aip - s * aiq;
            M[i * n + q] = s * aip + c * aiq;
            M[p * n + i] = M[i * n + p];
            M[q * n + i] = M[i * n + q];
          }
        }
        // Accumulate the rotation into V.
        if (V) for (let i = 0; i < n; i++) {
          const vip = V[i * n + p], viq = V[i * n + q];
          V[i * n + p] = c * vip - s * viq;
          V[i * n + q] = s * vip + c * viq;
        }
      }
    }
  }
  // Extract eigenvalues (diagonal of M).
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = M[i * n + i];
  return { values, vectors: V, n };
}

function _eigen(A: any): any {
  A = valueLib._logicalDense(A);
  if (A.im || A.shape[0] !== A.shape[1] || !_isSymmetric(A, A.shape[0], 0)) {
    return require('./general-eigen.ts').eigen(A);
  }
  const { values, vectors, n } = _jacobiSymmetric(A);
  return {
    values:  valueLib.complexValue(values, new Float64Array(n), [n]),
    vectors: valueLib.complexValue(vectors!, new Float64Array(n * n), [n, n]),
  };
}

function _generalEigenExtreme(A: any, maximum: boolean): number {
  const { values, vectors } = require('./general-eigen.ts').eigen(A);
  if (values.data.length === 0) throw new Error('eigenvalue extrema require a nonempty spectrum');
  let scale = 0;
  for (let i = 0; i < A.data.length; i++) scale = Math.max(scale, Math.abs(A.data[i]), Math.abs(A.im?.[i] || 0));
  scale ||= 1;
  const n = values.data.length;
  let result = maximum ? -Infinity : Infinity;
  for (let j = 0; j < n; j++) {
    let residual = 0, norm = 0;
    for (let i = 0; i < n; i++) {
      const vr = vectors.data[i * n + j], vi = vectors.im[i * n + j];
      let rr = -(values.data[j] / scale * vr - values.im[j] / scale * vi);
      let ri = -(values.data[j] / scale * vi + values.im[j] / scale * vr);
      for (let k = 0; k < n; k++) {
        const ar = A.data[i * n + k] / scale, ai = (A.im?.[i * n + k] || 0) / scale;
        rr += ar * vectors.data[k * n + j] - ai * vectors.im[k * n + j];
        ri += ar * vectors.im[k * n + j] + ai * vectors.data[k * n + j];
      }
      residual = Math.hypot(residual, rr, ri); norm = Math.hypot(norm, vr, vi);
    }
    // A large unrelated eigenvalue must not hide a small complex pair.
    // Classify roundoff per eigenpair using its actual backward residual.
    const tolerance = 64 * (residual * scale / norm
      + Number.EPSILON * Math.hypot(values.data[j], values.im[j]));
    if (Math.abs(values.im[j]) > tolerance) {
      throw new Error('eigenvalue extrema are undefined for complex eigenvalues (spec §09)');
    }
    result = maximum ? Math.max(result, values.data[j]) : Math.min(result, values.data[j]);
  }
  return result;
}

function _eigmax(A: any): number {
  A = valueLib._logicalDense(A);
  if (A.im || !_isSymmetric(A, A.shape[0], 0)) return _generalEigenExtreme(A, true);
  const { values } = _jacobiSymmetric(A, false);
  let max = values[0];
  for (let i = 1; i < values.length; i++) if (values[i] > max) max = values[i];
  return max;
}

function _eigmin(A: any): number {
  A = valueLib._logicalDense(A);
  if (A.im || !_isSymmetric(A, A.shape[0], 0)) return _generalEigenExtreme(A, false);
  const { values } = _jacobiSymmetric(A, false);
  let min = values[0];
  for (let i = 1; i < values.length; i++) if (values[i] < min) min = values[i];
  return min;
}

// =====================================================================
// Thin SVD via one-sided Jacobi rotations (spec §09)
// =====================================================================
//
// `_svd(A)` returns record(U, S, V) such that A = U · diag(S) · V^T,
// where:
//   - A is m × n with m ≥ n.
//   - U is m × n with orthonormal columns.
//   - S is a length-n vector of non-negative singular values (sorted
//     descending).
//   - V is n × n with orthonormal columns.
//
// Algorithm (one-sided Jacobi on A^T A, applied implicitly):
//   We rotate pairs of COLUMNS of A by Givens angles chosen so the
//   resulting column pair becomes orthogonal. Accumulate the rotations
//   into V. After convergence, the column norms of the rotated A are
//   the singular values, and the normalised columns form U.
//
// Pros: numerically stable for the size class FlatPPL targets;
// terminates in O(n) sweeps. Cons: O(m·n²) per sweep, worse than
// bidiag+QR for very tall thin matrices. Defer the more involved
// algorithms until a profile shows it.

function _columnNorm(B: Float64Array, m: number, n: number, j: number): number {
  let scale = 0;
  for (let i = 0; i < m; i++) scale = Math.max(scale, Math.abs(B[i * n + j]));
  if (scale === 0 || !Number.isFinite(scale)) return scale;
  let sum = 0;
  for (let i = 0; i < m; i++) {
    const x = B[i * n + j] / scale;
    sum += x * x;
  }
  return scale * Math.sqrt(sum);
}

function _svd(A: any): any {
  A = valueLib._logicalDense(A);
  const m = A.shape[0], n = A.shape[1];
  if (m < n) {
    // Aᴴ=U S Vᴴ implies A=V S Uᴴ, including complex and rank-deficient A.
    const f = _svd(valueLib.adjoint(A));
    return { U: f.V, S: f.S, V: f.U };
  }
  if (A.im) return require('./complex-svd.ts').svd(A);
  // Work on a fresh copy.
  const B = new Float64Array(m * n);
  for (let i = 0; i < m * n; i++) B[i] = A.data[i];
  // V starts as identity (n × n).
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;

  for (let sweep = 0; sweep < _JACOBI_MAX_SWEEPS; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        // Use raw products at normal scales; normalize columns separately
        // when squared norms overflow or underflow. A global scale can erase
        // a small singular value beside a large one.
        let a = 0, b = 0, c = 0;
        for (let i = 0; i < m; i++) {
          const xp = B[i * n + p], xq = B[i * n + q];
          a += xp * xp; b += xq * xq; c += xp * xq;
        }
        const normal = 2.2250738585072014e-308;
        if (a >= normal && b >= normal && Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c)) {
          if ((Math.abs(c) / Math.sqrt(a)) / Math.sqrt(b) <= _JACOBI_TOL_REL) continue;
        } else {
          const np = _columnNorm(B, m, n, p), nq = _columnNorm(B, m, n, q);
          if (np === 0 || nq === 0) continue;
          let corr = 0;
          for (let i = 0; i < m; i++) corr += (B[i * n + p] / np) * (B[i * n + q] / nq);
          if (Math.abs(corr) <= _JACOBI_TOL_REL) continue;
          const scale = Math.max(np, nq);
          const sp = np / scale, sq = nq / scale;
          a = sp * sp; b = sq * sq; c = (sp * sq) * corr;
        }
        if (c === 0) continue;
        // Givens rotation: zero c by rotating cols p, q by angle θ
        // with tan(2θ) = 2c / (a - b).
        const denom = a - b;
        let t: number;
        if (denom === 0) {
          t = (c > 0 ? 1 : -1);
        } else {
          const theta = (denom / c) / 2;
          const sign = theta >= 0 ? 1 : -1;
          t = Math.abs(theta) > 1e150 ? c / denom
            : sign / (Math.abs(theta) + Math.hypot(theta, 1));
        }
        if (t === 0) continue;
        rotated = true;
        const cs = 1 / Math.sqrt(t * t + 1);
        const sn = t * cs;
        // Rotate cols p, q of B:  new_col_p = cs·col_p + sn·col_q;
        //                          new_col_q = -sn·col_p + cs·col_q
        // (Convention chosen so new_col_p · new_col_q = 0.)
        for (let i = 0; i < m; i++) {
          const xp = B[i * n + p], xq = B[i * n + q];
          B[i * n + p] =  cs * xp + sn * xq;
          B[i * n + q] = -sn * xp + cs * xq;
        }
        // Accumulate the rotation into V (same column rotation).
        for (let i = 0; i < n; i++) {
          const vp = V[i * n + p], vq = V[i * n + q];
          V[i * n + p] =  cs * vp + sn * vq;
          V[i * n + q] = -sn * vp + cs * vq;
        }
      }
    }
    if (!rotated) break;
  }
  // Extract singular values from the column norms of B; U columns
  // are the normalised B columns.
  const S = new Float64Array(n);
  const U = new Float64Array(m * n);
  for (let j = 0; j < n; j++) {
    const norm = _columnNorm(B, m, n, j);
    S[j] = norm;
    if (norm > 0) {
      for (let i = 0; i < m; i++) U[i * n + j] = B[i * n + j] / norm;
    }
  }
  // Sort singular values descending. Permute U columns + V columns.
  const order = new Array(n);
  for (let i = 0; i < n; i++) order[i] = i;
  order.sort((a, b) => S[b] - S[a]);
  const Ssorted = new Float64Array(n);
  const Usorted = new Float64Array(m * n);
  const Vsorted = new Float64Array(n * n);
  for (let newJ = 0; newJ < n; newJ++) {
    const oldJ = order[newJ];
    Ssorted[newJ] = S[oldJ];
    for (let i = 0; i < m; i++) Usorted[i * n + newJ] = U[i * n + oldJ];
    for (let i = 0; i < n; i++) Vsorted[i * n + newJ] = V[i * n + oldJ];
  }
  // Complete the left basis after sorting, so every nonzero singular vector
  // is already present. Choose the coordinate with the largest residual and
  // reorthogonalize to avoid duplicating an existing vector.
  for (let j = 0; j < n; j++) {
    if (Ssorted[j] !== 0) continue;
    let best = 0, bestResidual = -Infinity;
    for (let i = 0; i < m; i++) {
      let residual = 1;
      for (let k = 0; k < j; k++) residual -= Usorted[i * n + k] ** 2;
      if (residual > bestResidual) { bestResidual = residual; best = i; }
    }
    Usorted[best * n + j] = 1;
    for (let pass = 0; pass < 2; pass++) for (let k = 0; k < j; k++) {
      let dot = 0;
      for (let i = 0; i < m; i++) dot += Usorted[i * n + k] * Usorted[i * n + j];
      for (let i = 0; i < m; i++) Usorted[i * n + j] -= dot * Usorted[i * n + k];
    }
    let norm = 0;
    for (let i = 0; i < m; i++) norm = Math.hypot(norm, Usorted[i * n + j]);
    for (let i = 0; i < m; i++) Usorted[i * n + j] /= norm;
  }
  return {
    U: _matValue(Usorted, m, n),
    S: _vecValue(Ssorted, n),
    V: _matValue(Vsorted, n, n),
  };
}

// =====================================================================
// rank(A) — numerical rank via SVD (spec §09)
// =====================================================================
//
// Counts singular values above the tolerance
//   tol = max(m, n) · σ_max · ε_machine
// (the conventional choice, matching NumPy / Octave defaults). Returns
// an integer.

function _rank(A: any, rtol = -1, atol = 0): number {
  valueLib.requireMatrix(A, 'rank');
  const [relative, absolute] = _cutoffs(A, rtol, atol), S = _svd(A).S.data;
  const cutoff = _singularCutoff(S, relative, absolute);
  let rank = 0;
  for (const s of S) if (s > cutoff) rank++;
  return rank;
}

/** Symmetric/Hermitian eigenpairs with a real ascending spectrum. Validate
 * both triangles before averaging roundoff; never treat a general matrix as
 * Hermitian by reading just its lower triangle. */
function _hermitianEigen(A: any, computeVectors: boolean): any {
  valueLib.requireMatrix(A, 'eigh');
  A = valueLib._logicalDense(A);
  const n = A.shape[0];
  if (n !== A.shape[1]) throw new Error('eigh: A must be square');
  let scale = 0;
  for (let i = 0; i < A.data.length; i++) {
    const ar = A.data[i], ai = A.im?.[i] ?? 0;
    if (!Number.isFinite(ar) || !Number.isFinite(ai)) throw new Error('eigh: A must be finite');
    scale = Math.max(scale, Math.abs(ar), Math.abs(ai));
  }
  scale ||= 1;
  const tolerance = 64 * Number.EPSILON * Math.max(1, n);
  for (let i = 0; i < n; i++) for (let j = i; j < n; j++) {
    const a = i * n + j, b = j * n + i;
    if (Math.abs(A.data[a] / scale - A.data[b] / scale) > tolerance
        || Math.abs((A.im?.[a] ?? 0) / scale + (A.im?.[b] ?? 0) / scale) > tolerance) {
      throw new Error('eigh: A must be symmetric or Hermitian');
    }
  }
  const f = A.im ? require('./hermitian-eigen.ts').eigen(A, computeVectors, scale)
    : _jacobiSymmetric(A, computeVectors, 8 * Number.EPSILON);
  const order = Array.from({ length: n }, (_, j) => j).sort((a, b) => f.values[a] - f.values[b]);
  const values = _vecValue(Float64Array.from(order, j => f.values[j]), n);
  if (!computeVectors) return values;
  const source = A.im ? f.vectors : _matValue(f.vectors, n, n);
  const re = new Float64Array(n * n), im = A.im ? new Float64Array(n * n) : null;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    re[i * n + j] = source.data[i * n + order[j]];
    if (im) im[i * n + j] = source.im[i * n + order[j]];
  }
  // Fix the arbitrary phase at the largest coordinate, breaking equal-magnitude
  // ties by the first row. This leaves repeated eigenspaces unconstrained.
  for (let j = 0; j < n; j++) {
    let pivot = 0, size = -1;
    for (let i = 0; i < n; i++) {
      const norm = Math.hypot(re[i * n + j], im?.[i * n + j] ?? 0);
      if (norm > size) { size = norm; pivot = i; }
    }
    const at = pivot * n + j, pr = re[at] / size, pi = (im?.[at] ?? 0) / size;
    for (let i = 0; i < n; i++) {
      const index = i * n + j, ar = re[index], ai = im?.[index] ?? 0;
      re[index] = ar * pr + ai * pi;
      if (im) im[index] = ai * pr - ar * pi;
    }
    re[at] = size;
    if (im) im[at] = 0;
  }
  return { values, vectors: im ? valueLib.complexValue(re, im, [n, n]) : _matValue(re, n, n) };
}

function _eigh(A: any): any { return _hermitianEigen(A, true); }
function _eigvalsh(A: any): any { return _hermitianEigen(A, false); }

module.exports = {
  _lu,
  _kron,
  _matexp,
  _qr,
  _lstsq,
  _eigen,
  _eigmax,
  _eigmin,
  _svd,
  _rank,
  _pinv,
  _cond,
  _eigh,
  _eigvalsh,
  // Test-only / internal helpers
  _internal: {
    _matmul,
    _matAdd,
    _matScale,
    _identity,
    _norm1,
  },
};
