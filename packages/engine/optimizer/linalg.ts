'use strict';

// =====================================================================
// optimizer/linalg.ts — small dense linear algebra for the optimizer
// =====================================================================
//
// Self-contained helpers the CMA-ES core needs on a symmetric covariance
// matrix C: a symmetric eigensolver (cyclic Jacobi rotations) and the
// derived C^{1/2} / C^{-1/2} (CMA-ES samples candidates as m + σ·C^{1/2}·z
// and updates the step-size path with C^{-1/2}). Dependency-free and
// low-dimensional by intent — optimizer parameter spaces are small, so a
// robust O(n³) Jacobi per generation is the right trade (clarity over the
// engine's general `ext-linalg.eigen`, which targets larger/possibly-
// complex problems). Pure CJS leaf (no engine imports).

/** A·v for a square matrix A (rows) and vector v. */
function matvec(A: any, v: any): number[] {
  const n = A.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    const Ai = A[i];
    let s = 0;
    for (let j = 0; j < n; j++) s += Ai[j] * v[j];
    out[i] = s;
  }
  return out;
}

/**
 * Symmetric eigendecomposition via cyclic Jacobi rotations.
 * Returns `{ values, vectors }` with `A ≈ vectors · diag(values) · vectorsᵀ`;
 * column `j` of `vectors` is a unit eigenvector for `values[j]`. `A` must be
 * (numerically) symmetric. Order of eigenpairs is unspecified.
 */
function symEig(Ain: any): { values: number[]; vectors: number[][] } {
  const n = Ain.length;
  if (n === 1) return { values: [Ain[0][0]], vectors: [[1]] };
  const A = Ain.map((row: any) => row.slice());
  const V: number[][] = [];
  for (let i = 0; i < n; i++) { V.push(new Array(n).fill(0)); V[i][i] = 1; }

  const MAX_SWEEPS = 100;
  for (let sweep = 0; sweep < MAX_SWEEPS; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-30) break;

    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        // Givens rotation G(p,q,θ) with G[p][p]=G[q][q]=c, G[p][q]=s,
        // G[q][p]=−s. A ← Gᵀ A G zeros the (p,q) entry when
        // tan(2θ) = 2·apq/(aqq − app), i.e. 2θ = atan2(2·apq, aqq − app).
        const theta = 0.5 * Math.atan2(2 * apq, A[q][q] - A[p][p]);
        const c = Math.cos(theta), s = Math.sin(theta);

        // A ← A·G (update columns p,q)
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        // A ← Gᵀ·A (update rows p,q)
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        // V ← V·G (accumulate eigenvectors as columns)
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const values: number[] = [];
  for (let i = 0; i < n; i++) values.push(A[i][i]);
  return { values, vectors: V };
}

/** V·diag(d)·Vᵀ for orthonormal-column V and a diagonal vector d. */
function vDiagVt(V: number[][], d: number[]): number[][] {
  const n = V.length;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    out.push(new Array(n).fill(0));
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += V[i][k] * d[k] * V[j][k];
      out[i][j] = s;
    }
  }
  return out;
}

/**
 * Symmetric matrix square root and inverse square root of a (numerically)
 * positive-semidefinite `A`. Eigenvalues are floored to a tiny positive
 * value so a degenerate/near-singular `C` still yields a usable factor.
 */
function matSqrtAndInvSqrt(A: any): {
  sqrt: number[][]; invSqrt: number[][]; values: number[]; vectors: number[][];
} {
  const { values, vectors } = symEig(A);
  const dpos = values.map((v) => Math.sqrt(Math.max(v, 1e-30)));
  const dinv = dpos.map((v) => 1 / v);
  return {
    sqrt: vDiagVt(vectors, dpos),
    invSqrt: vDiagVt(vectors, dinv),
    values, vectors,
  };
}

module.exports = { matvec, symEig, vDiagVt, matSqrtAndInvSqrt };
