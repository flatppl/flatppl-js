'use strict';

// Per-atom transforms from scalar leaf draws to the §08 multivariate
// variates. The materialiser (mat-multivariate.ts) and the value-position
// walker (sampler.ts) both draw the scalar leaves and share these maps.

// Wishart(nu, scale) by the Bartlett decomposition. `LS` is the dense n×n
// lower Cholesky factor of `scale` (of `inv(scale)` when `invert`), `chi2[i]`
// a ChiSquared(nu - i) draw and `normals` the n(n-1)/2 strictly-lower entries
// of A in row-major order. With T = LS·A, writes T·Tᵀ, or (T·Tᵀ)⁻¹ when
// `invert` (§08: InverseWishart(nu, scale) = pushfwd(inv, Wishart(nu, inv(scale)))).
function bartlettAtom(LS: ArrayLike<number>, n: number, chi2: ArrayLike<number>,
                      normals: ArrayLike<number>, invert: boolean,
                      out: Float64Array, outOff: number): void {
  const A = new Float64Array(n * n);
  for (let i = 0; i < n; i++) A[i * n + i] = Math.sqrt(chi2[i]);
  let o = 0;
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < i; j++) A[i * n + j] = normals[o++];
  }
  const T = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += LS[i * n + k] * A[k * n + j];
      T[i * n + j] = s;
    }
  }
  if (!invert) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += T[i * n + k] * T[j * n + k];
        out[outOff + i * n + j] = s;
      }
    }
    return;
  }
  // (T·Tᵀ)⁻¹ = T⁻ᵀ·T⁻¹ with T lower triangular.
  const tinv = new Float64Array(n * n);
  for (let j = 0; j < n; j++) {
    for (let i = j; i < n; i++) {
      let s = (i === j) ? 1 : 0;
      for (let k = j; k < i; k++) s -= T[i * n + k] * tinv[k * n + j];
      tinv[i * n + j] = s / T[i * n + i];
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += tinv[k * n + i] * tinv[k * n + j];
      out[outOff + i * n + j] = s;
    }
  }
}

// Beta parameters of the onion method (Lewandowski, Kurowicka and Joe 2009,
// §3.2) for row i ≥ 1 (0-indexed) of an LKJCholesky(n, eta) factor.
function onionBetaParams(n: number, eta: number, i: number): { alpha: number, beta: number } {
  return { alpha: i / 2, beta: eta + (n - i - 1) / 2 };
}

// LKJCholesky(n, eta) by the onion method. `betas[i - 1]` is the
// onionBetaParams draw for row i and `normals` holds n(n-1)/2 standard
// normals, row i using i of them for its uniform direction. Row i of L is
// sqrt(y)·u with diagonal sqrt(1 - y), so every row has unit norm.
function onionCholeskyAtom(n: number, betas: ArrayLike<number>, normals: ArrayLike<number>,
                           out: Float64Array, outOff: number): void {
  for (let k = 0; k < n * n; k++) out[outOff + k] = 0;
  out[outOff] = 1;
  let o = 0;
  for (let i = 1; i < n; i++) {
    const y = betas[i - 1];
    let norm2 = 0;
    for (let k = 0; k < i; k++) norm2 += normals[o + k] * normals[o + k];
    const r = Math.sqrt(y);
    const invNorm = norm2 > 0 ? 1 / Math.sqrt(norm2) : 0;
    for (let k = 0; k < i; k++) out[outOff + i * n + k] = r * normals[o + k] * invNorm;
    out[outOff + i * n + i] = Math.sqrt(1 - y);
    o += i;
  }
}

// §08: LKJ(n, eta) = pushfwd(row_gram, LKJCholesky(n, eta)).
function rowGramAtom(L: ArrayLike<number>, lOff: number, n: number,
                     out: Float64Array, outOff: number): void {
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      const maxK = Math.min(i, j);
      for (let k = 0; k <= maxK; k++) s += L[lOff + i * n + k] * L[lOff + j * n + k];
      out[outOff + i * n + j] = s;
    }
  }
}

module.exports = { bartlettAtom, onionBetaParams, onionCholeskyAtom, rowGramAtom };
