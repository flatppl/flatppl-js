// packages/engine/ram-kernel.ts
'use strict';

// Rank-1 Cholesky update/downdate. Given a lower-triangular S (flat row-major,
// dim×dim) with C = S Sᵀ, update S IN PLACE to the Cholesky factor of
// C + alpha·v vᵀ, in O(dim²). `alpha` may be negative (a downdate). `v` is
// scratch and is OVERWRITTEN. Returns false if a downdate loses positive-
// definiteness (the caller should discard the partially-updated S and keep the
// previous one). Standard cholupdate (Golub & Van Loan); the sign generalises
// it to both update and downdate.
function cholRank1Update(S: Float64Array, v: Float64Array, alpha: number, dim: number): boolean {
  const sgn = alpha >= 0 ? 1 : -1;
  const a = Math.sqrt(Math.abs(alpha));
  for (let i = 0; i < dim; i++) v[i] *= a;
  for (let k = 0; k < dim; k++) {
    const Skk = S[k * dim + k];
    const r2 = Skk * Skk + sgn * v[k] * v[k];
    if (r2 <= 0) return false;
    const r = Math.sqrt(r2);
    const c = r / Skk;
    const s = v[k] / Skk;
    S[k * dim + k] = r;
    for (let i = k + 1; i < dim; i++) {
      S[i * dim + k] = (S[i * dim + k] + sgn * s * v[i]) / c;
      v[i] = c * v[i] - s * S[i * dim + k];
    }
  }
  return true;
}

module.exports = { cholRank1Update };
