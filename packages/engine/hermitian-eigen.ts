'use strict';

// Complex Hermitian Jacobi similarities for the ext-linear-algebra module.
// The real path remains in ext-linalg.ts. Phase-aware plane rotations preserve
// orthonormal columns even when eigenvalues repeat; general Schur backsolves
// cannot provide that guarantee in a degenerate eigenspace.
const V = require('./value.ts');

/** Apply J=[c,sρ;-s*conj(ρ),c] to two columns without a dense product. */
function rotate(re: Float64Array, im: Float64Array, n: number, p: number, q: number,
    c: number, s: number, pr: number, pi: number): void {
  for (let i = 0; i < n; i++) {
    const a = i * n + p, b = i * n + q;
    const ar = re[a], ai = im[a], br = re[b], bi = im[b];
    re[a] = c * ar - s * (pr * br + pi * bi);
    im[a] = c * ai - s * (pr * bi - pi * br);
    re[b] = s * (pr * ar - pi * ai) + c * br;
    im[b] = s * (pr * ai + pi * ar) + c * bi;
  }
}

/** Cyclic unitary similarities H←JᴴHJ; the wrapper validates Hermiticity. */
function eigen(A: any, computeVectors: boolean, scale: number): any {
  const n = A.shape[0], re = new Float64Array(A.data), im = new Float64Array(A.im);
  const vr = computeVectors ? new Float64Array(n * n) : null;
  const vi = computeVectors ? new Float64Array(n * n) : null;
  if (vr) for (let i = 0; i < n; i++) vr[i * n + i] = 1;
  for (let i = 0; i < n; i++) {
    im[i * n + i] = 0;
    for (let j = i + 1; j < n; j++) {
      const a = i * n + j, b = j * n + i;
      const realSum = re[a] + re[b], imagDifference = im[a] - im[b];
      re[a] = re[b] = Number.isFinite(realSum) ? realSum / 2 : re[a] / 2 + re[b] / 2;
      im[a] = Number.isFinite(imagDifference) ? imagDifference / 2 : im[a] / 2 - im[b] / 2;
      im[b] = -im[a];
    }
  }
  const tolerance = 8 * Number.EPSILON;
  let converged = n < 2;
  for (let sweep = 0; sweep < 100 && !converged; sweep++) {
    let largest = 0;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      const pq = p * n + q, z = Math.hypot(re[pq], im[pq]);
      largest = Math.max(largest, z / scale);
      if (z / scale <= tolerance) continue;
      const pr = re[pq] / z, pi = im[pq] / z;
      const app = re[p * n + p], aqq = re[q * n + q];
      const theta = (aqq / scale - app / scale) / (2 * (z / scale));
      const t = theta === 0 ? 1 : Math.sign(theta) / (Math.abs(theta) + Math.hypot(theta, 1));
      const c = 1 / Math.hypot(1, t), s = c * t;
      // Only off-pivot rows are needed. Restore the conjugate rows after the
      // two-column update, then set the diagonal and eliminated pivot exactly.
      rotate(re, im, n, p, q, c, s, pr, pi);
      for (let i = 0; i < n; i++) if (i !== p && i !== q) {
        re[p * n + i] = re[i * n + p]; im[p * n + i] = -im[i * n + p];
        re[q * n + i] = re[i * n + q]; im[q * n + i] = -im[i * n + q];
      }
      re[p * n + p] = app - t * z; re[q * n + q] = aqq + t * z;
      im[p * n + p] = im[q * n + q] = 0;
      re[pq] = re[q * n + p] = im[pq] = im[q * n + p] = 0;
      if (vr && vi) rotate(vr, vi, n, p, q, c, s, pr, pi);
    }
    converged = largest <= tolerance;
  }
  if (!converged) throw new Error('eigh: Hermitian Jacobi iteration did not converge');
  return { values: Float64Array.from({ length: n }, (_, i) => re[i * n + i]),
    vectors: vr ? V.complexValue(vr, vi, [n, n]) : null };
}

module.exports = { eigen };
