// packages/engine/mh-kernel.ts
'use strict';
const { gaussianNoise } = require('./mcmc-driver.ts');

// Adaptive Metropolis (Haario, Saksman & Tamminen 2001) with a whole-vector
// block proposal: one logπ per walker per sweep (not dim× as Metropolis-within-
// Gibbs). The proposal covariance is the empirical posterior covariance
// estimated from warmup samples, applied via its Cholesky factor — so the
// sampler moves along the posterior's correlated directions (a plain diagonal
// proposal mixes terribly when parameters are correlated, e.g. α/β in a
// regression). A scalar step is also adapted toward the optimal accept rate.
//
//   proposal:  y' = y + scale · L z,   z ~ N(0, I),  L Lᵀ = Cov_warmup + εI
//   scale → 2.38/√dim  (the optimal RW-MH scaling), nudged toward accept 0.234.

// In-place Cholesky of a symmetric PD matrix A (flat row-major, dim×dim) into a
// lower-triangular L (flat). Returns false if A is not PD (caller falls back to
// a diagonal proposal).
function cholesky(A: Float64Array, L: Float64Array, dim: number): boolean {
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * dim + j];
      for (let k = 0; k < j; k++) s -= L[i * dim + k] * L[j * dim + k];
      if (i === j) {
        if (s <= 0) return false;
        L[i * dim + j] = Math.sqrt(s);
      } else {
        L[i * dim + j] = s / L[j * dim + j];
      }
    }
    for (let j = i + 1; j < dim; j++) L[i * dim + j] = 0;
  }
  return true;
}

const mhKernel = {
  init(_nWalkers: number, dim: number, opts: any) {
    // L starts as initStep·I (independent diagonal proposal) until enough
    // warmup samples accrue to estimate a covariance.
    const initStep = opts.initStep ?? 1;
    const L = new Float64Array(dim * dim);
    for (let d = 0; d < dim; d++) L[d * dim + d] = initStep;
    return {
      dim,
      scale: 2.38 / Math.sqrt(Math.max(dim, 1)), // RW-MH optimal scaling, accept-adapted
      target: 0.234,
      L,                                          // proposal Cholesky factor (lower)
      // Running mean + second-moment for the empirical covariance.
      sum: new Float64Array(dim),
      cross: new Float64Array(dim * dim),         // Σ x xᵀ
      count: 0,
      z: new Float64Array(dim),                   // scratch standard-normal vector
      iter: 0,
    };
  },
  step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, phase: string) {
    const dim = adaptState.dim, nWalkers = ensemble.length;
    const L = adaptState.L, z = adaptState.z;
    const warm = phase === 'warmup';
    let accepts = 0, proposals = 0;

    for (let w = 0; w < nWalkers; w++) {
      const y = ensemble[w];
      for (let d = 0; d < dim; d++) z[d] = gaussianNoise(prng);
      // yProp = y + scale · L z   (L lower-triangular)
      const yProp = Float64Array.from(y);
      const sc = adaptState.scale;
      for (let i = 0; i < dim; i++) {
        let acc = 0;
        for (let k = 0; k <= i; k++) acc += L[i * dim + k] * z[k];
        yProp[i] = y[i] + sc * acc;
      }
      const lpProp = mv.logPosterior(yProp);     // ONE logπ for the whole vector
      proposals++;
      const accepted = Math.log(prng() + 1e-300) < (lpProp - logp[w]);
      if (accepted) { ensemble[w] = yProp; logp[w] = lpProp; accepts++; }

      if (warm) {
        const g = (accepted ? 1 : 0) - adaptState.target;
        adaptState.scale *= Math.exp(g / Math.sqrt(adaptState.iter + 1));
        const cur = ensemble[w];                 // accumulate the (possibly updated) position
        const sum = adaptState.sum, cross = adaptState.cross;
        for (let i = 0; i < dim; i++) {
          sum[i] += cur[i];
          for (let j = 0; j <= i; j++) cross[i * dim + j] += cur[i] * cur[j];
        }
        adaptState.count++;
      }
    }

    if (warm) {
      adaptState.iter++;
      // Re-estimate the proposal covariance from the warmup samples every 50
      // sweeps once enough have accrued (need > dim for a non-degenerate cov).
      const n = adaptState.count;
      if (n > dim + 2 && adaptState.iter % 50 === 0) {
        const sum = adaptState.sum, cross = adaptState.cross;
        const cov = new Float64Array(dim * dim);
        const eps = 1e-6;
        for (let i = 0; i < dim; i++) {
          const mi = sum[i] / n;
          for (let j = 0; j <= i; j++) {
            const c = cross[i * dim + j] / n - mi * (sum[j] / n);
            cov[i * dim + j] = c + (i === j ? eps : 0);
            cov[j * dim + i] = cov[i * dim + j];
          }
        }
        const Lnew = new Float64Array(dim * dim);
        if (cholesky(cov, Lnew, dim)) {
          adaptState.L = Lnew;
        } else {
          // Not PD (e.g. a near-degenerate direction): fall back to a diagonal
          // proposal from the marginal std so we never propose with a broken L.
          const Ld = new Float64Array(dim * dim);
          for (let d = 0; d < dim; d++) {
            const v = cross[d * dim + d] / n - (sum[d] / n) * (sum[d] / n);
            Ld[d * dim + d] = Math.sqrt(v > eps ? v : eps);
          }
          adaptState.L = Ld;
        }
      }
    }
    return { accepts, proposals };
  },
};

module.exports = { mhKernel };
