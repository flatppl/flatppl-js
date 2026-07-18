// packages/engine/slice-kernel.ts
'use strict';

// General univariate slice sampling (Neal 2003): stepping-out + shrinkage, one
// coordinate at a time in a random order per sweep. Self-tuning — no proposal
// covariance, no Gaussian assumption (unlike elliptical slice). Per-coordinate
// widths w_i are seeded once from the prior-pool marginal std and held fixed;
// there is no ongoing adaptation, so chains are independent from step 1 (the
// worker pool distributes them directly). Each coordinate update conditions on
// the current values of the others via the full mv.logPosterior — evals are
// sequential within a chain (each shrink depends on the last), so slice does not
// batch; parallelism is across chains.
function makeSliceKernel(opts?: any) {
  const m = (opts && opts.m) || 50;   // max stepping-out expansions per coordinate
  return {
    init(_nWalkers: number, dim: number, _o: any, mv: any) {
      const w = new Float64Array(dim);
      // Seed w_i from the prior-pool marginal std (floored). initFromPrior may be
      // absent in a bare driver harness; fall back to unit widths.
      let seeded = false;
      if (mv && typeof mv.initFromPrior === 'function') {
        const n = Math.max(64, 8 * dim);
        const pool: Float64Array[] = mv.initFromPrior(n, Math.random);   // pool std only; RNG choice irrelevant to the width scale
        if (pool && pool.length) {
          for (let d = 0; d < dim; d++) {
            let mu = 0; for (let p = 0; p < pool.length; p++) mu += pool[p][d]; mu /= pool.length;
            let v = 0; for (let p = 0; p < pool.length; p++) { const e = pool[p][d] - mu; v += e * e; } v /= pool.length;
            w[d] = v > 1e-12 ? Math.sqrt(v) : 1;
          }
          seeded = true;
        }
      }
      if (!seeded) for (let d = 0; d < dim; d++) w[d] = 1;
      return { dim, w, m, order: new Int32Array(dim) };
    },
    step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, _phase: string) {
      const dim = adaptState.dim, nWalkers = ensemble.length, w = adaptState.w, order = adaptState.order, mCap = adaptState.m;
      let evals = 0, updates = 0;
      for (let c = 0; c < nWalkers; c++) {
        const y = ensemble[c];
        // Random coordinate order (Fisher-Yates) so the sweep isn't biased.
        for (let d = 0; d < dim; d++) order[d] = d;
        for (let d = dim - 1; d > 0; d--) { const j = Math.floor(prng() * (d + 1)); const t = order[d]; order[d] = order[j]; order[j] = t; }
        for (let oi = 0; oi < dim; oi++) {
          const i = order[oi];
          const x0 = y[i];
          const logu = logp[c] + Math.log(prng() + 1e-300);   // slice level under the current density
          const wi = w[i];
          // Stepping-out: a width-w_i interval placed randomly around x0, expanded
          // outward until both ends fall below logu, up to mCap total steps.
          let L = x0 - wi * prng(), R = L + wi;
          let j = Math.floor(mCap * prng()), k = mCap - 1 - j;
          y[i] = L; while (j > 0) { evals++; if (mv.logPosterior(y) > logu) { L -= wi; y[i] = L; j--; } else break; }
          y[i] = R; while (k > 0) { evals++; if (mv.logPosterior(y) > logu) { R += wi; y[i] = R; k--; } else break; }
          // Shrinkage: sample in [L,R]; on a reject, shrink the bound toward x0 on
          // the side of the rejected point and resample. Guard against interval
          // collapse (flat/degenerate density) by accepting x0.
          let accepted = false;
          while (!accepted) {
            const xp = L + prng() * (R - L);
            y[i] = xp; const lp = mv.logPosterior(y); evals++;
            if (lp > logu) { logp[c] = lp; updates++; accepted = true; }
            else if (xp < x0) L = xp;
            else R = xp;
            if (!accepted && R - L < 1e-12) { y[i] = x0; logp[c] = mv.logPosterior(y); evals++; accepted = true; }
          }
        }
      }
      return { accepts: updates, proposals: evals };
    },
  };
}

module.exports = { makeSliceKernel };
