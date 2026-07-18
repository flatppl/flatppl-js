// packages/engine/slice-kernel.ts
'use strict';

// General univariate slice sampling (Neal 2003): stepping-out + shrinkage, one
// coordinate at a time in a (per-sweep) random order. Self-tuning — no proposal
// covariance, no Gaussian assumption (unlike elliptical slice). Per-coordinate
// widths w_i are seeded once from the prior-pool marginal std and held fixed;
// there is no ongoing adaptation, so chains are independent from step 1.
//
// Cost: slice makes MANY logπ evals per sweep (stepping-out + shrinkage per
// coordinate) — inherently far more than a random-walk proposal. Evals are
// sequential WITHIN a chain (each shrink depends on the last), but ACROSS chains
// they lock-step: all chains process the same coordinate together, so each
// stepping-out / shrink round scores every still-active chain in ONE
// mv.logPosteriorBatch call (the elliptical-slice batching pattern). On an
// expensive density this is the dominant speedup — the whole ensemble runs in one
// worker to keep the batch as large as possible. A scalar fallback covers a
// ModelView without logPosteriorBatch.
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
      const dim = adaptState.dim, order = adaptState.order;
      // Shared random coordinate order for the sweep (so chains lock-step per coord).
      for (let d = 0; d < dim; d++) order[d] = d;
      for (let d = dim - 1; d > 0; d--) { const j = Math.floor(prng() * (d + 1)); const t = order[d]; order[d] = order[j]; order[j] = t; }
      // Batch across chains only when there is more than one to batch AND a
      // batch scorer exists; a lone chain (the common pooled case — one chain per
      // worker) takes the scalar path, avoiding batch-of-1 array overhead.
      if (ensemble.length > 1 && typeof mv.logPosteriorBatch === 'function') {
        return stepBatched(ensemble, logp, mv, prng, adaptState, order);
      }
      return stepScalar(ensemble, logp, mv, prng, adaptState, order);
    },
  };
}

// Across-chain-batched sweep: all chains process each coordinate together, and
// every stepping-out / shrink round scores its still-active chains in one
// logPosteriorBatch. `proposals` counts individual chain evals (so acceptRate =
// updates/evals matches the scalar path's meaning).
function stepBatched(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, A: any, order: Int32Array) {
  const dim = A.dim, nW = ensemble.length, w = A.w, mCap = A.m;
  let evals = 0, updates = 0;
  const props = ensemble.map((y) => Float64Array.from(y));   // per-chain working vectors
  const x0 = new Float64Array(nW), logu = new Float64Array(nW), L = new Float64Array(nW), R = new Float64Array(nW);
  const jj = new Int32Array(nW), kk = new Int32Array(nW), accepted = new Uint8Array(nW);
  const idx: number[] = [];
  const scoreActive = () => { const batch: Float64Array[] = new Array(idx.length); for (let a = 0; a < idx.length; a++) batch[a] = props[idx[a]]; evals += idx.length; return mv.logPosteriorBatch(batch); };
  for (let oi = 0; oi < dim; oi++) {
    const i = order[oi], wi = w[i];
    for (let c = 0; c < nW; c++) {
      x0[c] = ensemble[c][i]; logu[c] = logp[c] + Math.log(prng() + 1e-300);
      L[c] = x0[c] - wi * prng(); R[c] = L[c] + wi;
      jj[c] = Math.floor(mCap * prng()); kk[c] = mCap - 1 - jj[c]; accepted[c] = 0;
    }
    // Stepping-out, expand left: expand each chain's L while logπ(atL) > slice level.
    for (;;) {
      idx.length = 0; for (let c = 0; c < nW; c++) if (jj[c] > 0) idx.push(c);
      if (!idx.length) break;
      for (const c of idx) props[c][i] = L[c];
      const lps = scoreActive();
      for (let a = 0; a < idx.length; a++) { const c = idx[a]; if (lps[a] > logu[c]) { L[c] -= wi; jj[c]--; } else jj[c] = 0; }
    }
    // Stepping-out, expand right.
    for (;;) {
      idx.length = 0; for (let c = 0; c < nW; c++) if (kk[c] > 0) idx.push(c);
      if (!idx.length) break;
      for (const c of idx) props[c][i] = R[c];
      const lps = scoreActive();
      for (let a = 0; a < idx.length; a++) { const c = idx[a]; if (lps[a] > logu[c]) { R[c] += wi; kk[c]--; } else kk[c] = 0; }
    }
    // Shrinkage: sample in [L,R]; accept above the level, else shrink toward x0.
    for (;;) {
      idx.length = 0; for (let c = 0; c < nW; c++) if (!accepted[c]) idx.push(c);
      if (!idx.length) break;
      const xp = new Float64Array(nW);
      for (const c of idx) { xp[c] = L[c] + prng() * (R[c] - L[c]); props[c][i] = xp[c]; }
      const lps = scoreActive();
      for (let a = 0; a < idx.length; a++) {
        const c = idx[a];
        if (lps[a] > logu[c]) { ensemble[c][i] = xp[c]; logp[c] = lps[a]; accepted[c] = 1; updates++; }
        else {
          if (xp[c] < x0[c]) L[c] = xp[c]; else R[c] = xp[c];
          if (R[c] - L[c] < 1e-12) { ensemble[c][i] = x0[c]; accepted[c] = 1; }   // interval collapse: keep x0 (logp unchanged)
        }
      }
    }
    for (let c = 0; c < nW; c++) props[c][i] = ensemble[c][i];   // sync working vector to the accepted value
  }
  return { accepts: updates, proposals: evals };
}

// Scalar fallback (one chain, one coordinate at a time) for a ModelView without
// logPosteriorBatch.
function stepScalar(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, A: any, order: Int32Array) {
  const dim = A.dim, nWalkers = ensemble.length, w = A.w, mCap = A.m;
  let evals = 0, updates = 0;
  for (let c = 0; c < nWalkers; c++) {
    const y = ensemble[c];
    for (let oi = 0; oi < dim; oi++) {
      const i = order[oi];
      const x0 = y[i];
      const logu = logp[c] + Math.log(prng() + 1e-300);
      const wi = w[i];
      let L = x0 - wi * prng(), R = L + wi;
      let j = Math.floor(mCap * prng()), k = mCap - 1 - j;
      y[i] = L; while (j > 0) { evals++; if (mv.logPosterior(y) > logu) { L -= wi; y[i] = L; j--; } else break; }
      y[i] = R; while (k > 0) { evals++; if (mv.logPosterior(y) > logu) { R += wi; y[i] = R; k--; } else break; }
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
}

module.exports = { makeSliceKernel };
