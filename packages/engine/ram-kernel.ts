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

const driver = require('./mcmc-driver.ts');
const { gaussianNoise } = driver;

// In-place Cholesky of a symmetric PD matrix A (flat row-major, dim×dim) into a
// lower-triangular L (flat). Returns false if A is not PD.
function denseCholesky(A: Float64Array, L: Float64Array, dim: number): boolean {
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

// RAM: Robust Adaptive Metropolis (Vihola 2012), with an informed proposal.
// Proposes y' = y + S·z (z ~ N(0,I)) and, during warmup, adapts the Cholesky
// factor S toward a target acceptance rate via a rank-1 update of C = S Sᵀ:
//   C ← C + η_n (α − α*) · (S ẑ)(S ẑ)ᵀ,   ẑ = z/‖z‖,   η_n = n^{-γ}.
// The rank-1 update alone starts isotropic and adapts slowly, so it fails to
// capture a strongly anisotropic / stiff posterior (e.g. a funnel: one
// coordinate's marginal sd orders of magnitude below another's) within a short
// warmup — every block proposal is dominated by the over-large move on the stiff
// coordinate and rejected. So warmup ALSO periodically re-anchors S to the
// empirical posterior covariance (the Haario recipe the mh kernel uses:
// per-coordinate variances, shrunk off-diagonals once enough samples accrue,
// diagonal fallback), scaled by the optimal RW factor 2.38/√d; the rank-1
// acceptance update then refines that anchor. This gives RAM the anisotropy
// capture that makes it robust, keeping its acceptance-targeted refinement.
// S is stored in adaptState.L and adaptState.scale is fixed at 1 (scale folds
// into S) so the freeze-then-parallel machinery (which freezes {L, scale}) works
// unchanged. The sample phase is the shared fixed-proposal batched step.
function makeRamKernel(opts?: any) {
  const gamma = (opts && typeof opts.gamma === 'number') ? opts.gamma : 2 / 3;
  return {
    init(_nWalkers: number, dim: number, _o: any) {
      const target = dim === 1 ? 0.44 : 0.234;
      const S = new Float64Array(dim * dim);
      const s0 = 2.38 / Math.sqrt(Math.max(dim, 1));   // RW-MH optimal scale as the initial S
      for (let d = 0; d < dim; d++) S[d * dim + d] = s0;
      return {
        dim, scale: 1, target, gamma, L: S, iter: 0,
        z: new Float64Array(dim), step: new Float64Array(dim),
        // Running mean + second moment for the empirical covariance re-anchor.
        sum: new Float64Array(dim), cross: new Float64Array(dim * dim), count: 0, sweeps: 0,
      };
    },
    step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, phase: string) {
      const dim = adaptState.dim, nWalkers = ensemble.length;
      const S = adaptState.L, z = adaptState.z, stepv = adaptState.step;
      const warm = phase === 'warmup';
      // Sample phase: fixed proposal, one batched score (bit-identical to scalar).
      if (!warm && typeof mv.logPosteriorBatch === 'function') {
        return driver.fixedProposalBatchStep(ensemble, logp, mv, prng, S, adaptState.scale, dim, z);
      }
      const kappa = 2.38 / Math.sqrt(Math.max(dim, 1));
      let accepts = 0, proposals = 0;
      for (let w = 0; w < nWalkers; w++) {
        const y = ensemble[w];
        let zn2 = 0;
        for (let d = 0; d < dim; d++) { z[d] = gaussianNoise(prng); zn2 += z[d] * z[d]; }
        const yProp = Float64Array.from(y);
        for (let i = 0; i < dim; i++) {
          let acc = 0;
          for (let k = 0; k <= i; k++) acc += S[i * dim + k] * z[k];
          stepv[i] = acc;              // S·z (the proposal increment)
          yProp[i] = y[i] + acc;       // scale === 1
        }
        const lpProp = mv.logPosterior(yProp);
        proposals++;
        const dlp = lpProp - logp[w];
        const accepted = Math.log(prng() + 1e-300) < dlp;
        if (accepted) { ensemble[w] = yProp; logp[w] = lpProp; accepts++; }
        if (warm) {
          const alphaAcc = dlp >= 0 ? 1 : Math.exp(dlp);      // Metropolis accept probability
          adaptState.iter++;
          const eta = Math.pow(adaptState.iter, -adaptState.gamma);
          const beta = eta * (alphaAcc - adaptState.target);
          if (zn2 > 1e-300 && beta !== 0) {
            const inv = 1 / Math.sqrt(zn2);                   // ẑ = z/‖z‖ ⇒ S ẑ = stepv/‖z‖
            for (let d = 0; d < dim; d++) stepv[d] *= inv;
            cholRank1Update(S, stepv, beta, dim);             // keeps old S on downdate PD-loss
          }
          // Accumulate the (possibly updated) position for the covariance anchor.
          const cur = ensemble[w], sum = adaptState.sum, cross = adaptState.cross;
          for (let i = 0; i < dim; i++) {
            sum[i] += cur[i];
            for (let j = 0; j <= i; j++) cross[i * dim + j] += cur[i] * cur[j];
          }
          adaptState.count++;
        }
      }

      if (warm) {
        adaptState.sweeps++;
        // Re-anchor S to the empirical covariance (× optimal RW factor) every 25
        // sweeps once enough samples accrue — this injects the posterior's
        // per-coordinate scale/anisotropy the rank-1 update adapts too slowly to
        // discover; the rank-1 refinement then continues from the anchor.
        const n = adaptState.count;
        if (n > dim + 2 && adaptState.sweeps % 25 === 0) {
          const sum = adaptState.sum, cross = adaptState.cross, eps = 1e-9;
          const variance = new Float64Array(dim);
          for (let d = 0; d < dim; d++) {
            const md = sum[d] / n;
            const v = cross[d * dim + d] / n - md * md;
            variance[d] = v > eps ? v : eps;
          }
          const useFullCov = n > 20 * dim;   // enough samples for a stable correlation estimate
          const M = new Float64Array(dim * dim);
          for (let i = 0; i < dim; i++) {
            M[i * dim + i] = variance[i];
            if (useFullCov) {
              const mi = sum[i] / n;
              for (let j = 0; j < i; j++) {
                const c = (cross[i * dim + j] / n - mi * (sum[j] / n)) * 0.9;   // shrink off-diagonals
                M[i * dim + j] = c; M[j * dim + i] = c;
              }
            }
          }
          const Lnew = new Float64Array(dim * dim);
          if (denseCholesky(M, Lnew, dim)) {
            for (let t = 0; t < Lnew.length; t++) Lnew[t] *= kappa;
            adaptState.L = Lnew;
          } else {
            const Ld = new Float64Array(dim * dim);
            for (let d = 0; d < dim; d++) Ld[d * dim + d] = kappa * Math.sqrt(variance[d]);
            adaptState.L = Ld;
          }
        }
      }
      return { accepts, proposals };
    },
  };
}

module.exports = { cholRank1Update, makeRamKernel };
