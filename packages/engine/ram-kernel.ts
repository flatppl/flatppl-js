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
// Proposes y' = y + scale·S·z (z ~ N(0,I)) and, during warmup, adapts the
// Cholesky factor S toward a target acceptance rate via a rank-1 update of
// C = S Sᵀ:
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
//
// The rank-1 term alone corrects OVERALL scale only very slowly: it nudges one
// random direction at a time, so an isotropic scale error needs O(dim) "lucky"
// directions to average out, and every re-anchor discards whatever it had
// achieved and reinstates the SAME raw kappa·√Σ̂ magnitude (kappa = 2.38/√dim is
// the Gaussian-asymptotic optimum, routinely wrong-sized for a non-Gaussian /
// funnel posterior). Bug (sampler-quality regression): with no other
// mechanism, this left RAM's frozen sample-phase proposal systematically
// mis-scaled (observed as an acceptance rate persistently BELOW target on e.g.
// eight-schools, unlike the mh kernel), causing poor mixing / high split-R̂. Fix,
// mirroring the mh kernel's own (already-robust) design: `scale` is now a
// genuinely-adapted multiplier, Robbins-Monro toward the target acceptance rate
// every warmup step — a FAST, direction-independent correction layered on top of
// the slower rank-1 shape refinement (the standard two-timescale recipe: shape
// adapts the correlation structure, a decoupled scalar adapts the overall step
// size — Andrieu & Thoms 2008 §5). The freeze-then-parallel machinery already
// threads {L, scale} through unchanged; only the warmup step (which used to
// hardcode scale≡1) and `init` (which now seeds scale, not S, with the initial
// magnitude) change.
// Re-anchor cadence (sweeps). Kept as a named constant because the warmup-end
// guard below needs to reserve a matching settle window.
const ANCHOR_PERIOD = 25;

function makeRamKernel(opts?: any) {
  const gamma = (opts && typeof opts.gamma === 'number') ? opts.gamma : 2 / 3;
  return {
    init(_nWalkers: number, dim: number, _o: any) {
      const target = dim === 1 ? 0.44 : 0.234;
      const S = new Float64Array(dim * dim);
      const s0 = 2.38 / Math.sqrt(Math.max(dim, 1));   // RW-MH optimal scale as the initial S
      for (let d = 0; d < dim; d++) S[d * dim + d] = s0;
      // Total warmup length, so the re-anchor can leave itself a settle window
      // (see the `sweeps` guard below) instead of firing on the final sweep.
      const warmupTotal = (_o && typeof _o.warmup === 'number') ? _o.warmup : 1000;
      return {
        // `scale` starts at 1 (S already carries the kappa·I initial magnitude)
        // and is Robbins-Monro-adapted toward `target` every warmup step — see
        // the design note above.
        dim, scale: 1, target, gamma, L: S, iter: 0, warmupTotal,
        z: new Float64Array(dim), step: new Float64Array(dim),
        Sbak: new Float64Array(dim * dim),   // scratch: pre-update S, restored on downdate PD-loss
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
        const sc = adaptState.scale;
        for (let i = 0; i < dim; i++) {
          let acc = 0;
          for (let k = 0; k <= i; k++) acc += S[i * dim + k] * z[k];
          stepv[i] = acc;              // S·z (the proposal increment, pre-scale)
          yProp[i] = y[i] + sc * acc;
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
          // Fast, direction-independent scalar correction (Robbins-Monro toward
          // `target`) — same recipe as the mh kernel's `scale`, decoupled from
          // the slower rank-1 shape update below. Without this, the periodic
          // covariance re-anchor's fixed kappa magnitude has no way to correct
          // for a non-Gaussian target within a short warmup (see the file-top
          // note) and the frozen sample-phase proposal ends up systematically
          // mis-scaled.
          const g = (accepted ? 1 : 0) - adaptState.target;
          adaptState.scale *= Math.exp(g / Math.sqrt(adaptState.iter + 1));
          const beta = eta * (alphaAcc - adaptState.target);
          if (zn2 > 1e-300 && beta !== 0) {
            const inv = 1 / Math.sqrt(zn2);                   // ẑ = z/‖z‖ ⇒ S ẑ = stepv/‖z‖
            for (let d = 0; d < dim; d++) stepv[d] *= inv;
            // cholRank1Update mutates S IN PLACE as it walks k=0..dim-1, so a
            // downdate that loses positive-definiteness partway through leaves
            // S neither the old factor nor a valid new one — snapshot first and
            // restore on failure (the doc comment's "keeps old S" contract,
            // which the call alone does not honour: the return value must be
            // checked).
            adaptState.Sbak.set(S);
            if (!cholRank1Update(S, stepv, beta, dim)) S.set(adaptState.Sbak);
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
        //
        // BUG (sampler-quality regression): for any warmup length that is a
        // multiple of ANCHOR_PERIOD (1000, 1500, 500, … — the common defaults),
        // `sweeps % ANCHOR_PERIOD === 0` is true on the LAST warmup sweep too, so
        // the anchor's raw kappa·√Σ̂ estimate — a Gaussian-asymptotic scale that is
        // frequently mis-calibrated for a non-Gaussian/funnel posterior — got
        // frozen for the whole sample phase with ZERO rank-1 refinement
        // afterward, instead of being "refined" as the design intends. Guard: a
        // re-anchor may only fire if at least one more ANCHOR_PERIOD of warmup
        // remains, so every anchor gets its full refinement window before either
        // the next anchor or the freeze.
        const n = adaptState.count;
        const settleLeft = adaptState.warmupTotal - adaptState.sweeps;
        if (n > dim + 2 && adaptState.sweeps % ANCHOR_PERIOD === 0 && settleLeft >= ANCHOR_PERIOD) {
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
