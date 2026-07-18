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

// RAM: Robust Adaptive Metropolis (Vihola 2012). Proposes y' = y + S·z
// (z ~ N(0,I)); during warmup adapts the Cholesky factor S toward a target
// acceptance rate via a rank-1 update of C = S Sᵀ:
//   C ← C + η_n (α − α*) · (S ẑ)(S ẑ)ᵀ,   ẑ = z/‖z‖,   η_n = n^{-γ}.
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
      return { dim, scale: 1, target, gamma, L: S, iter: 0, z: new Float64Array(dim), step: new Float64Array(dim) };
    },
    step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, phase: string) {
      const dim = adaptState.dim, nWalkers = ensemble.length;
      const S = adaptState.L, z = adaptState.z, stepv = adaptState.step;
      const warm = phase === 'warmup';
      // Sample phase: fixed proposal, one batched score (bit-identical to scalar).
      if (!warm && typeof mv.logPosteriorBatch === 'function') {
        return driver.fixedProposalBatchStep(ensemble, logp, mv, prng, S, adaptState.scale, dim, z);
      }
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
        }
      }
      return { accepts, proposals };
    },
  };
}

module.exports = { cholRank1Update, makeRamKernel };
