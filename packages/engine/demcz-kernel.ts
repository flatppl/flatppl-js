// packages/engine/demcz-kernel.ts
'use strict';
// DEMCz — differential-evolution ensemble MCMC with a past-states archive
// (ter Braak & Vrugt 2008). Gradient-free, auto-adapting to the target's
// covariance and linear correlations via difference vectors drawn from an
// archive Z of past walker states. The proposal is symmetric, so acceptance
// is the plain Metropolis ratio — no Jacobian term (unlike emcee's stretch
// move). γ = 2.38/√(2·dim) normally; γ = 1 every 10th generation to enable
// jumps between well-separated modes.
const driver = require('./mcmc-driver.ts');
const { gaussianNoise } = driver;

function makeDemczKernel(opts?: any) {
  // Tuning knobs come from ctx.inferenceOpts (captured here), NOT from the
  // driver's init opts — the driver does not forward gamma/b/thin.
  const cfg = opts || {};
  return {
    init(nWalkers: number, dim: number) {
      if (nWalkers < 4) {
        throw new Error(`demcz: nWalkers must be >= 4 (got ${nWalkers})`);
      }
      return {
        Z: [] as Float64Array[],       // past-states archive; seeded on first step
        gen: 0,                        // generation counter (drives γ=1 + thinning)
        gamma: cfg.gamma ?? (2.38 / Math.sqrt(2 * dim)),
        b: cfg.b ?? 1e-6,              // jitter sd (ergodicity)
        K: cfg.thin ?? 10,            // archive-append interval
      };
    },
    step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, _phase: string) {
      const dim = mv.dim, n = ensemble.length;
      const Z = adaptState.Z;
      // Seed the archive from the initial (dispersed) ensemble on generation 0.
      if (Z.length === 0) for (let w = 0; w < n; w++) Z.push(Float64Array.from(ensemble[w]));
      const M = Z.length;
      const gamma = (adaptState.gen % 10 === 9) ? 1.0 : adaptState.gamma;
      const b = adaptState.b;
      const batch = typeof mv.logPosteriorBatch === 'function';

      // Every walker is proposed against the FROZEN archive snapshot Z (it reads
      // Z, never another walker's in-progress update), so the proposals are
      // mutually independent and can be scored in ONE batched pass. All random
      // draws (r1, r2, jitter, accept-uniform) are consumed in a fixed order so
      // a fixed seed yields bit-identical draws.
      const Ys: Float64Array[] = new Array(n);
      const us = new Float64Array(n);
      for (let k = 0; k < n; k++) {
        const r1 = Math.floor(prng() * M);
        let r2 = Math.floor(prng() * M);
        if (r2 === r1) r2 = (r2 + 1) % M;        // two distinct archive indices (M >= n >= 4)
        const Xk = ensemble[k], Zr1 = Z[r1], Zr2 = Z[r2];
        const Y = new Float64Array(dim);
        for (let d = 0; d < dim; d++) Y[d] = Xk[d] + gamma * (Zr1[d] - Zr2[d]) + b * gaussianNoise(prng);
        Ys[k] = Y;
        us[k] = prng();
      }
      const lps = batch ? mv.logPosteriorBatch(Ys) : Ys.map((Y) => mv.logPosterior(Y));
      let accepts = 0;
      for (let k = 0; k < n; k++) {
        const logAccept = lps[k] - logp[k];      // symmetric proposal ⇒ no Jacobian
        if (Math.log(us[k] + 1e-300) < logAccept) {
          ensemble[k] = Ys[k]; logp[k] = lps[k]; accepts++;
        }
      }
      // Grow the archive (thinned by K) with the post-update ensemble.
      if (adaptState.gen % adaptState.K === 0) {
        for (let w = 0; w < n; w++) Z.push(Float64Array.from(ensemble[w]));
      }
      adaptState.gen++;
      return { accepts, proposals: n };
    },
  };
}

module.exports = { makeDemczKernel };
