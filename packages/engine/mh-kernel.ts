// packages/engine/mh-kernel.ts
'use strict';
const { gaussianNoise } = require('./mcmc-driver.ts');

// Independent-walker random-walk Metropolis with a WHOLE-VECTOR (block)
// proposal: all coordinates are perturbed at once and scored with a SINGLE
// logπ per walker per sweep — not one logπ per coordinate (Metropolis-within-
// Gibbs), which cost dim× density evaluations. Walkers do not interact. A
// single scalar step scale is adapted toward the target accept rate during
// warmup (Robbins–Monro); per-coordinate scales pre-scale each dimension so a
// badly-scaled posterior still moves.
const mhKernel = {
  init(_nWalkers: number, dim: number, opts: any) {
    return {
      scale: (opts.initStep ?? (2.38 / Math.sqrt(Math.max(dim, 1)))), // RW-MH optimal × scalar accept-adapt
      perCoord: new Float64Array(dim).fill(1), // per-dimension scale, adapted from warmup variance
      target: 0.234,                           // optimal multi-dim RW-MH accept
      iter: 0,
      // Running per-coordinate mean/var over warmup positions (diagonal
      // adaptive Metropolis): perCoord[d] ← sd[d] so each coordinate moves on
      // its own scale — essential when coordinates differ in magnitude (e.g.
      // eight-schools tau vs theta) where a single shared step mixes terribly.
      sum: new Float64Array(dim),
      sumsq: new Float64Array(dim),
      count: 0,
    };
  },
  step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, phase: string) {
    const dim = mv.dim, nWalkers = ensemble.length;
    const scale = adaptState.scale, perCoord = adaptState.perCoord;
    const warm = phase === 'warmup';
    let accepts = 0, proposals = 0;
    for (let w = 0; w < nWalkers; w++) {
      const y = ensemble[w];
      const yProp = Float64Array.from(y);
      for (let d = 0; d < dim; d++) yProp[d] = y[d] + scale * perCoord[d] * gaussianNoise(prng);
      const lpProp = mv.logPosterior(yProp);    // ONE logπ for the whole vector
      proposals++;
      const accepted = Math.log(prng() + 1e-300) < (lpProp - logp[w]);
      if (accepted) { ensemble[w] = yProp; logp[w] = lpProp; accepts++; }
      if (warm) {
        const g = (accepted ? 1 : 0) - adaptState.target;
        adaptState.scale *= Math.exp(g / Math.sqrt(adaptState.iter + 1));
        const cur = ensemble[w];
        for (let d = 0; d < dim; d++) { adaptState.sum[d] += cur[d]; adaptState.sumsq[d] += cur[d] * cur[d]; }
        adaptState.count++;
      }
    }
    if (warm) {
      adaptState.iter++;
      // Refresh per-coordinate scales from the accumulated warmup variance
      // every 50 sweeps once enough samples have accrued.
      if (adaptState.count > 2 * nWalkers && adaptState.iter % 50 === 0) {
        const n = adaptState.count;
        for (let d = 0; d < dim; d++) {
          const m = adaptState.sum[d] / n;
          const v = adaptState.sumsq[d] / n - m * m;
          perCoord[d] = v > 1e-12 ? Math.sqrt(v) : perCoord[d];
        }
      }
    }
    return { accepts, proposals };
  },
};

module.exports = { mhKernel };
