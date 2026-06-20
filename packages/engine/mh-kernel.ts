// packages/engine/mh-kernel.ts
'use strict';
const { gaussianNoise } = require('./mcmc-driver.ts');

// Independent-walker random-walk Metropolis (Metropolis-within-Gibbs per
// coordinate). Walkers do not interact. A single per-coordinate step scale is
// adapted toward the target accept rate during warmup (Robbins-Monro).
const mhKernel = {
  init(nWalkers: number, dim: number, opts: any) {
    const initStep = opts.initStep ?? 1;
    return { step: new Float64Array(dim).fill(initStep), target: dim === 1 ? 0.44 : 0.234, iter: 0 };
  },
  step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, phase: string) {
    const dim = mv.dim, nWalkers = ensemble.length;
    let accepts = 0, proposals = 0;
    for (let w = 0; w < nWalkers; w++) {
      let y = ensemble[w], lp = logp[w];
      for (let d = 0; d < dim; d++) {
        const yProp = Float64Array.from(y);
        yProp[d] = y[d] + adaptState.step[d] * gaussianNoise(prng);
        const lpProp = mv.logPosterior(yProp);
        proposals++;
        const accepted = Math.log(prng() + 1e-300) < (lpProp - lp);
        if (accepted) { y = yProp; lp = lpProp; accepts++; }
        if (phase === 'warmup') {
          const g = (accepted ? 1 : 0) - adaptState.target;
          adaptState.step[d] *= Math.exp(g / Math.sqrt(adaptState.iter + 1));
        }
      }
      ensemble[w] = y; logp[w] = lp;
    }
    if (phase === 'warmup') adaptState.iter++;
    return { accepts, proposals };
  },
};

module.exports = { mhKernel };
