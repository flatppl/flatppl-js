// packages/engine/emcee-kernel.ts
'use strict';
// Affine-invariant ensemble sampler — the emcee stretch move
// (Goodman & Weare 2010; Foreman-Mackey et al. 2013). Gradient-free,
// self-adapting to the target's scale and linear correlations; no warmup
// adaptation. Walkers interact, so this is a genuine ensemble kernel: the
// ensemble is split in two halves and each half is updated against the
// (frozen) complementary half.
function makeEmceeKernel(a?: number) {
  const Adefault = a ?? 2;
  return {
    init(nWalkers: number, dim: number, opts: any) {
      if (nWalkers < 4 || nWalkers % 2 !== 0) {
        throw new Error(`emcee: nWalkers must be even and >= 4 (got ${nWalkers}); use >= 2*dim+2 for dim=${dim}`);
      }
      return { a: opts.a ?? Adefault };
    },
    step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, adaptState: any, _phase: string) {
      const dim = mv.dim, n = ensemble.length, half = n / 2;
      const a = adaptState.a;
      let accepts = 0, proposals = 0;
      // s=0 updates the first half against the second; s=1 the reverse. The
      // complementary set is frozen while its partner half is updated.
      for (let s = 0; s < 2; s++) {
        const lo = s === 0 ? 0 : half, hi = s === 0 ? half : n;
        const clo = s === 0 ? half : 0, chi = s === 0 ? n : half;
        for (let k = lo; k < hi; k++) {
          // z ~ g(z) ∝ 1/sqrt(z) on [1/a, a]:  z = ((a-1)*u + 1)^2 / a
          const z = Math.pow((a - 1) * prng() + 1, 2) / a;
          const j = clo + Math.floor(prng() * (chi - clo));   // random complementary walker
          const Xk = ensemble[k], Xj = ensemble[j];
          const Y = new Float64Array(dim);
          for (let d = 0; d < dim; d++) Y[d] = Xj[d] + z * (Xk[d] - Xj[d]);
          const lpY = mv.logPosterior(Y);
          proposals++;
          // accept with prob min(1, z^(dim-1) * π(Y)/π(Xk))
          const logAccept = (dim - 1) * Math.log(z) + lpY - logp[k];
          if (Math.log(prng() + 1e-300) < logAccept) {
            ensemble[k] = Y; logp[k] = lpY; accepts++;
          }
        }
      }
      return { accepts, proposals };
    },
  };
}

module.exports = { makeEmceeKernel };
