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
      const batch = typeof mv.logPosteriorBatch === 'function';
      let accepts = 0, proposals = 0;
      // s=0 updates the first half against the second; s=1 the reverse. The
      // complementary set is frozen while its partner half is updated.
      for (let s = 0; s < 2; s++) {
        const lo = s === 0 ? 0 : half, hi = s === 0 ? half : n;
        const clo = s === 0 ? half : 0, chi = s === 0 ? n : half;
        // Propose every walker in this half against the FROZEN complementary
        // half, drawing z/j and the accept-uniform up front, then score all
        // proposals in ONE logPosteriorBatch pass. Walkers in a half never see
        // each other's updates (each reads its own Xk + a frozen Xj), so the
        // draws are independent — batching is exact. The accept-uniform is drawn
        // before scoring (it does not depend on the score), so the prng stream
        // is consumed in the same order as the scalar path: bit-identical.
        const m = hi - lo;
        const Ys: Float64Array[] = new Array(m);
        const zs = new Float64Array(m), us = new Float64Array(m);
        for (let t = 0; t < m; t++) {
          const k = lo + t;
          // z ~ g(z) ∝ 1/sqrt(z) on [1/a, a]:  z = ((a-1)*u + 1)^2 / a
          const z = Math.pow((a - 1) * prng() + 1, 2) / a;
          const j = clo + Math.floor(prng() * (chi - clo));   // random complementary walker
          const u = prng();
          const Xk = ensemble[k], Xj = ensemble[j];
          const Y = new Float64Array(dim);
          for (let d = 0; d < dim; d++) Y[d] = Xj[d] + z * (Xk[d] - Xj[d]);
          Ys[t] = Y; zs[t] = z; us[t] = u;
        }
        const lps = batch ? mv.logPosteriorBatch(Ys) : Ys.map((Y) => mv.logPosterior(Y));
        for (let t = 0; t < m; t++) {
          const k = lo + t, lpY = lps[t];
          proposals++;
          // accept with prob min(1, z^(dim-1) * π(Y)/π(Xk))
          const logAccept = (dim - 1) * Math.log(zs[t]) + lpY - logp[k];
          if (Math.log(us[t] + 1e-300) < logAccept) {
            ensemble[k] = Ys[t]; logp[k] = lpY; accepts++;
          }
        }
      }
      return { accepts, proposals };
    },
  };
}

module.exports = { makeEmceeKernel };
