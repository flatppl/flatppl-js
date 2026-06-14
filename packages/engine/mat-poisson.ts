'use strict';

// =====================================================================
// mat-poisson.ts — PoissonProcess core (spec §08; engine-concepts §2.3)
// =====================================================================
//
// The mathematically load-bearing core of `PoissonProcess(intensity)`,
// decoupled from worker / classifier orchestration so it is verifiable
// in isolation. The full `matPoissonProcess` / `walkPoissonProcess`
// handlers wrap the worker draws + the consume/rest density contract
// around these.
//
// Spec §08: variates are arrays of points (permutation-invariant). For
// the canonical construction `intensity = weighted(M, shape)` (M = the
// expected count = `totalmass(intensity)`, `shape = normalize(intensity)`
// a normalized distribution):
//
//   SAMPLING (per atom i):  k_i ~ Poisson(M_i);  k_i points iid from shape.
//   DENSITY  (w.r.t. iid(Lebesgue, k), per atom i):
//       logp_i = Σ_j log λ(t_ij) − M_i
//              = k_i·log(M_i) + Σ_j shape_logpdf(t_ij) − M_i
//   because λ(t) = intensity(t) = M·shape_pdf(t).
//
// The per-atom output has VARIABLE length k_i → the ragged value kind
// (ragged.ts): VectorOfVectors for scalar points.

const R = require('./ragged.ts');

// Assemble the ragged process value from per-atom counts + the flat pool
// of all atoms' points (drawn iid from the shape, atom-major). `counts`
// length N; `pointsFlat` length Σ counts. Scalar points (kernelShape=[]).
function assemblePoissonRagged(counts: ArrayLike<number>,
                              pointsFlat: Float64Array): any {
  const N = counts.length;
  const offsets = new Int32Array(N + 1);
  let tot = 0;
  for (let i = 0; i < N; i++) { tot += counts[i] | 0; offsets[i + 1] = tot; }
  if (pointsFlat.length !== tot) {
    throw new Error('assemblePoissonRagged: pointsFlat length ' + pointsFlat.length
      + ' ≠ Σcounts ' + tot);
  }
  return R.ragged(pointsFlat, offsets, []);
}

// Per-atom log-density of a ragged PoissonProcess observation under
// `intensity = weighted(M, shape)`. Returns a uniform `[N]` Value
// (`{shape:[N], data}`) — one log-density per atom (the consume/rest
// walker sums these into the trace logp).
//   - `raggedObs`: a ragged value (scalar points).
//   - `M`: per-atom expected count — a scalar or a length-N array.
//   - `shapeLogpdf`: t → log shape_pdf(t) (the NORMALIZED shape density).
function poissonProcessLogDensity(raggedObs: any, M: number | ArrayLike<number>,
                                  shapeLogpdf: (t: number) => number): any {
  const N = R.raggedCount(raggedObs);
  const out = new Float64Array(N);
  const data = raggedObs.data, offsets = raggedObs.offsets;
  for (let i = 0; i < N; i++) {
    const Mi = (typeof M === 'number') ? M : M[i];
    const lo = offsets[i], hi = offsets[i + 1];
    const k = hi - lo;                       // scalar points: span = count
    let s = (k > 0 ? k * Math.log(Mi) : 0) - Mi;
    for (let p = lo; p < hi; p++) s += shapeLogpdf(data[p]);
    out[i] = s;
  }
  return { shape: [N], data: out };
}

module.exports = {
  assemblePoissonRagged,
  poissonProcessLogDensity,
};
