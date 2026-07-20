// packages/engine/mlfriends.ts
'use strict';
// MLFriends region (Buchner 2016/2019) on the unit cube: the union of balls of
// radius r (in a whitened, live-covariance metric) around the live points. Used
// by nested sampling to propose likelihood-constrained draws by rejection —
// far more efficient than a region-free slice once the live set concentrates.
function mean(live: Float64Array[], dim: number): Float64Array {
  const m = new Float64Array(dim);
  for (const u of live) for (let c = 0; c < dim; c++) m[c] += u[c];
  for (let c = 0; c < dim; c++) m[c] /= live.length;
  return m;
}
// Lower-Cholesky of the live covariance (+ ridge). Returns L (dim×dim, row-major).
function cholCov(live: Float64Array[], dim: number, m: Float64Array): Float64Array {
  const C = new Float64Array(dim * dim);
  for (const u of live) for (let i = 0; i < dim; i++) { const di = u[i] - m[i]; for (let j = 0; j < dim; j++) C[i * dim + j] += di * (u[j] - m[j]); }
  const n = Math.max(1, live.length - 1);
  for (let k = 0; k < dim * dim; k++) C[k] /= n;
  for (let i = 0; i < dim; i++) C[i * dim + i] += 1e-9;         // ridge for singular C
  const L = new Float64Array(dim * dim);
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j <= i; j++) {
      let s = C[i * dim + j];
      for (let k = 0; k < j; k++) s -= L[i * dim + k] * L[j * dim + k];
      if (i === j) L[i * dim + j] = Math.sqrt(Math.max(s, 1e-300));
      else L[i * dim + j] = s / L[j * dim + j];
    }
  }
  return L;
}
// Solve L y = x (lower-triangular forward substitution) → whitened coords.
function whiten(L: Float64Array, dim: number, x: Float64Array): Float64Array {
  const y = new Float64Array(dim);
  for (let i = 0; i < dim; i++) { let s = x[i]; for (let k = 0; k < i; k++) s -= L[i * dim + k] * y[k]; y[i] = s / L[i * dim + i]; }
  return y;
}
// Apply L: u = center + L z  (un-whiten a ball-space displacement z).
function unwhiten(L: Float64Array, dim: number, z: Float64Array, center: Float64Array): Float64Array {
  const u = new Float64Array(dim);
  for (let i = 0; i < dim; i++) { let s = center[i]; for (let k = 0; k <= i; k++) s += L[i * dim + k] * z[k]; u[i] = s; }
  return u;
}
function whitenedDist2(L: Float64Array, dim: number, a: Float64Array, b: Float64Array): number {
  const d = new Float64Array(dim); for (let i = 0; i < dim; i++) d[i] = a[i] - b[i];
  const z = whiten(L, dim, d); let s = 0; for (let i = 0; i < dim; i++) s += z[i] * z[i]; return s;
}

function buildRegion(liveU: Float64Array[], prng: () => number, opts: any = {}) {
  const K = liveU.length, dim = liveU[0].length;
  const B = opts.bootstrap || 50, enlarge = opts.enlarge || 1.0;
  const m = mean(liveU, dim);
  const L = cholCov(liveU, dim, m);
  // Leave-one-out bootstrap radius: max over resamples of each original point's
  // nearest-neighbour whitened distance to a DISTINCT resampled point.
  let r2 = 0;
  for (let b = 0; b < B; b++) {
    const idx = new Int32Array(K);
    for (let i = 0; i < K; i++) idx[i] = Math.floor(prng() * K);
    for (let i = 0; i < K; i++) {
      let best = Infinity;
      for (let j = 0; j < K; j++) { if (idx[j] === i) continue; const d2 = whitenedDist2(L, dim, liveU[i], liveU[idx[j]]); if (d2 < best) best = d2; }
      if (Number.isFinite(best) && best > r2) r2 = best;
    }
  }
  const radius = Math.sqrt(r2) * enlarge;
  // Count how many live-point balls contain u (whitened distance ≤ radius).
  function overlaps(u: Float64Array): number { let n = 0; const r2b = radius * radius; for (let c = 0; c < K; c++) if (whitenedDist2(L, dim, u, liveU[c]) <= r2b) n++; return n; }
  function contains(u: Float64Array): boolean { return overlaps(u) > 0; }
  // Sample uniformly from the union: pick a random center, sample a point in its
  // ball (uniform in the whitened ball → un-whiten), reject outside the cube, and
  // accept with prob 1/overlaps(u) (overlap correction → uniform over the union).
  function sample(): Float64Array | null {
    const center = liveU[Math.floor(prng() * K)];
    // uniform point in the unit d-ball (Muller method), scaled by radius
    const z = new Float64Array(dim); let nrm = 0;
    for (let i = 0; i < dim; i++) { // standard normal via Box–Muller pairs
      const u1 = Math.max(prng(), 1e-12), u2 = prng();
      z[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      nrm += z[i] * z[i];
    }
    nrm = Math.sqrt(nrm);
    const rr = radius * Math.pow(prng(), 1 / dim);              // uniform radius in the ball
    for (let i = 0; i < dim; i++) z[i] = (z[i] / (nrm || 1)) * rr;
    const u = unwhiten(L, dim, z, center);
    for (let i = 0; i < dim; i++) if (u[i] < 0 || u[i] > 1) return null;   // outside the cube
    const n = overlaps(u);
    if (n <= 0) return null;
    if (prng() > 1 / n) return null;                           // overlap correction
    return u;
  }
  return { sample, contains, radius, nCenters: K };
}
module.exports = { buildRegion };
