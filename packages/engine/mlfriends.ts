// packages/engine/mlfriends.ts
'use strict';
// Region-based constrained-draw proposals on the unit cube, sampled
// uniformly over the union of per-live-point balls via rejection + an
// overlap correction (1/overlaps(u)) so the union — not any single ball —
// is uniform. `opts.metric` selects the ball geometry:
//   - 'whitened' (default): MLFriends (Buchner 2016/2019) — radius in a
//     GLOBAL whitened, live-covariance metric. On a well-behaved posterior
//     the covariance shapes the balls to the mode; on a ridge/degenerate
//     posterior a single global covariance elongates every ball along the
//     ridge, making them huge (low likelihood-test acceptance) and the
//     Cholesky + whitened-distance overlap check expensive.
//   - 'identity': RadFriends (Buchner 2014) — radius in the RAW cube metric,
//     no Cholesky/covariance at all. All cube coords already live on
//     [0,1], so there is no scale disparity to whiten away; the bootstrap
//     radius is the plain Euclidean leave-one-out nearest-neighbour
//     distance. On a ridge, live points are dense ALONG it, so the k-NN
//     radius is small and the balls hug the ridge tightly instead of
//     ballooning across it.
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
// Raw squared Euclidean distance — the 'identity' metric (RadFriends): the
// cube coords are already unit-scale so no whitening is needed, and this
// also skips the mean/Cholesky computation entirely.
function euclidDist2(dim: number, a: Float64Array, b: Float64Array): number {
  let s = 0; for (let i = 0; i < dim; i++) { const d = a[i] - b[i]; s += d * d; } return s;
}

function buildRegion(liveU: Float64Array[], prng: () => number, opts: any = {}) {
  const K = liveU.length, dim = liveU[0].length;
  const B = opts.bootstrap || 50, enlarge = opts.enlarge || 1.0;
  const identity = opts.metric === 'identity';
  // dist2(a,b): squared distance in the region's metric. unwhitenFn(z,center):
  // maps a ball-space displacement z back to cube coords. 'identity' skips
  // the mean/Cholesky (L = identity, so whiten/unwhiten are no-ops); the
  // default 'whitened' metric keeps the MLFriends global live-covariance
  // shape.
  let dist2: (a: Float64Array, b: Float64Array) => number;
  let unwhitenFn: (z: Float64Array, center: Float64Array) => Float64Array;
  if (identity) {
    dist2 = (a, b) => euclidDist2(dim, a, b);
    unwhitenFn = (z, center) => { const u = new Float64Array(dim); for (let i = 0; i < dim; i++) u[i] = center[i] + z[i]; return u; };
  } else {
    const m = mean(liveU, dim);
    const L = cholCov(liveU, dim, m);
    dist2 = (a, b) => whitenedDist2(L, dim, a, b);
    unwhitenFn = (z, center) => unwhiten(L, dim, z, center);
  }
  // Leave-one-out bootstrap radius: max over resamples of each original point's
  // nearest-neighbour distance (in the region's metric) to a DISTINCT resampled point.
  let r2 = 0;
  for (let b = 0; b < B; b++) {
    const idx = new Int32Array(K);
    for (let i = 0; i < K; i++) idx[i] = Math.floor(prng() * K);
    for (let i = 0; i < K; i++) {
      let best = Infinity;
      for (let j = 0; j < K; j++) { if (idx[j] === i) continue; const d2 = dist2(liveU[i], liveU[idx[j]]); if (d2 < best) best = d2; }
      if (Number.isFinite(best) && best > r2) r2 = best;
    }
  }
  const radius = Math.sqrt(r2) * enlarge;
  // Count how many live-point balls contain u (distance ≤ radius, in the region's metric).
  function overlaps(u: Float64Array): number { let n = 0; const r2b = radius * radius; for (let c = 0; c < K; c++) if (dist2(u, liveU[c]) <= r2b) n++; return n; }
  function contains(u: Float64Array): boolean { return overlaps(u) > 0; }
  // Sample uniformly from the union: pick a random center, sample a point in its
  // ball (uniform in the ball → map back to cube coords), reject outside the
  // cube, and accept with prob 1/overlaps(u) (overlap correction → uniform
  // over the union). This correction is metric-agnostic — RadFriends needs
  // it exactly as much as MLFriends does.
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
    const u = unwhitenFn(z, center);
    for (let i = 0; i < dim; i++) if (u[i] < 0 || u[i] > 1) return null;   // outside the cube
    const n = overlaps(u);
    if (n <= 0) return null;
    if (prng() > 1 / n) return null;                           // overlap correction
    return u;
  }
  return { sample, contains, radius, nCenters: K };
}
module.exports = { buildRegion };
