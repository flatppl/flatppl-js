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
//   - 'cluster': per-cluster local ellipsoids (MultiNest/nestle multi-
//     ellipsoid recipe). A single GLOBAL covariance (whitened) can't bound a
//     curved/ridged constrained region — but the prior transform (e.g. a
//     Normal quantile) is nonlinear, so even a linear ridge in θ-space shows
//     up CURVED in cube u-space. Recursively 2-means-split the live set
//     (in the parent cluster's own whitened frame, so the split direction
//     reflects local shape, not raw coordinate scale) and keep the split
//     only if it shrinks the enclosed volume (sum of child ellipsoid
//     volumes < parent's) — the same accept-a-split test MultiNest/nestle
//     use. Each leaf gets its own bounding ellipsoid (mean + covariance
//     Cholesky, scaled so all its points sit inside); sampling picks a LIVE
//     POINT uniformly (not an ellipsoid via a volume-weighted threshold —
//     see the note above euclidDist2 for why) and draws uniformly inside
//     whichever leaf owns it, then applies a volume-and-size-weighted
//     "1/overlaps(u)" union correction (buildClusterRegion's own note has
//     the derivation). On a curved ridge this yields several tight local
//     ellipsoids instead of one huge global one — and building/sampling
//     costs come from a handful of leaf ellipsoids, not the
//     whitened/identity metrics' O(bootstrap·K²) build or O(K) per-proposal
//     overlap scan.
//
// OPT-IN, NOT the default (measured 2026-07-21, 3 seeds each vs region-free):
// 'cluster' is genuinely FASTER (rasch 1.21×, eight-schools 1.19×, partial-
// pooling 1.07×) with higher acceptance efficiency (1.13–1.28×) — the only
// region variant that beats region-free on speed (whitened/identity both lost
// outright). BUT it degrades the evidence: logZ variance blows up (rasch sd
// 0.024→0.369, ~15×) and the mean drifts inconsistently (eight-schools +0.21,
// partial-pooling −0.60). The overlap correction is unbiased GIVEN the region;
// the region itself under-covers the constrained prior (ellipsoids too tight /
// split too aggressive → misses mass → noisy shrinkage). So region-free stays
// the default; use 'cluster' for faster POSTERIOR-SAMPLE exploration where you
// don't need precision on logZ, NOT for evidence/Bayes-factor work. Making the
// coverage conservative (larger enlarge, stricter split) to restore logZ
// accuracy is tracked as a follow-up.
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
// A threshold/categorical draw picked straight off a single prng() call, at a
// FIXED stride through the stream (e.g. one call per sample() attempt when
// the accept probability doesn't vary), is measurably biased with the simple
// LCG this engine uses for deterministic tests and mat-density.ts's seeded
// runs — fixed-stride subsequences of an LCG are themselves low-quality LCGs
// (a classic defect), and a controlled 2-ellipsoid case showed the raw LCG's
// pick frequencies and resulting sample mean diverge from a grid-integral
// ground truth by several times their own Monte Carlo noise, while
// Math.random() reproduces that ground truth exactly. The cluster region
// below therefore never makes a weighted/threshold draw the FIRST decision
// in an attempt: it picks a LIVE POINT uniformly via `Math.floor(prng()*K)`
// over K≈hundreds of equally-weighted options — the same style the ball
// metrics already use safely — and only threshold-compares a smoothly
// varying accept probability (not a small fixed set of skewed cutoffs).

// Bounding ellipsoid for one cluster's points: mean + covariance Cholesky
// (via the shared mean/cholCov helpers), scaled so the whitened distance of
// the FARTHEST point is exactly 1 (a tiny relative pad avoids float-boundary
// flakiness at that point), times opts.enlarge for extra safety margin.
// logVol is log(volume) up to an additive dim-only constant (the unit
// d-ball's volume factor) that cancels whenever two ellipsoids of the same
// dim are compared — parent-vs-children (the split test) and leaf-vs-leaf
// (the volume-proportional sampling weight) both only ever compare same-dim
// ellipsoids, so the missing constant is never a problem.
function boundingEllipsoid(pts: Float64Array[], dim: number, enlarge: number, idx: number[]) {
  const m = mean(pts, dim);
  const L = cholCov(pts, dim, m);
  let maxD2 = 0;
  for (const p of pts) { const d2 = whitenedDist2(L, dim, p, m); if (d2 > maxD2) maxD2 = d2; }
  const scale = Math.sqrt(Math.max(maxD2, 1e-300)) * (1 + 1e-6) * enlarge;
  const Ls = new Float64Array(dim * dim);
  for (let k = 0; k < dim * dim; k++) Ls[k] = L[k] * scale;
  let logVol = 0; for (let i = 0; i < dim; i++) logVol += Math.log(Math.max(Ls[i * dim + i], 1e-300));
  return { mean: m, L: Ls, logVol, size: pts.length, idx };
}
// 2-means on displacement vectors (already centered + whitened by the
// parent's own covariance, so Euclidean distance here IS Mahalanobis
// distance in the parent's frame). Returns a 0/1 assignment per point, or
// null on a degenerate split (an empty side, or too few points to seed two
// distinct centroids) — the caller falls back to keeping the parent whole.
function kmeans2Assign(disp: Float64Array[], dim: number, prng: () => number, iters: number): Int8Array | null {
  const n = disp.length;
  if (n < 2) return null;
  const i0 = Math.floor(prng() * n);
  let i1 = Math.floor(prng() * n), guard = 0;
  while (i1 === i0 && guard++ < 30) i1 = Math.floor(prng() * n);
  if (i1 === i0) return null;
  const c0 = disp[i0].slice(), c1 = disp[i1].slice();
  const assign = new Int8Array(n);
  for (let it = 0; it < iters; it++) {
    let changed = false;
    for (let i = 0; i < n; i++) {
      const p = disp[i]; let d0 = 0, d1 = 0;
      for (let c = 0; c < dim; c++) { const e0 = p[c] - c0[c], e1 = p[c] - c1[c]; d0 += e0 * e0; d1 += e1 * e1; }
      const a = d0 <= d1 ? 0 : 1;
      if (assign[i] !== a) changed = true;
      assign[i] = a;
    }
    const sum0 = new Float64Array(dim), sum1 = new Float64Array(dim);
    let n0 = 0, n1 = 0;
    for (let i = 0; i < n; i++) {
      const p = disp[i];
      if (assign[i] === 0) { n0++; for (let c = 0; c < dim; c++) sum0[c] += p[c]; }
      else { n1++; for (let c = 0; c < dim; c++) sum1[c] += p[c]; }
    }
    if (n0 === 0 || n1 === 0) return null;
    for (let c = 0; c < dim; c++) { c0[c] = sum0[c] / n0; c1[c] = sum1[c] / n1; }
    if (!changed && it > 0) break;
  }
  return assign;
}
// Recursive 2-means split (MultiNest/nestle multi-ellipsoid recipe): fit a
// bounding ellipsoid to `pts`; if there are enough points to try a split,
// 2-means in the parent's whitened frame, fit child ellipsoids, and recurse
// into the children ONLY if doing so shrinks the total enclosed volume
// (sum of child volumes < parent's) — the same "does splitting help" test
// nestle uses. Depth-capped as a belt-and-suspenders guard (minClusterPts
// already bounds the recursion in practice). `idx` carries each point's
// ORIGINAL index into the live-point array through the recursion (points get
// partitioned into different leaves, but the caller needs to map any live
// point back to the leaf that ended up owning it). Returns the leaf
// ellipsoids, each tagged with the `idx` of the live points it owns.
function buildClusters(pts: Float64Array[], idx: number[], dim: number, prng: () => number, opts: any, depth: number): any[] {
  const minPts = opts.minClusterPts || Math.max(dim + 1, 2);
  const maxDepth = opts.maxClusterDepth || 6;
  const kIters = opts.kmeansIters || 10;
  const enlarge = opts.enlarge || 1.0;
  const parent = boundingEllipsoid(pts, dim, enlarge, idx);
  if (depth >= maxDepth || pts.length < 2 * minPts) return [parent];
  const m = mean(pts, dim);
  const L0 = cholCov(pts, dim, m);
  const disp: Float64Array[] = pts.map((p) => {
    const d = new Float64Array(dim); for (let i = 0; i < dim; i++) d[i] = p[i] - m[i];
    return whiten(L0, dim, d);
  });
  const assign = kmeans2Assign(disp, dim, prng, kIters);
  if (!assign) return [parent];
  const A: Float64Array[] = [], B: Float64Array[] = [], idxA: number[] = [], idxB: number[] = [];
  for (let i = 0; i < pts.length; i++) { if (assign[i] === 0) { A.push(pts[i]); idxA.push(idx[i]); } else { B.push(pts[i]); idxB.push(idx[i]); } }
  if (A.length < minPts || B.length < minPts) return [parent];
  const eA = boundingEllipsoid(A, dim, enlarge, idxA), eB = boundingEllipsoid(B, dim, enlarge, idxB);
  const mx = Math.max(eA.logVol, eB.logVol);
  const logVolChildren = mx + Math.log(Math.exp(eA.logVol - mx) + Math.exp(eB.logVol - mx));
  if (logVolChildren >= parent.logVol) return [parent];      // split doesn't help — keep parent whole
  return buildClusters(A, idxA, dim, prng, opts, depth + 1).concat(buildClusters(B, idxB, dim, prng, opts, depth + 1));
}
// Union-of-ellipsoids region: same rejection-sampling shape as the ball
// metrics (draw → cube-check → overlap-accept), but over the leaf ellipsoids
// from buildClusters instead of one ball per live point. UNLIKE a plain
// "pick ellipsoid ∝ volume via a threshold compare" (which measurably biases
// under this engine's simple deterministic-test LCG — see the note above
// mean()/cholCov()), sampling here picks a LIVE POINT uniformly (the same
// safe `Math.floor(prng()*K)` the ball metrics use) and uses whichever leaf
// owns it — i.e. picks leaf i with probability size_i/K, not Vol_i/ΣVol.
// That changes the accept-correction: with weight q_i = size_i/Vol_i per
// leaf, the proposal density at x is Σ_i (size_i/K)(1/Vol_i)1[x∈Ei] =
// (1/K)·Σ_{i:x∈Ei} q_i, so accepting x with probability
// min_j(q_j) / Σ_{i:x∈Ei} q_i (computed in log-space as logQ = log(size)−logVol,
// a plain "1/overlaps(u)" ball-style count) makes the surviving density
// constant over the union ∩ cube — the same rejection-sampling argument the
// ball metrics use, just with q_i in place of the ball metrics' equal
// per-point weight.
function buildClusterRegion(liveU: Float64Array[], dim: number, prng: () => number, opts: any) {
  // Cluster-specific enlarge default: 1.0 (exact-cover, the generic
  // ball-metric default) measurably UNDER-covers on a hard/ridged posterior
  // (rasch-1pl, 3 seeds, nLive 400, dlogz 0.5: logZ off by 0.63-1.10, i.e.
  // 7-12x the run's own logZerr — a real, reproducible bias, not noise).
  // Each leaf ellipsoid is fit from only its own (often small, post-split)
  // point subset, so an exact-cover ellipsoid systematically misses the true
  // constrained-prior mass just outside those specific points. 1.2 cuts the
  // bias substantially (though not perfectly — still opt-in, not the
  // default region; see mat-density.ts's nested branch / the cluster-region
  // measurement report for the full analysis).
  const o = opts.enlarge != null ? opts : Object.assign({ enlarge: 1.2 }, opts);
  const K = liveU.length;
  const idx0 = new Array(K); for (let i = 0; i < K; i++) idx0[i] = i;
  const leaves = buildClusters(liveU, idx0, dim, prng, o, 0);
  const nLeaves = leaves.length;
  const pointToLeaf = new Int32Array(K);
  for (let li = 0; li < nLeaves; li++) for (const oi of leaves[li].idx) pointToLeaf[oi] = li;
  const logQ = new Float64Array(nLeaves);
  let logQmin = Infinity;
  for (let i = 0; i < nLeaves; i++) { logQ[i] = Math.log(leaves[i].size) - leaves[i].logVol; if (logQ[i] < logQmin) logQmin = logQ[i]; }
  // Weighted overlap sum in log-space: logsumexp of logQ[i] over leaves whose
  // ellipsoid contains u, or -Infinity if none do.
  function overlapLogQ(u: Float64Array): number {
    let mx = -Infinity; const hit: number[] = [];
    for (let i = 0; i < nLeaves; i++) if (whitenedDist2(leaves[i].L, dim, u, leaves[i].mean) <= 1) { hit.push(logQ[i]); if (logQ[i] > mx) mx = logQ[i]; }
    if (hit.length === 0) return -Infinity;
    let s = 0; for (const lq of hit) s += Math.exp(lq - mx); return mx + Math.log(s);
  }
  function contains(u: Float64Array): boolean { return Number.isFinite(overlapLogQ(u)); }
  function sample(): Float64Array | null {
    const e = leaves[pointToLeaf[Math.floor(prng() * K)]];   // pick a live point uniformly, use its leaf
    const z = new Float64Array(dim); let nrm = 0;
    for (let i = 0; i < dim; i++) {
      const u1 = Math.max(prng(), 1e-12), u2 = prng();
      z[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      nrm += z[i] * z[i];
    }
    nrm = Math.sqrt(nrm);
    const rr = Math.pow(prng(), 1 / dim);                       // uniform radius in the UNIT ball (e.L already covers to radius 1)
    for (let i = 0; i < dim; i++) z[i] = (z[i] / (nrm || 1)) * rr;
    const u = unwhiten(e.L, dim, z, e.mean);
    for (let i = 0; i < dim; i++) if (u[i] < 0 || u[i] > 1) return null;    // outside the cube
    const logS = overlapLogQ(u);                                // u came from e, so this is always finite
    if (prng() > Math.exp(logQmin - logS)) return null;         // weighted overlap correction
    return u;
  }
  return { sample, contains, radius: NaN, nCenters: nLeaves, clusterSizes: leaves.map((e: any) => e.size) };
}

function buildRegion(liveU: Float64Array[], prng: () => number, opts: any = {}) {
  const K = liveU.length, dim = liveU[0].length;
  if (opts.metric === 'cluster') return buildClusterRegion(liveU, dim, prng, opts);
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
