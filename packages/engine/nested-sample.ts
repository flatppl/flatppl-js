'use strict';
const { buildRegion } = require('./mlfriends.ts');
// Static nested sampling (Skilling 2006). Live points live in the unit cube;
// the prior transform T maps them to θ. Each iteration removes the lowest-
// likelihood live point, accumulates it into the evidence with the shrinking
// prior mass X_i ≈ exp(−i/K), and replaces it with a likelihood-constrained
// draw obtained by a slice step on the cube coordinates. Posterior samples are
// the dead points weighted by w_i = L_i (X_{i−1} − X_i).
function logaddexp(a: number, b: number): number {
  if (a === -Infinity) return b; if (b === -Infinity) return a;
  const m = Math.max(a, b); return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

// One likelihood-constrained slice draw on the cube (Neal 2003 stepping-out +
// shrinkage per coordinate), constraint logLik(T(u)) > Lstar. Sweeps over all
// coordinates `nSweeps` times so the replacement point decorrelates from its
// seed instead of moving in just one coordinate (a single sweep is a weak MCMC
// move on real, non-separable likelihoods). On a coordinate whose shrink loop
// exhausts its iterations without finding an accepting candidate, that
// coordinate is left unchanged — the current point always satisfies the
// constraint, so it is always safe to keep it. Returns the new cube point +
// its record + logLik + the number of logLik(transform(...)) evaluations
// actually performed (for accounting the true sampling efficiency).
function constrainedSlice(u0: Float64Array, dim: number, transform: any, logLik: any, Lstar: number, prng: () => number, nSweeps: number = 5) {
  const u = u0.slice();
  const w = 0.3;                                   // initial slice width on [0,1]
  let nEval = 0;
  for (let sweep = 0; sweep < nSweeps; sweep++) {
    for (let c = 0; c < dim; c++) {
      let lo = Math.max(0, u[c] - w * prng());
      let hi = Math.min(1, lo + w);
      // stepping-out is bounded by the cube; shrink until the constraint holds.
      for (let it = 0; it < 60; it++) {
        const cand = lo + prng() * (hi - lo);
        const uu = u.slice(); uu[c] = cand;
        const rec = transform(uu);
        nEval++;
        if (logLik(rec) > Lstar) { u[c] = cand; break; }
        if (cand < u[c]) lo = cand; else hi = cand;    // shrink toward the current point
      }
    }
  }
  const rec = transform(u);
  nEval++;
  return { u, rec, logl: logLik(rec), nEval };
}

function runNested(transform: any, dim: number, logLik: any, opts: any = {}) {
  const K = opts.nLive || 400;
  if (K < 2) throw new Error(`runNested requires nLive >= 2 (got ${K}): the constrained-draw seed picks a live point other than the one being replaced, which is impossible with a single live point`);
  const dlogz = opts.dlogz != null ? opts.dlogz : 0.5;
  const prng = opts.prng || Math.random;
  const maxIter = opts.maxIter || 100000;
  const sliceSweeps = opts.sliceSweeps != null ? opts.sliceSweeps : 5;
  const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;
  // Progress is gauged by the remaining-evidence gap closing toward the dlogz
  // stop: gap = logZremain − logZ starts large and shrinks to log(dlogz) at
  // termination. There is no known iteration total, so this fraction (from the
  // first finite gap) is the honest progress signal. Reported throttled.
  const logStop = Math.log(dlogz);
  let gap0: number | null = null;

  // Initial live points: uniform cube → transform → logLik.
  const liveU: Float64Array[] = [], liveRec: any[] = [], liveL: number[] = [];
  for (let i = 0; i < K; i++) {
    const u = new Float64Array(dim); for (let c = 0; c < dim; c++) u[c] = prng();
    const rec = transform(u); liveU.push(u); liveRec.push(rec); liveL.push(logLik(rec));
  }

  const deadRec: any[] = [], deadLogW: number[] = [], deadL: number[] = [];
  let logZ = -Infinity, logZsq = -Infinity;      // logZsq tracks Σ w_i² for the error
  let logX = 0;                                  // log prior mass, X_0 = 1
  let nIter = 0, nEval = K;
  const logdV = Math.log(1 - Math.exp(-1 / K));  // E[log(X_{i-1}-X_i)] per step ≈ log(X_{i-1}(1-e^{-1/K}))

  // MLFriends region: proposes constrained draws by rejection (far cheaper
  // than a region-free slice once the live set concentrates around the
  // mode). Rebuilt every `rebuild` iterations so its radius tracks the
  // shrinking constrained prior; a stale region under-covers and biases the
  // evidence. Falls back to the slice step when region rejection stalls
  // (high-d / degenerate regions), so termination is never blocked on it.
  let region: any = null, sinceRebuild = 0;
  const rebuild = opts.rebuild || Math.max(1, (K / 10) | 0);
  // Region is OPT-IN, not the default: on rasch-1pl's non-identifiable ridge
  // a single global whitened metric can't bound the constrained region, so
  // MLFriends rebuilds thrash and cost 4.4x a plain region-free slice with no
  // acceptance gain. Region-free static NS is correct and not slower on hard
  // posteriors; pass useRegion:true for easy/low-dim posteriors where the
  // region's rejection sampling beats the slice fallback (see the efficiency
  // test in nested-sample.test.ts).
  const useRegion = opts.useRegion === true;
  const regionTries = opts.regionTries || 200;

  for (; nIter < maxIter; nIter++) {
    // lowest-likelihood live point
    let lo = 0; for (let i = 1; i < K; i++) if (liveL[i] < liveL[lo]) lo = i;
    const Lstar = liveL[lo];
    const logWi = Lstar + logX + logdV;          // w_i = L_i · (X_{i-1}-X_i)
    logZ = logaddexp(logZ, logWi);
    logZsq = logaddexp(logZsq, 2 * logWi);
    deadRec.push(liveRec[lo]); deadLogW.push(logWi); deadL.push(Lstar);
    logX += -1 / K;                              // X_i = exp(−i/K); must land BEFORE the
                                                   // termination check and any break, so both
                                                   // it and the post-loop closure see X_i, not
                                                   // the stale X_{i-1} used for logWi above.
    // termination: remaining live evidence fraction
    let maxLive = -Infinity; for (let i = 0; i < K; i++) maxLive = Math.max(maxLive, liveL[i]);
    const logZremain = maxLive + logX;
    if (onProgress && nIter % 40 === 0) {
      const gap = logZremain - logZ;
      if (gap0 === null && Number.isFinite(gap)) gap0 = gap;
      if (gap0 !== null && gap0 > logStop) {
        onProgress(Math.max(0, Math.min(0.99, (gap0 - gap) / (gap0 - logStop))), 'sampling');
      }
    }
    if (logZremain - logZ < Math.log(dlogz)) { nIter++; break; }
    // replace with a likelihood-constrained draw: region rejection first
    // (cheap once the region is tight), slice as the fallback.
    let drawn;
    if (useRegion) {
      if (!region || sinceRebuild >= rebuild) { region = buildRegion(liveU, prng, opts.region || {}); sinceRebuild = 0; }
      sinceRebuild++;
      let ev = 0, found: any = null;
      for (let t = 0; t < regionTries && !found; t++) {
        const u = region.sample();
        if (!u) continue;
        const rec = transform(u); ev++;
        const logl = logLik(rec);
        if (logl > Lstar) found = { u, rec, logl, nEval: ev };
      }
      if (found) {
        drawn = found;
      } else {
        let seed = lo; while (seed === lo && K > 1) seed = Math.floor(prng() * K);
        drawn = constrainedSlice(liveU[seed], dim, transform, logLik, Lstar, prng, sliceSweeps);
        drawn.nEval += ev;                             // account the wasted region tries
      }
    } else {
      let seed = lo; while (seed === lo && K > 1) seed = Math.floor(prng() * K);
      drawn = constrainedSlice(liveU[seed], dim, transform, logLik, Lstar, prng, sliceSweeps);
    }
    nEval += drawn.nEval;
    liveU[lo] = drawn.u; liveRec[lo] = drawn.rec; liveL[lo] = drawn.logl;
  }
  // add the remaining live points, each with mass X_final / K
  const logXk = logX - Math.log(K);
  for (let i = 0; i < K; i++) {
    const logWi = liveL[i] + logXk;
    logZ = logaddexp(logZ, logWi);
    logZsq = logaddexp(logZsq, 2 * logWi);
    deadRec.push(liveRec[i]); deadLogW.push(logWi); deadL.push(liveL[i]);
  }
  // information-based error: err ≈ sqrt(H/K), H from the weights; use the
  // moment estimate logZerr = sqrt(Σ w_i² )/Z as a robust proxy.
  const logZerr = Math.sqrt(Math.max(0, Math.exp(logZsq - 2 * logZ)));
  const logWeights = Float64Array.from(deadLogW);
  if (onProgress) onProgress(1, 'sampling');
  return {
    samples: deadRec, logWeights, logZ, logZerr,
    nLive: K, nIter, efficiency: deadRec.length / Math.max(1, nEval),
  };
}
module.exports = { runNested, logaddexp };
