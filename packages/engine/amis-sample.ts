'use strict';
// amis-sample.ts — Efficient Adaptive Multiple Importance Sampling (EAMIS).
//
// El-Laham, Martino, Elvira & Bugallo, "Efficient Adaptive Multiple Importance
// Sampling", EUSIPCO 2019 (Table II). A gradient-free adaptive importance
// sampler: a single Gaussian proposal q_t = N(μ_t, Σ_t) is adapted over T
// iterations by weighted moment-matching (as in AMIS, Table I), and the
// importance weights use the AMIS *temporal deterministic mixture* in the
// denominator. EAMIS's contribution is to APPROXIMATE that mixture once the
// proposal has stopped moving: after an automatically-detected iteration K
// (when ‖μ_t − μ_{t-1}‖₂ < ε), proposals K…t are collapsed into one component
// θ_{ℓ*} (ℓ* = max(τ, K)) carrying weight (t−K+1), so each sample needs only K
// proposal evaluations instead of t — O(MKT) vs AMIS's O(MT²).
//
// Target: π(x) = ℓ(y|x)h(x) ∝ posterior (paper eq. 1). Here logπ(x) is
// mv.logPosterior(y) over the unconstrained vector (prior + likelihood +
// change-of-variables Jacobian) — the same scorer the MCMC backends use.
//
// Produces WEIGHTED samples (self-normalised importance weights), so the
// posterior is reported with logWeights (like the IS path) and the viewer's
// ESS / PSIS-k̂ readout applies.

const rng = require('./rng.ts');
const sampler = require('./sampler.ts');

function gaussianNoise(prng: () => number): number {
  const u1 = Math.max(prng(), 1e-300), u2 = prng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Lower-triangular Cholesky of a symmetric PD matrix (flat row-major). Returns
// false if not PD (caller falls back to a diagonal sqrt).
function cholesky(A: Float64Array, L: Float64Array, dim: number): boolean {
  for (let i = 0; i < dim; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i * dim + j];
      for (let k = 0; k < j; k++) s -= L[i * dim + k] * L[j * dim + k];
      if (i === j) { if (s <= 0) return false; L[i * dim + j] = Math.sqrt(s); }
      else L[i * dim + j] = s / L[j * dim + j];
    }
    for (let j = i + 1; j < dim; j++) L[i * dim + j] = 0;
  }
  return true;
}

const LN_2PI = Math.log(2 * Math.PI);

// A Gaussian proposal: mean μ, lower-Cholesky L of Σ, and log|Σ| = 2 Σ log Lᵢᵢ.
function makeProposal(mu: Float64Array, Sigma: Float64Array, dim: number) {
  const L = new Float64Array(dim * dim);
  if (!cholesky(Sigma, L, dim)) {
    // Non-PD: diagonal fallback from the (floored) diagonal of Σ.
    for (let i = 0; i < dim * dim; i++) L[i] = 0;
    for (let d = 0; d < dim; d++) {
      const v = Sigma[d * dim + d];
      L[d * dim + d] = Math.sqrt(v > 1e-12 ? v : 1e-12);
    }
  }
  let logdet = 0;
  for (let d = 0; d < dim; d++) logdet += 2 * Math.log(L[d * dim + d]);
  return { mu: Float64Array.from(mu), L, logdet };
}

// log N(x; μ, Σ) using the proposal's Cholesky: solve L u = (x−μ), q = uᵀu.
function mvnLogpdf(x: Float64Array, p: any, dim: number): number {
  const u = new Float64Array(dim);
  for (let i = 0; i < dim; i++) {
    let s = x[i] - p.mu[i];
    for (let k = 0; k < i; k++) s -= p.L[i * dim + k] * u[k];
    u[i] = s / p.L[i * dim + i];
  }
  let q = 0;
  for (let i = 0; i < dim; i++) q += u[i] * u[i];
  return -0.5 * (dim * LN_2PI + p.logdet + q);
}

function logsumexp(a: number[]): number {
  let mx = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
  if (mx === -Infinity) return -Infinity;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - mx);
  return mx + Math.log(s);
}

// EAMIS. Returns { samples (unconstrained y vectors), logW (final IS weights),
// K, M, T } — the caller constrains + reshapes into the posterior measure.
function amisSample(mv: any, opts: any) {
  const dim = mv.dim;
  const M = opts.amisSamples ?? Math.max(50, Math.min(500, Math.round((opts.draws ?? 1000) / 4)));
  const T = opts.amisIters ?? 20;
  const seed = opts.seed ?? 0;
  const epsK = opts.amisEpsK ?? 0.005;      // auto-K threshold on ‖Δμ‖₂ (paper §IV-D)
  const ridge = 1e-6;
  // Per-coordinate proposal-variance floor, as a fraction of the initial
  // (prior-scale) variance. A single Gaussian proposal that overshoots toward a
  // tight mode can collapse: low ESS → weighted moment cov ≈ a single point →
  // proposal narrows → still low ESS → degenerate spiral. Flooring each variance
  // at floorFrac·Σ₁ keeps the proposal broad enough to keep covering the target
  // (so ESS recovers) while still allowing it to tighten ~10× toward the mode.
  const floorFrac = opts.amisFloorFrac ?? 0.01;
  const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;

  const baseKey = rng.keyFromSeed(seed);
  const prng = sampler.makePhiloxPrngAdapter(rng.stateFromKey(baseKey[0], baseKey[1]));

  // Initial proposal from the prior-draw pool: μ₁ = mean, Σ₁ = cov (+ridge),
  // inflated for over-dispersed coverage. Falls back to N(0, I) if no pool.
  let mu = new Float64Array(dim);
  let Sigma = new Float64Array(dim * dim);
  const initN = Math.max(dim + 2, 64);
  const initPts: Float64Array[] = (typeof mv.initFromPrior === 'function')
    ? mv.initFromPrior(initN, prng) : [];
  if (initPts.length >= 2) {
    for (const y of initPts) for (let d = 0; d < dim; d++) mu[d] += y[d];
    for (let d = 0; d < dim; d++) mu[d] /= initPts.length;
    for (const y of initPts) {
      for (let i = 0; i < dim; i++) for (let j = 0; j <= i; j++) {
        const c = (y[i] - mu[i]) * (y[j] - mu[j]);
        Sigma[i * dim + j] += c; Sigma[j * dim + i] += c;
      }
    }
    const inflate = 4 / initPts.length;   // 2× std over the empirical prior cov
    for (let i = 0; i < dim * dim; i++) Sigma[i] *= inflate;
  } else {
    for (let d = 0; d < dim; d++) Sigma[d * dim + d] = 1;
  }
  // Degenerate pool (e.g. all draws identical → zero variance, as when the
  // prior was materialised with a tiny sample count): fall back to a broad
  // isotropic proposal rather than a delta that can never cover the target.
  let totVar = 0;
  for (let d = 0; d < dim; d++) totVar += Sigma[d * dim + d];
  if (!(totVar > 1e-8)) {
    for (let i = 0; i < dim * dim; i++) Sigma[i] = 0;
    for (let d = 0; d < dim; d++) Sigma[d * dim + d] = 1;
  }
  for (let d = 0; d < dim; d++) Sigma[d * dim + d] += ridge;
  // Variance floor reference (fraction of the initial per-coordinate variance).
  const floorVar = new Float64Array(dim);
  for (let d = 0; d < dim; d++) floorVar[d] = floorFrac * Sigma[d * dim + d];

  const proposals: any[] = [];          // θ_1 … θ_t
  const X: Float64Array[] = [];         // all samples
  const logTarget: number[] = [];       // logπ(x) per sample
  const tau: number[] = [];             // generating iteration (1-based) per sample
  // logQ[i][j] = log q(X_i; θ_j). A sample's density under a proposal never
  // changes, so cache it: each entry is computed ONCE. Recomputing it every
  // iteration (as the naive temporal-mixture loop does) is O(M·T³) Gaussian
  // evals and dominated the run; caching makes it O(M·T²) computed-once.
  const logQ: number[][] = [];
  let K: number | null = null;          // frozen when adaptation stalls
  let prevMu: Float64Array | null = null;
  let finalLogW = new Float64Array(0);

  for (let t = 1; t <= T; t++) {
    const p = makeProposal(mu, Sigma, dim);
    proposals.push(p);
    const pIdx = t - 1;                  // 0-based index of this proposal
    // New proposal's column for all EXISTING samples.
    for (let i = 0; i < X.length; i++) logQ[i].push(mvnLogpdf(X[i], p, dim));

    // a. Draw M samples from the current proposal, then score them. AMIS's M
    // samples are independent, so score the whole iteration's batch in ONE pass
    // (mv.logPosteriorBatch → one batched likelihood eval) — the reason AMIS
    // suits the engine's atom-batched evaluator far better than sequential MCMC.
    const batch: Float64Array[] = new Array(M);
    for (let m = 0; m < M; m++) {
      const z = new Float64Array(dim);
      for (let d = 0; d < dim; d++) z[d] = gaussianNoise(prng);
      const x = new Float64Array(dim);
      for (let i = 0; i < dim; i++) {
        let acc = 0;
        for (let k = 0; k <= i; k++) acc += p.L[i * dim + k] * z[k];
        x[i] = p.mu[i] + acc;
      }
      batch[m] = x;
    }
    const lt = (typeof mv.logPosteriorBatch === 'function')
      ? mv.logPosteriorBatch(batch)
      : batch.map((x: Float64Array) => mv.logPosterior(x));
    for (let m = 0; m < M; m++) {
      const x = batch[m];
      X.push(x); tau.push(t); logTarget.push(lt[m]);
      // This new sample's density under every proposal so far (0..pIdx).
      const row: number[] = new Array(pIdx + 1);
      for (let j = 0; j <= pIdx; j++) row[j] = mvnLogpdf(x, proposals[j], dim);
      logQ.push(row);
    }

    // Auto-K (§IV-D): once the proposal mean stops moving, freeze K = t−1 and
    // switch to the reduced (efficient) mixture for subsequent re-weightings.
    if (K === null && prevMu) {
      let d2 = 0;
      for (let d = 0; d < dim; d++) { const e = mu[d] - prevMu[d]; d2 += e * e; }
      if (Math.sqrt(d2) < epsK) K = t - 1 > 0 ? t - 1 : 1;
    }

    // b. Re-weight ALL samples under the temporal mixture (full for t≤K, the
    //    EAMIS reduced mixture for t>K).
    const logW = new Float64Array(X.length);
    const useApprox = (K !== null && t > K);
    const logT = Math.log(t);
    for (let i = 0; i < X.length; i++) {
      const lq = logQ[i];                          // cached log q(X_i; θ_j), j=0..t-1
      let logmix: number;
      if (!useApprox) {
        logmix = logsumexp(lq) - logT;             // (1/t) Σ_{j=1}^t q_j
      } else {
        const lstar = Math.max(tau[i], K as number);
        const comps: number[] = [];
        for (let j = 0; j < (K as number) - 1; j++) comps.push(lq[j]);
        comps.push(Math.log(t - (K as number) + 1) + lq[lstar - 1]);
        logmix = logsumexp(comps) - logT;          // (1/t)(Σ_{j<K} q_j + (t−K+1) q_{ℓ*})
      }
      logW[i] = logTarget[i] - logmix;
    }
    finalLogW = logW;

    // c+d. Normalise weights and update the proposal by weighted moment matching.
    const arr: number[] = Array.prototype.slice.call(logW);
    const lse = logsumexp(arr);
    if (!Number.isFinite(lse)) break;              // all weights zero → stop
    const wbar = new Float64Array(X.length);
    for (let i = 0; i < X.length; i++) wbar[i] = Math.exp(logW[i] - lse);

    const newMu = new Float64Array(dim);
    for (let i = 0; i < X.length; i++) for (let d = 0; d < dim; d++) newMu[d] += wbar[i] * X[i][d];
    const newSigma = new Float64Array(dim * dim);
    for (let i = 0; i < X.length; i++) {
      const w = wbar[i], x = X[i];
      for (let a = 0; a < dim; a++) {
        const da = x[a] - newMu[a];
        for (let b = 0; b <= a; b++) {
          const c = w * da * (x[b] - newMu[b]);
          newSigma[a * dim + b] += c; newSigma[b * dim + a] += c;
        }
      }
    }
    for (let d = 0; d < dim; d++) newSigma[d * dim + d] += ridge;
    // Floor each variance so the proposal cannot collapse to a delta. If a
    // diagonal is lifted to the floor, shrink that coordinate's off-diagonals
    // proportionally so the correlation stays consistent (and Σ stays PD).
    for (let d = 0; d < dim; d++) {
      const v = newSigma[d * dim + d];
      if (v < floorVar[d] && floorVar[d] > 0) {
        const s = Math.sqrt(floorVar[d] / (v > 0 ? v : ridge));
        for (let j = 0; j < dim; j++) { newSigma[d * dim + j] *= s; newSigma[j * dim + d] *= s; }
        newSigma[d * dim + d] = floorVar[d];
      }
    }
    prevMu = mu;
    mu = newMu;
    Sigma = newSigma;
    if (onProgress) onProgress(t / T, 'amis');
  }

  return { samples: X, logW: finalLogW, K: K ?? T, M, T };
}

module.exports = { amisSample };
