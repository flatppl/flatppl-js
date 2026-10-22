'use strict';
// amis-sample.ts — Efficient Adaptive Multiple Importance Sampling (EAMIS).
//
// El-Laham, Martino, Elvira & Bugallo, "Efficient Adaptive Multiple Importance
// Sampling", EUSIPCO 2019 (Table II). A gradient-free adaptive importance
// sampler: a single Gaussian proposal q_t = N(μ_t, Σ_t) is adapted over T
// iterations by weighted moment-matching, and the
// importance weights use the AMIS *temporal deterministic mixture* in the
// denominator. EAMIS's contribution is to APPROXIMATE that mixture once the
// proposal has stopped moving: after an automatically-detected iteration K
// (when successive Gaussian KL divergence is below ε), proposals K…t are collapsed into one component
// θ_{ℓ*} (ℓ* = max(τ, K)) carrying weight (t−K+1), so each sample needs only K
// proposal evaluations instead of t — O(MKT) vs AMIS's O(MT²) Gaussian scores.
// Reweighting and covariance fitting still revisit all accumulated samples.
// Proposal fitting first follows prior·likelihood^β, with an ESS-controlled β
// rising from zero to one. This avoids a sharp likelihood selecting one prior
// point and collapsing the next covariance. Without a split scorer, the initial
// Gaussian supplies the reference for q_1^(1−β)·π^β. If q_t misses the current
// bridge target, a local bridge fits that target before β advances. This extension
// never tempers the returned posterior weights or changes their mixture denominator.
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
  return proposal(Float64Array.from(mu), L, logdet);
}

// Maglev OSR miscompile, nodejs/node#66457, fixed in V8 15.2.
function proposal(mu: Float64Array<ArrayBuffer>, L: Float64Array<ArrayBuffer>, logdet: number) {
  return { mu, L, logdet };
}

// log N(x; μ, Σ) using the proposal's Cholesky: solve L u = (x−μ), q = uᵀu.
function mvnLogpdf(x: Float64Array, p: any, dim: number, u: Float64Array): number {
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

/** KL(current || previous), including covariance changes in the stopping rule.
 * A stationary mean alone cannot establish that a narrowing proposal is stable. */
function proposalKL(p: any, previous: any, dim: number): number {
  const u = new Float64Array(dim);
  let trace = 0, distance = 0;
  for (let column = 0; column <= dim; column++) {
    for (let i = 0; i < dim; i++) {
      let s = column === dim ? p.mu[i] - previous.mu[i] : p.L[i * dim + column];
      for (let j = 0; j < i; j++) s -= previous.L[i * dim + j] * u[j];
      u[i] = s / previous.L[i * dim + i];
      if (column === dim) distance += u[i] * u[i];
      else trace += u[i] * u[i];
    }
  }
  return Math.max(0, 0.5 * (trace + distance - dim + previous.logdet - p.logdet));
}

// Maglev OSR miscompile, nodejs/node#66457, fixed in V8 15.2.
function bridgeFit(weights: Float64Array<ArrayBuffer>, ess: number, beta: number) {
  return { weights, ess, beta };
}

/** Fit a reachable intermediate target using fresh proposal draws. The ESS
 * bound limits weight concentration; these weights are never returned as samples. */
function bridgeWeights(logBase: number[], logTarget: ArrayLike<number>, logProposal: number[], beta: number, minEss: number) {
  const weightsAt = (b: number) => {
    const lw = logBase.map((base, i) => {
      if (b === 0) return base - logProposal[i];
      if (b === 1) return logTarget[i] - logProposal[i];
      return (1 - b) * base + b * logTarget[i] - logProposal[i];
    });
    const lse = logsumexp(lw);
    const weights = Float64Array.from(lw, w => Number.isFinite(lse) ? Math.exp(w - lse) : 0);
    let sumSq = 0;
    for (const w of weights) sumSq += w * w;
    return bridgeFit(weights, sumSq > 0 ? 1 / sumSq : 0, b);
  };
  const full = weightsAt(1);
  if (full.ess >= minEss) return full;
  let fit = weightsAt(beta);
  // Refit this rung when the Gaussian misses it. Do not assume ESS is monotone
  // away from a well-represented endpoint, or advance using a degenerate fit.
  if (fit.ess < minEss) return fit;
  let hi = 1;
  for (let i = 0; i < 40; i++) {
    const mid = (fit.beta + hi) / 2, candidate = weightsAt(mid);
    if (candidate.ess >= minEss) fit = candidate;
    else hi = mid;
  }
  return fit;
}

// EAMIS. Returns { samples (unconstrained y vectors), logW (final IS weights),
// K, M, T } — the caller constrains + reshapes into the posterior measure.
function amisSample(mv: any, opts: any) {
  const dim = mv.dim;
  const solveScratch = new Float64Array(dim);
  const M = opts.amisSamples ?? Math.max(50, Math.min(500, Math.round((opts.draws ?? 1000) / 4)));
  const T = opts.amisIters ?? 20;
  const seed = opts.seed ?? 0;
  const epsK = opts.amisEpsK ?? 0.005;      // auto-K Gaussian KL threshold (§IV-D)
  const ridge = 1e-6;
  // Numerical diagonal regularization, not an exploration floor: a fixed 1%
  // of prior variance can dwarf every posterior direction in an informative model.
  const floorFrac = opts.amisFloorFrac ?? 1e-12;
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
        Sigma[i * dim + j] += c; if (j !== i) Sigma[j * dim + i] += c;
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
  // Cache each proposal score once. After K freezes, keep only proposals
  // 1…K−1 and the sample's representative max(tau[i], K) in slot K−1.
  const logQ: number[][] = [];
  let K: number | null = null;          // frozen when adaptation stalls
  let finalLogW = new Float64Array(0);
  let beta = 0;

  for (let t = 1; t <= T; t++) {
    const p = makeProposal(mu, Sigma, dim);
    proposals.push(p);
    const pIdx = t - 1;                  // 0-based index of this proposal
    // A frozen mixture never reads new proposal columns for existing samples.
    if (K === null) {
      for (let i = 0; i < X.length; i++) logQ[i].push(mvnLogpdf(X[i], p, dim, solveScratch));
    }

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
    const split = typeof mv.logPriorLikBatch === 'function' ? mv.logPriorLikBatch(batch) : null;
    const lt = split ? Float64Array.from(split.prior, (x: number, i: number) => {
      const lp = x + split.lik[i];
      return Number.isFinite(lp) ? lp : -Infinity;
    }) : (typeof mv.logPosteriorBatch === 'function')
      ? mv.logPosteriorBatch(batch)
      : batch.map((x: Float64Array) => mv.logPosterior(x));
    // A first proposal (drawn at prior scale) whose every sample scores −∞ means
    // the target has no mass the prior can reach. The weights then normalise to
    // whichever single sample the max-subtraction picks, the proposal collapses
    // onto it, and AMIS returns that one point with ESS 0 — indistinguishable in
    // the viewer from a very sharp posterior. Refuse instead, matching the MCMC
    // driver's, SMC's and nested's refusals on the same condition.
    if (t === 1) {
      let anyFinite = false;
      for (let m = 0; m < M; m++) if (Number.isFinite(lt[m])) { anyFinite = true; break; }
      if (!anyFinite) {
        throw new Error(
          `backend 'amis': log-posterior is not finite at ANY of the ${M} samples of the `
          + 'initial prior-scale proposal, so no importance weight carries mass. Check that '
          + 'the likelihood scores finitely at a prior draw.');
      }
    }
    for (let m = 0; m < M; m++) {
      const x = batch[m];
      X.push(x); tau.push(t); logTarget.push(lt[m]);
      const row: number[] = new Array(K ?? (pIdx + 1));
      if (K === null) {
        for (let j = 0; j <= pIdx; j++) row[j] = mvnLogpdf(x, proposals[j], dim, solveScratch);
      } else {
        for (let j = 0; j < K - 1; j++) row[j] = mvnLogpdf(x, proposals[j], dim, solveScratch);
        row[K - 1] = mvnLogpdf(x, p, dim, solveScratch);
      }
      logQ.push(row);
    }

    // Freeze only after fitting the full target and checking both Gaussian
    // moments. Bridge stages can keep the mean fixed while changing the scale.
    if (K === null && pIdx > 0 && beta === 1) {
      if (proposalKL(p, proposals[pIdx - 1], dim) < epsK) {
        K = t - 1 > 0 ? t - 1 : 1;
        for (let i = 0; i < logQ.length; i++) {
          const row = logQ[i];
          row[K - 1] = row[Math.max(tau[i], K) - 1];
          row.length = K;
        }
      }
    }

    // b. Re-weight ALL samples under the temporal mixture (full for t≤K, the
    //    EAMIS reduced mixture for t>K).
    const logW = new Float64Array(X.length);
    const useApprox = (K !== null && t > K);
    const logT = Math.log(t);
    for (let i = 0; i < X.length; i++) {
      const lq = logQ[i];
      let logmix: number;
      if (!useApprox) {
        logmix = logsumexp(lq) - logT;             // (1/t) Σ_{j=1}^t q_j
      } else {
        const comps: number[] = [];
        for (let j = 0; j < (K as number) - 1; j++) comps.push(lq[j]);
        comps.push(Math.log(t - (K as number) + 1) + lq[(K as number) - 1]);
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

    const minEss = Math.min(M, Math.max(dim + 2, M / 10));
    let sumSq = 0;
    for (const w of wbar) sumSq += w * w;
    // Hard support can make the first positive temperature discontinuous.
    // Accumulated posterior hits can still support a full-target covariance.
    if (1 / sumSq >= minEss) beta = 1;
    let fitWeights = wbar, fitX = X;
    if (beta < 1) {
      const lq = batch.map(x => mvnLogpdf(x, p, dim, solveScratch));
      const lq0: number[] = split ? Array.from(split.prior)
        : batch.map(x => mvnLogpdf(x, proposals[0], dim, solveScratch));
      let fit = bridgeWeights(lq0, lt, lq, beta, minEss);
      if (fit.ess < minEss) {
        // A Gaussian may miss even the prior, especially with heavy tails.
        // Fit toward this rung from q_t without advancing the outer bridge.
        const logRung = lq0.map((v, i) => beta === 0 ? v : (1 - beta) * v + beta * lt[i]);
        fit = { ...bridgeWeights(lq, logRung, lq, 0, minEss), beta };
      }
      if (fit.ess <= dim) {
        if (onProgress) onProgress(t / T, 'amis');
        continue;
      }
      beta = fit.beta;
      fitX = batch;
      fitWeights = fit.weights;
    }
    const newMu = new Float64Array(dim);
    for (let i = 0; i < fitX.length; i++) for (let d = 0; d < dim; d++) newMu[d] += fitWeights[i] * fitX[i][d];
    const newSigma = new Float64Array(dim * dim);
    for (let i = 0; i < fitX.length; i++) {
      const w = fitWeights[i], x = fitX[i];
      for (let a = 0; a < dim; a++) {
        const da = x[a] - newMu[a];
        for (let b = 0; b <= a; b++) {
          const c = w * da * (x[b] - newMu[b]);
          newSigma[a * dim + b] += c; if (b !== a) newSigma[b * dim + a] += c;
        }
      }
    }
    // Sigma >= diag(floorVar) in PSD order; the same regularized proposal
    // supplies both draws and their Gaussian importance denominator.
    for (let d = 0; d < dim; d++) newSigma[d * dim + d] += floorVar[d];
    mu = newMu;
    // A new tail hit can dominate historical weights even after the bridge.
    // Retain the last covariance until enough points support another fit.
    if (fitX !== X || 1 / sumSq >= minEss) Sigma = newSigma;
    if (onProgress) onProgress(t / T, 'amis');
  }

  return { samples: X, logW: finalLogW, K: K ?? T, M, T };
}

module.exports = { amisSample };
