'use strict';
// smc-sample.ts — adaptive-tempered, waste-free Sequential Monte Carlo.
//
// The robust gradient-free workhorse: march a population of particles from the
// prior (β=0) to the posterior (β=1) along a likelihood-tempered path
//   log π_β(x) = prior'(x) + β·lik(x)          (prior' includes the Jacobian)
// adapting the temperature ladder so each rung drops the conditional ESS to a
// target (CESS-bisection), and moving particles with a population-covariance
// Metropolis kernel. Waste-free (Dau & Chopin 2022): resample M≪N ancestors and
// keep ALL M·P chain states. Reuses mv.logPriorLikBatch so re-tempering to a new
// β is arithmetic on the cached `lik` — only fresh move proposals are scored.
//
// El-tempered references: Del Moral–Doucet–Jasra 2006; Dai–Heng–Jacob–Whiteley
// (JASA 2022); Dau–Chopin (JRSS-B 2022). Evidence (log marginal likelihood) is
// accumulated from the per-rung normalising-constant increments — a free bonus.

const rng = require('./rng.ts');
const sampler = require('./sampler.ts');

function gaussianNoise(prng: () => number): number {
  const u1 = Math.max(prng(), 1e-300), u2 = prng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Lower-triangular Cholesky (flat row-major); false if not PD.
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

function logSumExpArr(a: Float64Array): number {
  let mx = -Infinity;
  for (let i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
  if (!Number.isFinite(mx)) return -Infinity;
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.exp(a[i] - mx);
  return mx + Math.log(s);
}

// Systematic resampling of M ancestor indices from unnormalised log-weights.
function systematicResample(logW: Float64Array, M: number, prng: () => number): Int32Array {
  const n = logW.length;
  let mx = -Infinity; for (let i = 0; i < n; i++) if (logW[i] > mx) mx = logW[i];
  const w = new Float64Array(n); let sw = 0;
  for (let i = 0; i < n; i++) { w[i] = Math.exp(logW[i] - mx); sw += w[i]; }
  const out = new Int32Array(M);
  const u0 = prng() / M;
  let c = w[0] / sw, j = 0;
  for (let m = 0; m < M; m++) {
    const u = u0 + m / M;
    while (u > c && j < n - 1) { j++; c += w[j] / sw; }
    out[m] = j;
  }
  return out;
}

// EAMIS-style return shape: { samples (unconstrained), logZ, rungs, betas, acceptRate }.
function smcSample(mv: any, opts: any) {
  const dim = mv.dim;
  const P = Math.max(2, opts.smcSteps ?? 12);                  // chain length (states kept per ancestor)
  const Ntarget = opts.smcParticles ?? opts.particles ?? 2000;
  const M = Math.max(2, Math.round(Ntarget / P));             // ancestors; total = M·P
  // CESS target fraction. A gentler ladder (0.7) than the textbook N/2 — our
  // random-walk move needs the rungs close enough to mix; 0.5 left the proposal
  // chasing too-large jumps and underestimated tails.
  const rho = opts.smcCESS ?? 0.7;
  const maxRungs = opts.smcMaxRungs ?? 200;
  const ridge = 1e-9;
  const seed = opts.seed ?? 0;
  const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;

  const baseKey = rng.keyFromSeed(seed);
  const prng = sampler.makePhiloxPrngAdapter(rng.stateFromKey(baseKey[0], baseKey[1]));

  // β=0: particles from the prior pool, scored once.
  let X: Float64Array[] = mv.initFromPrior(M * P, prng);
  let N = X.length;
  let { prior, lik } = mv.logPriorLikBatch(X);
  let beta = 0, logZ = 0, rungs = 0;
  const betas: number[] = [];
  let acceptSum = 0, acceptCount = 0;
  // Move-proposal scale, adapted across rungs toward the optimal RW acceptance
  // (~0.234). The population covariance sets the SHAPE; this sets the step size.
  // A fixed 2.38/√dim over-steps the cramped directions of a funnel (acceptance
  // collapses, the neck never gets explored) — adapting it is essential there.
  let scale = 2.38 / Math.sqrt(Math.max(dim, 1));

  // CESS of the incremental weights w_i = exp(db·lik_i) over the (equal-weight)
  // population: (Σw)² / Σw². Decreasing in db; lik=-Inf contributes zero weight.
  const cessAt = (db: number): number => {
    let mx = -Infinity;
    for (let i = 0; i < N; i++) { const lw = db * lik[i]; if (lw > mx) mx = lw; }
    if (!Number.isFinite(mx)) return 0;
    let s1 = 0, s2 = 0;
    for (let i = 0; i < N; i++) { const w = Math.exp(db * lik[i] - mx); s1 += w; s2 += w * w; }
    return s2 > 0 ? (s1 * s1) / s2 : 0;
  };

  while (beta < 1) {
    if (++rungs > maxRungs) {
      throw new Error('backend \'smc\': temperature ladder did not reach β=1 in '
        + maxRungs + ' rungs (stuck at β=' + beta.toFixed(3) + ') — raise the rung '
        + 'cap / particle count, or use backend \'mh\'/\'emcee\'');
    }
    const target = rho * N;

    // Adapt the next β: largest step that keeps CESS ≥ target, capped at β=1.
    let db: number;
    const full = 1 - beta;
    if (cessAt(full) >= target) {
      db = full;                                   // even reaching β=1 stays healthy
    } else {
      let lo = 0, hi = full;
      for (let it = 0; it < 60; it++) {
        const mid = 0.5 * (lo + hi);
        if (cessAt(mid) > target) lo = mid; else hi = mid;
      }
      db = 0.5 * (lo + hi);
    }
    if (!(db > 0)) {
      throw new Error('backend \'smc\': temperature failed to advance (likelihood '
        + 'too informative for the particle count) — try more particles or backend \'is\'');
    }
    const newBeta = Math.min(1, beta + db);
    const dbReal = newBeta - beta;

    // Reweight + accumulate evidence: logZ += log(mean exp(dbReal·lik)).
    const incLW = new Float64Array(N);
    for (let i = 0; i < N; i++) incLW[i] = dbReal * lik[i];
    const lse = logSumExpArr(incLW);
    if (!Number.isFinite(lse)) {
      throw new Error('backend \'smc\': all particle weights vanished at β=' + newBeta.toFixed(4)
        + ' — the tempered target lost all mass');
    }
    logZ += lse - Math.log(N);
    beta = newBeta; betas.push(beta);
    if (onProgress) onProgress(beta, 'smc');

    // Proposal covariance = weighted covariance of the current population.
    const wn = new Float64Array(N); { let s = 0; for (let i = 0; i < N; i++) { wn[i] = Math.exp(incLW[i] - lse); s += wn[i]; } /* ≈1 */ void s; }
    const mean = new Float64Array(dim);
    for (let i = 0; i < N; i++) { const xi = X[i]; const wi = wn[i]; for (let d = 0; d < dim; d++) mean[d] += wi * xi[d]; }
    const Sigma = new Float64Array(dim * dim);
    for (let i = 0; i < N; i++) {
      const xi = X[i], wi = wn[i];
      for (let a = 0; a < dim; a++) { const da = xi[a] - mean[a]; for (let b = 0; b <= a; b++) { const c = wi * da * (xi[b] - mean[b]); Sigma[a * dim + b] += c; Sigma[b * dim + a] += c; } }
    }
    for (let d = 0; d < dim; d++) Sigma[d * dim + d] += ridge;
    const L = new Float64Array(dim * dim);
    if (!cholesky(Sigma, L, dim)) {                // not PD → diagonal std fallback
      for (let i = 0; i < dim * dim; i++) L[i] = 0;
      for (let d = 0; d < dim; d++) { const v = Sigma[d * dim + d]; L[d * dim + d] = Math.sqrt(v > ridge ? v : ridge); }
    }

    // Resample M ancestors, then move (waste-free): keep all M·P chain states.
    const anc = systematicResample(incLW, M, prng);
    const keptX: Float64Array[] = new Array(M * P);
    const keptPrior = new Float64Array(M * P);
    const keptLik = new Float64Array(M * P);
    // Chain heads = the resampled ancestors (state 0 of each chain).
    const cur: Float64Array[] = new Array(M);
    const curPrior = new Float64Array(M);
    const curLik = new Float64Array(M);
    for (let m = 0; m < M; m++) { const a = anc[m]; cur[m] = Float64Array.from(X[a]); curPrior[m] = prior[a]; curLik[m] = lik[a]; }
    // Snapshot state 0.
    for (let m = 0; m < M; m++) { keptX[m] = Float64Array.from(cur[m]); keptPrior[m] = curPrior[m]; keptLik[m] = curLik[m]; }
    // P−1 Metropolis sweeps invariant to π_β; each sweep scores M proposals in
    // one batched call.
    let rungAcc = 0, rungProp = 0;
    for (let step = 1; step < P; step++) {
      const prop: Float64Array[] = new Array(M);
      for (let m = 0; m < M; m++) {
        const y = cur[m]; const z = new Float64Array(dim);
        for (let d = 0; d < dim; d++) z[d] = gaussianNoise(prng);
        const yp = new Float64Array(dim);
        for (let i = 0; i < dim; i++) { let acc = 0; for (let k = 0; k <= i; k++) acc += L[i * dim + k] * z[k]; yp[i] = y[i] + scale * acc; }
        prop[m] = yp;
      }
      const sc = mv.logPriorLikBatch(prop);
      const off = step * M;
      for (let m = 0; m < M; m++) {
        const curLp = curPrior[m] + beta * curLik[m];
        const propLp = sc.prior[m] + beta * sc.lik[m];
        acceptCount++; rungProp++;
        if (Number.isFinite(propLp) && Math.log(prng() + 1e-300) < (propLp - curLp)) {
          cur[m] = prop[m]; curPrior[m] = sc.prior[m]; curLik[m] = sc.lik[m]; acceptSum++; rungAcc++;
        }
        keptX[off + m] = Float64Array.from(cur[m]); keptPrior[off + m] = curPrior[m]; keptLik[off + m] = curLik[m];
      }
    }
    // Adapt the step size toward ~0.234 acceptance for the next rung (Robbins-
    // Monro-ish, capped). Clamped to a sane range so a degenerate rung can't
    // run it away.
    if (rungProp > 0) {
      const accRate = rungAcc / rungProp;
      scale *= Math.exp((accRate - 0.234) * 1.5);
      if (scale < 1e-3) scale = 1e-3; else if (scale > 10) scale = 10;
    }
    X = keptX; prior = keptPrior; lik = keptLik; N = keptX.length;
  }

  return {
    samples: X,
    logZ,
    rungs,
    betas,
    acceptRate: acceptCount > 0 ? acceptSum / acceptCount : 0,
    N,
  };
}

module.exports = { smcSample };
