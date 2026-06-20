// packages/engine/elliptical-slice-kernel.ts
'use strict';
// Elliptical slice sampling (Murray, Adams & MacKay 2010) as a runMcmc kernel.
//
// Gradient-free and TUNING-FREE (no step size): propose on an ellipse against a
// Gaussian reference g = N(μ,Σ) and slice-sample the ratio h = logπ − log g,
// shrinking an angle bracket until acceptance. The ellipse exactly preserves g,
// so the slice never has to fight the reference. Reference:
//   • EXACT  — when the unconstrained prior is itself an independent Gaussian
//     (mv.gaussianPrior); then h = logπ − log(prior) = the likelihood, the
//     textbook ESS. Most efficient.
//   • FITTED — otherwise a Gaussian fit to the prior-pool population, re-fit
//     from the warmup population (a preconditioned elliptical slice).
//
// Batching: all chains step out / shrink together — each shrink round scores the
// active chains' proposals in ONE mv.logPosteriorBatch call.

const { gaussianNoise } = require('./mcmc-driver.ts');

const LN_2PI = Math.log(2 * Math.PI);

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

// log N(x; μ, Σ) via the reference Cholesky L: solve L u = x−μ, q = uᵀu.
function logGauss(x: Float64Array, A: any, dim: number): number {
  const L = A.L_ref, mu = A.mu_ref, u = A.uScratch;
  for (let i = 0; i < dim; i++) {
    let s = x[i] - mu[i];
    for (let k = 0; k < i; k++) s -= L[i * dim + k] * u[k];
    u[i] = s / L[i * dim + i];
  }
  let q = 0;
  for (let i = 0; i < dim; i++) q += u[i] * u[i];
  return -0.5 * (dim * LN_2PI + A.logDet + q);
}

// Build the Gaussian reference (exact from mv.gaussianPrior, else fitted from a
// prior-pool sample). Lazy — runs on the first step, where a PRNG is available.
function buildRef(A: any, mv: any, prng: () => number, dim: number): void {
  const L = new Float64Array(dim * dim);
  const mu = new Float64Array(dim);
  if (A.mode === 'exact') {
    const g = mv.gaussianPrior;
    for (let d = 0; d < dim; d++) { mu[d] = g.mu[d]; L[d * dim + d] = g.sigma[d]; }
  } else {
    const pool: Float64Array[] = mv.initFromPrior(Math.max(256, 20 * dim), prng);
    fitGaussian(pool, mu, L, dim);
  }
  A.mu_ref = mu; A.L_ref = L;
  let logDet = 0; for (let d = 0; d < dim; d++) logDet += 2 * Math.log(L[d * dim + d]);
  A.logDet = logDet;
  A.uScratch = new Float64Array(dim);
  A.refBuilt = true;
}

// Mean + covariance of a population → μ and its lower-Cholesky L (diagonal-std
// fallback if the covariance isn't positive definite).
function fitGaussian(pts: Float64Array[], mu: Float64Array, L: Float64Array, dim: number): void {
  const n = pts.length, ridge = 1e-9;
  mu.fill(0);
  for (const p of pts) for (let d = 0; d < dim; d++) mu[d] += p[d];
  for (let d = 0; d < dim; d++) mu[d] /= n;
  const Sigma = new Float64Array(dim * dim);
  for (const p of pts) {
    for (let a = 0; a < dim; a++) { const da = p[a] - mu[a]; for (let b = 0; b <= a; b++) { const c = da * (p[b] - mu[b]); Sigma[a * dim + b] += c; Sigma[b * dim + a] += c; } }
  }
  for (let i = 0; i < dim * dim; i++) Sigma[i] /= n;
  for (let d = 0; d < dim; d++) Sigma[d * dim + d] += ridge;
  for (let i = 0; i < dim * dim; i++) L[i] = 0;
  if (!cholesky(Sigma, L, dim)) {
    for (let d = 0; d < dim; d++) { const v = Sigma[d * dim + d]; L[d * dim + d] = Math.sqrt(v > ridge ? v : ridge); }
  }
}

function makeEllipticalSliceKernel() {
  return {
    init(_nWalkers: number, dim: number, opts: any, mv: any) {
      return {
        dim,
        maxShrink: opts && opts.essMaxShrink ? opts.essMaxShrink : 50,
        mode: (mv && mv.gaussianPrior) ? 'exact' : 'fitted',
        refBuilt: false,
        // Warmup re-fit accumulators (fitted mode).
        sum: new Float64Array(dim),
        cross: new Float64Array(dim * dim),
        count: 0,
        iter: 0,
      };
    },
    step(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, A: any, phase: string) {
      const dim = A.dim, N = ensemble.length, maxShrink = A.maxShrink;
      if (!A.refBuilt) buildRef(A, mv, prng, dim);
      const mu = A.mu_ref, L = A.L_ref;

      const nu: Float64Array[] = new Array(N);
      const theta = new Float64Array(N), thMin = new Float64Array(N), thMax = new Float64Array(N), slice = new Float64Array(N);
      const active = new Array(N).fill(true); let nActive = N;
      for (let k = 0; k < N; k++) {
        // ν = μ + L z
        const z = new Float64Array(dim); for (let d = 0; d < dim; d++) z[d] = gaussianNoise(prng);
        const v = new Float64Array(dim);
        for (let i = 0; i < dim; i++) { let acc = 0; for (let j = 0; j <= i; j++) acc += L[i * dim + j] * z[j]; v[i] = mu[i] + acc; }
        nu[k] = v;
        const hx = logp[k] - logGauss(ensemble[k], A, dim);
        slice[k] = hx + Math.log(prng() + 1e-300);     // slice level (h(x) + log u)
        theta[k] = 2 * Math.PI * prng(); thMin[k] = theta[k] - 2 * Math.PI; thMax[k] = theta[k];
      }

      let totalEvals = 0;
      for (let it = 0; it < maxShrink && nActive > 0; it++) {
        const idxs: number[] = []; const props: Float64Array[] = [];
        for (let k = 0; k < N; k++) {
          if (!active[k]) continue;
          const c = Math.cos(theta[k]), s = Math.sin(theta[k]);
          const xk = ensemble[k], vk = nu[k];
          const xp = new Float64Array(dim);
          for (let d = 0; d < dim; d++) xp[d] = mu[d] + (xk[d] - mu[d]) * c + (vk[d] - mu[d]) * s;
          idxs.push(k); props.push(xp);
        }
        const lp = mv.logPosteriorBatch(props); totalEvals += props.length;
        for (let a = 0; a < idxs.length; a++) {
          const k = idxs[a];
          const hp = Number.isFinite(lp[a]) ? lp[a] - logGauss(props[a], A, dim) : -Infinity;
          if (hp > slice[k]) {
            ensemble[k] = props[a]; logp[k] = lp[a]; active[k] = false; nActive--;
          } else {
            // Shrink the bracket toward 0 on the side θ lies, redraw θ.
            if (theta[k] < 0) thMin[k] = theta[k]; else thMax[k] = theta[k];
            theta[k] = thMin[k] + (thMax[k] - thMin[k]) * prng();
          }
        }
      }
      // Chains still active hit maxShrink — keep their current state (θ→0 always
      // satisfies the slice, so this is rare; a safe no-op when it happens).

      if (phase === 'warmup' && A.mode === 'fitted') {
        const sum = A.sum, cross = A.cross;
        for (let k = 0; k < N; k++) {
          const x = ensemble[k];
          for (let i = 0; i < dim; i++) { sum[i] += x[i]; for (let j = 0; j <= i; j++) cross[i * dim + j] += x[i] * x[j]; }
        }
        A.count += N; A.iter++;
        // Re-fit the reference from accrued warmup positions every 25 sweeps.
        if (A.iter % 25 === 0 && A.count > dim + 2) {
          const n = A.count, ridge = 1e-9;
          const m = new Float64Array(dim); for (let d = 0; d < dim; d++) m[d] = sum[d] / n;
          const Sigma = new Float64Array(dim * dim);
          for (let i = 0; i < dim; i++) for (let j = 0; j <= i; j++) { const c = cross[i * dim + j] / n - m[i] * m[j]; Sigma[i * dim + j] = c; Sigma[j * dim + i] = c; }
          for (let d = 0; d < dim; d++) Sigma[d * dim + d] += ridge;
          const Lnew = new Float64Array(dim * dim);
          if (cholesky(Sigma, Lnew, dim)) {
            A.mu_ref = m; A.L_ref = Lnew;
            let logDet = 0; for (let d = 0; d < dim; d++) logDet += 2 * Math.log(Lnew[d * dim + d]); A.logDet = logDet;
          }
        }
      }
      return { accepts: N - nActive, proposals: totalEvals };
    },
  };
}

module.exports = { makeEllipticalSliceKernel };
