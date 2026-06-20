'use strict';

// =====================================================================
// optimizer/cmaes.ts — Covariance Matrix Adaptation Evolution Strategy
// =====================================================================
//
// The (µ/µ_w, λ)-CMA-ES with rank-µ + rank-one covariance adaptation and
// cumulative step-size adaptation (Hansen & Ostermeier 2001; Hansen 2016
// tutorial). MAXIMISES an injected async objective: `evalCloud(points)` →
// fitness per point. One generation = one `evalCloud` call (the batched
// engine scores the whole population at once); inflating λ trades cheap
// batch width for fewer generations.
//
// The core is pure and unconstrained over ℝᵈ: bounds and value-set
// transforms live in `coords.ts` (it injects `evalCloud` over transformed
// coordinates). Non-finite fitness (−∞/NaN out of support) sinks to worst
// rank and is dropped from recombination, so a partially-invalid landscape
// keeps the search inside the feasible region rather than being derailed.
//
// Dependency-free leaf (only ./linalg). The RNG is seedable for
// reproducibility; the viewer can inject a Philox-derived stream.

const { matvec, matSqrtAndInvSqrt } = require('./linalg.ts');

/** Deterministic, seedable PRNG (mulberry32) → uniform [0,1). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard-normal generator (Marsaglia polar) over a uniform source. */
function makeGaussian(rng: () => number): () => number {
  let spare: number | null = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0, v = 0, s = 0;
    do { u = 2 * rng() - 1; v = 2 * rng() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const m = Math.sqrt(-2 * Math.log(s) / s);
    spare = v * m;
    return u * m;
  };
}

/**
 * Run CMA-ES. `spec`:
 *   - `evalCloud(points: number[][]) => Promise<number[]>` — batch fitness, maximised.
 *   - `x0: number[]` — initial distribution mean.
 *   - `sigma0: number` — initial step size (in the same coordinates as x0).
 *   - `opts`: `{ lambda?, maxGenerations?, tolFun?, tolX?, seed?, rng? }`.
 * Returns `{ x, value, mean, sigma, C, generations, evals, reason }` where `x`
 * is the best feasible point found.
 */
async function cmaes(spec: any): Promise<any> {
  const { evalCloud, x0 } = spec;
  const opts = spec.opts || {};
  const N = x0.length;
  const rng = opts.rng || mulberry32(((opts.seed ?? 0x9e3779b9) >>> 0));
  const gauss = makeGaussian(rng);

  let sigma = typeof spec.sigma0 === 'number' ? spec.sigma0 : 1;

  // Strategy parameters (Hansen 2016, eqs. for the default scheme).
  const lambda = opts.lambda || (4 + Math.floor(3 * Math.log(N)));
  const mu = Math.floor(lambda / 2);
  const wRaw: number[] = [];
  for (let i = 0; i < mu; i++) wRaw.push(Math.log(mu + 0.5) - Math.log(i + 1));
  const wSum = wRaw.reduce((a, b) => a + b, 0);
  const w = wRaw.map((x) => x / wSum);
  const mueff = 1 / w.reduce((a, b) => a + b * b, 0);

  const cc = (4 + mueff / N) / (N + 4 + 2 * mueff / N);
  const cs = (mueff + 2) / (N + mueff + 5);
  const c1 = 2 / ((N + 1.3) ** 2 + mueff);
  const cmu = Math.min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((N + 2) ** 2 + mueff));
  const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (N + 1)) - 1) + cs;
  const chiN = Math.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N));

  let mean = x0.slice();
  const C: number[][] = [];
  for (let i = 0; i < N; i++) { C.push(new Array(N).fill(0)); C[i][i] = 1; }
  const pc = new Array(N).fill(0);
  const ps = new Array(N).fill(0);

  const maxGen = opts.maxGenerations || (100 + 100 * N);
  const tolFun = opts.tolFun ?? 1e-12;
  const tolX = opts.tolX ?? 1e-12;

  const best = { x: mean.slice(), value: -Infinity };
  let evals = 0;
  let gen = 0;
  let reason = 'maxGenerations';
  const histBest: number[] = [];

  for (gen = 0; gen < maxGen; gen++) {
    const { sqrt: Csqrt, invSqrt: Cinv, values: eigvals } = matSqrtAndInvSqrt(C);
    const emax = Math.max(...eigvals), emin = Math.min(...eigvals);
    if (emax / Math.max(emin, 1e-300) > 1e14) { reason = 'conditioning'; break; }

    // Sample λ candidates: y = C^{1/2}·z, x = mean + σ·y.
    const ys: number[][] = [];
    const xs: number[][] = [];
    for (let k = 0; k < lambda; k++) {
      const z = new Array(N);
      for (let i = 0; i < N; i++) z[i] = gauss();
      const y = matvec(Csqrt, z);
      ys.push(y);
      xs.push(mean.map((mi: number, i: number) => mi + sigma * y[i]));
    }
    const fitness = await evalCloud(xs);
    evals += lambda;

    // Rank by fitness DESC; non-finite sinks to worst, then is dropped.
    const ranked = fitness.map((_: any, i: number) => i)
      .filter((i: number) => Number.isFinite(fitness[i]))
      .sort((a: number, b: number) => fitness[b] - fitness[a]);
    if (ranked.length === 0) continue; // whole population infeasible — resample

    if (fitness[ranked[0]] > best.value) {
      best.x = xs[ranked[0]].slice();
      best.value = fitness[ranked[0]];
    }

    // Recombination over the feasible top-µ, weights renormalised to that subset.
    const m2 = Math.min(mu, ranked.length);
    let wsum2 = 0;
    for (let i = 0; i < m2; i++) wsum2 += w[i];
    const meanNew = new Array(N).fill(0);
    const yw = new Array(N).fill(0);
    for (let i = 0; i < m2; i++) {
      const wi = w[i] / wsum2;
      const xi = xs[ranked[i]], yi = ys[ranked[i]];
      for (let d = 0; d < N; d++) { meanNew[d] += wi * xi[d]; yw[d] += wi * yi[d]; }
    }

    // Step-size evolution path.
    const CinvYw = matvec(Cinv, yw);
    for (let d = 0; d < N; d++)
      ps[d] = (1 - cs) * ps[d] + Math.sqrt(cs * (2 - cs) * mueff) * CinvYw[d];
    const psNorm = Math.sqrt(ps.reduce((a: number, b: number) => a + b * b, 0));

    const hsig = (psNorm / Math.sqrt(1 - Math.pow(1 - cs, 2 * (gen + 1))) / chiN)
      < (1.4 + 2 / (N + 1)) ? 1 : 0;

    // Covariance evolution path.
    for (let d = 0; d < N; d++)
      pc[d] = (1 - cc) * pc[d] + hsig * Math.sqrt(cc * (2 - cc) * mueff) * yw[d];

    // Adapt C: (1−c1−cµ)·C + rank-one (pc pcᵀ + δ·C) + rank-µ.
    const deltaHsig = (1 - hsig) * cc * (2 - cc);
    for (let a = 0; a < N; a++) {
      for (let b = 0; b < N; b++) {
        let rankMu = 0;
        for (let i = 0; i < m2; i++) {
          const yi = ys[ranked[i]];
          rankMu += (w[i] / wsum2) * yi[a] * yi[b];
        }
        C[a][b] = (1 - c1 - cmu) * C[a][b]
          + c1 * (pc[a] * pc[b] + deltaHsig * C[a][b])
          + cmu * rankMu;
      }
    }
    for (let a = 0; a < N; a++) for (let b = a + 1; b < N; b++) {
      const s = 0.5 * (C[a][b] + C[b][a]); C[a][b] = s; C[b][a] = s;
    }

    // Adapt σ.
    sigma *= Math.exp((cs / damps) * (psNorm / chiN - 1));
    mean = meanNew;

    // Termination.
    histBest.push(best.value);
    if (histBest.length > 20) histBest.shift();
    if (sigma * Math.sqrt(emax) < tolX) { reason = 'tolX'; break; }
    if (histBest.length >= 20
      && (Math.max(...histBest) - Math.min(...histBest)) < tolFun) {
      reason = 'tolFun'; break;
    }
    if (!Number.isFinite(sigma) || sigma > 1e60) { reason = 'diverged'; break; }
  }

  return { x: best.x, value: best.value, mean, sigma, C, generations: gen, evals, reason };
}

module.exports = { cmaes, mulberry32, makeGaussian };
