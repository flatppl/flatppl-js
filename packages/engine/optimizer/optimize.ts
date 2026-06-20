'use strict';

// =====================================================================
// optimizer/optimize.ts — orchestration + modular optimizer registry
// =====================================================================
//
// `optimize(spec)` is the engine-side entry the viewer's "Find maximum"
// button drives. It maximises an objective over a model's free inputs,
// starting from the pivot, and returns a `ModeFit`:
//
//   { mode, value, covariance, curvatureSource, conditioning,
//     boundaryActive, noisy, scale, nEvals, nBatches, generations,
//     terminationReason, optimizer }
//
// `mode` is in the original parameter space; `covariance` is the Laplace
// fit in the normalised z-space (the inverse observed information) — the
// reusable artefact the adaptive-IS step (next) turns into a proposal
// measure. Pipeline: coords (value-set/plot-scale transform) → the chosen
// optimizer over z-space (CMA-ES by default, robust to a far pivot) →
// FD-Hessian polish (cheap/near-exact on smooth closed-form targets) →
// ModeFit. Multi-start disperses extra starts across the domain so a far
// or bad pivot still finds the global basin.
//
// The optimizer is chosen from a registry (`opts.optimizer`, default
// 'cmaes') so other optimizers can be added without touching callers.

const { makeCoords } = require('./coords.ts');
const { cmaes, mulberry32 } = require('./cmaes.ts');
const { polish: fdPolish } = require('./polish.ts');
const { symEig } = require('./linalg.ts');

// name → async (zspec) => { x (best z), value, [C], [sigma], generations, evals, reason }
const OPTIMIZERS: any = { cmaes };
function registerOptimizer(name: string, fn: any): void { OPTIMIZERS[name] = fn; }

function gaussFrom(rng: () => number): number {
  let u = 0, v = 0, s = 0;
  do { u = 2 * rng() - 1; v = 2 * rng() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}

/**
 * @param spec `{ evalCloud, x0, domains, scales, opts }`.
 *   - `evalCloud(points: number[][]) => Promise<number[]>` — original-space batch
 *     fitness, MAXIMISED (use logdensityof, not densityof).
 *   - `domains` / `scales` — per free input (value-set domain + plot-range scale).
 *   - `opts`: `{ optimizer?, sigma0?, lambda?, maxGenerations?, starts?, startSpread?,
 *               seed?, polish?, noisy?, hGrad?, hHess? }`.
 */
async function optimize(spec: any): Promise<any> {
  const { evalCloud, x0, domains } = spec;
  const opts = spec.opts || {};
  const n = domains.length;
  const coords = makeCoords({ domains, scales: spec.scales || [], x0 });

  const optName = opts.optimizer || 'cmaes';
  const optFn = OPTIMIZERS[optName];
  if (!optFn) throw new Error(`optimize: unknown optimizer '${optName}'`);

  let nBatches = 0, nEvals = 0;
  const evalZ = async (zCloud: number[][]) => {
    nBatches++; nEvals += zCloud.length;
    return evalCloud(zCloud.map((z) => coords.toX(coords.project(z))));
  };

  const sigma0 = opts.sigma0 ?? 0.3; // normalised z-units ≈ a fraction of a plot scale
  const seed = (opts.seed ?? 0x51ed) >>> 0;
  const rng = mulberry32(seed);
  const spread = opts.startSpread ?? 1.5;
  const lambdaDef = opts.lambda || (4 + Math.floor(3 * Math.log(Math.max(n, 1))));

  const sampleStartX = (): number[] => {
    const x = new Array(n);
    for (let i = 0; i < n; i++) {
      const d = domains[i];
      if (d.kind === 'interval') x[i] = d.lo + rng() * (d.hi - d.lo);
      else if (d.kind === 'posreals' || d.kind === 'nonnegreals') {
        const base = x0[i] > 0 ? x0[i] : Math.max(1e-6, coords.scales[i]);
        x[i] = base * Math.exp(gaussFrom(rng) * spread);
      } else x[i] = x0[i] + gaussFrom(rng) * spread * coords.scales[i];
    }
    return x;
  };

  // BIPOP-CMA-ES restart regime (Hansen 2009 — the BBOB front-runner). Restart 0
  // runs from the pivot at the default population. Subsequent restarts interleave
  // two regimes, always feeding the one that has spent LESS budget:
  //   • LARGE (IPOP): population doubled each time — global / multimodal coverage.
  //   • SMALL: a smaller population with a small random σ — local refinement of
  //     narrow peaks (the sharply-peaked-likelihood case a single run misses).
  // Budget-adaptive: capped by restart count, an optional eval budget, and an
  // early stop once several restarts in a row fail to improve. Strictly better
  // than a fixed count of equal-size starts. (v1 keeps only the global best;
  // collecting all distinct modes for a mixture proposal is the adaptive-IS
  // extension.)
  const maxRestarts = Math.max(0, opts.starts != null ? opts.starts - 1 : (opts.maxRestarts ?? 9));
  const maxEvals = opts.maxEvals ?? (2000 * (n + 1));
  const stagnateStop = opts.stagnateRestarts ?? 3;
  let bestRun: any = null;
  let budgetLarge = 0, budgetSmall = 0, iLarge = 0, stagnant = 0;
  for (let restart = 0; restart <= maxRestarts; restart++) {
    if (restart > 0 && (nEvals >= maxEvals || stagnant >= stagnateStop)) break;
    let lambda = lambdaDef, sig = sigma0, large = true;
    let zStart = coords.toZ(x0);
    if (restart > 0) {
      zStart = coords.toZ(sampleStartX());
      large = budgetLarge <= budgetSmall;
      if (large) { iLarge++; lambda = Math.floor(lambdaDef * Math.pow(2, iLarge)); }
      else {
        const u = rng();
        const lambdaLarge = lambdaDef * Math.pow(2, Math.max(1, iLarge));
        lambda = Math.max(2, Math.floor(lambdaDef * Math.pow(0.5 * lambdaLarge / lambdaDef, u * u)));
        sig = sigma0 * Math.pow(10, -2 * rng());   // small σ for fine local search
      }
    }
    const before = nEvals;
    const run = await optFn({
      evalCloud: evalZ, x0: zStart, sigma0: sig,
      opts: { lambda, maxGenerations: opts.maxGenerations, seed: (seed + 1013 * (restart + 1)) >>> 0 },
    });
    if (restart > 0) { if (large) budgetLarge += (nEvals - before); else budgetSmall += (nEvals - before); }
    if (!bestRun || run.value > bestRun.value + 1e-12) { bestRun = run; stagnant = 0; }
    else stagnant++;
  }

  let z = bestRun.x;
  let value = bestRun.value;
  let covariance: number[][] | null = null;
  let curvatureSource = 'none';

  if (opts.polish !== false && !opts.noisy) {
    const p = await fdPolish(evalZ, z, { hGrad: opts.hGrad, hHess: opts.hHess });
    if (Number.isFinite(p.value) && p.value >= value) { z = p.z; value = p.value; }
    if (p.ok) { covariance = p.covariance; curvatureSource = 'fd-hessian'; }
  }
  if (!covariance && bestRun.C && Number.isFinite(bestRun.sigma)) {
    const s2 = bestRun.sigma * bestRun.sigma;
    covariance = bestRun.C.map((row: number[]) => row.map((v) => v * s2));
    curvatureSource = 'cma-C';
  }

  const mode = coords.toX(coords.project(z));

  // Per-axis marginal std in ORIGINAL (x) space — the Laplace curvature width
  // a viewer plot uses to frame the peak (mode ± k·sd) instead of the much
  // wider prior. `covariance` is the z-space (normalised) Laplace fit; map its
  // per-axis 1σ displacement through the local coordinate jacobian (affine for
  // reals/interval, log for posreals — so the symmetric z step gives the right
  // x half-width even on the nonlinear axes). null when curvature is unavailable.
  let sd: number[] | null = null;
  if (covariance) {
    const zc = coords.toZ(mode);
    sd = covariance.map((row: number[], i: number) => {
      const sz = Math.sqrt(Math.max(0, row[i]));
      if (!(sz > 0) || !Number.isFinite(sz)) return 0;
      const zp = zc.slice(); zp[i] += sz;
      const zm = zc.slice(); zm[i] -= sz;
      const xp = coords.toX(coords.project(zp))[i];
      const xm = coords.toX(coords.project(zm))[i];
      const w = 0.5 * Math.abs(xp - xm);
      return Number.isFinite(w) ? w : 0;
    });
  }

  const boundaryActive = domains.map((d: any, i: number) => {
    if (d.kind !== 'interval') return false;
    const eps = 1e-6 * Math.max(1, Math.abs(d.hi - d.lo));
    return Math.abs(mode[i] - d.lo) < eps || Math.abs(mode[i] - d.hi) < eps;
  });

  let conditioning: number | null = null;
  if (covariance) {
    const { values } = symEig(covariance);
    const emax = Math.max(...values), emin = Math.min(...values);
    conditioning = emin > 0 ? emax / emin : Infinity;
  }

  return {
    mode, sd, value, covariance, curvatureSource, conditioning, boundaryActive,
    noisy: !!opts.noisy, scale: coords.scales.slice(),
    nEvals, nBatches, generations: bestRun.generations,
    terminationReason: bestRun.reason, optimizer: optName,
  };
}

module.exports = { optimize, registerOptimizer, OPTIMIZERS };
