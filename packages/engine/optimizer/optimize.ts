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
  const starts = Math.max(1, opts.starts || 1);
  const seed = (opts.seed ?? 0x51ed) >>> 0;
  const rng = mulberry32(seed);
  const spread = opts.startSpread ?? 1.5;

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

  // Run the optimizer from each start (start 0 = pivot); keep the best.
  // (v1 keeps only the global best; collecting all distinct modes for a
  // mixture proposal is the adaptive-IS extension.)
  let bestRun: any = null;
  for (let s = 0; s < starts; s++) {
    const zStart = s === 0 ? coords.toZ(x0) : coords.toZ(sampleStartX());
    const run = await optFn({
      evalCloud: evalZ, x0: zStart, sigma0,
      opts: {
        lambda: opts.lambda, maxGenerations: opts.maxGenerations,
        seed: (seed + 1013 * (s + 1)) >>> 0,
      },
    });
    if (!bestRun || run.value > bestRun.value) bestRun = run;
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
    mode, value, covariance, curvatureSource, conditioning, boundaryActive,
    noisy: !!opts.noisy, scale: coords.scales.slice(),
    nEvals, nBatches, generations: bestRun.generations,
    terminationReason: bestRun.reason, optimizer: optName,
  };
}

module.exports = { optimize, registerOptimizer, OPTIMIZERS };
