'use strict';

// =====================================================================
// optimizer/polish.ts — finite-difference local polish + Laplace covariance
// =====================================================================
//
// The cheap, near-exact refinement for smooth (closed-form) objectives.
// On the batched engine a central-difference gradient is 2d points = one
// batch; a parallel line search is one batch of step candidates; the
// Hessian is O(d²) second differences = a few batches. We iterate a few
// gradient-ascent + line-search steps to snap onto the peak, then evaluate
// the Hessian there and return the Laplace covariance Σ = (−H)⁻¹ (the
// inverse observed information — a Gaussian fit to the mode for the
// downstream importance-sampling proposal). `ok` is false when H is not
// negative-definite (the point is not a clean maximum), so a bad fit is
// reported rather than silently inverted. Operates in whatever coordinate
// `evalCloud` uses; the orchestrator runs it in the normalised z-space.

const { symEig, vDiagVt } = require('./linalg.ts');

/** Central-difference gradient at `z` (2d points in one batch). */
async function fdGradient(evalCloud: any, z: number[], h?: number): Promise<number[]> {
  const step = h || 1e-4;
  const n = z.length;
  const pts: number[][] = [];
  for (let i = 0; i < n; i++) {
    const zp = z.slice(); zp[i] += step; pts.push(zp);
    const zm = z.slice(); zm[i] -= step; pts.push(zm);
  }
  const v = await evalCloud(pts);
  const g = new Array(n);
  for (let i = 0; i < n; i++) g[i] = (v[2 * i] - v[2 * i + 1]) / (2 * step);
  return g;
}

/** Central-difference Hessian at `z` (one batch: center + axis + cross points). */
async function fdHessian(evalCloud: any, z: number[], h?: number): Promise<number[][]> {
  const step = h || 1e-3;
  const n = z.length;
  const pts: number[][] = [];
  const idx: any = {};
  const add = (key: string, p: number[]) => { idx[key] = pts.length; pts.push(p); };
  add('c', z.slice());
  for (let i = 0; i < n; i++) {
    const pp = z.slice(); pp[i] += step; add('p' + i, pp);
    const pm = z.slice(); pm[i] -= step; add('m' + i, pm);
  }
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const a = z.slice(); a[i] += step; a[j] += step; add(`pp${i}_${j}`, a);
    const b = z.slice(); b[i] += step; b[j] -= step; add(`pm${i}_${j}`, b);
    const c = z.slice(); c[i] -= step; c[j] += step; add(`mp${i}_${j}`, c);
    const d = z.slice(); d[i] -= step; d[j] -= step; add(`mm${i}_${j}`, d);
  }
  const v = await evalCloud(pts);
  const V = (k: string) => v[idx[k]];
  const fc = V('c');
  const H: number[][] = [];
  for (let i = 0; i < n; i++) H.push(new Array(n).fill(0));
  const h2 = step * step;
  for (let i = 0; i < n; i++) H[i][i] = (V('p' + i) - 2 * fc + V('m' + i)) / h2;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const hij = (V(`pp${i}_${j}`) - V(`pm${i}_${j}`) - V(`mp${i}_${j}`) + V(`mm${i}_${j}`)) / (4 * h2);
    H[i][j] = hij; H[j][i] = hij;
  }
  return H;
}

/**
 * Laplace covariance from a Hessian: Σ = (−H)⁻¹ when −H is positive-definite
 * (a genuine maximum), else `{ covariance: null, ok: false }`.
 */
function laplaceCovariance(H: number[][]): { covariance: number[][] | null; ok: boolean } {
  const negH = H.map((row) => row.map((v: number) => -v));
  const { values, vectors } = symEig(negH);
  for (const v of values) if (!(v > 1e-10)) return { covariance: null, ok: false };
  return { covariance: vDiagVt(vectors, values.map((v: number) => 1 / v)), ok: true };
}

/**
 * Iterated gradient ascent with a batched line search, then a Laplace fit.
 * Returns `{ z, value, covariance, ok, gradNorm, iterations, hessian }`.
 */
async function polish(evalCloud: any, z0: number[], opts?: any): Promise<any> {
  const o = opts || {};
  const maxIter = o.maxIter ?? 12;
  const hGrad = o.hGrad ?? 1e-4;
  const hHess = o.hHess ?? 1e-3;
  const baseSteps: number[] = o.steps || [2, 1, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01];

  let z = z0.slice();
  let val = (await evalCloud([z]))[0];
  let gradNorm = Infinity;
  let it = 0;

  for (it = 0; it < maxIter; it++) {
    const g = await fdGradient(evalCloud, z, hGrad);
    gradNorm = Math.sqrt(g.reduce((s, x) => s + x * x, 0));
    if (!Number.isFinite(gradNorm) || gradNorm === 0) break;
    const dir = g.map((x) => x / gradNorm);

    // Batched parallel line search: try the step set, then a 10× finer set
    // if none improved. Take the best improving step (no sequential backtrack).
    const tryAlphas = async (alphas: number[]) => {
      const cand = alphas.map((a) => z.map((zi, i) => zi + a * dir[i]));
      const cv = await evalCloud(cand);
      let bestA = -1, bestV = val, bestZ = z;
      for (let k = 0; k < alphas.length; k++) {
        if (Number.isFinite(cv[k]) && cv[k] > bestV) { bestV = cv[k]; bestZ = cand[k]; bestA = alphas[k]; }
      }
      return { bestA, bestV, bestZ };
    };

    let { bestA, bestV, bestZ } = await tryAlphas(baseSteps);
    if (bestA < 0) ({ bestA, bestV, bestZ } = await tryAlphas(baseSteps.map((a) => a * 0.1)));
    if (bestA < 0) break; // no improving step — converged
    z = bestZ; val = bestV;
  }

  const hessian = await fdHessian(evalCloud, z, hHess);
  const { covariance, ok } = laplaceCovariance(hessian);
  return { z, value: val, covariance, ok, gradNorm, iterations: it, hessian };
}

module.exports = { fdGradient, fdHessian, laplaceCovariance, polish };
