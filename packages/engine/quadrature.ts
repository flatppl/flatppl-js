'use strict';

// Gauss-Legendre nodes/weights on [0,1] (interior — never 0 or 1).
// 3-point (degree-5 exact) is the accurate rule; 2-point (degree-3) is the
// embedded coarse rule. The cell error estimate is |I3 - I2|.
const GL3 = [
  { x: 0.5 - 0.5 * Math.sqrt(3 / 5), w: 5 / 18 },
  { x: 0.5,                          w: 8 / 18 },
  { x: 0.5 + 0.5 * Math.sqrt(3 / 5), w: 5 / 18 },
];
const GL2 = [
  { x: 0.5 - 0.5 / Math.sqrt(3), w: 0.5 },
  { x: 0.5 + 0.5 / Math.sqrt(3), w: 0.5 },
];

// Tensor-product application of a 1-D rule over cell [lo,hi] in R^dims.
function tensorRule(
  integrand: (u: number[]) => number,
  lo: number[], hi: number[],
  rule: { x: number; w: number }[],
): number {
  const dims = lo.length;
  const h = lo.map((l, i) => hi[i] - l);
  let total = 0;
  const rec = (d: number, pt: number[], w: number) => {
    if (d === dims) { total += w * integrand(pt); return; }
    for (const { x, w: wt } of rule) rec(d + 1, pt.concat(lo[d] + x * h[d]), w * wt * h[d]);
  };
  rec(0, [], 1.0);
  return total;
}

// (I3, error) for one cell.
function cellEstimate(
  integrand: (u: number[]) => number, lo: number[], hi: number[],
): { I: number; E: number } {
  const fine = tensorRule(integrand, lo, hi, GL3);
  const coarse = tensorRule(integrand, lo, hi, GL2);
  return { I: fine, E: Math.abs(fine - coarse) };
}

interface Cell { lo: number[]; hi: number[]; I: number; E: number; }

export function adaptiveCubature(
  integrand: (u: number[]) => number,
  dims: number,
  opts?: { tol?: number; maxEvals?: number },
): { Z: number; err: number; evals: number } {
  const tol = (opts && opts.tol) ?? 1e-8;
  const maxEvals = (opts && opts.maxEvals) ?? 200000;
  const perCell = Math.pow(3, dims) + Math.pow(2, dims);

  // Seed with a minimum uniform subdivision (INIT_DIV cells per axis) rather
  // than a single cell: a feature narrower than the whole box (e.g. a sharp
  // interior peak) could be stepped over by one coarse cell's 2-/3-point rule,
  // making the embedded error estimate falsely tiny and terminating the loop at
  // a wrong Z with a wrong (small) `err` — the classic global-adaptive blind
  // spot, and a refuse-don't-mislower hazard. The seed grid guarantees the
  // integrand is sampled at least at this resolution before the convergence
  // test is trusted; adaptive refinement then concentrates where it matters.
  // Seed cost is bounded for dims ≤ 3 (8/64/512 cells).
  const INIT_DIV = 8;
  const step = 1 / INIT_DIV;
  // Simple array as a max-by-E "heap": we scan for the worst cell each step.
  // Cell counts stay small (hundreds–low thousands) for dims ≤ 3, so a linear
  // worst-cell scan is not the bottleneck; the integrand evals are.
  const cells: Cell[] = [];
  let totI = 0, totE = 0, evals = 0;
  const idx = new Array(dims).fill(0);
  const seedCount = Math.pow(INIT_DIV, dims);
  for (let c = 0; c < seedCount; c++) {
    const lo = idx.map((k) => k * step);
    const hi = idx.map((k) => (k + 1) * step);
    const est = cellEstimate(integrand, lo, hi);
    cells.push({ lo, hi, I: est.I, E: est.E });
    totI += est.I; totE += est.E; evals += perCell;
    for (let d = 0; d < dims; d++) { if (++idx[d] < INIT_DIV) break; idx[d] = 0; }
  }

  while (totE > tol * Math.abs(totI) && evals < maxEvals) {
    // worst cell by E
    let wi = 0;
    for (let i = 1; i < cells.length; i++) if (cells[i].E > cells[wi].E) wi = i;
    const c = cells[wi];
    totI -= c.I; totE -= c.E;
    // bisect the longest edge
    let ax = 0; let best = -Infinity;
    for (let i = 0; i < dims; i++) { const len = c.hi[i] - c.lo[i]; if (len > best) { best = len; ax = i; } }
    const mid = 0.5 * (c.lo[ax] + c.hi[ax]);
    const aLo = c.lo.slice(), aHi = c.hi.slice(); aHi[ax] = mid;
    const bLo = c.lo.slice(), bHi = c.hi.slice(); bLo[ax] = mid;
    const ea = cellEstimate(integrand, aLo, aHi);
    const eb = cellEstimate(integrand, bLo, bHi);
    evals += 2 * perCell;
    cells[wi] = { lo: aLo, hi: aHi, I: ea.I, E: ea.E };
    cells.push({ lo: bLo, hi: bHi, I: eb.I, E: eb.E });
    totI += ea.I + eb.I; totE += ea.E + eb.E;
  }
  return { Z: totI, err: totE, evals };
}

// 5-point Gauss-Legendre nodes/weights on [0,1] (degree-9 exact).
const GL5 = [
  { x: 0.5 * (1 - 0.9061798459386640), w: 0.5 * 0.2369268850561891 },
  { x: 0.5 * (1 - 0.5384693101056831), w: 0.5 * 0.4786286704993665 },
  { x: 0.5,                            w: 0.5 * 0.5688888888888889 },
  { x: 0.5 * (1 + 0.5384693101056831), w: 0.5 * 0.4786286704993665 },
  { x: 0.5 * (1 + 0.9061798459386640), w: 0.5 * 0.2369268850561891 },
];

// A FIXED composite Gauss-Legendre rule on (0,1) whose cells are graded
// geometrically towards BOTH endpoints: the breakpoints are 0, 2^-L, …, 1/2, …,
// 1 − 2^-L, 1, and each cell carries the 5-point rule above.
//
// WHY GRADED AND NOT UNIFORM. The caller integrates ∫₀¹ f(F⁻¹(u)) du, and F⁻¹
// carries u → 0 and u → 1 to the base measure's tails, so the integrand's whole
// structure sits in the two endpoint cells. A uniform grid resolves none of it.
// The dyadic grading resolves a tail decaying at any exponential rate, and its
// outermost cell doubles as the caller's accuracy probe: for an integrand this
// rule resolves, that cell's contribution is a negligible fraction of the total.
//
// Returned per CELL rather than as one flat node list: the caller needs the
// outermost cells separately, and the grouping is not recoverable from the
// nodes alone.
export function gradedUnitCells(levels: number): Array<{ us: number[]; ws: number[] }> {
  const bps: number[] = [0];
  for (let k = levels; k >= 1; k--) bps.push(Math.pow(2, -k));
  for (let k = 2; k <= levels; k++) bps.push(1 - Math.pow(2, -k));
  bps.push(1);
  const cells: Array<{ us: number[]; ws: number[] }> = [];
  for (let i = 0; i + 1 < bps.length; i++) {
    const lo = bps[i], hi = bps[i + 1];
    const h = hi - lo;
    cells.push({
      us: GL5.map((n) => lo + n.x * h),
      ws: GL5.map((n) => n.w * h),
    });
  }
  return cells;
}

// Fixed composite Gauss-Legendre estimate of ∫_lo^hi f over ONE 1-D window.
//
// WHY THIS IS SEPARATE FROM `adaptiveCubature`. The adaptive routine cannot
// tell a CONVERGENT endpoint singularity from a DIVERGENT one: both look
// locally smooth to the embedded error estimate, so it reports "converged" at
// whatever finite number its last subdivision reached. The discriminator is the
// WINDOW's own contribution — for a convergent endpoint ∫_{1-δ}^{1} f → 0 as
// δ → 0, and for a divergent one it does not. Callers probe a geometric ladder
// of δ with this and refuse when the contribution fails to shrink.
export function windowIntegral(
  f: (u: number[]) => number, lo: number, hi: number, cells = 32,
): number {
  const h = (hi - lo) / cells;
  let total = 0;
  for (let i = 0; i < cells; i++) {
    for (const { x, w } of GL3) total += w * h * f([lo + (i + x) * h]);
  }
  return total;
}
