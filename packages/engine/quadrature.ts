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

  const lo0 = new Array(dims).fill(0);
  const hi0 = new Array(dims).fill(1);
  const first = cellEstimate(integrand, lo0, hi0);
  // Simple array as a max-by-E "heap": we scan for the worst cell each step.
  // Cell counts stay small (hundreds–low thousands) for dims ≤ 3, so a linear
  // worst-cell scan is not the bottleneck; the integrand evals are.
  const cells: Cell[] = [{ lo: lo0, hi: hi0, I: first.I, E: first.E }];
  let totI = first.I, totE = first.E, evals = perCell;

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
