'use strict';

// Credible-region contours for the pair-plot 2D scatters (render-density.ts).
//
// Given the points already drawn in a scatter cell, estimate a 2D density on a
// regular grid, then extract the iso-density levels that enclose a target
// fraction of the total mass (highest-posterior-density regions). The classic
// corner-plot reading: the inner curve bounds the 68% credible region, the
// outer the 95%.
//
// Why histogram the plotted points (not the full weighted sample): the scatter
// cell already resolves weighting once — a weighted posterior is drawn as an
// importance-resampled uniform-weight subset (render-density.ts), an unweighted
// measure as an even stride. Binning that same subset keeps the contour and the
// cloud in exact agreement and shares their axis frame, so the overlaid `lines`
// series never pushes echarts' auto-range around.
//
// No stdlib pull-in — all math is JS-native — so this can run on the main
// thread beside the chart build.

/** One credible level: the fraction of mass it encloses plus the polyline
 *  segments (pairs of [x, y] endpoints in data coordinates) tracing it. */
export interface ContourLevel {
  frac: number;
  segments: Array<[[number, number], [number, number]]>;
}

/**
 * Density contours enclosing each fraction in `fracs` (e.g. [0.68, 0.95]).
 *
 * @param xs   x coordinate per point
 * @param ys   y coordinate per point (same length as xs)
 * @param fracs target enclosed-mass fractions, each in (0, 1)
 * @param opts.grid     grid resolution per axis (default 48)
 * @param opts.smooth   Gaussian smoothing sigma in bin units (default 1.0)
 * @param opts.minPoints below this many points the estimate is too noisy to be
 *                       meaningful; returns [] (default 30)
 * @returns one ContourLevel per requested fraction, in the SAME order; a level
 *          with no crossing (degenerate data) yields empty `segments`.
 */
export function densityContours(
  xs: ArrayLike<number>,
  ys: ArrayLike<number>,
  fracs: number[],
  opts: { grid?: number; smooth?: number; minPoints?: number } = {},
): ContourLevel[] {
  const n = Math.min(xs.length, ys.length);
  const minPoints = opts.minPoints != null ? opts.minPoints : 30;
  const empty = fracs.map(function (f) { return { frac: f, segments: [] as ContourLevel['segments'] }; });
  if (n < minPoints) return empty;

  // Data range. A degenerate (zero-width) axis has no 2D density to contour.
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  if (!(xmax > xmin) || !(ymax > ymin)) return empty;

  const G = Math.max(8, opts.grid != null ? opts.grid : 48);
  const dx = (xmax - xmin) / G;
  const dy = (ymax - ymin) / G;

  // Bin counts. Node j (0..G-1) is the CENTRE of bin j, at xmin+(j+0.5)*dx.
  const grid = new Float64Array(G * G);
  let total = 0;
  for (let i = 0; i < n; i++) {
    const x = xs[i], y = ys[i];
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    let bx = Math.floor((x - xmin) / dx);
    let by = Math.floor((y - ymin) / dy);
    if (bx < 0) bx = 0; else if (bx >= G) bx = G - 1;
    if (by < 0) by = 0; else if (by >= G) by = G - 1;
    grid[by * G + bx] += 1;
    total += 1;
  }
  if (total <= 0) return empty;

  const sigma = opts.smooth != null ? opts.smooth : 1.0;
  const field = sigma > 0 ? gaussianBlur(grid, G, sigma) : grid;

  // Node coordinates (bin centres).
  const gx = new Float64Array(G);
  const gy = new Float64Array(G);
  for (let j = 0; j < G; j++) { gx[j] = xmin + (j + 0.5) * dx; gy[j] = ymin + (j + 0.5) * dy; }

  return fracs.map(function (frac) {
    const level = hpdLevel(field, frac);
    const segments = level > 0 ? marchingSquares(field, G, gx, gy, level) : [];
    return { frac: frac, segments: segments };
  });
}

/** Separable Gaussian blur with reflect-at-edge padding. Mass-preserving
 *  (normalised kernel), so the total under `field` matches the input. */
function gaussianBlur(src: Float64Array, G: number, sigma: number): Float64Array {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const kernel = new Float64Array(2 * radius + 1);
  let ksum = 0;
  for (let k = -radius; k <= radius; k++) {
    const w = Math.exp(-(k * k) / (2 * sigma * sigma));
    kernel[k + radius] = w;
    ksum += w;
  }
  for (let k = 0; k < kernel.length; k++) kernel[k] /= ksum;

  const tmp = new Float64Array(G * G);
  const out = new Float64Array(G * G);
  // Horizontal pass.
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let cc = c + k;
        if (cc < 0) cc = -cc - 1; else if (cc >= G) cc = 2 * G - cc - 1;  // reflect
        if (cc < 0) cc = 0; else if (cc >= G) cc = G - 1;
        acc += src[r * G + cc] * kernel[k + radius];
      }
      tmp[r * G + c] = acc;
    }
  }
  // Vertical pass.
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let rr = r + k;
        if (rr < 0) rr = -rr - 1; else if (rr >= G) rr = 2 * G - rr - 1;  // reflect
        if (rr < 0) rr = 0; else if (rr >= G) rr = G - 1;
        acc += tmp[rr * G + c] * kernel[k + radius];
      }
      out[r * G + c] = acc;
    }
  }
  return out;
}

/** Highest-posterior-density threshold: the field value `t` such that the mass
 *  in cells with value ≥ t is `frac` of the total. Sort cell values descending,
 *  accumulate until the target fraction is reached. */
function hpdLevel(field: Float64Array, frac: number): number {
  let total = 0;
  for (let i = 0; i < field.length; i++) total += field[i];
  if (!(total > 0)) return 0;
  const sorted = Float64Array.from(field);
  sorted.sort();  // ascending
  const target = frac * total;
  let acc = 0;
  // Walk from the top (densest) down until the accumulated mass reaches target.
  for (let i = sorted.length - 1; i >= 0; i--) {
    acc += sorted[i];
    if (acc >= target) return sorted[i];
  }
  return sorted[0];
}

// Marching-squares edge lookup: for each of the 16 corner-sign cases, the pairs
// of cell edges the iso-line crosses. Edges are indexed 0=bottom,1=right,2=top,
// 3=left. Corners are 0=BL,1=BR,2=TR,3=TL (CCW). Ambiguous saddles (5, 10) emit
// both crossings; for a smooth density the visual cost of not disambiguating is
// negligible.
const MS_EDGES: number[][] = [
  [],            // 0
  [3, 0],        // 1  BL
  [0, 1],        // 2  BR
  [3, 1],        // 3  BL BR
  [1, 2],        // 4  TR
  [3, 0, 1, 2],  // 5  BL TR (saddle)
  [0, 2],        // 6  BR TR
  [3, 2],        // 7  BL BR TR
  [2, 3],        // 8  TL
  [2, 0],        // 9  BL TL
  [0, 1, 2, 3],  // 10 BR TL (saddle)
  [2, 1],        // 11 BL BR TL
  [1, 3],        // 12 TR TL
  [1, 0],        // 13 BL TR TL
  [0, 3],        // 14 BR TR TL
  [],            // 15
];

/** Extract iso-`level` line segments from a scalar field sampled at grid nodes
 *  (gx × gy coordinates), via marching squares with linear edge interpolation.
 *  Returns segments as endpoint pairs in data coordinates. */
function marchingSquares(
  field: Float64Array, G: number, gx: Float64Array, gy: Float64Array, level: number,
): Array<[[number, number], [number, number]]> {
  const segs: Array<[[number, number], [number, number]]> = [];

  // Interpolate the crossing point on the edge between two nodes.
  function interp(v0: number, v1: number, a: number, b: number): number {
    const d = v1 - v0;
    if (d === 0) return a;
    let t = (level - v0) / d;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    return a + t * (b - a);
  }

  for (let r = 0; r < G - 1; r++) {
    for (let c = 0; c < G - 1; c++) {
      const vBL = field[r * G + c];
      const vBR = field[r * G + c + 1];
      const vTR = field[(r + 1) * G + c + 1];
      const vTL = field[(r + 1) * G + c];
      let idx = 0;
      if (vBL >= level) idx |= 1;
      if (vBR >= level) idx |= 2;
      if (vTR >= level) idx |= 4;
      if (vTL >= level) idx |= 8;
      const edges = MS_EDGES[idx];
      if (edges.length === 0) continue;

      const x0 = gx[c], x1 = gx[c + 1];
      const y0 = gy[r], y1 = gy[r + 1];
      // Point on a given edge (0=bottom,1=right,2=top,3=left).
      function edgePt(e: number): [number, number] {
        switch (e) {
          case 0: return [interp(vBL, vBR, x0, x1), y0];           // bottom
          case 1: return [x1, interp(vBR, vTR, y0, y1)];           // right
          case 2: return [interp(vTL, vTR, x0, x1), y1];           // top
          default: return [x0, interp(vBL, vTL, y0, y1)];          // left (3)
        }
      }
      for (let e = 0; e + 1 < edges.length; e += 2) {
        segs.push([edgePt(edges[e]), edgePt(edges[e + 1])]);
      }
    }
  }
  return segs;
}
