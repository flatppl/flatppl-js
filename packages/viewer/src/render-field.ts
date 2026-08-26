// @flatppl/viewer — value-axis 2-D field renderer —
//
// renderField2D paints a scalar field sampled on a regular grid over two
// CONTINUOUS axes, with a colour bar and optional contour overlay. Distinct
// from renderMatrixHeatmap (render-samples.ts), which is a category-axis
// matrix display with 1-indexed row/col labels and y-inversion — that one
// cannot carry data coordinates, so it is not reusable here.
//
// jointDensityField / jointMeanField build the field from a sampled measure's
// columns. They are DOM-free so they can be tested headless.

import { binGrid, gaussianBlur } from './contour2d.js';
import type { ContourLevel } from './contour2d.js';
import { formatScalar } from './util.js';

/** A scalar field on a regular grid. `z` is row-major (`row * xs.length + col`);
 *  a NaN entry means "no data here" and renders transparent. */
export interface Field2D {
  /** x bin centres. */
  xs: Float64Array;
  /** y bin centres. */
  ys: Float64Array;
  /** Row-major field values, length ys.length * xs.length. */
  z: Float64Array;
}

/** Below this many points a binned surface is noise, not a picture. Same floor
 *  densityContours uses. */
const MIN_POINTS = 30;

/** Bins per axis. Kept well under the point count so most bins are occupied,
 *  and capped so a 10^5-atom posterior does not paint 10^4 rects. */
function gridSize(n: number): number {
  return Math.max(16, Math.min(64, Math.ceil(Math.sqrt(n / 10))));
}

const SMOOTH_SIGMA = 0.8;

/** Per-axis mass trimmed off each end before binning. */
const CLIP_TAIL = 0.005;

/** Weighted quantile of `v` at fraction `q`, ignoring non-finite entries. */
function weightedQuantile(v: ArrayLike<number>, w: ArrayLike<number> | null | undefined, q: number, n: number): number {
  const idx: number[] = [];
  let total = 0;
  for (let i = 0; i < n; i++) {
    const wi = w ? w[i] : 1;
    if (!Number.isFinite(v[i]) || !Number.isFinite(wi) || wi <= 0) continue;
    idx.push(i);
    total += wi;
  }
  if (idx.length === 0 || !(total > 0)) return NaN;
  idx.sort(function (a, b) { return v[a] - v[b]; });
  const target = q * total;
  let acc = 0;
  for (let k = 0; k < idx.length; k++) {
    acc += w ? w[idx[k]] : 1;
    if (acc >= target) return v[idx[k]];
  }
  return v[idx[idx.length - 1]];
}

/**
 * The window the surface spans: the central 99% of each axis by WEIGHT.
 *
 * Not the full data range. A heavy-tailed axis (a half-Cauchy scale, say) has a
 * data range orders of magnitude wider than its mass, and binning over that
 * range crushes the whole posterior into one row of bins. Weighted quantiles
 * also mean an importance-weighted measure is framed on its POSTERIOR support
 * rather than on the prior atoms' spread.
 *
 * Returns null when either axis is degenerate over that window.
 */
function clipRange(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  w: ArrayLike<number> | null | undefined,
  n: number,
): [number, number, number, number] | null {
  const xlo = weightedQuantile(x, w, CLIP_TAIL, n);
  const xhi = weightedQuantile(x, w, 1 - CLIP_TAIL, n);
  const ylo = weightedQuantile(y, w, CLIP_TAIL, n);
  const yhi = weightedQuantile(y, w, 1 - CLIP_TAIL, n);
  if (!(xhi > xlo) || !(yhi > ylo)) return null;
  return [xlo, xhi, ylo, yhi];
}

/**
 * Weighted 2-D density of (x, y), normalised so Σ z·dx·dy = 1.
 *
 * @param weights per-point weight (linear, not log); omitted means equal weights.
 * @returns null when the sample cannot carry a surface — too few points, or a
 *          degenerate axis.
 */
export function jointDensityField(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  weights?: ArrayLike<number> | null,
): Field2D | null {
  const n = Math.min(x.length, y.length);
  if (n < MIN_POINTS) return null;
  const G = gridSize(n);
  const range = clipRange(x, y, weights, n);
  if (!range) return null;
  const binned = binGrid(x, y, G, weights, range);
  if (!binned || !(binned.total > 0)) return null;
  const smoothed = gaussianBlur(binned.field, G, SMOOTH_SIGMA);
  // Cell area from the bin centres (uniform by construction).
  const dx = G > 1 ? binned.gx[1] - binned.gx[0] : 1;
  const dy = G > 1 ? binned.gy[1] - binned.gy[0] : 1;
  // The blur is mass-preserving, so Σ smoothed === binned.total; dividing by
  // total·dx·dy turns bin mass into a probability density.
  const scale = 1 / (binned.total * dx * dy);
  const z = new Float64Array(smoothed.length);
  for (let i = 0; i < z.length; i++) z[i] = smoothed[i] * scale;
  return { xs: binned.gx, ys: binned.gy, z: z };
}

/**
 * Per-bin weighted mean of a third quantity over the (x, y) plane:
 * `Σ w·v / Σ w`. Empty bins come back NaN.
 *
 * Not smoothed. Blurring a mean would bleed occupied bins into empty ones and
 * invent values where the sample has none.
 */
export function jointMeanField(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  v: ArrayLike<number>,
  weights?: ArrayLike<number> | null,
): Field2D | null {
  const n = Math.min(x.length, y.length, v.length);
  if (n < MIN_POINTS) return null;
  const G = gridSize(n);
  const range = clipRange(x, y, weights, n);
  if (!range) return null;
  const wv = new Float64Array(n);
  for (let i = 0; i < n; i++) wv[i] = (weights ? weights[i] : 1) * v[i];
  // Same G and same range for both, so numerator and denominator bins align.
  const den = binGrid(x, y, G, weights, range);
  const num = binGrid(x, y, G, wv, range);
  if (!den || !num) return null;
  const z = new Float64Array(G * G);
  for (let i = 0; i < z.length; i++) {
    z[i] = den.field[i] > 0 ? num.field[i] / den.field[i] : NaN;
  }
  return { xs: den.gx, ys: den.gy, z: z };
}

export interface Field2DOptions {
  xLabel: string;
  yLabel: string;
  zLabel: string;
  /** Overlaid as `lines` series in the theme foreground. */
  contours?: ContourLevel[];
  /** Centre the colour ramp on 0 (for a signed quantity). */
  diverging?: boolean;
  /** Bound the ramp by the 2nd / 98th percentile of the occupied bins instead
   *  of their min and max. For a per-bin MEAN, where a sparse bin's outlier
   *  would otherwise own the whole scale. Leave off for a density, whose peak
   *  is the signal and must not be clipped. */
  robustRamp?: boolean;
  /** Extra text appended to the colour-bar tooltip. */
  zTooltip?: string;
}

/**
 * Paint `field` into `hostEl` as a value-axis surface.
 *
 * Drawn with a `custom` series rather than echarts' `heatmap`: on a value axis
 * heatmap sizes its cells from the axis band width, which is one DATA unit, so
 * cells only line up with the bins when the bin width happens to be 1. A
 * custom series places each rect on its own bin edges via api.coord.
 */
export function renderField2D(hostEl: any, field: Field2D, opts: Field2DOptions): void {
  hostEl.innerHTML = '';
  const fg = getComputedStyle(document.body).color || '#ccc';
  const nx = field.xs.length, ny = field.ys.length;
  const dx = nx > 1 ? field.xs[1] - field.xs[0] : 1;
  const dy = ny > 1 ? field.ys[1] - field.ys[0] : 1;

  // Skip NaN bins entirely — an omitted rect is the transparent "no data" cell,
  // and it also keeps the colour-bar range over real values only.
  const vals: number[] = [];
  const cells: Array<[number, number, number]> = [];
  for (let r = 0; r < ny; r++) {
    for (let c = 0; c < nx; c++) {
      const v = field.z[r * nx + c];
      if (!Number.isFinite(v)) continue;
      cells.push([field.xs[c], field.ys[r], v]);
      vals.push(v);
    }
  }
  vals.sort(function (a, b) { return a - b; });
  const q = (p: number) => vals[Math.min(vals.length - 1, Math.max(0, Math.round(p * (vals.length - 1))))];
  const tail = opts.robustRamp ? 0.02 : 0;
  let lo = vals.length ? q(tail) : Infinity;
  let hi = vals.length ? q(1 - tail) : -Infinity;
  // value[3] is the ramp's input, clamped into those bounds; value[2] stays the
  // true value so the tooltip never lies about a clipped cell.
  const data = cells.map(function (cell) {
    return { value: [cell[0], cell[1], cell[2], Math.min(hi, Math.max(lo, cell[2]))] };
  });
  if (data.length === 0) {
    const empty = document.createElement('div');
    empty.textContent = 'No occupied bins to plot.';
    empty.style.opacity = '0.5';
    empty.style.padding = '24px';
    empty.style.textAlign = 'center';
    hostEl.appendChild(empty);
    return;
  }
  if (lo === hi) { lo -= 0.5; hi += 0.5; }

  const chartDiv = document.createElement('div');
  chartDiv.style.width = '100%';
  chartDiv.style.height = '100%';
  hostEl.appendChild(chartDiv);

  // Same two ramps renderMatrixHeatmap picks between, so the two surfaces read
  // as one family.
  const diverging = !!opts.diverging && lo < 0 && hi > 0;
  const bound = Math.max(Math.abs(lo), Math.abs(hi));
  const visualMap = {
    type: 'continuous',
    dimension: 3,
    seriesIndex: 0,
    min: diverging ? -bound : lo,
    max: diverging ? bound : hi,
    calculable: true,
    orient: 'vertical',
    right: 6,
    top: 'center',
    itemHeight: 120,
    text: [opts.zLabel, ''],
    formatter: formatScalar,
    inRange: diverging
      ? { color: ['#3b6ad6', '#f4f4f4', '#d6543b'] }
      : { color: ['#1a1d2e', '#4b6aa8', '#a8c4e8'] },
    textStyle: { color: fg, fontSize: 10 },
  };

  const contourSeries = (opts.contours || [])
    .filter(function (lvl) { return lvl.segments.length > 0; })
    .map(function (lvl) {
      const inner = lvl.frac <= 0.7;
      return {
        type: 'lines',
        coordinateSystem: 'cartesian2d',
        data: lvl.segments.map(function (s) { return { coords: s }; }),
        silent: true,
        z: 5,
        lineStyle: {
          color: fg,
          width: inner ? 1.5 : 1.1,
          opacity: inner ? 0.85 : 0.55,
          type: inner ? 'solid' : 'dashed',
        },
      };
    });

  const axisCommon = {
    type: 'value',
    axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
    axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
    splitLine: { show: false },
    nameLocation: 'middle',
    nameTextStyle: { color: fg, fontSize: 11, fontFamily: 'var(--vscode-editor-font-family, monospace)' },
  };

  const ec = echarts.init(chartDiv);
  ec.setOption({
    backgroundColor: 'transparent',
    animation: false,
    grid: { left: 64, right: 96, top: 16, bottom: 52, containLabel: false },
    tooltip: {
      formatter: function (p: any) {
        return opts.xLabel + ': ' + formatScalar(p.value[0])
          + '<br/>' + opts.yLabel + ': ' + formatScalar(p.value[1])
          + '<br/>' + opts.zLabel + ': ' + formatScalar(p.value[2]);
      },
    },
    xAxis: Object.assign({}, axisCommon, {
      min: field.xs[0] - dx / 2,
      max: field.xs[nx - 1] + dx / 2,
      name: opts.xLabel, nameGap: 32,
      axisLabel: { color: fg, opacity: 0.6, fontSize: 10, formatter: formatScalar },
    }),
    yAxis: Object.assign({}, axisCommon, {
      min: field.ys[0] - dy / 2,
      max: field.ys[ny - 1] + dy / 2,
      name: opts.yLabel, nameGap: 46,
      axisLabel: { color: fg, opacity: 0.6, fontSize: 10, formatter: formatScalar },
    }),
    visualMap: visualMap,
    series: ([{
      type: 'custom',
      data: data,
      encode: { x: 0, y: 1, tooltip: [0, 1, 2] },
      renderItem: function (params: any, api: any) {
        const cx = api.value(0), cy = api.value(1);
        const lt = api.coord([cx - dx / 2, cy + dy / 2]);
        const rb = api.coord([cx + dx / 2, cy - dy / 2]);
        return {
          type: 'rect',
          shape: {
            // +1px so adjacent cells overlap rather than leaving seams at
            // fractional pixel boundaries.
            x: lt[0], y: lt[1],
            width: rb[0] - lt[0] + 1, height: rb[1] - lt[1] + 1,
          },
          style: { fill: api.visual('color') },
        };
      },
    }] as any[]).concat(contourSeries as any[]),
  });
}
