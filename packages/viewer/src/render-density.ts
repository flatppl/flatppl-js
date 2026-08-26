// @flatppl/viewer — density-strip + corner-grid renderers —
//
// renderDensityStrips lays out one density column per axis (the
// marginals view); renderCornerGrid is the NxN matrix view
// (marginals on the diagonal, joint scatters off-diagonal).

import { colorForBinding } from './palette.js';
import { densityContours, fieldContours } from './contour2d.js';
import { jointDensityField, jointMeanField, renderField2D } from './render-field.js';
/**
 * Render the selected axes as a 2D density-strip view: one
 * column per axis, where each column shades by the per-axis
 * marginal density along y. Useful for array-shaped data where
 * each index slot has its own marginal — corner plots scale
 * O(N²) cells, this scales O(N).
 *
 * Implementation: per axis, compute an FD histogram. For each
 * bin in that axis, draw a horizontal rect spanning the column
 * width with opacity proportional to bin density (relative to
 * the axis's max). The y-axis is shared across all columns
 * (global value range) so columns are visually comparable.
 *
 * ECharts has a heatmap series that could do this for free,
 * but the value-axis variant restricts to a regular grid; we
 * want each axis's bin grid to align to its own FD-derived
 * edges. Custom render gives that flexibility cheaply.
 */
import { nameSeed } from './orchestration.js';
import { listScalarAxes, makeMainThreadPrng } from './util.js';
import { formatScalar } from './util.js';
import type { Ctx } from './types';
export function renderDensityStrips(ctx: Ctx, hostEl: any, measure: any, bindingName: any, axesArg: any) {
  hostEl.innerHTML = '';
  // Marginals mode passes the full axis list (no selection cap); we
  // fall back to listScalarAxes for legacy callers.
  const axes = axesArg || listScalarAxes(measure);
  const n = axes.length;
  if (n === 0) {
    const empty = document.createElement('div');
    empty.textContent = 'No scalar axes to plot.';
    empty.style.opacity = '0.5';
    empty.style.padding = '24px';
    empty.style.textAlign = 'center';
    hostEl.appendChild(empty);
    return;
  }

  const fg = getComputedStyle(document.body).color || '#ccc';
  const color = colorForBinding(ctx, bindingName);
  const logWeights = measure.logWeights;
  const histOptsBase = logWeights ? { logWeights: logWeights } : {};

  // Per-axis FD histograms + global y range.
  const hists = axes.map(function(a: any) {
    return FlatPPLEngine.histogram.freedmanDiaconisHistogram(a.samples, histOptsBase);
  });
  // Per-axis peak densities: a tightly-concentrated marginal has
  // a much higher per-bin density than a broad one (Σ density ×
  // binwidth ≈ 1 in either case), so a single global peak makes
  // broad columns look near-empty. Normalising each column to
  // its own peak keeps every column's shape readable.
  let yMin = Infinity, yMax = -Infinity;
  const peakDensities = new Array(hists.length);
  for (let i = 0; i < hists.length; i++) {
    const h = hists[i];
    peakDensities[i] = 0;
    if (!h.binEdges || h.binEdges.length === 0) continue;
    if (h.binEdges[0] < yMin) yMin = h.binEdges[0];
    if (h.binEdges[h.binEdges.length - 1] > yMax) yMax = h.binEdges[h.binEdges.length - 1];
    for (let j = 0; j < h.ys.length; j++) {
      if (h.ys[j] > peakDensities[i]) peakDensities[i] = h.ys[j];
    }
  }
  if (!isFinite(yMin) || !isFinite(yMax) || yMin === yMax) {
    const info = document.createElement('div');
    info.textContent = 'No variation across selected axes.';
    info.style.opacity = '0.5';
    info.style.padding = '24px';
    info.style.textAlign = 'center';
    hostEl.appendChild(info);
    return;
  }

  // One echarts instance hosts the whole strip view. Categories on
  // x (one per axis); value y (continuous, shared range). Bins
  // rendered as semi-transparent rects via custom series.
  hostEl.style.display = '';
  hostEl.style.gridTemplateColumns = '';
  hostEl.style.gridTemplateRows = '';
  hostEl.innerHTML = '';
  const chartDiv = document.createElement('div');
  chartDiv.style.width = '100%';
  chartDiv.style.height = '100%';
  hostEl.appendChild(chartDiv);

  // Group adjacent axes that belong to the same parent (an iid
  // array's slots all share the prefix before "[k]"). Within a
  // group, columns render edge-to-edge so obs[1]..obs[10] reads
  // as a continuous shaded sequence. Between groups, the
  // boundary cells get an inset on the gap side so a small
  // visible gap separates groups (a full empty slot was too
  // wide). axisGroup extracts the prefix before any trailing
  // "[…]" — same-group axes share that prefix.
  function axisGroup(label: any) {
    const i = label.lastIndexOf('[');
    return i >= 0 ? label.slice(0, i) : label;
  }
  const groups = axes.map(function(a: any) { return axisGroup(a.label); });
  // Per-axis gap flags. Boundary cells (group differs from
  // neighbour) shrink on the gap side; the renderer reads these
  // off the rect data. GAP_FRACTION is the inset depth as a
  // fraction of bandSize — 0.18 gives a ~36% combined visible
  // gap between groups (much tighter than a full empty slot)
  // while still leaving the bulk of the column for the data.
  const GAP_FRACTION = 0.18;
  // Build the rect data: one entry per (axis_idx, bin) pair.
  // Each entry carries [axis_idx, bin_y_center, density] plus
  // the bin's [lo, hi] and per-side gap insets.
  const data: Array<{ value: number[]; edges: number[]; gapLeft: boolean; gapRight: boolean }> = [];
  for (let ai2 = 0; ai2 < hists.length; ai2++) {
    const hh = hists[ai2];
    const gapLeft  = (ai2 > 0)             && groups[ai2] !== groups[ai2 - 1];
    const gapRight = (ai2 < axes.length-1) && groups[ai2] !== groups[ai2 + 1];
    for (let bi = 0; bi < hh.ys.length; bi++) {
      data.push({
        value: [ai2, hh.xs[bi], hh.ys[bi]],
        edges: [hh.binEdges[bi], hh.binEdges[bi + 1]],
        gapLeft: gapLeft, gapRight: gapRight,
      });
    }
  }
  const catLabels = axes.map(function(a: any) { return a.label; });
  const seriesColor = color;
  const ec = echarts.init(chartDiv);
  ec.setOption({
    backgroundColor: 'transparent',
    animation: false,
    grid: { left: 60, right: 25, top: 10, bottom: 60, containLabel: false },
    xAxis: {
      type: 'category',
      data: catLabels,
      axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
      axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
      axisLabel: {
        color: fg, opacity: 0.7, fontSize: 11,
        interval: 0,
        rotate: axes.length > 8 ? 60 : 0,
        fontFamily: 'var(--vscode-editor-font-family, monospace)',
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value', scale: true,
      min: yMin, max: yMax,
      axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
      axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
      axisLabel: { color: fg, opacity: 0.6, fontSize: 11, formatter: formatScalar },
      splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
    },
    series: [{
      type: 'custom',
      data: data,
      renderItem: function(_p: any, api: any) {
        const d = data[_p.dataIndex];
        const density = d.value[2];
        // Column-centred horizontal extent: ~70% of slot width.
        const cx = api.coord([d.value[0], (d.edges[0] + d.edges[1]) / 2]);
        const top = api.coord([d.value[0], d.edges[1]]);
        const bot = api.coord([d.value[0], d.edges[0]]);
        // ECharts category axes give bandWidth via api.size.
        // Use the full band width so adjacent columns share
        // their edges — the marginal strips read as a continuous
        // shaded sequence within a group. At group boundaries
        // (gapLeft / gapRight set on the rect data) we inset the
        // boundary cell by GAP_FRACTION of bandSize on the gap
        // side, so different-prefix axes get a small visible
        // separator without a full empty column.
        const bandSize = api.size([1, 0])[0];
        const leftEdge  = cx[0] - bandSize * 0.5
                      + (d.gapLeft  ? bandSize * GAP_FRACTION : 0);
        const rightEdge = cx[0] + bandSize * 0.5
                      - (d.gapRight ? bandSize * GAP_FRACTION : 0);
        // Opacity scales linearly with density relative to the
        // PER-AXIS peak — concentrated and broad marginals both
        // read at full intensity at their mode rather than the
        // broad ones fading under a tightly-concentrated peer.
        const axisPeak = peakDensities[d.value[0]];
        const opacity = axisPeak > 0
          ? Math.max(0.04, 0.85 * density / axisPeak)
          : 0;
        return {
          type: 'rect',
          shape: {
            x: leftEdge,
            y: top[1],
            width: rightEdge - leftEdge,
            height: bot[1] - top[1],
          },
          style: api.style({ fill: seriesColor, opacity: opacity, stroke: 'none' }),
        };
      },
      encode: { x: 0, y: 1 },
    }],
  });
}

/**
 * Build the corner-plot grid (diagonal marginals + below-diagonal
 * scatters) for the currently-selected axes. ctx.host is the parent
 * div whose contents we replace; it must be a flex/block child
 * with a fixed height so the inner grid expands correctly.
 */
export function renderCornerGrid(ctx: Ctx, hostEl: any, measure: any, bindingName: any) {
  hostEl.innerHTML = '';
  const axes = listScalarAxes(measure)
    .filter(function(a: any) { return ctx.recordSelection!.selected.indexOf(a.key) >= 0; });
  const n = axes.length;
  if (n === 0) {
    const empty = document.createElement('div');
    empty.textContent = 'Select at least one axis to plot.';
    empty.style.opacity = '0.5';
    empty.style.padding = '24px';
    empty.style.textAlign = 'center';
    hostEl.appendChild(empty);
    return;
  }

  const fg = getComputedStyle(document.body).color || '#ccc';
  const color = colorForBinding(ctx, bindingName);
  const logWeights = measure.logWeights;
  const histOptsBase = logWeights ? { logWeights: logWeights } : {};

  // Grid layout with two extra tracks: a leftmost column for
  // vertical y-axis labels (one per plot row), and a bottom row
  // for horizontal x-axis labels (one per plot column). The y
  // labels' track is auto-sized to the label width; the x
  // labels' track to the label height.
  //
  //   col:    [auto] [1fr] [1fr] ... [1fr]       n+1 columns
  //   row:    [1fr]                              n rows of plots
  //           [1fr]
  //           ...
  //           [auto]                             1 row of x labels
  hostEl.style.display = 'grid';
  hostEl.style.gridTemplateColumns = 'auto repeat(' + n + ', 1fr)';
  hostEl.style.gridTemplateRows    = 'repeat(' + n + ', 1fr) auto';
  hostEl.style.gap = '6px';
  hostEl.style.minHeight = '0';

  // ---- y-axis labels (left column, vertical) -----------------
  // Each label sits in column 1 (the auto-sized leftmost
  // track) at row r+1, naming the variable on the y-axis of
  // every cell in that row. Rotated 90deg counterclockwise so
  // it reads bottom-up.
  for (let yi = 0; yi < n; yi++) {
    const ylab = document.createElement('div');
    ylab.textContent = axes[yi].label;
    ylab.style.gridColumn = '1 / span 1';
    ylab.style.gridRow = (yi + 1) + ' / span 1';
    ylab.style.writingMode = 'vertical-rl';
    ylab.style.transform = 'rotate(180deg)';
    ylab.style.display = 'flex';
    ylab.style.alignItems = 'center';
    ylab.style.justifyContent = 'center';
    ylab.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
    ylab.style.fontSize = '0.92em';
    ylab.style.opacity = '0.85';
    ylab.style.padding = '0 0.3em';
    hostEl.appendChild(ylab);
  }

  // ---- x-axis labels (bottom row, horizontal) ----------------
  // Each label sits in row n+1 (the auto-sized bottom track)
  // at column c+2 (skipping the leftmost y-label track),
  // naming the variable on the x-axis of every cell in that
  // column.
  for (let xi = 0; xi < n; xi++) {
    const xlab = document.createElement('div');
    xlab.textContent = axes[xi].label;
    xlab.style.gridColumn = (xi + 2) + ' / span 1';
    xlab.style.gridRow = (n + 1) + ' / span 1';
    xlab.style.textAlign = 'center';
    xlab.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
    xlab.style.fontSize = '0.92em';
    xlab.style.opacity = '0.85';
    xlab.style.padding = '0.2em 0';
    hostEl.appendChild(xlab);
  }

  // Per-cell builder: chart container only — no internal label,
  // axis names live on the grid edges. (Plot row r, plot col c
  // → grid row r+1, grid col c+2 because of the two label tracks.)
  function makeCell(row: any, col: any) {
    const cell = document.createElement('div');
    cell.style.gridRow    = (row + 1) + ' / span 1';
    cell.style.gridColumn = (col + 2) + ' / span 1';
    cell.style.minHeight = '0';
    cell.style.minWidth  = '0';
    cell.style.background = 'rgba(255,255,255,0.02)';
    cell.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.08))';
    cell.style.borderRadius = '3px';
    hostEl.appendChild(cell);
    return cell;
  }

  // ---- Diagonals: 1D marginals --------------------------------
  // Each cell delegates to a helper so the renderItem closure
  // captures *that call's* `rects` / `color` parameters by name
  // — not a `var`-hoisted loop variable that gets overwritten on
  // the next iteration. (The bug: var seriesRects/seriesColor
  // are function-scoped, so all closures end up reading the LAST
  // iteration's values; the first paint looks correct because
  // setOption renders synchronously, but resize-triggered
  // re-renders pick up the wrong data and the upper cells go
  // blank.)
  function renderDiagonalCell(inner: any, rects: any, color: any) {
    const ec1 = echarts.init(inner);
    ec1.setOption({
      backgroundColor: 'transparent',
      animation: false,
      grid: { left: 50, right: 12, top: 6, bottom: 24, containLabel: false },
      xAxis: {
        type: 'value', scale: true,
        axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
        axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
        axisLabel: { color: fg, opacity: 0.6, fontSize: 10, formatter: formatScalar },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value', scale: true,
        axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
        axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
        axisLabel: { color: fg, opacity: 0.5, fontSize: 10, formatter: formatScalar },
        splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
      },
      series: [{
        type: 'custom',
        data: rects,
        renderItem: function(_p: any, api: any) {
          const d = rects[_p.dataIndex];
          const lt = api.coord([d.x0, d.value[1]]);
          const rb = api.coord([d.x1, 0]);
          return {
            type: 'rect',
            shape: { x: lt[0], y: lt[1], width: rb[0] - lt[0], height: rb[1] - lt[1] },
            style: api.style({ fill: color, opacity: 0.55, stroke: color, lineWidth: 0.5 }),
          };
        },
        encode: { x: 0, y: 1 },
      }],
    });
  }
  for (let i = 0; i < n; i++) {
    const samples = axes[i].samples;
    const inner = makeCell(i, i);
    const hist = FlatPPLEngine.histogram.freedmanDiaconisHistogram(samples, histOptsBase);
    const rects: Array<{ value: number[]; x0: number; x1: number }> = [];
    for (let k = 0; k < hist.xs.length; k++) {
      rects.push({
        value: [hist.xs[k], hist.ys[k]],
        x0: hist.binEdges[k],
        x1: hist.binEdges[k + 1],
      });
    }
    renderDiagonalCell(inner, rects, color);
  }

  if (n < 2) return;   // single-field record: only the diagonal

  // ---- Below-diagonal: 2D joint scatters ----------------------
  // Subsample if N is large enough that overplotting kills
  // readability. ECharts handles 50k points fine; 100k starts to
  // chug.
  //
  // Two paths:
  //   * Unweighted measure: take an even slice (deterministic
  //     stride) so the visual is stable across re-renders.
  //   * Weighted measure (e.g. an importance-weighted posterior
  //     from bayesupdate): plain stride would render the
  //     *prior's* atom positions with no regard for weights —
  //     the resulting scatter looks like the prior, even though
  //     the diagonal histograms (which use logWeights) correctly
  //     show the posterior. Fix: importance-resample atom
  //     indices via systematicResample, producing a uniform-
  //     weight subset that visualises the actual posterior.
  //     Per-binding seeded PRNG keeps the resample deterministic
  //     across re-renders.
  const anyN = axes[0].samples.length;
  const maxPoints = 20000;
  let indices;
  if (measure.logWeights) {
    const nOut = Math.min(anyN, maxPoints);
    const rsPrng = makeMainThreadPrng(nameSeed(ctx, bindingName + ':scatter'));
    indices = FlatPPLEngine.empirical.systematicResample(
      measure.logWeights, nOut, rsPrng);
  } else {
    const stride = anyN > maxPoints ? Math.ceil(anyN / maxPoints) : 1;
    const len = Math.ceil(anyN / stride);
    indices = new Int32Array(len);
    for (let ii = 0, jj = 0; ii < anyN; ii += stride, jj++) indices[jj] = ii;
  }
  // Pre-build one positional list per axis to avoid repeated
  // .samples lookups in the inner loop below.
  const cols = axes.map(function(a) { return a.samples; });

  for (let row = 1; row < n; row++) {
    for (let col = 0; col < row; col++) {
      const xCol = cols[col], yCol = cols[row];
      const inner2 = makeCell(row, col);
      const pts = new Array(indices.length);
      for (let p = 0; p < indices.length; p++) {
        const idx = indices[p];
        pts[p] = [xCol[idx], yCol[idx]];
      }
      // Point opacity scales with point count — denser data
      // gets more transparency so clouds don't saturate.
      const alpha = Math.max(0.05, Math.min(0.6, 800 / pts.length));
      // Credible-region contours over the SAME plotted points, so the curves
      // share the scatter's axis frame (the overlaid `lines` series stays
      // inside the cloud's data range and never perturbs echarts' auto-scale).
      // Inner curve = 68% HPD region, outer = 95% (corner-plot convention).
      const cxs = new Float64Array(pts.length);
      const cys = new Float64Array(pts.length);
      for (let p = 0; p < pts.length; p++) { cxs[p] = pts[p][0]; cys[p] = pts[p][1]; }
      const contourLevels = densityContours(cxs, cys, [0.68, 0.95]);
      const contourSeries = contourLevels
        .filter(function (lvl) { return lvl.segments.length > 0; })
        .map(function (lvl) {
          const inner = lvl.frac <= 0.7;
          // Draw in the theme foreground (near-white on dark, near-black on
          // light) so the curves read over the same-hued point cloud; the outer
          // (95%) level is dashed and fainter to separate it from the inner 68%.
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
      const ec2 = echarts.init(inner2);
      ec2.setOption({
        backgroundColor: 'transparent',
        animation: false,
        grid: { left: 50, right: 12, top: 6, bottom: 24, containLabel: false },
        xAxis: {
          type: 'value', scale: true,
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, fontSize: 10, formatter: formatScalar },
          splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
        },
        yAxis: {
          type: 'value', scale: true,
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.5, fontSize: 10, formatter: formatScalar },
          splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
        },
        series: [{
          type: 'scatter',
          data: pts,
          symbolSize: 3,
          large: true,
          largeThreshold: 2000,
          itemStyle: { color: color, opacity: alpha },
        }].concat(contourSeries as any[]),
      });
    }
  }
  // Above-diagonal cells are intentionally left empty (corner-
  // plot convention).
}

/** Credible levels the joint surface overlays, inner first. */
const JOINT_FRACS = [0.68, 0.95];

/**
 * Joint 2D view: the selected axes' joint density as a value-axis surface,
 * with a third selected axis optionally taking over the colour.
 *
 * Roles are positional in selection order — `selected[0]` → x, `[1]` → y,
 * `[2]` → colour — so the Variates dropdown doubles as the role picker and no
 * second control is needed. `onRotate` re-renders the chart after the rotate
 * buttons permute those roles.
 */
export function renderJointField(ctx: Ctx, hostEl: any, measure: any, onRotate: () => void) {
  hostEl.innerHTML = '';
  const byKey: Record<string, any> = {};
  listScalarAxes(measure).forEach(function (a: any) { byKey[a.key] = a; });
  // Selection ORDER, not axis order — that ordering is the role assignment.
  const picked = ctx.recordSelection!.selected
    .map(function (k: string) { return byKey[k]; })
    .filter(function (a: any) { return !!a; });

  function message(text: string) {
    const div = document.createElement('div');
    div.textContent = text;
    div.style.opacity = '0.5';
    div.style.padding = '24px';
    div.style.textAlign = 'center';
    hostEl.appendChild(div);
  }
  if (picked.length < 2) {
    message('Select at least two variates to plot a joint surface.');
    return;
  }
  const roleCount = Math.min(3, picked.length);
  const xAxis = picked[0], yAxis = picked[1];
  const cAxis = roleCount >= 3 ? picked[2] : null;

  // Bin with weights rather than importance-resampling. A surface needs no
  // equal-weight subset, and binning uses every atom — deliberately unlike the
  // corner-plot scatters above, which resample because a scatter can only draw
  // one point per atom.
  let weights: Float64Array | null = null;
  if (measure.logWeights) {
    const lw = measure.logWeights;
    let maxLw = -Infinity;
    for (let i = 0; i < lw.length; i++) if (lw[i] > maxLw) maxLw = lw[i];
    weights = new Float64Array(lw.length);
    // Shift by the max before exponentiating: raw logWeights can sit far
    // enough below 0 that every exp() underflows to zero.
    if (Number.isFinite(maxLw)) {
      for (let i = 0; i < lw.length; i++) weights[i] = Math.exp(lw[i] - maxLw);
    }
  }

  const density = jointDensityField(xAxis.samples, yAxis.samples, weights);
  if (!density) {
    message('Not enough variation in the selected variates to bin a surface.');
    return;
  }
  const surface = cAxis
    ? jointMeanField(xAxis.samples, yAxis.samples, cAxis.samples, weights)
    : density;
  if (!surface) {
    message('Not enough variation in the selected variates to bin a surface.');
    return;
  }

  // ---- Role legend + rotate buttons ----
  hostEl.style.display = 'flex';
  hostEl.style.flexDirection = 'column';
  hostEl.style.minHeight = '0';

  const legend = document.createElement('div');
  legend.style.display = 'flex';
  legend.style.alignItems = 'center';
  legend.style.gap = '1.2em';
  legend.style.padding = '0.2em 0.4em 0.4em';
  legend.style.fontSize = '0.92em';
  legend.style.flex = '0 0 auto';

  const roles = document.createElement('span');
  roles.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
  roles.style.opacity = '0.85';
  roles.textContent = 'x: ' + xAxis.label + '   y: ' + yAxis.label
    + (cAxis ? '   colour: ' + cAxis.label : '');
  roles.title = cAxis
    ? 'Colour is the weighted mean of ' + cAxis.label + ' per bin.\n'
      + 'The contours stay the 68% / 95% credible regions of the '
      + xAxis.label + ' / ' + yAxis.label + ' DENSITY, not of the colour.'
    : 'Colour is the joint density of ' + xAxis.label + ' and ' + yAxis.label
      + '.\nContours bound the 68% and 95% credible regions.';
  legend.appendChild(roles);

  function rotateBtn(glyph: string, title: string, step: number) {
    const b = document.createElement('button');
    b.textContent = glyph;
    b.title = title;
    b.style.cursor = 'pointer';
    b.style.fontSize = '0.95em';
    b.style.lineHeight = '1';
    b.style.padding = '0.15em 0.45em';
    b.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
    b.style.borderRadius = '3px';
    b.style.background = 'var(--vscode-button-secondaryBackground, #3a3d41)';
    b.style.color = 'var(--vscode-button-secondaryForeground, #ccc)';
    b.addEventListener('click', function (ev) {
      ev.stopPropagation();
      // Permute only the role-bearing prefix; any further checked axes keep
      // their place in the queue.
      const sel = ctx.recordSelection!.selected;
      const head = sel.slice(0, roleCount);
      const shift = ((step % roleCount) + roleCount) % roleCount;
      const rotated = head.slice(shift).concat(head.slice(0, shift));
      for (let i = 0; i < roleCount; i++) sel[i] = rotated[i];
      onRotate();
    });
    return b;
  }
  const btns = document.createElement('span');
  btns.style.display = 'inline-flex';
  btns.style.gap = '0.3em';
  btns.appendChild(rotateBtn('↑', 'Rotate roles: y becomes x', 1));
  btns.appendChild(rotateBtn('↓', 'Rotate roles: x becomes y', -1));
  legend.appendChild(btns);

  if (ctx.recordSelection!.selected.length > 3) {
    const note = document.createElement('span');
    note.textContent = 'Joint 2D uses the first three axes';
    note.style.opacity = '0.6';
    legend.appendChild(note);
  }
  hostEl.appendChild(legend);

  const chartHost = document.createElement('div');
  chartHost.style.flex = '1 1 auto';
  chartHost.style.minHeight = '0';
  hostEl.appendChild(chartHost);

  // Contours always describe the (x, y) DENSITY — with a third axis on the
  // colour the surface no longer shows it. Taken off the density field's own
  // grid, so they are its exact iso-levels (and weight-aware, unlike
  // densityContours' re-bin of the raw points).
  const contours = fieldContours(density.z, density.xs, density.ys, JOINT_FRACS);
  renderField2D(chartHost, surface, {
    xLabel: xAxis.label,
    yLabel: yAxis.label,
    zLabel: cAxis ? cAxis.label : 'density',
    contours: contours,
    diverging: !!cAxis,
    robustRamp: !!cAxis,
  });
}
