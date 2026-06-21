// @flatppl/viewer — BAT-like summary-statistics table view.
//
// A third record-measure view mode alongside corner (Correlations) and
// strips (Marginals): one row per scalar variate with weighted summary
// stats and an inline HTML histogram. Stats reuse engine primitives;
// the histogram cell is translated from LazyReports.jl (bars tinted where
// the mean/median fall), rendered as HTML <div> bars rather than SVG so the
// cell inherits surrounding text colour + theming.

import { listScalarAxes, esc } from './util.js';

declare const FlatPPLEngine: any;

// Bin cap for the inline ~120px histogram cell — coarse enough that bars
// stay legible (FD's natural count can reach 100+).
const TABLE_HIST_BINS = 32;

/** Index of the histogram bin containing value v, or -1 if outside support. */
function binIndexOf(hist: any, v: number): number {
  const edges = hist.binEdges;
  if (!edges || edges.length < 2 || !Number.isFinite(v)) return -1;
  if (v < edges[0] || v > edges[edges.length - 1]) return -1;
  let bin = Math.floor((v - edges[0]) / hist.binWidth);
  if (bin < 0) bin = 0;
  if (bin >= hist.ys.length) bin = hist.ys.length - 1;
  return bin;
}

/** Weighted per-variate summary stats. logWeights null/undefined → uniform. */
export function variateSummary(samples: Float64Array, logWeights: ArrayLike<number> | null | undefined) {
  const H = FlatPPLEngine.histogram;
  const n = samples.length;
  // Coarse bin cap: the inline cell is ~120px, so FD's natural bin count
  // (can be 40–100+) renders as unreadable hairlines. Cap at TABLE_HIST_BINS
  // for legible bars; also coarsens `mode` to the same resolution (fine for
  // a summary readout).
  const hist = H.freedmanDiaconisHistogram(samples,
    logWeights ? { logWeights, maxBins: TABLE_HIST_BINS } : { maxBins: TABLE_HIST_BINS });

  // Normalised weights (uniform 1/n when no logWeights).
  let w: Float64Array;
  if (logWeights) {
    w = H.normaliseWeights(logWeights);
  } else {
    w = new Float64Array(n);
    w.fill(n > 0 ? 1 / n : 0);
  }

  // Weighted mean + std.
  let mean = 0;
  for (let i = 0; i < n; i++) mean += w[i] * samples[i];
  let varSum = 0;
  for (let i = 0; i < n; i++) { const d = samples[i] - mean; varSum += w[i] * d * d; }
  const std = Math.sqrt(varSum);

  // Mode = centre of the tallest histogram bar.
  let mode = NaN;
  if (hist.ys && hist.ys.length > 0) {
    let best = 0;
    for (let i = 1; i < hist.ys.length; i++) if (hist.ys[i] > hist.ys[best]) best = i;
    mode = hist.xs[best];
  }

  // Weighted quantiles need samples sorted ascending with weights re-paired.
  const idx = Array.from({ length: n }, (_, i) => i);
  idx.sort((a, b) => samples[a] - samples[b]);
  const sortedSamples = new Float64Array(n);
  const sortedW = new Float64Array(n);
  for (let i = 0; i < n; i++) { sortedSamples[i] = samples[idx[i]]; sortedW[i] = w[idx[i]]; }
  const median = H.weightedQuantileSorted(sortedSamples, sortedW, 0.5);
  const q05 = H.weightedQuantileSorted(sortedSamples, sortedW, 0.05);
  const q95 = H.weightedQuantileSorted(sortedSamples, sortedW, 0.95);

  return { mean, std, mode, median, q05, q95, hist };
}

/** Inline HTML histogram: a flex row of <div> bars, each height = ys[i]/max(ys),
 *  tinted by where the mean/median fall (teal both, green mean, steelblue median).
 *  HTML (not SVG) so the cell inherits the surrounding text colour + theming and
 *  has no separate render namespace. */
export function inlineHistogramHtml(hist: any, marks: { meanValue: number; medianValue: number }): HTMLElement {
  const ys = hist.ys || new Float64Array(0);
  const nb = ys.length;
  const box = document.createElement('div');
  box.style.display = 'inline-flex';
  box.style.alignItems = 'flex-end';
  box.style.gap = '0';
  box.style.width = '120px';
  box.style.height = '18px';
  box.style.verticalAlign = 'middle';
  if (nb === 0) return box;

  let maxY = 0;
  for (let i = 0; i < nb; i++) if (ys[i] > maxY) maxY = ys[i];
  if (!(maxY > 0)) maxY = 1;

  const meanBin = binIndexOf(hist, marks.meanValue);
  const medianBin = binIndexOf(hist, marks.medianValue);

  for (let i = 0; i < nb; i++) {
    const h = ys[i] / maxY;
    const bar = document.createElement('div');
    bar.style.flex = '1 1 0';
    // Min 1px so a non-empty bin is always visible; 0-count bins stay flat.
    bar.style.height = (ys[i] > 0 ? Math.max(h * 100, 4) : 0) + '%';
    let color = 'currentColor';
    if (i === meanBin && i === medianBin) color = 'teal';
    else if (i === meanBin) color = 'green';
    else if (i === medianBin) color = 'steelblue';
    bar.style.background = color;
    box.appendChild(bar);
  }
  return box;
}

/** Format a number for a stat cell: em-dash for non-finite, else ~4 sig figs. */
function fmt(x: number): string {
  if (!Number.isFinite(x)) return '—';
  const a = Math.abs(x);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return x.toExponential(2);
  return x.toFixed(a >= 100 ? 1 : a >= 1 ? 3 : 4);
}

/** Measure-level ESS as a percentage string. Uniform measures (kHat NaN)
 *  carry no IS information → "100%". */
function essPercent(measure: any): string {
  try {
    const dof = FlatPPLEngine.empirical.estimateDof(measure);
    const q = FlatPPLEngine.empirical.importanceSamplingQuality(measure, dof);
    if (!Number.isFinite(q.kHat)) return '100%';
    const pct = q.ratio * 100;
    return (pct >= 10 ? pct.toFixed(0) : pct.toFixed(1)) + '%';
  } catch (_) { return '—'; }
}

const COLS = ['variate', 'mean', 'std', 'mode', 'median', '5%', '95%', 'ESS%', 'histogram'];

/** Third record-measure view mode: a per-variate summary-statistics table. */
export function renderRecordTable(ctx: any, hostEl: HTMLElement, measure: any, bindingName: string): void {
  const axes = listScalarAxes(measure);
  if (axes.length === 0) {
    // Mirror the marginals/corner empty message (built as trusted markup; name escaped).
    const empty = document.createElement('div');
    empty.style.opacity = '0.5';
    empty.style.padding = '24px';
    empty.style.textAlign = 'center';
    empty.innerHTML = 'No scalar fields to plot for <strong>' + esc(bindingName) + '</strong>.';
    hostEl.appendChild(empty);
    return;
  }

  const logWeights = measure.logWeights;
  const essStr = essPercent(measure);

  const table = document.createElement('table');
  table.style.width = '100%';
  table.style.borderCollapse = 'collapse';
  table.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
  table.style.fontSize = '0.9em';

  const thead = document.createElement('thead');
  const hrow = document.createElement('tr');
  for (let c = 0; c < COLS.length; c++) {
    const th = document.createElement('th');
    th.textContent = COLS[c];
    th.style.textAlign = c === 0 ? 'left' : (c === COLS.length - 1 ? 'center' : 'right');
    th.style.padding = '2px 8px';
    th.style.borderBottom = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.2))';
    th.style.opacity = '0.7';
    hrow.appendChild(th);
  }
  thead.appendChild(hrow);
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  for (let i = 0; i < axes.length; i++) {
    const a = axes[i];
    const s = variateSummary(a.samples, logWeights);
    const tr = document.createElement('tr');

    const nameTd = document.createElement('td');
    nameTd.textContent = a.label;       // textContent: never injects markup
    nameTd.style.textAlign = 'left';
    nameTd.style.padding = '2px 8px';
    tr.appendChild(nameTd);

    const cells = [fmt(s.mean), fmt(s.std), fmt(s.mode), fmt(s.median), fmt(s.q05), fmt(s.q95), essStr];
    for (let c = 0; c < cells.length; c++) {
      const td = document.createElement('td');
      td.textContent = cells[c];
      td.style.textAlign = 'right';
      td.style.padding = '2px 8px';
      tr.appendChild(td);
    }

    const histTd = document.createElement('td');
    histTd.style.textAlign = 'center';
    histTd.style.padding = '2px 8px';
    histTd.appendChild(inlineHistogramHtml(s.hist, { meanValue: s.mean, medianValue: s.median }));
    tr.appendChild(histTd);

    tbody.appendChild(tr);
  }
  table.appendChild(tbody);

  // hostEl may have been styled as a grid/flex by a previous mode; the
  // caller (rerenderChart) already resets display/grid before calling us,
  // but set overflow so a tall table scrolls inside the pane.
  hostEl.style.overflow = 'auto';
  hostEl.appendChild(table);
}
