// @flatppl/viewer — posterior-predictive check (PPC) panel renderer.
//
// For each observed field in a PPC object, renders one echarts panel that
// overlays two histograms on a shared value x-axis:
//   - the replicated predictive samples (y_rep), a weighted, area-normalised
//     histogram (filled), and
//   - the observed data, binned over the SAME bin edges (outline), so the
//     shapes compare directly.
//
// Both series are drawn as `custom` rect series (the same primitive the corner-
// plot marginals use), so the bars sit at true data coordinates and the two
// histograms register exactly.
//
// Weighted domain clipping
// ─────────────────────────
// Heavy-tailed likelihoods can produce y_rep samples with extreme finite
// outliers that carry near-zero weight. Using raw min/max to set the visible
// x-range would collapse the histogram to a single bar. Instead we clip the
// display domain to weighted quantiles ([0.5%, 99.5%]); samples outside the
// range still land in the boundary bins (they are clamped, not discarded).

import { esc, formatScalar } from './util.js';
import { colorForBinding } from './palette.js';

declare const FlatPPLEngine: any;

/** Weighted quantile clip fractions for the histogram x-domain. */
const DOMAIN_LO_Q = 0.005;
const DOMAIN_HI_Q = 0.995;

/** Number of bins for the predictive histogram. */
const PPC_HIST_BINS = 60;

/**
 * Compute sorted sample/weight pairs needed by weightedQuantileSorted.
 * Returns [sortedSamples, sortedWeights]; `logWeights` null → uniform weights.
 */
function sortedPairs(
  samples: Float64Array,
  logWeights: Float64Array | null,
): [Float64Array, Float64Array] {
  const H = FlatPPLEngine.histogram;
  const n = samples.length;
  const w: Float64Array = logWeights
    ? H.normaliseWeights(logWeights)
    : (() => { const u = new Float64Array(n); u.fill(n > 0 ? 1 / n : 0); return u; })();

  const idx = Array.from({ length: n }, (_: unknown, i: number) => i);
  idx.sort((a: number, b: number) => samples[a] - samples[b]);
  const ss = new Float64Array(n);
  const sw = new Float64Array(n);
  for (let i = 0; i < n; i++) { ss[i] = samples[idx[i]]; sw[i] = w[idx[i]]; }
  return [ss, sw];
}

/**
 * Area-normalised (PDF-scale) histogram of `values` over the given bin edges,
 * so it overlays a weighted predictive histogram built on the same edges.
 * Values outside the edge range are clamped into the boundary bins.
 */
function densityOnEdges(values: number[], binEdges: ArrayLike<number>): Float64Array {
  const nb = binEdges.length - 1;
  if (nb <= 0) return new Float64Array(0);
  const counts = new Float64Array(nb);
  let n = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    let b: number;
    if (v <= binEdges[0]) b = 0;
    else if (v >= binEdges[nb]) b = nb - 1;
    else { b = 0; while (b < nb - 1 && v > binEdges[b + 1]) b++; }
    counts[b]++; n++;
  }
  const ys = new Float64Array(nb);
  for (let b = 0; b < nb; b++) {
    const w = binEdges[b + 1] - binEdges[b];
    ys[b] = (n > 0 && w > 0) ? counts[b] / (n * w) : 0;
  }
  return ys;
}

/**
 * Scale a histogram to unit peak. Predictive (many samples, spread out) and
 * observed (often just a handful of points, concentrated) have wildly different
 * per-bin densities; area-normalising both makes the observed spikes dwarf the
 * predictive. Peak-normalising each to 1 keeps both visible and makes the
 * comparison about SHAPE and LOCATION, which is what a PPC reads.
 */
function normalizePeak(ys: ArrayLike<number>): Float64Array {
  let m = 0;
  for (let i = 0; i < ys.length; i++) if (ys[i] > m) m = ys[i];
  const out = new Float64Array(ys.length);
  if (!(m > 0)) return out;
  for (let i = 0; i < ys.length; i++) out[i] = ys[i] / m;
  return out;
}

/** Build `custom`-series rect data for a histogram (bin centre + height + edges). */
function rectsFor(binEdges: ArrayLike<number>, ys: ArrayLike<number>): Array<{ value: number[]; x0: number; x1: number }> {
  const rects: Array<{ value: number[]; x0: number; x1: number }> = [];
  for (let k = 0; k < ys.length; k++) {
    rects.push({
      value: [((binEdges[k] as number) + (binEdges[k + 1] as number)) / 2, ys[k] as number],
      x0: binEdges[k] as number,
      x1: binEdges[k + 1] as number,
    });
  }
  return rects;
}

/** A custom rect-bar series drawing `rects` at true data coordinates. */
function barSeries(
  rects: Array<{ value: number[]; x0: number; x1: number }>,
  name: string,
  fill: string,
  stroke: string,
  lineWidth: number,
  opacity: number,
): any {
  return {
    name,
    type: 'custom',
    data: rects,
    renderItem: function (_p: any, api: any) {
      const d = rects[_p.dataIndex];
      const lt = api.coord([d.x0, d.value[1]]);
      const rb = api.coord([d.x1, 0]);
      return {
        type: 'rect',
        shape: { x: lt[0], y: lt[1], width: rb[0] - lt[0], height: rb[1] - lt[1] },
        style: api.style({ fill, opacity, stroke, lineWidth }),
      };
    },
    encode: { x: 0, y: 1 },
  };
}

/**
 * Build and append one PPC panel into `hostEl`: an echarts chart overlaying the
 * weighted predictive histogram and the observed-data histogram.
 */
function appendPpcPanel(
  ctx: any,
  hostEl: HTMLElement,
  fieldName: string,
  yRepSamples: Float64Array,
  logWeights: Float64Array | null,
  observed: number[],
): void {
  const H = FlatPPLEngine.histogram;
  const fg = getComputedStyle(document.body).color || '#ccc';
  const color = colorForBinding(ctx, fieldName);

  const panel = document.createElement('div');
  panel.style.marginBottom = '16px';

  const heading = document.createElement('div');
  heading.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
  heading.style.fontSize = '0.9em';
  heading.style.fontWeight = 'bold';
  heading.style.marginBottom = '4px';
  heading.style.opacity = '0.85';
  heading.innerHTML = esc(fieldName);
  panel.appendChild(heading);

  const n = yRepSamples.length;
  if (n === 0) {
    const empty = document.createElement('div');
    empty.style.opacity = '0.5';
    empty.style.fontSize = '0.85em';
    empty.textContent = '(no y_rep samples)';
    panel.appendChild(empty);
    hostEl.appendChild(panel);
    return;
  }

  // ── Weighted-quantile x-domain clip ────────────────────────────────────────
  const [ss, sw] = sortedPairs(yRepSamples, logWeights);
  const rawLo = H.weightedQuantileSorted(ss, sw, DOMAIN_LO_Q);
  const rawHi = H.weightedQuantileSorted(ss, sw, DOMAIN_HI_Q);
  let domainLo = Number.isFinite(rawLo) ? rawLo : ss[0];
  let domainHi = Number.isFinite(rawHi) ? rawHi : ss[ss.length - 1];
  if (!(domainHi > domainLo)) { domainLo -= 1; domainHi += 1; }

  // ── Predictive histogram (weighted, clipped to the display domain) ─────────
  const clipped = new Float64Array(n);
  let clippedWeights: Float64Array | null = null;
  if (logWeights) {
    clippedWeights = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      clipped[i] = Math.min(Math.max(yRepSamples[i], domainLo), domainHi);
      clippedWeights[i] = logWeights[i];
    }
  } else {
    for (let i = 0; i < n; i++) clipped[i] = Math.min(Math.max(yRepSamples[i], domainLo), domainHi);
  }

  const hist = H.freedmanDiaconisHistogram(
    clipped,
    clippedWeights ? { logWeights: clippedWeights, maxBins: PPC_HIST_BINS } : { maxBins: PPC_HIST_BINS },
  );
  const binEdges: ArrayLike<number> = hist.binEdges || new Float64Array(0);
  const predYs: Float64Array = normalizePeak(hist.ys || new Float64Array(0));

  if (predYs.length === 0) {
    const empty = document.createElement('div');
    empty.style.opacity = '0.5';
    empty.style.fontSize = '0.85em';
    empty.textContent = '(degenerate predictive histogram)';
    panel.appendChild(empty);
    hostEl.appendChild(panel);
    return;
  }

  // ── Observed histogram over the SAME bin edges (peak-normalised to match) ──
  const obsYs = normalizePeak(densityOnEdges(observed, binEdges));

  // ── echarts panel: predictive filled + observed outline, overlaid ──────────
  const chart = document.createElement('div');
  chart.style.width = '100%';
  chart.style.height = '160px';
  panel.appendChild(chart);
  hostEl.appendChild(panel);   // attach before init so echarts measures a real size

  const ec = echarts.init(chart);
  ec.setOption({
    backgroundColor: 'transparent',
    animation: false,
    grid: { left: 48, right: 12, top: 22, bottom: 24, containLabel: false },
    legend: {
      data: ['predictive', 'observed'],
      textStyle: { color: fg }, top: 0, right: 8, itemWidth: 14, itemHeight: 8, itemGap: 12,
    },
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'value', scale: true,
      axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
      axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
      axisLabel: { color: fg, opacity: 0.6, fontSize: 10, formatter: formatScalar },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value', scale: true,
      axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
      axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
      axisLabel: { color: fg, opacity: 0.5, fontSize: 10, formatter: formatScalar },
      splitLine: { lineStyle: { color: fg, opacity: 0.1 } },
    },
    series: [
      barSeries(rectsFor(binEdges, predYs), 'predictive', color, color, 0.5, 0.6),
      barSeries(rectsFor(binEdges, obsYs), 'observed', 'transparent', fg, 1.2, 1),
    ],
  });
}

/**
 * Render a posterior-predictive check object into `hostEl`: one echarts panel
 * per field (declaration order), each overlaying the weighted y_rep histogram
 * and the observed-data histogram.
 *
 * `ppc` shape: `{ fields: { [name]: { yRep: { samples, logWeights }, observed } } }`.
 */
export function renderRecordPpc(ctx: any, hostEl: HTMLElement, ppc: any): void {
  hostEl.style.overflow = 'auto';

  if (!ppc || !ppc.fields || Object.keys(ppc.fields).length === 0) {
    const empty = document.createElement('div');
    empty.style.opacity = '0.5';
    empty.style.padding = '24px';
    empty.style.textAlign = 'center';
    empty.textContent = 'No posterior-predictive fields available.';
    hostEl.appendChild(empty);
    return;
  }

  const fieldNames = Object.keys(ppc.fields);
  for (let i = 0; i < fieldNames.length; i++) {
    const name = fieldNames[i];
    const field = ppc.fields[name];
    if (!field) continue;
    const yRep = field.yRep;
    const observed: number[] = Array.isArray(field.observed) ? field.observed : [];
    if (!yRep || !(yRep.samples instanceof Float64Array)) continue;
    appendPpcPanel(ctx, hostEl, name, yRep.samples, yRep.logWeights ?? null, observed);
  }
}
