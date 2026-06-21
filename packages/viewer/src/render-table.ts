// @flatppl/viewer — BAT-like summary-statistics table view.
//
// A third record-measure view mode alongside corner (Correlations) and
// strips (Marginals): one row per scalar variate with weighted summary
// stats and an inline SVG histogram. Stats reuse engine primitives;
// the histogram cell is translated from LazyReports.jl (SVG <rect> bars,
// tinted where the mean/median fall).

declare const FlatPPLEngine: any;
const SVG_NS = 'http://www.w3.org/2000/svg';

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
  const hist = H.freedmanDiaconisHistogram(samples, logWeights ? { logWeights } : {});

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

/** Inline SVG histogram: <rect> bars, height = ys[i]/max(ys), uniform width 1,
 *  tinted by where the mean/median fall (teal both, green mean, steelblue median). */
export function inlineHistogramSvg(hist: any, marks: { meanValue: number; medianValue: number }): SVGSVGElement {
  const ys = hist.ys || new Float64Array(0);
  const nb = ys.length;
  const svg = document.createElementNS(SVG_NS, 'svg') as SVGSVGElement;
  svg.setAttribute('width', '120');
  svg.setAttribute('height', '18');
  svg.setAttribute('viewBox', '0 0 ' + Math.max(nb, 1) + ' 1');
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('shape-rendering', 'crispEdges');
  if (nb === 0) return svg;

  let maxY = 0;
  for (let i = 0; i < nb; i++) if (ys[i] > maxY) maxY = ys[i];
  if (!(maxY > 0)) maxY = 1;

  const meanBin = binIndexOf(hist, marks.meanValue);
  const medianBin = binIndexOf(hist, marks.medianValue);

  for (let i = 0; i < nb; i++) {
    const h = ys[i] / maxY;
    const rect = document.createElementNS(SVG_NS, 'rect');
    rect.setAttribute('x', String(i));
    rect.setAttribute('y', String(1 - h));
    rect.setAttribute('width', '1');
    rect.setAttribute('height', String(h));
    let fill = 'currentColor';
    if (i === meanBin && i === medianBin) fill = 'teal';
    else if (i === meanBin) fill = 'green';
    else if (i === medianBin) fill = 'steelblue';
    rect.setAttribute('fill', fill);
    svg.appendChild(rect);
  }
  return svg;
}
