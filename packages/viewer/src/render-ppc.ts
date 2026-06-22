// @flatppl/viewer — posterior-predictive check (PPC) panel renderer.
//
// For each observed field in a PPC object, renders a panel containing:
//   - A full-width weighted histogram of the replicated predictive samples
//     (y_rep), drawn as HTML <div> bars (no chart library, no external deps).
//   - A rug strip beneath the histogram: one thin tick per observed value,
//     positioned by linear interpolation over the histogram's x-range.
//
// Weighted domain clipping
// ─────────────────────────
// Heavy-tailed likelihoods can produce y_rep samples with extreme finite
// outliers that carry near-zero weight. Using raw min/max to set the visible
// x-range would collapse the histogram to a single bar that misrepresents the
// actual predictive mass. Instead we use weighted quantiles ([0.5%, 99.5%]) to
// clip the histogram domain so the bulk of the predictive distribution is
// visible. Samples outside [lo, hi] still contribute to the histogram bars
// that fall at the boundary — they are not discarded, only the display range
// is narrowed. The approach mirrors the weighted-quantile path in
// render-table.ts (sort ascending, re-pair weights, call
// FlatPPLEngine.histogram.weightedQuantileSorted).

import { esc } from './util.js';

declare const FlatPPLEngine: any;

/** Maximum rug ticks to draw; subsample when observed.length exceeds this. */
const MAX_RUG_TICKS = 200;

/** Weighted quantile clip fractions for the histogram x-domain. */
const DOMAIN_LO_Q = 0.005;
const DOMAIN_HI_Q = 0.995;

/** Number of bins for the PPC histogram (larger than the table inline cell). */
const PPC_HIST_BINS = 60;

/**
 * Compute sorted sample/weight pairs needed by weightedQuantileSorted.
 * Returns [sortedSamples, sortedWeights]; both have the same length as
 * `samples`. `logWeights` null → uniform weights.
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
 * Build and append one PPC panel into `hostEl` for the given field.
 *
 * The histogram uses a weighted-quantile-clipped x-domain ([0.5%, 99.5%])
 * so heavy-tailed y_rep outliers with near-zero weight do not collapse the
 * visible range to a single bar. See module-level comment for rationale.
 */
function appendPpcPanel(
  hostEl: HTMLElement,
  fieldName: string,
  yRepSamples: Float64Array,
  logWeights: Float64Array | null,
  observed: number[],
): void {
  const H = FlatPPLEngine.histogram;

  // ── Container ─────────────────────────────────────────────────────────────
  const panel = document.createElement('div');
  panel.style.marginBottom = '16px';

  // ── Heading ───────────────────────────────────────────────────────────────
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

  // ── Compute weighted-quantile x-domain ────────────────────────────────────
  // Sort once for both quantile queries and reuse the pairs.
  const [ss, sw] = sortedPairs(yRepSamples, logWeights);
  const rawLo = H.weightedQuantileSorted(ss, sw, DOMAIN_LO_Q);
  const rawHi = H.weightedQuantileSorted(ss, sw, DOMAIN_HI_Q);

  // Guard: if domain collapsed (e.g. all samples equal) widen slightly so the
  // histogram call gets a valid range, and the rug interpolation doesn't divide
  // by zero.
  let domainLo = Number.isFinite(rawLo) ? rawLo : ss[0];
  let domainHi = Number.isFinite(rawHi) ? rawHi : ss[ss.length - 1];
  if (!(domainHi > domainLo)) {
    domainLo = domainLo - 1;
    domainHi = domainHi + 1;
  }

  // ── Weighted histogram ────────────────────────────────────────────────────
  // Clip samples to [domainLo, domainHi] before passing to the histogram so
  // the FD algorithm operates on the clipped range (avoids degenerate bins from
  // extreme outlier coordinates), then collect the returned bin heights.
  const clipped = new Float64Array(n);
  let clippedWeights: Float64Array | null = null;
  if (logWeights) {
    clippedWeights = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      clipped[i] = Math.min(Math.max(yRepSamples[i], domainLo), domainHi);
      clippedWeights[i] = logWeights[i];
    }
  } else {
    for (let i = 0; i < n; i++) {
      clipped[i] = Math.min(Math.max(yRepSamples[i], domainLo), domainHi);
    }
  }

  const hist = H.freedmanDiaconisHistogram(
    clipped,
    clippedWeights
      ? { logWeights: clippedWeights, maxBins: PPC_HIST_BINS }
      : { maxBins: PPC_HIST_BINS },
  );

  const ys: Float64Array = hist.ys || new Float64Array(0);
  const nb = ys.length;
  const binEdges: ArrayLike<number> = hist.binEdges || new Float64Array(0);

  // ── Histogram bars ────────────────────────────────────────────────────────
  const histBox = document.createElement('div');
  histBox.style.position = 'relative';
  histBox.style.display = 'flex';
  histBox.style.alignItems = 'flex-end';
  histBox.style.gap = '0';
  histBox.style.width = '100%';
  histBox.style.height = '120px';
  histBox.style.boxSizing = 'border-box';

  if (nb > 0) {
    let maxY = 0;
    for (let i = 0; i < nb; i++) if (ys[i] > maxY) maxY = ys[i];
    if (!(maxY > 0)) maxY = 1;

    for (let i = 0; i < nb; i++) {
      const bar = document.createElement('div');
      bar.style.flex = '1 1 0';
      bar.style.height = (ys[i] > 0 ? Math.max((ys[i] / maxY) * 100, 2) : 0) + '%';
      bar.style.background = 'currentColor';
      bar.style.opacity = '0.75';
      histBox.appendChild(bar);
    }
  }

  panel.appendChild(histBox);

  // ── Rug strip ─────────────────────────────────────────────────────────────
  // Determine x-range from binEdges. binEdges has nb+1 entries.
  const edgeCount = binEdges instanceof Float64Array
    ? (binEdges as Float64Array).length
    : (binEdges as number[]).length;
  const xMin = edgeCount > 0 ? (binEdges as any)[0] : domainLo;
  const xMax = edgeCount > 1 ? (binEdges as any)[edgeCount - 1] : domainHi;
  const xRange = xMax - xMin;

  let rugValues = observed;
  let capped = false;
  if (observed.length > MAX_RUG_TICKS) {
    // Evenly subsample to MAX_RUG_TICKS entries (not random, for determinism).
    const step = observed.length / MAX_RUG_TICKS;
    rugValues = new Array<number>(MAX_RUG_TICKS);
    for (let i = 0; i < MAX_RUG_TICKS; i++) {
      rugValues[i] = observed[Math.round(i * step)];
    }
    capped = true;
  }

  const rugBox = document.createElement('div');
  rugBox.style.position = 'relative';
  rugBox.style.width = '100%';
  rugBox.style.height = '8px';
  rugBox.style.marginTop = '2px';
  rugBox.style.boxSizing = 'border-box';
  rugBox.style.overflow = 'hidden';

  if (xRange > 0) {
    for (let i = 0; i < rugValues.length; i++) {
      const v = rugValues[i];
      if (!Number.isFinite(v)) continue;
      const pct = (v - xMin) / xRange;
      if (pct < 0 || pct > 1) continue;
      const tick = document.createElement('div');
      tick.style.position = 'absolute';
      tick.style.left = (pct * 100) + '%';
      tick.style.top = '0';
      tick.style.width = '1px';
      tick.style.height = '100%';
      tick.style.background = 'currentColor';
      tick.style.opacity = '0.6';
      rugBox.appendChild(tick);
    }
  }

  panel.appendChild(rugBox);

  // ── Rug cap note ─────────────────────────────────────────────────────────
  if (capped) {
    const note = document.createElement('div');
    note.style.fontSize = '0.8em';
    note.style.opacity = '0.55';
    note.style.marginTop = '1px';
    note.textContent = '(showing 200 of ' + observed.length + ' observed values)';
    panel.appendChild(note);
  }

  hostEl.appendChild(panel);
}

/**
 * Render a posterior-predictive check object into `hostEl`: one panel per
 * field (declaration order), each with a weighted y_rep histogram and an
 * observed-values rug.
 *
 * `ppc` shape: `{ fields: { [name]: { yRep: { samples, logWeights }, observed } } }`.
 * `ctx` is passed for API consistency with other renderers but is not used here.
 */
export function renderRecordPpc(_ctx: unknown, hostEl: HTMLElement, ppc: any): void {
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
    appendPpcPanel(hostEl, name, yRep.samples, yRep.logWeights ?? null, observed);
  }
}
