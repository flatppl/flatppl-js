// @flatppl/viewer — record/sample-stats renderers —
//
// renderRecordMarginals + renderRecordToolbar drive the correlations/
// marginals view modes for record-shaped measures; renderAxisDropdown
// / renderGroupDropdown build the axis/group multi-select popovers;
// renderSampleStats shows the N/ESS readout; renderConstantRecord
// short-circuits the constant case; measureIsConstant /
// formatConstantMeasure detect + format constant measures.

import type { Ctx } from './types';
import { renderCornerGrid, renderDensityStrips } from './render-density.js';
import { renderRecordTable } from './render-table.js';
import { detectGeneratedQuantities, fixedEnvFor } from './generated-quantities.js';
import { buildInferenceControl, buildDrawControl } from './render-controls.js';
import { renderRecordPpc } from './render-ppc.js';
import { sendWorker } from './worker.js';

import { showPlotMessage } from './render-frame.js';
import { esc, formatCount, formatLogTotalmass, formatSampleCount, formatScalar, formatValue } from './util.js';
import { renderPlotFrame, renderConstantValue } from './render-frame.js';
import { listScalarAxes, qualityTooltip, samplesAreConstant } from './util.js';
export function measureIsConstant(ctx: Ctx, m: any): boolean {
  if (!m) return false;
  if (m.fields) {
    for (const k in m.fields) {
      if (!measureIsConstant(ctx, m.fields[k])) return false;
    }
    return true;
  }
  if (Array.isArray(m.elems)) {
    for (let i = 0; i < m.elems.length; i++) {
      if (!measureIsConstant(ctx, m.elems[i])) return false;
    }
    return true;
  }
  if (m.shape === 'array' && m.samples instanceof Float64Array && m.dims) {
    // Atom-major SoA: stride k = prod(dims) per atom. The whole
    // array is constant iff slot s has the same value at every
    // atom, for every s in [0, k).
    const stride = m.dims.reduce(function(p: any, n: any) { return p * n; }, 1);
    if (stride === 0) return true;
    const N = m.samples.length / stride;
    for (let s = 0; s < stride; s++) {
      const v = m.samples[s];
      for (let ai = 1; ai < N; ai++) {
        if (m.samples[ai * stride + s] !== v) return false;
      }
    }
    return true;
  }
  if (m.samples instanceof Float64Array) {
    // Length-mismatch with SAMPLE_COUNT identifies a literal-array
    // measure (kind: 'array' derivation): per-atom these are
    // deterministic, even though the array's own elements differ.
    if (m.samples.length !== ctx.SAMPLE_COUNT) return true;
    return samplesAreConstant(m.samples);
  }
  return false;
}

export function formatConstantMeasure(ctx: Ctx, m: any): string {
  if (!m) return '?';
  if (m.fields) {
    const ks = Object.keys(m.fields);
    const fparts = new Array(ks.length);
    for (let i = 0; i < ks.length; i++) {
      fparts[i] = ks[i] + ' = ' + formatConstantMeasure(ctx, m.fields[ks[i]]);
    }
    return 'record(' + fparts.join(', ') + ')';
  }
  if (Array.isArray(m.elems)) {
    const eparts = new Array(m.elems.length);
    for (let ei = 0; ei < m.elems.length; ei++) {
      // Tuple element may be null when fixedValueToMeasure
      // couldn't represent it (an rngstate, typically). Surface
      // a placeholder so the rest of the tuple's structure stays
      // visible — e.g. `(record(obs = […]), <rngstate>)` for a
      // single-LHS `rand(rs, m)` result.
      eparts[ei] = m.elems[ei] ? formatConstantMeasure(ctx, m.elems[ei]) : '<rngstate>';
    }
    return '(' + eparts.join(', ') + ')';
  }
  if (m.shape === 'array' && m.samples instanceof Float64Array && m.dims) {
    const stride = m.dims.reduce(function(p: any, n: any) { return p * n; }, 1);
    return formatValue(m.samples.subarray(0, stride), undefined);
  }
  if (m.samples instanceof Float64Array && m.samples.length > 0) {
    // Two cases distinguished by sample length:
    //   - length === SAMPLE_COUNT: a per-atom scalar measure
    //     (caller verified samples are constant across atoms);
    //     surface a single number.
    //   - length !== SAMPLE_COUNT: a literal-data array
    //     (kind:'array' derivation surfaced as a record field);
    //     surface every element with array ellipsis.
    if (m.samples.length === ctx.SAMPLE_COUNT) return formatScalar(m.samples[0]);
    return formatValue(m.samples, undefined);
  }
  return '?';
}

export function renderConstantRecord(ctx: Ctx, measure: any, bindingName: string, toolbarControls?: any) {
  // A callable (kernel / function) is a MAPPING inputs→output: keep the
  // input-selection toolbar mounted even when THIS input yields a degenerate
  // (constant) output — never trap the user at the degenerate point (e.g. a
  // kernel at its 0-valued type-default inputs collapsing to record(obs=[0,…])).
  // renderConstantValue falls back to plain hero text when no toolbar is given
  // (a genuinely fixed-phase record binding).
  renderConstantValue(ctx, bindingName, formatConstantMeasure(ctx, measure), toolbarControls);
}

export function renderRecordMarginals(ctx: Ctx, measure: any, bindingName: string, extraToolbarControls: any) {
  // Detect whether this binding is a bayesupdate posterior. If so,
  // generated-quantity detection is available and the toolbar
  // will offer a "Generated" toggle group.
  const isBayesupdate = !!(
    ctx.derivationsState &&
    ctx.derivationsState.derivations &&
    ctx.derivationsState.derivations[bindingName] &&
    ctx.derivationsState.derivations[bindingName].kind === 'bayesupdate'
  );

  // Compute generated-quantity candidates once (stable across rerenders
  // of the same focused binding). Empty when not a posterior.
  const genCandidates = isBayesupdate ? detectGeneratedQuantities(ctx, measure) : [];

  // displayMeasure: the measure passed to chart renderers — either the
  // raw measure (no generated quantities toggled on) or a shallow clone
  // with toggled generated-quantity fields appended. Cached per
  // (measure identity, genQuantities list identity) to avoid recomputing
  // on every rerenderChart call. The cache invalidates when either the
  // measure or the toggle list reference changes.
  let _dmCacheMeasure: any = null;
  let _dmCacheKey: string = '';
  let _dmCacheResult: any = null;

  function displayMeasure() {
    const on = (ctx.recordSelection && ctx.recordSelection.genQuantities) || [];
    const cacheKey = on.slice().sort().join(',');
    if (_dmCacheMeasure === measure && _dmCacheKey === cacheKey) return _dmCacheResult;
    if (!on.length) {
      _dmCacheMeasure = measure;
      _dmCacheKey = cacheKey;
      _dmCacheResult = measure;
      return measure;
    }
    const specs = genCandidates.filter(function(s: any) { return on.indexOf(s.name) >= 0; });
    if (!specs.length) {
      _dmCacheMeasure = measure;
      _dmCacheKey = cacheKey;
      _dmCacheResult = measure;
      return measure;
    }
    // Plain {name: value} env for any fixed-phase constants the toggled
    // quantities reference — evaluateExprN reads baseEnv via `name in baseEnv`,
    // so the FixedValues resolver itself won't do (its names live behind .get).
    const baseEnv = fixedEnvFor(ctx, measure, specs);
    let result: any;
    try {
      result = FlatPPLEngine.generatedQuantities.appendGeneratedQuantities(measure, specs, baseEnv);
    } catch (e) {
      try { console.error('[viewer] generated-quantity eval failed:', e); } catch (_) {}
      result = measure;
    }
    _dmCacheMeasure = measure;
    _dmCacheKey = cacheKey;
    _dmCacheResult = result;
    return result;
  }

  // Axes for toolbar / selection state come from the DERIVED measure
  // (displayMeasure()). With genQuantities empty this is the raw measure
  // (unchanged first render); after a toggle it includes the derived
  // fields, so allGroups, the per-axis Variates dropdown, the corner
  // grid (filters by recordSelection.selected) and the marginals strips
  // (filter by recordSelection.marginalGroups) all see them — not just
  // the Table (which lists displayMeasure's fields directly).
  const axes = listScalarAxes(displayMeasure());
  if (axes.length === 0) {
    showPlotMessage(ctx, 'No scalar fields to plot for <strong>' + esc(bindingName) + '</strong>.', { hint: true });
    return;
  }

  // Group prefix per axis (drop any trailing "[k]"). Used by
  // marginals view's group-level selector and (separately) by
  // its boundary insets between groups. Same definition both
  // places — kept here so selection state and rendering stay in
  // sync via a single source of truth.
  function axisGroupKey(label: any) {
    const i = label.lastIndexOf('[');
    return i >= 0 ? label.slice(0, i) : label;
  }
  const allGroups: string[] = [];
  const seenGroup: Record<string, boolean> = {};
  for (let gi = 0; gi < axes.length; gi++) {
    const g = axisGroupKey(axes[gi].label);
    if (!seenGroup[g]) { seenGroup[g] = true; allGroups.push(g); }
  }

  // Reset selection when the focused binding changes. Defaults:
  //   mode='correlations'; selected = first CORRELATIONS_MAX_AXES
  //                                    axes (per-axis selection)
  //   marginalGroups = all groups (group-level selection used in
  //                                marginals mode)
  if (!ctx.recordSelection || ctx.recordSelection!.bindingName !== bindingName) {
    ctx.recordSelection = {
      bindingName: bindingName,
      mode: 'correlations',
      selected: axes.slice(0, ctx.CORRELATIONS_MAX_AXES).map(function(a: any) { return a.key; }),
      marginalGroups: allGroups.slice(),
      genQuantities: [],
    };
  } else {
    // Drop any selections that no longer exist (rare — defensive).
    // If the present-filter empties a previously non-empty selection,
    // every selected key vanished from this measure's axes — a transient
    // shape mismatch across re-renders of the "same" binding (e.g. a
    // single-axis posterior whose one key (`lambda`) didn't survive an
    // earlier render's seeding). Re-seed the default rather than strand
    // the user on "Select at least one axis to plot" with no way back
    // except a manual re-pick. This mirrors the marginalGroups empty-
    // guard just below; the two selection states now self-heal alike.
    // (A deliberate in-panel deselect-all routes through rerenderChart,
    // not this reconciliation, so it is never undone here.)
    const present: Record<string, boolean> = {}; axes.forEach(function(a: any) { present[a.key] = true; });
    const prevSelected = ctx.recordSelection!.selected;
    const keptSelected = prevSelected.filter(function(k: any) { return present[k]; });
    ctx.recordSelection!.selected = (prevSelected.length > 0 && keptSelected.length === 0)
      ? axes.slice(0, ctx.CORRELATIONS_MAX_AXES).map(function(a: any) { return a.key; })
      : keptSelected;
    if (!ctx.recordSelection!.marginalGroups) ctx.recordSelection!.marginalGroups = allGroups.slice();
    else {
      const presentGroups: Record<string, boolean> = {}; allGroups.forEach(function(g: any) { presentGroups[g] = true; });
      ctx.recordSelection!.marginalGroups = ctx.recordSelection!.marginalGroups.filter(
        function(g: any) { return presentGroups[g]; });
      if (ctx.recordSelection!.marginalGroups.length === 0) ctx.recordSelection!.marginalGroups = allGroups.slice();
    }
    if (!ctx.recordSelection!.genQuantities) ctx.recordSelection!.genQuantities = [];
  }

  // PPC state: built lazily on first switch to 'ppc' mode.
  // ppcAvailable starts false; set to true once a non-null PPC
  // is successfully built so the dropdown option is shown after
  // the first successful build. ppcBuilt guards against re-runs.
  let ppc: any = null;
  let ppcAvailable = false;
  let ppcBuilt = false;

  // Build the PPC once — called when the user first selects 'ppc'
  // mode and ppcBuilt is false. Builds a matCtx mirroring the one
  // used by engine-facade.getMeasure (same derivations/bindings/
  // fixedValues/rootKey/sendWorker shape the engine expects).
  // On success sets ppc + ppcAvailable = true and triggers a full
  // rerender so the dropdown option appears and the chart renders.
  // On failure ppc stays null and we fall through to correlations.
  async function ensurePpc() {
    if (ppcBuilt) return;
    ppcBuilt = true;
    if (!isBayesupdate || !ctx.derivationsState) return;
    const deriv = ctx.derivationsState.derivations[bindingName];
    if (!deriv || deriv.kind !== 'bayesupdate') return;
    const matCtx: any = {
      derivations: ctx.derivationsState.derivations,
      bindings:    ctx.derivationsState.bindings,
      fixedValues: ctx.derivationsState.fixedValues,
      rootKey:     ctx.rootSeed,
      sendWorker:  function(m: any) { return sendWorker(ctx, m); },
    };
    try {
      const result = await FlatPPLEngine.posteriorPredictive.buildPosteriorPredictive(
        deriv, matCtx, measure,
      );
      if (result) {
        ppc = result;
        ppcAvailable = true;
        // Full rerender so the toolbar gains the PPC dropdown option
        // and the chart area switches to the PPC renderer.
        rerenderAll();
      }
    } catch (_) {
      // Silently ignore — ppc stays null, ppcAvailable stays false.
    }
  }

  // chartHostRef captures the chart-area div from the frame, so
  // rerenderChart can clear and repopulate it without rebuilding
  // the toolbar (which would close any open dropdown).
  let chartHostRef: HTMLElement | null = null;

  // Two-tier re-render. rerenderAll rebuilds the entire frame
  // (including the toolbar) — used when mode-button styling
  // changes. rerenderChart only repaints the chart host —
  // used by axis-selection toggles so the open dropdown survives.
  function rerenderChart() {
    if (!chartHostRef) return;
    chartHostRef.innerHTML = '';
    // Reset inline styles the strip / grid renderer may have set
    // on a previous pass (display:grid for cornerGrid; flex for
    // strips). We re-establish from scratch each draw.
    chartHostRef.style.display = '';
    chartHostRef.style.gridTemplateColumns = '';
    chartHostRef.style.gridTemplateRows = '';
    chartHostRef.style.gap = '';
    chartHostRef.style.overflow = '';
    // Use the derived measure (with any toggled generated quantities
    // appended as extra fields) for all chart renderers.
    const dm = displayMeasure();
    if ((ctx.recordSelection!.mode as any) === 'ppc') {
      if (ppc) {
        renderRecordPpc(ctx, chartHostRef, ppc);
      } else {
        // PPC not yet available (build pending or failed) — fall back
        // to correlations so the pane is never blank. If the build
        // completes successfully it triggers a full rerender.
        if (!ppcBuilt) ensurePpc();
        renderCornerGrid(ctx, chartHostRef, dm, bindingName);
      }
    } else if (ctx.recordSelection!.mode === 'table') {
      renderRecordTable(ctx, chartHostRef, dm, bindingName);
    } else if (ctx.recordSelection!.mode === 'marginals') {
      // Marginals mode: filter axes by selected groups (group =
      // axis label's prefix before any "[k]"). Default is all
      // groups → full axis list; users uncheck to narrow.
      // Re-enumerate axes from the DERIVED measure each draw (not the
      // outer `axes`): the gen-quantity toggle uses the chart-only
      // rerender, which does NOT recompute the outer `axes`, so a
      // freshly-toggled quantity is only visible by re-listing `dm` here.
      const dmAxes = listScalarAxes(dm);
      const selSet: Record<string, boolean> = {};
      (ctx.recordSelection!.marginalGroups || allGroups).forEach(function(g: any) {
        selSet[g] = true;
      });
      const picked = dmAxes.filter(function(a: any) { return selSet[axisGroupKey(a.label)]; });
      renderDensityStrips(ctx, chartHostRef, dm, bindingName, picked);
    } else {
      renderCornerGrid(ctx, chartHostRef, dm, bindingName);
    }
  }
  function rerenderAll() {
    // extraToolbarControls is a builder thunk (or null) — resolve
    // to a fresh Element/Fragment each rebuild. A static Element
    // captured once gets emptied on the first appendChild (for
    // DocumentFragments) or destroyed by renderPlotFrame's
    // innerHTML='' before the next rebuild can re-use it.
    const extra = typeof extraToolbarControls === 'function'
      ? extraToolbarControls()
      : extraToolbarControls;
    const toolbarControls = renderRecordToolbar(ctx,
      axes, allGroups, rerenderAll, rerenderChart, extra,
      isBayesupdate, isBayesupdate ? genCandidates : [], ppcAvailable);
    renderPlotFrame(ctx, {
      measure: measure,
      toolbarControls: toolbarControls,
      chartCallback: function(chartHost: any) {
        chartHostRef = chartHost;
        rerenderChart();
      },
    });
  }

  rerenderAll();
  // Eagerly kick off the PPC build for bayesupdate posteriors so the
  // "Posterior predictive" dropdown option appears automatically once
  // it completes — the user does not need to switch to 'ppc' mode first.
  if (isBayesupdate) ensurePpc();
}

/**
 * Build the inner controls of the corner-plot toolbar: view-mode
 * toggle on the left, axis (or group) selector to its right, and
 * the kernel-sample preset dropdown (when supplied) further right.
 *
 * Returns a DocumentFragment that the caller hands to
 * renderPlotFrame as `toolbarControls`. The frame owns the
 * outer toolbar styling and pins the N+ESS readout to the right
 * — this builder no longer touches sample-stats.
 *
 * Rebuilt on every full rerender (cheap; <100 elements) so the
 * mode buttons reflect active state and the selector visibility
 * tracks the mode.
 */
export function renderRecordToolbar(ctx: Ctx, axes: any[], groups: string[], onModeChange: () => void, onSelectionChange: () => void, extraToolbarControls: any, isBayesupdatePosterior: boolean, genCandidates?: Array<{ name: string; ir: any }>, ppcAvailable?: boolean) {
  const bar = document.createDocumentFragment();

  // ---- Mode dropdown ----
  // Single <select> replaces the former Correlations/Marginals/Table button
  // group to reclaim toolbar width. Option list is data-driven so Spec 2's
  // "Posterior predictive" (bayesupdate-only) can be appended later.
  const MODE_OPTIONS: Array<{ key: string; label: string; title: string }> = [
    { key: 'correlations', label: 'Correlations', title: 'Pairwise corner plot: marginals on the diagonal, joint scatters below' },
    { key: 'marginals',    label: 'Marginals',    title: 'One column per axis with vertical density shading; plots every axis' },
    { key: 'table',        label: 'Table',        title: 'Summary-statistics table: per-variate mean, std, median, credible interval, ESS, R̂, MCSE, and an inline histogram' },
  ];
  if (ppcAvailable) {
    MODE_OPTIONS.push({ key: 'ppc', label: 'Posterior predictive', title: 'Replicated observations forward-sampled at each posterior draw, overlaid on the observed data' });
  }
  const modeSel = document.createElement('select');
  modeSel.style.cursor = 'pointer';
  modeSel.style.fontSize = '1em';
  modeSel.style.padding = '0.2em 0.4em';
  modeSel.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
  modeSel.style.color = 'var(--vscode-dropdown-foreground, #ccc)';
  modeSel.style.border = '1px solid var(--vscode-dropdown-border, #555)';
  modeSel.style.borderRadius = '3px';
  for (const o of MODE_OPTIONS) {
    const opt = document.createElement('option');
    opt.value = o.key; opt.textContent = o.label; opt.title = o.title;
    modeSel.appendChild(opt);
  }
  modeSel.value = ctx.recordSelection!.mode;
  modeSel.title = 'View mode for this record measure';
  modeSel.addEventListener('change', function () {
    const modeKey = modeSel.value;
    if (ctx.recordSelection!.mode === modeKey) return;
    ctx.recordSelection!.mode = modeKey as any;
    if (modeKey === 'correlations'
        && ctx.recordSelection!.selected.length > ctx.CORRELATIONS_MAX_AXES) {
      ctx.recordSelection!.selected = ctx.recordSelection!.selected.slice(0, ctx.CORRELATIONS_MAX_AXES);
    }
    onModeChange();
  });
  bar.appendChild(modeSel);

  // Sampler selector — only for bayesupdate posteriors (the only measures that
  // use a sampler). Hidden for IID priors and every other measure, so the bar
  // never shows a stale backend label for a prior that is sampled IID.
  if (isBayesupdatePosterior && ctx.onInferenceChange) {
    const sep0 = document.createElement('div');
    sep0.style.width = '1px';
    sep0.style.alignSelf = 'stretch';
    sep0.style.background = 'rgba(255,255,255,0.1)';
    bar.appendChild(sep0);
    bar.appendChild(buildInferenceControl(ctx, ctx.onInferenceChange));
  }

  // Forward-draw control — for ALL sampled record plots (prior IID and
  // posterior alike), unlike the posterior-only Sampler control above.
  if (ctx.onForwardDrawChange) {
    const sepD = document.createElement('div');
    sepD.style.width = '1px';
    sepD.style.alignSelf = 'stretch';
    sepD.style.background = 'rgba(255,255,255,0.1)';
    bar.appendChild(sepD);
    bar.appendChild(buildDrawControl(ctx, ctx.onForwardDrawChange));
  }

  // Axis-level selector in correlations mode (per-leaf
  // checkboxes, capped at CORRELATIONS_MAX_AXES); group-level
  // selector in marginals mode (one entry per name-prefix —
  // obs[1]…obs[10] collapse into a single "obs" toggle).
  if (ctx.recordSelection!.mode === 'correlations') {
    const sep = document.createElement('div');
    sep.style.width = '1px';
    sep.style.alignSelf = 'stretch';
    sep.style.background = 'rgba(255,255,255,0.1)';
    bar.appendChild(sep);
    // Axis-checkbox toggles only need to redraw the chart (the
    // toolbar's button styling is unaffected) — pass the
    // chart-only callback so the dropdown doesn't get rebuilt
    // out from under its open popup.
    bar.appendChild(renderAxisDropdown(ctx, axes, onSelectionChange));
  } else if (ctx.recordSelection!.mode === 'marginals' && groups && groups.length > 1) {
    const sep2 = document.createElement('div');
    sep2.style.width = '1px';
    sep2.style.alignSelf = 'stretch';
    sep2.style.background = 'rgba(255,255,255,0.1)';
    bar.appendChild(sep2);
    bar.appendChild(renderGroupDropdown(ctx, groups, onSelectionChange));
  }

  // Generated-quantities toggle group — only for bayesupdate posteriors
  // when there is at least one qualifying deterministic binding. Toggles
  // default off; toggling on appends derived fields (sharing posterior
  // logWeights) before chart rendering so they appear as extra variates.
  if (genCandidates && genCandidates.length > 0) {
    const sep3 = document.createElement('div');
    sep3.style.width = '1px';
    sep3.style.alignSelf = 'stretch';
    sep3.style.background = 'rgba(255,255,255,0.1)';
    bar.appendChild(sep3);
    // Chart-only rerender (onSelectionChange), same as the Variates /
    // group dropdowns — keeps THIS popup open across clicks. rerenderChart
    // re-derives the display measure (displayMeasure() is cache-keyed on
    // genQuantities) and re-enumerates its axes per draw, so a toggled
    // quantity still appears in Table + Marginals + Correlations.
    bar.appendChild(renderGenQuantitiesDropdown(ctx, genCandidates, onSelectionChange));
  }

  // Caller-supplied controls (currently: the kernel-sample
  // preset dropdown) sit after the axis selector so the
  // toolbar reads left-to-right as
  //   [plot style] [axes] [generated] [preset] [...N + ESS pinned right by frame]
  if (extraToolbarControls) bar.appendChild(extraToolbarControls);
  return bar;
}

export function renderSampleStats(ctx: Ctx, measure: any) {
  const wrap = document.createElement('span');
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.4em';
  wrap.style.opacity = '0.85';
  wrap.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
  wrap.style.fontSize = '0.92em';

  // Totalmass badge — shows when the measure is non-normalized
  // (weighted, superpose, bayesupdate's posterior carrying the
  // marginal likelihood Z, etc.). Normalized measures (every leaf
  // distribution, normalize(...), lawof(...)) show no badge so the
  // readout stays uncluttered.
  if (measure && typeof measure.logTotalmass === 'number') {
    const massText = formatLogTotalmass(measure.logTotalmass);
    if (massText != null) {
      const massSpan = document.createElement('span');
      massSpan.textContent = 'total mass: ' + massText;
      massSpan.title = 'log total mass: ' + measure.logTotalmass.toFixed(4)
        + '\nThe measure is unnormalized — its total mass differs from 1. '
        + 'Wrap in normalize(...) to rescale.';
      massSpan.style.opacity = '0.9';
      wrap.appendChild(massSpan);
      // Visual separator between badges. A middle dot collides
      // with the math context here — `exp(-20.564) · 10⁵` reads as
      // a product. A pipe stays neutral and is the conventional
      // "and now a different stat" separator in technical UIs.
      // We also bump the surrounding gap so the boundary is
      // visually distinct without needing a heavy glyph.
      const sep = document.createElement('span');
      sep.textContent = '│';   // U+2502 BOX DRAWINGS LIGHT VERTICAL
      sep.style.opacity = '0.35';
      sep.style.margin = '0 0.25em';
      wrap.appendChild(sep);
    }
  }

  // Defensive try/catch: a thrown error here would propagate up
  // through renderPlotFrame → renderRecordMarginals' rerenderAll,
  // poisoning the entire plot render. Diagnostic-readout failure
  // is non-fatal — fall back to a count-only display so the chart
  // still draws. console.error surfaces real bugs in the quality
  // classifier without breaking user-facing rendering.
  try {
    const dof = FlatPPLEngine.empirical.estimateDof(measure);
    const q = FlatPPLEngine.empirical.importanceSamplingQuality(measure, dof);

    // Sampler backends (mh/emcee/amis/smc/ess) RESAMPLE their output to
    // SAMPLE_COUNT atoms for plotting, so q.N is an unconditional ~10^5 that
    // misrepresents the run. Their record carries diagnostics.nSamples = the TRUE
    // draw count — show that, labelled "draws", with the resample noted in the
    // title. Plain measures keep the atom-count "draws".
    const nLabel = document.createElement('span');
    const dg: any = measure && measure.diagnostics;
    const trueN = (dg && Number.isFinite(dg.nSamples) && dg.nSamples > 0) ? dg.nSamples : null;
    if (trueN != null) {
      nLabel.textContent = formatSampleCount(trueN) + ' draws';
      nLabel.title = trueN + ' sampler draws (resampled to ' + formatCount(q.N) + ' atoms for plotting)';
    } else {
      nLabel.textContent = formatSampleCount(q.N) + ' draws';
      nLabel.title = 'Total draw count in the empirical measure'
                   + (q.N >= 100 && Math.log10(q.N) === Math.floor(Math.log10(q.N))
                      ? ' (' + formatCount(q.N) + ')'
                      : '');
    }
    wrap.appendChild(nLabel);

    // Effectively-uniform measures (no logWeights, or logWeights
    // all-equal-within-epsilon) carry no IS-quality information:
    // every atom has equal weight by construction, so PSIS k̂
    // doesn't apply and the ESS readout would just say "100%".
    // The engine signals this via kHat NaN — we skip the
    // diagnostic span and show the bare count. The diagnostic
    // only appears when it actually carries information
    // (importance-reweighted measures: bayesupdate / weighted /
    // logweighted / posterior outputs).
    //
    // MCMC posterior (backend mh / emcee): equal-weight draws (so kHat is NaN,
    // the IS readout below is skipped) carrying a `diagnostics` object instead.
    // Surface acceptance + the worst-across-parameters split-R̂ / bulk-ESS,
    // colour-coded with the same is-quality classes as the IS readout.
    if (measure && measure.diagnostics) {
      const d = measure.diagnostics;

      // Elliptical slice: equal-weight MCMC draws, but no accept rate (it's
      // near-rejection-free) — report split-R̂ + bulk ESS (disambiguated from the
      // sampler's "ESS" name), mean shrink steps, and the reference mode.
      if (d.method === 'ess-slice') {
        const pp = d.perParam || {};
        let maxRhat = 0, minEss = Infinity;
        for (const k of Object.keys(pp)) { if (Number.isFinite(pp[k].rHat)) maxRhat = Math.max(maxRhat, pp[k].rHat); if (Number.isFinite(pp[k].essBulk)) minEss = Math.min(minEss, pp[k].essBulk); }
        const label = (maxRhat <= 1.01 && minEss >= 400) ? 'good' : (maxRhat <= 1.05 && minEss >= 100) ? 'ok' : (maxRhat <= 1.10) ? 'bad' : 'unusable';
        const es = document.createElement('span');
        es.className = 'is-quality is-' + label;
        es.textContent = 'ESS-slice: R̂ ' + (maxRhat > 0 ? maxRhat.toFixed(3) : '—') + ', ESS(eff) ' + (Number.isFinite(minEss) ? formatSampleCount(Math.round(minEss)) : '—') + ', ~' + (Number.isFinite(d.meanShrinks) ? d.meanShrinks.toFixed(1) : '—') + ' shrinks, ' + (d.mode || '');
        es.title = 'Elliptical slice sampling (gradient- and tuning-free):'
          + '\nmax split-R̂ ' + (maxRhat > 0 ? maxRhat.toFixed(3) : '—') + ' (want < 1.01)'
          + '\nmin bulk effective sample size ' + (Number.isFinite(minEss) ? Math.round(minEss) : '—')
          + '\nmean shrink steps/iteration ' + (Number.isFinite(d.meanShrinks) ? d.meanShrinks.toFixed(2) : '—')
          + '\nGaussian reference: ' + (d.mode === 'exact' ? 'exact (Normal prior)' : 'fitted to the population');
        wrap.appendChild(es);
        return wrap;
      }

      // SMC: equal-weight particles (no IS weights / R̂). Headline the log
      // marginal likelihood (evidence) plus ladder length and move acceptance.
      if (d.method === 'smc') {
        const acc = Number.isFinite(d.acceptRate) ? (d.acceptRate * 100).toFixed(0) + '%' : '—';
        const label = (d.acceptRate >= 0.15 && d.acceptRate <= 0.6) ? 'good' : (d.acceptRate >= 0.08) ? 'ok' : 'bad';
        const sm = document.createElement('span');
        sm.className = 'is-quality is-' + label;
        sm.textContent = 'SMC: logZ ' + (Number.isFinite(d.logZ) ? d.logZ.toFixed(2) : '—') + ', ' + (d.rungs != null ? d.rungs : '—') + ' rungs, accept ' + acc;
        sm.title = 'Sequential Monte Carlo (adaptive-tempered, waste-free):'
          + '\nlog marginal likelihood (evidence) ' + (Number.isFinite(d.logZ) ? d.logZ.toFixed(3) : '—')
          + '\ntemperature rungs ' + (d.rungs != null ? d.rungs : '—')
          + '\nmove acceptance ' + acc + ' (healthy ~0.2–0.4)';
        wrap.appendChild(sm);
        return wrap;
      }

      // AMIS (adaptive importance sampling) is not MCMC — report the combined
      // effective-sample-size fraction (its IS quality) and the auto-detected
      // mixture-freeze iteration K, not acceptance / R̂.
      if (d.method === 'amis') {
        const frac = Number.isFinite(d.essFrac) ? d.essFrac : 0;
        const label = frac >= 0.5 ? 'good' : frac >= 0.1 ? 'ok' : frac >= 0.01 ? 'bad' : 'unusable';
        const pct = frac >= 0.1 ? (frac * 100).toFixed(0) : (frac * 100).toFixed(1);
        const am = document.createElement('span');
        am.className = 'is-quality is-' + label;
        am.textContent = 'AMIS: ESS ' + pct + '%, K ' + (d.K != null ? d.K : '—');
        am.title = 'Adaptive multiple importance sampling (EAMIS):'
          + '\neffective sample size ' + (Number.isFinite(d.ess) ? formatSampleCount(Math.round(d.ess)) : '—')
          + ' of ' + (d.nSamples != null ? formatSampleCount(d.nSamples) : '—') + ' (' + pct + '%)'
          + '\nlow ESS ⇒ the single-Gaussian proposal fits the posterior poorly; try emcee'
          + '\nK = iteration where the proposal adaptation froze (auto-detected)';
        wrap.appendChild(am);
        return wrap;
      }

      const pp = d.perParam || {};
      let maxRhat = 0, minEss = Infinity;
      for (const k of Object.keys(pp)) {
        if (Number.isFinite(pp[k].rHat)) maxRhat = Math.max(maxRhat, pp[k].rHat);
        if (Number.isFinite(pp[k].essBulk)) minEss = Math.min(minEss, pp[k].essBulk);
      }
      const label = (maxRhat <= 1.01 && minEss >= 400) ? 'good'
                  : (maxRhat <= 1.05 && minEss >= 100) ? 'ok'
                  : (maxRhat <= 1.10)                  ? 'bad'
                  :                                      'unusable';
      const acc = Number.isFinite(d.acceptRate) ? (d.acceptRate * 100).toFixed(0) + '%' : '—';
      const rh  = maxRhat > 0 ? maxRhat.toFixed(3) : '—';
      const es  = Number.isFinite(minEss) ? formatSampleCount(Math.round(minEss)) : '—';
      const mc = document.createElement('span');
      mc.className = 'is-quality is-' + label;
      mc.textContent = 'accept ' + acc + ', R̂ ' + rh + ', ESS ' + es;
      mc.title = 'MCMC diagnostics (worst across parameters):'
        + '\nacceptance rate ' + acc
        + '\nmax split-R̂ ' + rh + ' (want < 1.01)'
        + '\nmin bulk ESS ' + es;
      wrap.appendChild(mc);
      return wrap;
    }

    if (!Number.isFinite(q.kHat)) return wrap;

    const diag = document.createElement('span');
    diag.className = 'is-quality is-' + q.label;
    const ratioPct = (q.ratio * 100);
    const ratioStr = ratioPct >= 10 ? ratioPct.toFixed(0)
                                  : ratioPct.toFixed(1);
    diag.textContent = '(' + q.label + ': ESS ' + ratioStr + '%, PSIS k̂ ' + q.kHat.toFixed(2) + ')';
    diag.title = qualityTooltip(q);
    wrap.appendChild(diag);
  } catch (err) {
    try { console.error('IS-quality classifier failed:', err); } catch (_) {}
    wrap.textContent = '— draws';
  }
  return wrap;
}

/**
 * Compact dropdown axis selector for correlations mode. Button
 * shows the count ("Plot axes (3 / 12) ▾"); click opens a
 * popup-anchored panel with a scrollable checkbox list. Outside
 * clicks close it. Cap enforcement (max 4) shows an inline red
 * note in the panel when the user tries to exceed.
 */
export function renderAxisDropdown(ctx: Ctx, axes: any[], onChange: () => void) {
  const wrap = document.createElement('div');
  wrap.style.position = 'relative';
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.4em';

  const hint = document.createElement('span');
  hint.textContent = 'Variates:';
  hint.style.opacity = '0.6';
  wrap.appendChild(hint);

  const btn = document.createElement('button');
  btn.style.cursor = 'pointer';
  btn.style.fontSize = '1em';
  btn.style.padding = '0.2em 0.6em';
  btn.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
  btn.style.borderRadius = '3px';
  btn.style.background = 'var(--vscode-button-secondaryBackground, #3a3d41)';
  btn.style.color = 'var(--vscode-button-secondaryForeground, #ccc)';
  btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
  btn.textContent = ctx.recordSelection!.selected.length
    + ' / ' + axes.length + '  ▾';
  wrap.appendChild(btn);

  // Popup panel — absolutely positioned beneath the button.
  const panel = document.createElement('div');
  panel.style.position = 'absolute';
  panel.style.top = 'calc(100% + 4px)';
  panel.style.left = '0';
  panel.style.zIndex = '50';
  panel.style.minWidth = '14em';
  panel.style.maxHeight = '20em';
  panel.style.overflowY = 'auto';
  panel.style.padding = '0.4em';
  panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
  panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
  panel.style.borderRadius = '3px';
  panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
  panel.style.display = 'none';
  wrap.appendChild(panel);

  // Cap-error slot inside the panel (red note shown briefly when
  // the user tries to add a 5th).
  const capErr = document.createElement('div');
  capErr.style.color = '#E57373';
  capErr.style.fontSize = '0.92em';
  capErr.style.padding = '0.3em 0.4em';
  capErr.style.opacity = '0';
  capErr.style.transition = 'opacity 0.2s';
  panel.appendChild(capErr);

  axes.forEach(function(axis) {
    const label = document.createElement('label');
    label.style.display = 'flex';
    label.style.alignItems = 'center';
    label.style.gap = '0.4em';
    label.style.padding = '0.2em 0.4em';
    label.style.cursor = 'pointer';
    label.style.userSelect = 'none';
    label.style.borderRadius = '2px';
    label.addEventListener('mouseenter', function() { label.style.background = 'rgba(255,255,255,0.05)'; });
    label.addEventListener('mouseleave', function() { label.style.background = ''; });

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = ctx.recordSelection!.selected.indexOf(axis.key) >= 0;
    cb.addEventListener('change', function(ev) {
      // Don't bubble up to the wrap's outside-click closer.
      ev.stopPropagation();
      const idx = ctx.recordSelection!.selected.indexOf(axis.key);
      if (cb.checked) {
        if (idx >= 0) return;
        if (ctx.recordSelection!.selected.length >= ctx.CORRELATIONS_MAX_AXES) {
          cb.checked = false;
          capErr.textContent = 'At most ' + ctx.CORRELATIONS_MAX_AXES
            + ' axes — uncheck one first.';
          capErr.style.opacity = '1';
          return;
        }
        ctx.recordSelection!.selected.push(axis.key);
      } else {
        if (idx >= 0) ctx.recordSelection!.selected.splice(idx, 1);
      }
      capErr.style.opacity = '0';
      // Update the count on the button without rebuilding the
      // toolbar (which would tear down this dropdown's open
      // panel). The axis-dropdown stays open until the user
      // clicks outside.
      btn.textContent = ctx.recordSelection!.selected.length
        + ' / ' + axes.length + '  ▾';
      onChange();
    });
    label.appendChild(cb);

    const name = document.createElement('span');
    name.textContent = axis.label;
    name.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
    label.appendChild(name);
    panel.appendChild(label);
  });

  // Toggle on button click; close on outside click.
  btn.addEventListener('click', function(ev) {
    ev.stopPropagation();
    const open = panel.style.display !== 'none';
    panel.style.display = open ? 'none' : 'block';
    if (!open) {
      // One-shot outside-click handler — registers on this open,
      // tears itself down on close so we don't accumulate handlers.
      const off = function(ev2: any) {
        if (panel.contains(ev2.target) || btn.contains(ev2.target)) return;
        panel.style.display = 'none';
        document.removeEventListener('click', off, true);
      };
      // capture phase so we close before any inner click is processed
      setTimeout(function() {
        document.addEventListener('click', off, true);
      }, 0);
    }
  });

  return wrap;
}

/**
 * Group-level checkbox dropdown for marginals view. Same shape
 * as renderAxisDropdown but operates on group prefixes (obs[1]
 * …obs[10] collapse to a single "obs" entry) and has no
 * selection cap. State lives in ctx.recordSelection!.marginalGroups.
 */
export function renderGroupDropdown(ctx: Ctx, groups: string[], onChange: () => void) {
  const wrap = document.createElement('div');
  wrap.style.position = 'relative';
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.4em';

  const hint = document.createElement('span');
  hint.textContent = 'Variates:';
  hint.style.opacity = '0.6';
  wrap.appendChild(hint);

  const btn = document.createElement('button');
  btn.style.cursor = 'pointer';
  btn.style.fontSize = '1em';
  btn.style.padding = '0.2em 0.6em';
  btn.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
  btn.style.borderRadius = '3px';
  btn.style.background = 'var(--vscode-button-secondaryBackground, #3a3d41)';
  btn.style.color = 'var(--vscode-button-secondaryForeground, #ccc)';
  btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
  function updateBtn() {
    btn.textContent = ctx.recordSelection!.marginalGroups.length
      + ' / ' + groups.length + '  ▾';
  }
  updateBtn();
  wrap.appendChild(btn);

  const panel = document.createElement('div');
  panel.style.position = 'absolute';
  panel.style.top = 'calc(100% + 4px)';
  panel.style.left = '0';
  panel.style.zIndex = '50';
  panel.style.minWidth = '12em';
  panel.style.maxHeight = '20em';
  panel.style.overflowY = 'auto';
  panel.style.padding = '0.4em';
  panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
  panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
  panel.style.borderRadius = '3px';
  panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
  panel.style.display = 'none';
  wrap.appendChild(panel);

  groups.forEach(function(g) {
    const label = document.createElement('label');
    label.style.display = 'flex';
    label.style.alignItems = 'center';
    label.style.gap = '0.4em';
    label.style.padding = '0.2em 0.4em';
    label.style.cursor = 'pointer';
    label.style.userSelect = 'none';
    label.style.borderRadius = '2px';
    label.addEventListener('mouseenter', function() { label.style.background = 'rgba(255,255,255,0.05)'; });
    label.addEventListener('mouseleave', function() { label.style.background = ''; });

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = ctx.recordSelection!.marginalGroups.indexOf(g) >= 0;
    cb.addEventListener('change', function(ev) {
      ev.stopPropagation();
      const idx = ctx.recordSelection!.marginalGroups.indexOf(g);
      if (cb.checked) {
        if (idx < 0) ctx.recordSelection!.marginalGroups.push(g);
      } else {
        if (idx >= 0) ctx.recordSelection!.marginalGroups.splice(idx, 1);
      }
      updateBtn();
      onChange();
    });
    label.appendChild(cb);

    const name = document.createElement('span');
    name.textContent = g;
    name.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
    label.appendChild(name);
    panel.appendChild(label);
  });

  btn.addEventListener('click', function(ev) {
    ev.stopPropagation();
    const open = panel.style.display !== 'none';
    panel.style.display = open ? 'none' : 'block';
    if (!open) {
      const off = function(ev2: any) {
        if (panel.contains(ev2.target) || btn.contains(ev2.target)) return;
        panel.style.display = 'none';
        document.removeEventListener('click', off, true);
      };
      setTimeout(function() {
        document.addEventListener('click', off, true);
      }, 0);
    }
  });

  return wrap;
}

/**
 * Generated-quantities checkbox dropdown for bayesupdate posterior views.
 * Lists qualifying deterministic bindings (kind:'evaluate', self-refs ⊆
 * record fields). Toggling on appends the binding as a derived field
 * (inheriting posterior logWeights) before chart rendering.
 * Default: all OFF. State lives in ctx.recordSelection!.genQuantities.
 */
export function renderGenQuantitiesDropdown(ctx: Ctx, candidates: Array<{ name: string; ir: any }>, onChange: () => void) {
  const wrap = document.createElement('div');
  wrap.style.position = 'relative';
  wrap.style.display = 'inline-flex';
  wrap.style.alignItems = 'center';
  wrap.style.gap = '0.4em';

  const hint = document.createElement('span');
  hint.textContent = 'Generated:';
  hint.style.opacity = '0.6';
  wrap.appendChild(hint);

  const btn = document.createElement('button');
  btn.style.cursor = 'pointer';
  btn.style.fontSize = '1em';
  btn.style.padding = '0.2em 0.6em';
  btn.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
  btn.style.borderRadius = '3px';
  btn.style.background = 'var(--vscode-button-secondaryBackground, #3a3d41)';
  btn.style.color = 'var(--vscode-button-secondaryForeground, #ccc)';
  btn.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
  function updateBtn() {
    const on = ctx.recordSelection!.genQuantities || [];
    btn.textContent = on.length + ' / ' + candidates.length + '  ▾';
  }
  updateBtn();
  wrap.appendChild(btn);

  const panel = document.createElement('div');
  panel.style.position = 'absolute';
  panel.style.top = 'calc(100% + 4px)';
  panel.style.left = '0';
  panel.style.zIndex = '50';
  panel.style.minWidth = '12em';
  panel.style.maxHeight = '20em';
  panel.style.overflowY = 'auto';
  panel.style.padding = '0.4em';
  panel.style.background = 'var(--vscode-editorWidget-background, #252526)';
  panel.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.15))';
  panel.style.borderRadius = '3px';
  panel.style.boxShadow = '0 2px 8px rgba(0,0,0,0.4)';
  panel.style.display = 'none';
  wrap.appendChild(panel);

  candidates.forEach(function(c) {
    const label = document.createElement('label');
    label.style.display = 'flex';
    label.style.alignItems = 'center';
    label.style.gap = '0.4em';
    label.style.padding = '0.2em 0.4em';
    label.style.cursor = 'pointer';
    label.style.userSelect = 'none';
    label.style.borderRadius = '2px';
    label.addEventListener('mouseenter', function() { label.style.background = 'rgba(255,255,255,0.05)'; });
    label.addEventListener('mouseleave', function() { label.style.background = ''; });

    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = (ctx.recordSelection!.genQuantities || []).indexOf(c.name) >= 0;
    cb.addEventListener('change', function(ev) {
      ev.stopPropagation();
      const sel = ctx.recordSelection!;
      if (!sel.genQuantities) sel.genQuantities = [];
      if (!sel.selected) sel.selected = [];
      if (!sel.marginalGroups) sel.marginalGroups = [];
      const idx = sel.genQuantities.indexOf(c.name);
      // A scalar generated quantity surfaces as a single axis whose key
      // and group prefix are both the bare binding name (no "[k]" suffix).
      // Seed it on-by-default in BOTH the correlations selection
      // (recordSelection.selected) and the marginals group selection
      // (recordSelection.marginalGroups) so the new field shows up in
      // every view, not just the Table. Toggling off removes it from all
      // three. The selection-reset else-branch in renderRecordMarginals
      // re-filters these by present axis keys, so stale entries self-heal.
      if (cb.checked) {
        if (idx < 0) sel.genQuantities.push(c.name);
        // Respect the correlations axis cap (the Variates dropdown enforces the
        // same limit): register the derived field and seed it into marginals
        // unconditionally, but only auto-add it to the correlations selection
        // when there's room — otherwise a gen-quantity toggle would silently
        // push `selected` past CORRELATIONS_MAX_AXES.
        if (sel.selected.indexOf(c.name) < 0
            && sel.selected.length < ctx.CORRELATIONS_MAX_AXES) {
          sel.selected.push(c.name);
        }
        if (sel.marginalGroups.indexOf(c.name) < 0) sel.marginalGroups.push(c.name);
      } else {
        if (idx >= 0) sel.genQuantities.splice(idx, 1);
        const si = sel.selected.indexOf(c.name);
        if (si >= 0) sel.selected.splice(si, 1);
        const mi = sel.marginalGroups.indexOf(c.name);
        if (mi >= 0) sel.marginalGroups.splice(mi, 1);
      }
      updateBtn();
      // Chart-only rerender (onChange === onSelectionChange) so this dropdown
      // stays open across clicks, matching the Variates/group dropdowns.
      // rerenderChart re-derives the measure + re-enumerates axes from it, so
      // the toggled field shows in Table/Marginals/Correlations without a full
      // toolbar rebuild. (Trade-off: the outer Variates/group dropdowns, built
      // once at rerenderAll, don't list the derived field — deselect it here.)
      onChange();
    });
    label.appendChild(cb);

    const name = document.createElement('span');
    name.textContent = c.name;
    name.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
    label.appendChild(name);
    panel.appendChild(label);
  });

  btn.addEventListener('click', function(ev) {
    ev.stopPropagation();
    const open = panel.style.display !== 'none';
    panel.style.display = open ? 'none' : 'block';
    if (!open) {
      const off = function(ev2: any) {
        if (panel.contains(ev2.target) || btn.contains(ev2.target)) return;
        panel.style.display = 'none';
        document.removeEventListener('click', off, true);
      };
      setTimeout(function() {
        document.addEventListener('click', off, true);
      }, 0);
    }
  });

  return wrap;
}
