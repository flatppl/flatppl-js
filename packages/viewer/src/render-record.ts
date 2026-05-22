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

import { showPlotMessage } from './render-frame.js';
import { esc, formatCount, formatLogTotalmass, formatSampleCount, formatScalar, formatValue } from './util.js';
import { renderPlotFrame, renderTextValue } from './render-frame.js';
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

export function renderConstantRecord(ctx: Ctx, measure: any, bindingName: string) {
  renderTextValue(ctx, bindingName, formatConstantMeasure(ctx, measure));
}

export function renderRecordMarginals(ctx: Ctx, measure: any, bindingName: string, extraToolbarControls: any) {
  const axes = listScalarAxes(measure);
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
    };
  } else {
    // Drop any selections that no longer exist (rare — defensive).
    const present: Record<string, boolean> = {}; axes.forEach(function(a: any) { present[a.key] = true; });
    ctx.recordSelection!.selected = ctx.recordSelection!.selected.filter(function(k: any) { return present[k]; });
    if (!ctx.recordSelection!.marginalGroups) ctx.recordSelection!.marginalGroups = allGroups.slice();
    else {
      const presentGroups: Record<string, boolean> = {}; allGroups.forEach(function(g: any) { presentGroups[g] = true; });
      ctx.recordSelection!.marginalGroups = ctx.recordSelection!.marginalGroups.filter(
        function(g: any) { return presentGroups[g]; });
      if (ctx.recordSelection!.marginalGroups.length === 0) ctx.recordSelection!.marginalGroups = allGroups.slice();
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
    if (ctx.recordSelection!.mode === 'marginals') {
      // Marginals mode: filter axes by selected groups (group =
      // axis label's prefix before any "[k]"). Default is all
      // groups → full axis list; users uncheck to narrow.
      const selSet: Record<string, boolean> = {};
      (ctx.recordSelection!.marginalGroups || allGroups).forEach(function(g: any) {
        selSet[g] = true;
      });
      const picked = axes.filter(function(a: any) { return selSet[axisGroupKey(a.label)]; });
      renderDensityStrips(ctx, chartHostRef, measure, bindingName, picked);
    } else {
      renderCornerGrid(ctx, chartHostRef, measure, bindingName);
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
      axes, allGroups, rerenderAll, rerenderChart, extra);
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
export function renderRecordToolbar(ctx: Ctx, axes: any[], groups: string[], onModeChange: () => void, onSelectionChange: () => void, extraToolbarControls: any) {
  const bar = document.createDocumentFragment();

  // ---- Mode toggle group ----
  const modeGroup = document.createElement('div');
  modeGroup.style.display = 'flex';
  modeGroup.style.gap = '0.25em';

  function makeModeBtn(modeKey: any, label: any, title: any) {
    const b = document.createElement('button');
    b.textContent = label;
    b.title = title;
    b.style.cursor = 'pointer';
    b.style.fontSize = '1em';
    b.style.padding = '0.2em 0.8em';
    b.style.border = '1px solid var(--vscode-button-border, transparent)';
    b.style.borderRadius = '3px';
    const active = ctx.recordSelection!.mode === modeKey;
    b.style.background = active
      ? 'var(--vscode-button-background, #0e639c)'
      : 'var(--vscode-button-secondaryBackground, #3a3d41)';
    b.style.color = active
      ? 'var(--vscode-button-foreground, #fff)'
      : 'var(--vscode-button-secondaryForeground, #ccc)';
    b.addEventListener('click', function() {
      if (ctx.recordSelection!.mode === modeKey) return;
      ctx.recordSelection!.mode = modeKey;
      // Clip selection to correlations cap when switching back.
      if (modeKey === 'correlations'
          && ctx.recordSelection!.selected.length > ctx.CORRELATIONS_MAX_AXES) {
        ctx.recordSelection!.selected = ctx.recordSelection!.selected.slice(0, ctx.CORRELATIONS_MAX_AXES);
      }
      // Mode toggle changes button styling → full toolbar rebuild.
      onModeChange();
    });
    return b;
  }
  modeGroup.appendChild(makeModeBtn('correlations', 'Correlations',
    'Pairwise corner plot: marginals on the diagonal, joint scatters below'));
  modeGroup.appendChild(makeModeBtn('marginals', 'Marginals',
    'One column per axis with vertical density shading; plots every axis'));
  bar.appendChild(modeGroup);

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

  // Caller-supplied controls (currently: the kernel-sample
  // preset dropdown) sit after the axis selector so the
  // toolbar reads left-to-right as
  //   [plot style] [axes] [preset] [...N + ESS pinned right by frame]
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

    const nLabel = document.createElement('span');
    nLabel.textContent = formatSampleCount(q.N) + ' samples';
    nLabel.title = 'Total atom count in the empirical measure'
                 + (q.N >= 100 && Math.log10(q.N) === Math.floor(Math.log10(q.N))
                    ? ' (' + formatCount(q.N) + ')'
                    : '');
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
    wrap.textContent = '— samples';
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
