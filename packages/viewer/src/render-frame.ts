// @ts-check
// @flatppl/viewer — plot-pane frame + per-binding errors —
//
// setPlotEnabled toggles the plot pane; renderPlotFrame builds the
// stable toolbar + chart-host scaffold every renderer fills.
// renderTextValue shows a constant scalar (literal / Dirac /
// deterministic-arithmetic result). errorsForBinding surfaces
// type-error rows the info panel echoes. makeActionButton is the
// codicon-based icon button used by preset/domain controls.

import { renderPlotForCurrent } from './render-plot.js';
import { renderSampleStats } from './render-record.js';
import { $, esc } from './util.js';
/**
 * Return the analyzer-level error diagnostics that landed on a
 * binding (typeinfer mismatches, undefined refs, etc.), or null
 * if there are none. Source for both the plot pane's
 * "semantically invalid" message and the DAG's red error border.
 */
import { cancelAllSampling } from './worker.js';
import type { Ctx } from './types';
export function errorsForBinding(ctx: Ctx, bindingName: any) {
  if (!bindingName || !ctx.currentState || !ctx.currentState.data
      || !ctx.currentState.data.nodes) return null;
  const nodes = ctx.currentState.data.nodes;
  for (let i = 0; i < nodes.length; i++) {
    if (nodes[i].id === bindingName) return nodes[i].errors || null;
  }
  return null;
}

/**
 * Reset plot-content's inline style. The marginals view sets
 * display:grid with several layout properties; subsequent
 * single-chart views need a clean slate so their content fills
 * the pane without inheriting a stale grid.
 */
export function resetPlotContentStyle(ctx: Ctx) {
  const el = $('plot-content');
  el.style.display = '';
  el.style.gridTemplateColumns = '';
  el.style.gridTemplateRows = '';
  el.style.gap = '';
  el.style.padding = '';
  el.style.boxSizing = '';
  el.style.flexDirection = '';
}

/**
 * Render a message into the plot pane in place of a chart.
 *
 * SECURITY CONTRACT: `html` is TRUSTED MARKUP rendered via innerHTML so callers
 * can emphasise text (`<strong>…</strong>`, `<ul>`, error colouring). Callers
 * MUST `esc()` any model-derived interpolation (binding names, error messages)
 * before passing it in. FlatPPL identifiers cannot contain `<`, but error
 * strings and future inputs can — every current call site escapes; keep it so.
 */
export function showPlotMessage(ctx: Ctx, html: string, options?: { cancellable?: boolean; hint?: boolean; progress?: boolean }) {
  if (ctx.plotEchart) { ctx.plotEchart.dispose(); ctx.plotEchart = null; }
  resetPlotContentStyle(ctx);
  const el = $('plot-content');
  const cancellable = options && options.cancellable;
  const hint       = options && options.hint;
  const progress   = options && options.progress;
  // A determinate progress bar for off-thread samplers (MH / emcee / AMIS),
  // updated via updatePlotProgress as the worker streams mcmcProgress. Starts
  // empty; if no progress arrives the bar simply stays at 0 (the Stop button
  // still works), so it degrades to the old indeterminate behaviour.
  const progHtml = progress
    ? '<div class="plot-progress"><div class="plot-progress-fill" id="plot-progress-fill"></div></div>'
      + '<div class="plot-progress-label" id="plot-progress-label"></div>'
    : '';
  const stopHtml = cancellable
    ? '<div><button class="plot-stop-btn" id="plot-stop-btn">Stop</button></div>'
    : '';
  const cls = hint ? ' class="hint"' : '';
  el.innerHTML = '<div id="plot-empty"' + cls + '>' + html + progHtml + stopHtml + '</div>';
  if (cancellable) {
    const btn = document.getElementById('plot-stop-btn');
    // Wrap to bind ctx — the
    // click handler invokes its callback with the MouseEvent.
    if (btn) btn.addEventListener('click', function () { cancelAllSampling(ctx); });
  }
}

// Update the determinate progress bar drawn by showPlotMessage({progress:true}).
// `frac` in [0,1]; `phase` labels the stage ('warmup' / 'sample' / 'amis').
export function updatePlotProgress(_ctx: Ctx, frac: number, phase: string) {
  const fill = document.getElementById('plot-progress-fill');
  const label = document.getElementById('plot-progress-label');
  if (!fill) return;
  const pct = Math.max(0, Math.min(1, frac)) * 100;
  fill.style.width = pct.toFixed(1) + '%';
  if (label) {
    const nice = phase === 'amis' ? 'AMIS' : phase === 'warmup' ? 'warmup' : 'sampling';
    label.textContent = nice + ' ' + pct.toFixed(0) + '%';
  }
}

export function makeActionButton(ctx: Ctx, iconKey: any, title: any) {
  const b = document.createElement('button');
  b.type = 'button';
  b.title = title;
  b.setAttribute('aria-label', title);
  b.style.background = 'transparent';
  b.style.color = 'var(--vscode-foreground, #cccccc)';
  b.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
  b.style.borderRadius = '3px';
  b.style.padding = '2px 4px';
  b.style.display = 'inline-flex';
  b.style.alignItems = 'center';
  b.style.justifyContent = 'center';
  b.style.cursor = 'pointer';
  b.style.opacity = '0.75';
  b.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" '
    + 'xmlns="http://www.w3.org/2000/svg" fill="currentColor" '
    + 'aria-hidden="true"><path d="' + ctx.CODICON_PATHS[iconKey] + '"/></svg>';
  b.addEventListener('mouseenter', function() { b.style.opacity = '1'; });
  b.addEventListener('mouseleave', function() { b.style.opacity = '0.75'; });
  return b;
}

/**
 * Action button whose face is a Unicode glyph rather than a codicon SVG.
 * Same subtle styling as `makeActionButton` so the two read as one button
 * family in the toolbar. Used for the profile-plot "find maximum" (⛰) and
 * "auto-fit domain" (✨) actions, which sit next to the input / domain
 * selectors. Disabling drops opacity and blocks pointer events; re-enable
 * with `setGlyphButtonEnabled`.
 */
export function makeGlyphButton(glyph: string, title: string) {
  const b = document.createElement('button');
  b.type = 'button';
  b.title = title;
  b.setAttribute('aria-label', title);
  b.style.background = 'transparent';
  b.style.color = 'var(--vscode-foreground, #cccccc)';
  b.style.border = '1px solid var(--vscode-button-border, rgba(255,255,255,0.15))';
  b.style.borderRadius = '3px';
  b.style.padding = '1px 5px';
  b.style.display = 'inline-flex';
  b.style.alignItems = 'center';
  b.style.justifyContent = 'center';
  b.style.cursor = 'pointer';
  b.style.opacity = '0.75';
  b.style.fontSize = '1.05em';
  b.style.lineHeight = '1';
  b.textContent = glyph;
  b.addEventListener('mouseenter', function() { if (!b.disabled) b.style.opacity = '1'; });
  b.addEventListener('mouseleave', function() { if (!b.disabled) b.style.opacity = '0.75'; });
  return b;
}

/** Toggle a glyph action button's enabled state (used while its async
    action runs so it can't be double-fired). */
export function setGlyphButtonEnabled(b: HTMLButtonElement, enabled: boolean) {
  b.disabled = !enabled;
  b.style.opacity = enabled ? '0.75' : '0.4';
  b.style.pointerEvents = enabled ? '' : 'none';
}

export function setPlotEnabled(ctx: Ctx, enabled: any) {
  ctx.plotEnabled = !!enabled;
  const plot    = $('plot-panel');
  const graph   = $('graph-panel');
  const divider = $('plot-divider');
  const btn     = $('plot-toggle');
  plot.classList.toggle('hidden', !ctx.plotEnabled);
  graph.classList.toggle('full',  !ctx.plotEnabled);
  divider.classList.toggle('hidden', !ctx.plotEnabled);
  btn.classList.toggle('on', ctx.plotEnabled);
  btn.textContent = 'Plot: ' + (ctx.plotEnabled ? 'on' : 'off');
  // Drop any user-dragged inline flex so the class-based defaults
  // (flex: 1 1 100% on graph-full, flex: 0 0 0 on plot-hidden, or
  // the regular 60/40 split when both are showing) take effect.
  // Inline-style takes precedence over our class rules; clearing
  // it here means a toggle-off-then-on resets the split rather
  // than holding the previous drag position into the hidden state.
  graph.style.flex = '';
  plot.style.flex = '';
  // Persist across panel reopens. VS Code restores webview state
  // automatically when the panel is shown again.
  if (ctx.host.saveState) { try { ctx.host.saveState({ plotEnabled: ctx.plotEnabled, inferenceOpts: ctx.inferenceOpts }); } catch (_) {} }
  if (ctx.plotEnabled) {
    // Render whatever the current plan says — including the
    // "not plottable" message if the focused binding isn't
    // chainable. Echarts also needs resize after becoming visible
    // (it measures 0×0 while collapsed).
    renderPlotForCurrent(ctx);
    if (ctx.plotEchart) ctx.plotEchart.resize();
  } else if (ctx.plotEchart) {
    // Tear down the echart instance to avoid keeping its canvas /
    // event listeners alive while the panel is collapsed. It'll
    // be reconstructed on the next renderDensity call.
    try { ctx.plotEchart.dispose(); } catch (_) {}
    ctx.plotEchart = null;
  }
  // Cytoscape skipped resize while the graph pane was at a
  // different height — kick it now so the layout fills correctly.
  if (ctx.cy) {
    // requestAnimationFrame so the flex re-layout has settled
    // before we ask cytoscape for the new size.
    requestAnimationFrame(function() { ctx.cy.resize(); ctx.cy.fit(undefined, 40); });
  }
}

/**
 * Single entry-point for laying out a plot. Owns:
 *   - the flex-column structure of #plot-content
 *   - an optional toolbar row (controls on the left, sample-stats
 *     readout pinned right when `measure` is supplied)
 *   - the chart ctx.host that fills the remaining vertical space
 *   - disposal of any prior `ctx.plotEchart` and reset of inline styles
 *
 * Every measure-backed renderer (samples / corner / strips / kernel-
 * sample / profile / array-step) goes through here so the visual
 * framing is consistent across binding kinds. Plain text views
 * (constant scalars / records) use `renderTextValue` instead.
 *
 * opts:
 *   measure          — optional EmpiricalMeasure; drives N+ESS
 *                      readout (always shown when given, including
 *                      for unweighted measures where ESS = N).
 *   toolbarControls  — optional Element (or DocumentFragment)
 *                      appended to the LEFT of the toolbar. The
 *                      sample-stats readout (if `measure`) sits to
 *                      the RIGHT via `margin-left: auto`.
 *   chartCallback    — function(chartHost) called once the layout
 *                      is in place. The ctx.host is a div that fills
 *                      the remaining vertical space; the callback
 *                      writes its chart DOM (echarts.init,
 *                      grid layout, etc.) directly into it.
 */
export function renderPlotFrame(ctx: Ctx, opts: any) {
  resetPlotContentStyle(ctx);
  if (ctx.plotEchart) { try { ctx.plotEchart.dispose(); } catch (_) {} ctx.plotEchart = null; }
  const el = $('plot-content');
  el.innerHTML = '';
  el.style.display = 'flex';
  el.style.flexDirection = 'column';
  el.style.padding = '10px';
  el.style.boxSizing = 'border-box';
  el.style.gap = '8px';

  const hasToolbarLeft = opts.toolbarControls != null;
  const hasMeasureStats = opts.measure != null;
  if (hasToolbarLeft || hasMeasureStats) {
    const bar = document.createElement('div');
    bar.className = 'plot-frame-toolbar';
    bar.style.display = 'flex';
    bar.style.flexWrap = 'wrap';
    bar.style.gap = '0.75em';
    bar.style.alignItems = 'center';
    bar.style.padding = '0.4em 0.6em';
    bar.style.background = 'rgba(255,255,255,0.02)';
    bar.style.border = '1px solid var(--vscode-panel-border, rgba(255,255,255,0.08))';
    bar.style.borderRadius = '3px';
    bar.style.fontSize = '0.92em';
    bar.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
    bar.style.flexShrink = '0';
    if (hasToolbarLeft) bar.appendChild(opts.toolbarControls);
    if (hasMeasureStats) {
      // margin-left:auto on the spacer pushes the stats readout
      // to the right edge regardless of how many controls are
      // on the left.
      const spacer = document.createElement('div');
      spacer.style.marginLeft = 'auto';
      bar.appendChild(spacer);
      bar.appendChild(renderSampleStats(ctx, opts.measure));
    }
    el.appendChild(bar);
  }

  const chartHost = document.createElement('div');
  chartHost.style.flex = '1 1 auto';
  chartHost.style.minHeight = '0';
  chartHost.style.minWidth = '0';
  chartHost.style.position = 'relative';
  el.appendChild(chartHost);

  // Optional row beneath the chart. Used by the profile-plot path
  // to surface the lo/hi x-axis limit inputs alongside the axis
  // name, vertically aligned with where echarts' axis-name would
  // otherwise sit. Other plot types pass nothing and get the
  // previous layout (chart fills remaining height).
  if (opts.bottomRow) {
    const bottom = document.createElement('div');
    bottom.style.flexShrink = '0';
    bottom.appendChild(opts.bottomRow);
    el.appendChild(bottom);
  }

  opts.chartCallback(chartHost);
}

/**
 * Render a constant value (literal, deterministic arithmetic of
 * literals, or a degenerate distribution) as plain text in the
 * scalar-display block. Used by:
 *   - constant scalar bindings (samplesAreConstant short-circuit)
 *   - phase=fixed records / tuples (renderConstantRecord)
 *   - kernel-sample bindings whose substituted body collapses to
 *     a single value
 * The font-size auto-shrinks for long renderings (record(...) with
 * many fields) so the value still fits within the pane.
 */
// Render a constant value as text, but KEEP an input-selection toolbar mounted
// when one is supplied (a kernel / function plot whose current input yields a
// degenerate / constant output). A callable is a mapping inputs→output, so its
// input controls must always be reachable — never trap the user at a degenerate
// input with no way to pick another. With no toolbar (a genuinely fixed-phase
// binding) this is exactly renderTextValue. One place so the record and scalar
// constant paths stay unified.
export function renderConstantValue(ctx: Ctx, bindingName: any, text: any, toolbarControls?: any) {
  if (!toolbarControls) { renderTextValue(ctx, bindingName, text); return; }
  const resolved = typeof toolbarControls === 'function' ? toolbarControls() : toolbarControls;
  const composite = ('' + text).length > 16 && /[(\[]/.test(text);
  const name = bindingName ? esc(bindingName) : '';
  renderPlotFrame(ctx, {
    measure: null,
    toolbarControls: resolved,
    chartCallback: function(chartHost: any) {
      const d = document.createElement('div');
      d.className = 'scalar-display';
      d.style.display = 'flex';
      d.style.flexDirection = 'column';
      d.style.justifyContent = 'center';
      d.style.height = '100%';
      d.innerHTML = (name ? '<div class="name">' + name + '</div>' : '')
        + '<div class="' + (composite ? 'value composite' : 'value') + '">' + esc(text) + '</div>';
      chartHost.appendChild(d);
    },
  });
}

export function renderTextValue(ctx: Ctx, bindingName: any, text: any) {
  resetPlotContentStyle(ctx);
  if (ctx.plotEchart) { try { ctx.plotEchart.dispose(); } catch (_) {} ctx.plotEchart = null; }
  const el = $('plot-content');
  const name = bindingName ? esc(bindingName) : '';
  // Atomic values (e.g. "5", "Dirac(5)", "true") get the hero
  // 36px treatment so the value pops as the answer. Composite
  // values (records, multi-element arrays, Dirac wrappers around
  // structured bodies) fall back to a comfortable monospace
  // size — the .composite class flip is enough; the threshold
  // is "contains structural punctuation AND non-trivial length",
  // which catches both "record(a = 1.5, …)" (long) and
  // "[1.2, 3.4, 5.1, …, 3.9]" (medium). Short Dirac wraps like
  // "Dirac(5)" stay big.
  const composite = text.length > 16 && /[(\[]/.test(text);
  const valueClass = composite ? 'value composite' : 'value';
  el.innerHTML =
    '<div class="scalar-display">'
    + (name ? '<div class="name">' + name + '</div>' : '')
    + '<div class="' + valueClass + '">' + esc(text) + '</div>'
    + '</div>';
}
