// @ts-check
// @flatppl/viewer — plot-pane frame + per-binding errors (Phase 4d).
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
export function errorsForBinding(ctx, bindingName) {
  if (!bindingName || !ctx.currentState || !ctx.currentState.data
      || !ctx.currentState.data.nodes) return null;
  var nodes = ctx.currentState.data.nodes;
  for (var i = 0; i < nodes.length; i++) {
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
export function resetPlotContentStyle(ctx) {
  var el = $('plot-content');
  el.style.display = '';
  el.style.gridTemplateColumns = '';
  el.style.gridTemplateRows = '';
  el.style.gap = '';
  el.style.padding = '';
  el.style.boxSizing = '';
  el.style.flexDirection = '';
}

export function showPlotMessage(ctx: any, html: string, options?: { cancellable?: boolean; hint?: boolean }) {
  if (ctx.plotEchart) { ctx.plotEchart.dispose(); ctx.plotEchart = null; }
  resetPlotContentStyle(ctx);
  var el = $('plot-content');
  var cancellable = options && options.cancellable;
  var hint       = options && options.hint;
  var stopHtml = cancellable
    ? '<div><button class="plot-stop-btn" id="plot-stop-btn">Stop</button></div>'
    : '';
  var cls = hint ? ' class="hint"' : '';
  el.innerHTML = '<div id="plot-empty"' + cls + '>' + html + stopHtml + '</div>';
  if (cancellable) {
    var btn = document.getElementById('plot-stop-btn');
    // Wrap to bind ctx — Phase 3 added ctx as first param, but the
    // click handler invokes its callback with the MouseEvent.
    if (btn) btn.addEventListener('click', function () { cancelAllSampling(ctx); });
  }
}

export function makeActionButton(ctx, iconKey, title) {
  var b = document.createElement('button');
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

export function setPlotEnabled(ctx, enabled) {
  ctx.plotEnabled = !!enabled;
  var plot    = $('plot-panel');
  var graph   = $('graph-panel');
  var divider = $('plot-divider');
  var btn     = $('plot-toggle');
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
  if (ctx.host.saveState) { try { ctx.host.saveState({ plotEnabled: ctx.plotEnabled }); } catch (_) {} }
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
export function renderPlotFrame(ctx, opts) {
  resetPlotContentStyle(ctx);
  if (ctx.plotEchart) { try { ctx.plotEchart.dispose(); } catch (_) {} ctx.plotEchart = null; }
  var el = $('plot-content');
  el.innerHTML = '';
  el.style.display = 'flex';
  el.style.flexDirection = 'column';
  el.style.padding = '10px';
  el.style.boxSizing = 'border-box';
  el.style.gap = '8px';

  var hasToolbarLeft = opts.toolbarControls != null;
  var hasMeasureStats = opts.measure != null;
  if (hasToolbarLeft || hasMeasureStats) {
    var bar = document.createElement('div');
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
      var spacer = document.createElement('div');
      spacer.style.marginLeft = 'auto';
      bar.appendChild(spacer);
      bar.appendChild(renderSampleStats(ctx, opts.measure));
    }
    el.appendChild(bar);
  }

  var chartHost = document.createElement('div');
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
    var bottom = document.createElement('div');
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
export function renderTextValue(ctx, bindingName, text) {
  resetPlotContentStyle(ctx);
  if (ctx.plotEchart) { try { ctx.plotEchart.dispose(); } catch (_) {} ctx.plotEchart = null; }
  var el = $('plot-content');
  var name = bindingName ? esc(bindingName) : '';
  // Atomic values (e.g. "5", "Dirac(5)", "true") get the hero
  // 36px treatment so the value pops as the answer. Composite
  // values (records, multi-element arrays, Dirac wrappers around
  // structured bodies) fall back to a comfortable monospace
  // size — the .composite class flip is enough; the threshold
  // is "contains structural punctuation AND non-trivial length",
  // which catches both "record(a = 1.5, …)" (long) and
  // "[1.2, 3.4, 5.1, …, 3.9]" (medium). Short Dirac wraps like
  // "Dirac(5)" stay big.
  var composite = text.length > 16 && /[(\[]/.test(text);
  var valueClass = composite ? 'value composite' : 'value';
  el.innerHTML =
    '<div class="scalar-display">'
    + (name ? '<div class="name">' + name + '</div>' : '')
    + '<div class="' + valueClass + '">' + esc(text) + '</div>'
    + '</div>';
}
