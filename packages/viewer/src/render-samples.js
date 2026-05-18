// @flatppl/viewer — samples / empirical / array renderers (Phase 4e).
//
// renderArrayStepPlot draws a fixed-length numeric array as a step
// plot; renderEmpiricalMeasure renders the sample histogram + density
// for a scalar measure; renderSamplesAndDensity is the combined
// "histogram + smoothed density" pane.

import { measureIsConstant, renderConstantRecord, renderRecordMarginals } from './render-record.js';

import { colorForBinding } from './palette.js';
import { complexReBadge, esc, formatComplexScalar, formatScalar } from './util.js';
import { sendWorker } from './worker.js';
import { renderPlotFrame, renderTextValue } from './render-frame.js';
import { plotZoomOptions, samplesAreConstant } from './util.js';
export function renderArrayStepPlot(ctx, arr) {
  var fg = getComputedStyle(document.body).color || '#ccc';
  var n = arr.length;
  // Build piecewise-constant step data: each value v at index i
  // contributes two points (i, v) and (i+1, v). echarts then
  // draws segments connecting consecutive entries.
  var stepData = new Array(n * 2);
  for (var i = 0; i < n; i++) {
    stepData[2 * i]     = [i, arr[i]];
    stepData[2 * i + 1] = [i + 1, arr[i]];
  }
  // Same DAG-aligned color resolution the histogram path uses —
  // colorForBinding maps the binding's type/kind to the same
  // palette the DAG view paints. For literal arrays that's the
  // shared phaseFixed grey (post the literal-color unification);
  // for other shapes it picks up the node.kind overrides.
  var color = colorForBinding(ctx, ctx.currentPlotBindingName);
  var distLabel = ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'array';
  var arrayLegendLabel = n + ' values';
  // No measure passed — fixed array data isn't a sampled empirical
  // measure, so the frame skips the N+ESS readout. (A future
  // refinement could surface "length: n" in the toolbar instead.)
  renderPlotFrame(ctx, {
    chartCallback: function(chartHost) {
      ctx.plotEchart = echarts.init(chartHost);
      var zoomOpts = plotZoomOptions(fg);
      ctx.plotEchart.setOption({
        animation: false,
        dataZoom: zoomOpts.dataZoom,
        toolbox: zoomOpts.toolbox,
        grid: { left: 60, right: 25, top: 30, bottom: 50, containLabel: false },
        title: {
          text: distLabel,
          left: 'center', top: 4,
          textStyle: { color: fg, fontSize: 13, fontWeight: 'normal' },
        },
        legend: {
          data: [arrayLegendLabel],
          top: 4, right: 12,
          textStyle: { color: fg, fontSize: 11 },
          itemWidth: 14, itemHeight: 8,
        },
        tooltip: { show: false },
        xAxis: {
          type: 'value',
          name: 'index', nameLocation: 'center', nameGap: 28,
          min: 0, max: n,
          minInterval: 1,
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
          splitLine: { show: false },
        },
        yAxis: {
          type: 'value',
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
          splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
        },
        series: [{
          name: arrayLegendLabel,
          type: 'line', data: stepData, symbol: 'none',
          lineStyle: { color: color, width: 2 },
        }],
      });
    },
  });
}

/**
 * Single dispatch entry point for all empirical-measure plots.
 * A measure is just a nullary kernel — so kernel-sample bindings
 * (with substituted inputs) and ordinary measure bindings render
 * through exactly the same path; the only difference is whether
 * the toolbar carries a preset selector. Higher up, this is
 * called by:
 *
 *   - renderPlotForCurrent's measure path → opts.toolbarControls
 *     is null; opts.analyticalIR drives the optional density
 *     overlay.
 *   - renderKernelSampleMeasure → opts.toolbarControls is the
 *     preset dropdown; opts.analyticalIR is null (kernel bodies
 *     aren't closed-form).
 *
 * Dispatch:
 *   - record / tuple / array shape  → corner / marginals view
 *     (constant short-circuit → renderConstantRecord)
 *   - mode === 'array' (fixed array) → step plot
 *   - constant scalar samples         → renderTextValue
 *   - otherwise scalar              → bar histogram (+ optional
 *                                       analytical density curve)
 *
 * opts:
 *   name             — binding name (display + cache key seed)
 *   mode             — plan.mode (only 'array' is read here)
 *   discrete         — drives integerHistogram vs FD
 *   analyticalIR     — optional IR to send to the density worker
 *   toolbarControls  — optional builder THUNK () → Element|Fragment
 *                      that produces fresh toolbar DOM each call.
 *                      Must be a thunk (not a static Element /
 *                      Fragment) because the corner-plot rerender
 *                      path rebuilds the toolbar repeatedly, and
 *                      appendChild on a DocumentFragment empties
 *                      it after the first append. Static Elements
 *                      get destroyed by renderPlotFrame's
 *                      innerHTML='' on the next frame rebuild.
 *                      Pass null when no extra controls are needed.
 *   staleGuard       — optional () → bool; when supplied and false,
 *                      the async density round-trip's reply is
 *                      ignored. Lets the caller cancel via plan
 *                      identity comparison without leaking a
 *                      stale plot across binding navigation.
 */
export function renderEmpiricalMeasure(ctx, measure, opts) {
  var name = opts.name;
  // Multivariate measure (record / tuple / array shapes): route
  // to the corner / 2D-strip renderer.
  if (measure.shape === 'record' || measure.shape === 'tuple' || measure.shape === 'array') {
    // Constant short-circuit: a record / tuple measure whose
    // every scalar leaf is the same across atoms is a literal
    // value masquerading as a measure. N copies of the same
    // point histogram-mash into N tall bars per axis —
    // uninformative — so render the surface form as text instead.
    // Top-level array measures stay on the corner-plot path:
    // even when "constant" they're more useful as per-slot
    // histograms.
    if ((measure.shape === 'record' || measure.shape === 'tuple')
        && measureIsConstant(ctx, measure)) {
      renderConstantRecord(ctx, measure, name);
      return;
    }
    renderRecordMarginals(ctx, measure, name, opts.toolbarControls);
    return;
  }
  var samples = measure.samples;
  // Complex-valued binding (engine sets dtype:'complex' + .imag,
  // planar with .samples = Re). v1 renders the real part — honest
  // about it via a toolbar badge — rather than silently showing
  // Re with no indication it's one projection of a complex value.
  // |z| / Im / Argand modes + a mode toggle are a tracked
  // follow-up (TODO-flatppl-js.md §03); the badge is the seam
  // they'll grow from. Array/record-shaped complex (per-atom
  // vectors) keep their existing corner rendering of Re for now.
  var isComplex = measure.dtype === 'complex'
    && measure.imag instanceof Float64Array;
  // Array-mode: skip histogram + density entirely; the data
  // is a fixed-length sequence to plot as index→value, not
  // a sample of a distribution.
  if (opts.mode === 'array') {
    renderArrayStepPlot(ctx, samples);
    return;
  }
  // Constant scalar samples: render as text (same path as
  // phase=fixed scalars and degenerate distributions). A constant
  // complex value shows both parts ("a + b i") — showing only Re
  // would be actively misleading for a fixed complex constant.
  if (samplesAreConstant(samples)) {
    if (isComplex) {
      renderTextValue(ctx, name, formatComplexScalar(samples[0], measure.imag[0]));
    } else {
      renderTextValue(ctx, name, formatScalar(samples[0]));
    }
    return;
  }
  // Histogram lives on the main thread now — no round-trip.
  // Cache by (name, discrete) so click-flipping a binding is
  // instant. Cache lives only as long as the underlying measure:
  // rebuildDerivations and configUpdate (sampleCount change)
  // clear it.
  var histKey = name + '|' + (opts.discrete ? 'd' : 'c');
  var hist = ctx.histogramCache.get(histKey);
  if (!hist) {
    // Pass logWeights through so weighted measures (post
    // weighted/bayesupdate/normalize) render their bars
    // correctly. For unweighted measures this is null and the
    // histogram takes its fast count/N path.
    var histOpts = measure.logWeights ? { logWeights: measure.logWeights } : {};
    hist = opts.discrete
      ? FlatPPLEngine.histogram.integerHistogram(samples, histOpts)
      : FlatPPLEngine.histogram.freedmanDiaconisHistogram(samples, histOpts);
    ctx.histogramCache.set(histKey, hist);
  }
  var staleGuard = opts.staleGuard || function() { return true; };
  // Scalar histogram path renders once (no internal rerenders), so
  // we resolve the toolbar thunk to a static Element here. The
  // record-marginals path above keeps the thunk so each rebuild
  // produces fresh DOM.
  var resolvedToolbar = typeof opts.toolbarControls === 'function'
    ? opts.toolbarControls()
    : opts.toolbarControls;
  // Complex scalar bindings carry no toolbar of their own — surface
  // the "showing Re(z)" badge so the histogram isn't mistaken for
  // the whole value.
  if (isComplex && resolvedToolbar == null) {
    resolvedToolbar = complexReBadge();
  }
  // Only fetch analytical density when applicable. This is the
  // only worker round-trip per plot for measure bindings, and
  // it's skipped entirely for variates and chain-mode (stochastic-
  // parent) measures.
  if (opts.analyticalIR) {
    // Anchor the density curve's x-range to the histogram's
    // first/last bin edges. Otherwise the curve uses its own
    // quantile-derived grid which can extend past the bars
    // (and into impossible regions, e.g. x<0 for Exponential).
    var range;
    if (hist.binEdges && hist.binEdges.length > 1) {
      range = [hist.binEdges[0], hist.binEdges[hist.binEdges.length - 1]];
    } else if (hist.support) {
      range = [hist.support[0], hist.support[1]];
    }
    var densOpts = { gridPoints: 256 };
    if (range) densOpts.range = range;
    return sendWorker(ctx, { type: 'density', ir: opts.analyticalIR, opts: densOpts })
      .then(function(densReply) {
        if (!staleGuard()) return;
        renderSamplesAndDensity(ctx, 
          { samples: samples, histogram: hist, density: densReply, measure: measure },
          { mode: opts.mode, toolbarControls: resolvedToolbar });
      });
  }
  renderSamplesAndDensity(ctx, 
    { samples: samples, histogram: hist, density: null, measure: measure },
    { mode: opts.mode, toolbarControls: resolvedToolbar });
}

export function renderSamplesAndDensity(ctx, reply, plan) {
  // Array-data short-circuit: render an index→value step plot.
  // Skips the constant check below — a five-element array of all
  // 1s is a legitimate data sequence, not a scalar to be displayed
  // as text. (Reachable only when callers bypass
  // renderEmpiricalMeasure, which already handles the array
  // short-circuit.)
  if (plan && plan.mode === 'array') {
    renderArrayStepPlot(ctx, reply.samples);
    return;
  }

  // Constant-value short-circuit: render the value as text.
  // (Same defensive duplicate of the renderEmpiricalMeasure
  // short-circuit — keeps direct callers safe.)
  if (samplesAreConstant(reply.samples)) {
    renderTextValue(ctx, ctx.currentPlotBindingName, formatScalar(reply.samples[0]));
    return;
  }

  var fg = getComputedStyle(document.body).color || '#ccc';

  // Look up the binding's DAG-view color so the plot reads as
  // belonging to the same node the user is hovering on the graph.
  // Match the DAG renderer's color choice exactly — including its
  // node.kind override that maps a measure-typed binding to the
  // lawof blue rather than the generic 'call' grey. See
  // colorForBinding above.
  var color = colorForBinding(ctx, ctx.currentPlotBindingName);

  var hist = reply.histogram;
  var dens = reply.density;
  var discrete = (hist && hist.reference === 'counting');

  // Empirical histogram bars. For continuous, echarts' bar series
  // doesn't naturally do equal-width bars on a value xAxis; we use
  // a custom-render with the precomputed bin edges so the bars sit
  // at their actual x positions and widths regardless of zoom.
  // For discrete, the simpler bar series with categoryGap=40% gives
  // the spaced "lollipop"-ish look standard for pmfs.
  var samplesSeries;
  if (discrete) {
    var pairs = new Array(hist.xs.length);
    for (var i = 0; i < hist.xs.length; i++) pairs[i] = [hist.xs[i], hist.ys[i]];
    samplesSeries = {
      name: 'samples',
      type: 'bar',
      data: pairs,
      itemStyle: { color: color, opacity: 0.5 },
      barCategoryGap: '40%',
      z: 1,
    };
  } else {
    // Continuous: render bars via custom shape so widths track the
    // actual bin edges (not echarts' auto category spacing).
    var rects = [];
    for (var i = 0; i < hist.xs.length; i++) {
      rects.push({
        value: [hist.xs[i], hist.ys[i]],
        x0: hist.binEdges[i],
        x1: hist.binEdges[i + 1],
      });
    }
    samplesSeries = {
      name: 'samples',
      type: 'custom',
      data: rects,
      renderItem: function(_params, api) {
        var pt = api.value(0); // unused — we use the explicit edges below
        var rec = api.value(2); // also unused; we close over rects instead
        // Use the data point reference rather than pt/rec so this
        // closure stays compatible with echarts' value indexing across
        // versions. api.coord maps [x, y] data → pixel coordinates.
        var idx = _params.dataIndex;
        var d = rects[idx];
        var lt = api.coord([d.x0, d.value[1]]);  // left-top corner
        var rb = api.coord([d.x1, 0]);           // right-bottom corner
        return {
          type: 'rect',
          shape: { x: lt[0], y: lt[1], width: rb[0] - lt[0], height: rb[1] - lt[1] },
          style: api.style({ fill: color, opacity: 0.5, stroke: color, lineWidth: 0.5 }),
        };
      },
      encode: { x: 0, y: 1 },
      z: 1,
    };
  }

  // Density curve overlay (when available). Discrete + analytical
  // renders as scatter dots at integer atoms; continuous as a line.
  var densitySeries = null;
  if (dens && dens.xs && dens.xs.length > 0) {
    var dPairs = new Array(dens.xs.length);
    for (var j = 0; j < dens.xs.length; j++) dPairs[j] = [dens.xs[j], dens.ys[j]];
    if (discrete) {
      densitySeries = {
        name: 'density',
        type: 'scatter',
        data: dPairs,
        symbol: 'circle', symbolSize: 8,
        itemStyle: { color: color, borderColor: fg, borderWidth: 1, opacity: 1 },
        z: 2,
      };
    } else {
      densitySeries = {
        name: 'density',
        type: 'line',
        data: dPairs,
        symbol: 'none', smooth: false,
        lineStyle: { color: color, width: 2, opacity: 1 },
        z: 2,
      };
    }
  }

  // Density overlay is only ever the analytical PDF/pmf — there's
  // no smoothed-from-samples curve. (KDE was tried but dropped: it
  // smears mass past the support, which mis-suggests density where
  // there is none. See worker.js.)
  samplesSeries.name = 'samples';
  var series = densitySeries ? [samplesSeries, densitySeries] : [samplesSeries];
  var legendData = densitySeries ? ['samples', 'density'] : ['samples'];

  var distLabel = ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'distribution';

  // Frame owns the N + ESS readout (in the toolbar above the
  // chart). Pass reply.measure so the frame can compute it; the
  // chart itself only carries the binding-name title.
  // toolbarControls (e.g. kernel-sample preset selector) are
  // appended to the LEFT of the toolbar; N+ESS sits on the right.
  renderPlotFrame(ctx, {
    measure: reply.measure,
    toolbarControls: plan && plan.toolbarControls ? plan.toolbarControls : null,
    chartCallback: function(chartHost) {
      ctx.plotEchart = echarts.init(chartHost);
      var zoomOpts2 = plotZoomOptions(fg);
      ctx.plotEchart.setOption({
        animation: false,
        dataZoom: zoomOpts2.dataZoom,
        toolbox: zoomOpts2.toolbox,
        grid: { left: 60, right: 25, top: 30, bottom: 50, containLabel: false },
        title: {
          text: distLabel,
          left: 'center', top: 4,
          textStyle: { color: fg, fontSize: 13, fontWeight: 'normal' },
        },
        legend: {
          data: legendData,
          top: 4, right: 12,
          textStyle: { color: fg, fontSize: 11 },
          itemWidth: 14, itemHeight: 8,
        },
        // No tooltip / axisPointer — the user doesn't need to read off
        // exact values from a hover crosshair, and the moving lines
        // are visually noisy. Re-enable here if a future plot view
        // (e.g. trace diagnostics) actually needs precise readouts.
        tooltip: { show: false },
        xAxis: {
          type: 'value',
          name: 'x', nameLocation: 'center', nameGap: 28,
          axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
          splitLine: { show: false },
          minInterval: discrete ? 1 : null,
          // Anchor the visible range to the histogram support — the
          // density curve, computed on a wider quantile-padded grid,
          // can extend a bit beyond; let echarts auto-fit so both fit.
        },
        yAxis: {
          type: 'value',
          name: discrete ? 'P(X=x)' : 'p(x)',
          nameLocation: 'center', nameGap: 45,
          axisLine: { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick: { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
          splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
          min: 0,
        },
        series: series,
      });
    },
  });
}
