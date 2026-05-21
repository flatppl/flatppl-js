// @flatppl/viewer — profile-plot renderer family (Phase 4e).
//
// renderProfilePlotForCurrent is the entry; buildProfileControls
// + buildProfileBottomRow build the axis-selector and lo/hi inputs;
// renderProfileLine renders the value-line for the swept axis;
// commitSliceX commits user edits to the x-axis bounds back into
// ctx.profileRangeCache.

import type { Ctx, ProfilePlan } from './types';
import { tryGetMeasure } from './engine-facade.js';
import { activeDomainRangesFor, activeFixedNamesFor, activePresetFor, computeAutoValues, ensureDomainOverrideFor, ensureOverrideFor, resolveSweepRange, setDomainOverrideFor, setOverrideFor } from './overrides.js';
import { colorForBinding } from './palette.js';
import { buildDomainControl, buildPresetControl } from './render-controls.js';
import { showPlotMessage } from './render-frame.js';
import { defaultRangeForLeafType, defaultValueForLeafType, esc, formatScalar } from './util.js';
import { sendWorker } from './worker.js';
import { renderPlotFrame } from './render-frame.js';
import { plotZoomOptions } from './util.js';
export function renderProfilePlotForCurrent(ctx: Ctx) {
  const planAny = ctx.currentPlotPlan;
  if (!planAny || planAny.mode !== 'profile') return;
  const plan: ProfilePlan = planAny;
  var sig = plan.signature;
  var axes = plan.axes;
  let sweepAxis: any = null;
  for (var i = 0; i < axes.length; i++) {
    if (axes[i].key === plan.sweepKey) { sweepAxis = axes[i]; break; }
  }
  if (!sweepAxis) {
    showPlotMessage(ctx, 'Profile plot: no axis selected for <strong>'
      + esc(plan.name) + '</strong>.', { hint: true });
    return;
  }
  // Mode dispatch. Kernels are routed to renderKernelSampleForCurrent
  // by buildPlotPlan; profile mode here only sees function /
  // likelihood bindings.
  var mode = sig.kind === 'function' ? 'function' : 'logdensity';
  // Build paramName ↔ kwargName index for env construction.
  var inputByKwarg: Record<string, any> = {};
  for (var k = 0; k < sig.inputs.length; k++) {
    inputByKwarg[sig.inputs[k].kwargName] = sig.inputs[k];
  }
  // Top-level-scalar-only restriction for F4a.
  for (var a = 0; a < axes.length; a++) {
    if (axes[a].path && axes[a].path.length > 0) {
      showPlotMessage(ctx, 'Profile plot: record / array inputs not yet supported — '
        + 'try a binding with scalar inputs only.',
        { hint: true });
      return;
    }
  }
  // Build fixedEnv with three-tier precedence:
  //   1. Active preset's value for the kwarg (highest) — base
  //      preset values overridden by (modified) overrides.
  //   2. Source binding's samples[0] for binding-source axes
  //      (resolved later, after materialisation).
  //   3. Type-aware default for the leaf type (lowest).
  // The activePreset lookup short-circuits the materialise path
  // for any axis whose kwarg the preset (incl. modified
  // overrides) covers — so when the user picks a preset, we
  // don't even fetch the source binding.
  var active = activePresetFor(ctx, plan);
  var fixedEnv: Record<string, any> = {};
  const nonSweptBindingSources: Array<{ paramName: string; sourceName: string }> = [];
  for (var a2 = 0; a2 < axes.length; a2++) {
    if (axes[a2].key === plan.sweepKey) continue;
    var inp = inputByKwarg[axes[a2].kwargName];
    if (!inp) continue;
    if (active.values && Object.prototype.hasOwnProperty.call(active.values, axes[a2].kwargName)) {
      fixedEnv[inp.paramName] = active.values[axes[a2].kwargName];
      continue;
    }
    fixedEnv[inp.paramName] = defaultValueForLeafType(axes[a2].leafType);
    if (axes[a2].source && axes[a2].source.kind === 'binding') {
      // tryGetMeasure (below) soft-fails to null for inputs and
      // any other source that has no samples — the leaf-type
      // default then stays in fixedEnv.
      nonSweptBindingSources.push({
        paramName: inp.paramName,
        sourceName: axes[a2].source.name,
      });
    }
  }
  var sweepInput = inputByKwarg[sweepAxis.kwargName];
  var sweepParamName = sweepInput && sweepInput.paramName;
  if (!sweepParamName) {
    showPlotMessage(ctx, 'Profile plot: cannot resolve sweep parameter.', { hint: true });
    return;
  }
  // For kernels / likelihoods we walk the kernel body via
  // traceeval — peel any outer lawof and substitute self-refs to
  // other measure bindings via expandMeasureRefsInIR (the same
  // helper bayesupdate uses). Functions evaluate the body
  // verbatim through evaluateExpr.
  var ir = sig.body;
  // Multi-output: extract the sub-IR for the currently-selected
  // output leaf. For scalar outputs (single leaf, empty path)
  // this is a no-op pass-through.
  if (plan.outputs && plan.outputs.length > 1 && plan.outputKey) {
    let selectedOut: any = null;
    for (var oj = 0; oj < plan.outputs.length; oj++) {
      if (plan.outputs[oj].key === plan.outputKey) {
        selectedOut = plan.outputs[oj];
        break;
      }
    }
    if (selectedOut) {
      var extracted = FlatPPLEngine.orchestrator.extractOutputIR(
        ir, selectedOut.path);
      if (extracted) ir = extracted;
    }
  }
  if (mode === 'logdensity') {
    ir = FlatPPLEngine.orchestrator.expandMeasureRefsInIR(
      ir, ctx.derivationsState!.derivations);
  }
  // Propagate the swept axis (and other params) through transitive
  // deterministic deps. Without this, e.g. sweeping `theta1` in a
  // kernel whose body references `a = c * theta1` leaves `a` at a
  // single fixed value (collected once by the pre-materialise
  // step) — the plot is flat because the swept axis never reaches
  // the leaf. inlineForProfile inlines `a`'s IR into the body and
  // rewrites self.theta1 → %local.theta1.
  var paramNames = sig.inputs.map(function(inp: any) { return inp.paramName; });
  ir = FlatPPLEngine.orchestrator.inlineForProfile(
    ir, paramNames, ctx.derivationsState!.bindings, ctx.derivationsState!.derivations);
  var POINT_COUNT = 200;
  showPlotMessage(ctx, 'Profiling…', { cancellable: true, hint: true });
  var planForCall = plan;
  // The body may reference other bindings via (ref self <name>) —
  // e.g. `f_a = functionof(c * _par_, ...)` where `c` is an outer
  // literal. Pre-materialise those, take their samples[0] as a
  // single fixed value, and merge into fixedEnv. F4b will let
  // the user pick a different atom or override these. For
  // stochastic self refs this picks the first atom, which is
  // arbitrary but deterministic — a "good enough" first cut.
  const selfRefs: string[] = [];
  FlatPPLEngine.orchestrator.collectSelfRefs(ir).forEach(function(n: string) {
    selfRefs.push(n);
  });
  // Range resolution per (binding, axis, domain):
  //   1. Active domain has a range for the sweep kwarg (named
  //      source binding, override, or both) → use it.
  //   2. profileRangeCache hit for the auto-fit of this
  //      (binding, axis, domainName) → reuse.
  //   3. Otherwise compute via resolveSweepRange and cache the
  //      auto-fit. The cache stores auto-fits only.
  // Note: presetOverrides are orthogonal — they drive non-swept
  // input values, not x-axis ranges.
  var domainRanges = activeDomainRangesFor(ctx, plan);
  var domainRangeForSweep = domainRanges[plan.sweepKey];
  var cacheKey = plan.name + '|' + plan.sweepKey + '|D=' + (plan.domainName || '');
  var rangePromise;
  if (domainRangeForSweep) {
    rangePromise = Promise.resolve(
      [domainRangeForSweep.lo, domainRangeForSweep.hi]);
  } else {
    var cached = ctx.profileRangeCache.get(cacheKey);
    rangePromise = cached
      ? Promise.resolve([cached.lo, cached.hi])
      : resolveSweepRange(ctx, sweepAxis).then(function(r: any) {
          ctx.profileRangeCache.set(cacheKey, { lo: r[0], hi: r[1], fromAuto: true });
          return r;
        });
  }
  var rangeRef = [defaultRangeForLeafType(sweepAxis.leafType)];
  Promise.all([
    rangePromise,
    Promise.all(selfRefs.map(function(n) { return tryGetMeasure(ctx, n); })),
    Promise.all(nonSweptBindingSources.map(function(s) {
      return tryGetMeasure(ctx, s.sourceName);
    })),
  ]).then(function(arr) {
    rangeRef[0] = arr[0];
    var measures = arr[1];
    for (var i = 0; i < selfRefs.length; i++) {
      var m = measures[i];
      if (m && m.samples && m.samples.length > 0) {
        fixedEnv[selfRefs[i]] = m.samples[0];
      }
    }
    var srcMeasures = arr[2];
    for (var k = 0; k < nonSweptBindingSources.length; k++) {
      var sm = srcMeasures[k];
      if (sm && sm.samples && sm.samples.length > 0) {
        fixedEnv[nonSweptBindingSources[k].paramName] = sm.samples[0];
      }
    }
    // Integer-typed sweep axis (e.g. a count parameter, a
    // Bernoulli k, a Poisson observation): only integer x values
    // are mathematically meaningful. Snap the range to integer
    // bounds and pick a sweep count so profileN's evenly-spaced
    // grid x_i = lo + (hi−lo)·i/(n−1) lands on integers exactly
    // (n = hi−lo+1 with integer lo, hi). Cap at POINT_COUNT for
    // very wide ranges; renderProfileLine then draws a step
    // plot rather than smoothing between integer values.
    var pointCount = POINT_COUNT;
    var isIntegerAxis = sweepAxis.leafType
      && sweepAxis.leafType.kind === 'scalar'
      && sweepAxis.leafType.prim === 'integer';
    if (isIntegerAxis) {
      var ilo = Math.ceil(rangeRef[0][0]);
      var ihi = Math.floor(rangeRef[0][1]);
      if (ihi >= ilo) {
        rangeRef[0] = [ilo, ihi];
        pointCount = Math.min(ihi - ilo + 1, POINT_COUNT);
      }
    }
    // For likelihood profile plots, resolve the obs IR to a JS
    // value at sample time. sig.obsIR is set by
    // signatureOfLikelihood; non-likelihood signatures don't
    // carry one. Resolution failures (e.g. an observation we
    // can't materialise) propagate as a clean plot-time error,
    // same as the bayesupdate path.
    var observed;
    if (sig.obsIR != null) {
      observed = FlatPPLEngine.orchestrator.resolveIRToValue(
        sig.obsIR, ctx.derivationsState!.bindings, ctx.derivationsState!.fixedValues);
    }
    return sendWorker(ctx, {
      type: 'profileN',
      ir: ir,
      sweepName: sweepParamName,
      range: rangeRef[0],
      count: pointCount,
      mode: mode,
      fixedEnv: fixedEnv,
      observed: observed,
      tally: 'clamped',
    });
  }).then(function(reply) {
    if (!reply) return;
    if (ctx.currentPlotPlan !== planForCall) return;
    renderProfileLine(ctx, reply.samples, rangeRef[0], plan, sweepAxis);
  }).catch(function(err) {
    if (ctx.currentPlotPlan !== planForCall) return;
    showPlotMessage(ctx, 'Profile plot failed: ' + esc(err && err.message || String(err)));
  });
}

/**
 * Build the profile-plot toolbar controls (axis dropdown, preset
 * dropdown, y-cutoff selector, x-range inputs). Returns a
 * DocumentFragment that the caller hands to renderPlotFrame as
 * `toolbarControls`. Logic mirrors the original inline build; only
 * the styling ctx.host moved.
 */
export function buildProfileControls(ctx: Ctx, plan: ProfilePlan, range: any) {
  var frag = document.createDocumentFragment();
  var isLogDensity = plan.signature.kind === 'kernel'
                  || plan.signature.kind === 'likelihood';
  var hasAxes = plan.axes && plan.axes.length > 1;
  // "hasInputs" tells whether the plan has any input axes at all
  // (single or multiple). The input/preset dropdown is shown
  // whenever the callable has inputs, even with no user-declared
  // presets — the "auto" option in buildPresetControl still
  // surfaces the values being used.
  var hasInputs = plan.axes && plan.axes.length > 0;
  var hasMultiOutput = plan.outputs && plan.outputs.length > 1;
  // Output selector — appears for callables whose specialized
  // output is multi-leaf (record / tuple / array). Single-leaf
  // outputs (scalar functions) skip this control. Picking a
  // leaf rewrites the IR sent to profileN to the matching
  // sub-expression of the body, so the sweep evaluates that
  // specific scalar component along the chosen input axis.
  if (hasMultiOutput) {
    var outLabel = document.createElement('label');
    outLabel.textContent = 'Output:';
    outLabel.style.opacity = '0.6';
    outLabel.style.marginRight = '0.25em';
    var outSelect = document.createElement('select');
    outSelect.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
    outSelect.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
    outSelect.style.border = '1px solid var(--vscode-dropdown-border, #555)';
    outSelect.style.padding = '2px 4px';
    outSelect.style.fontSize = '1em';
    for (var oi = 0; oi < plan.outputs.length; oi++) {
      var oOpt = document.createElement('option');
      oOpt.value = plan.outputs[oi].key;
      oOpt.textContent = plan.outputs[oi].label || '<scalar>';
      if (plan.outputs[oi].key === plan.outputKey) oOpt.selected = true;
      outSelect.appendChild(oOpt);
    }
    outSelect.addEventListener('change', function(e) {
      plan.outputKey = (e.target as HTMLSelectElement).value;
      renderProfilePlotForCurrent(ctx);
    });
    frag.appendChild(outLabel);
    frag.appendChild(outSelect);
  }
  // Output-side controls go LEFT of Inputs in the toolbar.
  // Rel. cut-off is a y-axis (output) display filter for
  // log-density / log-likelihood plots, so it belongs with the
  // Output: selector rather than between Inputs: and the
  // x-Axis block. The order ends up reading:
  //   [Output:] [Rel. cut-off:] [Inputs:] [x-Axis: lo ≤ axis ≤ hi]
  // — output-related ←left, input-related→right.
  if (isLogDensity) {
    if (plan.yCutoff == null) plan.yCutoff = 100;
    var cutLabel = document.createElement('label');
    cutLabel.textContent = 'Rel. cut-off:';
    cutLabel.style.opacity = '0.6';
    cutLabel.style.marginRight = '0.25em';
    var cutSel = document.createElement('select');
    cutSel.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
    cutSel.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
    cutSel.style.border = '1px solid var(--vscode-dropdown-border, #555)';
    cutSel.style.padding = '2px 4px';
    cutSel.style.fontSize = '1em';
    var cutoffs = [10, 100, 1000, 10000];
    for (var ci = 0; ci < cutoffs.length; ci++) {
      var copt = document.createElement('option');
      copt.value = String(cutoffs[ci]);
      copt.textContent = '−' + cutoffs[ci];
      if (cutoffs[ci] === plan.yCutoff) copt.selected = true;
      cutSel.appendChild(copt);
    }
    cutSel.addEventListener('change', function(e) {
      plan.yCutoff = parseInt((e.target as HTMLSelectElement).value, 10);
      renderProfilePlotForCurrent(ctx);
    });
    frag.appendChild(cutLabel);
    frag.appendChild(cutSel);
  }
  if (hasInputs) {
    // Reuse buildPresetControl so the option text (name + value
    // record) and styling stay consistent with the kernel-sample
    // path. Always shown when there are input axes — the "auto"
    // option carries the default values even without user-
    // declared presets.
    frag.appendChild(buildPresetControl(ctx, plan, function() {
      renderProfilePlotForCurrent(ctx);
    }));
    // Domain selector — same row, drives x-axis ranges from
    // cartprod(...) bindings. Falls back to a no-op fragment when
    // the binding has no axes; we already returned early for that.
    frag.appendChild(buildDomainControl(ctx, plan, function() {
      renderProfilePlotForCurrent(ctx);
    }));
  }
  // The lo/hi limit inputs live under the plot now (see
  // buildProfileBottomRow). The toolbar carries only the axis
  // selector (or static label for single-axis), as
  //   x-Axis: <axis selector | static name>
  var xBlock = document.createElement('span');
  xBlock.style.display = 'inline-flex';
  xBlock.style.alignItems = 'center';
  xBlock.style.gap = '0.35em';

  var xLabel = document.createElement('label');
  xLabel.textContent = 'x-Axis:';
  xLabel.style.opacity = '0.6';

  // Hide kwargs the active preset wrapped in `fixed(...)` —
  // spec §03 marks those as "held constant during optimization",
  // so offering them as a sweep axis contradicts the annotation.
  // Edge case: if filtering removes every option (every kwarg was
  // marked fixed), fall back to the unfiltered list so the user
  // can still pick something; the absence of any sweepable axis
  // is a model-authoring decision, not a viewer constraint.
  var fixedNames = activeFixedNamesFor(ctx, plan);
  var visibleAxes = plan.axes;
  if (fixedNames && fixedNames.size > 0) {
    var kept = plan.axes.filter(function(a) {
      return !fixedNames.has(a.key) || a.key === plan.sweepKey;
    });
    if (kept.length > 0) visibleAxes = kept;
  }

  var axisEl;
  if (hasAxes) {
    axisEl = document.createElement('select');
    axisEl.style.background = 'var(--vscode-dropdown-background, #3c3c3c)';
    axisEl.style.color = 'var(--vscode-dropdown-foreground, #cccccc)';
    axisEl.style.border = '1px solid var(--vscode-dropdown-border, #555)';
    axisEl.style.padding = '2px 4px';
    axisEl.style.fontSize = '1em';
    axisEl.title = 'Axis to sweep';
    for (var ai = 0; ai < visibleAxes.length; ai++) {
      var opt = document.createElement('option');
      opt.value = visibleAxes[ai].key;
      opt.textContent = visibleAxes[ai].label;
      if (visibleAxes[ai].key === plan.sweepKey) opt.selected = true;
      axisEl.appendChild(opt);
    }
    axisEl.addEventListener('change', function(e) {
      plan.sweepKey = (e.target as HTMLSelectElement).value;
      renderProfilePlotForCurrent(ctx);
    });
  } else if (plan.axes && plan.axes.length === 1) {
    axisEl = document.createElement('span');
    axisEl.textContent = plan.axes[0].label;
    axisEl.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
    axisEl.style.padding = '2px 4px';
    axisEl.style.opacity = '0.85';
  }

  xBlock.appendChild(xLabel);
  if (axisEl) xBlock.appendChild(axisEl);
  frag.appendChild(xBlock);
  return frag;
}

/**
 * Row that sits under the echarts plot in profile mode:
 *
 *   [lo input]    <axis name>    [hi input]
 *
 * Wrapper-edge-aligned (lo pinned left, hi pinned right; axis
 * name centred). The axis name comes from plan.axes[…].label;
 * the (default = V) decoration is added in a later step.
 * Range edits commit to ctx.profileRangeCache keyed by
 * (binding, sweepKey, preset) — same store as before, same
 * effect on the re-render path.
 */
export function buildProfileBottomRow(ctx: Ctx, plan: ProfilePlan, range: any) {
  var fg = getComputedStyle(document.body).color || '#ccc';
  var row = document.createElement('div');
  row.style.display = 'flex';
  row.style.alignItems = 'baseline';
  row.style.gap = '0.6em';
  // Horizontal padding matches the echarts grid insets used by
  // renderProfileLine (grid.left:60, grid.right:25). With this
  // padding the lo input's left edge aligns with the chart's
  // leftmost data extent and the hi input's right edge with the
  // rightmost — so the values read as labels of the extents
  // rather than floating off in the pane margins.
  row.style.padding = '0.2em 25px 0.4em 60px';
  row.style.fontFamily = 'var(--vscode-font-family, sans-serif)';
  row.style.fontSize = '0.92em';

  var xLoInput = document.createElement('input');
  xLoInput.type = 'number'; xLoInput.step = 'any';
  xLoInput.value = formatScalar(range[0]);
  xLoInput.title = 'x-axis lower limit';
  var xHiInput = document.createElement('input');
  xHiInput.type = 'number'; xHiInput.step = 'any';
  xHiInput.value = formatScalar(range[1]);
  xHiInput.title = 'x-axis upper limit';
  [xLoInput, xHiInput].forEach(function(inp) {
    inp.style.background = 'var(--vscode-input-background, #3c3c3c)';
    inp.style.color = 'var(--vscode-input-foreground, #cccccc)';
    inp.style.border = '1px solid var(--vscode-input-border, #555)';
    inp.style.padding = '2px 4px';
    inp.style.fontSize = '1em';
    inp.style.width = '6.5em';
    inp.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';
  });

  // Limit edits commit into the active *domain* override (auto
  // pseudo-domain on plan.domainAutoOverride, named domains in
  // module-wide domainOverrides). Per-kwarg ranges live here
  // because a cartprod(...) source binding spans every input
  // axis; rebuildDerivations reconciles each kwarg independently
  // against the source intervals on the next refresh.
  var commitRange = function() {
    var newLo = parseFloat(xLoInput.value);
    var newHi = parseFloat(xHiInput.value);
    if (!Number.isFinite(newLo) || !Number.isFinite(newHi) || newLo >= newHi) {
      xLoInput.value = formatScalar(range[0]);
      xHiInput.value = formatScalar(range[1]);
      return;
    }
    var key = plan.sweepKey;
    if (!key) return;
    var entry = ensureDomainOverrideFor(ctx, plan);
    entry.ranges = entry.ranges || {};
    entry.ranges[key] = { lo: newLo, hi: newHi };
    setDomainOverrideFor(ctx, plan, entry);
    renderProfilePlotForCurrent(ctx);
  };
  xLoInput.addEventListener('change', commitRange);
  xHiInput.addEventListener('change', commitRange);

  // Centred axis name, with a "(default = V)" suffix showing
  // the value this axis is pinned at when another axis is the
  // sweep. V comes from the active preset's user-set values
  // when present (named preset value, or click-set override
  // on a modified entry); otherwise from the same auto
  // computation the dropdown's "auto: …" label uses (type
  // default, or samples[0] from a source binding when cached).
  // Lets the user read the current slice and "navigate" through
  // axes by clicking + switching the sweep direction.
  var axisName = plan.sweepKey;
  var sweepKwarg = plan.sweepKey;
  if (plan.axes) {
    for (var i = 0; i < plan.axes.length; i++) {
      if (plan.axes[i].key === plan.sweepKey) {
        axisName = plan.axes[i].label;
        sweepKwarg = plan.axes[i].kwargName;
        break;
      }
    }
  }
  var defaultText = '';
  if (sweepKwarg) {
    var activeForLabel = activePresetFor(ctx, plan);
    var v;
    if (activeForLabel.values
        && Object.prototype.hasOwnProperty.call(activeForLabel.values, sweepKwarg)) {
      v = activeForLabel.values[sweepKwarg];
    } else {
      var av = computeAutoValues(ctx, plan);
      v = av[sweepKwarg];
    }
    if (v !== undefined && v !== null) {
      defaultText = '  (default = ' + formatScalar(v) + ')';
    }
  }
  var nameSpan = document.createElement('span');
  nameSpan.textContent = axisName + defaultText;
  nameSpan.style.flex = '1';
  nameSpan.style.textAlign = 'center';
  nameSpan.style.color = fg;
  nameSpan.style.opacity = '0.75';
  nameSpan.style.fontFamily = 'var(--vscode-editor-font-family, monospace)';

  row.appendChild(xLoInput);
  row.appendChild(nameSpan);
  row.appendChild(xHiInput);
  return row;
}

export function renderProfileLine(ctx: Ctx, values: any, range: any, plan: ProfilePlan, sweepAxis: any) {
  var fg = getComputedStyle(document.body).color || '#ccc';
  var color = colorForBinding(ctx, ctx.currentPlotBindingName);
  var n = values.length;
  var lo = range[0], hi = range[1];
  // Integer-typed sweep axis: only integer x values are
  // mathematically meaningful, so render as a step plot
  // (piecewise-constant between adjacent integers) plus a dot at
  // each evaluated point. Echarts' step:'middle' on a line series
  // gives the right shape — the value at integer k extends to
  // (k − 0.5, k + 0.5).
  var isIntegerAxis = sweepAxis.leafType
    && sweepAxis.leafType.kind === 'scalar'
    && sweepAxis.leafType.prim === 'integer';
  // For log-density / log-likelihood, find the maximum finite
  // value across the sweep and clamp the y-axis to
  // [yMax − cutoff, yMax]. Below that we show no-man's-land —
  // values get clamped to the cutoff line so the curve stays
  // visible rather than disappearing under a -∞ singularity.
  var yMax = -Infinity, yMin = Infinity;
  for (var yi = 0; yi < n; yi++) {
    var v = values[yi];
    if (Number.isFinite(v)) {
      if (v > yMax) yMax = v;
      if (v < yMin) yMin = v;
    }
  }
  let yClipMin: number | null = null, yClipMax: number | null = null;
  if ((plan.signature.kind === 'kernel' || plan.signature.kind === 'likelihood')
      && Number.isFinite(yMax)) {
    var cut = (plan.yCutoff != null) ? plan.yCutoff : 100;
    yClipMin = yMax - cut;
    // Add a small upper headroom (~5% of the cutoff) so the peak
    // doesn't sit on the chart's top edge.
    yClipMax = yMax + 0.05 * cut;
  }
  // Build (x, y) pairs. Three cases produce gaps (null) so the
  // line renders as broken segments rather than misleading
  // connections:
  //   1. NaN / ±∞                        — domain-of-definition holes.
  //   2. y < yClipMin (below the cutoff) — same conceptual gap:
  //      "no useful information here". Drawing them at the floor
  //      would suggest the curve sits at the cutoff value there,
  //      which it doesn't. echarts honours { connectNulls: false }
  //      and stops the line at each null.
  var data = new Array(n);
  for (var i = 0; i < n; i++) {
    var t = n === 1 ? 0 : i / (n - 1);
    var x = lo + t * (hi - lo);
    var y = values[i];
    if (!Number.isFinite(y) || (yClipMin != null && y < yClipMin)) {
      data[i] = [x, null];
    } else {
      data[i] = [x, y];
    }
  }
  var titleText = (ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'profile')
    + ' — ' + esc(sweepAxis.label);
  // Per spec / convention: a kernel with obs fixed (likelihoodof)
  // computes the log-LIKELIHOOD; a bare kernel (or any other
  // measure-bodied callable) computes the log-DENSITY at obs.
  // Functions evaluate to a value.
  var legendLabel = plan.signature.kind === 'function'   ? 'value'
                   : plan.signature.kind === 'likelihood' ? 'log-likelihood'
                   :                                        'log-density';
  // Profile plots evaluate a function/kernel at a grid of points;
  // they aren't sampled empirical measures, so no measure / N+ESS
  // readout. The toolbar carries the controls only.
  renderPlotFrame(ctx, {
    toolbarControls: buildProfileControls(ctx, plan, range),
    bottomRow:       buildProfileBottomRow(ctx, plan, range),
    chartCallback: function(chartHost: any) {
      ctx.plotEchart = echarts.init(chartHost);
      var zoomOpts = plotZoomOptions(fg);
      ctx.plotEchart.setOption({
        animation: false,
        dataZoom: zoomOpts.dataZoom,
        toolbox: zoomOpts.toolbox,
        // The axis-name is rendered by the bottom-row instead of
        // echarts (so the lo/hi limit inputs can sit beside it,
        // wrapper-edge aligned). bottom can shrink because
        // there's no axis-name eating vertical space inside the
        // grid.
        grid: { left: 60, right: 25, top: 30, bottom: 25, containLabel: false },
        title: {
          text: titleText,
          left: 'center', top: 4,
          textStyle: { color: fg, fontSize: 13, fontWeight: 'normal' },
        },
        legend: {
          data: [legendLabel],
          top: 4, right: 12,
          textStyle: { color: fg, fontSize: 11 },
          itemWidth: 14, itemHeight: 8,
        },
        // Vertical-only crosshair on hover. Communicates "click to
        // slice on this x-axis value" — y is irrelevant, so no
        // horizontal indicator. Keep the tooltip text minimal:
        // just the rounded x value, so users can read off the
        // value they're about to commit before clicking.
        tooltip: {
          show: true,
          trigger: 'axis',
          triggerOn: 'mousemove',
          axisPointer: { type: 'line', snap: false, animation: false,
                         label: { show: false } },
          backgroundColor: 'rgba(40, 40, 40, 0.92)',
          borderColor: 'rgba(120, 120, 120, 0.6)',
          borderWidth: 1,
          padding: [3, 6],
          textStyle: { color: fg, fontSize: 11 },
          formatter: function(params: any) {
            if (!params || !params.length) return '';
            var x = params[0].axisValue;
            return sweepAxis.label + ' = ' + formatScalar(x);
          },
        },
        xAxis: {
          type: 'value',
          name: '',
          min: lo, max: hi,
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
          splitLine: { show: false },
        },
        yAxis: Object.assign({
          type: 'value',
          axisLine:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisTick:  { lineStyle: { color: fg, opacity: 0.4 } },
          axisLabel: { color: fg, opacity: 0.6, formatter: formatScalar },
          splitLine: { lineStyle: { color: fg, opacity: 0.15 } },
          // No y-axis pointer — only the x-axis crosshair is
          // meaningful for click-to-slice.
          axisPointer: { show: false },
        }, yClipMin != null ? { min: yClipMin, max: yClipMax } : {}),
        series: [{
          name: legendLabel,
          type: 'line', data: data,
          // Integer-axis profile: piecewise-constant 'middle' step
          // (value at integer k extends to k±0.5) with dots at
          // each evaluated point so the discrete grid is visible.
          // Continuous axis: smooth line, no markers.
          step: isIntegerAxis ? 'middle' : false,
          symbol: isIntegerAxis ? 'circle' : 'none',
          symbolSize: 5,
          lineStyle: { color: color, width: 2 },
          itemStyle: isIntegerAxis ? { color: color } : undefined,
          connectNulls: false,
        }],
      });

      // Click-to-slice. The click lands at some x in the chart's
      // grid; we convert pixel → data, then commit that x as the
      // sweep-axis value in a modified preset (creating one if
      // we were on the base). The plot itself doesn't change
      // (the sweep axis still spans lo..hi) but the user has
      // pinned this axis's value for when they switch the
      // sweep direction to another axis. The under-plot axis
      // name picks up `(default = V)` immediately.
      ctx.plotEchart.getZr().on('click', function(ev: any) {
        var pt = [ev.offsetX, ev.offsetY];
        if (!ctx.plotEchart.containPixel('grid', pt)) return;
        // xAxisIndex finder takes a scalar pixel and returns a
        // scalar data value (echarts API quirk — only the grid/
        // series finders take arrays). Passing the [x,y] array
        // here returns NaN.
        var clickedX = ctx.plotEchart.convertFromPixel({ xAxisIndex: 0 }, ev.offsetX);
        if (!Number.isFinite(clickedX)) return;
        commitSliceX(ctx, plan, clickedX);
        renderProfilePlotForCurrent(ctx);
      });
    },
  });
}

export function commitSliceX(ctx: Ctx, plan: ProfilePlan, x: number) {
  if (!plan || !plan.axes) return;
  let kwarg: string | null = null;
  for (var i = 0; i < plan.axes.length; i++) {
    if (plan.axes[i].key === plan.sweepKey) {
      kwarg = plan.axes[i].kwargName;
      break;
    }
  }
  if (!kwarg) return;
  var entry = ensureOverrideFor(ctx, plan);
  entry.values[kwarg] = x;
  setOverrideFor(ctx, plan, entry);
}
