// @flatppl/viewer — kernel-sample renderers (Phase 4f).
//
// renderKernelSampleForCurrent is the entry; renderKernelSample-
// Measure renders an in-memory empirical measure after
// materialiseConcreteMeasure has done the substitution.
// renderFixedRecord handles a record-shaped fixed-phase value.

import { nameSeed } from './orchestration.js';
import { materialiseConcreteMeasure } from './plot-plan.js';
import { buildPresetControl } from './render-controls.js';

import { getMeasure, tryGetMeasure } from './engine-facade.js';
import { activePresetFor } from './overrides.js';
import { showPlotMessage } from './render-frame.js';
import { renderConstantRecord } from './render-record.js';
import { defaultValueForLeafType, esc } from './util.js';
import { renderEmpiricalMeasure } from './render-samples.js';
export function renderFixedRecord(ctx, plan) {
  showPlotMessage(ctx, 'Loading…', { hint: true });
  var planForCall = plan;
  getMeasure(ctx, plan.name).then(function(measure) {
    if (ctx.currentPlotPlan !== planForCall) return;
    renderConstantRecord(ctx, measure, plan.name);
  }).catch(function(err) {
    if (ctx.currentPlotPlan !== planForCall) return;
    showPlotMessage(ctx, 'Failed to load <strong>' + esc(plan.name) + '</strong>: '
      + esc(err && err.message || String(err)));
  });
}

export function renderKernelSampleForCurrent(ctx) {
  var plan = ctx.currentPlotPlan;
  if (!plan || plan.mode !== 'kernel-sample') return;
  var sig = plan.signature;
  var inputByKwarg = {};
  for (var k = 0; k < sig.inputs.length; k++) {
    inputByKwarg[sig.inputs[k].kwargName] = sig.inputs[k];
  }
  // Restrict (for now) to top-level scalar inputs — same limit
  // as the function/likelihood profile path.
  for (var ai = 0; ai < plan.axes.length; ai++) {
    if (plan.axes[ai].path && plan.axes[ai].path.length > 0) {
      showPlotMessage(ctx, 'Kernel plot: record / array inputs not yet supported '
        + '— try a kernel with scalar inputs only.',
        { hint: true });
      return;
    }
  }
  var active = activePresetFor(ctx, plan);
  // Cache key embeds the active preset's values directly so two
  // states of the same preset (with vs. without overrides, or
  // two different override sets) don't collide on cached
  // samples. Stable JSON suffices for our short kwarg lists.
  var cacheKey = plan.name + '|kernel-sample|' + (plan.presetName || '')
    + '|' + JSON.stringify(active.values || {});
  // Build the input env (paramName → number). Auto values for
  // axes not covered by the active preset (incl. modified
  // overrides) come from source-binding samples[0] (or type-
  // aware defaults for placeholder sources).
  const env: Record<string, number | boolean> = {};
  const bindingSourceLookups: Array<{ paramName: string; sourceName: string }> = [];
  for (var a = 0; a < plan.axes.length; a++) {
    var ax = plan.axes[a];
    var inp = inputByKwarg[ax.kwargName];
    if (!inp) continue;
    if (active.values && Object.prototype.hasOwnProperty.call(active.values, ax.kwargName)) {
      env[inp.paramName] = active.values[ax.kwargName];
      continue;
    }
    env[inp.paramName] = defaultValueForLeafType(ax.leafType);
    if (ax.source && ax.source.kind === 'binding') {
      // Queue an empirical-sample lookup unconditionally —
      // tryGetMeasure soft-fails to null for sources that can't
      // produce samples (pure inputs like elementof). The
      // leaf-type default stays in env in that case.
      bindingSourceLookups.push({
        paramName: inp.paramName,
        sourceName: ax.source.name,
      });
    }
  }
  // Build the substituted measure IR. expandMeasureRefsInIR peels
  // any outer lawof and inlines measure-typed self-refs;
  // inlineForProfile (with all params named) inlines value-position
  // deterministic deps and rewrites self.<param> → %local.<param>;
  // substituteLocals replaces %local refs with their concrete env
  // values. Result: a self-contained measure IR with no refs.
  var paramNames = sig.inputs.map(function(inp) { return inp.paramName; });
  // We can do most of the IR work synchronously, but we need
  // the binding-source samples first to fill env entries.
  showPlotMessage(ctx, 'Sampling…', { cancellable: true, hint: true });
  var planForCall = plan;
  // Cache hit: use previously-sampled measure directly.
  if (ctx.measureCache.has(cacheKey)) {
    return Promise.resolve(ctx.measureCache.get(cacheKey)).then(function(m) {
      if (ctx.currentPlotPlan !== planForCall) return;
      renderKernelSampleMeasure(ctx, m, plan);
    });
  }
  // Two-phase pre-materialise:
  //   (1) binding-typed input sources we already know about (for
  //       a kernel with input `mu` whose source is `mu` in scope,
  //       we want samples[0] of that binding to seed the env);
  //   (2) self-refs captured from the outer scope by the kernel
  //       body (e.g. `sigma` referenced inside `iid(Normal(mu=0,
  //       sigma=sigma), 3)` even though `sigma` isn't a kernel
  //       input). These appear as (ref self sigma) after
  //       inlineForProfile because sigma is stochastic and so
  //       isn't inlined as a deterministic dep. substituteLocals
  //       only touches %local refs, so the materialise would
  //       otherwise fail with "unbound self reference".
  //
  // The actual captured self-refs are collected *after*
  // inlineForProfile because that pass inlines deterministic
  // deps. Anything still self-ref'd is genuinely a captured
  // stochastic/fixed dep from the outer scope.
  Promise.all(bindingSourceLookups.map(function(s) {
    return tryGetMeasure(ctx, s.sourceName);
  })).then(function(srcMeasures) {
    for (var i = 0; i < bindingSourceLookups.length; i++) {
      var sm = srcMeasures[i];
      if (sm && sm.samples && sm.samples.length > 0) {
        env[bindingSourceLookups[i].paramName] = sm.samples[0];
      }
    }
    var ir = sig.body;
    ir = FlatPPLEngine.orchestrator.expandMeasureRefsInIR(
      ir, ctx.derivationsState.derivations);
    // expandMeasureRefsInIR fails closed for refs whose derivation
    // was pruned by buildDerivations (e.g. `x` here, because its
    // distIR depends on the parameterized `mu`). The kernel-sample
    // path substitutes that parameter via env at materialise time,
    // so it still needs the structural shape. Re-run with the
    // bindings fallback to recover from binding.ir directly.
    if (ir && ir.kind === 'ref' && ir.ns === 'self') {
      var expanded = FlatPPLEngine.orchestrator.expandMeasureIR(
        ir.name, ctx.derivationsState.derivations,
        undefined, ctx.derivationsState.bindings);
      if (expanded) ir = expanded;
    }
    ir = FlatPPLEngine.orchestrator.inlineForProfile(
      ir, paramNames, ctx.derivationsState.bindings, ctx.derivationsState.derivations);
    ir = FlatPPLEngine.orchestrator.substituteLocals(ir, env);

    // Captured self-refs (outer-scope stochastic / fixed bindings
    // that aren't kernel inputs) are no longer collapsed to
    // samples[0] here. materialiseConcreteMeasure threads
    // refArrays through to the worker's sampleN — atom i of the
    // kernel sample uses atom i of every captured ref, matching
    // the per-atom semantics of the closed-measure getMeasure
    // path. Per spec §04, stochastic ancestors that aren't
    // boundary inputs participate in the kernel's randomness.
    return materialiseConcreteMeasure(ctx, ir, ctx.SAMPLE_COUNT, nameSeed(ctx, plan.name));
  }).then(function(measure) {
    if (ctx.currentPlotPlan !== planForCall) return;
    ctx.measureCache.set(cacheKey, measure);
    renderKernelSampleMeasure(ctx, measure, plan);
  }).catch(function(err) {
    if (ctx.currentPlotPlan !== planForCall) return;
    showPlotMessage(ctx, 'Kernel plot failed: ' + esc(err && err.message || String(err)));
  });
}

export function renderKernelSampleMeasure(ctx, measure, plan) {
  // Always wire the input-selection toolbar when the plan has
  // axes — the "auto" option still carries useful information
  // even without user-declared presets, and the control stays
  // visible across bindings for consistency.
  var hasAxes = plan.axes && plan.axes.length > 0;
  // Kernel-sample renders a histogram of empirical draws with an
  // auto-fit x-axis range — the Domain selector (which drives a
  // swept x-axis range) doesn't apply here, so we only mount the
  // Inputs control.
  var toolbarBuilder = hasAxes
    ? function() {
        return buildPresetControl(ctx, plan, function() {
          renderKernelSampleForCurrent(ctx);
        });
      }
    : null;
  renderEmpiricalMeasure(ctx, measure, {
    name: plan.name,
    mode: 'samples',
    discrete: false,
    analyticalIR: null,
    toolbarControls: toolbarBuilder,
  });
}
