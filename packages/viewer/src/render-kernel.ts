// @flatppl/viewer — kernel-sample renderers —
//
// renderKernelSampleForCurrent is the entry; renderKernelSample-
// Measure renders an in-memory empirical measure after
// materialiseConcreteMeasure has done the substitution.
// renderFixedRecord handles a record-shaped fixed-phase value.

import type { Ctx, FixedRecordPlan, KernelSamplePlan } from './types';
import { nameSeed } from './orchestration.js';
import { materialiseConcreteMeasure } from './plot-plan.js';
import { buildPresetControl } from './render-controls.js';

import { getMeasure, tryGetMeasure } from './engine-facade.js';
import { activePresetFor } from './overrides.js';
import { showPlotMessage } from './render-frame.js';
import { renderConstantRecord } from './render-record.js';
import { arrayInputLength, defaultValueForInputType, esc } from './util.js';
import { renderEmpiricalMeasure } from './render-samples.js';
export function renderFixedRecord(ctx: Ctx, plan: FixedRecordPlan) {
  showPlotMessage(ctx, 'Loading…', { hint: true });
  const planForCall = plan;
  getMeasure(ctx, plan.name).then(function(measure: any) {
    if (ctx.currentPlotPlan !== planForCall) return;
    renderConstantRecord(ctx, measure, plan.name);
  }).catch(function(err: any) {
    if (ctx.currentPlotPlan !== planForCall) return;
    showPlotMessage(ctx, 'Failed to load <strong>' + esc(plan.name) + '</strong>: '
      + esc(err && err.message || String(err)));
  });
}

export function renderKernelSampleForCurrent(ctx: Ctx) {
  const planAny = ctx.currentPlotPlan;
  if (!planAny || planAny.mode !== 'kernel-sample') return;
  const plan: KernelSamplePlan = planAny;
  const sig = plan.signature;
  const inputByKwarg: Record<string, any> = {};
  for (let k = 0; k < sig.inputs.length; k++) {
    inputByKwarg[sig.inputs[k].kwargName] = sig.inputs[k];
  }
  const active = activePresetFor(ctx, plan);
  // Cache key embeds the active preset's values directly so two
  // states of the same preset (with vs. without overrides, or
  // two different override sets) don't collide on cached
  // samples. Stable JSON suffices for our short kwarg lists.
  const cacheKey = plan.name + '|kernel-sample|' + (plan.presetName || '')
    + '|' + JSON.stringify(active.values || {});
  // Build the input env (paramName → number | array). Auto values
  // for axes not covered by the active preset (incl. modified
  // overrides) come from source-binding samples (or type-aware
  // defaults for placeholder sources).
  //
  // Array-typed inputs (e.g. `theta = elementof(cartpow(reals,8))`
  // or `theta ~ iid(Normal(0,1),8)` used as a kernel input) live
  // as ONE env entry: `env[paramName] = [v_1, ..., v_J]`.
  // `walkType` emits one axis per slot, but the kwargName is shared
  // across all slots — so we group by kwargName and do a single
  // source lookup per array-typed input. `substituteLocals`
  // handles the array env value by emitting a `vector(lit,…)` IR.
  const env: Record<string, any> = {};
  const bindingSourceLookups: Array<{ paramName: string; sourceName: string; arrayLen: number | null }> = [];
  const seenKwargs = new Set<string>();
  for (let a = 0; a < plan.axes.length; a++) {
    const ax = plan.axes[a];
    if (seenKwargs.has(ax.kwargName)) continue;
    seenKwargs.add(ax.kwargName);
    const inp = inputByKwarg[ax.kwargName];
    if (!inp) continue;
    const arrayLen = arrayInputLength(inp.type);
    if (active.values && Object.prototype.hasOwnProperty.call(active.values, ax.kwargName)) {
      env[inp.paramName] = active.values[ax.kwargName];
      continue;
    }
    // Structural default for the WHOLE input (scalar / array / record),
    // so a record-typed input (`pars = elementof(cartprod(a,b,mu))`)
    // seeds a record `{a,b,mu}` rather than a scalar. `arrayLen` is
    // still used below for the binding-source-sample override path.
    env[inp.paramName] = defaultValueForInputType(inp.type);
    if (ax.source && ax.source.kind === 'binding') {
      // Queue an empirical-sample lookup unconditionally —
      // tryGetMeasure soft-fails to null for sources that can't
      // produce samples (pure inputs like elementof). The
      // leaf-type default stays in env in that case.
      bindingSourceLookups.push({
        paramName: inp.paramName,
        sourceName: ax.source.name,
        arrayLen,
      });
    }
  }
  // Build the substituted measure IR. expandMeasureRefsInIR peels
  // any outer lawof and inlines measure-typed self-refs;
  // inlineForProfile (with all params named) inlines value-position
  // deterministic deps and rewrites self.<param> → %local.<param>;
  // substituteLocals replaces %local refs with their concrete env
  // values. Result: a self-contained measure IR with no refs.
  const paramNames = sig.inputs.map(function(inp: any) { return inp.paramName; });
  // We can do most of the IR work synchronously, but we need
  // the binding-source samples first to fill env entries.
  showPlotMessage(ctx, 'Sampling…', { cancellable: true, hint: true });
  const planForCall = plan;
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
  // Latent guard (engine-concepts §19): filter function-like and
  // fixed-phase source bindings from the samples[0] override loop.
  // The same array-collapse failure mode we fixed in render-profile
  // (flatppl-js commit e9984f3) would apply here once non-scalar
  // input axes are supported (F4a restriction in render-profile
  // lines 44-49). Today it's latent — fixing now keeps the code
  // resilient when F4a lifts.
  const bindings = ctx.derivationsState && ctx.derivationsState.bindings;
  Promise.all(bindingSourceLookups.map(function(s) {
    const src = bindings && bindings.get(s.sourceName);
    const isFunctionLike = FlatPPLEngine.materialiser.isFunctionLikeBinding(src);
    // Demand-driven (§17.4): "is this source fixed-phase?" is a PHASE
    // question — read binding.phase, don't `fixedValues.has` (which under
    // the lazy resolver would force-resolve the value just to answer it).
    const isFixedPhase = !!(src && src.phase === 'fixed');
    if (isFunctionLike || isFixedPhase) return null;
    return tryGetMeasure(ctx, s.sourceName);
  })).then(function(srcMeasures) {
    for (let i = 0; i < bindingSourceLookups.length; i++) {
      const lookup = bindingSourceLookups[i];
      const sm = srcMeasures[i];
      if (!sm || !sm.samples || sm.samples.length === 0) continue;
      if (lookup.arrayLen != null && sm.samples.length >= lookup.arrayLen) {
        // Array input: take atom-0's J-element slice (sm.samples is
        // atom-major flattened — atom 0 occupies the first arrayLen
        // entries). Convert to plain Array so substituteLocals'
        // length-based dispatch picks the array branch cleanly.
        env[lookup.paramName] = Array.from(sm.samples.slice(0, lookup.arrayLen)) as number[];
      } else {
        env[lookup.paramName] = sm.samples[0];
      }
    }
    let ir = sig.body;
    ir = FlatPPLEngine.orchestrator.expandMeasureRefsInIR(
      ir, ctx.derivationsState!.derivations);
    // expandMeasureRefsInIR fails closed for refs whose derivation
    // was pruned by buildDerivations (e.g. `x` here, because its
    // distIR depends on the parameterized `mu`). The kernel-sample
    // path substitutes that parameter via env at materialise time,
    // so it still needs the structural shape. Re-run with the
    // bindings fallback to recover from binding.ir directly.
    if (ir && ir.kind === 'ref' && ir.ns === 'self') {
      const expanded = FlatPPLEngine.orchestrator.expandMeasureIR(
        ir.name, ctx.derivationsState!.derivations,
        undefined, ctx.derivationsState!.bindings);
      if (expanded) ir = expanded;
    }
    ir = FlatPPLEngine.orchestrator.inlineForProfile(
      ir, paramNames, ctx.derivationsState!.bindings, ctx.derivationsState!.derivations);
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

export function renderKernelSampleMeasure(ctx: Ctx, measure: any, plan: KernelSamplePlan) {
  // Always wire the input-selection toolbar when the plan has
  // axes — the "auto" option still carries useful information
  // even without user-declared presets, and the control stays
  // visible across bindings for consistency.
  const hasAxes = plan.axes && plan.axes.length > 0;
  // Kernel-sample renders a histogram of empirical draws with an
  // auto-fit x-axis range — the Domain selector (which drives a
  // swept x-axis range) doesn't apply here, so we only mount the
  // Inputs control.
  const toolbarBuilder = hasAxes
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
