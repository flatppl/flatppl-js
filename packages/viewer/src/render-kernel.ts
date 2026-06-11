// @flatppl/viewer — kernel-sample renderers —
//
// renderKernelSampleForCurrent is the entry; renderKernelSample-
// Measure renders an in-memory empirical measure after
// materialiseConcreteMeasure has done the substitution.
// renderFixedRecord handles a record-shaped fixed-phase value.

import type { Ctx, FixedRecordPlan, KernelSamplePlan } from './types';
import { nameSeed } from './orchestration.js';
import { materialiseConcreteMeasure, materialiseAppliedKernelByName } from './plot-plan.js';
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
  // source lookup per array-typed input (clm's feedInputs handles
  // scalar / array / record env values uniformly).
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
    //
    // A LAMBDA param types as `any` (no declared boundary type), so the
    // structural default would collapse a record input to a scalar and
    // every body field access dies ("get_field target is not a record")
    // — the k_model trap (`k_model = pars -> …` with `pars` mirroring
    // the module binding). Enrich the default's type from the axis
    // SOURCE binding, or from a module binding of the same KWARG name
    // (the intended mirror). Defaults only — semantics are untouched.
    let defType = inp.type;
    if (!defType || defType.kind === 'any' || defType.kind === 'deferred') {
      const dsBindings = ctx.derivationsState && ctx.derivationsState.bindings;
      const srcName = (ax.source && ax.source.kind === 'binding' && ax.source.name)
        || ax.kwargName;
      const srcBinding = dsBindings && srcName ? dsBindings.get(srcName) : null;
      if (srcBinding && srcBinding.inferredType) defType = srcBinding.inferredType;
    }
    env[inp.paramName] = defaultValueForInputType(defType);
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
  // We need the binding-source samples first to fill env entries; the
  // canonical lowering + materialisation runs after they land.
  showPlotMessage(ctx, 'Sampling…', { cancellable: true, hint: true });
  const planForCall = plan;
  // Cache hit: use previously-sampled measure directly.
  if (ctx.measureCache.has(cacheKey)) {
    return Promise.resolve(ctx.measureCache.get(cacheKey)).then(function(m) {
      if (ctx.currentPlotPlan !== planForCall) return;
      renderKernelSampleMeasure(ctx, m, plan);
    });
  }
  // Pre-materialise binding-typed input sources we already know about
  // (for a kernel with input `mu` whose source is `mu` in scope, we
  // want samples[0] of that binding to seed the env). Captured
  // outer-scope refs need NO pre-pass: the canonical lowering declares
  // them as `shared` inputs and matClm feeds them per-atom.
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
    // PRIMARY (CLM Phase 5b): one canonical lowering. `lowerMeasure`
    // expands the kernel's reified output, transitively inlines derived
    // value bindings down to the FED inputs (H5/H3 — a swept/fed
    // boundary reaches the leaves), applies the MC-marginal form when
    // the body is a generative composite, and DECLARES every input;
    // `matClm` (dispatched by materialiseMeasureIR on the clm node)
    // feeds the env values through the ONE feeding contract sample +
    // density share. Captured outer-scope refs are declared `shared`
    // inputs and feed per-atom — atom i of the kernel sample uses atom
    // i of every captured ref (spec §04: stochastic ancestors that
    // aren't boundary inputs participate in the kernel's randomness).
    const dst = ctx.derivationsState!;
    const lowCtx = {
      derivations: dst.derivations,
      bindings: dst.bindings,
      fixedValues: dst.fixedValues,
    };
    let clmNode: any = null;
    try {
      clmNode = FlatPPLEngine.clm.lowerMeasure(sig.body, lowCtx, { boundaries: env });
    } catch (_e) { clmNode = null; }
    const primary = clmNode
      ? materialiseConcreteMeasure(ctx, clmNode, ctx.SAMPLE_COUNT, nameSeed(ctx, plan.name))
      : Promise.reject(new Error('lowerMeasure returned null for ' + plan.name));
    return primary.catch(function(clmErr: any) {
      // Concrete-application route: an APPLIED composite kernel
      // (k_model / k_model_n — the body is a user-call of another
      // kernel) doesn't lower as a bare measure; synthesize
      // `<kernel>(<env point>)` over the RAW bindings, re-derive, and
      // materialise the synthetic binding through its by-name
      // derivation — the same path `model_dist = k_model(glob_pars)`
      // uses (gallery-verified).
      let applied: any;
      try {
        applied = FlatPPLEngine.orchestrator.deriveAppliedKernel(
          ctx.currentBindings, plan.name, sig, env);
      } catch (e) { applied = null; }
      if (applied && applied.derivations && applied.derivations[applied.name]) {
        return materialiseAppliedKernelByName(ctx, applied, ctx.SAMPLE_COUNT, nameSeed(ctx, plan.name));
      }
      // Neither route applies: surface the canonical-lowering error (the
      // legacy substitute-IR fallback is retired — CLM Phase 5b, removed
      // after interactive gallery verification).
      return Promise.reject(clmErr);
    });
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
