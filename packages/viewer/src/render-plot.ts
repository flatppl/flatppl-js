// @flatppl/viewer — top-of-stack plot dispatchers —
//
// renderPlotForCurrent is the per-binding plot router (samples /
// density / corner / profile / array / kernel-sample), dispatching
// to the appropriate renderer family. updatePlotForBinding is the
// public hook the DAG tap handler + applySourceUpdate call to
// re-target the plot pane.

import { buildPlotPlan } from './plot-plan.js';
import { renderFixedRecord, renderKernelSampleForCurrent } from './render-kernel.js';

import { getMeasure } from './engine-facade.js';
import { applyRememberedSelections, rememberPlanSelections } from './overrides.js';
import { showPlotMessage, updatePlotProgress } from './render-frame.js';
import { renderProfilePlotForCurrent } from './render-profile.js';
import { buildInferenceControl, shouldDeferAutoSample } from './render-controls.js';
import { esc } from './util.js';
import { errorsForBinding } from './render-frame.js';
import { renderEmpiricalMeasure } from './render-samples.js';
import type { Ctx } from './types';
/**
 * Render the plot pane for the currently-focused binding.
 *
 * `opts.autoTrigger` marks an AUTOMATIC call — one not originating from the
 * explicit Sample button (e.g. a model edit re-focusing the same binding via
 * applySourceUpdate → focusNode). On such a call, a bayesupdate posterior
 * whose effective backend is a stateful sampler (MH/RAM/slice/emcee/AMIS/
 * SMC/nested/ESS — see SAMPLING_BACKENDS in render-controls.ts) is NOT
 * re-sampled; instead the pane shows a "press Sample" hint.
 * Explicit triggers (Sample button, DAG node click/drill-down, cog changes
 * once applied) omit `autoTrigger` and always sample. Cheap non-sampling
 * plots (IS / forward / tractable / array / matrix) are unaffected either
 * way and keep updating live on edit.
 */
export function renderPlotForCurrent(ctx: Ctx, opts?: { autoTrigger?: boolean }) {
  // The plot panel stays mounted whenever plotEnabled is true. When
  // the focused binding isn't plottable (lawof, modules, etc.) we
  // still show *something* — a "Not plottable" message — so the
  // panel doesn't appear/disappear under the user as they click
  // around the DAG.
  //
  // Type errors take priority over EVERYTHING. The orchestrator
  // is structural and may produce a derivation for a binding
  // whose body has a type error — e.g. weighted(exp(pow(M),1), N)
  // where pow(measure) is invalid. The plot pane must short-
  // circuit on errors before sampling, otherwise we'd render a
  // valid-looking empty histogram with NaN samples instead of
  // the actionable diagnostic.
  const name = ctx.currentPlotBindingName ? esc(ctx.currentPlotBindingName) : 'this binding';
  const typeErrors = errorsForBinding(ctx, ctx.currentPlotBindingName);
  if (typeErrors && typeErrors.length > 0) {
    let msg = '<strong>' + name + '</strong> is semantically invalid:'
      + '<ul>';
    for (let i = 0; i < typeErrors.length; i++) {
      msg += '<li style="color: #E57373;">' + esc(typeErrors[i].message) + '</li>';
    }
    msg += '</ul>';
    showPlotMessage(ctx, msg);
    return;
  }
  if (!ctx.currentPlotPlan) {
    if (ctx.currentState && ctx.currentState.targetName === ctx.MODULE_TARGET) {
      showPlotMessage(ctx, 'Click a binding in the graph to plot it.', { hint: true });
      return;
    }
    // Synthetic / internal nodes (anonymous lifted subexpressions,
    // placeholders, holes, lawof / kernelof / draw bridge nodes,
    // disintegration outputs that don't carry a user binding name)
    // fail the binding lookup in updatePlotForBinding, which sets
    // currentPlotBindingName=null. Surface a generic message here
    // — there's nothing user-meaningful to plot, and pointing at
    // a different binding would be guesswork.
    if (ctx.currentPlotBindingName == null) {
      showPlotMessage(ctx, 'Internal nodes are not plottable.', { hint: true });
      return;
    }
    showPlotMessage(ctx, 'Not plottable for <strong>' + name + '</strong>.', { hint: true });
    return;
  }
  // Profile mode (function / likelihood bindings) dispatches to
  // its own worker primitive (profileN) and renderer; the rest
  // of this function handles the sample-mode pipeline.
  if (ctx.currentPlotPlan.mode === 'profile') {
    renderProfilePlotForCurrent(ctx);
    return;
  }
  // Kernel-sample mode: kernel binding rendered like any
  // sampled measure, with a preset dropdown selecting the
  // kernel's input parameters before sampling.
  if (ctx.currentPlotPlan.mode === 'kernel-sample') {
    renderKernelSampleForCurrent(ctx);
    return;
  }
  // Phase=fixed value-typed bindings: render the surface form
  // directly (text for scalars/records/tuples; existing step plot
  // for arrays). Scalars whose per-atom samples differ (engine
  // broadcast) fall through to the sample histogram path.
  if (ctx.currentPlotPlan.mode === 'fixed-record') {
    renderFixedRecord(ctx, ctx.currentPlotPlan);
    return;
  }
  // mode='fixed-scalar' falls through to the sample pipeline
  // below. renderSamplesAndDensity already short-circuits to
  // scalar-text when samplesAreConstant — phase=fixed bindings
  // whose samples are uniform get the text rendering, while
  // engine-broadcast cases (lp_obs, where phase says fixed but
  // each atom's logp differs) keep the histogram.
  // Array-mode loads the cached array synchronously (no worker
  // round-trip), so a Stop button is pointless for it. Sampling
  // mode shows the Stop button so the user can abort long
  // operations (per-i ref chains under huge sample counts).
  const arrayMode = ctx.currentPlotPlan.mode === 'array';
  const matrixMode = ctx.currentPlotPlan.mode === 'matrix';
  // Off-thread samplers (MH / emcee / AMIS) stream progress; show a determinate
  // bar for them. IS and array/matrix loads finish too fast (or aren't pooled)
  // to warrant one.
  const io = ctx.inferenceOpts;
  const sampling = !(arrayMode || matrixMode);
  const showBar = sampling && io && (io.backend === 'mh' || io.backend === 'ram' || io.backend === 'slice' || io.backend === 'emcee' || io.backend === 'amis' || io.backend === 'smc' || io.backend === 'nested' || io.backend === 'elliptical-slice-sampler');

  // Sampling only starts on the explicit Sample button: an AUTOMATIC
  // trigger (opts.autoTrigger — a model edit re-focusing this same
  // binding, see applySourceUpdate in orchestration.ts) must not silently
  // kick off a possibly-slow MCMC / nested / AMIS / SMC / ESS run. isPosterior
  // mirrors the same bayesupdate check the "cancelled" catch branch below
  // (and buildInferenceControl's other callers) use — every non-posterior
  // plot (prior / tractable / array / matrix) draws directly via the
  // synthetic 'forward' backend and stays cheap, so it's never deferred.
  const gateBindingName = ctx.currentPlotBindingName;
  const gateIsPosterior = !!(
    gateBindingName && ctx.derivationsState && ctx.derivationsState.derivations &&
    ctx.derivationsState.derivations[gateBindingName] &&
    ctx.derivationsState.derivations[gateBindingName].kind === 'bayesupdate'
  );
  const effectiveBackend = gateIsPosterior ? io.backend : 'forward';
  if (shouldDeferAutoSample({ autoTrigger: !!(opts && opts.autoTrigger), sampling, effectiveBackend })) {
    showPlotMessage(ctx,
      'Model changed — the <strong>' + esc(effectiveBackend) + '</strong> sampler is stale. Press <strong>Sample</strong> to refresh.',
      { hint: true });
    const host = document.getElementById('plot-empty');
    if (host && ctx.onInferenceChange) {
      const row = document.createElement('div');
      row.style.marginTop = '0.6em';
      row.style.display = 'flex';
      row.style.justifyContent = 'center';
      row.appendChild(buildInferenceControl(ctx, ctx.onInferenceChange, gateIsPosterior));
      host.appendChild(row);
    }
    return;
  }

  showPlotMessage(ctx,
    (arrayMode || matrixMode) ? 'Loading…' : 'Sampling…',
    { cancellable: sampling, hint: true, progress: showBar });
  if (showBar) ctx.onSamplingProgress = function (frac: number, phase: string) { updatePlotProgress(ctx, frac, phase); };
  // Cast to any: the remaining plan modes here (samples / array /
  // fixed-scalar) have differing fields; renderEmpiricalMeasure
  // dispatches on opts.mode inside, so reading `.discrete` /
  // `.analyticalIR` from an ArrayPlan (where they're absent and
  // unused) is benign.
  const planForCall: any = ctx.currentPlotPlan;

  // Cache hit avoids the worker entirely. We still defer through
  // a microtask so the UI flush is uniform and the stale-reply
  // guard pattern stays the same.
  Promise.resolve()
    .then(function() { return getMeasure(ctx, planForCall.name); })
    .then(function(measure) {
      if (ctx.currentPlotPlan !== planForCall) return null;
      return renderEmpiricalMeasure(ctx, measure, {
        name: planForCall.name,
        mode: planForCall.mode,
        discrete: planForCall.discrete,
        analyticalIR: planForCall.analyticalIR,
        // MatrixPlan carries .shape; benign undefined for other modes.
        shape: planForCall.shape,
        toolbarControls: null,
        staleGuard: function() { return ctx.currentPlotPlan === planForCall; },
      });
    })
    .catch(function(err) {
      if (ctx.currentPlotPlan !== planForCall) return;
      const msg = err && err.message ? err.message : String(err);
      if (msg === 'cancelled') {
        // User clicked Stop. Keep the pane actionable rather than a dead end.
        // For a posterior, mount the sampler picker inline so the user can
        // switch backend (or just re-roll) and press Sample to retry WITHOUT
        // round-tripping through the graph — a graph re-tap would re-sample
        // with the old backend before they could change it. Non-posteriors
        // have no sampler to pick, so keep the graph-retry hint.
        const nm = ctx.currentPlotBindingName;
        const isPosterior = !!(
          nm && ctx.derivationsState && ctx.derivationsState.derivations &&
          ctx.derivationsState.derivations[nm] &&
          ctx.derivationsState.derivations[nm].kind === 'bayesupdate'
        );
        if (isPosterior && ctx.onInferenceChange) {
          showPlotMessage(ctx, 'Sampling cancelled. Pick a sampler and press <strong>Sample</strong> to retry.', { hint: true });
          const host = document.getElementById('plot-empty');
          if (host) {
            const row = document.createElement('div');
            row.style.marginTop = '0.6em';
            row.style.display = 'flex';
            row.style.justifyContent = 'center';
            row.appendChild(buildInferenceControl(ctx, ctx.onInferenceChange));
            host.appendChild(row);
          }
        } else {
          const name = nm ? esc(nm) : 'this binding';
          showPlotMessage(ctx, 'Sampling cancelled. Click <strong>' + name + '</strong> in the graph to retry.', { hint: true });
        }
      } else {
        // Real errors are actionable; not italic/dimmed.
        showPlotMessage(ctx, 'Could not compute plot: ' + esc(msg));
      }
    })
    .then(function () { ctx.onSamplingProgress = null; }, function () { ctx.onSamplingProgress = null; });
}

export function updatePlotForBinding(ctx: Ctx, bindingName: string | null, opts?: { autoTrigger?: boolean }) {
  // Snapshot the outgoing plan first — the user may have
  // mutated it since it was first built (selected a different
  // preset, edited an override value, picked a sweep axis).
  // rememberPlanSelections re-keys on plan.name, so this also
  // captures same-binding edits in time for applyRemembered…
  // to restore them onto the rebuilt plan below.
  rememberPlanSelections(ctx, ctx.currentPlotPlan);
  const binding = ctx.currentBindings && bindingName != null
    ? ctx.currentBindings.get(bindingName)
    : null;
  const plan = buildPlotPlan(ctx, binding);
  // Restore user-driven plan state across rebuilds — both same-
  // binding rebuilds (source edit) and cross-binding navigation
  // (click away and back). pendingPresetName / pendingDomainName
  // (set by auto-save-as) take precedence over remembered
  // selection so a freshly-coined name lands selected.
  applyRememberedSelections(ctx, plan);
  // Preset / domain pending-name routing only applies to plans with
  // matchedPresets / matchedDomains — i.e. 'profile' and 'kernel-sample'.
  // Use any-cast through the union; tightening per-mode lands when
  // each renderer's narrows tighten.
  if (plan && (plan.mode === 'profile' || plan.mode === 'kernel-sample')) {
    const p: any = plan;
    if (ctx.pendingPresetName != null) {
      const pn = ctx.pendingPresetName;
      ctx.pendingPresetName = null;
      if (p.matchedPresets
          && p.matchedPresets.some(function(pe: any) { return pe.name === pn; })) {
        p.presetName = pn;
      }
    }
    if (ctx.pendingDomainName != null) {
      const dn = ctx.pendingDomainName;
      ctx.pendingDomainName = null;
      if (p.matchedDomains
          && p.matchedDomains.some(function(d: any) { return d.name === dn; })) {
        p.domainName = dn;
      }
    }
  }
  ctx.currentPlotPlan = plan;
  // Save the freshly-hydrated plan too so a save-as pending name
  // or applyRemembered's filter decisions are reflected in memory
  // before the next mutation. The matching outgoing snapshot at
  // the top of this function captures user edits between calls.
  rememberPlanSelections(ctx, plan);
  // Only surface the clicked name in the plot UI when it actually
  // names a binding. Synthetic nodes (anonymous inline expressions,
  // placeholders, holes) carry IDs like 'prior:target' that aren't
  // useful to the user — fall back to a generic message.
  ctx.currentPlotBindingName = binding ? bindingName : null;
  // Plot pane stays visible whenever plotEnabled is true. When the
  // current binding isn't plottable, renderPlotForCurrent() shows
  // a "Not plottable" message in place of a chart. `opts` (autoTrigger)
  // passes through unchanged — see renderPlotForCurrent's doc comment.
  if (ctx.plotEnabled) renderPlotForCurrent(ctx, opts);
}
