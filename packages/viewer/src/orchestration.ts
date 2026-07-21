// @ts-check
// @flatppl/viewer — orchestration / source updates / nameSeed —
//
// applySourceUpdate processes a sourceUpdate or showModule message:
// re-parses the source, rebuilds derivations, re-renders the DAG,
// restores the focused binding. resizeAllEchartsInPlot /
// resizeAndFitCy back the host's ResizeObserver. nameSeed mixes the
// per-mount rootSeed with a binding name into a per-binding worker
// seed, deterministic across runs.

import { rebuildDerivations } from './derivations.js';
import { updatePlotForBinding } from './render-plot.js';

/**
 * FNV-1a 32-bit string hash, then XOR the root seed. Used to give
 * each binding its own RNG stream for sampleN(). Independent of
 * arrival order — two independent variables stay independent
 * regardless of which one the user clicked first.
 */
import { enterModuleView, focusNode } from './dag.js';
import type { Ctx } from './types';
export function nameSeed(ctx: Ctx, name: any) {
  let h = 2166136261;
  for (let i = 0; i < name.length; i++) {
    h = h ^ name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h ^ ctx.rootSeed) >>> 0;
}

/**
 * The module context (resolved `path` + dependency `bundleSources`) a
 * sourceUpdate lowers against. Both are STICKY to the current model: the host
 * signals a model SWITCH by sending `path` (a string, or `null` for a path-less
 * module such as an embedded block), which (re)establishes both path and
 * bundle; a SAME-MODEL update (target navigation, edit) omits `path`, so the
 * tracked context persists.
 *
 * Why sticky: a same-model re-lower that dropped the path would reprocess the
 * source with an empty modulePath, and the engine would resolve relative
 * `load_module` deps against an empty base — so a remote module's
 * `load_module("priors.flatppl")` collapsed to a bare gallery path and the
 * cross-URL drill-down 404'd (engine registry built in pir.ts via
 * `resolveModulePath(modulePath, relPath)`).
 */
export function moduleContextOnUpdate(
  prev: { path: any; bundleSources: any },
  msg: any,
): { path: any; bundleSources: any } {
  if (msg.path !== undefined) {
    return { path: msg.path, bundleSources: msg.bundleSources || null };
  }
  return { path: prev.path, bundleSources: prev.bundleSources };
}

export function applySourceUpdate(ctx: Ctx, msg: any) {
  const sourceChanged = (msg.source !== ctx.currentSource);
  // Captured BEFORE the sourceChanged block below (re)assigns
  // ctx.currentBindings, so this distinguishes an edit of an ALREADY-open
  // model from the very first sourceUpdate that opens one (hadPriorBindings
  // is false there). Only the former defers a stateful-sampler backend's
  // auto re-render (see autoTrigger below) — the initial paint still
  // samples immediately so the pane isn't left empty on load.
  const hadPriorBindings = !!ctx.currentBindings;
  // The module's path + dep bundle are STICKY to the current model
  // (moduleContextOnUpdate): a same-model re-lower (target-nav / edit) keeps
  // them so relative load_module deps still resolve against THIS module — and
  // a cross-module back-navigation can re-sync the editor source (spec §04).
  const modCtx = moduleContextOnUpdate(
    { path: ctx.currentPath, bundleSources: ctx.currentBundleSources }, msg);
  ctx.currentPath = modCtx.path;
  ctx.currentBundleSources = modCtx.bundleSources;
  // currentVariantId is initialised to 'flatppl' in main.ts and
  // there is only one canonical surface (spec §05); the host MAY
  // override via msg.variant for forward-compatibility but no
  // current host does. Retained as the explicit seam should a
  // genuinely-different surface form ever land.
  if (msg.variant) ctx.currentVariantId = msg.variant;
  if (sourceChanged) {
    ctx.currentSource = msg.source;
    try {
      // Multi-file (spec §04): the host pre-resolves `load_module` deps
      // into `msg.bundleSources` (keyed by resolved path) and supplies the
      // primary's own path for relative resolution. `currentBindings`
      // stays the PRIMARY module (the DAG renders it); `currentLinked
      // Bindings` is the engine-internal flattened graph that drives
      // derivation building / materialisation (cross-module refs resolved).
      const result = FlatPPLEngine.processSource(msg.source, {
        variant: ctx.currentVariantId,
        bundle: ctx.currentBundleSources ? { sources: ctx.currentBundleSources } : undefined,
        path: ctx.currentPath || undefined,
      });
      ctx.currentBindings = result.bindings;
      ctx.currentLinkedBindings = result.linkedBindings || result.bindings;
      ctx.currentLoweredModule = result.loweredModule;
      // ctx.currentBundleSources was set above via moduleContextOnUpdate — the
      // raw bundle (spec §04) the off-thread MCMC pool re-processes in workers
      // to resolve `load_module` deps (else a cross-module posterior has no
      // derivation there).
      // Source change → rebuild derivations and clear sample cache.
      // The orchestrator's derivations key the cache, so any change
      // (renamed bindings, edited dist params, new dependencies)
      // requires a full reset.
      rebuildDerivations(ctx);
    } catch (e) {
      // Parse error: keep the previous bindings so the visualizer
      // stays usable while the user fixes their syntax. The host's
      // own diagnostics (VS Code editor squiggles, embed page
      // markers, …) surface the error to the user.
      console.error('FlatPPL parse error:', e);
      return;
    }
  }
  if (msg.type === 'showModule') {
    enterModuleView(ctx, msg.pushHistory);
    return;
  }
  // The DAG view tracks two distinct foci:
  //   currentState.targetName   — sub-DAG root (set by initial
  //                                nav and DAG dbltap; mirrored
  //                                to URL via host.setTarget).
  //   currentPlotBindingName    — node whose plot is rendered in
  //                                the right pane (set additionally
  //                                by single-tap on a DAG node).
  // On a source-only refresh (persist's debounced re-render) we
  // want to KEEP whatever the user was looking at. focusNode
  // already preserves currentState.targetName when msg.targetName
  // is null, and re-renders the same sub-DAG. But its
  // updatePlotForBinding call resets the plot pane back to the
  // sub-DAG root, losing any divergent single-tap selection.
  // Capture currentPlotBindingName here and restore it after
  // focusNode finishes.
  let preservedPlotBinding = null;
  if (sourceChanged && ctx.currentPlotBindingName
      && ctx.currentState
      && ctx.currentPlotBindingName !== ctx.currentState.targetName) {
    preservedPlotBinding = ctx.currentPlotBindingName;
  }
  // autoTrigger: this call chain is a same-model TEXT EDIT (not the initial
  // load, not a cross-model switch, not an explicit DAG click) re-focusing
  // whatever binding is already shown — renderPlotForCurrent defers a
  // stateful-sampler backend behind the Sample button rather than silently
  // re-running it. `msg.path === undefined` is the same same-model signal
  // moduleContextOnUpdate uses above: a model SWITCH
  // always carries `path` (even `null` for a path-less module), so this
  // excludes opening a different/new model from the gate — that still
  // samples immediately, same as the very first load.
  const autoTrigger = sourceChanged && hadPriorBindings && msg.path === undefined;
  focusNode(ctx, msg.targetName, msg.pushHistory, { autoTrigger });
  if (preservedPlotBinding
      && ctx.currentBindings && ctx.currentBindings.has(preservedPlotBinding)
      && ctx.currentPlotBindingName !== preservedPlotBinding) {
    updatePlotForBinding(ctx, preservedPlotBinding, { autoTrigger });
  }
}

export function resizeAllEchartsInPlot(ctx: Ctx) {
  const root = document.getElementById('plot-content');
  if (!root) return;
  const nodes = root.querySelectorAll('div');
  for (let i = 0; i < nodes.length; i++) {
    const inst = echarts.getInstanceByDom(nodes[i]);
    if (inst) try { inst.resize(); } catch (_) {}
  }
  // The root itself may host a single chart (samples / array /
  // profile single-line modes) — resize that too.
  const rootInst = echarts.getInstanceByDom(root);
  if (rootInst) try { rootInst.resize(); } catch (_) {}
}

export function resizeAndFitCy(ctx: Ctx) {
  if (!ctx.cy) return;
  // Resize the canvas PROMPTLY (so rendering matches the container) but
  // DEBOUNCE the re-fit. The container settles its size in several steps
  // during page/flex layout, and firing cy.fit() on each step re-centers
  // the graph — shifting node screen positions repeatedly. A double-click
  // landing in that window has its second tap miss the moved node (the
  // node never moves once settled — layout is animate:false). cy.resize()
  // alone preserves pan/zoom (no node movement); the single trailing
  // cy.fit() re-centers once, after resizes have stopped.
  try { ctx.cy.resize(); } catch (_) {}
  if (ctx._cyFitTimer) { try { clearTimeout(ctx._cyFitTimer); } catch (_) {} }
  ctx._cyFitTimer = setTimeout(function () {
    ctx._cyFitTimer = null;
    if (!ctx.cy) return;
    try { ctx.cy.fit(undefined, 40); } catch (_) {}
  }, 150);
}
