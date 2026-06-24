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

export function applySourceUpdate(ctx: Ctx, msg: any) {
  const sourceChanged = (msg.source !== ctx.currentSource);
  // Track which module (source file) the DAG currently belongs to, so a
  // cross-module back-navigation can re-sync the editor source (spec §04).
  if (msg.path !== undefined) ctx.currentPath = msg.path;
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
        bundle: msg.bundleSources ? { sources: msg.bundleSources } : undefined,
        path: msg.path || undefined,
      });
      ctx.currentBindings = result.bindings;
      ctx.currentLinkedBindings = result.linkedBindings || result.bindings;
      ctx.currentLoweredModule = result.loweredModule;
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
  focusNode(ctx, msg.targetName, msg.pushHistory);
  if (preservedPlotBinding
      && ctx.currentBindings && ctx.currentBindings.has(preservedPlotBinding)
      && ctx.currentPlotBindingName !== preservedPlotBinding) {
    updatePlotForBinding(ctx, preservedPlotBinding);
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
