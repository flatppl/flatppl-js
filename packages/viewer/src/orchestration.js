// @flatppl/viewer — orchestration / source updates / nameSeed (Phase 4f).
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
export function nameSeed(ctx, name) {
  var h = 2166136261;
  for (var i = 0; i < name.length; i++) {
    h = h ^ name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h ^ ctx.rootSeed) >>> 0;
}

export function applySourceUpdate(ctx, msg) {
  var sourceChanged = (msg.source !== ctx.currentSource);
  // Track the surface-syntax variant of the in-memory source so
  // (a) processSource picks the right grammar and (b) persist
  // write-back chooses matching syntax (e.g. `True` vs `true`).
  // Variant comes from the host as an id string ('flatppl' /
  // 'flatppy' / 'flatppj') in msg.variant; if absent, falls back
  // to canonical FlatPPL.
  if (msg.variant) ctx.currentVariantId = msg.variant;
  if (sourceChanged) {
    ctx.currentSource = msg.source;
    try {
      var result = FlatPPLEngine.processSource(msg.source,
        { variant: ctx.currentVariantId });
      ctx.currentBindings = result.bindings;
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
  var preservedPlotBinding = null;
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

export function resizeAllEchartsInPlot(ctx) {
  var root = document.getElementById('plot-content');
  if (!root) return;
  var nodes = root.querySelectorAll('div');
  for (var i = 0; i < nodes.length; i++) {
    var inst = echarts.getInstanceByDom(nodes[i]);
    if (inst) try { inst.resize(); } catch (_) {}
  }
  // The root itself may host a single chart (samples / array /
  // profile single-line modes) — resize that too.
  var rootInst = echarts.getInstanceByDom(root);
  if (rootInst) try { rootInst.resize(); } catch (_) {}
}

export function resizeAndFitCy(ctx) {
  if (!ctx.cy) return;
  // requestAnimationFrame so the layout pass that triggered the
  // resize has settled before we ask cytoscape for the new size.
  requestAnimationFrame(function () {
    try { ctx.cy.resize(); ctx.cy.fit(undefined, 40); }
    catch (_) {}
  });
}
