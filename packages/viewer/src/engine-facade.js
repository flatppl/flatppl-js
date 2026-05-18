// @flatppl/viewer — engine facade (Phase 4c).
//
// Thin per-mount wrappers around FlatPPLEngine.materialiser.
// getMeasure memoises materialised EmpiricalMeasures in
// ctx.measureCache; tryGetMeasure soft-fails to null;
// fixedValueToMeasure adapts a fixed-phase JS value into the SoA
// empirical-measure shape; collectRefArrays threads getMeasure +
// fixedValues into the engine's ref-collection.
//
// Two callbacks (getMeasure, sendWorker) are bound into 1-arg
// closures at the engine boundary — see the wrap inside getMeasure.
// sendWorker is imported from ./worker.js.

import { sendWorker } from './worker.js';

export function tryGetMeasure(ctx, name) {
  return getMeasure(ctx, name).then(
    function(m) { return m; },
    function(_err) { return null; });
}

export function getMeasure(ctx, name) {
  if (ctx.measureCache.has(name)) return Promise.resolve(ctx.measureCache.get(name));
  if (!ctx.derivationsState) return Promise.reject(new Error('no model loaded'));
  // All per-kind materialisation lives in the engine — the viewer's
  // job here is just to memoise the result against the cache. The
  // engine-side materialiser dispatches by derivation kind, computes
  // samples + logWeights + logTotalmass + n_eff, and returns the
  // Measure record. Recursion is handled by passing getMeasure
  // itself back in so child materialisations hit the same cache.
  var promise = FlatPPLEngine.materialiser.materialiseMeasure(name, {
    derivations: ctx.derivationsState.derivations,
    bindings:    ctx.derivationsState.bindings,
    fixedValues: ctx.derivationsState.fixedValues,
    // Bind ctx into 1-arg callbacks: the engine's
    // materialiseMeasure expects callbacks with the original
    // signatures (`getMeasure(name)`, `sendWorker(msg)`); our
    // hoisted versions added `ctx` as a first parameter, so we
    // close over `ctx` here to keep the engine ABI unchanged.
    getMeasure:  function (n) { return getMeasure(ctx, n); },
    sendWorker:  function (m) { return sendWorker(ctx, m); },
    sampleCount: ctx.SAMPLE_COUNT,
    rootSeed:    ctx.rootSeed,
    rejectionBudget: ctx.REJECTION_BUDGET,
  });
  promise.then(function(m) { ctx.measureCache.set(name, m); });
  return promise;
}

export function fixedValueToMeasure(ctx, v) {
  return FlatPPLEngine.materialiser.fixedValueToMeasure(v, ctx.SAMPLE_COUNT);
}

export function collectRefArrays(ctx, ir) {
  var fv = ctx.derivationsState && ctx.derivationsState.fixedValues;
  return FlatPPLEngine.materialiser.collectRefArrays(
    ir, fv, function (n) { return getMeasure(ctx, n); });
}
