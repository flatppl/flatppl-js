// @ts-check
// @flatppl/viewer — engine facade —
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

import { sendWorker, runMcmcPool } from './worker.js';
import type { Ctx } from './types';

export function tryGetMeasure(ctx: Ctx, name: any) {
  return getMeasure(ctx, name).then(
    function(m: any) { return m; },
    function(_err: any) { return null; });
}

export function getMeasure(ctx: Ctx, name: any) {
  if (ctx.measureCache.has(name)) return Promise.resolve(ctx.measureCache.get(name));
  if (!ctx.derivationsState) return Promise.reject(new Error('no model loaded'));

  // MCMC/sampling backends (mh / emcee / nested / ...) for a bayesupdate
  // posterior run OFF the main thread in a worker pool — non-blocking, and
  // parallel across independent chains (MH) / ensembles (emcee) where the
  // backend supports it (nested runs in a single worker — see runMcmcPool).
  // Only the posterior binding takes this path; its sub-measures (priors,
  // etc.) materialise normally inside each worker. Other bindings under any
  // backend use the main-thread path below.
  const io = ctx.inferenceOpts;
  const deriv = ctx.derivationsState.derivations[name];
  if (io && (io.backend === 'mh' || io.backend === 'ram' || io.backend === 'slice' || io.backend === 'emcee' || io.backend === 'demcz' || io.backend === 'amis' || io.backend === 'smc' || io.backend === 'elliptical-slice-sampler' || io.backend === 'nested')
      && deriv && deriv.kind === 'bayesupdate' && (ctx as any).currentSource) {
    const p = runMcmcPool(ctx, name, io);
    p.then((m: any) => ctx.measureCache.set(name, m), () => {});
    return p;
  }
  // All per-kind materialisation lives in the engine — the viewer's
  // job here is just to memoise the result against the cache. The
  // engine-side materialiser dispatches by derivation kind, computes
  // samples + logWeights + logTotalmass + n_eff, and returns the
  // Measure record. Recursion is handled by passing getMeasure
  // itself back in so child materialisations hit the same cache.
  const promise = FlatPPLEngine.materialiser.materialiseMeasure(name, {
    derivations: ctx.derivationsState.derivations,
    bindings:    ctx.derivationsState.bindings,
    fixedValues: ctx.derivationsState.fixedValues,
    // Bind ctx into 1-arg callbacks: the engine's
    // materialiseMeasure expects callbacks with the original
    // signatures (`getMeasure(name)`, `sendWorker(msg)`); our
    // hoisted versions added `ctx` as a first parameter, so we
    // close over `ctx` here to keep the engine ABI unchanged.
    getMeasure:  function (n: any) { return getMeasure(ctx, n); },
    sendWorker:  function (m: any) { return sendWorker(ctx, m); },
    sampleCount: ctx.SAMPLE_COUNT,
    rootSeed:    ctx.rootSeed,
    rejectionBudget: ctx.REJECTION_BUDGET,
    // Posterior backend selection (read by the engine's matBayesupdate).
    // Propagated to child materialisations too; only bayesupdate bindings act
    // on it, so forward measures are unaffected.
    inferenceOpts: ctx.inferenceOpts,
  });
  // Cache-set rides its own subscription; give it a no-op rejection handler
  // (like the MCMC branch above) so a materialise failure — e.g. getMeasure
  // on a free `elementof` input with no derivation, which callers reach via
  // the soft-failing tryGetMeasure — doesn't surface as an unhandled
  // rejection. The returned `promise` still rejects for the caller.
  promise.then(function(m: any) { ctx.measureCache.set(name, m); }, function() {});
  return promise;
}

export function fixedValueToMeasure(ctx: Ctx, v: any) {
  return FlatPPLEngine.materialiser.fixedValueToMeasure(v, ctx.SAMPLE_COUNT);
}

export function collectRefArrays(ctx: Ctx, ir: any) {
  const fv = ctx.derivationsState && ctx.derivationsState.fixedValues;
  return FlatPPLEngine.materialiser.collectRefArrays(
    ir, fv, function (n: any) { return getMeasure(ctx, n); });
}
