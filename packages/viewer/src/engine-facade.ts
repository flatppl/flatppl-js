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
  const cache = ctx.measureCache;
  const state = ctx.derivationsState;
  if (cache.has(name)) return Promise.resolve(cache.get(name));
  if (!state) return Promise.reject(new Error('no model loaded'));
  // A source/config update replaces the cache. Keep completions in their
  // original cache and refuse recursion across that model boundary.
  const isCurrent = () => ctx.measureCache === cache && ctx.derivationsState === state;

  // MCMC/sampling backends (mh / emcee / nested / ...) for a bayesupdate
  // posterior run OFF the main thread in a worker pool — non-blocking, and
  // parallel across independent chains (MH) / ensembles (emcee) where the
  // backend supports it (nested runs in a single worker — see runMcmcPool).
  // Only the posterior binding takes this path; its sub-measures (priors,
  // etc.) materialise normally inside each worker. Other bindings under any
  // backend use the main-thread path below.
  const io = ctx.inferenceOpts ? { ...ctx.inferenceOpts } : ctx.inferenceOpts;
  const deriv = state.derivations[name];
  if (io && (io.backend === 'mh' || io.backend === 'ram' || io.backend === 'slice' || io.backend === 'emcee' || io.backend === 'demcz' || io.backend === 'amis' || io.backend === 'smc' || io.backend === 'elliptical-slice-sampler' || io.backend === 'nested')
      && deriv && deriv.kind === 'bayesupdate' && (ctx as any).currentSource) {
    const p = runMcmcPool(ctx, name, io);
    p.then((m: any) => cache.set(name, m), () => {});
    return p;
  }
  // A dataset (iid over a record law) is ONE k-row table, at one atom: an
  // ensemble of tables has no value, so the engine refuses any other count.
  // It is drawn in a context of its own, whose upstream draws are at one
  // atom too, and shown as one dataset (`datasetMeasure`).
  const engineCtx = { derivations: state.derivations, bindings: state.bindings };
  if (FlatPPLEngine.materialiser.isRecordIidDataset(name, engineCtx)) {
    const own = new Map();
    const dsCtx: any = {
      derivations: state.derivations,
      bindings:    state.bindings,
      fixedValues: state.fixedValues,
      moduleRegistry: state.moduleRegistry,
      getMeasure: function (n: any) {
        if (own.has(n)) return own.get(n);
        const p = isCurrent() ? FlatPPLEngine.materialiser.materialiseMeasure(n, dsCtx)
          : Promise.reject(new Error('model changed'));
        own.set(n, p);
        return p;
      },
      sendWorker: function (m: any) {
        return isCurrent() ? sendWorker(ctx, m) : Promise.reject(new Error('model changed'));
      },
      sampleCount: 1,
      rootSeed:    ctx.rootSeed,
      rejectionBudget: ctx.REJECTION_BUDGET,
      inferenceOpts: io,
    };
    const p = dsCtx.getMeasure(name).then(datasetMeasure);
    p.then(function (m: any) { cache.set(name, m); }, function () {});
    return p;
  }
  // All per-kind materialisation lives in the engine — the viewer's
  // job here is just to memoise the result against the cache. The
  // engine-side materialiser dispatches by derivation kind, computes
  // samples + logWeights + logTotalmass + n_eff, and returns the
  // Measure record. Recursion is handled by passing getMeasure
  // itself back in so child materialisations hit the same cache.
  const promise = FlatPPLEngine.materialiser.materialiseMeasure(name, {
    derivations: state.derivations,
    bindings:    state.bindings,
    fixedValues: state.fixedValues,
    moduleRegistry: state.moduleRegistry,
    // Bind ctx into 1-arg callbacks: the engine's
    // materialiseMeasure expects callbacks with the original
    // signatures (`getMeasure(name)`, `sendWorker(msg)`); our
    // hoisted versions added `ctx` as a first parameter, so we
    // close over `ctx` here to keep the engine ABI unchanged.
    getMeasure:  function (n: any) {
      return isCurrent() ? getMeasure(ctx, n) : Promise.reject(new Error('model changed'));
    },
    sendWorker:  function (m: any) {
      return isCurrent() ? sendWorker(ctx, m) : Promise.reject(new Error('model changed'));
    },
    sampleCount: ctx.SAMPLE_COUNT,
    rootSeed:    ctx.rootSeed,
    rejectionBudget: ctx.REJECTION_BUDGET,
    // Posterior backend selection (read by the engine's matBayesupdate).
    // Propagated to child materialisations too; only bayesupdate bindings act
    // on it, so forward measures are unaffected.
    inferenceOpts: io,
  });
  // Cache-set rides its own subscription; give it a no-op rejection handler
  // (like the MCMC branch above) so a materialise failure — e.g. getMeasure
  // on a free `elementof` input with no derivation, which callers reach via
  // the soft-failing tryGetMeasure — doesn't surface as an unhandled
  // rejection. The returned `promise` still rejects for the caller.
  promise.then(function(m: any) { cache.set(name, m); }, function() {});
  return promise;
}

// A dataset table as a record measure over its rows: one field per scalar
// column, and one per component of a vector column (`p[1]`, `p[2]`, …), so the
// record views show its columns and row histograms. `dataset` carries the row
// count and, for rows drawn by importance resampling, the pool's ESS and K.
export function datasetMeasure(t: any) {
  const emp = FlatPPLEngine.empirical;
  const rows = t.nrows;
  const fields: any = {};
  for (const name of Object.keys(t.columns || {})) {
    const col = t.columns[name];
    if (!col || !col.data || !Array.isArray(col.shape)) continue;
    const inner = col.shape.slice(1);
    const width = inner.reduce((p: number, n: number) => p * n, 1);
    if (width === 1) {
      fields[name] = { shape: 'scalar', samples: Float64Array.from(col.data), logWeights: null };
      continue;
    }
    for (let j = 0; j < width; j++) {
      const xs = new Float64Array(rows);
      for (let r = 0; r < rows; r++) xs[r] = col.data[r * width + j];
      fields[name + '[' + (j + 1) + ']'] = { shape: 'scalar', samples: xs, logWeights: null };
    }
  }
  return Object.assign(emp.recordMeasure(fields, null), {
    n_eff: rows,
    dataset: { rows, n_eff: t.n_eff, sir: t.sir || null },
  });
}

export function fixedValueToMeasure(ctx: Ctx, v: any) {
  return FlatPPLEngine.materialiser.fixedValueToMeasure(v, ctx.SAMPLE_COUNT);
}

export function collectRefArrays(ctx: Ctx, ir: any) {
  const fv = ctx.derivationsState && ctx.derivationsState.fixedValues;
  return FlatPPLEngine.materialiser.collectRefArrays(
    ir, fv, function (n: any) { return getMeasure(ctx, n); });
}
