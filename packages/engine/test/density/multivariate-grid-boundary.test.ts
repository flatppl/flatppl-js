'use strict';
// Boundary pin for multi-axis kernel-broadcast (PR: N-D §04 grids).
//
// v1 scope is SCALAR-OUTPUT distributions only. A vector-output / vector-param
// distribution (Dirichlet alpha, MvNormal mu, …) carries a trailing EVENT axis
// that `collectionAxesOf` cannot distinguish from a grid axis (see the NOTE in
// materialiser-shared.collectionAxesOf). The out-of-scope case must therefore
// fail LOUD — never silently mis-score the event axis as another grid cell.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const densityPrims = require('../../density-prims.ts');
const { processSource, orchestrator, materialiser } = require('../..');
const { createWorkerHandler } = require('../../worker.ts');

// ---------------------------------------------------------------------------
// 1. The loud backstop, pinned directly.
//
// `builtinLogdensityofPositional` (blp) is what the scalar grid path calls
// per cell. A multivariate kernel is not a univariate REGISTRY entry, so blp
// throws rather than scoring a scalar slice of a vector variate. This is the
// guard that makes the out-of-scope case loud; if a future refactor of the
// grid path drops it, the path could silently mis-score — this test catches
// that.
// ---------------------------------------------------------------------------
test('blp guard: positional logdensityof of a multivariate kernel throws loud', () => {
  assert.equal(densityPrims.isMultivariateKernel('Dirichlet'), true);
  assert.equal(densityPrims.isMultivariateKernel('MvNormal'), true);
  assert.throws(
    () => densityPrims.builtinLogdensityofPositional('Dirichlet', [1.0, 1.0, 1.0], 0.2),
    /univariate|record form|multivariate/i,
    'Dirichlet (vector-output) must not be scored via the scalar positional path',
  );
  assert.throws(
    () => densityPrims.builtinLogdensityofPositional('MvNormal', [0.0, 0.0], 0.1),
    /univariate|record form|multivariate/i,
    'MvNormal (vector-output) must not be scored via the scalar positional path',
  );
});

// ---------------------------------------------------------------------------
// 2. End-to-end: a vector-output distribution broadcast over a grid does not
// silently score. We assert only that it REJECTS (loud) — the exact failure
// (currently a lowering `no derivation`, not the blp guard) is an out-of-scope
// limitation, and either loud error is acceptable; what matters is that no
// finite mis-scored density is returned.
// ---------------------------------------------------------------------------
const SEED = 0xFEED;

function materialise(src: string, target: string, N: number): Promise<any> {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: N,
    rootSeed: SEED,
    rootKey: SEED,
    marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => {
      const r = w.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message))
        : Promise.resolve(r);
    },
  };
  return ctx.getMeasure(target);
}

test('broadcast boundary: vector-output kernel over a grid does not silently score', async () => {
  // Dirichlet has a single vector param (alpha); broadcasting it over a 2-cell
  // grid of 3-vectors is out of scope for v1 and must not return a finite
  // (mis-scored) density.
  const src = [
    'alphas = [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]',
    '__score__ = logdensityof(Dirichlet.(alpha = alphas), [[0.2, 0.3, 0.5], [0.1, 0.4, 0.5]])',
  ].join('\n');
  await assert.rejects(
    async () => {
      const m = await materialise(src, '__score__', 1);
      return m && m.samples != null
        ? m.samples[0]
        : (m && m.value && m.value.data != null ? m.value.data[0] : NaN);
    },
    'broadcasting a vector-output distribution over a grid must fail loud, not return a mis-scored density',
  );
});
