'use strict';
const { test }   = require('node:test');
const assert     = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const MV         = require('../model-view.ts');

const SEED = 0xBA5E;

// Build a ctx where getMeasure for `missingPoolLatent` is forced to throw,
// simulating a forward-sampling failure (e.g. a composite [G,N] latent whose
// sampler errors out). All OTHER bindings materialise normally.
function ctxWithMissingPool(src: string, N: number, missingPoolLatent: string): any {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: SEED });
  const cache = new Map<string, any>();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: N,
    rootKey: SEED,
    rootSeed: SEED,
    marginalizationCount: 64,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    sendWorker: (m: any) => {
      const r = w.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message))
        : Promise.resolve(r);
    },
  };
  ctx.getMeasure = (n: string) => {
    if (n === missingPoolLatent) throw new Error(`simulated pool failure for '${n}'`);
    if (cache.has(n)) return cache.get(n);
    const m = materialiser.materialiseMeasure(n, ctx);
    cache.set(n, m);
    return m;
  };
  return ctx;
}

function postDeriv(ctx: any): any {
  for (const [, v] of Object.entries(ctx.derivations)) if (v && (v as any).kind === 'bayesupdate') return v;
  return null;
}

// Model with:
//   a_plus_b ~ iid(pushfwd(sq, Exponential(1.5)), 2)  — INTRACTABLE (sq = z*z non-injective)
//   mu       ~ iid(Beta(1, 1), 2)                      — tractable
//   p        ~ beta_row_K.(a_plus_b, mu)               — composite, pool FORCED MISSING
//
// The intractable prior MUST still be refused, not skipped because p lacks a pool.
// Old code: havePools=false (p missing) → probe skipped → resolves silently.
// Fixed code: probes all pooled latents (a_plus_b + mu) in one combined point,
// catches the density error for a_plus_b, rejects loudly.
const SRC = `
weird = pushfwd(sq, Exponential(1.5))
sq = z -> z * z
a_plus_b ~ iid(weird, 2)
mu ~ iid(Beta(1, 1), 2)
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), 3)
p ~ beta_row_K.(a_plus_b, mu)
binom_row = (n_row, p_row) -> Binomial.(n_row, p_row)
r ~ binom_row.([[5,5,5],[5,5,5]], p)
prior = lawof(record(a_plus_b = a_plus_b, mu = mu, p = p))
forward_kernel = kernelof(record(r = r), a_plus_b = a_plus_b, mu = mu, p = p)
L = likelihoodof(forward_kernel, record(r = [[3,2,4],[1,5,0]]))
posterior = bayesupdate(L, prior)
`;

test('intractable pushfwd prior is refused even when a co-latent lacks a pool', async () => {
  // Force p's pool to be missing (simulates a composite [G,N] sampling failure).
  // a_plus_b and mu still get pools, so the combined probe fires for them.
  const ctx = ctxWithMissingPool(SRC, 300, 'p');
  await assert.rejects(
    () => MV.buildModelViewFromCtx(ctx, postDeriv(ctx)),
    /tractable density|bijection/,
    'must refuse the intractable prior, not skip the probe',
  );
});
