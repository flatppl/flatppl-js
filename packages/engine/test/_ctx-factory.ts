'use strict';
// Shared factory for building a materialiser context from FlatPPL source.
// Used by regression-baseline.test.ts and module-registry-score.test.ts.
// The leading underscore keeps this file outside the `test/**/*.test.ts` glob.

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SEED = 0xBA5E;

// Build a materialiser context for a given source text and sample count.
function ctxFor(src: string, N: number) {
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
    rootKey: SEED,
    rootSeed: SEED,
    marginalizationCount: 64,
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
  return { proc, ctx };
}

module.exports = { ctxFor, SEED };
