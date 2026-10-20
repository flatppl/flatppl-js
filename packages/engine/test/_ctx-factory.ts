'use strict';
// Shared factory for building a materialiser context from FlatPPL source.
// Used by regression-baseline.test.ts and module-registry-score.test.ts.
// The leading underscore keeps this file outside the `test/**/*.test.ts` glob.

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const { errorFromReply } = require('../worker-errors.ts');

const SEED = 0xBA5E;

// Build a materialiser context for a given source text and sample count.
function ctxFor(src: string, N: number, opts?: any) {
  const proc = processSource(src, opts);
  const built = orchestrator.buildDerivations(proc.linkedBindings, {
    moduleRegistry: proc.linkedModuleRegistry,
  });
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
    moduleRegistry: built.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => {
      return Promise.resolve(w.handle(m)).then((r: any) => {
        if (r?.type === 'error') throw errorFromReply(r);
        return r;
      });
    },
  };
  return { proc, ctx };
}

module.exports = { ctxFor, SEED };
