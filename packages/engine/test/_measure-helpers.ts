'use strict';

// Shared `makeCtx` harness for tests that drive a FlatPPL source through
// the engine and inspect the resulting measures via materialiseMeasure.
//
// Why this exists
// ===============
// Four test files (joint-likelihood, joint-likelihood-refchain, locscale,
// locscale-affine) each carried a byte-identical ~30-line `makeCtx()`
// that: processes a source, builds derivations, spins up a worker handler
// (seeded), and exposes a context with a memoising `getMeasure` plus a
// promise-shaped `sendWorker`. The only per-file difference was the
// default sample count and root seed (module-level constants).
//
// `makeCtxFactory({ sampleCount, rootSeed })` returns a `makeCtx(source,
// opts?)` closure identical in behaviour to the originals: `opts.sampleCount`
// overrides the factory default; the seed is fixed per factory.

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

/**
 * Build a `makeCtx(source, opts?)` bound to a default sample count and a
 * fixed root seed. `makeCtx` returns { ctx, lifted, built }:
 *   - ctx.getMeasure(name)  — memoised materialiseMeasure for `name`.
 *   - ctx.sendWorker(msg)   — worker.handle wrapped to a Promise that
 *                             rejects on a worker `error` reply.
 *   - ctx.derivations / bindings / fixedValues / sampleCount / rootSeed.
 */
function makeCtxFactory(defaults: { sampleCount: number; rootSeed: number }) {
  const SAMPLE_COUNT = defaults.sampleCount;
  const ROOT_SEED    = defaults.rootSeed;
  return function makeCtx(source: any, opts?: any) {
    opts = opts || {};
    const lifted = processSource(source);
    const built  = orchestrator.buildDerivations(lifted.bindings);
    const worker = createWorkerHandler();
    worker.handle({ type: 'init', seed: ROOT_SEED });
    const cache = new Map();
    const ctx: any = {
      derivations: built.derivations,
      bindings:    built.bindings,
      fixedValues: built.fixedValues || new Map(),
      getMeasure:  (name: any) => {
        if (cache.has(name)) return cache.get(name);
        const p = materialiser.materialiseMeasure(name, ctx);
        cache.set(name, p);
        return p;
      },
      sendWorker:  (msg: any) => {
        const reply = worker.handle(msg);
        if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
        return Promise.resolve(reply);
      },
      sampleCount: opts.sampleCount != null ? opts.sampleCount : SAMPLE_COUNT,
      rootSeed:    ROOT_SEED,
    };
    return { ctx, lifted, built };
  };
}

module.exports = { makeCtxFactory };
