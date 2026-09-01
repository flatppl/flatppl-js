'use strict';

// =====================================================================
// peratom-weighted-bench.ts — hot-path gate for `_perAtomFallback`
// =====================================================================
//
// `_perAtomFallback` (sampler-eval-batched.ts) is the residue branch of
// the batched evaluator: every body with a non-scalar op lands there, so
// its per-atom cost is what plotting a `weighted` measure pays. This
// harness materialises one over a 2-D Lebesgue box at the sample counts
// a host actually asks for, on both routes.
//
// Run:  node packages/engine/bench/peratom-weighted-bench.ts
//       N=50000 node packages/engine/bench/peratom-weighted-bench.ts
//
// NOT a test — it prints a table and exits 0. FLATPPL_NO_PERATOM_COMPILE
// selects the route inside the process, so the two columns differ only
// in the loop under measurement.

const fs = require('fs');
const path = require('path');
const { processSource, orchestrator, materialiser } = require('../index.ts');
const { createWorkerHandler } = require('../worker.ts');
const batched = require('../sampler-eval-batched.ts');

const MODEL = process.env.MODEL || path.join(
  __dirname, '..', 'test', 'fixtures', 'dminus-to-3pi-amplitude.flatppl');
const BINDING = process.env.BINDING || 'amplitude_measure';
const COUNTS = (process.env.N || '1000,10000,50000')
  .split(',').map((s: string) => parseInt(s, 10));

const src = fs.readFileSync(MODEL, 'utf8');

function buildCtx(N: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: N, rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

async function timeOnce(N: number, compile: boolean) {
  batched._setCompilePerAtom(compile);
  const ctx = buildCtx(N);
  const t0 = process.hrtime.bigint();
  const m = await ctx.getMeasure(BINDING);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  return { ms, logTotalmass: m.logTotalmass, n_eff: m.n_eff };
}

async function main() {
  const rows: any[] = [];
  for (const N of COUNTS) {
    const cold = await timeOnce(N, false);
    const hot = await timeOnce(N, true);
    rows.push({
      N,
      'interpreted ms': +cold.ms.toFixed(0),
      'compiled ms': +hot.ms.toFixed(0),
      speedup: +(cold.ms / hot.ms).toFixed(2) + 'x',
      'us/atom before': +((cold.ms * 1000) / N).toFixed(0),
      'us/atom after': +((hot.ms * 1000) / N).toFixed(0),
      // Same number on both routes or the speedup is meaningless.
      identical: cold.logTotalmass === hot.logTotalmass
        && cold.n_eff === hot.n_eff,
    });
  }
  console.table(rows);
  console.log(JSON.stringify({
    node: process.version, model: path.basename(MODEL), binding: BINDING,
  }));
}

main().catch((e) => { console.error(e); process.exit(1); });
