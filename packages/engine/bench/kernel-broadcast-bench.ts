'use strict';

// =====================================================================
// kernel-broadcast-bench.ts — Phase 8 (fusion (b) sub-2) baseline + gate
// =====================================================================
//
// Measures materialise wall-time for the composite kernel-broadcast
// paths in `mat-broadcast.ts`. Today those paths loop PER CELL (one
// worker call per broadcast cell j; nested loops K_outer × K_inner) —
// see the executors in mat-broadcast.ts. Phase 8 folds the cell axis
// into the worker batch axis (one call at count N·K), so the speedup
// scales with the cell count K. This harness sweeps K to expose that:
// at the fixtures' native K=3 the fold is only ~3×; the ≥10× acceptance
// gate (TODO Phase 8) lives at K ≳ 16.
//
// Run:  npx tsx packages/engine/bench/kernel-broadcast-bench.ts
//
// NOT a test — it prints a table and exits 0. Re-run before/after a
// sub-2 change; the "median ms" columns are the comparison.

const fs = require('node:fs');
const path = require('node:path');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

// --- timing -----------------------------------------------------------

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Median materialise time (ms) for `targetName` at N atoms, over `runs`
// timed repetitions after one warm-up. Each repetition uses a fresh ctx
// + worker + cache so no measurement reuses a cached measure.
async function timeMaterialise(
  source: string, targetName: string, N: number, runs: number,
): Promise<{ ms: number; ok: boolean; note?: string }> {
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  if (errs.length) {
    return { ms: NaN, ok: false, note: 'parse: ' + errs[0].message };
  }
  const built = orchestrator.buildDerivations(lifted.bindings);
  if (!built.derivations[targetName]) {
    return { ms: NaN, ok: false, note: 'no derivation for ' + targetName };
  }

  const once = async (): Promise<number> => {
    const worker = createWorkerHandler();
    worker.handle({ type: 'init', seed: 42 });
    const cache = new Map();
    const ctx: any = {
      derivations: built.derivations, bindings: built.bindings,
      fixedValues: built.fixedValues || new Map(),
      getMeasure: (n: string) => {
        if (cache.has(n)) return cache.get(n);
        const p = materialiser.materialiseMeasure(n, ctx);
        cache.set(n, p);
        return p;
      },
      sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
      sampleCount: N, rootKey: [42, 0],
    };
    const t0 = performance.now();
    await ctx.getMeasure(targetName);
    return performance.now() - t0;
  };

  try {
    await once();                                   // warm-up (JIT)
    const ts: number[] = [];
    for (let r = 0; r < runs; r++) ts.push(await once());
    return { ms: median(ts), ok: true };
  } catch (e: any) {
    return { ms: NaN, ok: false, note: e.message };
  }
}

// --- scaled model generators (cell axis = K) --------------------------

// Random-intercepts (iid-composite kernel body), cell axis K = G groups,
// inner iid axis = Npg. x_per_group is a G×Npg fixed regressor matrix.
function genRandomIntercepts(G: number, Npg: number): string {
  const rows: string[] = [];
  for (let i = 0; i < G; i++) {
    const cols: number[] = [];
    for (let j = 0; j < Npg; j++) cols.push(Number(((i * Npg + j + 1) * 0.1).toFixed(3)));
    rows.push('[' + cols.join(', ') + ']');
  }
  return `flatppl_compat = "0.1"
G = ${G}
N_per_group = ${Npg}
mu_int ~ Normal(0, 10)
tau_int ~ truncate(Normal(0, 5), interval(0, inf))
intercepts ~ iid(Normal(mu_int, tau_int), G)
slope ~ Normal(0, 1)
sigma ~ truncate(Normal(0, 1), interval(0, inf))
x_per_group = [${rows.join(', ')}]
group_kernel = kernelof(
  iid(Normal(mu = intercept + slope .* x_group, sigma = sigma), N_per_group),
  intercept = intercept,
  x_group = x_group)
y_obs = broadcast(group_kernel, intercept = intercepts, x_group = x_per_group)
`;
}

// --- native reference fixtures ---------------------------------------

const FIXTURE_DIR = path.join(__dirname, '..', 'test', 'fixtures');
const NATIVE: Array<{ file: string; target: string }> = [
  { file: 'eight-schools.flatppl',          target: 'posterior' },
  { file: 'random-intercepts.flatppl',      target: 'y_obs' },
  { file: 'vector-obs-mvnormal.flatppl',    target: 'y' },
  { file: 'hierarchical-state-space.flatppl', target: 'y' },
  { file: 'nested-broadcast.flatppl',       target: 'y' },
];

// --- main -------------------------------------------------------------

function row(cols: Array<string | number>, w: number[]): string {
  return cols.map((c, i) => String(c).padStart(w[i])).join('  ');
}

(async () => {
  const N = 2000, RUNS = 5;
  const W = [34, 8, 8, 12];

  // eslint-disable-next-line no-console
  console.log(`\nPhase 8 baseline — materialise median ms (N=${N}, runs=${RUNS})\n`);

  console.log(row(['random-intercepts: cell axis K = G', 'K', 'Npg', 'median ms'], W));
  console.log('-'.repeat(70));
  for (const G of [4, 8, 16, 32, 64]) {
    const r = await timeMaterialise(genRandomIntercepts(G, 4), 'y_obs', N, RUNS);
    console.log(row(['  scaled', G, 4, r.ok ? r.ms.toFixed(2) : ('FAIL: ' + r.note)], W));
  }

  console.log('\n' + row(['native reference fixtures', '', '', 'median ms'], W));
  console.log('-'.repeat(70));
  for (const fx of NATIVE) {
    const src = fs.readFileSync(path.join(FIXTURE_DIR, fx.file), 'utf-8');
    const r = await timeMaterialise(src, fx.target, N, RUNS);
    console.log(row(['  ' + fx.file + ' :: ' + fx.target, '', '',
      r.ok ? r.ms.toFixed(2) : ('FAIL: ' + r.note)], W));
  }
  console.log('');
})();
