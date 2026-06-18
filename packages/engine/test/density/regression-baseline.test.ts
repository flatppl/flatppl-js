'use strict';
// Regression baseline: scores a fixed set of models at a fixed seed and diffs
// against the committed golden (test/fixtures/regression-baseline.json). The
// golden was validated against independent oracles (Distributions.jl /
// ROOT) at capture time — see each entry's "oracle" note in the JSON. ANY
// movement on an existing entry after an engine change is a regression: stop
// and investigate. New multi-axis models are ADDED to the golden as they
// start working; existing entries never move.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('../..');
const { createWorkerHandler } = require('../../worker.ts');

const SEED = 0xBA5E;
const TOL = 1e-9; // density (deterministic); sampling stats use 0.05

// Build a materialiser context for a given source text and sample count.
// Exported so later tasks can reuse the pattern without copying it.
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

// Score one model spec: returns a flat map of { 'density:<name>': value,
// 'mean:<binding>': value }. Exported for reuse by later tasks.
async function scoreModel(spec: any) {
  const baseSrc = fs.readFileSync(
    path.join(__dirname, '../fixtures/baseline', spec.file),
    'utf8'
  );
  const out: Record<string, number> = {};

  // Density extractions: append the probe line then materialise __score__
  for (const ex of (spec.density || [])) {
    const src = baseSrc + '\n' + ex.score + '\n';
    const { ctx } = ctxFor(src, 1);
    const m = await ctx.getMeasure('__score__');
    const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
    if (!s || s.length === 0) {
      throw new Error(`scoreModel: ${spec.file} density '${ex.name}' produced no `
        + `data (measure shape unexpected) — the harness cannot score this point`);
    }
    out['density:' + ex.name] = s[0];
  }

  // Sample-mean extractions: materialise binding, take arithmetic mean
  for (const ex of (spec.samplemean || [])) {
    const { ctx } = ctxFor(baseSrc, ex.N ?? 20000);
    const m = await ctx.getMeasure(ex.binding);
    const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
    if (!s || s.length === 0) {
      throw new Error(`scoreModel: ${spec.file} binding '${ex.binding}' produced no `
        + `samples (measure shape unexpected)`);
    }
    let mu = 0;
    for (let i = 0; i < s.length; i++) mu += s[i];
    out['mean:' + ex.binding] = mu / s.length;
  }

  return out;
}

const GOLDEN = JSON.parse(
  fs.readFileSync(path.join(__dirname, 'regression-baseline.json'), 'utf8')
);

for (const spec of GOLDEN.models) {
  test(`baseline: ${spec.file} matches golden`, async () => {
    const got = await scoreModel(spec);
    for (const k of Object.keys(spec.expect)) {
      const exp = spec.expect[k] as number;
      const tol = k.startsWith('mean:') ? 0.05 : TOL;
      assert.ok(
        Math.abs(got[k] - exp) <= tol,
        `${spec.file} [${k}]: got ${got[k]}, golden ${exp} (Δ ${Math.abs(got[k] - exp)})`
      );
    }
  });
}

module.exports = { scoreModel, ctxFor };
