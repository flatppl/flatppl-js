'use strict';

// Sampling a superpose of IMPORTANCE-WEIGHTED density-by-formula measures
// (Buffy #307, second bug). `normalize(weighted(fn, Lebesgue(S)))` materialises
// as an importance-weighted empirical measure: UNIFORM sample positions whose
// density lives in the per-atom logWeights. matSuperpose used to run its
// block-aware K=1 per-index selection directly on those parents — picking
// between the two parents' atoms at their (uniform) positions without ever
// concentrating by weight — so a mixture of such components plotted FLAT
// (identical to the bare Lebesgue reference). The density path was correct
// (#103); only the sampled/empirical measure the viewer plots was wrong — a
// density-vs-sampling divergence (scar zone).
//
// Fix: at K=1, SIR-resample each importance-weighted parent to equal-weight
// atoms (baking its density into the sample positions) BEFORE the mixture
// selection; equal-weight parents (positions already represent their law, e.g.
// a per-atom Normal(mu_i, …) draw) and the K>1 repeat-block path are untouched.
//
// INDEPENDENT ORACLE (closed form, NOT engine output):
//   D1 = normalize(weighted(x -> x,       Lebesgue(0,1))) has density 2x
//   D2 = normalize(weighted(x -> 1 - x,   Lebesgue(0,1))) has density 2(1-x)
//   mixture = superpose(weighted(0.7, D1), weighted(0.3, D2))
//     density = 0.7·2x + 0.3·2(1-x) = 0.8x + 0.6   (rising 0.6 → 1.4 on [0,1])
//   ∫ over [0,¼],[¼,½],[½,¾],[¾,1] = 0.175, 0.225, 0.275, 0.325 (sums to 1).
// A flat/uniform mixture (the bug) would give 0.25 each.

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function buildCtx(src: string, N: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 7 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: 7, rootSeed: 7, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

// Weight-aware quarter-bin fractions on [0,1] — mirrors how the viewer
// histograms an empirical measure (importance weights applied when present).
// A resampled equal-weight measure has uniform weights, so this reduces to the
// plain position histogram there.
function quarterFracs(measure: any): number[] {
  const s = measure.samples as Float64Array;
  const lw = measure.logWeights as Float64Array | null;
  const c = [0, 0, 0, 0]; let tot = 0;
  for (let k = 0; k < s.length; k++) {
    const x = s[k]; if (!Number.isFinite(x)) continue;
    const wk = lw ? Math.exp(lw[k]) : 1;
    if (!(wk > 0)) continue;
    const b = Math.min(3, Math.max(0, Math.floor(x * 4)));
    c[b] += wk; tot += wk;
  }
  return c.map((v) => v / Math.max(1e-300, tot));
}

const MODEL = `
flatppl_compat = "0.1"
up   = x -> x
down = x -> 1 - x
D1 = normalize(weighted(up,   Lebesgue(support = interval(0.0, 1.0))))
D2 = normalize(weighted(down, Lebesgue(support = interval(0.0, 1.0))))
mixture = superpose(weighted(0.7, D1), weighted(0.3, D2))
`;

test('superpose of importance-weighted components samples the mixture density, not flat (#307)', async () => {
  const ORACLE = [0.175, 0.225, 0.275, 0.325]; // ∫(0.8x+0.6) per quarter; closed form
  const ctx = buildCtx(MODEL, 40000);
  const m = await ctx.getMeasure('mixture');
  const f = quarterFracs(m);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(f[i] - ORACLE[i]) < 0.02,
      `quarter ${i}: sampled ${f[i].toFixed(3)} ≈ oracle ${ORACLE[i]} (Δ ${Math.abs(f[i] - ORACLE[i]).toFixed(3)})`);
  }
  // And explicitly NOT the flat/uniform shape the bug produced.
  const flat = f.every((v) => Math.abs(v - 0.25) < 0.02);
  assert.ok(!flat, `mixture must not sample flat/uniform; got ${f.map((v) => v.toFixed(3)).join(' ')}`);
});

test('a single normalize(weighted(fn, Lebesgue)) also samples its density (#307 primitive)', async () => {
  // D1 alone = triangular 2x: quarter fractions ∫2x = 0.0625, 0.1875, 0.3125, 0.4375.
  const ORACLE = [0.0625, 0.1875, 0.3125, 0.4375];
  const ctx = buildCtx(MODEL, 40000);
  const m = await ctx.getMeasure('D1');
  const f = quarterFracs(m);
  for (let i = 0; i < 4; i++) {
    assert.ok(Math.abs(f[i] - ORACLE[i]) < 0.02,
      `D1 quarter ${i}: sampled ${f[i].toFixed(3)} ≈ oracle ${ORACLE[i]} (Δ ${Math.abs(f[i] - ORACLE[i]).toFixed(3)})`);
  }
});
