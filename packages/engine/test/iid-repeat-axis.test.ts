'use strict';

// =====================================================================
// matIid repeat axis — within-atom conditional independence (spec §06)
// =====================================================================
//
// `iid(M, k)` over a COMPOSITE M (superpose / select / pushfwd / …)
// has no single worker primitive, so the materialiser re-enters the
// kind-dispatch pipeline for M at an inflated count N·k and reshapes
// to [N, k]. The subtle part: when M conditions on atom-level
// stochastic VALUE draws (a per-atom `mu`, mixing weight `psi`, …),
// spec §06 requires "draw the value per atom, then k iid draws from M
// *at that value*" — atom i's k inner draws must SHARE atom i's value.
//
// The repeat-axis fix (materialiser.ts matIid + materialiser-shared
// `tileMeasureAtomMajor`) materialises the value once at parent-N and
// tiles it ×k (atom-major) while M's measure structure redraws freshly,
// so the k inner positions of atom i all see value_i. A naive inflate
// would redraw the value N·k times → correct marginal, WRONG joint.
//
// These tests discriminate the two: a shared per-atom value makes the
// k inner draws (near-)identical within an atom while still varying
// across atoms; an independent redraw would make them vary within an
// atom too.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    'source parses/analyses cleanly: ' + errs.map((d: any) => d.message).join('; '));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 42 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
    sampleCount: N,
    rootKey: [42, 0],
  };
  return { ctx, built };
}

// Read an [N, k] (outerRank=1) iid measure's flat data into per-atom rows.
function perAtomRows(m: any, N: number, k: number): number[][] {
  assert.ok(m && m.value && m.value.data, 'iid measure exposes a [N,k] Value');
  assert.deepEqual(m.value.shape, [N, k],
    'iid value shape is [N, k]; got ' + JSON.stringify(m.value.shape));
  const data = m.value.data;
  const rows: number[][] = [];
  for (let i = 0; i < N; i++) {
    const row: number[] = [];
    for (let j = 0; j < k; j++) row.push(data[i * k + j]);
    rows.push(row);
  }
  return rows;
}

function mean(xs: number[]): number {
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}
function std(xs: number[]): number {
  const m = mean(xs);
  let s = 0;
  for (const x of xs) s += (x - m) * (x - m);
  return Math.sqrt(s / xs.length);
}

// ---------------------------------------------------------------------
// 1. Shared per-atom component MEAN — the strong discriminator.
//    M's near-deterministic component is centred at the per-atom draw
//    `mu`, so each inner draw ≈ mu_i. Shared mu ⇒ the k draws collapse
//    onto mu_i (tiny within-atom spread); independent redraw would
//    spread them across the mu-prior (~10).
// ---------------------------------------------------------------------

test('matIid repeat axis: shared per-atom mu — k inner draws collapse onto mu_i', async () => {
  const N = 3000, k = 4;
  // superpose (two identical near-Dirac components at mu) forces the
  // composite fallback; both branches draw ≈ mu_i.
  const src = `
mu = draw(Normal(0, 10))
m = superpose(weighted(0.5, Normal(mu, 0.001)), weighted(0.5, Normal(mu, 0.001)))
y = iid(m, ${k})
`;
  const { ctx } = setupCtx(src, N);
  const rows = perAtomRows(await ctx.getMeasure('y'), N, k);

  // (a) Within-atom: the k draws share mu_i ⇒ spread ≈ component sd
  //     (0.001), NOT the mu-prior width (10). A redraw would give a
  //     within-atom spread ~10.
  let maxWithinSpread = 0;
  for (const row of rows) {
    maxWithinSpread = Math.max(maxWithinSpread, Math.max(...row) - Math.min(...row));
  }
  assert.ok(maxWithinSpread < 0.05,
    'within-atom spread must be ~component-sd (shared mu_i); got '
    + maxWithinSpread.toFixed(4) + ' (a redraw would be O(10))');

  // (b) Across atoms: per-atom means still span the mu-prior (~10) —
  //     the fix preserves between-atom variation, doesn't flatten it.
  const perAtomMeans = rows.map(mean);
  assert.ok(std(perAtomMeans) > 7,
    'across-atom std of per-atom means tracks the mu-prior (~10); got '
    + std(perAtomMeans).toFixed(3));
});

test('matIid repeat axis: marginal is preserved (mean≈0, std≈10)', async () => {
  const N = 4000, k = 4;
  const src = `
mu = draw(Normal(0, 10))
m = superpose(weighted(0.5, Normal(mu, 0.001)), weighted(0.5, Normal(mu, 0.001)))
y = iid(m, ${k})
`;
  const { ctx } = setupCtx(src, N);
  const rows = perAtomRows(await ctx.getMeasure('y'), N, k);
  const all = rows.flat();
  // The fix changes the JOINT, not the marginal: pooled draws are still
  // ~Normal(0, 10) (each draw is mu + tiny noise, mu ~ Normal(0,10)).
  assert.ok(Math.abs(mean(all)) < 1.0,
    'pooled mean ≈ 0; got ' + mean(all).toFixed(3));
  assert.ok(Math.abs(std(all) - 10) < 1.5,
    'pooled std ≈ 10; got ' + std(all).toFixed(3));
});

// ---------------------------------------------------------------------
// 2. Shared per-atom mixing WEIGHT — confirms the fix generalises to a
//    value that enters as a superpose weight (not a component param).
//    psi_i shared ⇒ atom i's per-atom mean ≈ (1 − 2·psi_i), so it
//    correlates strongly (negatively) with the canonical psi_i. An
//    independent redraw would wash psi out (per-atom mean ≈ 0, no
//    correlation). The fix also makes y reuse the SAME psi the rest of
//    the model sees, so we can read psi back via getMeasure('psi').
// ---------------------------------------------------------------------

test('matIid repeat axis: shared mixing weight psi correlates with per-atom mean', async () => {
  const N = 3000, k = 24;
  const src = `
psi = draw(Beta(1, 1))
m = superpose(weighted(psi, Normal(-1, 0.001)), weighted(1 - psi, Normal(1, 0.001)))
y = iid(m, ${k})
`;
  const { ctx } = setupCtx(src, N);
  const rows = perAtomRows(await ctx.getMeasure('y'), N, k);
  const psiM = await ctx.getMeasure('psi');
  const psi: number[] = Array.from(psiM.samples as Float64Array);
  assert.equal(psi.length, N, 'psi materialises at N atoms');

  const perAtomMeans = rows.map(mean);
  // Pearson correlation between psi_i and per-atom mean.
  const mp = mean(psi), my = mean(perAtomMeans);
  let cov = 0, vp = 0, vy = 0;
  for (let i = 0; i < N; i++) {
    const dp = psi[i] - mp, dy = perAtomMeans[i] - my;
    cov += dp * dy; vp += dp * dp; vy += dy * dy;
  }
  const corr = cov / Math.sqrt(vp * vy);
  // Shared psi ⇒ per-atom mean ≈ 1 − 2·psi_i ⇒ strong negative corr.
  assert.ok(corr < -0.8,
    'per-atom mean must track the shared psi_i (corr ≈ -1); got '
    + corr.toFixed(3) + ' (independent redraw would give ≈ 0)');
});

// ---------------------------------------------------------------------
// 3. pushfwd — ORDER-PRESERVING composite. `pushfwd(fn(_ + mu), base)`
//    has out[a] = base[a] + mu[a], so tiling the per-atom `mu` alone
//    yields within-atom conditional independence (no resampling
//    handler involved). Also a regression for the general gap: the
//    scalar pushfwd path must pass the fn body's EXTERNAL refs (`mu`),
//    not just f's own parameter — else the worker eval of `_ + mu`
//    leaves `mu` unbound.
// ---------------------------------------------------------------------

test('pushfwd: fn body referencing an external per-atom draw materialises', async () => {
  const N = 2000;
  const src = `
mu = draw(Normal(0, 10))
shifted = pushfwd(fn(_ + mu), Normal(0, 0.001))
`;
  const { ctx } = setupCtx(src, N);
  const m = await ctx.getMeasure('shifted');
  assert.ok(m && m.samples && m.samples.length === N,
    'pushfwd of fn(_ + mu) materialises N scalar samples (mu passed to the worker)');
  // shifted ≈ mu ~ Normal(0, 10): pooled mean ≈ 0, std ≈ 10.
  const xs = Array.from(m.samples as Float64Array);
  assert.ok(Math.abs(mean(xs)) < 1.0, 'mean ≈ 0; got ' + mean(xs).toFixed(3));
  assert.ok(Math.abs(std(xs) - 10) < 1.5, 'std ≈ 10; got ' + std(xs).toFixed(3));
});

test('matIid repeat axis: pushfwd composite (order-preserving) shares mu via tiling', async () => {
  const N = 2000, k = 4;
  const src = `
mu = draw(Normal(0, 10))
shifted = pushfwd(fn(_ + mu), Normal(0, 0.001))
y = iid(shifted, ${k})
`;
  const { ctx } = setupCtx(src, N);
  const rows = perAtomRows(await ctx.getMeasure('y'), N, k);
  // Order-preserving composite: tiling alone (no resampling handler)
  // gives within-atom collapse onto mu_i.
  let maxWithinSpread = 0;
  for (const row of rows) {
    maxWithinSpread = Math.max(maxWithinSpread, Math.max(...row) - Math.min(...row));
  }
  assert.ok(maxWithinSpread < 0.05,
    'within-atom spread ≈ base sd (shared mu_i); got ' + maxWithinSpread.toFixed(4));
  assert.ok(std(rows.map(mean)) > 7,
    'across-atom std tracks the mu-prior (~10); got ' + std(rows.map(mean)).toFixed(3));
});
