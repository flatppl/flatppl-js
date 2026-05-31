'use strict';

// =====================================================================
// hierarchical-models.test.ts — Phase F/G motivating test cases
// =====================================================================
//
// Documents the CURRENT behavior of two canonical SOTA hierarchical-
// model patterns (composite kernel bodies + outer-axis propagation —
// engine-concepts §20.10.9 / TODO-flatppl-js fusion thread (b)
// remaining sub-items). Each test fixes the expected end-state once
// Phase F/G land, and the present-day behavior so regressions in
// either direction are caught.
//
// Patterns under test:
//   1. Hierarchical repeated-measures (Pyro-style `plate` over
//      groups with `plate` over observations).
//   2. Random-intercepts regression (lme4 / Bambi / Stan-style
//      hierarchical linear model with per-group intercept +
//      shared slope).
//
// Both fixtures land in flatppl-examples + a copy in the engine
// test/fixtures/ tree (per CONVENTIONS.md: examples canonical,
// tests carry their own copy for resilience).
//
// **Today (post fusion (b) MVP):** both fixtures CLASSIFY but
// materialise fails because matKernelBroadcast rejects the
// user-kernel head — the kernel body is `lawof(iid(<Dist>, n))`
// rather than `lawof(<Dist>)`, so the fusion (b) MVP refuses to
// inline.
//
// **Target (Phase F):** matKernelBroadcast or the dissolver
// recognises the composite kernel body, lifts the kernel application
// into a tiled broadcast + reshape, and produces a shape
// [N_atom, G, N_per_group] obs measure.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function readFixture(name: string): string {
  const p = path.join(__dirname, 'fixtures', name);
  return fs.readFileSync(p, 'utf-8');
}

function setupCtx(src: string, N: number) {
  const lifted = processSource(src);
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
    rootSeed: 42,
  };
  return { ctx, derivations: built.derivations, bindings: built.bindings };
}

// =====================================================================
// 1. Hierarchical repeated-measures — `kernelof(iid(Normal,N), mu)`
//    broadcast over a per-group mu vector
// =====================================================================
//
// Per spec §04 + §06: broadcast iterates over its collection-arg
// outer axis; per cell the kernel's body is drawn. With body
// `iid(Normal(mu, s), N_per_group)`, each cell yields a length-
// N_per_group vector — so the whole result is shape [G, N_per_group]
// per engine atom.

test('hierarchical-repeated-measures: classifies + materialises (Phase F)', async () => {
  const src = readFixture('hierarchical-repeated-measures.flatppl');
  const { ctx, derivations } = setupCtx(src, 50);

  // obs classifies as kernelbroadcast (Phase F extends
  // classifyKernelBroadcast to recognise user-kernel heads with
  // iid-body shape).
  assert.ok(derivations.obs, 'obs has a derivation');
  assert.equal(derivations.obs.kind, 'kernelbroadcast',
    'kernel-of-iid composite routes via matKernelBroadcast');

  // Phase F result shape: [N_atom=50, G=4, N_per_group=5].
  const m = await ctx.getMeasure('obs');
  assert.ok(m && m.value && Array.isArray(m.value.shape),
    'obs materialises to a shape-tagged Value');
  assert.deepEqual(m.value.shape, [50, 4, 5],
    'Phase F result shape: [N_atom, G, N_per_group]');
});

// =====================================================================
// 2. Random-intercepts regression — per-group intercept + per-group
//    regressors, kernel body is `iid(Normal(int + slope * x, s), N)`
// =====================================================================
//
// Same shape as eight-schools BUT each group has multiple obs (the
// per-group N_per_group dim). The kernel body uses both the
// placeholder `intercept` (scalar per group) and `x_group` (rank-1
// per group), making it a more complex composite-body case than
// the simpler hierarchical test above.
//
// **Phase 4.1 unblocks this case.** The vec-per-cell broadcast arg
// `x_group` (shape [G, N_per_group] atom-indep) triggers the new
// `_executeIidCompositeVecPerCell` path in mat-broadcast.ts: each
// (j, r) is sampled independently with the (j, r) per-atom slice
// of every broadcast arg, side-stepping the substituted-body
// batched-eval shape mismatch that the existing scalar-per-cell
// path produced.

test('random-intercepts: vec-per-cell iid composite materialises (Phase 4.1)', async () => {
  const src = readFixture('random-intercepts.flatppl');
  const { ctx, derivations } = setupCtx(src, 100);

  // Classify.
  assert.ok(derivations.y_obs,
    'y_obs has a derivation (classifier accepts the broadcast IR)');
  assert.equal(derivations.y_obs.kind, 'kernelbroadcast',
    'composite-body iid kernel routes via matKernelBroadcast');

  // Materialise.
  const m = await ctx.getMeasure('y_obs');
  assert.deepEqual(m.value.shape, [100, 3, 4],
    'Phase 4.1 result shape: [N_atom, G, N_per_group]');

  // No NaN: the Phase 4.1 substitute-then-evaluate path produces
  // finite samples (vs. NaN that earlier scalar-per-cell-only path
  // produced for vec-per-cell args).
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      'y_obs sample at flat index ' + i + ' is finite');
  }

  // Conformance oracle: SAMPLES VARY ACROSS (j, r). For each atom i
  // the (j, r) layout should produce distinguishable values because
  // x_per_group[j][r] varies with r (linear-predictor changes) AND
  // intercepts[j] varies with j (per-group intercept). A trivial
  // collapsed-axis bug would produce identical samples across (j, r);
  // a misaligned-stride bug would produce identical samples along
  // one of the two axes.
  //
  // We probe by computing the cross-(j, r) std at fixed atom across
  // the (j, r) plane — this excludes per-atom RNG variance and
  // isolates structural variation. Expected: substantially > 0.
  const N = 100, G = 3, NPG = 4;
  function flatStd(buf: Float64Array, start: number, end: number, step = 1): number {
    let n = 0, mean = 0, m2 = 0;
    for (let k = start; k < end; k += step) {
      n++;
      const d = buf[k] - mean;
      mean += d / n;
      m2 += d * (buf[k] - mean);
    }
    return Math.sqrt(m2 / Math.max(1, n - 1));
  }
  let nonzeroVariationCount = 0;
  for (let i = 0; i < N; i++) {
    const std = flatStd(m.value.data, i * G * NPG, (i + 1) * G * NPG);
    if (std > 1e-9) nonzeroVariationCount++;
  }
  assert.ok(nonzeroVariationCount >= N * 0.95,
    'samples vary across (j, r) for almost every atom (got '
    + nonzeroVariationCount + '/' + N + ')');
});

// =====================================================================
// 3. Eight-schools reference fixture — baseline (Phase 1.1 Normal hot
//    path, copied from flatppl-examples per CONVENTIONS.md examples-
//    vs-test-fixtures convention)
// =====================================================================
//
// Used by the Phase 4 reference-fixture set as the "scalar-per-cell"
// baseline: each school's `y` is a single Normal observation
// parameterised by `theta[j]` and `std_errs_data[j]`. Kernel broadcast
// hits the Normal closed-form fast path (Phase 1.1
// kernel-broadcast-handlers.ts). Result shape: [N_atom, J=8].

test('eight-schools: kernel broadcast (scalar-per-cell baseline)', async () => {
  const src = readFixture('eight-schools.flatppl');
  const { ctx, derivations } = setupCtx(src, 50);

  assert.ok(derivations.y, 'y has a derivation');
  // y = Normal.(theta, std_errs_data) — broadcast over theta and
  // std_errs_data. Both are length-J vectors; per cell j the params
  // are scalars. The analyser lifts the inline `Normal.(...)` into
  // an `__anon` binding (alias on `y`); the anon classifies as
  // kernelbroadcast (the Normal hot path in
  // kernel-broadcast-handlers.ts handles it).
  assert.equal(derivations.y.kind, 'alias',
    'y aliases the lifted broadcast binding');
  // The aliased binding is the actual kernel-broadcast derivation.
  const anonName = (derivations.y as any).target;
  if (anonName && derivations[anonName]) {
    assert.equal(derivations[anonName].kind, 'kernelbroadcast',
      'y\'s lifted alias target is kernel-broadcast');
  }

  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [50, 8],
    'eight-schools y shape: [N_atom, J=8]');

  // No NaN.
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      'y sample at flat index ' + i + ' is finite');
  }
});
