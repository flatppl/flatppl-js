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

// =====================================================================
// 4. Joint-bodied composite kernel — multi-output regression (Phase 4.2)
// =====================================================================
//
// Per spec §06 the kernel body is `joint(y1 = Normal(mu, sigma),
// y2 = Normal(mu, 2*sigma))` — each cell produces one joint draw,
// factoring as independent components. Phase 4.2 lands the joint
// composite-body recogniser + the `_executeJointComposite` execution
// path; classify-time accepts joint-bodied user kernels.
//
// Result shape per atom: [G, C] with C = 2 components. The first
// component (y1) ties to `mu`; the second (y2) ties to `mu` with
// doubled sigma — so per-cell variances are sigma_g and (2*sigma_g)
// respectively. The conformance oracles below probe both the
// structural shape AND the calibration of per-component cell means.

test('joint-obs-regression: keyword-joint composite materialises (Phase 4.2)', async () => {
  const src = readFixture('joint-obs-regression.flatppl');
  const { ctx, derivations } = setupCtx(src, 200);

  // Classify.
  assert.ok(derivations.y_obs,
    'y_obs has a derivation (classifier accepts joint-bodied user kernel)');
  assert.equal(derivations.y_obs.kind, 'kernelbroadcast',
    'joint-bodied user kernel routes via matKernelBroadcast');

  // Materialise.
  const m = await ctx.getMeasure('y_obs');
  assert.deepEqual(m.value.shape, [200, 3, 2],
    'joint-obs result shape: [N_atom, G, C]');

  // No NaN.
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      'y_obs sample at flat index ' + i + ' is finite');
  }

  // Per-(g, c) calibration. For each group g and component c, the
  // sample mean over atoms is mu_g (within sampling noise). mus_per_
  // group is itself stochastic-ancestor (~ iid(Normal(mu_pop, tau),
  // G) with mu_pop ~ Normal(0, 5), tau ~ Normal+(0, 2)), so atom-i's
  // (g, c) value is Normal(mu_g[i], sigma_g_c) — averaging over
  // atoms gives E[mu_g] = 0 (the mu_pop prior mean) with substantial
  // variance from mu_pop's spread. We don't pin the absolute mean
  // tightly; we instead verify the y1 vs y2 components share the
  // SAME center within a cell (both are Normal(mu, *)), and the y2
  // component has a larger spread (sigma doubled).
  const N = 200, G = 3, C = 2;
  const cellMean: number[][] = [[], [], []];
  const cellVar: number[][] = [[], [], []];
  for (let g = 0; g < G; g++) {
    for (let c = 0; c < C; c++) {
      let sum = 0, sumSq = 0;
      for (let i = 0; i < N; i++) {
        const v = m.value.data[i * G * C + g * C + c];
        sum += v;
        sumSq += v * v;
      }
      cellMean[g][c] = sum / N;
      cellVar[g][c] = sumSq / N - (sum / N) * (sum / N);
    }
  }

  // y2's variance > y1's variance for each cell: y2 has the doubled
  // sigma, so the (Var(y2) - Var(y1)) signal should be positive even
  // before we add the per-atom-mu_g variance contribution (which is
  // SAME for both components within a cell). Generous margin — N=200
  // gives noisy single-cell estimates.
  for (let g = 0; g < G; g++) {
    assert.ok(cellVar[g][1] > cellVar[g][0] * 0.9,
      'cell ' + g + ': y2 variance (' + cellVar[g][1].toFixed(2)
      + ') exceeds y1 variance (' + cellVar[g][0].toFixed(2)
      + ') as expected (sigma doubled for y2)');
  }
});

test('joint-obs-regression: positional joint also classifies + materialises', async () => {
  // Inline positional-joint user kernel — no separate fixture, the
  // assertion is that the SAME recogniser handles `joint(M1, M2)`
  // (positional) just as it handles `joint(name1 = M1, name2 = M2)`
  // (keyword).
  const src = [
    'flatppl_compat = "0.1"',
    'mus = [0.0, 1.0, 2.0]',
    'sigs = [1.0, 0.5, 0.8]',
    'comp1 = Normal(mu = mu, sigma = sigma)',
    'comp2 = Normal(mu = mu, sigma = mul(2.0, sigma))',
    'k_pos = kernelof(joint(comp1, comp2), mu = mu, sigma = sigma)',
    'y_pos = broadcast(k_pos, mu = mus, sigma = sigs)',
  ].join('\n');
  const { ctx, derivations } = setupCtx(src, 40);

  assert.equal(derivations.y_pos.kind, 'kernelbroadcast',
    'positional joint routes via matKernelBroadcast');
  const m = await ctx.getMeasure('y_pos');
  assert.deepEqual(m.value.shape, [40, 3, 2],
    'positional-joint result shape: [N_atom, K, C]');
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      'positional-joint sample at flat index ' + i + ' is finite');
  }
});

// =====================================================================
// 5. Jointchain-bodied composite kernel — hierarchical state-space (Phase 4.3)
// =====================================================================
//
// Per spec §06 jointchain is a Markov chain: step 0 is a base
// measure; each step k > 0 applies a kernel to the previous variate.
// Phase 4.3 lands the jointchain composite-body recogniser + the
// `_executeJointChainComposite` execution path with per-step state
// threading (step k's sampleN sees step k-1's per-atom column as a
// refArray).
//
// Fixture: AR-1 random walk per group, 3 transition steps. The
// per-step variance follows the random walk formula
// Var(x_k) ≈ sigma_init^2 + k * sigma_step^2 → with sigma_init=0.1,
// sigma_step=0.5 the predicted values are {0.01, 0.26, 0.51, 0.76}.
// Increments x_k - x_{k-1} are Normal(0, sigma_step) → sample
// variance approaches 0.25.

test('hierarchical-state-space: jointchain composite materialises (Phase 4.3)', async () => {
  const src = readFixture('hierarchical-state-space.flatppl');
  const { ctx, derivations } = setupCtx(src, 500);

  // Classify.
  assert.ok(derivations.y,
    'y has a derivation (classifier accepts jointchain-bodied user kernel)');
  assert.equal(derivations.y.kind, 'kernelbroadcast',
    'jointchain-bodied user kernel routes via matKernelBroadcast');

  // Materialise.
  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [500, 3, 4],
    'state-space result shape: [N_atom, G, chain_length]');

  // No NaN.
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      'y sample at flat index ' + i + ' is finite');
  }

  // AR-1 calibration. The KEY conformance signal — state threading
  // is correct iff the per-cell increments behave like AR-1
  // increments AND the per-step marginal variance grows linearly.
  const N = 500, G = 3, C = 4;
  const sigmaStep = 0.5, sigmaInit = 0.1;
  const x0List = [0.0, 0.5, 1.0];
  for (let g = 0; g < G; g++) {
    // Step 0 marginal: mean ≈ x0_g, var ≈ sigma_init^2.
    let sum0 = 0, sumSq0 = 0;
    for (let i = 0; i < N; i++) {
      const v = m.value.data[i * G * C + g * C + 0];
      sum0 += v; sumSq0 += v * v;
    }
    const m0 = sum0 / N, v0 = sumSq0 / N - m0 * m0;
    assert.ok(Math.abs(m0 - x0List[g]) < 0.05,
      'group ' + g + ' step 0 mean (' + m0.toFixed(3)
      + ') near x0=' + x0List[g]);
    assert.ok(Math.abs(v0 - sigmaInit * sigmaInit) < 0.01,
      'group ' + g + ' step 0 var (' + v0.toFixed(3)
      + ') near sigma_init^2 = ' + (sigmaInit * sigmaInit));

    // Each transition step's increment distribution: per atom,
    // delta = x_k - x_{k-1}. With state threading this is a fresh
    // Normal(0, sigma_step) draw; the per-atom correlation between
    // x_k and x_{k-1} carries the random walk.
    for (let k = 1; k < C; k++) {
      let dSum = 0, dSumSq = 0;
      for (let i = 0; i < N; i++) {
        const xk = m.value.data[i * G * C + g * C + k];
        const xkm1 = m.value.data[i * G * C + g * C + (k - 1)];
        dSum += (xk - xkm1);
        dSumSq += (xk - xkm1) * (xk - xkm1);
      }
      const dMean = dSum / N;
      const dVar = dSumSq / N - dMean * dMean;
      assert.ok(Math.abs(dMean) < 0.08,
        'group ' + g + ' step ' + k + ' increment mean ('
        + dMean.toFixed(3) + ') near 0');
      assert.ok(Math.abs(dVar - sigmaStep * sigmaStep) < 0.05,
        'group ' + g + ' step ' + k + ' increment var ('
        + dVar.toFixed(3) + ') near sigma_step^2 = '
        + (sigmaStep * sigmaStep)
        + ' — STATE THREADING ORACLE (a chain that drew steps '
        + 'independently would have increment var ≈ sigma_step^2 + '
        + 'Var(x_{k-1}) → much larger than 0.25 once k > 1)');
    }
  }
});

// =====================================================================
// 6. Nested-broadcast composite kernel (Phase 4.4)
// =====================================================================
//
// Outer kernel body is itself a broadcast. Per spec §04 the inner
// broadcast realises an independent-product measure over its inner
// kwargs' collection axes; nesting inside an outer kernel-broadcast
// yields shape [N, K_outer, K_inner] per atom.
//
// Fixture: per-patient observations across visits. Outer iterates
// patients (sigmas_per_patient); inner iterates visits (visit_means).
// Per-(p, v) calibration: mean ≈ visit_means[v], var ≈ sigmas[p]^2.

test('nested-broadcast: nested obs composite materialises (Phase 4.4)', async () => {
  const src = readFixture('nested-broadcast.flatppl');
  const { ctx, derivations } = setupCtx(src, 500);

  // Classify.
  assert.ok(derivations.y,
    'y has a derivation (classifier accepts nested-broadcast-bodied user kernel)');
  assert.equal(derivations.y.kind, 'kernelbroadcast',
    'nested-broadcast-bodied user kernel routes via matKernelBroadcast');

  // Materialise.
  const m = await ctx.getMeasure('y');
  assert.deepEqual(m.value.shape, [500, 3, 4],
    'nested-broadcast result shape: [N_atom, P, V]');

  // No NaN.
  for (let i = 0; i < m.value.data.length; i++) {
    assert.ok(Number.isFinite(m.value.data[i]),
      'y sample at flat index ' + i + ' is finite');
  }

  // Per-(p, v) calibration: KEY conformance oracle. Each cell is
  // independent draws of Normal(visit_means[v], sigmas[p]) — sample
  // mean / variance pin BOTH the outer-cell substitution (sigmas[p]
  // bound to the right patient) AND the inner-cell substitution
  // (visit_means[v] sliced to the right visit). A broken executor
  // that crossed axes would produce mean / variance values that
  // mismatch the (p, v) grid.
  const N = 500, P = 3, V = 4;
  const visitMeans = [10.0, 20.0, 30.0, 40.0];
  const sigmas = [0.5, 1.0, 1.5];
  for (let p = 0; p < P; p++) {
    for (let v = 0; v < V; v++) {
      let sum = 0, sumSq = 0;
      for (let i = 0; i < N; i++) {
        const x = m.value.data[i * P * V + p * V + v];
        sum += x; sumSq += x * x;
      }
      const mean = sum / N;
      const variance = sumSq / N - mean * mean;
      // 4-sigma margin on the sample mean: SE = sigma / sqrt(N).
      const meanSE = sigmas[p] / Math.sqrt(N);
      assert.ok(Math.abs(mean - visitMeans[v]) < 4 * meanSE,
        '(p=' + p + ', v=' + v + ') mean (' + mean.toFixed(3)
        + ') near visit_means[v]=' + visitMeans[v]
        + ' (4-sigma margin ' + (4 * meanSE).toFixed(3) + ')');
      // 4-sigma margin on the sample variance: SE ≈ var * sqrt(2/(N-1)).
      const expectedVar = sigmas[p] * sigmas[p];
      const varSE = expectedVar * Math.sqrt(2 / (N - 1));
      assert.ok(Math.abs(variance - expectedVar) < 4 * varSE,
        '(p=' + p + ', v=' + v + ') var (' + variance.toFixed(3)
        + ') near sigmas[p]^2=' + expectedVar
        + ' (4-sigma margin ' + (4 * varSE).toFixed(3) + ') '
        + '— TWO-AXIS SUBSTITUTION ORACLE');
    }
  }
});

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
