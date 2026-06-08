'use strict';

// H2 / L2 conformance — `locscale(m, shift, scale)` for non-scalar scale
// (spec §06 sec:locscale).
//
// `locscale` desugars to an affine pushforward. Before the H2 fix the
// lift unconditionally synthesized the ELEMENTWISE scalar inverse
// `(_ − shift)/scale` + scalar log-volume `log|scale|`, which is correct
// only for a SCALAR scale:
//   - MATRIX scale (the spec form `locscale(MvNormal(zeros(n),eye(n)),
//     mu, lower_cholesky(cov))` ≡ `MvNormal(mu, cov)`): the density path
//     did `scalar / matrix` through the non-shape-aware `div` → NaN, and
//     that NaN slipped past the `typeof x !== 'number'` guard → silent
//     garbage log-density.
//   - VECTOR (per-component) scale: unsupported.
//   - scale = 0 (L2): non-invertible → +Inf/NaN density.
//
// This file pins:
//   1. SCALAR scale (incl. scale < 0) still matches Normal(mu, |sigma|)
//      density (regression).
//   2. MATRIX-scale locscale density is CORRECT (vs MvNormal(mu, cov)
//      logpdf) and NEVER returns NaN/Inf silently. Forward sampling
//      recovers the right mean.
//   3. scale = 0 (L2) yields a clear non-invertible diagnostic.
//   4. VECTOR-scale yields a clear unsupported diagnostic.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 20000;
const ROOT_SEED    = 0x10C5CA1E;

function makeCtx(source: any, opts?: any) {
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
}

// =====================================================================
// 1. SCALAR scale regression — incl. scale < 0
// =====================================================================

test('scalar locscale density matches Normal(mu, |sigma|) logpdf (incl. sigma < 0)', async () => {
  // Normal(mu,|sigma|) closed-form logpdf — the abs makes a negative
  // scale equivalent to its magnitude (the affine log-volume is
  // log|scale|).
  const normLogpdf = (x: number, mu: number, sigma: number) =>
    -0.5 * Math.log(2 * Math.PI) - Math.log(Math.abs(sigma))
    - 0.5 * ((x - mu) / sigma) ** 2;

  for (const [MU, SIGMA] of [[3.0, 2.0], [3.0, -2.0], [-1.0, 0.5]]) {
    for (const yVal of [MU - 1, MU, MU + 1.3]) {
      const { ctx } = makeCtx(`
        base = Normal(mu = 0, sigma = 1)
        y = locscale(base, ${MU}, ${SIGMA})
        lp = logdensityof(y, ${yVal})
      `);
      const lp = await ctx.getMeasure('lp');
      const got = lp.samples[0];
      const want = normLogpdf(yVal, MU, SIGMA);
      assert.ok(Number.isFinite(got), `density at ${yVal} must be finite, got ${got}`);
      assert.ok(Math.abs(got - want) < 1e-9,
        `scalar locscale (mu=${MU}, sigma=${SIGMA}) logpdf at ${yVal}: got ${got}, want ${want}`);
    }
  }
});

// =====================================================================
// 2. MATRIX scale — CORRECT density (made correct, not diagnosed)
// =====================================================================
//
// `locscale(MvNormal(zeros(2),eye(2)), mu, lower_cholesky(cov))`
// ≡ `MvNormal(mu, cov)`. We use INLINE literals so the affine
// paramIRs are self-contained (no binding-env resolution needed by the
// top-level density entry point).

const MU_VEC  = [1.0, 2.0];
const COV     = [[2.0, 0.5], [0.5, 1.0]];

const MATRIX_MODEL = `
base = MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))
Y = locscale(base, [${MU_VEC[0]}, ${MU_VEC[1]}], lower_cholesky(rowstack([[${COV[0][0]}, ${COV[0][1]}], [${COV[1][0]}, ${COV[1][1]}]])))
`;

function toValue(arr: number[]) {
  return { shape: [arr.length], data: Float64Array.from(arr) };
}
function toMatValue(m: number[][]) {
  const r = m.length, c = m[0].length;
  const d = new Float64Array(r * c);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) d[i * c + j] = m[i][j];
  return { shape: [r, c], data: d };
}

test('matrix-scale locscale routes through the affine registry (pushfwd, registryName=affine)', () => {
  const { built } = makeCtx(MATRIX_MODEL);
  const der = built.derivations['Y'];
  assert.ok(der, 'Y has a derivation');
  assert.equal(der.kind, 'pushfwd', 'Y derives as a pushfwd');
  const expanded = orchestrator.expandMeasure('Y',
    { derivations: built.derivations, bindings: built.bindings });
  assert.ok(expanded && expanded.bijection, 'Y expands with bijection metadata');
  assert.equal(expanded.bijection.registryName, 'affine',
    'matrix-scale locscale uses the affine bijection registry');
});

test('matrix-scale locscale density is CORRECT (== MvNormal(mu,cov) logpdf), never NaN/Inf', () => {
  const { built } = makeCtx(MATRIX_MODEL);
  const expanded = orchestrator.expandMeasure('Y',
    { derivations: built.derivations, bindings: built.bindings });

  const density = require('../density.ts');
  const densityPrims = require('../density-prims.ts');
  for (const obsArr of [[0.5, 1.5], [1.0, 2.0], [-0.3, 3.1]]) {
    const obs = toValue(obsArr);
    const lp = density.logDensity(expanded, obs, {}, {});
    const expected = densityPrims.MV_DENSITY_FNS.MvNormal(obs,
      { mu: toValue(MU_VEC), cov: toMatValue(COV) });
    assert.ok(Number.isFinite(lp),
      `matrix-scale locscale density at ${obsArr} must be finite, got ${lp}`);
    assert.ok(Math.abs(lp - expected) < 1e-9,
      `matrix-scale locscale density at ${obsArr}: got ${lp}, want ${expected}`);
  }
});

test('matrix-scale locscale samples recover the right mean (forward sampling works)', async () => {
  const { ctx } = makeCtx(MATRIX_MODEL);
  const Y = await ctx.getMeasure('Y');
  assert.ok(Y && Y.value, 'Y materialises to a vector-batched value');
  const N = Y.value.shape[0] as number;
  const D = Y.value.shape[1] as number;
  assert.equal(D, 2, 'event dim is 2');
  const d = Y.value.data;
  let m0 = 0, m1 = 0;
  for (let i = 0; i < N; i++) { m0 += d[i * D]; m1 += d[i * D + 1]; }
  m0 /= N; m1 /= N;
  assert.ok(Math.abs(m0 - MU_VEC[0]) < 0.1,
    `component-0 sample mean ${m0} should be near ${MU_VEC[0]}`);
  assert.ok(Math.abs(m1 - MU_VEC[1]) < 0.1,
    `component-1 sample mean ${m1} should be near ${MU_VEC[1]}`);
});

// =====================================================================
// 2b. NON-lower-triangular MATRIX scale — distribution still correct
// =====================================================================
//
// For a non-triangular scale S, lower_cholesky(S·Sᵀ) ≠ S, but L·Lᵀ =
// S·Sᵀ, so the pushforward distribution is N(mu, S·Sᵀ) regardless of
// which factor the registry picks. Density must equal MvNormal(mu, S·Sᵀ)
// and never be NaN/Inf.

const S_NONTRI = [[1.0, 0.5], [0.3, 1.0]];   // upper entry 0.5 ≠ 0
// S·Sᵀ = [[1.25, 0.8], [0.8, 1.09]]
function matGram(m: number[][]) {
  const r = m.length, c = m[0].length;
  const g: number[][] = [];
  for (let i = 0; i < r; i++) {
    g[i] = [];
    for (let j = 0; j < r; j++) {
      let s = 0;
      for (let k = 0; k < c; k++) s += m[i][k] * m[j][k];
      g[i][j] = s;
    }
  }
  return g;
}

const NONTRI_MODEL = `
base = MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))
Y = locscale(base, [${MU_VEC[0]}, ${MU_VEC[1]}], rowstack([[${S_NONTRI[0][0]}, ${S_NONTRI[0][1]}], [${S_NONTRI[1][0]}, ${S_NONTRI[1][1]}]]))
`;

test('non-triangular matrix-scale locscale density == MvNormal(mu, S·Sᵀ), never NaN/Inf', () => {
  const { built } = makeCtx(NONTRI_MODEL);
  const expanded = orchestrator.expandMeasure('Y',
    { derivations: built.derivations, bindings: built.bindings });
  assert.equal(expanded.bijection.registryName, 'affine',
    'non-triangular matrix scale must still route through the affine registry');

  const density = require('../density.ts');
  const densityPrims = require('../density-prims.ts');
  const cov = matGram(S_NONTRI);               // S·Sᵀ
  for (const obsArr of [[0.5, 1.5], [1.0, 2.0], [-0.3, 3.1]]) {
    const obs = toValue(obsArr);
    const lp = density.logDensity(expanded, obs, {}, {});
    const expected = densityPrims.MV_DENSITY_FNS.MvNormal(obs,
      { mu: toValue(MU_VEC), cov: toMatValue(cov) });
    assert.ok(Number.isFinite(lp),
      `density at ${obsArr} must be finite, got ${lp}`);
    assert.ok(Math.abs(lp - expected) < 1e-9,
      `non-triangular locscale density at ${obsArr}: got ${lp}, want ${expected}`);
  }
});

test('non-triangular matrix-scale locscale uses S·Sᵀ, not Sᵀ·S or raw S (adversarial)', () => {
  // S_NONTRI is non-symmetric, so S·Sᵀ ≠ Sᵀ·S (diagonals swap). The density
  // must match MvNormal(mu, S·Sᵀ) and NOT MvNormal(mu, Sᵀ·S) (a row/col-gram
  // swap) nor MvNormal(mu, S) (treating the scale as covariance directly).
  const { built } = makeCtx(NONTRI_MODEL);
  const expanded = orchestrator.expandMeasure('Y',
    { derivations: built.derivations, bindings: built.bindings });
  const density = require('../density.ts');
  const densityPrims = require('../density-prims.ts');

  const colGram = (m: number[][]) => {        // Sᵀ·S
    const r = m.length, c = m[0].length, g: number[][] = [];
    for (let i = 0; i < c; i++) { g[i] = [];
      for (let j = 0; j < c; j++) { let s = 0;
        for (let k = 0; k < r; k++) s += m[k][i] * m[k][j]; g[i][j] = s; } }
    return g;
  };
  const cov = matGram(S_NONTRI);              // S·Sᵀ (the correct one)
  const wrongCol = colGram(S_NONTRI);         // Sᵀ·S
  // Sanity: the two grams genuinely differ (else the test proves nothing).
  assert.ok(Math.abs(cov[0][0] - wrongCol[0][0]) > 1e-6,
    'S·Sᵀ and Sᵀ·S must differ for this adversarial test to bite');

  const obs = toValue([0.0, 2.0]);            // x−mu=[-1,0]: asymmetric, breaks the S·Sᵀ vs Sᵀ·S tie
  const lp = density.logDensity(expanded, obs, {}, {});
  const right = densityPrims.MV_DENSITY_FNS.MvNormal(obs,
    { mu: toValue(MU_VEC), cov: toMatValue(cov) });
  const wrong = densityPrims.MV_DENSITY_FNS.MvNormal(obs,
    { mu: toValue(MU_VEC), cov: toMatValue(wrongCol) });
  assert.ok(Math.abs(lp - right) < 1e-9, `must equal S·Sᵀ density: got ${lp}, want ${right}`);
  assert.ok(Math.abs(lp - wrong) > 1e-6, `must NOT equal Sᵀ·S density (${wrong})`);
});

// =====================================================================
// 3. L2 — scale = 0 degenerate → clear diagnostic
// =====================================================================

test('locscale with literal zero scale (L2) yields a clear non-invertible diagnostic', () => {
  let err: any = null;
  try {
    makeCtx(`
      base = Normal(mu = 0, sigma = 1)
      y = locscale(base, 3.0, 0)
    `);
  } catch (e) {
    err = e;
  }
  assert.ok(err, 'zero scale must throw');
  const msg = String(err && err.message);
  assert.ok(/locscale/.test(msg) && /(non-zero|invertible)/.test(msg),
    `diagnostic must mention non-zero/invertible scale, got: ${msg}`);
});

// =====================================================================
// 4. VECTOR (per-component) scale → clear unsupported diagnostic
// =====================================================================

test('locscale with vector (per-component) scale yields a clear unsupported diagnostic', () => {
  let err: any = null;
  try {
    makeCtx(`
      base = MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))
      y = locscale(base, [1.0, 2.0], [2.0, 0.5])
    `);
  } catch (e) {
    err = e;
  }
  assert.ok(err, 'vector scale must throw a diagnostic rather than silently misbehave');
  const msg = String(err && err.message);
  assert.ok(/locscale/.test(msg) && /(vector|per-component)/.test(msg),
    `diagnostic must mention vector/per-component scale, got: ${msg}`);
});

// =====================================================================
// H1 — the non-finite-inverse density guard (density.ts H2(a)) fires
// =====================================================================
//
// A RUNTIME-zero scalar scale (`sub(1.0, 1.0)` = 0) is NOT a NumberLiteral,
// so `__isLiteralZero` misses it and it routes through the SCALAR path.
// Its closed-form inverse `(y - shift)/0` is ±Inf, which must trip
// density.walkPushfwd's non-finite guard and THROW, rather than recurse
// into a silently-garbage log-density.
test('H1: a runtime-zero scalar scale trips the non-finite-inverse density guard (throws, not garbage)', async () => {
  const { ctx } = makeCtx(`
    base = Normal(mu = 0, sigma = 1)
    Y = locscale(base, 3.0, sub(1.0, 1.0))
    lp = logdensityof(Y, 5.0)
  `);
  await assert.rejects(
    async () => { await ctx.getMeasure('lp'); },
    /non-finite/,
    'a non-finite closed-form inverse must throw the guard error, not return a finite-but-garbage density');
});

// =====================================================================
// H2 — NAMED scale bindings (the documented workaround) — Identifier
// branch of __discoveredScaleRank
// =====================================================================

test('H2: a named matrix-scale binding (rank-2) routes through the affine registry and density is correct', () => {
  // S = lower_cholesky(COV) ⇒ S·Sᵀ = COV, so Y ≡ MvNormal(MU_VEC, COV).
  const { built } = makeCtx(`
    base = MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))
    S = lower_cholesky(rowstack([[${COV[0][0]}, ${COV[0][1]}], [${COV[1][0]}, ${COV[1][1]}]]))
    Y = locscale(base, [${MU_VEC[0]}, ${MU_VEC[1]}], S)
  `);
  const expanded = orchestrator.expandMeasure('Y',
    { derivations: built.derivations, bindings: built.bindings });
  assert.equal(expanded.bijection.registryName, 'affine',
    'a NAMED matrix scale must route to the affine registry, not the scalar path');

  const density = require('../density.ts');
  const densityPrims = require('../density-prims.ts');
  // The bijection's paramIRs reference the NAMED binding `S`, so the density
  // entry point needs its value in the env — sourced from the precomputed
  // fixedValues (cf. the inline matrix tests, which are self-contained).
  const env: any = {};
  for (const [k, v] of built.fixedValues) env[k] = v;
  for (const obsArr of [[0.5, 1.5], [1.0, 2.0]]) {
    const obs = toValue(obsArr);
    const lp = density.logDensity(expanded, obs, env, {});
    const expected = densityPrims.MV_DENSITY_FNS.MvNormal(obs,
      { mu: toValue(MU_VEC), cov: toMatValue(COV) });
    assert.ok(Number.isFinite(lp) && Math.abs(lp - expected) < 1e-9,
      `named matrix-scale density at ${obsArr}: got ${lp}, want ${expected}`);
  }
});

test('H2: a named vector-scale binding yields the unsupported diagnostic (same as inline)', () => {
  let err: any = null;
  try {
    makeCtx(`
      base = MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))
      s = [2.0, 0.5]
      Y = locscale(base, [1.0, 2.0], s)
    `);
  } catch (e) { err = e; }
  assert.ok(err, 'a named vector scale must throw, not fall through to the scalar path');
  const msg = String(err && err.message);
  assert.ok(/locscale/.test(msg) && /(vector|per-component)/.test(msg),
    `diagnostic must mention vector/per-component scale, got: ${msg}`);
});

// =====================================================================
// M6 — remaining __discoveredScaleRank classification arms
// =====================================================================

test('M6: a bare nested-array-literal scale ([[…],[…]]) classifies rank-2 → affine registry', () => {
  const { built } = makeCtx(`
    base = MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))
    Y = locscale(base, [1.0, 2.0], [[1.0, 0.0], [0.0, 1.0]])
  `);
  const expanded = orchestrator.expandMeasure('Y',
    { derivations: built.derivations, bindings: built.bindings });
  assert.equal(expanded.bijection.registryName, 'affine',
    'a bare [[…],[…]] literal scale must classify as a matrix (rank-2)');
});

test('M6: a rowstack of a FLAT array classifies rank-1 → vector diagnostic', () => {
  let err: any = null;
  try {
    makeCtx(`
      base = MvNormal(mu = [0.0, 0.0, 0.0], cov = rowstack([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
      Y = locscale(base, [0.0, 0.0, 0.0], rowstack([1.0, 2.0, 3.0]))
    `);
  } catch (e) { err = e; }
  assert.ok(err, 'rowstack of a flat array is a rank-1 (vector) scale and must throw');
  assert.ok(/(vector|per-component)/.test(String(err && err.message)),
    `expected a vector diagnostic, got: ${err && err.message}`);
});

// =====================================================================
// H3 — an off-list inline matrix-valued scale must FAIL LOUDLY
// =====================================================================
//
// `matprod(A, A)` is matrix-valued but is NOT in __discoveredScaleRank's
// MATRIX_OPS, so it falls through to the scalar path. It must NOT silently
// sample/score garbage — today it fails loudly (sampler/density rejects it).
// This test PINS that loud failure: if a future change ever lets this case
// return a value silently, this test fails and flags the regression.
test('H3: an unrecognised inline matrix-valued scale fails loudly (regression pin against silent sampling)', async () => {
  const { ctx } = makeCtx(`
    base = MvNormal(mu = [0.0, 0.0], cov = rowstack([[1.0, 0.0], [0.0, 1.0]]))
    A = rowstack([[1.0, 0.0], [0.5, 1.0]])
    Y = locscale(base, [1.0, 2.0], matprod(A, A))
    lp = logdensityof(Y, [0.5, 1.5])
  `);
  await assert.rejects(
    async () => { await ctx.getMeasure('lp'); },
    /.+/,
    'an off-list matrix scale must throw rather than silently return a density');
});
