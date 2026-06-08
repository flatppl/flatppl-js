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
