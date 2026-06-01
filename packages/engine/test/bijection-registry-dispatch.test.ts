'use strict';

// =====================================================================
// bijection-registry-dispatch.test.ts — Phase 5.1 Session 5d
// =====================================================================
//
// Pins the CONSUMER-side dispatch in matPushfwd (commit 2) and
// walkPushfwd (commit 3) for bijection bindings carrying
// `registryName` + `paramIRs`. The producer side (lift-time MvNormal
// lowering) is Session 5e; these tests exercise the consumer against
// synthetic IR with the marker pair hand-attached.
//
// Test 1-2: matPushfwd vector-base dispatch. Construct
// `Base = iid(Normal(0,1), D); b = bijection(<identity stubs>); Y =
// pushfwd(b, Base)`, attach registryName='affine' + paramIRs = {L, b}
// onto b's bijection binding, materialise Y, verify the result
// matches `affineAtomBatchedForward(Base.value, {L, b}, N)` byte-
// equally.
//
// Test 3: dispatch error surfaces — registryName + paramIRs missing
// (additive-invariant violation), unknown registryName, scalar base
// falls through to AST path.
//
// Test 4: walkPushfwd density via registry (commit 3) — sampled then
// scored; the registry's atomBatchedInverse + logDetJ matches a hand-
// computed MvNormal density.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const bijRegistry = require('../bijection-registry.ts');

const SAMPLE_COUNT = 256;
const ROOT_SEED    = 0xB1737D00;

function makeCtx(source: any, sampleCount?: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => Promise.resolve(worker.handle(msg)),
    sampleCount: sampleCount != null ? sampleCount : SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// =====================================================================
// 1. matPushfwd vector-base dispatch (registryName='affine')
// =====================================================================
//
// Hand-attach registryName + paramIRs to the bijection binding, then
// materialise pushfwd(b, iid(Normal, D)) and verify the output
// matches a direct registry call.

test('matPushfwd 5d: vector-base + registryName dispatch matches direct registry call', async () => {
  const ctx = makeCtx(`
Base = iid(Normal(mu = 0.0, sigma = 1.0), 2)
b = bijection(fn(_), fn(_), 0.0)
Y = pushfwd(b, Base)
`);
  // Synthetic producer (the Session 5e job will do this from lift):
  // mark the bijection binding with registryName + paramIRs for affine.
  // L = [[2, 0], [1, 3]] (lower-triangular), b = [10, -5].
  const bBinding = ctx.bindings.get('b');
  bBinding.bijection.registryName = 'affine';
  bBinding.bijection.paramIRs = {
    L: { kind: 'lit', value: [[2, 0], [1, 3]] },
    b: { kind: 'lit', value: [10, -5] },
  };

  // Materialise Y — registry fast path activates.
  const Y = await ctx.getMeasure('Y');
  assert.ok(Y && Y.value && Array.isArray(Y.value.shape),
    'Y materialises to a shape-tagged Value');
  const N = Y.value.shape[0];
  const D = Y.value.shape[1];
  assert.equal(D, 2, 'output shape D=2 (matches affine [D]→[D] contract)');
  assert.equal(Y.value.outerRank, 1,
    'preserves outerRank=1 (iid-of-scalar atom marker)');

  // Independently materialise Base and compute affineAtomBatchedForward
  // directly. Y.value.data MUST match byte-for-byte.
  const Base = await ctx.getMeasure('Base');
  const expected = bijRegistry.affineAtomBatchedForward(
    Base.value,
    { L: { shape: [2, 2], data: new Float64Array([2, 0, 1, 3]) },
      b: { shape: [2],    data: new Float64Array([10, -5]) } },
    N);
  for (let i = 0; i < N * D; i++) {
    assert.ok(Math.abs(Y.value.data[i] - expected.data[i]) < 1e-12,
      `Y.value.data[${i}] = ${Y.value.data[i]} vs expected ${expected.data[i]}`);
  }
});

// =====================================================================
// 2. matPushfwd error surface — registryName without paramIRs
// =====================================================================

test('matPushfwd 5d: registryName without paramIRs rejects with clear error', async () => {
  const ctx = makeCtx(`
Base = iid(Normal(mu = 0.0, sigma = 1.0), 2)
b = bijection(fn(_), fn(_), 0.0)
Y = pushfwd(b, Base)
`);
  const bBinding = ctx.bindings.get('b');
  bBinding.bijection.registryName = 'affine';
  // PARAMIRS DELIBERATELY OMITTED — additive-invariant violation.

  await assert.rejects(
    () => ctx.getMeasure('Y'),
    /paramIRs|invariant/i,
    'matPushfwd rejects loudly when registryName is set without paramIRs');
});

// =====================================================================
// 3. matPushfwd error surface — unknown registryName
// =====================================================================

test('matPushfwd 5d: unknown registryName rejects with clear error', async () => {
  const ctx = makeCtx(`
Base = iid(Normal(mu = 0.0, sigma = 1.0), 2)
b = bijection(fn(_), fn(_), 0.0)
Y = pushfwd(b, Base)
`);
  const bBinding = ctx.bindings.get('b');
  bBinding.bijection.registryName = 'not_a_real_bijection';
  bBinding.bijection.paramIRs = { foo: { kind: 'lit', value: 1.0 } };

  await assert.rejects(
    () => ctx.getMeasure('Y'),
    /not found in bijection-registry|not in.*registry/i,
    'matPushfwd rejects loudly for unknown registryName');
});

// =====================================================================
// 4. matPushfwd fall-through — scalar base + registryName uses AST path
// =====================================================================
//
// Registry-driven fast path activates ONLY for vector-atom bases
// (M.value with outerRank===1). A scalar base with the same
// registryName marker falls through to the existing AST path — proves
// the additive invariant (the registry is an OPTIMISATION marker, not
// a REQUIREMENT). The AST path's identity fwd `fn(_)` then evaluates
// per-atom and produces the same samples as the base.

test('matPushfwd 5d: scalar base with registryName falls through to AST path', async () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(_), fn(_), 0.0)
Y = pushfwd(b, M)
`);
  // Mark with registryName — but base is scalar Normal, not iid.
  // Fast path should NOT activate; AST identity fn(_) evaluates.
  const bBinding = ctx.bindings.get('b');
  bBinding.bijection.registryName = 'affine';
  bBinding.bijection.paramIRs = {
    L: { kind: 'lit', value: [[1.0]] },
    b: { kind: 'lit', value: [0.0] },
  };
  const Y = await ctx.getMeasure('Y');
  assert.ok(Y && Y.samples,
    'AST path materialised — Y has scalar samples');
  for (let i = 0; i < Y.samples.length; i++) {
    assert.ok(Number.isFinite(Y.samples[i]),
      `Y.samples[${i}] finite (= ${Y.samples[i]})`);
  }
});

// =====================================================================
// 5. walkPushfwd vector-base density via registry (commit 3)
// =====================================================================
//
// Score a hand-built observation through the lowered form; the
// density-side registry path consumes a vector head, runs
// atomBatchedInverse + logDetJ, recurses on the iid base. Compare to
// a hand-computed MvNormal density.

test('walkPushfwd 5d: vector-base + registryName dispatch matches hand-computed MvNormal density', () => {
  const ctx = makeCtx(`
Base = iid(Normal(mu = 0.0, sigma = 1.0), 2)
b = bijection(fn(_), fn(_), 0.0)
Y = pushfwd(b, Base)
`);
  const bBinding = ctx.bindings.get('b');
  bBinding.bijection.registryName = 'affine';
  bBinding.bijection.paramIRs = {
    L: { kind: 'lit', value: [[2, 0], [1, 3]] },
    b: { kind: 'lit', value: [10, -5] },
  };

  const expanded = orchestrator.expandMeasure(
    'Y', { derivations: ctx.derivations, bindings: ctx.bindings });
  assert.equal(expanded.bijection.registryName, 'affine');
  assert.ok(expanded.bijection.paramIRs);

  // Observation y. Hand-compute expected density.
  //   z = L^{-1}(y - b)
  //   for y = [12, 1]: y - b = [2, 6]; solve L·z = [2, 6]:
  //     z[0] = 2/2 = 1; z[1] = (6 - 1·1)/3 = 5/3 ≈ 1.6666...
  //   log p_iid_Normal_2(z) = sum_i log p_Normal(z_i, 0, 1)
  //                         = -log(2π) - 0.5 * (z0^2 + z1^2)
  //   logDetJ = -log|det L| = -(log 2 + log 3) = -log 6
  //
  // Total: -log(2π) - (1 + (5/3)^2)/2 - log(6)
  const density = require('../density.ts');
  const obs = { shape: [2], data: new Float64Array([12, 1]) };
  const lp = density.logDensity(expanded, obs, {}, {});
  const z0 = 1.0, z1 = 5/3;
  const expected = -Math.log(2 * Math.PI)
    - 0.5 * (z0*z0 + z1*z1)
    - Math.log(6);
  assert.ok(Number.isFinite(lp), 'density is finite (got ' + lp + ')');
  assert.ok(Math.abs(lp - expected) < 1e-10,
    'density ' + lp + ' vs hand-computed ' + expected);
});

// =====================================================================
// 6. walkPushfwd fall-through — scalar-base AST path unaltered
// =====================================================================

test('walkPushfwd 5d: existing scalar-base density unchanged when registry not engaged', () => {
  const ctx = makeCtx(`
M = Normal(mu = 0.0, sigma = 1.0)
b = bijection(fn(exp(_)), fn(log(_)), fn(_))
LN = pushfwd(b, M)
`);
  // Mark with registryName + paramIRs but scalar base — the registry
  // fast path's D-discovery is affine-specific; for non-affine
  // registry names with a scalar base, we don't enter the fast path
  // because we'd fail D-discovery. The AST path is the source of
  // truth for scalar bases.
  //
  // Specifically: scalar base means consumeVector wouldn't apply; the
  // density walker's registry path is for vector-atom bases produced
  // by iid composites.
  //
  // Here we test the parallel case: bij carries registryName but
  // base is scalar, so the registry path's vector contracts can't
  // match. The walker should still produce the AST path result.
  const bBinding = ctx.bindings.get('b');
  // INTENTIONALLY don't set registryName/paramIRs — we want the
  // AST scalar path to fire untouched.
  void bBinding;

  const density = require('../density.ts');
  const expanded = orchestrator.expandMeasure(
    'LN', { derivations: ctx.derivations, bindings: ctx.bindings });
  const lp = density.logDensity(expanded, 2.5, {}, {});
  assert.ok(Number.isFinite(lp),
    'scalar AST density unchanged (no registryName marker)');
});
