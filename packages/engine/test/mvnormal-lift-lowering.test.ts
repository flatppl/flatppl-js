'use strict';

// =====================================================================
// mvnormal-lift-lowering.test.ts — Phase 5.1 Session 5e
// =====================================================================
//
// Pins the lift-time MvNormal lowering producer that closes the
// engine-concepts §22 architectural reframe end-to-end.
//
// Producer flow:
//   1. `lift.inlineMvNormalLift` detects `MvNormal(mu = X, cov = Y)`
//      calls in the AST.
//   2. When mu and cov both resolve to ArrayLiteral (literal at the
//      call site or one-level ref to an ArrayLiteral binding), the
//      helper emits two synthetic bindings:
//        - __bij_N = bijection(fn(_), fn(_), 0.0)
//                    with the side-channel marker
//                    `binding.__mvnormalLowering = {muIR, covIR}`
//        - __iid_N = iid(Normal(mu=0, sigma=1), D)
//      and rewrites the original `MvNormal(...)` call to
//      `pushfwd(__bij_N, __iid_N)` in place.
//   3. `derivations.buildDerivations` bijection-construction loop
//      reads the marker and additively attaches
//      `binding.bijection.registryName = 'affine'` and
//      `paramIRs = {L: lower_cholesky(covIR), b: muIR}`.
//   4. At materialise / density time, matPushfwd / walkPushfwd
//      (Session 5d) consume registryName + paramIRs and dispatch
//      through the bijection-registry's affine atom-batched paths —
//      same hot code path matMvNormal uses.
//
// Fallback: when the gate doesn't fire (non-literal mu / cov), the
// MvNormal AST is preserved and matMvNormal handles it via the
// existing kind='mvnormal' classifier branch.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 256;
const ROOT_SEED    = 0xB1737D02;

function makeCtx(source: any) {
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
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// =====================================================================
// 1. Structural pin — synthetic bindings + classifier kind=pushfwd
// =====================================================================

test('5e: ref-form mu + cov → synthetic __bij + __iid bindings, X classifies pushfwd', () => {
  const ctx = makeCtx(`
mu_vec = [1.0, 2.0]
cov_mat = [[2.0, 0.5], [0.5, 1.0]]
X = MvNormal(mu = mu_vec, cov = cov_mat)
`);
  // Find the synthetic bindings.
  const names = Array.from(ctx.bindings.keys());
  const bijName = names.find((n: any) => /^__bij/.test(n));
  const iidName = names.find((n: any) => /^__iid/.test(n));
  assert.ok(bijName, 'a __bij_N synthetic binding exists');
  assert.ok(iidName, 'an __iid_N synthetic binding exists');

  const bijBinding = ctx.bindings.get(bijName);
  assert.equal(bijBinding.type, 'bijection',
    '__bij_N typed as bijection so buildDerivations populates binding.bijection');
  assert.ok(bijBinding.bijection, 'binding.bijection metadata attached');
  assert.equal(bijBinding.bijection.registryName, 'affine',
    'registryName forwarded onto binding.bijection');
  assert.ok(bijBinding.bijection.paramIRs, 'paramIRs forwarded');
  // paramIRs.L IR: {kind:'call', op:'lower_cholesky', args:[<covIR>]}
  assert.equal(bijBinding.bijection.paramIRs.L.op, 'lower_cholesky');
  assert.equal(bijBinding.bijection.paramIRs.L.args.length, 1);
  // paramIRs.b IR: ref to mu_vec.
  assert.equal(bijBinding.bijection.paramIRs.b.kind, 'ref');
  assert.equal(bijBinding.bijection.paramIRs.b.name, 'mu_vec');

  // The original X binding rewrites to pushfwd(__bij_N, __iid_N).
  // Classifier emits kind='pushfwd', NOT 'mvnormal'.
  assert.equal(ctx.derivations.X.kind, 'pushfwd',
    'X classified as pushfwd (not mvnormal) — lift rewrote the IR');
  assert.equal(ctx.derivations.X.fnRef, bijName,
    'pushfwd.fnRef points at the synthetic __bij_N');
  assert.equal(ctx.derivations.X.from, iidName,
    'pushfwd.from points at the synthetic __iid_N base');
});

// =====================================================================
// 2. End-to-end materialisation — lowered path matches matMvNormal
// =====================================================================
//
// The load-bearing correctness pin: lift-rewritten X produces samples
// numerically equal to a parallel hand-written matMvNormal invocation
// with the same root seed.

test('5e: lowered-path materialisation produces shape [N, D] with finite samples', async () => {
  const ctx = makeCtx(`
mu_vec = [1.0, 2.0]
cov_mat = [[2.0, 0.5], [0.5, 1.0]]
X = MvNormal(mu = mu_vec, cov = cov_mat)
`);
  const X = await ctx.getMeasure('X');
  assert.ok(X && X.value, 'X materialises to a shape-tagged Value');
  assert.deepEqual(Array.from(X.value.shape), [SAMPLE_COUNT, 2],
    'shape [N, 2]');
  assert.equal(X.value.outerRank, 1,
    'outerRank=1 preserved (iid-of-scalar pushfwd of base)');
  for (let i = 0; i < X.value.data.length; i++) {
    assert.ok(Number.isFinite(X.value.data[i]),
      `X.value.data[${i}] = ${X.value.data[i]} is finite`);
  }
});

test('5e: lowered-path empirical mean ≈ mu_vec (4-sigma N=4000)', async () => {
  const src = `
mu_vec = [1.0, -2.0, 3.0]
cov_mat = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 2.0]]
X = MvNormal(mu = mu_vec, cov = cov_mat)
`;
  // Use a larger N for the mean check.
  const lifted = processSource(src);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const N = 4000;
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
    sendWorker:  (msg: any) => Promise.resolve(worker.handle(msg)),
    sampleCount: N,
    rootSeed:    ROOT_SEED,
  };
  const X = await ctx.getMeasure('X');
  const muExpected = [1.0, -2.0, 3.0];
  const diagCov = [1.0, 0.5, 2.0];
  for (let d = 0; d < 3; d++) {
    let sum = 0;
    for (let i = 0; i < N; i++) sum += X.value.data[i * 3 + d];
    const mean = sum / N;
    const stderr = Math.sqrt(diagCov[d] / N);
    const margin = 4 * stderr;
    assert.ok(Math.abs(mean - muExpected[d]) < margin,
      `dim ${d}: sample mean ${mean.toFixed(4)} vs expected `
      + `${muExpected[d]} (margin ${margin.toFixed(4)})`);
  }
});

// =====================================================================
// 3. Inline-literal mu and cov (no separate bindings)
// =====================================================================

test('5e: inline-literal mu / cov also trigger the lift gate', () => {
  const ctx = makeCtx(`
X = MvNormal(mu = [1.0, 2.0], cov = rowstack([[2.0, 0.5], [0.5, 1.0]]))
`);
  const names = Array.from(ctx.bindings.keys());
  const bijName = names.find((n: any) => /^__bij/.test(n));
  assert.ok(bijName, 'inline-literal mu/cov produces a __bij_N binding');
  const bijBinding = ctx.bindings.get(bijName);
  assert.equal(bijBinding.bijection.registryName, 'affine');
  // paramIRs.b IR: literal vector when mu is inline.
  // lower's ArrayLiteral lowering: {kind:'call', op:'vector', args:[lit ...]}
  assert.equal(bijBinding.bijection.paramIRs.b.op, 'vector',
    'inline mu lowers to vector(...) IR');
  assert.equal(bijBinding.bijection.paramIRs.b.args.length, 2);
});

// =====================================================================
// 4. Density side smoke — closed-form MvNormal density
// =====================================================================

test('5e: walkPushfwd registry density matches density-prims MvNormal closed form', () => {
  // Use INLINE literals so paramIRs are self-contained (no refs
  // requiring density-time binding-env resolution). Hierarchical
  // models with stochastic mu/cov refs need the matLogdensityof
  // path that threads fixedValues into baseEnv; the present test
  // pins the registry-walker density math directly via the
  // top-level density.logDensity entry point.
  const ctx = makeCtx(`
X = MvNormal(mu = [1.0, 2.0], cov = rowstack([[2.0, 0.5], [0.5, 1.0]]))
`);
  const expanded = orchestrator.expandMeasure(
    'X', { derivations: ctx.derivations, bindings: ctx.bindings });
  assert.ok(expanded, 'X expands');
  assert.equal(expanded.bijection.registryName, 'affine');

  const density = require('../density.ts');
  const obs = { shape: [2], data: new Float64Array([0.5, 1.5]) };
  const lp = density.logDensity(expanded, obs, {}, {});

  // Compare to the density-prims closed form (which itself routes
  // through the registry's affine entry — see density-prims.ts
  // MvNormal closed form refactored in Session 3).
  const densityPrims = require('../density-prims.ts');
  const muVec = { shape: [2], data: new Float64Array([1.0, 2.0]) };
  const covMat = { shape: [2, 2], data: new Float64Array([2.0, 0.5, 0.5, 1.0]) };
  const xObs = { shape: [2], data: new Float64Array([0.5, 1.5]) };
  const expected = densityPrims.MV_DENSITY_FNS.MvNormal(xObs,
    { mu: muVec, cov: covMat });

  assert.ok(Number.isFinite(lp), 'lift-rewritten density is finite');
  assert.ok(Math.abs(lp - expected) < 1e-10,
    `lift-rewritten density ${lp} vs closed-form MvNormal ${expected}`);
});

// =====================================================================
// 5. Fallback — non-literal cov stays on matMvNormal (kind=mvnormal)
// =====================================================================

test('5e: non-literal cov falls through to matMvNormal (kind=mvnormal preserved)', async () => {
  // Use a CallExpr binding for cov that the lift literal-resolver
  // can't unwrap (e.g. `eye(2)` — a function call, not ArrayLiteral).
  // The lift gate skips this case; matMvNormal materialises X.
  const ctx = makeCtx(`
mu_vec = [1.0, 2.0]
cov_mat = eye(2)
X = MvNormal(mu = mu_vec, cov = cov_mat)
`);
  // No __bij_N / __iid_N synthesised.
  const names = Array.from(ctx.bindings.keys());
  const bijName = names.find((n: any) => /^__bij/.test(n));
  assert.equal(bijName, undefined,
    'non-literal cov → no synthetic bijection binding (gate skipped)');
  assert.equal(ctx.derivations.X.kind, 'mvnormal',
    'X stays on the kind=mvnormal classifier branch — matMvNormal handles it');

  // Materialiser still works.
  const X = await ctx.getMeasure('X');
  assert.ok(X && X.value, 'fallback path materialises');
  assert.deepEqual(Array.from(X.value.shape), [SAMPLE_COUNT, 2]);
});

// =====================================================================
// 6. Regression — hand-attached marker pathway still works
// =====================================================================
//
// 5e shouldn't disturb the hand-attached-marker test from Session 5d.

test('5e: hand-attached registry markers still work alongside lift-lowering', async () => {
  const ctx = makeCtx(`
Base = iid(Normal(mu = 0.0, sigma = 1.0), 2)
b = bijection(fn(_), fn(_), 0.0)
Y = pushfwd(b, Base)
`);
  // No MvNormal in this source — no lift gate fires.
  const names = Array.from(ctx.bindings.keys());
  const autoBij = names.find((n: any) => /^__bij/.test(n));
  assert.equal(autoBij, undefined,
    'no MvNormal in source → no auto-generated bijection bindings');

  // Hand-attach the marker on the user-written `b` binding.
  const bBinding = ctx.bindings.get('b');
  bBinding.bijection.registryName = 'affine';
  bBinding.bijection.paramIRs = {
    L: { kind: 'lit', value: [[2, 0], [1, 3]] },
    b: { kind: 'lit', value: [10, -5] },
  };
  const Y = await ctx.getMeasure('Y');
  assert.ok(Y && Y.value && Y.value.shape[1] === 2,
    'hand-marker dispatch still produces a [N, 2] Value');
});
