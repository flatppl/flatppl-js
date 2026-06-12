'use strict';

// Matrix/vector locscale (P3): locscale(iid(Normal,D), shift, scale) is the
// multivariate affine pushforward scale@x + shift, routed by inferred shape
// to the bijection-registry 'affine' path (the same entry MvNormal rides).
//
// Oracle (INDEPENDENT — scipy.stats.multivariate_normal, not the engine):
//   locscale(iid(Normal,2), [1,-1], [[2,0],[0,1.5]]) == MvNormal([1,-1], [[4,0],[0,2.25]])
//   logpdf([1.5,-0.5]) = -3.0232949106330103
//   logpdf([1.0,-1.0]) = -2.936489355077455

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const ROOT_SEED = 0x10C5CA1E;

function makeCtx(source: any, opts?: any) {
  opts = opts || {};
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [], 'unexpected diagnostics: ' + JSON.stringify(errs));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx); cache.set(name, p); return p;
    },
    sendWorker: (msg: any) => {
      const r = worker.handle(msg);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
    sampleCount: opts.sampleCount != null ? opts.sampleCount : 256,
    rootSeed: ROOT_SEED,
  };
  return ctx;
}

// The iid base is written with the distribution CALL form `Normal(0,1)` —
// the engine's iid contract (classifyIid's resolveMeasureBaseName) and every
// flatppl-examples idiom use `iid(Normal(...), D)`, not a bare `Normal`
// symbol. The lift hoists the inner Normal call to its own anon ref so the
// base scores as iid(<ref-to-scalar-dist>, D), exactly as MvNormal's
// synthetic base does.
const MAT_SRC =
  `L = [[2.0, 0.0], [0.0, 1.5]]\n`
  + `mu = [1.0, -1.0]\n`
  + `X = locscale(iid(Normal(0.0, 1.0), 2), mu, L)\n`;

test('matrix locscale classifies as a registry-affine pushfwd', () => {
  const ctx = makeCtx(MAT_SRC + `lp = logdensityof(X, [1.5, -0.5])\n`);
  // X rewrites to pushfwd(__bij_N, <iid base>). The registry-affine metadata
  // lives on the synthetic bijection BINDING (mirrors the MvNormal lowering
  // test — DerivationPushfwd carries fnRef/from, not the bijection meta,
  // which density-side dispatch reads via resolveBijection from the binding).
  const d = ctx.derivations['X'];
  assert.equal(d.kind, 'pushfwd', `X kind = ${d && d.kind}`);
  const bijName = Array.from(ctx.bindings.keys()).find((n: any) => /^__bij/.test(n));
  assert.ok(bijName, 'a __bij_N synthetic bijection binding exists');
  const bij = ctx.bindings.get(bijName).bijection;
  assert.ok(bij, 'binding.bijection metadata attached');
  assert.equal(bij.registryName, 'affine');
  assert.ok(bij.paramIRs && bij.paramIRs.L && bij.paramIRs.b,
    'paramIRs {L,b} present');
});

test('matrix locscale density matches the MvNormal scipy oracle', async () => {
  for (const [pt, expected] of [
    ['[1.5, -0.5]', -3.0232949106330103],
    ['[1.0, -1.0]', -2.936489355077455],
  ] as any[]) {
    const ctx = makeCtx(MAT_SRC + `lp = logdensityof(X, ${pt})\n`);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-9,
      `pt=${pt}: got ${lp.samples[0]}, expected ${expected}`);
  }
});

test('matrix locscale with named lower_cholesky scale routes the same way', () => {
  // scale given as a named matrix expression (the spec's MvNormal idiom).
  // The matrix literal is wrapped in rowstack(...) per spec §03 (commits the
  // row-major storage order so lower_cholesky reads a 2d array, not a
  // vector-of-vectors); this also makes Lc's inferredType a rank-2 [D,D],
  // exercising the lift gate's ref-inferredType square-confirm branch.
  const ctx = makeCtx(
    `cov = rowstack([[4.0, 0.0], [0.0, 2.25]])\n`
    + `Lc = lower_cholesky(cov)\n`
    + `mu = [1.0, -1.0]\n`
    + `X = locscale(iid(Normal(0.0, 1.0), 2), mu, Lc)\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  const d = ctx.derivations['X'];
  assert.equal(d.kind, 'pushfwd', `X kind = ${d && d.kind}`);
  const bijName = Array.from(ctx.bindings.keys()).find((n: any) => /^__bij/.test(n));
  assert.ok(bijName, 'a __bij_N synthetic bijection binding exists');
  assert.equal(ctx.bindings.get(bijName).bijection.registryName, 'affine');
});

test('matrix locscale over an MvNormal base gives a clean diagnostic (deferred)', () => {
  const src =
    `mu = [0.0, 0.0]\n`
    + `cov = [[1.0, 0.0], [0.0, 1.0]]\n`
    + `L = [[2.0, 0.0], [0.0, 1.5]]\n`
    + `shift = [1.0, -1.0]\n`
    + `X = locscale(MvNormal(mu = mu, cov = cov), shift, L)\n`;
  // The deferred composition case must surface a clear diagnostic somewhere
  // in the pipeline (analyzer diagnostics OR a buildDerivations throw with a
  // locscale-tagged message), NOT a cryptic registry shape error.
  let msg = '';
  try {
    const lifted2 = processSource(src);
    const errs = lifted2.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length) msg = errs.map((e: any) => e.message).join(' | ');
    else { orchestrator.buildDerivations(lifted2.bindings); }
  } catch (e: any) { msg = e.message; }
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged diagnostic, got: ${msg || '(none)'}`);
  assert.match(msg, /pushfwd|iid|base/i,
    `expected guidance to use iid base / pushfwd directly, got: ${msg}`);
});

test('vector-shift + scalar-scale locscale errors cleanly (no [D,D] matrix scale)', () => {
  // A vector shift forces the call to survive the scalar-expansion pre-pass,
  // but a SCALAR scale (not a [D,D] matrix) cannot lower to an affine-registry
  // pushfwd. This must surface a clean locscale-tagged error (analyzer shape
  // diagnostic OR the buildDerivations safety net), NOT a cryptic downstream
  // failure or a silently-dropped binding.
  const src =
    `X = locscale(iid(Normal(0.0, 1.0), 2), [0.0, 0.0], 2.0)\n`;
  let msg = '';
  try {
    const lifted2 = processSource(src);
    const errs = lifted2.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length) msg = errs.map((e: any) => e.message).join(' | ');
    else { orchestrator.buildDerivations(lifted2.bindings); }
  } catch (e: any) { msg = e.message; }
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged error, got: ${msg || '(none)'}`);
});

test('square transpose scale still routes to the affine registry', () => {
  // After Fix 1 (transpose removed from the lift gate's square-op set), a
  // SQUARE transpose still routes via the normal concrete-[D,D] inferredType
  // path — its result type IS a static [2,2] so square-confirm passes. The
  // matrix is wrapped in rowstack(...) per spec §03 so it is a rank-2 matrix
  // (a bare [[...]] literal is a vector-of-vectors, whose transpose is a
  // transposed-vector, not a square matrix).
  const ctx = makeCtx(
    `M = rowstack([[2.0, 0.0], [0.0, 1.5]])\n`
    + `Lt = transpose(M)\n`
    + `X = locscale(iid(Normal(0.0, 1.0), 2), [0.0, 0.0], Lt)\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  const d = ctx.derivations['X'];
  assert.equal(d.kind, 'pushfwd', `X kind = ${d && d.kind}`);
  const bijName = Array.from(ctx.bindings.keys()).find((n: any) => /^__bij/.test(n));
  assert.ok(bijName, 'a __bij_N synthetic bijection binding exists');
  assert.equal(ctx.bindings.get(bijName).bijection.registryName, 'affine');
});

test('iid base dim mismatched with scale dim errors cleanly (not "value exhausted")', () => {
  // Base D=3, scale/shift D=2. The affine map scale@x+shift needs a length-D
  // base; a mismatch must surface a clean locscale-tagged error (analyzer
  // dimension diagnostic OR the buildDerivations safety net), NOT the cryptic
  // density-time "value exhausted" / undefined-length throw.
  const src =
    `B = locscale(iid(Normal(0.0,1.0), 3), [0.0, 0.0], [[1.0,0.0],[0.0,1.0]])\n`
    + `lp = logdensityof(B, [1.0,1.0,1.0])\n`;
  let msg = '';
  try {
    const lifted2 = processSource(src);
    const errs = lifted2.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length) msg = errs.map((e: any) => e.message).join(' | ');
    else {
      const built = orchestrator.buildDerivations(lifted2.bindings);
      const worker = createWorkerHandler();
      worker.handle({ type: 'init', seed: ROOT_SEED });
      const cache = new Map();
      const ctx2: any = {
        derivations: built.derivations, bindings: built.bindings,
        fixedValues: built.fixedValues || new Map(),
        getMeasure: (name: any) => {
          if (cache.has(name)) return cache.get(name);
          const p = materialiser.materialiseMeasure(name, ctx2); cache.set(name, p); return p;
        },
        sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
        sampleCount: 4, rootSeed: ROOT_SEED,
      };
      ctx2.getMeasure('lp');
    }
  } catch (e: any) { msg = e.message; }
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged error, got: ${msg || '(none)'}`);
  assert.doesNotMatch(msg, /value exhausted/i,
    `expected a clean diagnostic, not the cryptic density throw: ${msg}`);
});

// ── Named-reference resolution branches (analyzer pre-pass + lift gate) ──
// The analyzer pre-pass (baseIsIid / baseIidCount) and lift gate
// (baseIidCount, isSquareRefType, isSquareOpRef) each carry a 1-level
// Identifier-resolution branch that the inline-literal tests above don't
// reach. These tests bind the base / scale to NAMED refs so the resolver
// must follow the Identifier → AssignStatement edge.

// 1. NAMED iid base ref. `B = iid(Normal(0,1), 2)` then locscale(B, ...).
//    Exercises the analyzer baseIsIid / baseIidCount Identifier-ref branch
//    (the base arg is an Identifier, resolved 1-level to the iid call) and
//    lift's baseIidCount Identifier-ref branch.
test('named iid base ref: density matches the MvNormal scipy oracle', async () => {
  const ctx = makeCtx(
    `B = iid(Normal(0.0, 1.0), 2)\n`
    + `X = locscale(B, [1.0, -1.0], [[2.0, 0.0], [0.0, 1.5]])\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-3.0232949106330103)) < 1e-9,
    `got ${lp.samples[0]}, expected -3.0232949106330103`);
});

// 2. NAMED matrix-literal scale ref. `Lm = [[...]]` then scale = Lm.
//    Exercises the analyzer resolveLit Identifier-ref branch (litRows follows
//    the ref to the ArrayLiteral) and lift's named-matrix isSquareRefType
//    branch (Lm's inferredType is a static rank-2 [2,2]).
test('named matrix-literal scale ref: density matches the MvNormal scipy oracle', async () => {
  const ctx = makeCtx(
    `Lm = [[2.0, 0.0], [0.0, 1.5]]\n`
    + `X = locscale(iid(Normal(0.0, 1.0), 2), [1.0, -1.0], Lm)\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-3.0232949106330103)) < 1e-9,
    `got ${lp.samples[0]}, expected -3.0232949106330103`);
});

// 3. NON-SQUARE matrix scale over an iid base → clean locscale diagnostic.
//    Exercises the analyzer non-square diagnostic block (a [2,3] literal scale
//    fails the squareMatrix check). Must NOT surface the cryptic density
//    "value exhausted" throw.
test('non-square matrix scale over iid base errors cleanly (not "value exhausted")', () => {
  const src =
    `X = locscale(iid(Normal(0.0,1.0), 2), [0.0, 0.0], [[1.0,2.0,3.0],[4.0,5.0,6.0]])\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`;
  let msg = '';
  try {
    const lifted2 = processSource(src);
    const errs = lifted2.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length) msg = errs.map((e: any) => e.message).join(' | ');
    else {
      const built = orchestrator.buildDerivations(lifted2.bindings);
      const worker = createWorkerHandler();
      worker.handle({ type: 'init', seed: ROOT_SEED });
      const cache = new Map();
      const ctx2: any = {
        derivations: built.derivations, bindings: built.bindings,
        fixedValues: built.fixedValues || new Map(),
        getMeasure: (name: any) => {
          if (cache.has(name)) return cache.get(name);
          const p = materialiser.materialiseMeasure(name, ctx2); cache.set(name, p); return p;
        },
        sendWorker: (m: any) => Promise.resolve(worker.handle(m)),
        sampleCount: 4, rootSeed: ROOT_SEED,
      };
      ctx2.getMeasure('lp');
    }
  } catch (e: any) { msg = e.message; }
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged error, got: ${msg || '(none)'}`);
  assert.doesNotMatch(msg, /value exhausted/i,
    `expected a clean diagnostic, not the cryptic density throw: ${msg}`);
});

// 4. NAMED base ref with DIM MISMATCH. `B = iid(Normal(0,1), 3)` (D=3) but a
//    [2,2] scale (D=2). Exercises the named-base dimension-reconcile path:
//    baseIidCount resolves B's Identifier to its iid call (K=3), then the
//    K !== D diagnostic fires.
test('named iid base ref with dim mismatch errors cleanly', () => {
  const src =
    `B = iid(Normal(0.0, 1.0), 3)\n`
    + `X = locscale(B, [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])\n`;
  let msg = '';
  try {
    const lifted2 = processSource(src);
    const errs = lifted2.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length) msg = errs.map((e: any) => e.message).join(' | ');
    else { orchestrator.buildDerivations(lifted2.bindings); }
  } catch (e: any) { msg = e.message; }
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged error, got: ${msg || '(none)'}`);
});

// 5. NAMED lower_cholesky scale ref. `Lc = lower_cholesky(cov)` is a
//    shape-preserving square-op ref with a %dynamic inferredType, so lift
//    admits it via isSquareOpRef (NOT isSquareRefType). cov factors so that
//    Lc = [[2,0],[0,1.5]], reproducing the shared MvNormal oracle.
test('named lower_cholesky scale ref: density matches the MvNormal scipy oracle', async () => {
  const ctx = makeCtx(
    `cov = rowstack([[4.0, 0.0], [0.0, 2.25]])\n`
    + `Lc = lower_cholesky(cov)\n`
    + `X = locscale(iid(Normal(0.0, 1.0), 2), [1.0, -1.0], Lc)\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-3.0232949106330103)) < 1e-9,
    `got ${lp.samples[0]}, expected -3.0232949106330103`);
});

// 6. DYNAMIC iid base count. `iid(Normal(0,1), n)` with `n` a NAMED ref
//    (not a NumberLiteral) makes both the analyzer pre-pass baseIidCount and
//    lift's baseIidCount return null (count not statically integer), so the
//    K !== D cross-check is skipped and the call routes on the scale shape
//    alone. Exercises analyzer baseIidCount's non-static-size `return null`
//    and lift baseIidCount's non-static-size `return null`. Still scores the
//    same MvNormal oracle (n resolves to 2 at materialisation).
test('dynamic iid base count routes and matches the MvNormal scipy oracle', async () => {
  const ctx = makeCtx(
    `n = 2\n`
    + `X = locscale(iid(Normal(0.0, 1.0), n), [1.0, -1.0], [[2.0, 0.0], [0.0, 1.5]])\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-3.0232949106330103)) < 1e-9,
    `got ${lp.samples[0]}, expected -3.0232949106330103`);
});

// 6b. SAMPLE-bound (`~`) vector shift. The analyzer's 1-level resolvers only
//     follow `=` (AssignStatement) edges; a `~`-bound name has no
//     AssignStatement, so looksNonScalar(shift) and resolveLit(shift) each run
//     their Identifier loop to completion and fall through (analyzer 2549,
//     2608). The pre-pass leaves the call to lift (no diagnostic). lift can't
//     statically pin the latent shift's dimension, so it does NOT route and
//     the unrouted locscale reaches the buildDerivations safety net, which
//     throws the clean locscale-tagged error (not a cryptic downstream crash).
test('sample-bound vector shift: clean pre-pass + safety-net error (resolver fall-through)', () => {
  const lifted = processSource(
    `mu ~ MvNormal(mean = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]])\n`
    + `X = locscale(iid(Normal(0.0, 1.0), 2), mu, [[2.0, 0.0], [0.0, 1.5]])\n`);
  const preErrs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(preErrs, [],
    `analyzer pre-pass should not diagnose; got: ${JSON.stringify(preErrs.map((e: any) => e.message))}`);
  let msg = '';
  try { orchestrator.buildDerivations(lifted.bindings); }
  catch (e: any) { msg = e.message; }
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged safety-net error, got: ${msg || '(none)'}`);
});

// 6c. SAMPLE-bound (`~`) NON-iid base. baseIsIid(base) runs its Identifier
//     loop to completion (no AssignStatement for a `~`-bound name) and falls
//     through to `return false` (analyzer 2567), so the pre-pass emits the
//     clean "requires an iid base" diagnostic rather than letting a deferred
//     MvNormal-base locscale survive.
test('sample-bound non-iid base errors cleanly (analyzer baseIsIid fall-through)', () => {
  const lifted = processSource(
    `base ~ MvNormal(mean = [0.0, 0.0], cov = [[1.0, 0.0], [0.0, 1.0]])\n`
    + `X = locscale(base, [1.0, -1.0], [[2.0, 0.0], [0.0, 1.5]])\n`);
  const msg = lifted.diagnostics
    .filter((d: any) => d.severity === 'error').map((e: any) => e.message).join(' | ');
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged diagnostic, got: ${msg || '(none)'}`);
  assert.match(msg, /iid/i, `expected iid-base guidance, got: ${msg}`);
});

// 7. INLINE square-op scale (not a literal, not a named ref). Passing
//    `lower_cholesky(...)` DIRECTLY as the scale arg means scaleAst is a
//    CallExpr — lift's gate has no literal/named-ref handle for it (the
//    square-op admit fires only for a 1-level NAMED ref), so it falls through
//    to the `neither literal nor a named ref` fallback and the unrouted
//    locscale reaches the buildDerivations safety net, which throws the clean
//    locscale-tagged error. Exercises lift's else-branch fallback (1202-1203).
test('inline square-op scale falls back to a clean locscale error', () => {
  const src =
    `X = locscale(iid(Normal(0.0,1.0), 2), [0.0, 0.0], `
    + `lower_cholesky(rowstack([[4.0, 0.0], [0.0, 2.25]])))\n`
    + `lp = logdensityof(X, [1.5, -0.5])\n`;
  let msg = '';
  try {
    const lifted2 = processSource(src);
    const errs = lifted2.diagnostics.filter((d: any) => d.severity === 'error');
    if (errs.length) msg = errs.map((e: any) => e.message).join(' | ');
    else { orchestrator.buildDerivations(lifted2.bindings); }
  } catch (e: any) { msg = e.message; }
  assert.match(msg, /locscale/i,
    `expected a locscale-tagged error, got: ${msg || '(none)'}`);
});
