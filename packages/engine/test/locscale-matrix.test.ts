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
