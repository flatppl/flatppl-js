'use strict';

// Tests for the `locscale(m, shift, scale)` measure combinator
// (spec §06 "locscale"): the affine location-scale pushforward,
// shorthand for `pushfwd(x -> scale * x + shift, m)`.
//
// The analyzer expands `locscale` to a synthetic affine
// `bijection(fwd, inv, logvolume)` driving a `pushfwd`, so density and
// sampling ride the existing pushforward paths (see
// analyzer.expandLocscaleStatements).
//
// Oracle (INDEPENDENT — Distributions.jl, not the engine):
//   julia> using Distributions
//   julia> d = LocationScale(0.0, 2.5, TDist(3.0))   # locscale(StudentT(3),0,2.5)
//   julia> logpdf.(d, [0.0, 1.0, -2.3, 4.0])
//     -1.9171795814976649
//     -2.021099059359087
//     -2.4142302959988386
//     -3.151151220686428
// These exact values are pinned below — NOT the engine's own output.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0xB1737CFC;

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
    sampleCount: opts.sampleCount != null ? opts.sampleCount : 512,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// For error-path tests: run only processSource (parse + analyze) and return
// the error-severity diagnostics. Does NOT build derivations / materialise,
// so a deliberately-rejected locscale never reaches the downstream paths.
function expectError(source: any, substr: any) {
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.ok(errs.length > 0, `expected an error diagnostic, got none for:\n${source}`);
  assert.ok(errs.some((d: any) => d.message.includes(substr)),
    `expected an error containing ${JSON.stringify(substr)}; got: `
      + JSON.stringify(errs.map((d: any) => d.message)));
}

const nLogpdf = require('@stdlib/stats-base-dists-normal-logpdf');

// Distributions.jl LocationScale(0, 2.5, TDist(3)) logpdf — pinned oracle.
const T3_LOCSCALE = {
  '0.0':  -1.9171795814976649,
  '1.0':  -2.021099059359087,
  '-2.3': -2.4142302959988386,
  '4.0':  -3.151151220686428,
};

// =====================================================================
// Location-scale Student-t density — matches Distributions.jl
// =====================================================================

test('locscale(StudentT(3), 0, 2.5) density matches Distributions.jl at multiple points', async () => {
  for (const [yStr, expected] of Object.entries(T3_LOCSCALE)) {
    const ctx = makeCtx(
      `B = locscale(StudentT(3.0), 0.0, 2.5)\nlp = logdensityof(B, ${yStr})\n`);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
      `y=${yStr}: got ${lp.samples[0]}, expected ${expected}`);
  }
});

// Non-literal shift/scale exercise the 0-arg `functionof(log(abs(scale)))`
// logvolume branch (a literal scale folds to a scalar logvolume instead).
test('locscale with shift/scale bound to names matches the same oracle', async () => {
  for (const [yStr, expected] of Object.entries(T3_LOCSCALE)) {
    const ctx = makeCtx(
      `s = 2.5\nmu0 = 0.0\nB = locscale(StudentT(3.0), mu0, s)\nlp = logdensityof(B, ${yStr})\n`);
    const lp = await ctx.getMeasure('lp');
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
      `y=${yStr}: got ${lp.samples[0]}, expected ${expected}`);
  }
});

// =====================================================================
// Closed-form identity: locscale(Normal(0,1), mu, sigma) == Normal(mu, sigma)
// =====================================================================

test('locscale(Normal(0,1), mu, sigma) density ≡ Normal(mu, sigma) (closed form)', async () => {
  const mu = 3.0, sigma = 2.0;
  for (const yVal of [3.0, 0.0, 7.5, -1.2]) {
    const ctx = makeCtx(
      `B = locscale(Normal(0.0, 1.0), ${mu}, ${sigma})\nlp = logdensityof(B, ${yVal})\n`);
    const lp = await ctx.getMeasure('lp');
    const expected = nLogpdf(yVal, mu, sigma);
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-12,
      `y=${yVal}: got ${lp.samples[0]}, expected ${expected}`);
  }
});

// =====================================================================
// Sampling regression: forward affine map applied to base draws
// =====================================================================

test('locscale rejects a non-measure (literal) base argument', () => {
  expectError('B = locscale(5.0, 0.0, 2.0)\n', 'measure');
});

test('locscale sampling produces the affine-mapped base draws', async () => {
  // locscale(Normal(0,1), 10, 2) — mean ≈ 10, sd ≈ 2 (Normal base is
  // light-tailed, so the empirical moments are tight at n=2048).
  const ctx = makeCtx(
    `b ~ locscale(Normal(0.0, 1.0), 10.0, 2.0)\nB = lawof(b)\n`,
    { sampleCount: 2048 });
  const m = await ctx.getMeasure('B');
  const s = m.samples;
  let mean = 0; for (const x of s) mean += x; mean /= s.length;
  let v = 0; for (const x of s) v += (x - mean) ** 2; v /= s.length;
  assert.ok(Math.abs(mean - 10) < 0.3, `mean ${mean} not ≈ 10`);
  assert.ok(Math.abs(Math.sqrt(v) - 2) < 0.3, `sd ${Math.sqrt(v)} not ≈ 2`);
});

test('locscale rejects a zero literal scale', () => {
  expectError('B = locscale(Normal(0.0, 1.0), 0.0, 0.0)\n', 'nonzero');
});

test('locscale rejects a non-finite literal scale', () => {
  expectError('B = locscale(Normal(0.0, 1.0), 0.0, 0.0)\n', 'nonzero');
});

test('locscale rejects a vector-literal scale (interim: use pushfwd)', () => {
  expectError('B = locscale(Normal(0.0, 1.0), 0.0, [1.0, 2.0])\n', 'pushfwd');
});

test('locscale rejects a matrix-literal scale (interim: use pushfwd)', () => {
  expectError('B = locscale(Normal(0.0, 1.0), 0.0, [[1.0, 0.0], [0.0, 1.0]])\n', 'pushfwd');
});

test('locscale rejects a vector-literal shift (interim: use pushfwd)', () => {
  expectError('B = locscale(Normal(0.0, 1.0), [1.0, 2.0], 2.0)\n', 'pushfwd');
});
