'use strict';

// =====================================================================
// ksuperpose-sampling.test.ts — spec §06 `ksuperpose` component draw
// =====================================================================
//
// SPEC ANCHOR — docs/06-measure-algebra.md, "Additive superposition",
// `ksuperpose` entry, quoted verbatim: "Because the weights do not depend
// on the variate, the mixture is sampleable whenever `kernel` is."
//
// SPEC ANCHOR — same file, "Joint composition", `iid` entry: "the product
// measure $M^{\otimes N}$ over arrays of shape `size`". A product of sums
// is a sum over the component choices of every coordinate independently,
// so under `iid(ksuperpose(…), k)` EVERY COORDINATE SELECTS ITS OWN
// COMPONENT. flatppl-dev/flatppl-engine-concepts.md §22.4 "The repeat
// axis" states the failure mode this pins: "a single resample per block
// pins every position in the block to one component, collapsing the k
// draws that §06's product measure makes independent (correct per-position
// marginal, wrong joint — the iid(superpose) branch-pinning defect). Each
// output position needs its own draw within the block."
//
// The mixture is expanded to `superpose(weighted(w[i], κ(θᵢ)), …)`
// (ksuperpose-expand.ts), so the component draw is `matSuperpose`'s
// per-output-index selection and the freshening is inherited rather than
// re-derived. These tests are the evidence that it IS inherited — the
// expansion is not self-evidently enough, because a construct whose
// measure components sit outside the standard child edges falls through
// to TILING, which is the wrong-answer direction (§22.4: "a node the walk
// cannot reach falls through to TILING, which is the wrong-answer
// direction, not the safe one").
//
// ORACLES ARE INDEPENDENT of this engine — Distributions.jl `MixtureModel`
// moments, or closed-form conditioning by hand:
//   MixtureModel([Normal(-3,1), Normal(3,1)], [0.5, 0.5])   mean 0,   var 10
//   MixtureModel([Normal(-3,1), Normal(3,1)], [0.25, 0.75]) mean 1.5, var 7.75
//   MixtureModel([N(-3,1), N(0,1), N(3,1)], [1/3,1/3,1/3])  mean 0,   var 7
//
// TOLERANCES follow test/iid-superpose-branch-freshness.test.ts: Monte-Carlo
// slack at N = 20000 draws, about 6 sigma, against a defect that sat 9
// variance units and 3 mean units away.

const { test } = require('node:test');
const assert = require('node:assert');
const { processSource } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const N = 20000;
const MEAN_TOL = 0.15;
const VAR_TOL = 0.30;
const COV_TOL = 0.45;

// Per-coordinate means and variances, plus coordinate 0's covariance against
// each coordinate, off a materialised iid measure's atom-major samples.
async function momentsOf(src: string, name: string, k: number) {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure(name);
  const flat = Array.from(m.samples as any).map(Number);
  assert.equal(flat.length, N * k, 'sample count');
  const mean = new Array(k).fill(0);
  for (let i = 0; i < flat.length; i++) mean[i % k] += flat[i] / N;
  const varr = new Array(k).fill(0);
  const cov = new Array(k).fill(0);
  for (let a = 0; a < N; a++) {
    const d0 = flat[a * k] - mean[0];
    for (let i = 0; i < k; i++) {
      const di = flat[a * k + i] - mean[i];
      varr[i] += (di * di) / (N - 1);
      cov[i] += (d0 * di) / (N - 1);
    }
  }
  return { mean, varr, cov };
}

const errorsOf = (src: string) => (processSource(src).diagnostics || [])
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);

// The ±3 equal-weight mixture in the ksuperpose spelling. §06's headline
// example wraps the lift in `normalize`, and it is load-bearing: the mass is
// Σᵢ wᵢ, which §06 says "need not be normalized", so the bare lift is
// %finite and `draw` rejects it (see the mass test below).
const KMIX = 'w = [0.5, 0.5]\nmus = [-3.0, 3.0]\n'
  + 'M = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))\n';

// ── Every coordinate selects its own component ───────────────────────────────

for (const k of [2, 3, 4]) {
  test(`iid(ksuperpose(…), ${k}): every coordinate is its own mixture draw`,
    async () => {
      const { mean, varr, cov } = await momentsOf(KMIX + `b ~ iid(M, ${k})\n`, 'b', k);
      for (let i = 0; i < k; i++) {
        assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL,
          `mean[${i}] = ${mean[i]}, want 0 — a per-coordinate mean at a `
          + 'component centre is the branch-pinning signature');
        assert.ok(Math.abs(varr[i] - 10.0) < VAR_TOL,
          `var[${i}] = ${varr[i]}, want 10 — a variance near a single `
          + "component's is the branch-pinning signature");
      }
      for (let i = 1; i < k; i++) {
        assert.ok(Math.abs(cov[i]) < COV_TOL,
          `cov[0,${i}] = ${cov[i]}, want 0 (§06's product measure)`);
      }
    });
}

test('nested iid(iid(ksuperpose(…), 2), 3) freshens all six coordinates',
  async () => {
    const { mean, varr, cov } = await momentsOf(
      KMIX + 'b ~ iid(iid(M, 2), 3)\n', 'b', 6);
    for (let i = 0; i < 6; i++) {
      assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
      assert.ok(Math.abs(varr[i] - 10.0) < VAR_TOL, `var[${i}] = ${varr[i]}`);
    }
    for (let i = 1; i < 6; i++) {
      assert.ok(Math.abs(cov[i]) < COV_TOL, `cov[0,${i}] = ${cov[i]}`);
    }
  });

test('UNEQUAL weights select in proportion in every coordinate', async () => {
  const { mean, varr, cov } = await momentsOf(
    'w = [0.25, 0.75]\nmus = [-3.0, 3.0]\n'
    + 'M = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))\n'
    + 'b ~ iid(M, 3)\n', 'b', 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(mean[i] - 1.5) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
    assert.ok(Math.abs(varr[i] - 7.75) < VAR_TOL, `var[${i}] = ${varr[i]}`);
  }
  for (let i = 1; i < 3; i++) {
    assert.ok(Math.abs(cov[i]) < COV_TOL, `cov[0,${i}] = ${cov[i]}`);
  }
});

test('a THREE-component mixture freshens every coordinate', async () => {
  const third = '0.3333333333333333';
  const { mean, varr, cov } = await momentsOf(
    `w3 = [${third}, ${third}, ${third}]\nmus3 = [-3.0, 0.0, 3.0]\n`
    + 'M = normalize(ksuperpose(Normal, w3)(mu = mus3, sigma = 1.0))\n'
    + 'b ~ iid(M, 3)\n', 'b', 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
    assert.ok(Math.abs(varr[i] - 7.0) < VAR_TOL, `var[${i}] = ${varr[i]}`);
  }
  for (let i = 1; i < 3; i++) {
    assert.ok(Math.abs(cov[i]) < COV_TOL, `cov[0,${i}] = ${cov[i]}`);
  }
});

test('an unreplicated mixture (no iid) draws the mixture', async () => {
  const { mean, varr } = await momentsOf(KMIX + 'b ~ M\n', 'b', 1);
  assert.ok(Math.abs(mean[0] - 0.0) < MEAN_TOL, `mean = ${mean[0]}`);
  assert.ok(Math.abs(varr[0] - 10.0) < VAR_TOL, `var = ${varr[0]}`);
});

// ── Genuine shared ancestors must STAY shared ────────────────────────────────
//
// §06: "shared stochastic ancestors retained". The wrong fix freshens the
// whole subtree, swapping a correct joint for a correct marginal over a wrong
// joint. Cross-coordinate covariance is the only statistic that separates the
// two: 0 when the ancestor is freshened, Var(E[X | ancestor]) when it is
// shared. So each control asserts the COVARIANCE; the mean and variance are
// unmoved by the distinction and are there only to catch a broken model.

test('CONTROL: PARAMETERIZED weights stay shared across the coordinates',
  async () => {
    // wp ~ Dirichlet(1, 1) is uniform on the 2-simplex, so wp[1] = psi with
    // psi ~ U(0,1) and X | psi ~ psi·N(−3,1) + (1−psi)·N(3,1). Closed form by
    // conditioning: E[X|psi] = 3 − 6psi, E[X²|psi] = 10, so
    //   Cov(X_i, X_j) = Var(3 − 6psi) = 36·(1/12) = 3
    //   Var(X)        = E[10 − (3 − 6psi)²] + 3 = 10 − 3 + 3 = 10
    //   E[X]          = 0
    // Confirmed at 2e6 draws in Julia: mean 0.0042, var 10.004, cov 2.997.
    // Freshening the weights would send the covariance to 0 and leave the
    // mean and variance untouched, so the covariance is the whole assertion.
    const { mean, varr, cov } = await momentsOf(
      'wp ~ Dirichlet(alpha = [1.0, 1.0])\nmus = [-3.0, 3.0]\n'
      + 'M = normalize(ksuperpose(Normal, wp)(mu = mus, sigma = 1.0))\n'
      + 'b ~ iid(M, 3)\n', 'b', 3);
    for (let i = 0; i < 3; i++) {
      assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
      assert.ok(Math.abs(varr[i] - 10.0) < VAR_TOL, `var[${i}] = ${varr[i]}`);
    }
    for (let i = 1; i < 3; i++) {
      assert.ok(Math.abs(cov[i] - 3.0) < COV_TOL,
        `cov[0,${i}] = ${cov[i]}, want 3 — a covariance near 0 means the `
        + 'parameterized weights were freshened per coordinate');
    }
  });

test('CONTROL: a HELD-CONSTANT family argument from an outer draw stays shared '
  + 'across the coordinates', async () => {
  // m ~ N(0,1) is the shared location; the family axis is sigma = [0.5, 4.0].
  // X | m ~ 0.5·N(m, 0.5²) + 0.5·N(m, 4²), so E[X|m] = m and
  //   Cov(X_i, X_j) = Var(m) = 1
  //   Var(X)        = (0.5·0.25 + 0.5·16) + 1 = 9.125
  //   E[X]          = 0
  // Confirmed at 2e6 draws in Julia: mean −0.0026, var 9.106, cov 1.0005.
  // `m` is reached only as a distribution parameter, never in measure
  // position, so the freshening carve-out must leave it tiled
  // (engine-concepts §22.4: distIR/weightIR are the fields the walk never
  // enters, by design).
  //
  // VAR_TOL does not apply here: the 4-sigma component makes the fourth
  // moment large, so the variance estimator's own standard error at 20000
  // draws is 0.132 (measured in the same Julia session) against 0.044 for
  // the ±3 mixture. 0.60 is the same ~4.5 sigma of slack.
  const VAR_TOL_HEAVY = 0.60;
  const { mean, varr, cov } = await momentsOf(
    'm ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'w = [0.5, 0.5]\nsigs = [0.5, 4.0]\n'
    + 'M = normalize(ksuperpose(Normal, w)(mu = m, sigma = sigs))\n'
    + 'b ~ iid(M, 3)\n', 'b', 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(mean[i] - 0.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
    assert.ok(Math.abs(varr[i] - 9.125) < VAR_TOL_HEAVY, `var[${i}] = ${varr[i]}`);
  }
  for (let i = 1; i < 3; i++) {
    assert.ok(Math.abs(cov[i] - 1.0) < COV_TOL,
      `cov[0,${i}] = ${cov[i]}, want 1 — a covariance near 0 means the `
      + 'shared family argument was freshened per coordinate');
  }
});

// ── Mass: §06 makes the bare lift unnormalized ───────────────────────────────

test('§06: the mixture mass is Σᵢ wᵢ and "need not be normalized", so `draw` '
  + 'rejects the bare lift and §06\'s normalize wrapper is load-bearing', () => {
  const bare = errorsOf('w = [0.3, 1.2]\nmus = [-3.0, 3.0]\n'
    + 'M = ksuperpose(Normal, w)(mu = mus, sigma = 1.0)\nb ~ M\n');
  assert.ok(bare.some((m: string) => /draw requires a probability measure/.test(m)),
    `want the draw mass gate, got ${JSON.stringify(bare)}`);
  const wrapped = errorsOf('w = [0.3, 1.2]\nmus = [-3.0, 3.0]\n'
    + 'M = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))\nb ~ M\n');
  assert.deepEqual(wrapped, [], 'the normalize wrapper clears the gate');
});

// ── All-zero weights: §06 makes SAMPLING undefined ──────────────────────────
//
// §06: "when every weight is zero it is the zero measure (density 0,
// log-density −∞, sampling undefined)". The density half is pinned in
// ksuperpose-density.test.ts (−∞ exactly, at three points). This is the
// sampling half, which used to return ONE CONSTANT repeated for every atom:
// the selection resampled a cumulative distribution built from all-−∞ weights,
// which pins index 0. A plausible-looking number for a measure with no draws.
//
// The check lives in `matSuperpose` — where the superposition computes its
// total mass — so it is the SHARED layer: the hand-written variadic spelling
// reaches the same guard whenever it materialises.

test('§06: drawing from an all-zero-weight mixture is a located error, not a '
  + 'repeated constant', async () => {
  const src = 'wz = [0.0, 0.0]\nmus = [-3.0, 3.0]\n'
    + 'M = normalize(ksuperpose(Normal, wz)(mu = mus, sigma = 1.0))\n'
    + 'b ~ M\n';
  const { ctx } = ctxFor(src, 6);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('b')), (e: any) => {
    assert.match(e.message, /every component has zero mass/);
    assert.match(e.message, /spec §06/);
    assert.match(e.message, /sampling undefined/);
    // Names the binding, so the error is located in the model.
    assert.match(e.message, /^superpose '/);
    return true;
  });
});

test('§06: the all-zero refusal also fires under iid, where the pinning was '
  + 'worst', async () => {
  const src = 'wz = [0.0, 0.0]\nmus = [-3.0, 3.0]\n'
    + 'M = normalize(ksuperpose(Normal, wz)(mu = mus, sigma = 1.0))\n'
    + 'b ~ iid(M, 3)\n';
  const { ctx } = ctxFor(src, 6);
  await assert.rejects(() => Promise.resolve(ctx.getMeasure('b')),
    /every component has zero mass/);
});

test('CONTROL: ONE zero weight still samples — the guard tests the TOTAL mass, '
  + 'not any single weight', async () => {
  // w = [0.0, 1.2] leaves the N(3, 1) component carrying all the mass, so
  // every draw comes from it: mean 3, var 1. A guard that fired on any zero
  // weight rather than on Σw would break this legal model.
  const { mean, varr } = await momentsOf(
    'w1 = [0.0, 1.2]\nmus = [-3.0, 3.0]\n'
    + 'M = normalize(ksuperpose(Normal, w1)(mu = mus, sigma = 1.0))\n'
    + 'b ~ iid(M, 3)\n', 'b', 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(mean[i] - 3.0) < MEAN_TOL, `mean[${i}] = ${mean[i]}, want 3`);
    assert.ok(Math.abs(varr[i] - 1.0) < VAR_TOL, `var[${i}] = ${varr[i]}, want 1`);
  }
});

// The MCMC route resolves `normalize` in its OWN copy of the rewrite
// (mcmc-density.ts), separately from the IS route's (mat-density.ts). With the
// mark on only one of them, `{backend: 'mh'}` on an all-zero mixture scored NaN,
// MH rejected every proposal, and the run reported a CONSTANT CHAIN with zero
// diagnostics — while the IS route refused the same model. Two routes, two
// answers, one measure. Both copies are marked now, and the refusal has to
// SURFACE as well as fire: `likWith` swallows density throws to −∞ on purpose
// (a proposal outside support must be rejected, not crash), so the §06 refusal
// is tagged and re-thrown through a one-time probe.
test('§06: the all-zero refusal reaches the MCMC route too, not just the IS '
  + 'route', async () => {
  const orchestrator = require('../orchestrator.ts');
  const materialiser = require('../materialiser.ts');
  const { createWorkerHandler } = require('../worker.ts');
  const SRC = 'wz = [0.0, 0.0]\noffs = [-1.0, 2.0]\n'
    + 'mu = draw(Normal(mu = 0.0, sigma = 10.0))\n'
    + 'mix = normalize(ksuperpose(Normal, wz)(mu = offs, sigma = 1.0))\n'
    + 'obs_dist = joint(y = mix)\n'
    + 'K = functionof(obs_dist, mu = mu)\n'
    + 'L = likelihoodof(K, record(y = 0.5))\n'
    + 'prior = lawof(record(mu = mu))\n'
    + 'posterior = bayesupdate(L, prior)\n';
  const lifted = processSource(SRC);
  assert.deepEqual(
    lifted.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 7 });
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
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: 200, rootSeed: 42, rootKey: 42,
  };
  await assert.rejects(
    () => materialiser.materialiseMeasure('posterior', ctx,
      { backend: 'mh', chains: 2, warmup: 50, draws: 50, seed: 1 }),
    (e: any) => {
      assert.match(e.message, /ZERO total mass/);
      assert.match(e.message, /spec §06/);
      assert.match(e.message, /backend 'mh'/);
      return true;
    });
});

test('§06: an unnormalized mixture draws in proportion to the weights — the '
  + 'weights live in the mass, the selection in the sample', async () => {
  // normalize(w = [0.25, 0.75]) is the same selection law as the unnormalized
  // [1.0, 3.0], so the moments are MixtureModel([N(-3,1),N(3,1)],[.25,.75])'s.
  const { mean, varr } = await momentsOf(
    'w = [1.0, 3.0]\nmus = [-3.0, 3.0]\n'
    + 'M = normalize(ksuperpose(Normal, w)(mu = mus, sigma = 1.0))\n'
    + 'b ~ iid(M, 3)\n', 'b', 3);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(mean[i] - 1.5) < MEAN_TOL, `mean[${i}] = ${mean[i]}`);
    assert.ok(Math.abs(varr[i] - 7.75) < VAR_TOL, `var[${i}] = ${varr[i]}`);
  }
});
