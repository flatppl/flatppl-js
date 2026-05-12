'use strict';

// Statistical tests for the closed-measure and kernel-sample paths
// the viewer drives. We test the underlying primitives — orchestrator's
// IR-expansion helpers plus the worker's sampleN with refArrays — on
// a small Normal-Exponential scale-mixture model whose marginal moments
// are analytical, then verify the empirical mean / variance of N
// sampled atoms lands within statistical tolerance of the closed form.
//
// Source model (`packages/engine/test/fixtures/scale-mixture.flatppl`
// keeps an authoritative copy):
//
//   mu = elementof(reals)
//   sigma ~ Exponential(rate = 1.0)
//   x ~ Normal(mu = mu, sigma = sigma)
//   kernel = kernelof(x, mu = mu)
//
// Analytical marginals (sigma ~ Exp(1) ⇒ E[sigma]=1, Var(sigma)=1,
// E[sigma²]=2):
//
//   closed measure `x` with mu=0:
//     E[x] = 0
//     Var(x) = E[sigma²] = 2
//   kernel(mu_val):
//     E[x | mu_val] = mu_val
//     Var(x | mu_val) = 2  (same as closed; mu shifts the mean only)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');
const orchestrator = require('../orchestrator');
const { createWorkerHandler } = require('../worker');

const N = 50000;  // sample count

// -- statistics ---------------------------------------------------------

function meanVar(samples) {
  let m = 0;
  for (let i = 0; i < samples.length; i++) m += samples[i];
  m /= samples.length;
  let v = 0;
  for (let i = 0; i < samples.length; i++) v += (samples[i] - m) ** 2;
  v /= samples.length;
  return { mean: m, variance: v };
}

// 3-σ-ish tolerance bands for the empirical mean and variance of N
// atoms of a Normal-Exponential scale mixture.
// - Var(sample mean) = Var(X)/N = 2/N. 3σ = 3·sqrt(2/N).
// - Sample variance under heavy-tailed (Laplace-ish) marginal is
//   noisier; allow ~3% absolute for N=50000.
const MEAN_TOL = 3 * Math.sqrt(2 / N);  // ~0.019
const VAR_TOL  = 0.10;                  // ~5% of 2.0, very forgiving

// -- shared pipeline ----------------------------------------------------

// Build derivations once and reuse across tests.
function buildModel() {
  const src = `
mu = elementof(reals)
sigma = draw(Exponential(rate = 1.0))
x = draw(Normal(mu = mu, sigma = sigma))
kernel = kernelof(x, mu = mu)
`;
  const r = processSource(src);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const ds = orchestrator.buildDerivations(lifted);
  return { src, bindings: lifted, derivations: ds.derivations };
}

// Sample N atoms of a binding through the worker, with refArrays
// resolving every captured self-ref to per-atom values. Mirrors what
// the viewer's getMeasure + materialiseConcreteMeasure do for a leaf
// distribution: collect refs, recursively sample each, then sample
// the target distribution with the resolved refArrays.
function sampleClosed(wh, distIR, derivations, count, seed) {
  const refs = Array.from(orchestrator.collectSelfRefs(distIR));
  const refPromises = refs.map((name, i) => {
    const innerIR = orchestrator.leafSampleIR(name, derivations);
    if (!innerIR) throw new Error(`no leaf sample IR for captured ref '${name}'`);
    const reply = wh.handle({
      type: 'sampleN', ir: innerIR, count, seed: seed + 1000 + i,
    });
    if (reply.type !== 'samples') {
      throw new Error('captured-ref sampleN failed: ' + JSON.stringify(reply));
    }
    return [name, reply.samples];
  });
  const refArrays = Object.fromEntries(refPromises);
  const reply = wh.handle({
    type: 'sampleN', ir: distIR, count, refArrays, seed,
  });
  if (reply.type !== 'samples') {
    throw new Error('target sampleN failed: ' + JSON.stringify(reply));
  }
  return reply.samples;
}

// =====================================================================
// Closed-measure path: x with mu = 0 (replace the elementof mu with a
// literal so x has a derivation).
// =====================================================================

test('marginal: closed measure `x ~ Normal(0, sigma)` with sigma ~ Exp(1)', () => {
  // Substitute mu=0 directly in source so we exercise the same path the
  // viewer's getMeasure(name) takes for a closed-measure binding.
  const src = `
sigma = draw(Exponential(rate = 1.0))
x = draw(Normal(mu = 0, sigma = sigma))
`;
  const r = processSource(src);
  const lifted = orchestrator.liftInlineSubexpressions(r.bindings);
  const ds = orchestrator.buildDerivations(lifted);
  const xIR = orchestrator.leafSampleIR('x', ds.derivations);
  assert.ok(xIR, 'x must have a leaf sample IR');

  const wh = createWorkerHandler();
  wh.handle({ type: 'init', seed: 1 });
  const samples = sampleClosed(wh, xIR, ds.derivations, N, 42);

  const { mean, variance } = meanVar(samples);
  assert.ok(Math.abs(mean - 0) < MEAN_TOL,
    `closed x: mean ${mean} not within ${MEAN_TOL} of 0`);
  assert.ok(Math.abs(variance - 2) < VAR_TOL,
    `closed x: variance ${variance} not within ${VAR_TOL} of 2`);
});

// =====================================================================
// Kernel-sample path: kernel(mu = mu_val). Drives the same pipeline
// the viewer uses — expandMeasureIR with bindings fallback (because
// x's derivation is pruned due to depending on the parameterized mu),
// inlineForProfile (rewrites `ref self mu` → `ref %local mu`),
// substituteLocals (binds the env value), then materialise with
// refArrays for the captured `sigma`.
// =====================================================================

function sampleKernelAtMu(muVal, seed) {
  const { bindings, derivations } = buildModel();
  const wh = createWorkerHandler();
  wh.handle({ type: 'init', seed: 1 });

  // 1. Resolve kernel's signature → body IR (lawof(ref self x)).
  const sig = orchestrator.signatureOf('kernel', bindings);
  assert.equal(sig.kind, 'kernel');
  assert.equal(sig.inputs.length, 1);
  const paramNames = sig.inputs.map((inp) => inp.paramName);

  // 2. Expand the body. expandMeasureRefsInIR strips lawof and tries
  //    to expand the inner ref via derivations; x's derivation is
  //    pruned (Normal kwargs reference the parameterized mu), so we
  //    fall back to the bindings-walking branch of expandMeasureIR.
  let ir = orchestrator.expandMeasureRefsInIR(sig.body, derivations);
  if (ir && ir.kind === 'ref' && ir.ns === 'self') {
    ir = orchestrator.expandMeasureIR(
      ir.name, derivations, undefined, bindings);
  }
  assert.ok(ir, 'kernel body should expand to a measure IR');

  // 3. Substitute the kernel input (mu) — first inlineForProfile
  //    rewrites `ref self mu` → `ref %local mu`, then substituteLocals
  //    binds %local.mu to the literal value.
  ir = orchestrator.inlineForProfile(ir, paramNames, bindings, derivations);
  ir = orchestrator.substituteLocals(ir, { mu: muVal });

  // 4. Sample the target distribution with refArrays for the
  //    captured `sigma`. After the substitutions above, ir is
  //    `Normal(mu = <lit>, sigma = ref self sigma)` (no iid in this
  //    simplified model — x was draw'd directly, not iid'd).
  assert.equal(ir.op, 'Normal');
  return sampleClosed(wh, ir, derivations, N, seed);
}

test('marginal: kernel(mu = 5) has E[x] ≈ 5, Var(x) ≈ 2', () => {
  const samples = sampleKernelAtMu(5.0, 100);
  const { mean, variance } = meanVar(samples);
  assert.ok(Math.abs(mean - 5) < MEAN_TOL,
    `kernel(5): mean ${mean} not within ${MEAN_TOL} of 5`);
  assert.ok(Math.abs(variance - 2) < VAR_TOL,
    `kernel(5): variance ${variance} not within ${VAR_TOL} of 2`);
});

test('marginal: kernel(mu = -3) shifts the mean by -3, variance unchanged', () => {
  const samples = sampleKernelAtMu(-3.0, 200);
  const { mean, variance } = meanVar(samples);
  assert.ok(Math.abs(mean + 3) < MEAN_TOL,
    `kernel(-3): mean ${mean} not within ${MEAN_TOL} of -3`);
  assert.ok(Math.abs(variance - 2) < VAR_TOL,
    `kernel(-3): variance ${variance} not within ${VAR_TOL} of 2`);
});

test('marginal: kernel-sample mean tracks the input across multiple mu values', () => {
  // The kernel input is the LOCATION parameter; varying it should
  // shift the empirical mean linearly while leaving the variance
  // unchanged.
  for (const muVal of [0, 1, 10]) {
    const samples = sampleKernelAtMu(muVal, 300 + muVal);
    const { mean, variance } = meanVar(samples);
    assert.ok(Math.abs(mean - muVal) < MEAN_TOL,
      `kernel(${muVal}): mean ${mean} not within ${MEAN_TOL}`);
    assert.ok(Math.abs(variance - 2) < VAR_TOL,
      `kernel(${muVal}): variance ${variance} not within ${VAR_TOL}`);
  }
});

// =====================================================================
// Regression guard for the original bug — closing `sigma` to its
// samples[0] (one value) collapses the variance to that single
// sigma². The current per-atom refArrays path produces ~2; the
// broken path would produce a (random, much smaller) value drawn
// from one Exp(1) sample squared.
// =====================================================================

test('marginal: kernel variance reflects mixture, not a single sigma sample', () => {
  // Re-sample sigma's first atom independently and verify that the
  // kernel's empirical variance is *not* close to that single value's
  // square. A pre-fix run collapsed all atoms to sigma_0^2, which is
  // a different number for each test seed.
  const wh = createWorkerHandler();
  wh.handle({ type: 'init', seed: 1 });
  const sigmaIR = orchestrator.leafSampleIR('sigma',
    orchestrator.buildDerivations(
      orchestrator.liftInlineSubexpressions(
        processSource('sigma = draw(Exponential(rate = 1.0))').bindings)
    ).derivations);
  const oneSigma = wh.handle({
    type: 'sampleN', ir: sigmaIR, count: 1, seed: 100,
  }).samples[0];
  const wrongVar = oneSigma * oneSigma;

  const samples = sampleKernelAtMu(0.0, 100);
  const { variance } = meanVar(samples);
  // Variance should be near 2, NOT near wrongVar (which is Exp(1)²
  // — anywhere from near-zero to several units, but with very high
  // probability ≠ 2). We allow a 0.3 separation as the trip wire.
  assert.ok(Math.abs(variance - 2) < 0.3,
    `variance ${variance} not near analytical 2`);
  assert.ok(Math.abs(variance - wrongVar) > 0.1
            || Math.abs(wrongVar - 2) < 0.5,
    `variance ${variance} suspiciously close to single-sigma² ${wrongVar} — `
    + 'the per-atom refArrays path may have regressed');
});
