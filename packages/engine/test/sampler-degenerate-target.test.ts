'use strict';
// Two ways a sampler run used to produce a confident-looking answer from a
// target it could not sample at all.
//
// 1. A log-posterior that is −∞ at every initial position. No proposal can be
//    accepted, so the driver returned the init positions verbatim — one frozen
//    value per chain, accept rate 0 — which the viewer renders as a posterior.
//    Observed on corpora/coverage/b_mass_peak, whose §09 standard-module
//    distribution members do not resolve on the pure-JS path: 4 unique draws
//    across 16000, accept rate 0, R̂ ≈ 2e13.
// 2. The slice kernel's shrinkage loop had no iteration bound. Its only exit
//    besides acceptance is `R − L < 1e-12`, which is FALSE once an endpoint is
//    NaN, so a chain position that has gone NaN leaves the loop with no
//    reachable exit. Guard (1) now catches a NaN init before the kernel runs, so
//    the bound is defence in depth rather than the fix for an observed hang.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');
const { makeSliceKernel } = require('../slice-kernel.ts');

// `Uniform(interval(2, 3))` prior against a likelihood truncated to [0, 1]:
// every prior draw scores −∞, and no reparameterisation can reach the support.
const NO_MASS = `
theta ~ Uniform(interval(2.0, 3.0))
obs_dist = normalize(truncate(Normal(mu = theta, sigma = 1.0), interval(0.0, 1.0)))
z ~ obs_dist
K = kernelof(record(z = z), theta = theta)
L = likelihoodof(K, record(z = 5.0))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;

for (const backend of ['mh', 'ram', 'slice', 'emcee', 'demcz', 'elliptical-slice-sampler',
  'amis', 'smc', 'nested']) {
  test(`backend:${backend} refuses a target that is -Infinity at every initial position`, async () => {
    const { ctx } = ctxFor(NO_MASS, 200);
    await assert.rejects(
      () => materialiser.materialiseMeasure('posterior', ctx,
        { backend, chains: 4, warmup: 20, draws: 20, seed: 1, nLive: 20, dlogz: 0.5,
          smcParticles: 100, smcSteps: 4, amisIters: 3, amisSamples: 50 }),
      (e: any) => {
        assert.match(e.message, /not finite at ANY|cannot move|no mass|vanished/,
          `unexpected message: ${e.message}`);
        return true;
      },
    );
  });
}

test('the slice kernel terminates from a NaN chain position', () => {
  // A NaN coordinate makes x0, L, R and every proposal NaN. `lps > logu` is
  // false for NaN, and so is the interval-collapse test `R - L < 1e-12`, so the
  // shrinkage loop has NO reachable exit without SHRINK_CAP. Verified: dropping
  // the cap makes this test hang rather than fail.
  const dim = 2;
  const mv: any = {
    dim,
    names: ['a', 'b'],
    logPosterior: () => NaN,
    logPosteriorBatch: (ys: Float64Array[]) => {
      const out = new Float64Array(ys.length);
      out.fill(NaN);
      return out;
    },
  };
  const kernel = makeSliceKernel();
  let s = 12345;
  const prng = () => { s = (1103515245 * s + 12345) >>> 0; return s / 4294967296; };

  for (const nWalkers of [1, 4]) {          // 1 → scalar path, 4 → batched path
    const ensemble: Float64Array[] = [];
    for (let w = 0; w < nWalkers; w++) ensemble.push(new Float64Array([NaN, NaN]));
    const logp = new Float64Array(nWalkers);
    logp.fill(NaN);
    const st = kernel.init(nWalkers, dim, {}, mv);
    // A sync kernel step cannot be timed out from here, so the assertion IS
    // that this call returns at all — an unbounded loop hangs the test file and
    // the runner reports it as a timeout.
    const r = kernel.step(ensemble, logp, mv, prng, st, 'sample');
    assert.ok(r && Number.isFinite(r.proposals), `nWalkers=${nWalkers}: no result`);
    // Bounded by dim × (stepping-out cap + shrink cap) × walkers.
    assert.ok(r.proposals <= dim * 250 * nWalkers,
      `nWalkers=${nWalkers}: ${r.proposals} evals exceeds the shrink cap`);
    assert.equal(r.accepts, 0, 'a NaN target must not accept');
  }
});
