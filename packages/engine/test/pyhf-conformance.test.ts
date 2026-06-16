'use strict';

// Cross-engine numerical conformance: a FlatPPL model in the shape the
// pyhf/HS3 importer emits (point-free `likelihoodof(measure, data)` with
// free `elementof` params, terms combined via `joint_likelihood`) must
// score the SAME log-density as pyhf's `model.logpdf(θ, data)`.
//
// Oracle numbers are pyhf 0.7.x on the canonical 2-bin/1-channel
// uncorrelated-background workspace (crates/hs3/tests/fixtures/
// 2bin_1channel.json), independently reproduced by scipy:
//   main Poisson term @ μ=1, γ=[1,1]      = -6.829566590859514
//   aux  ContinuedPoisson term @ γ=[1,1]  = -5.754066558029820
//   combined (pyhf model.logpdf)          = -12.583633148889334

const test = require('node:test');
const assert = require('node:assert');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function score(src: string, target: string) {
  const ps = processSource(src);
  const built = orchestrator.buildDerivations(ps.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 8,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    // Thread the alias→standard-module map so measure bodies that call
    // standard-module functions (e.g. `hepphys.interp_*`) dispatch.
    moduleRegistry: ps.loweredModule && ps.loweredModule.moduleRegistry,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m; },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx.getMeasure(target).then((m: any) => m.samples[0]);
}

const TOL = 1e-9;

test('point-free Poisson term scores like pyhf main term', async () => {
  const v = await score(`
mu = elementof(reals)
uncorr_bkguncrt = elementof(cartpow(posreals, 2))
obs_model = Poisson.([5.0, 10.0] .* mu .+ [50.0, 60.0] .* uncorr_bkguncrt)
L = likelihoodof(obs_model, [50.0, 60.0])
ld = logdensityof(L, record(mu = 1.0, uncorr_bkguncrt = [1.0, 1.0]))
`, 'ld');
  assert.ok(Math.abs(v - (-6.829566590859514)) < TOL, `got ${v}`);
});

test('joint_likelihood of two point-free terms folds to the sum', async () => {
  // Two independent Poisson terms; logdensityof(joint_likelihood(...), θ)
  // = sum of the per-term log-densities.
  const a = -6.829566590859514;          // Poisson([55,70] | [50,60])
  const b = await score(`
mu = elementof(reals)
g = elementof(cartpow(posreals, 2))
m1 = Poisson.([5.0, 10.0] .* mu .+ [50.0, 60.0] .* g)
m2 = Poisson.([5.0, 10.0] .* mu .+ [50.0, 60.0] .* g)
L = joint_likelihood(likelihoodof(m1, [50.0, 60.0]), likelihoodof(m2, [50.0, 60.0]))
ld = logdensityof(L, record(mu = 1.0, g = [1.0, 1.0]))
`, 'ld');
  assert.ok(Math.abs(b - 2 * a) < TOL, `got ${b}, want ${2 * a}`);
});

test('ContinuedPoisson aux term scores like pyhf constraint term', async () => {
  const v = await score(`
hepphys = standard_module("particle-physics", "0.1")
g = elementof(cartpow(posreals, 2))
aux_model = hepphys.ContinuedPoisson.(g .* ([50.0, 60.0] ./ [5.0, 12.0]) .^ 2)
L = likelihoodof(aux_model, ([50.0, 60.0] ./ [5.0, 12.0]) .^ 2)
ld = logdensityof(L, record(g = [1.0, 1.0]))
`, 'ld');
  assert.ok(Math.abs(v - (-5.754066558029820)) < TOL, `got ${v}`);
});

test('full pyhf 2bin joint (main Poisson + ContinuedPoisson aux) = pyhf logpdf', async () => {
  // The exact shape the importer emits for 2bin_1channel.json: point-free
  // terms (free elementof params), main + shapesys-aux combined via
  // joint_likelihood, scored in ONE logdensityof call.
  const v = await score(`
hepphys = standard_module("particle-physics", "0.1")
mu = elementof(reals)
uncorr_bkguncrt = elementof(cartpow(posreals, 2))
obs_model = Poisson.([5.0, 10.0] .* mu .+ [50.0, 60.0] .* uncorr_bkguncrt)
aux_model = hepphys.ContinuedPoisson.(uncorr_bkguncrt .* ([50.0, 60.0] ./ [5.0, 12.0]) .^ 2)
L = joint_likelihood(
  likelihoodof(obs_model, [50.0, 60.0]),
  likelihoodof(aux_model, ([50.0, 60.0] ./ [5.0, 12.0]) .^ 2)
)
ld = logdensityof(L, record(mu = 1.0, uncorr_bkguncrt = [1.0, 1.0]))
`, 'ld');
  assert.ok(Math.abs(v - (-12.583633148889334)) < TOL, `got ${v}`);
});

test('lumi modifier (Gaussian constraint) = pyhf logpdf', async () => {
  // Importer output for a single-channel model with a `lumi` modifier
  // (luminosity scales the rate; Gaussian-constrained at 1.0, σ=0.1).
  // pyhf model.logpdf @ lumi=1 = -4.5829601350863065.
  const v = await score(`
hepphys = standard_module("particle-physics", "0.1")
lumi = elementof(posreals)
obs_model = Poisson.([50.0, 55.0] .* lumi)
L = joint_likelihood(
  likelihoodof(obs_model, [52.0, 58.0]),
  likelihoodof(Normal(mu = lumi, sigma = 0.1), 1.0)
)
ld = logdensityof(L, record(lumi = 1.0))
`, 'ld');
  assert.ok(Math.abs(v - (-4.5829601350863065)) < TOL, `got ${v}`);
});

test('staterror modifier (per-bin Gaussian constraint) = pyhf logpdf', async () => {
  // Importer output for a single-channel 2-bin model with a `staterror`
  // modifier (per-bin MC-stat γ, Gaussian-constrained at 1.0 with relative
  // δ = err/nominal). Aux observation is the per-bin vector [1.0, 1.0].
  // pyhf model.logpdf @ γ=[1,1] = -3.440475632113821.
  const v = await score(`
se = elementof(cartpow(posreals, 2))
obs_model = Poisson.([50.0, 55.0] .* se)
L = joint_likelihood(
  likelihoodof(obs_model, [52.0, 58.0]),
  likelihoodof(Normal.(se, [0.1, 0.12727272727272726]), [1.0, 1.0])
)
ld = logdensityof(L, record(se = [1.0, 1.0]))
`, 'ld');
  assert.ok(Math.abs(v - (-3.440475632113821)) < TOL, `got ${v}`);
});

test('normsys modifier (code4 / interp_poly6_exp) = pyhf logpdf', async () => {
  // Importer output: a multiplicative normsys systematic interpolated by
  // hepphys.interp_poly6_exp (pyhf code4), Gaussian-constrained α at 0.
  // pyhf model.logpdf @ μ=1, α=0 = -6.8849658436465315.
  const v = await score(`
hepphys = standard_module("particle-physics", "0.1")
mu = elementof(reals)
theta_ns = elementof(reals)
obs_model = Poisson.([8.0, 9.0] .* mu .+ [50.0, 55.0] .* hepphys.interp_poly6_exp(0.88, 1.0, 1.15, theta_ns))
L = joint_likelihood(
  likelihoodof(obs_model, [58.0, 62.0]),
  likelihoodof(Normal(mu = theta_ns, sigma = 1.0), 0.0)
)
ld = logdensityof(L, record(mu = 1.0, theta_ns = 0.0))
`, 'ld');
  assert.ok(Math.abs(v - (-6.8849658436465315)) < TOL, `got ${v}`);
});

test('histosys modifier (code4p / interp_poly6_lin, per-bin) = pyhf logpdf', async () => {
  // Importer output: an additive histosys systematic interpolated per-bin by
  // hepphys.interp_poly6_lin (pyhf code4p) on the template VECTORS, Gaussian-
  // constrained α at 0. pyhf model.logpdf @ α=0, μ=1 = -6.8849658436465315.
  const v = await score(`
hepphys = standard_module("particle-physics", "0.1")
mu = elementof(reals)
theta_hs = elementof(reals)
obs_model = Poisson.([8.0, 9.0] .* mu .+ hepphys.interp_poly6_lin([47.0, 52.0], [50.0, 55.0], [53.0, 59.0], theta_hs))
L = joint_likelihood(
  likelihoodof(obs_model, [58.0, 62.0]),
  likelihoodof(Normal(mu = theta_hs, sigma = 1.0), 0.0)
)
ld = logdensityof(L, record(theta_hs = 0.0, mu = 1.0))
`, 'ld');
  assert.ok(Math.abs(v - (-6.8849658436465315)) < TOL, `got ${v}`);
});
