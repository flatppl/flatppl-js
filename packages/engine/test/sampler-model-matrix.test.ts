'use strict';
// sampler-model-matrix.test.ts — ROBUSTNESS matrix: every backend must
// materialise a posterior for every example model, with a non-empty record of
// fields (each carrying samples). This is a no-crash / no-empty / no-regression
// guard across the cross-product — the layer above the per-model density-oracle
// tests. It exists because this class of model (matrix latents, kernel-broadcast
// likelihoods over vector observations, joint-of-distributions priors, mixture
// normalizers) repeatedly hard-failed in a single backend or a single shape; a
// new shape bug should fail HERE for the whole matrix, not be discovered in the
// viewer. (Numeric correctness is gated by the dedicated oracle tests; this gate
// is structural robustness only.)
//
// Models = every fixture in test/fixtures/baseline (auto-discovered) + a few
// inline "shape coverage" models that exposed past bugs (joint-of-distributions
// prior, Gaussian-mixture normalize likelihood, hierarchical-logistic with an
// iid-of-broadcast [L,D] matrix latent + Bernoulli vector observation).

const { test }   = require('node:test');
const assert     = require('node:assert/strict');
const fs         = require('node:fs');
const path       = require('node:path');
const { ctxFor } = require('./density/regression-baseline.test.ts');

const BACKENDS = ['is', 'mh', 'emcee', 'amis', 'smc', 'elliptical-slice-sampler'];

const FIX = path.join(__dirname, 'fixtures/baseline');
const fixtureModels = fs.readdirSync(FIX)
  .filter((f: string) => f.endsWith('.flatppl'))
  .map((f: string) => ({ name: f.replace('.flatppl', ''), src: fs.readFileSync(path.join(FIX, f), 'utf8') }));

// Inline shape-coverage models (past regression shapes not in the fixture set).
const inlineModels = [
  { name: 'joint-of-distributions-prior', src: `
theta1_dist = Normal(0, 1)
theta2_dist = Exponential(1)
prior = joint(theta1 = theta1_dist, theta2 = theta2_dist)
theta1 = elementof(reals)
theta2 = elementof(reals)
obs ~ iid(Normal(mu = 5 * theta2, sigma = abs(theta1) * theta2), 10)
forward_kernel = kernelof(record(obs = obs))
L = likelihoodof(forward_kernel, record(obs = [1.2,3.4,5.1,2.8,4.0,3.7,5.5,2.1,4.3,3.9]))
posterior = bayesupdate(L, prior)
` },
  { name: 'gaussian-mixture-normalize', src: `
theta ~ Beta(1, 1)
mu ~ iid(Normal(0, 5), 2)
mix = normalize(superpose(weighted(theta, Normal(mu[1], 1)), weighted(1 - theta, Normal(mu[2], 1))))
y ~ iid(mix, 20)
prior = lawof(record(theta = theta, mu = mu))
forward_kernel = kernelof(record(y = y), theta = theta, mu = mu)
L = likelihoodof(forward_kernel, record(y = [1.3,2.2,-1.8,2.3,2.5,-1.5,1.9,1.5,-1.3,2.2,1.3,-3.3,0.6,1.4,0.4,-0.8,-0.5,1.5,1.7,1.3]))
posterior = bayesupdate(L, prior)
` },
  { name: 'hier-logistic-matrix-latent', src: `
x_data = [[1.0,0.5],[1.0,-1.2],[1.0,0.3],[1.0,1.8],[1.0,-0.7],[1.0,2.1],[1.0,0.9],[1.0,-1.5],[1.0,0.2]]
ll_data = [1,1,1,2,2,2,3,3,3]
y_data  = [1,0,0,1,0,1,1,0,0]
mu ~ iid(Normal(0, 100), 2)
sigma ~ iid(Exponential(1), 2)
beta ~ iid(Normal.(mu, sigma), 3)
prior = lawof(record(mu = mu, sigma = sigma, beta = beta))
logit_p = (x_row, group) -> invlogit(sum(x_row .* get(beta, group)))
p = logit_p.(x_data, ll_data)
y ~ Bernoulli.(p)
forward_kernel = kernelof(record(y = y), mu = mu, sigma = sigma, beta = beta)
L = likelihoodof(forward_kernel, record(y = y_data))
posterior = bayesupdate(L, prior)
` },
];

const MODELS = fixtureModels.concat(inlineModels);

for (const model of MODELS) {
  for (const backend of BACKENDS) {
    test(`matrix: ${model.name} × ${backend}`, async () => {
      const { ctx } = ctxFor(model.src, 120);
      ctx.inferenceOpts = {
        backend, chains: 8, warmup: 80, draws: 120,
        smcParticles: 400, smcSteps: 6, amisSamples: 300, amisIters: 3, seed: 1,
      };
      const m = await ctx.getMeasure('posterior');
      assert.ok(m && m.fields && Object.keys(m.fields).length > 0,
        `${model.name} × ${backend}: posterior has no fields`);
      for (const fn of Object.keys(m.fields)) {
        const f = m.fields[fn];
        const s = (f && f.samples) || (f && f.value && f.value.data);
        assert.ok(s && s.length > 0, `${model.name} × ${backend}: field '${fn}' carries no samples`);
      }
    });
  }
}
