'use strict';
// ess-e2e.test.ts — elliptical slice (backend 'elliptical-slice-sampler') through
// materialiseMeasure: the FITTED path on hierarchical/funnel models, and the
// EXACT path on an all-Normal-prior model (detectGaussianPrior fires).

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor }       = require('./density/regression-baseline.test.ts');
const { materialiser } = require('..');

const FIX = path.join(__dirname, 'fixtures/baseline');
function mean(m: any) { const s = m.samples || (m.value && m.value.data); let x = 0; for (let i = 0; i < s.length; i++) x += s[i]; return x / s.length; }

test('eight-schools: elliptical slice (fitted) recovers mu', async () => {
  const src = fs.readFileSync(path.join(FIX, 'eight-schools.flatppl'), 'utf8');
  const m = await materialiser.materialiseMeasure('posterior', ctxFor(src, 1000).ctx,
    { backend: 'elliptical-slice-sampler', chains: 4, warmup: 400, draws: 400, seed: 1 });
  assert.equal(m.diagnostics.method, 'ess-slice');
  assert.equal(m.diagnostics.mode, 'fitted');
  assert.ok(Number.isFinite(m.diagnostics.meanShrinks) && m.diagnostics.meanShrinks >= 1, 'meanShrinks ≥ 1');
  const mu = mean(m.fields.mu);
  assert.ok(mu > 1 && mu < 8, `mu ${mu} ≈ 4.4`);
  assert.ok(mean(m.fields.tau) > 0, 'tau positive');
});

test('partial-pooling: elliptical slice recovers phi + kappa (where AMIS collapsed)', async () => {
  const src = fs.readFileSync(path.join(FIX, 'partial-pooling.flatppl'), 'utf8');
  const m = await materialiser.materialiseMeasure('posterior', ctxFor(src, 500).ctx,
    { backend: 'elliptical-slice-sampler', chains: 4, warmup: 300, draws: 300, seed: 1 });
  const phi = mean(m.fields.phi), kappa = mean(m.fields.kappa);
  assert.ok(phi > 0.25 && phi < 0.37, `phi ${phi} ≈ 0.31`);
  assert.ok(kappa > 30, `kappa ${kappa} not collapsed`);
});

// All-Normal-prior model with a fixed likelihood scale ⇒ the unconstrained prior
// is exactly N(0, diag) ⇒ detectGaussianPrior fires ⇒ mode 'exact'.
const EXACT_MODEL = `
x_data = [1.1, 1.5, 1.3, 1.4, 0.9, 2.0]
y_data = [3.2, 4.1, 3.4, 3.9, 2.8, 5.1]

alpha ~ Normal(0, 5)
beta ~ Normal(0, 5)

prior = lawof(record(alpha = alpha, beta = beta))

means = alpha .+ beta .* x_data

y ~ Normal.(means, 0.5)

forward_kernel = kernelof(record(y = y), alpha = alpha, beta = beta)

L = likelihoodof(forward_kernel, record(y = y_data))

posterior = bayesupdate(L, prior)
`;

test('all-Normal-prior model: elliptical slice uses the EXACT Gaussian reference', async () => {
  const m = await materialiser.materialiseMeasure('posterior', ctxFor(EXACT_MODEL, 1000).ctx,
    { backend: 'elliptical-slice-sampler', chains: 4, warmup: 400, draws: 400, seed: 1 });
  assert.equal(m.diagnostics.method, 'ess-slice');
  assert.equal(m.diagnostics.mode, 'exact', 'all-Normal prior ⇒ exact reference');
  // Regression of y on x: OLS slope ≈ 2.0 for this data.
  const a = mean(m.fields.alpha), b = mean(m.fields.beta);
  assert.ok(Number.isFinite(a) && Number.isFinite(b), 'finite posterior means');
  assert.ok(b > 1.4 && b < 2.6, `beta ${b} ≈ 2.0 (positive slope recovered)`);
});
