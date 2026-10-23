'use strict';
// nested-backend-e2e.test.ts — end-to-end nested-sampling evidence through
// materialiseMeasure on a 1-D Gaussian-conjugate model with a closed-form
// oracle for logZ.

const { test } = require('node:test');
const assert   = require('node:assert/strict');

const { ctxFor }       = require('./_ctx-factory.ts');
const { materialiser } = require('..');

test('nested backend: 1-D Gaussian evidence end-to-end', async () => {
  // theta ~ Normal(0,1); y ~ Normal(theta, 2); observe y=0.
  // Z = ∫ N(θ;0,1) N(0;θ,2) dθ = N(0;0,√5) = 1/√(2π·5).  logZ = −½ log(2π·5).
  const src = `
flatppl_compat = "0.1"
theta ~ Normal(0.0, 1.0)
prior = lawof(record(theta = theta))
y ~ iid(Normal(theta, 2.0), 1)
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = [0.0]))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 100);
  const m = await materialiser.materialiseMeasure('posterior', ctx,
    { backend: 'nested', nLive: 400, dlogz: 0.01, seed: 7 });
  const logZexact = -0.5 * Math.log(2 * Math.PI * 5);
  assert.equal(m.diagnostics.method, 'nested');
  assert.ok(Math.abs(m.logTotalmass - logZexact) < 0.1, `logZ ${m.logTotalmass} vs ${logZexact}`);
});
