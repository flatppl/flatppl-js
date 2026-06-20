const { test } = require('node:test');
const assert = require('node:assert/strict');
const MV = require('../model-view.ts');

// Model: mu ~ Normal(0, 10); y ~ Normal(mu, 1) observed at 3.0. One latent (mu, real support).
// logπ(mu) = logNormal(mu;0,10) + logNormal(3; mu,1)  (+ 0 Jacobian, real support).
function logNormal(x: number, m: number, s: number) {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
}

test('logPosterior matches analytic unnormalised posterior (Normal-Normal, real support)', () => {
  const spec = {
    latents: [{ name: 'mu', distOp: 'Normal', params: { mu: 0, sigma: 10 }, support: { kind: 'real' }, discrete: false }],
    logLikelihood: (th: any) => logNormal(3.0, th.mu, 1),
  };
  const mv = MV.buildModelView(spec);
  assert.equal(mv.dim, 1);
  assert.deepEqual(mv.names, ['mu']);
  for (const mu of [-2, 0, 1.5, 4]) {
    const y = Float64Array.from([mu]);            // real support: y == theta
    const want = logNormal(mu, 0, 10) + logNormal(3.0, mu, 1);
    assert.ok(Math.abs(mv.logPosterior(y) - want) < 1e-9, `mu=${mu}`);
  }
});

test('logPosterior includes the Jacobian for a positive-support latent', () => {
  // sigma ~ Exponential(rate=1) on (0,inf); no likelihood. y = log sigma.
  const spec = {
    latents: [{ name: 'sigma', distOp: 'Exponential', params: { rate: 1 }, support: { kind: 'positive' }, discrete: false }],
    logLikelihood: () => 0,
  };
  const mv = MV.buildModelView(spec);
  for (const sigma of [0.1, 1, 5]) {
    const y = Float64Array.from([Math.log(sigma)]);
    const wantPrior = Math.log(1) - 1 * sigma;       // Exponential(1) logpdf at sigma
    const wantJac = Math.log(sigma);                 // logDetJ = y = log sigma
    assert.ok(Math.abs(mv.logPosterior(y) - (wantPrior + wantJac)) < 1e-9, `sigma=${sigma}`);
  }
});
