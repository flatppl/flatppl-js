const { test } = require('node:test');
const assert = require('node:assert/strict');
const MV = require('../model-view.ts');
const { ctxFor } = require('./density/regression-baseline.test.ts');

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

// Regression (#73 / Buffy #267): a VECTOR iid latent with a positive-support
// (non-identity, log) element transform scores correctly through the ctx-built
// ModelView. The scorer-format API is keyed by latent NAME with the whole vector
// value ({ B: Float64Array([...]) }) — NOT the flat per-coordinate `mv.names`
// keys ({ 'B[0]': v }), which are only for the unconstrained Float64Array that
// logPosterior(y) consumes. Feeding flat keys makes scorerPt['B'] undefined ->
// +undefined = NaN -> the posreals unconstrain clamps to log(1e-300) = -690.77
// on every coord and logPi is -Inf; that was a harness misuse, not an engine bug.
test('buildModelViewFromCtx: vector iid LogNormal latent scores vs oracle (scorer-format API)', async () => {
  const src = `
flatppl_compat = "0.1"
B ~ iid(LogNormal(0.0, 0.5), 5)
prior = lawof(record(B = B))
obs ~ Normal.(B, 1.0)
fwd = kernelof(record(obs = obs), B = B)
L = likelihoodof(fwd, record(obs = [0.9, 1.0, 1.1, 0.95, 1.05]))
posterior = bayesupdate(L, prior)
`;
  const ctx = ctxFor(src, 1).ctx;
  const mv = await MV.buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  assert.deepEqual(mv.names, ['B[0]', 'B[1]', 'B[2]', 'B[3]', 'B[4]']);

  const Bpt = [0.9, 1.0, 1.1, 0.95, 1.05];
  const scorerPt = { B: Float64Array.from(Bpt) };

  // unconstrainAll maps each coord through the element's log transform (posreals).
  const y = mv.unconstrainAll(scorerPt);
  Bpt.forEach((b, i) => assert.ok(Math.abs(y[i] - Math.log(b)) < 1e-12, `unconstrain B[${i}]`));

  // logPosteriorConstrained == sum LogNormal(0,0.5).logpdf(B[i]) + sum Normal(B[i],1).logpdf(obs[i]).
  // scipy oracle (obs == B): prior -1.1667959987576646, lik -4.594692666023363.
  const ORACLE = -5.7614886647810275;
  const lpc = mv.logPosteriorConstrained(scorerPt);
  assert.ok(Number.isFinite(lpc), `logPosteriorConstrained finite (got ${lpc})`);
  assert.ok(Math.abs(lpc - ORACLE) < 1e-9, `logPosteriorConstrained ${lpc} vs oracle ${ORACLE}`);
});
