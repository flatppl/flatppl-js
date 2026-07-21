'use strict';
// DEMCz quality gate on a strongly-correlated bivariate Gaussian.
// x ~ N(0,1);  y ~ N(0.95 x, sqrt(1-0.95^2)) ⇒ joint N(0, [[1,.95],[.95,1]]).
// Likelihood flat in (x,y) (obs_dist independent of the latents), so the
// posterior = prior: means 0, vars 1, cov 0.95 (closed-form oracle).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

// sqrt(1 - 0.95^2) = 0.31224989991991997
const MODEL = `
x = draw(Normal(mu = 0.0, sigma = 1.0))
y = draw(Normal(mu = 0.95 * x, sigma = 0.31224989991991997))
prior = lawof(record(x = x, y = y))
obs_dist = joint(z = Normal(mu = 0.0, sigma = 1.0))
K = functionof(obs_dist, x = x, y = y)
L = likelihoodof(K, record(z = 0.0))
posterior = bayesupdate(L, prior)
`;

const mean = (a: Float64Array) => { let s = 0; for (const v of a) s += v; return s / a.length; };
const cov = (a: Float64Array, b: Float64Array) => {
  const ma = mean(a), mb = mean(b); let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - ma) * (b[i] - mb);
  return s / (a.length - 1);
};

const OPTS = { warmup: 200, draws: 3000, seed: 1 };

test('backend:demcz recovers the correlated-Gaussian moments (closed form)', async () => {
  const { proc, ctx } = ctxFor(MODEL, 3000);
  const errs = (proc.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, `parse errors: ${errs.map((e: any) => e.message).join('; ')}`);
  const m = await materialiser.materialiseMeasure('posterior', ctx, Object.assign({ backend: 'demcz', walkers: 8 }, OPTS));
  const x = m.fields.x.samples, y = m.fields.y.samples;
  assert.ok(Math.abs(mean(x)) < 0.1, `mean x ${mean(x)} vs 0`);
  assert.ok(Math.abs(mean(y)) < 0.1, `mean y ${mean(y)} vs 0`);
  assert.ok(Math.abs(cov(x, x) - 1) < 0.15, `var x ${cov(x, x)} vs 1`);
  assert.ok(Math.abs(cov(y, y) - 1) < 0.15, `var y ${cov(y, y)} vs 1`);
  assert.ok(Math.abs(cov(x, y) - 0.95) < 0.15, `cov xy ${cov(x, y)} vs 0.95`);
});

test('backend:demcz ESS exceeds mh at equal short-warmup budget, and agrees with mh on the mean', async () => {
  const runEss = async (backend: string) => {
    const { ctx } = ctxFor(MODEL, 3000);
    const m = await materialiser.materialiseMeasure('posterior', ctx, Object.assign({ backend, chains: 8, walkers: 8 }, OPTS));
    return { m, essX: m.diagnostics.perParam.x.essBulk, essY: m.diagnostics.perParam.y.essBulk };
  };
  const dz = await runEss('demcz');
  const mh = await runEss('mh');
  // Headline value proposition: covariance-adapting DEMCz mixes the correlated
  // ridge better than diagonal-start mh before mh's warmup covariance matures.
  assert.ok(dz.essX > mh.essX, `demcz essBulk x ${dz.essX} should exceed mh ${mh.essX}`);
  assert.ok(dz.essY > mh.essY, `demcz essBulk y ${dz.essY} should exceed mh ${mh.essY}`);
  // Cross-check (disagreement detector, not the oracle): same posterior mean.
  assert.ok(Math.abs(mean(dz.m.fields.x.samples) - mean(mh.m.fields.x.samples)) < 0.15, 'demcz vs mh mean x disagree');
});
