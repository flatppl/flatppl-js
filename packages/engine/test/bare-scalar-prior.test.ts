'use strict';

// Spec §06 `likelihoodof`: the parameter point is a record with one field per input of the kernel,
// "so a kernel with the single input `mu` is evaluated at `record(mu = 1.0)`, not at `1.0`". The
// variate of a prior in `bayesupdate` is that point, so a prior written as a bare measure
// (`prior = Normal(0, 1)`, `prior = Lebesgue(posreals)`) is a static error naming the record
// spelling, and the sampler's setup names it too instead of "no stochastic draws to sample". The
// valid spelling `joint(theta = ...)` has no draws, and must work in the model view and the MCMC
// density.
//
// ORACLES, closed form. A Normal(0, 1) prior and y | b ~ Normal(b, 1) observed at 0.5 give the
// unnormalised log posterior logN(b; 0, 1) + logN(0.5; b, 1) and the posterior Normal(0.25, sqrt(0.5)).
// A Lebesgue(posreals) prior and y = 0 give logN(0; b, 1) with the log-Jacobian log(b) of the positive
// chart, and the half-Normal posterior of mean sqrt(2 / pi) and variance 1 - 2 / pi.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser, processSource } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const logN = (x: number, m: number, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
const errors = (src: string) => processSource(src).diagnostics.filter((d: any) => d.severity === 'error');

const NORMAL = (prior: string) => `
b = elementof(reals)
prior = ${prior}
y ~ Normal(b, 1)
K = kernelof(y, b = b)
L = likelihoodof(K, 0.5)
posterior = bayesupdate(L, prior)
`;
const LEBESGUE = (prior: string) => `
b = elementof(posreals)
prior = ${prior}
y ~ Normal(b, 1)
K = kernelof(y, b = b)
L = likelihoodof(K, 0)
posterior = bayesupdate(L, prior)
`;

test('a bare scalar prior is a located static error naming the record spelling', () => {
  for (const [src, variate] of [[NORMAL('Normal(0, 1)'), 'real'], [LEBESGUE('Lebesgue(posreals)'), 'real']]) {
    const errs = errors(src);
    assert.equal(errs.length, 1, JSON.stringify(errs.map((e: any) => e.message)));
    assert.match(errs[0].message, new RegExp(
      `^bayesupdate: the prior's variate is ${variate}, but the parameter point of a likelihood is a record `
      + 'with one field per input \\("b"\\)'));
    assert.match(errs[0].message, /even a kernel with one input is evaluated at record\(b = \.\.\)/);
    assert.match(errs[0].message, /`joint\(b = \.\.\.\)`/);
    assert.equal(errs[0].loc.start.line, 6);
  }
});

test('the sampler setup names the record spelling when the diagnostics were not consulted', async () => {
  const { ctx } = ctxFor(NORMAL('Normal(0, 1)'), 1);
  await assert.rejects(() => buildLogPi(ctx, ctx.derivations.posterior),
    /^Error: buildLogPi: the prior's variate is real, but the parameter point of a likelihood is a record .*joint\(b = \.\.\.\)/);
});

test('a joint(b = Normal(0, 1)) prior scores the closed-form log posterior', async () => {
  const src = NORMAL('joint(b = Normal(0, 1))');
  assert.deepEqual(errors(src), []);
  const { ctx } = ctxFor(src, 1);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  assert.equal(view.dim, 1);
  assert.deepEqual(view.names, ['b']);
  for (const b of [-0.7, 0.3, 1.9]) {
    const got = view.logPosterior(view.unconstrainAll({ b }));
    assert.ok(Math.abs(got - (logN(b, 0) + logN(0.5, b))) < 1e-12, `b = ${b}: ${got}`);
  }
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ b: 0.3 }) - logN(0.5, 0.3)) < 1e-12);
  assert.ok(Math.abs(scorer.priorOf({ b: 0.3 }) - logN(0.3, 0)) < 1e-12);
});

test('MCMC over a joint(b = Normal(0, 1)) prior recovers the conjugate posterior', async () => {
  const { ctx } = ctxFor(NORMAL('joint(b = Normal(0, 1))'), 4000);
  const m = await materialiser.materialiseMeasure('posterior', ctx,
    { backend: 'mh', chains: 4, warmup: 500, draws: 1000, seed: 3 });
  const x = Array.from(m.fields.b.samples) as number[];
  const mean = x.reduce((a, v) => a + v, 0) / x.length;
  const variance = x.reduce((a, v) => a + (v - mean) ** 2, 0) / x.length;
  assert.ok(Math.abs(mean - 0.25) < 0.08, `mean ${mean}`);
  assert.ok(Math.abs(variance - 0.5) < 0.1, `variance ${variance}`);
});

test('a joint(b = Lebesgue(posreals)) prior has the chart Jacobian and the half-Normal posterior', async () => {
  const src = LEBESGUE('joint(b = Lebesgue(posreals))');
  assert.deepEqual(errors(src), []);
  const { ctx } = ctxFor(src, 4000);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  assert.equal(view.dim, 1);
  for (const b of [0.5, 1, 2.5]) {
    const got = view.logPosterior(view.unconstrainAll({ b }));
    assert.ok(Math.abs(got - (logN(0, b) + Math.log(b))) < 1e-12, `b = ${b}: ${got}`);
  }
  const m = await materialiser.materialiseMeasure('posterior', ctx,
    { backend: 'mh', chains: 4, warmup: 500, draws: 2000, seed: 5 });
  const x = Array.from(m.fields.b.samples) as number[];
  assert.ok(x.every(v => v > 0));
  const mean = x.reduce((a, v) => a + v, 0) / x.length;
  const variance = x.reduce((a, v) => a + (v - mean) ** 2, 0) / x.length;
  assert.ok(Math.abs(mean - Math.sqrt(2 / Math.PI)) < 0.08, `half-Normal mean ${mean}`);
  assert.ok(Math.abs(variance - (1 - 2 / Math.PI)) < 0.08, `half-Normal variance ${variance}`);
});
