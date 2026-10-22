'use strict';

// `buildLogPi` binds a whole-record adapter's point from the prior's record
// fields (mcmc-density.ts). The prior's type comes from its binding, so an
// inline prior, which the lift hoists to a synthetic binding, needs that
// binding typed. Without it the point fell back to a bare draw and the
// adapter's `pars.theta` projected out of a scalar: logPi was -Infinity with
// no diagnostic.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const MODEL = `
theta ~ Uniform(unitinterval)
y ~ Binomial(10, theta)
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(pars -> K(theta = pars.theta), record(y = 3))
`;
// Binomial(10, 0.3) pmf at 3 under the Uniform(unitinterval) prior density 1.
const CLOSED_FORM = Math.log(120 * 0.3 ** 3 * 0.7 ** 7);

async function logPi(prior: string, point: any) {
  const src = MODEL + prior;
  assert.deepEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.post);
  return { logPi: scorer.logPi(point), lik: scorer.likOf(point) };
}

test('a named record prior scores the closed form through a whole-record adapter', async () => {
  const r = await logPi('prior = lawof(record(theta = theta))\npost = bayesupdate(L, prior)\n', { theta: 0.3 });
  assert.ok(Math.abs(r.logPi - CLOSED_FORM) < 1e-12);
  assert.ok(Math.abs(r.lik - CLOSED_FORM) < 1e-12);
});

test('an inline record prior scores the closed form through a whole-record adapter', async () => {
  const r = await logPi('post = bayesupdate(L, lawof(record(theta = theta)))\n', { theta: 0.3 });
  assert.ok(Math.abs(r.logPi - CLOSED_FORM) < 1e-12);
  assert.ok(Math.abs(r.lik - CLOSED_FORM) < 1e-12);
});

test('an inline prior over several fields keeps every field in the point', async () => {
  const src = `
a ~ Normal(0, 1)
b ~ Normal(0, 1)
y ~ Normal(a + b, 1)
K = kernelof(record(y = y), a = a, b = b)
L = likelihoodof(pars -> K(a = pars.a, b = pars.b), record(y = 0.0))
post = bayesupdate(L, lawof(record(a = a, b = b)))
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.post);
  const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - x * x / 2;
  const expected = logN(0.5) + logN(0.25) + logN(0.75);
  assert.ok(Math.abs(scorer.logPi({ a: 0.5, b: 0.25 }) - expected) < 1e-12);
});

test('an untyped prior is refused instead of guessing the point shape', async () => {
  const { ctx } = ctxFor(MODEL + 'prior = lawof(record(theta = theta))\npost = bayesupdate(L, prior)\n', 4);
  const name = ctx.derivations.post.from;
  ctx.bindings.set(name, { ...ctx.bindings.get(name), inferredType: undefined });
  await assert.rejects(() => buildLogPi(ctx, ctx.derivations.post),
    new RegExp(`the type of prior '${name}' is unknown, so the likelihood point cannot be bound`));
});
