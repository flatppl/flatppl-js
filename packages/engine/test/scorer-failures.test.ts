'use strict';

// An evaluation that throws is not a zero density and not a missing value.
// Each scorer below once turned such a throw into -Infinity, NaN, an empty
// check, or a dropped coordinate, with no message. Each must now report the
// cause. A density that is truly zero stays -Infinity.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi, evaluateDerivedBindings, derivedFieldFailure } = require('../mcmc-density.ts');
const { buildModelViewFromCtx } = require('../model-view.ts');
const modelSpec = require('../model-spec.ts');
const pp = require('../posterior-predictive.ts');
const { materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const OUT_OF_BOUNDS = 'evaluateExpr: get index 3 out of bounds for length 2';
const logN = (x: number, m: number, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

// `w` throws at every point. The likelihood reads it.
const THROWING_LIK = `
b = elementof(reals)
w = [b, b][3]
y ~ Normal(w, 1)
K = kernelof(y, b = b)
L = likelihoodof(K, 0.5)
posterior = bayesupdate(L, joint(b = Normal(0, 1)))
`;

test('a likelihood that throws is an error, not a flat -Infinity target', async () => {
  const { ctx } = ctxFor(THROWING_LIK, 1);
  const s = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(s.priorOf({ b: 0.3 }) - logN(0.3, 0)) < 1e-12);
  const want = `the likelihood failed to evaluate: ${OUT_OF_BOUNDS}`;
  assert.throws(() => s.likOf({ b: 0.3 }), { message: want });
  assert.throws(() => s.logPi({ b: 0.3 }), { message: want });
  assert.throws(() => s.priorLikBatch([{ b: 0.3 }]), { message: want });
});

test('the legacy posterior spec reports a likelihood throw', () => {
  const { ctx } = ctxFor(THROWING_LIK, 1);
  const spec = modelSpec.buildPosteriorSpec(ctx.derivations.posterior, ctx);
  assert.throws(() => spec.logLikelihood({ b: 0.3 }),
    { message: `the likelihood failed to evaluate: ${OUT_OF_BOUNDS}` });
});

// Oracle: Exponential(1) at bnd = 1 is -1, and Uniform(0, 1) at 0.5 is 0.
// Outside [0, bnd] the Uniform density is zero, so the prior is -Infinity.
test('a draw outside a stochastic support stays -Infinity on a real chart', async () => {
  const src = `
bnd ~ Exponential(1.0)
x ~ Uniform(interval(0.0, bnd))
prior = lawof(record(bnd = bnd, x = x))
y ~ Normal(x, 1.0)
K = kernelof(record(y = y), bnd = bnd, x = x)
L = likelihoodof(K, record(y = 0.5))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 64);
  const latents = modelSpec.enumerateLatents(ctx.derivations.posterior, ctx);
  assert.deepEqual(latents.find((l: any) => l.name === 'x').support, { kind: 'real' });
  const s = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(s.priorOf({ bnd: 1, x: 0.5 }) - -1) < 1e-12);
  assert.equal(s.priorOf({ bnd: 1, x: 1.5 }), -Infinity);
  assert.equal(s.priorOf({ bnd: 1, x: -0.5 }), -Infinity);
  assert.equal(s.logPi({ bnd: 1, x: 1.5 }), -Infinity);
});

test('a derived binding that throws names itself to a posterior-field reader', () => {
  const { ctx } = ctxFor(`flatppl_compat = "0.1"
mu ~ Normal(0.0, 1.0)
v = [mu, mu]
w = v[3]
`, 1);
  const env = evaluateDerivedBindings(ctx, { mu: 0.3 });
  assert.deepEqual(Array.from(env.v.data), [0.3, 0.3]);
  assert.ok(!('w' in env));
  assert.equal(derivedFieldFailure(env, 'w'), `binding 'w' failed to evaluate: ${OUT_OF_BOUNDS}`);
});

// A composite latent has no static width. Without a prior draw it once kept
// width 0 and left the sampler with no coordinate for it.
test('a composite latent whose prior draw fails names the latent and the cause', async () => {
  const { ctx } = ctxFor(`
N = 3
a = [2.0, 3.0]
b = [5.0, 4.0]
beta_row_K = (a_g, b_g) -> iid(Beta(a_g, b_g), N)
p ~ beta_row_K.(a, b)
binomial_row_K = (n_row, p_row) -> Binomial.(n_row, p_row)
r ~ binomial_row_K.([[10, 10, 10], [10, 10, 10]], p)
prior = lawof(record(p = p))
forward_kernel = kernelof(record(r = r), p = p)
L = likelihoodof(forward_kernel, record(r = [[7, 8, 6], [5, 6, 4]]))
posterior = bayesupdate(L, prior)
`, 50);
  const getMeasure = ctx.getMeasure;
  ctx.getMeasure = (n: string) => (n === 'p' || n === 'prior')
    ? Promise.reject(new Error('sampler down')) : getMeasure(n);
  await assert.rejects(() => buildModelViewFromCtx(ctx, ctx.derivations.posterior), {
    message: "model-view: latent 'p' takes its shape from a prior draw, and the prior could not be drawn: sampler down",
  });
});

test('a posterior predictive field that fails to draw names the field', async (t: any) => {
  const { ctx } = ctxFor(`
mu ~ Normal(0.0, 2.0)
prior = lawof(record(mu = mu))
y ~ iid(Normal(mu, 1.0), 2)
forward_kernel = kernelof(record(y = y), mu = mu)
L = likelihoodof(forward_kernel, record(y = [1.0, 2.0]))
posterior = bayesupdate(L, prior)
`, 200);
  const posterior = await ctx.getMeasure('posterior');
  t.mock.method(materialiser, 'materialiseMeasureIR', () => Promise.reject(new Error('sampler down')));
  t.mock.method(materialiser, 'materialiseMeasure', () => Promise.reject(new Error('sampler down')));
  await assert.rejects(() => pp.buildPosteriorPredictive(ctx.derivations.posterior, ctx, posterior),
    { message: "posterior predictive: field 'y' failed to draw: sampler down" });
});

test('a profile with no defined point reports the cause', () => {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const ir = {
    kind: 'call', op: 'get', args: [
      { kind: 'call', op: 'vector', args: [{ kind: 'ref', ns: 'self', name: 'x' }, { kind: 'ref', ns: 'self', name: 'x' }] },
      { kind: 'lit', value: 3, numType: 'integer' },
    ],
  };
  const r = w.handle({ type: 'profileN', ir, sweepName: 'x', range: [0, 1], count: 3, mode: 'function' });
  assert.equal(r.type, 'error');
  assert.match(r.message, /^profile over 'x' failed at every point: .*out of bounds/);
});
