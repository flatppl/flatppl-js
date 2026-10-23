'use strict';

// A kernel body that draws a latent the prior does not carry (`mu` below) has a
// likelihood that integrates that latent out (spec §06 "Density of composed
// measures"). For a linear-Gaussian latent that integral is exact, and its mean
// is affine in the point the likelihood is scored at. The MCMC scorer takes it
// in closed form, as `logdensityof` does at a fed boundary. A latent with no
// exact marginal is refused when the scorer is built.
//
// Before, the latent stayed unbound in the scoring env, `likWith` swallowed the
// unbound-reference error to -Infinity, and an MH run failed at initialisation
// with "supply finite initPositions", which names the wrong cause.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const materialiser = require('../materialiser.ts');

const logN = (x: number, m: number, s: number) =>
  -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

const LATENT = `
theta ~ Normal(0, 1)
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), theta = theta)
`;
// pdf of Normal(theta, 1) convolved with Normal(0, 1): y | theta ~ Normal(theta, sqrt(2)).
const closedForm = (theta: number, y = 0.5) => logN(y, theta, Math.SQRT2);

const FORMS: Record<string, { tail: string; point: string }> = {
  direct: {
    tail: 'L = likelihoodof(K, record(y = 0.5))\n'
      + 'prior = lawof(record(theta = theta))\nposterior = bayesupdate(L, prior)\n',
    point: 'record(theta = 0.3)',
  },
};

for (const [form, { tail, point }] of Object.entries(FORMS)) {
  test(`${form}: logdensityof scores the latent-closure likelihood at the fed point`, async () => {
    const src = LATENT + tail + `s = logdensityof(L, ${point})\n`;
    assert.deepEqual(processSource(src).diagnostics.filter((d: any) => d.severity === 'error'), []);
    const { ctx } = ctxFor(src, 4);
    const v = (await ctx.getMeasure('s')).samples[0];
    assert.ok(Math.abs(v - closedForm(0.3)) < 1e-12, `got ${v}, want ${closedForm(0.3)}`);
  });

  test(`${form}: buildLogPi scores the closed form at each point, scalar and batched`, async () => {
    const { ctx } = ctxFor(LATENT + tail, 4);
    const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
    const thetas = [0.3, -1.2, 2.5];
    for (const th of thetas) {
      assert.ok(Math.abs(scorer.likOf({ theta: th }) - closedForm(th)) < 1e-12, `likOf(${th})`);
      assert.ok(Math.abs(scorer.logPi({ theta: th }) - (closedForm(th) + logN(th, 0, 1))) < 1e-12,
        `logPi(${th})`);
    }
    const { lik } = scorer.priorLikBatch(thetas.map((theta) => ({ theta })));
    thetas.forEach((th, i) => assert.ok(Math.abs(lik[i] - closedForm(th)) < 1e-12, `batch ${th}`));
  });
}

test('an MH run over the latent-closure model recovers the closed-form posterior', async () => {
  // Prior Normal(0, 1), likelihood Normal(y = 0.5; theta, sqrt(2)): the posterior
  // is Normal(0.5 / 3, sqrt(2 / 3)).
  const { ctx } = ctxFor(LATENT + FORMS.direct.tail, 400);
  const m = await materialiser.materialiseMeasure('posterior', ctx,
    { backend: 'mh', chains: 4, warmup: 500, draws: 1000, seed: 3 });
  const draws: Float64Array = m.fields?.theta?.samples ?? m.samples;
  let mean = 0;
  for (const x of draws) mean += x / draws.length;
  assert.ok(Math.abs(mean - 0.5 / 3) < 0.12, `posterior mean ${mean}, want ${0.5 / 3}`);
});

test('a discrete latent is enumerated with a mean that depends on the point', async () => {
  const src = `
theta ~ Normal(0, 1)
k ~ Bernoulli(0.3)
y ~ Normal(theta + k, 1)
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = 0.5))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  for (const th of [0.3, -0.7]) {
    const want = Math.log(0.7 * Math.exp(logN(0.5, th, 1)) + 0.3 * Math.exp(logN(0.5, th + 1, 1)));
    assert.ok(Math.abs(scorer.likOf({ theta: th }) - want) < 1e-12, `theta ${th}`);
  }
});

test('a latent the prior carries scores the closed form', async () => {
  const src = `
mu ~ Normal(0, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), mu = mu)
L = likelihoodof(K, record(y = 0.5))
prior = lawof(record(mu = mu))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ mu: 0.3 }) - logN(0.5, 0.3, 1)) < 1e-12);
});

test('a draw independent of the boundary is a variate, not a latent, and still scores', async () => {
  const src = `
theta ~ Normal(0, 1)
z ~ Normal(1, 2)
w ~ Normal(theta, 1)
K = kernelof(record(w = w, z = z), theta = theta)
L = likelihoodof(K, record(w = 0.5, z = 0.7))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ theta: 0.3 }) - (logN(0.5, 0.3, 1) + logN(0.7, 1, 2))) < 1e-12);
});

// A latent with no exact marginal is refused when the scorer is built, naming
// the latent and the reason, instead of scoring -Infinity.
const NON_GAUSSIAN = `
theta ~ Normal(0, 1)
mu ~ Exponential(1)
y ~ Normal(mu + theta, 1)
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = 0.5))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;

test('buildLogPi refuses a latent with no exact marginal, naming it and the reason', async () => {
  assert.deepEqual(processSource(NON_GAUSSIAN).diagnostics.filter((d: any) => d.severity === 'error'), []);
  const { ctx } = ctxFor(NON_GAUSSIAN, 4);
  await assert.rejects(() => buildLogPi(ctx, ctx.derivations.posterior), (e: any) => {
    assert.match(e.message, /the likelihood marginalises the stochastic ancestor\(s\) mu\b/);
    assert.match(e.message, /not a Normal/);
    assert.match(e.message, /spec §06/);
    assert.equal(e.modelRefusal, true);
    return true;
  });
});

test('an MH run over such a model refuses with the latent, not with an initialisation failure', async () => {
  const { ctx } = ctxFor(NON_GAUSSIAN, 200);
  await assert.rejects(
    () => materialiser.materialiseMeasure('posterior', ctx,
      { backend: 'mh', chains: 2, warmup: 50, draws: 50, seed: 1 }),
    (e: any) => {
      assert.match(e.message, /marginalises the stochastic ancestor\(s\) mu\b/);
      assert.doesNotMatch(e.message, /initPositions/);
      return true;
    });
});

// A latent the kernel exposes as a field of its record is an observed variate:
// the density walker threads its observed value, so there is nothing to
// integrate out and nothing to refuse. (The first scorer check wrongly refused it.)
test('a latent exposed as an observed field of the kernel is a variate, not a marginal', async () => {
  const src = `
theta ~ Normal(0, 1)
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(record(mu = mu, y = y), theta = theta)
L = likelihoodof(K, record(mu = 0.2, y = 0.5))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ theta: 0.3 }) - (logN(0.2, 0.3, 1) + logN(0.5, 0.2, 1))) < 1e-12);
});

test('a kernel with two inputs closes over the latent with a mean affine in both', async () => {
  const src = `
a ~ Normal(0, 1)
b ~ Normal(0, 1)
mu ~ Normal(a + 2 * b, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), a = a, b = b)
L = likelihoodof(K, record(y = 0.5))
prior = lawof(record(a = a, b = b))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  for (const [a, b] of [[0.3, -0.4], [1.1, 0.7]]) {
    assert.ok(Math.abs(scorer.likOf({ a, b }) - logN(0.5, a + 2 * b, Math.SQRT2)) < 1e-12, `a=${a} b=${b}`);
  }
});

test('a kernel input renamed from the prior latent closes over the latent too', async () => {
  const src = `
theta ~ Normal(0, 1)
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(record(y = y), t = theta)
L = likelihoodof(K, record(y = 0.5))
prior = lawof(record(t = theta))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ theta: 0.3 }) - closedForm(0.3)) < 1e-12);
});

test('an inline draw inside the kernel is called an unnamed draw in the refusal', async () => {
  const src = `
theta ~ Normal(0, 1)
y ~ Normal(theta + draw(Uniform(interval(0, 1))), 1)
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = 0.5))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(src, 4);
  await assert.rejects(() => buildLogPi(ctx, ctx.derivations.posterior), (e: any) => {
    assert.match(e.message, /ancestor\(s\) an unnamed draw,/);
    return true;
  });
});

test('the recognised marginal needs its inputs at the point it is scored', async () => {
  const clm = require('../clm.ts');
  const lg = require('../linear-gaussian.ts');
  const { ctx } = ctxFor(LATENT + FORMS.direct.tail, 4);
  const d = ctx.derivations.posterior;
  const node = clm.lowerMeasure(d.bodyIR || d.bodyName, ctx,
    { derivation: d, inputMarginal: { scalars: ['theta'], fields: {} } });
  assert.equal(node.reduce.method, 'analytic-gaussian');
  const g = node.reduce.gaussian;
  assert.ok(Math.abs(lg.scoreGaussianMarginal(g, { y: 0.5 }, { theta: 0.3 }) - closedForm(0.3)) < 1e-12);
  assert.throws(() => lg.scoreGaussianMarginal(g, { y: 0.5 }),
    /depends on input 'theta', which has no value at this point/);
});

test('a scored likelihood with no latent of its own keeps the unreduced body', async () => {
  const clm = require('../clm.ts');
  const { ctx } = ctxFor(`
theta ~ Normal(0, 1)
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(record(mu = mu, y = y), theta = theta)
L = likelihoodof(K, record(mu = 0.2, y = 0.5))
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`, 4);
  const d = ctx.derivations.posterior;
  const node = clm.lowerMeasure(d.bodyIR || d.bodyName, ctx,
    { derivation: d, inputMarginal: { scalars: ['theta'], fields: {} } });
  assert.equal(node.reduce, null);
});

test('a scalar variate with a latent closes over it, and a mean free of the point stays constant', async () => {
  const scalar = `
theta ~ Normal(0, 1)
mu ~ Normal(theta, 1)
y ~ Normal(mu, 1)
K = kernelof(y, theta = theta)
L = likelihoodof(K, 0.5)
prior = lawof(record(theta = theta))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(scalar, 4);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  assert.ok(Math.abs(scorer.likOf({ theta: 0.3 }) - closedForm(0.3)) < 1e-12);
  // The latent is centred at 0 whatever the point, so the likelihood is constant.
  const free = scalar.replace('mu ~ Normal(theta, 1)', 'mu ~ Normal(0, 1)');
  const { ctx: ctx2 } = ctxFor(free, 4);
  const scorer2 = await buildLogPi(ctx2, ctx2.derivations.posterior);
  for (const th of [0.3, -2]) assert.ok(Math.abs(scorer2.likOf({ theta: th }) - logN(0.5, 0, Math.SQRT2)) < 1e-12);
});
