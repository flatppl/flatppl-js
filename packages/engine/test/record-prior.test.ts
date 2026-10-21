const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { materialiser } = require('..');
const { buildModelViewFromCtx } = require('../model-view.ts');

test('record reference priors join a proper dependent law for density and MCMC', async () => {
  const source = `
beta = elementof(cartpow(reals, 2))
s ~ Exponential(2)
b ~ Normal(s, 1)
proper = lawof((s = s, b = b))
improper = Lebesgue(cartprod(beta = cartpow(reals, 2)))
prior = joint(improper, proper)
y ~ Normal.(beta .+ b, s)
K = kernelof(y, beta, s, b)
posterior = bayesupdate(likelihoodof(K, [1, 3]), prior)
score = logdensityof(prior, (beta = [0.5, -0.5], s = 1.5, b = 2))
`;
  const { proc, ctx } = ctxFor(source, 256);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const normal = (x: number, mu: number, sigma: number) =>
    -Math.log(sigma) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((x - mu) / sigma) ** 2;
  const prior = Math.log(2) - 3 + normal(2, 1.5, 1);
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0] - prior) < 1e-12);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const point = { beta: Float64Array.of(0.5, -0.5), s: 1.5, b: 2 };
  const expected = prior + normal(1, 2.5, 1.5) + normal(3, 1.5, 1.5) + Math.log(1.5);
  assert.ok(Math.abs(view.logPosterior(view.unconstrainAll(point)) - expected) < 1e-12);
  const posterior = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'slice', chains: 4, warmup: 64, draws: 64, seed: 20261016,
  });
  for (const field of Object.values(posterior.fields) as any[]) {
    assert.ok(Array.from(field.samples).every(Number.isFinite));
    assert.ok(new Set(field.samples).size > 4);
  }
});

test('a reference field and a renamed law coordinate remain independent', async () => {
  const { ctx } = ctxFor(`
alpha = elementof(cartpow(reals, 1))
z ~ Normal(0, 1)
prior = joint(Lebesgue(cartprod(z = cartpow(reals, 1))), lawof((b = z)))
y ~ Normal(alpha[1] + z, 1)
K = kernelof(y, z = alpha, b = z)
posterior = bayesupdate(likelihoodof(K, 1), prior)
`);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const point = view.unconstrainAll({ z: [0.25], b: 0.5 });
  const expected = -Math.log(2 * Math.PI) - (0.5 ** 2 + 0.25 ** 2) / 2;
  assert.ok(Math.abs(view.logPosterior(point) - expected) < 1e-12);
});

test('record prior chains score dependent references between proper law blocks', async () => {
  const {ctx} = ctxFor(`
s ~ Exponential(2)
b = elementof(reals)
m ~ Normal(b, s)
proper = lawof((s = s))
improper = joint(b = Lebesgue(interval(0, s)))
reference_K = functionof(improper, s)
proper_K = kernelof((m = m), b, s)
prior = jointchain(proper, reference_K, proper_K)
y ~ Normal(m, 1)
K = kernelof(y, s, b, m)
posterior = bayesupdate(likelihoodof(K, 1), prior)
`, 1);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const normal = (x: number, mu: number, sigma: number) =>
    -Math.log(sigma) - 0.5 * Math.log(2 * Math.PI) - 0.5 * ((x - mu) / sigma) ** 2;
  const score = (s: number, b: number) => view.logPosterior(view.unconstrainAll({s, b, m: 0.5}));
  assert.equal(view.dim, 3);
  assert.ok(Math.abs(score(2, 1) - (Math.log(2) - 4
    + normal(0.5, 1, 2) + normal(1, 0.5, 1) + Math.log(2))) < 1e-12);
  assert.equal(score(2, 3), -Infinity);
  assert.ok(Math.abs(score(4, 3) - (Math.log(2) - 8
    + normal(0.5, 3, 4) + normal(1, 0.5, 1) + Math.log(4))) < 1e-12);
});

test('an improper scale retains its chart when a proper prior depends on it', async () => {
  const {ctx} = ctxFor(`
s = elementof(posreals)
b ~ Normal(0, s)
improper = joint(s = Lebesgue(posreals))
proper_K = kernelof((b = b), s)
prior = jointchain(improper, proper_K)
y ~ Normal(b, 1)
K = kernelof(y, s, b)
posterior = bayesupdate(likelihoodof(K, 1), prior)
`, 1);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const expected = -Math.log(2 * Math.PI) - (0.5 / 2) ** 2 / 2 - 0.5 ** 2 / 2;
  assert.equal(view.dim, 2);
  assert.ok(Math.abs(view.logPosterior(view.unconstrainAll({s: 2, b: 0.5})) - expected) < 1e-12);
});

test('a retained record prior scores an observed point without sampling its improper base', async () => {
  const { proc, ctx } = ctxFor(`
a = elementof(reals)
b ~ iid(Normal(a, 1), 1)
prior = jointchain(joint(a = Lebesgue(record(support = reals))), functionof(lawof((b = b)), a))
y ~ Normal.([-0.5, 0.5] .* b[1] .+ a, 1)
K = kernelof(y, a, b)
posterior = bayesupdate(likelihoodof(K, [1, 3]), prior)
score = logdensityof(posterior, (a = 0.3, b = [0.5]))
`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const normal = (x: number, mu: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - mu) ** 2;
  const expected = normal(0.5, 0.3) + normal(1, 0.05) + normal(3, 0.55);
  assert.ok(Math.abs((await ctx.getMeasure('score')).samples[0] - expected) < 1e-12);
});

test('a dependent vector reference keeps its dimension and pointwise bounds', async () => {
  const { ctx } = ctxFor(`
s ~ Exponential(2)
b = elementof(cartpow(reals, 2))
proper = lawof((s = s))
improper = joint(b = Lebesgue(cartpow(interval(0, s), 2)))
prior = jointchain(proper, functionof(improper, s))
y ~ Normal.(b, 1)
K = kernelof(y, s, b)
posterior = bayesupdate(likelihoodof(K, [1, 1]), prior)
`, 1);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.posterior);
  const score = (s: number) => view.logPosterior(view.unconstrainAll({ s, b: [1, 3] }));
  assert.equal(view.dim, 3);
  assert.equal(score(2), -Infinity);
  const expected = Math.log(2) - 8 - Math.log(2 * Math.PI) - 2 + Math.log(4);
  assert.ok(Math.abs(score(4) - expected) < 1e-12);
});
