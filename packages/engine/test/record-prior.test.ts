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

test('a reference field cannot silently replace a distinct proper draw coordinate', async () => {
  const { ctx } = ctxFor(`
alpha = elementof(cartpow(reals, 1))
z ~ Normal(0, 1)
prior = joint(Lebesgue(cartprod(z = cartpow(reals, 1))), lawof((b = z)))
y ~ Normal(alpha[1] + z, 1)
K = kernelof(y, z = alpha, b = z)
posterior = bayesupdate(likelihoodof(K, 1), prior)
`);
  await assert.rejects(buildModelViewFromCtx(ctx, ctx.derivations.posterior), /conflicts with a draw name/);
});

