'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { buildModelViewFromCtx } = require('../model-view.ts');
const { ctxFor } = require('./_ctx-factory.ts');

const normalLogpdf = (x: number, mean: number, sigma: number) =>
  -Math.log(2 * Math.PI) / 2 - Math.log(sigma) - ((x - mean) / sigma) ** 2 / 2;

function near(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) <= 16 * Number.EPSILON * Math.max(1, Math.abs(expected)),
    `${actual} != ${expected}`);
}

test('complex matrix inference uses interleaved row-major coordinates and unit Jacobian', async () => {
  const { ctx, proc } = ctxFor(`
extent = [2, 2]
Z = elementof(cartpow(complexes, extent))
R = Lebesgue(cartpow(complexes, extent))
lp(A) = -sum(abs2.(A)) / 2
prior = joint(Z = logweighted(lp, R))
y ~ Normal(real(Z[1, 2]), 1)
w ~ Normal(imag(Z[2, 1]), 2)
K = kernelof((y = y, w = w), Z)
post = bayesupdate(likelihoodof(K, (y = 4, w = -2)), prior)
V = rowstack([[complex(1.0, 2.0), complex(3.0, 4.0)], [complex(5.0, -6.0), complex(-7.0, 8.0)]])
score = logdensityof(post, (Z = V))
iidScore = logdensityof(iid(Lebesgue(complexes), extent), V)
`, 64);
  assert.deepEqual(proc.diagnostics, []);
  const view = await buildModelViewFromCtx(ctx, ctx.derivations.post);
  const first = Float64Array.of(1, 2, 3, 4, 5, -6, -7, 8);
  const second = Float64Array.of(3, 2, 7, 2, 5, -4, -7, 7);
  assert.equal(view.dim, 8);
  const coordinates = view.constrainAll(first);
  const matrixPoint = view.complexScorerValue('Z', (name: string) => coordinates[name]);
  assert.deepEqual(matrixPoint.shape, [2, 2]);
  assert.deepEqual(Array.from(matrixPoint.data), [1, 3, 5, -7]);
  assert.deepEqual(Array.from(matrixPoint.im), [2, 4, -6, 8]);
  assert.deepEqual(view.unconstrainAll({ Z: matrixPoint }), first);
  const expected = (u: Float64Array) =>
    -u.reduce((s, x) => s + x * x, 0) / 2 + normalLogpdf(4, u[2], 1) + normalLogpdf(-2, u[5], 2);
  near((await ctx.getMeasure('score')).samples[0], expected(first));
  near((await ctx.getMeasure('iidScore')).samples[0], 0);
  near(view.logPosterior(first), expected(first));
  const batch = view.logPosteriorBatch([first, second]);
  near(batch[0], expected(first));
  near(batch[1], expected(second));

  const posterior = await materialiser.materialiseMeasure('post', ctx, {
    backend: 'ram', chains: 4, draws: 16, warmup: 24, seed: 12,
  });
  const matrix = posterior.fields.Z;
  assert.deepEqual(matrix.value.shape, [64, 2, 2]);
  assert.equal(matrix.imag.length, matrix.samples.length);
  assert.ok(matrix.imag.every(Number.isFinite) && new Set(matrix.imag).size > 4);
});

test('RAM recovers an asymmetric complex latent alongside a real latent', async () => {
  const draws = 800, chains = 4;
  const { ctx, proc } = ctxFor(`
z = elementof(complexes)
b = elementof(reals)
R = Lebesgue(complexes)
lp(q) = -log(4 * pi) - real(q)^2 / 2 - imag(q)^2 / 8
prior = joint(z = logweighted(lp, R), b = Normal(-1, 0.5))
y ~ Normal(real(z), 1)
w ~ Normal(imag(z), 1)
o ~ Normal(b, 0.5)
K = kernelof((y = y, w = w, o = o), z, b)
post = bayesupdate(likelihoodof(K, (y = 4, w = -4, o = 2)), prior)
`, draws * chains);
  assert.deepEqual(proc.diagnostics, []);
  const posterior = await materialiser.materialiseMeasure('post', ctx, {
    backend: 'ram', chains, draws, warmup: 400, seed: 12,
  });
  // Completing the squares gives N(2, 1/2), N(-16/5, 4/5), and N(1/2, 1/8),
  // using variance as the second argument.
  const { z, b } = posterior.fields;
  assert.deepEqual(z.value.shape, [draws * chains]);
  const exact = [
    { name: 'z[0]', stream: z.samples, targetMean: 2, targetVariance: 1 / 2 },
    { name: 'z[1]', stream: z.imag, targetMean: -16 / 5, targetVariance: 4 / 5 },
    { name: 'b', stream: b.samples, targetMean: 1 / 2, targetVariance: 1 / 8 },
  ];
  for (const { name, stream, targetMean, targetVariance } of exact) {
    const ess = posterior.diagnostics.perParam[name].essBulk;
    assert.ok(ess > 100);
    const mean = stream.reduce((s: number, x: number) => s + x, 0) / stream.length;
    const variance = stream.reduce((s: number, x: number) => s + (x - mean) ** 2, 0) / stream.length;
    assert.ok(Math.abs(mean - targetMean) < 4 * Math.sqrt(targetVariance / ess),
      `component ${name} mean: ${mean}`);
    assert.ok(Math.abs(variance - targetVariance) < 4 * targetVariance * Math.sqrt(2 / ess),
      `component ${name} variance: ${variance}`);
  }
});
