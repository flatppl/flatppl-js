'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const SOURCE = `
mu ~ Normal(0, 1)
scale ~ Uniform(interval(1, 3))
location = [mu, mu, mu]
transform = -scale .* eye(3)
theta ~ locscale(iid(Normal(0.5, 0.7), 3), location, transform)
prior = lawof(record(mu = mu, scale = scale, theta = theta))
`;

function normalLogpdf(x: number, mu: number, sigma: number) {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma)
    - 0.5 * ((x - mu) / sigma) ** 2;
}

test('broadcast affine density preserves bounded scalar support', async () => {
  const { ctx } = ctxFor(`
M = pushfwd(x -> (x .* 2) .+ 3, iid(Uniform(interval(-1, 2)), 2))
inside = logdensityof(M, [2, 4])
outside = logdensityof(M, [0, 4])
`, 1);
  const inside = await ctx.getMeasure('inside');
  const outside = await ctx.getMeasure('outside');
  assert.ok(Math.abs(inside.samples[0] + 2 * Math.log(6)) < 1e-11);
  assert.equal(outside.samples[0], -Infinity);
});

test('a zero broadcast shift preserves the iid density through a named base', async () => {
  const { ctx } = ctxFor(`
J = 3
base = iid(Normal(0.5, 0.7), J)
shift(x) = x .+ 0
M = pushfwd(shift, base)
lp = logdensityof(M, [0, 1, 3])
`, 1);
  const lp = await ctx.getMeasure('lp');
  const expected = [0, 1, 3].reduce((sum, y) => sum + normalLogpdf(y, 0.5, 0.7), 0);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-11);
});

test('affine density keeps explicit live parameters and the full Jacobian', async () => {
  // An off-centre base distinguishes mu - scale*x from mu + scale*x.
  for (const [mu, scale] of [[1, 2], [-0.5, 1.5]]) {
    const point = [0, 1, 3];
    const { proc, ctx } = ctxFor(SOURCE + `
lp = logdensityof(prior, record(mu = ${mu}, scale = ${scale}, theta = [0, 1, 3]))
`, 1);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const lp = await ctx.getMeasure('lp');
    const expected = normalLogpdf(mu, 0, 1) - Math.log(2)
      + point.reduce((sum, y) => sum + normalLogpdf(y, mu - 0.5 * scale, 0.7 * scale), 0);
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-11,
      `${lp.samples[0]} != ${expected}`);
  }
});

test('affine draws retain their explicit shared location and scale', async () => {
  const N = 2048;
  const { proc, ctx } = ctxFor(SOURCE, N);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const prior = await ctx.getMeasure('prior');
  const mu = prior.fields.mu.samples;
  const scale = prior.fields.scale.samples;
  const theta = prior.fields.theta.value;
  assert.deepEqual(theta.shape, [N, 3]);
  let sum = 0, sumSquares = 0;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < 3; j++) {
      const z = (mu[i] - theta.data[3 * i + j]) / scale[i];
      sum += z;
      sumSquares += z * z;
    }
  }
  // Source Normal(0.5, 0.7): E[z] = 0.5 and E[z^2] = 0.74.
  assert.ok(Math.abs(sum / (3 * N) - 0.5) < 0.05);
  assert.ok(Math.abs(sumSquares / (3 * N) - 0.74) < 0.08);
});

test('an outer shift stays shared when the iid base reifies the same node', async () => {
  const { proc, ctx } = ctxFor(`
mu ~ Normal(0, 1)
location = [mu, mu]
theta ~ locscale(iid(lawof(mu), 2), location, [[1, 0], [0, 1]])
prior = lawof(record(mu = mu, theta = theta))
`, 2048);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const prior = await ctx.getMeasure('prior');
  const mu = prior.fields.mu.samples;
  const theta = prior.fields.theta.value.data;
  let cross = 0, squares = 0;
  for (let i = 0; i < mu.length; i++) {
    cross += mu[i] * theta[2 * i];
    squares += theta[2 * i] ** 2;
  }
  // theta_j = shared mu + a fresh independent Normal copy.
  assert.ok(Math.abs(cross / mu.length - 1) < 0.15);
  assert.ok(Math.abs(squares / mu.length - 2) < 0.2);
});

test('a reified iid affine density preserves the explicit outer parameter', async () => {
  const { proc, ctx } = ctxFor(`
mu ~ Normal(0, 1)
location = [mu, mu]
theta ~ locscale(iid(lawof(mu), 2), location, [[2, 0], [0, 2]])
prior = lawof(record(mu = mu, theta = theta))
lp = logdensityof(prior, record(mu = 1, theta = [0, 1]))
`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const lp = await ctx.getMeasure('lp');
  const expected = normalLogpdf(1, 0, 1)
    + normalLogpdf(0, 1, 2) + normalLogpdf(1, 1, 2);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-11);
});

test('a broadcast affine map transforms a reified vector law', async () => {
  const { ctx } = ctxFor(`
z ~ iid(Normal(0.5, 0.7), 3)
M = pushfwd(x -> (2 .* x) .+ 3, lawof(z))
lp = logdensityof(M, [0, 1, 3])
`, 1024);
  const lp = await ctx.getMeasure('lp');
  const expected = [0, 1, 3].reduce((s, y) => s + normalLogpdf(y, 4, 1.4), 0);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-11);
  const M = await ctx.getMeasure('M');
  const values = M.value.data;
  assert.deepEqual(M.value.shape, [1024, 3]);
  const mean = values.reduce((s: number, y: number) => s + y, 0) / values.length;
  assert.ok(Math.abs(mean - 4) < 0.1);
});
