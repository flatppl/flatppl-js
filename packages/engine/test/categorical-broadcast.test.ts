'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('categorical row callbacks preserve direct, conditional and posterior scores', async () => {
  const source = `
p ~ Normal(0, 1)
f(a) = Categorical(softmax([0., a]))
y ~ f.([p, p + .3, p - .5])
K = kernelof(record(y = y), p = p)
L = likelihoodof(K, record(y = [1, 2, 1]))
P = bayesupdate(L, lawof(record(p = p)))
`;
  const score = async (p: number) => {
    const { ctx } = makeMatCtx(source + `
direct = logdensityof(f.([${p}, ${p + .3}, ${p - .5}]), [1, 2, 1])
forward = logdensityof(K(p = ${p}), record(y = [1, 2, 1]))
posterior = logdensityof(P, record(p = ${p}))
`, { sampleCount: 1 });
    return Promise.all(['direct', 'forward', 'posterior'].map(async (name) =>
      (await ctx.getMeasure(name)).samples[0]));
  };
  for (const p of [-1, .2, 1.2]) {
    const logits = [p, p + .3, p - .5];
    // Independent two-category softmax and standard Normal identities.
    const likelihood = logits[1] - logits.reduce((s, x) => s + Math.log1p(Math.exp(x)), 0);
    const prior = -.5 * (Math.log(2 * Math.PI) + p * p);
    const actual = await score(p);
    for (let i = 0; i < actual.length; i++) {
      assert.ok(Math.abs(actual[i] - likelihood - (i === 2 ? prior : 0)) < 1e-12);
    }
    const gradient = 1 - logits.reduce((s, x) => s + 1 / (1 + Math.exp(-x)), 0);
    const h = 1e-5, lo = await score(p - h), hi = await score(p + h);
    for (let i = 0; i < actual.length; i++) {
      assert.ok(Math.abs((hi[i] - lo[i]) / (2 * h)
        - gradient + (i === 2 ? p : 0)) < 1e-8);
    }
  }
});

test('categorical callbacks retain category origins, keyword inputs and held vectors', async () => {
  const { ctx } = makeMatCtx(`
f(a) = Categorical0(softmax([0., a]))
scalar ~ f.(a = .2)
rows ~ f.(a = [.2, .5, -.3])
scalar_score = logdensityof(lawof(scalar), 1)
row_score = logdensityof(lawof(rows), [0, 1, 0])
p = [.2, .8]
held(a) = Categorical(p)
held_score = logdensityof(held.([1, 2]), [2, 2])
`, { sampleCount: 64, rootSeed: 117 });
  assert.deepEqual((await ctx.getMeasure('scalar')).value.shape, [64]);
  const rows = (await ctx.getMeasure('rows')).value;
  assert.deepEqual(rows.shape, [64, 3]);
  assert.ok(rows.data.every((x: number) => x === 0 || x === 1));
  assert.ok(Math.abs((await ctx.getMeasure('scalar_score')).samples[0]
    - .2 + Math.log1p(Math.exp(.2))) < 1e-12);
  const expected = .5 - [.2, .5, -.3].reduce((s, x) => s + Math.log1p(Math.exp(x)), 0);
  assert.ok(Math.abs((await ctx.getMeasure('row_score')).samples[0] - expected) < 1e-12);
  assert.ok(Math.abs((await ctx.getMeasure('held_score')).samples[0] - 2 * Math.log(.8)) < 1e-12);
});

test('categorical rows draw independently conditional on the shared captured parameter', async () => {
  const { ctx } = makeMatCtx(`
p ~ Normal(0, .4)
f(a) = Categorical(softmax([0., a + p]))
y ~ f.([-1, 1])
`, { sampleCount: 4096, rootSeed: 117 });
  const p = (await ctx.getMeasure('p')).samples;
  const y = (await ctx.getMeasure('y')).value;
  assert.deepEqual(y.shape, [4096, 2]);
  const sums = [0, 0], variances = [0, 0];
  let cross = 0;
  for (let i = 0; i < p.length; i++) {
    const residuals = [0, 0];
    for (let j = 0; j < 2; j++) {
      const draw = y.data[2 * i + j];
      assert.ok(draw === 1 || draw === 2);
      const probability = 1 / (1 + Math.exp(-(p[i] + 2 * j - 1)));
      residuals[j] = (draw === 2 ? 1 : 0) - probability;
      sums[j] += residuals[j];
      variances[j] += probability * (1 - probability);
    }
    cross += residuals[0] * residuals[1];
  }
  for (let j = 0; j < 2; j++) assert.ok(Math.abs(sums[j]) < 5 * Math.sqrt(variances[j]));
  assert.ok(Math.abs(cross / p.length) < .02);
});

test('a scalar distribution callback reduces its constructed vector inside each row', async () => {
  const { ctx } = makeMatCtx(`
f(a) = Normal(sum([a, a]), 1)
y ~ f.([1, 2])
r = logdensityof(lawof(y), [2, 4])
`, { sampleCount: 1024, rootSeed: 117 });
  assert.ok(Math.abs((await ctx.getMeasure('r')).samples[0] + Math.log(2 * Math.PI)) < 1e-12);
  const y = (await ctx.getMeasure('y')).value;
  assert.deepEqual(y.shape, [1024, 2]);
  for (let j = 0; j < 2; j++) {
    let sum = 0;
    for (let i = 0; i < 1024; i++) sum += y.data[2 * i + j];
    assert.ok(Math.abs(sum / 1024 - 2 * (j + 1)) < .15);
  }
});
