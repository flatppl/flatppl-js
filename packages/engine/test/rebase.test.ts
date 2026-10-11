'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { processSource } = require('..');

const close = (actual: number, expected: number, tol = 1e-12) =>
  assert.ok(Math.abs(actual - expected) < tol, `${actual} != ${expected}`);
const zeroScore = (s: number) => Math.log(3) / 2 - 2 * Math.log(s * Math.sqrt(2 * Math.PI)) - 1 / s ** 2;

test('rebase scores ordered and zero-sum Normal laws with the coordinate reference', async () => {
  const { ctx } = makeMatCtx(`
n = 3
A = rebase(iid(Normal(mu = 1, sigma = 2), n), Lebesgue(support = ordered(n)))
B = rebase(iid(Normal(7, 2), 3), Lebesgue(zerosum(3)))
a = logdensityof(A, [-1, 0, 3])
b = logdensityof(B, [-1, 0, 1])
outside = logdensityof(A, [0, 0, 1])
unit = rebase(iid(Normal(8, 2), 1), Lebesgue(zerosum(1)))
z = logdensityof(unit, [0])
tiny = logdensityof(rebase(iid(Normal(8, 1e-200), 1), Lebesgue(zerosum(1))), [0])
`, { sampleCount: 4 });
  close((await ctx.getMeasure('a')).samples[0], Math.log(6) - 3 * Math.log(2 * Math.sqrt(2 * Math.PI)) - 9 / 8);
  close((await ctx.getMeasure('b')).samples[0], zeroScore(2));
  assert.equal((await ctx.getMeasure('outside')).samples[0], -Infinity);
  close((await ctx.getMeasure('z')).samples[0], 0);
  close((await ctx.getMeasure('tiny')).samples[0], 0);
  const unit = await ctx.getMeasure('unit');
  assert.deepEqual(unit.value.shape, [4, 1]);
  assert.ok(unit.value.data.every((x: number) => x === 0));
});

test('rebase retains inferred scales and the normalizer derivative through reification', async () => {
  const { ctx } = makeMatCtx(`
s = elementof(posreals)
M = rebase(iid(Normal(7, s), 3), Lebesgue(zerosum(3)))
score = logdensityof(M, [-1, 0, 1])
f = functionof(score, scale = s)
a = f(2)
b = f(3)
lo = f(1.99999)
hi = f(2.00001)
`, { sampleCount: 4 });
  close((await ctx.getMeasure('a')).samples[0], zeroScore(2));
  close((await ctx.getMeasure('b')).samples[0], zeroScore(3));
  const lo = (await ctx.getMeasure('lo')).samples[0];
  const hi = (await ctx.getMeasure('hi')).samples[0];
  close((hi - lo) / 0.00002, -0.75, 1e-9);
});

test('rebase generic Normal-product fallback and matching named kernel inputs', async () => {
  const { ctx } = makeMatCtx(`
s = elementof(posreals)
m = elementof(reals)
N = Normal(m, s)
K = functionof(N, location = m, scale = s)
B = functionof(Normal(m, 1), scale = s, location = m)
R = rebase(K, B)
Q = R(scale = 2, location = 1)
score = logdensityof(Q, 0)
`, { sampleCount: 4 });
  // Product N(1,2)N(1,1), normalized: N(1,sqrt(4/5)).
  close((await ctx.getMeasure('score')).samples[0], -0.5 * Math.log(2 * Math.PI * 0.8) - 1 / 1.6);
});

test('rebase rejects mismatched variates, interfaces, and keyword operands', () => {
  for (const src of [
    'R = rebase(Normal(0,1), Lebesgue(zerosum(3)))',
    'R = rebase(M = Normal(0,1), B = Normal(0,1))',
    'x = elementof(reals)\nK = functionof(Normal(x,1), a=x)\nB = functionof(Normal(x,1), b=x)\nR = rebase(K,B)',
  ]) assert.ok(processSource(src).diagnostics.some((d: any) => d.severity === 'error'));
});

test('rebase sampling matches ordered moments and zero-sum conditional covariance', async () => {
  const count = 16384;
  const { ctx } = makeMatCtx(`
ordered_draw ~ rebase(iid(Normal(1, 2), 2), Lebesgue(ordered(2)))
scale ~ Exponential(1)
effects ~ rebase(iid(Normal(7, scale), 3), Lebesgue(zerosum(3)))
other ~ rebase(iid(Normal(0, 1), 3), Lebesgue(zerosum(3)))
prior = lawof(record(scale = scale, effects = effects))
score = logdensityof(prior, record(scale = 2, effects = [-1, 0, 1]))
reverse = lawof(record(effects = effects, scale = scale))
reverse_score = logdensityof(reverse, record(effects = [-1, 0, 1], scale = 2))
`, { sampleCount: count });
  close((await ctx.getMeasure('score')).samples[0], -2 + zeroScore(2));
  close((await ctx.getMeasure('reverse_score')).samples[0], -2 + zeroScore(2));
  const ordered = (await ctx.getMeasure('ordered_draw')).value.data;
  let left = 0, right = 0;
  for (let i = 0; i < count; i++) {
    assert.ok(ordered[2 * i] < ordered[2 * i + 1]);
    left += ordered[2 * i]; right += ordered[2 * i + 1];
  }
  // E[min/max of 2 iid N(mu,sigma)] = mu +/- sigma/sqrt(pi).
  close(left / count, 1 - 2 / Math.sqrt(Math.PI), 0.06);
  close(right / count, 1 + 2 / Math.sqrt(Math.PI), 0.06);
  const prior = await ctx.getMeasure('prior');
  const scales = prior.fields.scale.samples;
  const effects = prior.fields.effects.value.data;
  const other = (await ctx.getMeasure('other')).value.data;
  let variance = 0, covariance = 0, cross = 0;
  for (let i = 0; i < count; i++) {
    const x = effects[3 * i] / scales[i];
    const y = effects[3 * i + 1] / scales[i];
    const z = effects[3 * i + 2] / scales[i];
    close(x + y + z, 0, 1e-12);
    variance += x * x; covariance += x * y;
    cross += x * other[3 * i];
  }
  // Cov(effects / scale) = I - 11'/3. Normalizing EACH atom's scale
  // distinguishes conditional sampling from a frozen or independently fed scale.
  close(variance / count, 2 / 3, 0.03);
  close(covariance / count, -1 / 3, 0.03);
  close(cross / count, 0, 0.03);
});

test('rebase supports closed operands, kernel application and explicit likelihood boundaries', async () => {
  const { ctx } = makeMatCtx(`
s = elementof(posreals)
M = iid(Normal(0, s), 3)
K = functionof(M, scale = s)
R = rebase(K, Lebesgue(zerosum(3)))
effects ~ R(2)
y ~ Normal(sum(effects), 1)
model = kernelof(y, scale = s, effects = effects)
L = likelihoodof(model, 0)
score = logdensityof(L, record(scale = 3, effects = [-1, 0, 1]))
`, { sampleCount: 8 });
  close((await ctx.getMeasure('score')).samples[0], -0.5 * Math.log(2 * Math.PI));
  const effects = await ctx.getMeasure('effects');
  assert.deepEqual(effects.value.shape, [8, 3]);
});

test('exact rebase draws retain the iid Normal RNG continuation', async () => {
  const { ctx } = makeMatCtx(`
rng = rnginit([1, 2, 3, 4])
x, xs = rand(iid(Normal(0, 2), 3), rng)
z, zs = rand(rebase(iid(Normal(7, 2), 3), Lebesgue(zerosum(3))), rng)
o, os = rand(rebase(iid(Normal(0, 2), 3), Lebesgue(ordered(3))), rng)
after_x, _ = rand(iid(Normal(0, 1), 4), xs)
after_z, _ = rand(iid(Normal(0, 1), 4), zs)
after_o, _ = rand(iid(Normal(0, 1), 4), os)
`, { sampleCount: 4 });
  const data = async (name: string) => Array.from((await ctx.getMeasure(name)).value.data) as number[];
  const x = await data('x');
  const z = await data('z');
  const mean = x.reduce((a, b) => a + b, 0) / x.length;
  for (let i = 0; i < x.length; i++) close(z[i], x[i] - mean);
  assert.deepEqual(await data('o'), x.slice().sort((a, b) => a - b));
  assert.deepEqual(await data('after_z'), await data('after_x'));
  assert.deepEqual(await data('after_o'), await data('after_x'));
});

test('rebase applies inline kernels and refuses an unsupported sampler', async () => {
  const { ctx } = makeMatCtx(`
s = elementof(posreals)
R = rebase(functionof(iid(Normal(0, s), 3), scale = s), Lebesgue(zerosum(3)))
a = logdensityof(R(2), [-1, 0, 1])
b = logdensityof(rebase(functionof(iid(Normal(0, s), 3), scale = s), Lebesgue(zerosum(3)))(3), [-1, 0, 1])
G = rebase(Normal(0, 1), Normal(1, 1))
`, { sampleCount: 4 });
  close((await ctx.getMeasure('a')).samples[0], zeroScore(2));
  close((await ctx.getMeasure('b')).samples[0], zeroScore(3));
  await assert.rejects(ctx.getMeasure('G'), /rebase:.*sampl/);
});

test('rebase normalizers stay live in the MCMC point scorer', async () => {
  const { buildLogPi } = require('../mcmc-density.ts');
  const { ctx } = makeMatCtx(`
scale ~ Exponential(1)
effects ~ rebase(iid(Normal(7, scale), 3), Lebesgue(zerosum(3)))
prior = lawof(record(scale = scale, effects = effects))
y ~ Normal(sum(effects), 1)
K = kernelof(y, scale = scale, effects = effects)
L = likelihoodof(K, 0)
post = bayesupdate(L, prior)
`, { sampleCount: 4 });
  const { priorOf } = await buildLogPi(ctx, ctx.derivations.post);
  const score = (scale: number) => priorOf({ scale, effects: [-1, 0, 1] });
  for (const scale of [0.5, 2, 3]) close(score(scale), -scale + zeroScore(scale));
  close((score(2.00001) - score(1.99999)) / 0.00002, -1.75, 1e-9);
});

test('rebase refuses invalid Normal parameters even when the mean cancels', async () => {
  for (const [mu, sigma] of [['inf', '2'], ['0', '0'], ['0', 'inf']]) {
    const { ctx } = makeMatCtx(`
M = rebase(iid(Normal(${mu}, ${sigma}), 3), Lebesgue(zerosum(3)))
score = logdensityof(M, [-1, 0, 1])
`, { sampleCount: 4 });
    await assert.rejects(ctx.getMeasure('score'), /rebase:.*finite.*sigma > 0/);
    await assert.rejects(ctx.getMeasure('M'), /rebase:.*finite.*sigma > 0/);
  }
});
