'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

function countScore(first: number, second: number): number {
  return Math.log(first) + 2 * Math.log(second) - Math.log(2) - first - second;
}

test('polynomial bin sampling and density retain unnormalized rates and zero bins', async () => {
  const N = 2048;
  const { ctx } = ctxFor(`
I = weighted(x -> polynomial([1.0, 2.0], x), Lebesgue(interval(0, 1)))
J = I
M = BinnedPoissonProcess([-1, 0, 0.5, 1, 2], J)
lp = logdensityof(M, [0, 1, 2, 0])
impossible = logdensityof(M, [1, 0, 0, 0])
`, N);
  const m = await ctx.getMeasure('M');
  const means = [0, 0, 0, 0];
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < 4; k++) means[k] += m.samples[4 * i + k] / N;
  }
  // Integrating 1 + 2x gives rates [0, 3/4, 5/4, 0].
  assert.equal(means[0], 0);
  assert.equal(means[3], 0);
  assert.ok(Math.abs(means[1] - 0.75) < 0.12);
  assert.ok(Math.abs(means[2] - 1.25) < 0.15);
  assert.ok(Math.abs((await ctx.getMeasure('lp')).samples[0] - countScore(0.75, 1.25)) < 1e-12);
  assert.equal((await ctx.getMeasure('impossible')).samples[0], -Infinity);
});

test('polynomial bins intersect reference and truncation supports and retain reference scales', async () => {
  for (const [intensity, first, second] of [
    ['truncate(weighted(x -> polynomial([1.0,2.0],x), Lebesgue(reals)), interval(0,1))', 0.75, 1.25],
    ['truncate(truncate(weighted(x -> polynomial([1.0,2.0],x), Lebesgue(interval(0,2))), interval(0.25,1.5)), interval(0,0.75))', 0.4375, 0.5625],
    ['weighted(x -> polynomial([1.0,2.0],x), Uniform(interval(0,2)))', 0.375, 0.625],
    ['superpose(weighted(2, weighted(x -> polynomial([1.0,2.0],x), Lebesgue(interval(0,1)))), weighted(3, Lebesgue(interval(0,1))))', 3, 4],
  ] as const) {
    const { ctx } = ctxFor(`I = ${intensity}\nlp = logdensityof(BinnedPoissonProcess([0,0.5,1], I), [1,2])`, 1);
    assert.ok(Math.abs((await ctx.getMeasure('lp')).samples[0] - countScore(first, second)) < 1e-12);
  }
  // Integral of 1+2x² on [-1,0] and [0,2]: 5/3 and 22/3.
  const { ctx } = ctxFor(`
I = weighted(x -> polynomial([1.0,0.0,2.0],x), Lebesgue(interval(-1,2)))
lp = logdensityof(BinnedPoissonProcess([-1,0,2], I), [1,2])
`, 1);
  assert.ok(Math.abs((await ctx.getMeasure('lp')).samples[0] - countScore(5 / 3, 22 / 3)) < 1e-12);
});

test('polynomial coefficient extras and reified aliases remain live', async () => {
  const { ctx } = ctxFor(`
theta ~ Uniform(interval(0,4))
I = weighted((x,p) -> polynomial([1.0,p],x), Lebesgue(interval(0,1)), theta)
M = BinnedPoissonProcess([0,0.5,1],I)
score = functionof(logdensityof(M,[1,2]),theta=theta)
a = score(0.6)
b = score(2.4)
c = score(0.6)
`, 2048);
  for (const [name, theta] of [['a', 0.6], ['b', 2.4], ['c', 0.6]] as const) {
    const expected = countScore(0.5 + theta / 8, 0.5 + 3 * theta / 8);
    assert.ok(Math.abs((await ctx.getMeasure(name)).samples[0] - expected) < 1e-12);
  }
  const parent = await ctx.getMeasure('theta'), counts = await ctx.getMeasure('M');
  const mean = [0, 0], expectedMean = [0, 0];
  for (let i = 0; i < 2048; i++) {
    mean[0] += counts.samples[2 * i] / 2048;
    mean[1] += counts.samples[2 * i + 1] / 2048;
    expectedMean[0] += (0.5 + parent.samples[i] / 8) / 2048;
    expectedMean[1] += (0.5 + 3 * parent.samples[i] / 8) / 2048;
  }
  assert.ok(Math.abs(mean[0] - expectedMean[0]) < 0.12);
  assert.ok(Math.abs(mean[1] - expectedMean[1]) < 0.15);
  const { ctx: reified } = ctxFor(`
x ~ Uniform(interval(0.1,2))
a = elementof(reals)
f = functionof(polynomial([a,1.0],x),x=x,a=a)
I = weighted(f,Lebesgue(interval(0,1)),x)
score = functionof(logdensityof(BinnedPoissonProcess([0,0.5,1],I),[1,2]),x=x)
u = score(0.4)
v = score(1.2)
`, 1);
  for (const [name, x] of [['u', 0.4], ['v', 1.2]] as const) {
    assert.ok(Math.abs((await reified.getMeasure(name)).samples[0] - countScore(x / 2 + 1 / 8, x / 2 + 3 / 8)) < 1e-12);
  }
});

test('host-evaluated finite support bounds retain the current parameter slice', async () => {
  const { ctx } = ctxFor(`
hi ~ Uniform(interval(0.5,2))
I = weighted(x -> polynomial([1.0,2.0],x),Lebesgue(interval(0,hi)))
score = functionof(logdensityof(BinnedPoissonProcess([0,0.5,1],I),[1,2]),hi=hi)
a = score(0.75)
b = score(1.5)
`, 1);
  for (const [name, second] of [['a', 0.5625], ['b', 1.25]] as const) {
    assert.ok(Math.abs((await ctx.getMeasure(name)).samples[0] - countScore(0.75, second)) < 1e-12);
  }
});

test('polynomial bin moments avoid separate endpoint power overflow and underflow', async () => {
  for (const [coefficients, lo, hi, mass] of [
    ['[0.0,1e-200]', '1e200', '2e200', 1.5e200],
    ['[0.0,1e300]', '0', '1e-200', 5e-101],
    ['[0.0,1e-16]', '1e16', '10000000000000002', 2],
  ] as const) {
    const { ctx } = ctxFor(`I = weighted(x -> polynomial(${coefficients},x),Lebesgue(interval(${lo},${hi})))\nlp = logdensityof(BinnedPoissonProcess([${lo},${hi}],I),[0])`, 1);
    const score = (await ctx.getMeasure('lp')).samples[0];
    assert.ok(Math.abs(score + mass) <= 1e-12 * mass, `${score} vs ${-mass}`);
  }
});

test('unsupported event weights and unbounded full polynomial support refuse', async () => {
  for (const intensity of [
    'weighted(x -> exp(x),Lebesgue(interval(0,1)))',
    'weighted(x -> polynomial([1.0,2.0],2*x),Lebesgue(interval(0,1)))',
    'weighted(x -> polynomial([x,1.0],x),Lebesgue(interval(0,1)))',
    'weighted(x -> polynomial([1.0,2.0],x),Lebesgue(reals))',
  ]) {
    const { ctx } = ctxFor(`I = ${intensity}\nM = BinnedPoissonProcess([0,1],I)\nlp = logdensityof(M,[0])`, 1);
    await assert.rejects(ctx.getMeasure('M'), /BinnedPoissonProcess/);
    await assert.rejects(ctx.getMeasure('lp'), /BinnedPoissonProcess/);
  }
  const { ctx } = ctxFor(`
x = elementof(reals)
a = x
f = functionof(polynomial([a,1.0],x),x=x)
I = weighted(f,Lebesgue(interval(0,1)))
M = BinnedPoissonProcess([0,1],I)
lp = logdensityof(M,[0])
`, 1);
  await assert.rejects(ctx.getMeasure('lp'), /BinnedPoissonProcess/);
  const { ctx: infiniteBin } = ctxFor(`
I = weighted(x -> polynomial([1.0,2.0],x),Lebesgue(interval(0,1)))
lp = logdensityof(BinnedPoissonProcess([0,inf],I),[0])
`, 1);
  await assert.rejects(infiniteBin.getMeasure('lp'), /BinnedPoissonProcess/);
});
