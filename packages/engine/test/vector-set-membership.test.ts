'use strict';

// Membership in the §03 vector sets `stdsimplex(n)`, `ordered(n)` and
// `zerosum(n)`, through `x in S`, `Lebesgue(support = S)`, a weighted Lebesgue
// prior and `truncate`. Both inequalities and affine sums are part of the set;
// a coordinate reference does not extend the density off its support. Expected
// values are by hand; valid chart outputs may differ by summation roundoff.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { ctxFor } = require('./_ctx-factory.ts');

async function scoreValue(src: string, name: string): Promise<any> {
  const { ctx } = makeMatCtx(src, { sampleCount: 1 });
  const m = await ctx.getMeasure(name);
  if (m && m.value && m.value.data) return m.value.data[0];
  return m.samples[0];
}

async function score(src: string): Promise<number> {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('__score__');
  return m.value ? m.value.data[0] : m.samples[0];
}

const lebesgue = (set: string, x: string) =>
  score(`__score__ = logdensityof(Lebesgue(support = ${set}), ${x})`);

// log N(x; 0, s) by hand.
const normalLogpdf = (x: number, s: number) =>
  -0.5 * Math.log(2 * Math.PI * s * s) - (x * x) / (2 * s * s);

test('`v in ordered(3)` needs strictly increasing entries', async () => {
  const inSet = async (v: string) => scoreValue(`lp = ifelse(${v} in ordered(3), 1.0, 0.0)`, 'lp');
  assert.equal(await inSet('[-1.0, 0.5, 2.0]'), 1.0);
  assert.equal(await inSet('[0.0, 0.0, 1.0]'), 0.0);
  assert.equal(await inSet('[2.0, 1.0, 3.0]'), 0.0);
});

test('`v in stdsimplex(3)` checks signs and the unit sum', async () => {
  const inSet = async (v: string) => scoreValue(`lp = ifelse(${v} in stdsimplex(3), 1.0, 0.0)`, 'lp');
  assert.equal(await inSet('[0.2, 0.3, 0.5]'), 1.0);
  assert.equal(await inSet('[0.2, 0.3, 0.6]'), 0.0);
  assert.equal(await inSet('[-0.1, 0.6, 0.5]'), 0.0);
});

test('`v in zerosum(3)` checks the sum without an absolute tolerance floor', async () => {
  const inSet = async (v: string) => scoreValue(`lp = ifelse(${v} in zerosum(3), 1.0, 0.0)`, 'lp');
  assert.equal(await inSet('[0.5, 0.25, -0.75]'), 1.0);
  assert.equal(await inSet('[0.5, 0.25, -0.7]'), 0.0);
  assert.equal(await inSet('[1e-8, 1e-8, 1e-8]'), 0.0);
  assert.equal(await inSet('[0, 0, 0]'), 1.0);
});

test('`in` over a vector set refuses a point of the wrong length', async () => {
  await assert.rejects(
    () => scoreValue('lp = ifelse([1.0, 2.0] in ordered(3), 1.0, 0.0)', 'lp'),
    /`in` over ordered\(3\) expects a length-3 1-D point, got length 2/);
});

test('Lebesgue(support = ordered(3)) scores 0 on the set and -inf at a tie', async () => {
  assert.equal(await lebesgue('ordered(3)', '[-1.0, 0.5, 2.0]'), 0);
  assert.equal(await lebesgue('ordered(3)', '[0.0, 0.0, 1.0]'), -Infinity);
});

test('Lebesgue(support = zerosum(3)) has zero density off the hyperplane', async () => {
  assert.equal(await lebesgue('zerosum(3)', '[0.5, 0.25, -0.75]'), 0);
  assert.equal(await lebesgue('zerosum(3)', '[0.5, 0.25, -0.7]'), -Infinity);
});

test('Lebesgue(support = stdsimplex(3)) checks signs and the unit sum', async () => {
  assert.equal(await lebesgue('stdsimplex(3)', '[0.2, 0.3, 0.5]'), 0);
  assert.equal(await lebesgue('stdsimplex(3)', '[-0.1, 0.6, 0.5]'), -Infinity);
  assert.equal(await lebesgue('stdsimplex(3)', '[0.2, 0.3, 0.6]'), -Infinity);
});

test('affine set membership retains valid chart outputs at different scales', async () => {
  const { simplexTransform, zerosumTransform } = require('../transforms.ts');
  const cases = [
    ['stdsimplex', simplexTransform(5), [-8, 2, -1, 4]],
    ['stdsimplex', simplexTransform(3), [0, 0]],
    ['zerosum', zerosumTransform(4), [0.1, -0.7, 0.2]],
    ['zerosum', zerosumTransform(4), [1e8, 0.1, -2e8]],
    ['zerosum', zerosumTransform(4), [1e-8, 2e-8, -4e-8]],
  ];
  for (const [set, transform, unconstrained] of cases) {
    const point = (transform as any).constrain(Float64Array.from(unconstrained as number[]));
    const literal = JSON.stringify(Array.from(point));
    assert.equal(await scoreValue(`valid = ifelse(${literal} in ${set}(${point.length}), 1, 0)`, 'valid'), 1);
    assert.equal(await lebesgue(`${set}(${point.length})`, literal), 0);
  }
});

test('a weighted Lebesgue prior on ordered(3) scores its weight on the set', async () => {
  const prior = 'logweighted(fn(logdensityof(iid(Normal(0, 5), 3), _)), Lebesgue(support = ordered(3)))';
  const want = [-1.0, 0.5, 2.0].reduce((acc, x) => acc + normalLogpdf(x, 5), 0);
  const got = await score(`__score__ = logdensityof(${prior}, [-1.0, 0.5, 2.0])`);
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
  assert.equal(await score(`__score__ = logdensityof(${prior}, [0.0, 0.0, 1.0])`), -Infinity);
});

test('a logdensityof weight on a scalar Lebesgue interval resolves its measure', async () => {
  const got = await score(
    '__score__ = logdensityof(logweighted(fn(logdensityof(Normal(0, 5), _)), '
    + 'Lebesgue(support = interval(-10.0, 10.0))), 0.5)');
  assert.ok(Math.abs(got - normalLogpdf(0.5, 5)) < 1e-12, `got ${got}`);
});

test('truncate(iid(Normal(0, 1), 3), ordered(3)) keeps the base density on the set', async () => {
  const want = [0.5, 0.75, 1.0].reduce((acc, x) => acc + normalLogpdf(x, 1), 0);
  const got = await score(
    '__score__ = logdensityof(truncate(iid(Normal(0, 1), 3), ordered(3)), [0.5, 0.75, 1.0])');
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
  assert.equal(await score(
    '__score__ = logdensityof(truncate(iid(Normal(0, 1), 3), ordered(3)), [0.5, 0.25, 1.0])'),
  -Infinity);
});

test('a Lebesgue measure on a vector set refuses to draw', async () => {
  await assert.rejects(
    () => scoreValue('m = Lebesgue(support = zerosum(3))', 'm'),
    /Lebesgue\(support = zerosum\(3\)\) in 'm' has no draws/);
});

test('`in` over a vector set refuses a scalar point', async () => {
  await assert.rejects(
    () => scoreValue('lp = ifelse(0.5 in zerosum(3), 1.0, 0.0)', 'lp'),
    /`in` over zerosum\(3\) expects a length-3 1-D point, got number/);
});

test('a NaN entry is outside every vector set', async () => {
  assert.equal(await lebesgue('zerosum(3)', '[0.5, inf - inf, -0.5]'), -Infinity);
});

test('Lebesgue on a vector set needs a fixed size to classify', async () => {
  const { ctx } = ctxFor('n = elementof(posintegers)\nm = Lebesgue(support = ordered(n))\n__score__ = logdensityof(m, [1.0, 2.0])', 1);
  await assert.rejects(() => ctx.getMeasure('__score__'), /no derivation/);
});

test('MCMC refuses a latent truncated to a vector set', async () => {
  const { materialiser } = require('..');
  const SRC = `
x ~ truncate(iid(Normal(0.0, 1.0), 3), ordered(3))
prior = lawof(record(x = x))
K = functionof(joint(z = Normal(mu = sum(x), sigma = 1.0)), x = x)
L = likelihoodof(pars -> K(x = pars.x), record(z = 0.0))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC, 50);
  await assert.rejects(
    () => materialiser.materialiseMeasure('posterior', ctx, { backend: 'mh', warmup: 5, draws: 5 }),
    /truncate\(ordered\(3\)\)/);
});

test('Lebesgue(support = cartpow(reals, 3)) scores 0 at a finite point', async () => {
  assert.equal(await lebesgue('cartpow(reals, 3)', '[-2.0, 0.5, 7.0]'), 0);
});

test('Lebesgue on a cartpow of a half-line gates every cell', async () => {
  assert.equal(await lebesgue('cartpow(posreals, 3)', '[0.5, 0.25, 1.0]'), 0);
  assert.equal(await lebesgue('cartpow(posreals, 3)', '[0.5, 0.0, 1.0]'), -Infinity);
  assert.equal(await lebesgue('cartpow(nonnegreals, 2)', '[0.0, 3.0]'), 0);
  assert.equal(await lebesgue('cartpow(interval(1.0, inf), 2)', '[1.0, 0.5]'), -Infinity);
});

test('a Lebesgue measure on an unbounded cartpow refuses to draw', async () => {
  await assert.rejects(
    () => scoreValue('m = Lebesgue(support = cartpow(reals, 3))', 'm'),
    /Lebesgue\(support = cartpow\(…, 3\)\) in 'm' has no draws/);
});

test('a logdensityof weight on a bounded Lebesgue box resolves its measure', async () => {
  // The box case failed with "measure ref '__anonN' without resolveMeasureRef opt".
  const want = [0.5, -1.0, 2.0].reduce((acc, x) => acc + normalLogpdf(x, 1), 0);
  const got = await score(
    '__score__ = logdensityof(logweighted(fn(logdensityof(iid(Normal(0, 1), 3), _)), '
    + 'Lebesgue(support = cartpow(interval(-10.0, 10.0), 3))), [0.5, -1.0, 2.0])');
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
});

test('the plant-growth posterior scores its log prior plus its log-likelihood', async () => {
  // Weights of R's PlantGrowth, three groups of ten. The log-likelihood at the
  // preset point is the closed-form normal sum, -26.815734505382686.
  const src = `
weight_data = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69, 6.31, 5.12, 5.54, 5.50, 5.37, 5.29, 4.92, 6.15, 5.80, 5.26]
group = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
mu = elementof(reals)
effects = elementof(zerosum(3))
sigma = elementof(posreals)
weight ~ Normal.(mu .+ effects[group], sigma)
K_model = kernelof(record(weight = weight), mu = mu, effects = effects, sigma = sigma)
L = likelihoodof(K_model, record(weight = weight_data))
effects_prior = logweighted(fn(logdensityof(iid(Normal(0, 1), 3), _)), Lebesgue(support = zerosum(3)))
Pi_prior = joint(mu = Normal(5, 2), effects = effects_prior, sigma = Exponential(1))
Pi_post = bayesupdate(L, Pi_prior)
__score__ = logdensityof(Pi_post, record(mu = 5.073, effects = [-0.041, -0.412, 0.453], sigma = 0.6))
`;
  const logPrior = -0.5 * Math.log(2 * Math.PI * 4) - (5.073 - 5) ** 2 / 8
    + [-0.041, -0.412, 0.453].reduce((acc, x) => acc + normalLogpdf(x, 1), 0)
    - 0.6;
  const want = logPrior + -26.815734505382686;
  const got = await score(src);
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
});
