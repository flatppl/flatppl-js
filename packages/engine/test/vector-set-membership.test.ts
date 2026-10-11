'use strict';

// Membership in the §03 vector sets `stdsimplex(n)`, `ordered(n)` and
// `zerosum(n)`, through `x in S`, `Lebesgue(support = S)`, a weighted Lebesgue
// prior and `truncate`. Only the inequalities are checked (simplex entries
// >= 0, ordered entries strictly increasing). §03 makes the Lebesgue measure on
// `stdsimplex(n)` and `zerosum(n)` the coordinate measure of the n-1 free
// coordinates, so the linear equality is not tested. Expected values are by hand.

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

test('`v in stdsimplex(3)` checks signs, not the unit sum', async () => {
  const inSet = async (v: string) => scoreValue(`lp = ifelse(${v} in stdsimplex(3), 1.0, 0.0)`, 'lp');
  assert.equal(await inSet('[0.2, 0.3, 0.5]'), 1.0);
  assert.equal(await inSet('[0.2, 0.3, 0.6]'), 1.0);
  assert.equal(await inSet('[-0.1, 0.6, 0.5]'), 0.0);
});

test('`v in zerosum(3)` does not check the zero sum', async () => {
  const inSet = async (v: string) => scoreValue(`lp = ifelse(${v} in zerosum(3), 1.0, 0.0)`, 'lp');
  assert.equal(await inSet('[0.5, 0.25, -0.75]'), 1.0);
  assert.equal(await inSet('[0.5, 0.25, -0.7]'), 1.0);
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

test('Lebesgue(support = zerosum(3)) scores 0 off the zero sum too: equality not checked', async () => {
  assert.equal(await lebesgue('zerosum(3)', '[0.5, 0.25, -0.75]'), 0);
  assert.equal(await lebesgue('zerosum(3)', '[0.5, 0.25, -0.7]'), 0);
});

test('Lebesgue(support = stdsimplex(3)) scores -inf at a negative entry', async () => {
  assert.equal(await lebesgue('stdsimplex(3)', '[0.2, 0.3, 0.5]'), 0);
  assert.equal(await lebesgue('stdsimplex(3)', '[-0.1, 0.6, 0.5]'), -Infinity);
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
L = likelihoodof(K, record(z = 0.0))
posterior = bayesupdate(L, prior)
`;
  const { ctx } = ctxFor(SRC, 50);
  await assert.rejects(
    () => materialiser.materialiseMeasure('posterior', ctx, { backend: 'mh', warmup: 5, draws: 5 }),
    /truncate\(ordered\(3\)\)/);
});
