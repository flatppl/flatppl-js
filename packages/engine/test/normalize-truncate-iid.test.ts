'use strict';

// `normalize(truncate(iid(D, n), cartpow(S, n)))` is a valid model: the box
// truncation of an iid plate normalised by its total mass. It used to refuse
// with "sampler: 'iid' is not a known distribution", because the truncation
// mass and its rejection sampler only knew a scalar base leaf. The broadcast
// scorer, in turn, normalised an iid plate (or a weighted leaf) with Z = 1.
//
// ORACLE: closed form. The box [0,5]^2 under iid Normal(0, 1) has mass
// Z^2 with Z = Phi(5) - Phi(0) = 0.4999997133484282, so
//   log Z                 -0.6931477538632532
//   Z^2                   0.24999971334851034
//   normalised at [1, 2]  log N(1) + log N(2) - 2 log Z = -2.9515815586828387
//   normalised at [.5, 1] log N(.5) + log N(1) - 2 log Z = -1.0765815586828387
// and each coordinate is a Normal(0, 1) truncated to [0, 5]: mean
// (phi(0) - phi(5)) / Z = 0.7978820447921204.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const LOG_Z = -0.6931477538632532;
const Z2 = 0.24999971334851034;
const logN = (x: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * x * x;

const NORM = 'normalize(truncate(iid(Normal(0, 1), 2), cartpow(interval(0, 5), 2)))';

async function scalar(src: string, name: string) {
  const { ctx } = ctxFor(src, 1);
  return (await ctx.getMeasure(name)).samples[0];
}

function near(got: number, want: number, tol = 1e-12) {
  assert.ok(Math.abs(got - want) < tol, `got ${got}, want ${want}`);
}

test('the normalised density divides the box density by Z^2', async () => {
  near(await scalar(`M = ${NORM}\ns = logdensityof(M, [1.0, 2.0])\n`, 's'), -2.9515815586828387);
  near(await scalar(`M = ${NORM}\ns = logdensityof(M, [0.5, 1.0])\n`, 's'), -1.0765815586828387);
  near(await scalar(`M = ${NORM}\ns = logdensityof(M, [0.5, 1.0])\n`, 's'),
    logN(0.5) + logN(1) - 2 * LOG_Z);
});

test('a coordinate outside the box is -Infinity', async () => {
  assert.equal(await scalar(`M = ${NORM}\ns = logdensityof(M, [1.0, 9.0])\n`, 's'), -Infinity);
  assert.equal(await scalar(`M = ${NORM}\ns = logdensityof(M, [-1.0, 1.0])\n`, 's'), -Infinity);
});

test('totalmass is Z^2 raw and 1 normalised', async () => {
  const raw = 'truncate(iid(Normal(0, 1), 2), cartpow(interval(0, 5), 2))';
  near(await scalar(`M = ${raw}\nt = totalmass(M)\n`, 't'), Z2);
  near(await scalar(`M = ${NORM}\nt = totalmass(M)\n`, 't'), 1);
});

test('a named iid base normalises the same way', async () => {
  const src = 'D = iid(Normal(0, 1), 2)\nM = normalize(truncate(D, cartpow(interval(0, 5), 2)))\n'
    + 's = logdensityof(M, [1.0, 2.0])\nt = totalmass(truncate(D, cartpow(interval(0, 5), 2)))\n';
  near(await scalar(src, 's'), -2.9515815586828387);
  near(await scalar(src, 't'), Z2);
});

test('a named element set and a matrix size normalise by Z^count', async () => {
  // posreals keeps half the mass of each coordinate: Z = 1/2.
  near(await scalar('M = normalize(truncate(iid(Normal(0, 1), 3), cartpow(posreals, 3)))\n'
    + 's = logdensityof(M, [1.0, 2.0, 0.5])\n', 's'), -3.3023740579341827);
  near(await scalar('M = normalize(truncate(iid(Normal(0, 1), [2, 2]), cartpow(interval(0, 5), [2, 2])))\n'
    + 's = logdensityof(M, rowstack([[1.0, 2.0], [0.5, 1.0]]))\n', 's'), -4.028163117365677);
});

test('the unit interval box has mass (Phi(1) - Phi(0))^2', async () => {
  const src = 'M = truncate(iid(Normal(0, 1), 2), cartpow(unitinterval, 2))\n'
    + 't = totalmass(M)\ns = logdensityof(M, [0.5, 0.25])\n';
  near(await scalar(src, 't'), 0.11651623566859805);
  near(await scalar(src, 's'), logN(0.5) + logN(0.25));
});

test('draws are truncated Normal(0, 1) on [0, 5] in every coordinate', async () => {
  const { ctx } = ctxFor(`M = ${NORM}\nx ~ M\n`, 20000);
  const m = await ctx.getMeasure('x');
  const data = m.value ? m.value.data : m.samples;
  const n = data.length / 2;
  const sum = [0, 0];
  for (let i = 0; i < data.length; i++) {
    assert.ok(data[i] >= 0 && data[i] <= 5, `draw ${data[i]} outside [0, 5]`);
    sum[i % 2] += data[i];
  }
  // sd of one draw is 0.6028, so 20000 draws give a standard error of 0.0043.
  for (const s of sum) assert.ok(Math.abs(s / n - 0.7978820447921204) < 0.025, `mean ${s / n}`);
});

test('a set whose size differs from the plate still refuses', async () => {
  await assert.rejects(scalar('M = truncate(iid(Normal(0, 1), 3), cartpow(interval(0, 5), 2))\n'
    + 't = totalmass(M)\n', 't'));
});

async function batched(measure: string, pts: string) {
  const { ctx } = ctxFor(`M = ${measure}\nlp = broadcast(logdensityof, M, ${pts})\n`, 1);
  const m = await ctx.getMeasure('lp');
  return Array.from(m.value ? m.value.data : m.samples) as number[];
}

const PTS = '[[1.0, 2.0], [0.5, 1.0], [3.0, 0.1]]';
const NORMALISED = [-2.9515815586828387, -1.0765815586828387,
  logN(3) + logN(0.1) - 2 * LOG_Z];

test('broadcast(logdensityof, ...) normalises an iid plate', async () => {
  const got = await batched(NORM, PTS);
  NORMALISED.forEach((want, i) => near(got[i], want));
  const plate = await batched(
    'normalize(iid(truncate(Normal(0, 1), interval(0, 5)), 2))', PTS);
  NORMALISED.forEach((want, i) => near(plate[i], want));
});

test('broadcast(logdensityof, ...) normalises a weighted leaf', async () => {
  // normalize(weighted(2, Normal(0, 1))) is Normal(0, 1): the weight cancels.
  const got = await batched('normalize(weighted(2.0, Normal(0, 1)))', '[0.5, 1.0]');
  near(got[0], logN(0.5));
  near(got[1], logN(1));
  const plate = await batched('normalize(iid(weighted(2.0, Normal(0, 1)), 2))', PTS);
  near(plate[0], logN(1) + logN(2));
  near(plate[1], logN(0.5) + logN(1));
});

test('a normalised weighted measure whose base cannot be resolved refuses', () => {
  const density = require('../density.ts');
  const normalized = { kind: 'call', op: 'normalize', args: [{ kind: 'call', op: 'weighted',
    args: [{ kind: 'lit', value: 2 }, { kind: 'ref', ns: 'self', name: 'base' }] }] };
  assert.throws(() => density.logDensity(normalized, 0.5, {},
    { resolveMeasureRef: () => null }), /normalize\(weighted\).*total mass Z is not known/);
});

test('an iid plate counts its copies from a plain-array shape as well as a typed one', () => {
  const density = require('../density.ts');
  const normal = { kind: 'call', op: 'Normal', args: [{ kind: 'lit', value: 0 }, { kind: 'lit', value: 1 }] };
  const plate = { kind: 'call', op: 'iid', args: [normal, { kind: 'ref', ns: 'self', name: 'shape' }] };
  const want = [0.1, 0.2, 0.3, 0.4].reduce((sum, x) => sum + logN(x), 0);
  for (const shape of [[2, 2], Float64Array.from([2, 2])]) {
    near(density.logDensity(plate, [0.1, 0.2, 0.3, 0.4], { shape }, {}), want);
  }
});

test('a box over a joint is not an iid plate and keeps its own route', async () => {
  const got = await scalar('M = truncate(joint(Normal(0, 1), Normal(0, 1)), cartpow(interval(0, 5), 2))\n'
    + 's = logdensityof(M, [1.0, 2.0])\n', 's');
  near(got, logN(1) + logN(2));
});
