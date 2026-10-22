'use strict';

// `broadcast(logdensityof, M, pts)` scores all points in one pass whose N-axis indexes
// the points. `walkPushfwd` consumed the first point as the variate and left the rest,
// so every pushforward over scalar points threw "value has unconsumed leftover after
// walking IR (op=pushfwd)". It now inverts every point, scores the base once over the
// preimages and subtracts the logvolume per point. A point outside the image of f
// (log of a non-positive number) has density 0, where the walker returned NaN.
//
// ORACLE: closed form.
//   exp(Normal(0, 1)) is LogNormal(0, 1): log N(log y) - log y
//   2 x + 1 of Normal(0, 1) is Normal(1, 2)
//   log(Exponential(2)) has density 2 e^y exp(-2 e^y)
//   2 * Poisson(3) puts mass Poisson(y / 2) on even y and none on odd y

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number, m = 0, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
const logPoisson = (k: number, rate: number) => {
  let logFactorial = 0;
  for (let i = 2; i <= k; i++) logFactorial += Math.log(i);
  return k * Math.log(rate) - rate - logFactorial;
};

async function score(measure: string, pts: number[]) {
  const { ctx } = ctxFor(`M = ${measure}\nlp = broadcast(logdensityof, M, [${pts.join(', ')}])\n`
    + pts.map((p, i) => `s${i} = logdensityof(M, ${p})\n`).join(''), 1);
  const m = await ctx.getMeasure('lp');
  const batched = Array.from(m.value ? m.value.data : m.samples) as number[];
  const single: number[] = [];
  for (let i = 0; i < pts.length; i++) single.push((await ctx.getMeasure('s' + i)).samples[0]);
  return { batched, single };
}

function agree(got: number[], want: number[], tol = 1e-12) {
  assert.equal(got.length, want.length);
  want.forEach((w, i) => {
    if (w === -Infinity) assert.equal(got[i], -Infinity, `point ${i}: got ${got[i]}`);
    else assert.ok(Math.abs(got[i] - w) < tol, `point ${i}: got ${got[i]}, want ${w}`);
  });
}

test('exp of a Normal is LogNormal(0, 1) at every point, in the batch and one by one', async () => {
  const pts = [0.5, 1.5, 2.0, 0.1];
  const want = pts.map(y => logN(Math.log(y)) - Math.log(y));
  const { batched, single } = await score('pushfwd(fn(exp(_)), Normal(0, 1))', pts);
  agree(batched, want);
  agree(single, want);
});

test('points outside the image of exp have density 0', async () => {
  const pts = [-1.0, 1.0, 0.0, 2.0];
  const want = [-Infinity, logN(0), -Infinity, logN(Math.log(2)) - Math.log(2)];
  const { batched, single } = await score('pushfwd(fn(exp(_)), Normal(0, 1))', pts);
  agree(batched, want);
  agree(single, want);
});

test('an affine map has the scaled Normal density', async () => {
  const pts = [0.5, 1.5, 2.0, -1.0];
  const want = pts.map(y => logN(y, 1, 2));
  const { batched, single } = await score('pushfwd(fn(2.0 * _ + 1.0), Normal(0, 1))', pts);
  agree(batched, want);
  agree(single, want);
});

test('log of an Exponential carries its logvolume at every point', async () => {
  const pts = [0.5, -1.0, 0.0];
  const want = pts.map(y => Math.log(2) + y - 2 * Math.exp(y));
  const { batched } = await score('pushfwd(fn(log(_)), Exponential(2.0))', pts);
  agree(batched, want);
});

test('a discrete base scores each point on its own', async () => {
  const pts = [0, 2, 4, 3];
  const want = [logPoisson(0, 3), logPoisson(1, 3), logPoisson(2, 3), -Infinity];
  const { batched } = await score('pushfwd(fn(2 * _), Poisson(3.0))', pts);
  agree(batched, want);
});

test('an annotated bijection with a constant logvolume scores a Value of points', () => {
  const density = require('../density.ts');
  const values = require('../value.ts');
  const { parseSetIR } = require('../ir-shared.ts');
  const lit = (value: number) => ({ kind: 'lit', value });
  const local = (name: string) => ({ kind: 'ref', ns: '%local', name });
  // y = 2 x + 1 with the constant forward logvolume log 2.
  const affine = { kind: 'call', op: 'pushfwd',
    args: [{ kind: 'ref', ns: 'self', name: 'f' }, { kind: 'call', op: 'Normal', args: [lit(0), lit(1)] }],
    bijection: { fInv: { paramName: 'y', body: { kind: 'call', op: 'divide',
      args: [{ kind: 'call', op: 'sub', args: [local('y'), lit(1)] }, lit(2)] } },
    logVolume: { kind: 'scalar', value: Math.log(2) } } };
  const pts = [0.5, 1.5, -1.0];
  const got = density.logDensityN(affine, values.vector(Float64Array.from(pts)), null, 3,
    { pointsBatched: true, parseSet: (set: any) => parseSetIR(set, null) });
  pts.forEach((y, i) => assert.ok(Math.abs(got[i] - logN(y, 1, 2)) < 1e-12, `point ${i}: ${got[i]}`));
});
