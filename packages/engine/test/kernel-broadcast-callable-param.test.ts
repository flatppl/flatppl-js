'use strict';

// A kernel broadcast whose parameter broadcasts a user function,
// `y ~ Normal.(g.(z), 1.0)`, got no derivation. The cascade-prune read every
// self-ref of the parameter as a dependency, including `g`, which sits in the
// callable-head position of `broadcast(g, z)` and has neither a derivation nor
// a fixed value. The two-argument form `g.(z, z)` worked only because the
// dissolver had rewritten `broadcast(g, z, z)` to `add(z, z)`, and the
// bound-first form worked because its parameter was a plain `ref m`. The prune
// now leaves callable heads out in every arm, as the sample and weighted arms
// already did.
//
// Closed form: z_i ~ N(0, 1) and y_i | z ~ N(2 z_i, 1), so
//   log p = Σ log N(z_i; 0, 1) + Σ log N(y_i; 2 z_i, 1).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const lnN = (x: number, m: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - m) ** 2;
const closedForm = (z: number[], y: number[]) =>
  z.reduce((s, zi, i) => s + lnN(zi, 0) + lnN(y[i], 2 * zi), 0);

const FORMS: Record<string, string> = {
  'one argument': 'g(a) = 2.0 * a\nz ~ iid(Normal(0.0, 1.0), 3)\ny ~ Normal.(g.(z), 1.0)\n',
  'two arguments': 'g(a, b) = a + b\nz ~ iid(Normal(0.0, 1.0), 3)\ny ~ Normal.(g.(z, z), 1.0)\n',
  'bound first': 'g(a) = 2.0 * a\nz ~ iid(Normal(0.0, 1.0), 3)\nm = g.(z)\ny ~ Normal.(m, 1.0)\n',
};

async function score(src: string, z: number[], y: number[]): Promise<number> {
  const { ctx, proc } = ctxFor(src + 'Pi = lawof(record(z = z, y = y))\n'
    + `lp = logdensityof(Pi, record(z = ${JSON.stringify(z)}, y = ${JSON.stringify(y)}))\n`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  return (await ctx.getMeasure('lp')).samples[0];
}

const Z = [0.0, 0.1, 0.2];
const Y = [0.3, 0.3, 0.3];

for (const [label, src] of Object.entries(FORMS)) {
  test(`a user function broadcast in a kernel broadcast parameter scores (${label})`, async () => {
    const got = await score(src, Z, Y);
    assert.ok(Math.abs(got - (-5.593631199228037)) < 1e-12, `${got}`);
    assert.ok(Math.abs(closedForm(Z, Y) - (-5.593631199228037)) < 1e-12);
  });
}

test('the one-argument form moves with the closed form gradient', async () => {
  // Central differences of the engine density along each coordinate, against
  // the analytic gradient: d/dz_i = -z_i + 2 (y_i - 2 z_i), d/dy_i = -(y_i - 2 z_i).
  const src = FORMS['one argument'];
  const h = 1e-5;
  for (let i = 0; i < 3; i++) {
    const zp = Z.slice(), zm = Z.slice();
    zp[i] += h; zm[i] -= h;
    const dz = (await score(src, zp, Y) - await score(src, zm, Y)) / (2 * h);
    assert.ok(Math.abs(dz - (-Z[i] + 2 * (Y[i] - 2 * Z[i]))) < 1e-7, `dz_${i} ${dz}`);
    const yp = Y.slice(), ym = Y.slice();
    yp[i] += h; ym[i] -= h;
    const dy = (await score(src, Z, yp) - await score(src, Z, ym)) / (2 * h);
    assert.ok(Math.abs(dy - (-(Y[i] - 2 * Z[i]))) < 1e-7, `dy_${i} ${dy}`);
  }
});

test('the one-argument form samples y with variance 1 + 2^2', async () => {
  const N = 20000;
  const { ctx } = ctxFor(FORMS['one argument'], N);
  const y = await ctx.getMeasure('y');
  assert.deepEqual(y.value.shape, [N, 3]);
  const data = y.value.data;
  let s = 0, s2 = 0;
  for (let i = 0; i < data.length; i++) { s += data[i]; s2 += data[i] * data[i]; }
  const mean = s / data.length, variance = s2 / data.length - mean * mean;
  assert.ok(Math.abs(mean) < 0.05, `mean ${mean}`);
  assert.ok(Math.abs(variance - 5) < 0.15, `variance ${variance}`);
});
