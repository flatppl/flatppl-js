'use strict';

// §06 Joint composition: constructor coordinates are fresh. Reified stochastic
// nodes and constructor parameters retain their identity through nesting.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function covariance(a: Float64Array, b: Float64Array): number {
  const ma = a.reduce((s, x) => s + x, 0) / a.length;
  const mb = b.reduce((s, x) => s + x, 0) / b.length;
  return a.reduce((s, x, i) => s + (x - ma) * (b[i] - mb), 0) / a.length;
}

for (const rootSeed of [101, 202]) {
  test(`nested joint coordinates match the independent Normal law, seed ${rootSeed}`, async () => {
    const { ctx } = makeMatCtx(`
q = Normal(0.0, 1.0)
m = joint(u = joint(a = q, b = q), c = q)
lp = logdensityof(m, record(u = record(a = 0.0, b = 1.0), c = 2.0))
`, { sampleCount: 30000, rootSeed });
    const m = await ctx.getMeasure('m');
    const cols: Float64Array[] = [m.fields.u.fields.a.samples, m.fields.u.fields.b.samples, m.fields.c.samples];
    for (let i = 0; i < cols.length; i++) {
      for (let j = i + 1; j < cols.length; j++) {
        assert.ok(Math.abs(covariance(cols[i], cols[j])) < 0.06,
          'independent coordinates have covariance zero, not one');
      }
    }
    const lp = await ctx.getMeasure('lp');
    const oracle = -1.5 * Math.log(2 * Math.PI) - (0 + 1 + 4) / 2;
    assert.ok(Math.abs(lp.samples[0] - oracle) < 1e-12);
  });
}

test('nested constructor wrappers preserve the same stochastic parameter', async () => {
  const { ctx } = makeMatCtx(`
z ~ Normal(0.5, 2.0)
q = Normal(z, 0.6)
r = normalize(q)
m = joint(u = joint(a = q, b = q), c = r)
`, { sampleCount: 30000, rootSeed: 303 });
  const m = await ctx.getMeasure('m');
  const z = (await ctx.getMeasure('z')).samples;
  const cols: Float64Array[] = [m.fields.u.fields.a.samples, m.fields.u.fields.b.samples, m.fields.c.samples];
  const noise = cols.map(xs => Float64Array.from(xs, (x, i) => x - z[i]));
  // Each coordinate is this same z plus independent N(0,.6) noise.
  for (let i = 0; i < cols.length; i++) {
    assert.ok(Math.abs(covariance(noise[i], noise[i]) - 0.36) < 0.03);
    for (let j = i + 1; j < cols.length; j++) {
      assert.ok(Math.abs(covariance(cols[i], cols[j]) - 4) < 0.2);
      assert.ok(Math.abs(covariance(noise[i], noise[j])) < 0.03);
    }
  }
});

for (const fields of ['a = L, b = q', 'b = q, a = L']) {
  test(`a reified coordinate stays shared beside a fresh constructor: ${fields}`, async () => {
    const { ctx } = makeMatCtx(`
q = Normal(0.0, 1.0)
x ~ q
L = lawof(x)
m = joint(${fields})
same = joint(a = L, b = L)
`, { sampleCount: 30000, rootSeed: 404 });
    const m = await ctx.getMeasure('m');
    const x = (await ctx.getMeasure('x')).samples;
    assert.ok(m.fields.a.samples.every((v: number, i: number) => v === x[i]));
    assert.ok(Math.abs(covariance(m.fields.a.samples, m.fields.b.samples)) < 0.06);
    const same = await ctx.getMeasure('same');
    assert.ok(same.fields.a.samples.every((v: number, i: number) => v === x[i]));
    assert.ok(same.fields.b.samples.every((v: number, i: number) => v === x[i]));
  });
}

test('projection of a reified draw retains its coordinate identity', async () => {
  const { ctx } = makeMatCtx(`
M = joint(a = Normal(0.0, 1.0), b = Normal(5.0, 1.0))
x ~ M
P = pushfwd(fn(get(_, "a")), lawof(x))
j = joint(x = lawof(x), p = P)
`, { sampleCount: 256, rootSeed: 606 });
  const j = await ctx.getMeasure('j');
  assert.deepEqual(j.fields.p.samples, j.fields.x.fields.a.samples);
});

test('separate joint bindings have distinct streams with order-independent replay', async () => {
  const source = 'q = Normal(0.0, 1.0)\nj1 = joint(a = q, b = q)\nj2 = joint(a = q, b = q)';
  const first = makeMatCtx(source, { sampleCount: 30000, rootSeed: 505 }).ctx;
  const a = await first.getMeasure('j1');
  const b = await first.getMeasure('j2');
  assert.ok(Math.abs(covariance(a.fields.a.samples, b.fields.a.samples)) < 0.06);
  const reverse = makeMatCtx(source, { sampleCount: 30000, rootSeed: 505 }).ctx;
  const rb = await reverse.getMeasure('j2');
  const ra = await reverse.getMeasure('j1');
  assert.ok(ra.fields.a.samples.every((v: number, i: number) => v === a.fields.a.samples[i]));
  assert.ok(rb.fields.a.samples.every((v: number, i: number) => v === b.fields.a.samples[i]));
});
