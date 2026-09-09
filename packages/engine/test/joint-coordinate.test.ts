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

test('a fresh draw copies a reified trace, while joining its laws retains identity', async () => {
  const { ctx } = makeMatCtx(`
x ~ Normal(0.0, 1.0)
f = functionof(lawof(2.0 * x))
M = f()
y ~ M
z ~ M
traced = joint(x = lawof(x), y = M)
drawn = lawof(record(x = x, y = y, z = z))
`, { sampleCount: 30000, rootSeed: 707 });
  const traced = await ctx.getMeasure('traced');
  assert.deepEqual(traced.fields.y.samples,
    Float64Array.from(traced.fields.x.samples, (x: number) => 2 * x));
  const drawn = await ctx.getMeasure('drawn');
  const { x, y, z } = drawn.fields;
  assert.ok(Math.abs(covariance(y.samples, y.samples) - 4) < 0.15);
  for (const [a, b] of [[x, y], [x, z], [y, z]]) {
    assert.ok(Math.abs(covariance(a.samples, b.samples)) < 0.1,
      'fresh draws have zero covariance with each other and the original trace');
  }
});

test('separate draws from a constructor share its parameter, not its noise', async () => {
  const { ctx } = makeMatCtx(`
p ~ Normal(0.0, 2.0)
M = Normal(p, 1.0)
a ~ M
b ~ M
`, { sampleCount: 30000, rootSeed: 808 });
  const p = (await ctx.getMeasure('p')).samples;
  const a = (await ctx.getMeasure('a')).samples;
  const b = (await ctx.getMeasure('b')).samples;
  const ea = Float64Array.from(a, (x: number, i: number) => x - p[i]);
  const eb = Float64Array.from(b, (x: number, i: number) => x - p[i]);
  assert.ok(Math.abs(covariance(ea, ea) - 1) < 0.05);
  assert.ok(Math.abs(covariance(eb, eb) - 1) < 0.05);
  assert.ok(Math.abs(covariance(ea, eb)) < 0.05);
  assert.ok(Math.abs(covariance(a, b) - 4) < 0.15);
});

test('a fresh joint draw retains external ancestors shared with a reified component', async () => {
  const { ctx } = makeMatCtx(`
p ~ Normal(0.0, 2.0)
x ~ Normal(p, 1.0)
M = joint(a = lawof(x), b = Normal(p, 1.0), p = lawof(p))
y ~ M
`, { sampleCount: 30000, rootSeed: 909 });
  const p = (await ctx.getMeasure('p')).samples;
  const x = (await ctx.getMeasure('x')).samples;
  const y = (await ctx.getMeasure('y')).fields;
  assert.deepEqual(y.p.samples, p);
  const residual = (xs: Float64Array) => Float64Array.from(xs, (v, i) => v - p[i]);
  assert.ok(Math.abs(covariance(y.a.samples, y.a.samples) - 5) < 0.15);
  assert.ok(Math.abs(covariance(y.a.samples, y.b.samples) - 4) < 0.15);
  assert.ok(Math.abs(covariance(residual(y.a.samples), residual(x))) < 0.05);
  assert.ok(Math.abs(covariance(residual(y.a.samples), residual(y.b.samples))) < 0.05);
});

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
