'use strict';

// §06 products scope each law independently. A free constructor parameter is
// shared; repeated coordinates inside one §04 law retain their dependence.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

function covariance(a: Float64Array, b: Float64Array): number {
  const ma = a.reduce((s, x) => s + x, 0) / a.length;
  const mb = b.reduce((s, x) => s + x, 0) / b.length;
  return a.reduce((s, x, i) => s + (x - ma) * (b[i] - mb), 0) / a.length;
}

test('products and fresh draws use independent copies of a law', async () => {
  const { ctx } = makeMatCtx(`
x ~ Normal(0.0, 1.0)
M = lawof(2.0 * x)
y ~ M
z ~ M
product = joint(x = lawof(x), y = M)
drawn = lawof(record(x = x, y = y, z = z))
diagonal = lawof(record(x = x, y = 2.0 * x))
`, { sampleCount: 4096, rootSeed: 707 });
  const product = (await ctx.getMeasure('product')).fields;
  assert.ok(Math.abs(covariance(product.x.samples, product.y.samples)) < 0.2);
  assert.ok(Math.abs(covariance(product.y.samples, product.y.samples) - 4) < 0.5);
  const drawn = await ctx.getMeasure('drawn');
  const { x, y, z } = drawn.fields;
  for (const [a, b] of [[x, y], [x, z], [y, z]]) {
    assert.ok(Math.abs(covariance(a.samples, b.samples)) < 0.4,
      'fresh draws have zero covariance with each other and the original trace');
  }
  const diagonal = (await ctx.getMeasure('diagonal')).fields;
  assert.ok(diagonal.y.samples.every((v: number, i: number) =>
    v === 2 * diagonal.x.samples[i]));
});

test('separate draws from a constructor share its parameter, not its noise', async () => {
  const { ctx } = makeMatCtx(`
p ~ Normal(0.0, 2.0)
M = Normal(p, 1.0)
a ~ M
b ~ M
`, { sampleCount: 4096, rootSeed: 808 });
  const p = (await ctx.getMeasure('p')).samples;
  const a = (await ctx.getMeasure('a')).samples;
  const b = (await ctx.getMeasure('b')).samples;
  const ea = Float64Array.from(a, (x: number, i: number) => x - p[i]);
  const eb = Float64Array.from(b, (x: number, i: number) => x - p[i]);
  assert.ok(Math.abs(covariance(ea, ea) - 1) < 0.15);
  assert.ok(Math.abs(covariance(eb, eb) - 1) < 0.15);
  assert.ok(Math.abs(covariance(ea, eb)) < 0.1);
  assert.ok(Math.abs(covariance(a, b) - 4) < 0.6);
});

test('a product separates absorbed ancestors from its free constructor parameter', async () => {
  const { ctx } = makeMatCtx(`
p ~ Normal(0.0, 2.0)
x ~ Normal(p, 1.0)
M = joint(a = lawof(x), b = Normal(p, 1.0), p = lawof(p))
y ~ M
`, { sampleCount: 8192, rootSeed: 909 });
  const p = (await ctx.getMeasure('p')).samples;
  const y = (await ctx.getMeasure('y')).fields;
  assert.ok(Math.abs(covariance(y.a.samples, y.a.samples) - 5) < 0.5);
  assert.ok(Math.abs(covariance(y.a.samples, y.b.samples)) < 0.4);
  assert.ok(Math.abs(covariance(y.p.samples, p)) < 0.4);
  assert.ok(Math.abs(covariance(y.b.samples, p) - 4) < 0.5);
});

test('nested products have independent coordinates and their product density', async () => {
    const { ctx } = makeMatCtx(`
q = Normal(0.0, 1.0)
m = joint(u = joint(a = q, b = q), c = q)
lp = logdensityof(m, record(u = record(a = 0.0, b = 1.0), c = 2.0))
`, { sampleCount: 4096, rootSeed: 101 });
    const m = await ctx.getMeasure('m');
    const cols: Float64Array[] = [m.fields.u.fields.a.samples, m.fields.u.fields.b.samples, m.fields.c.samples];
    for (let i = 0; i < cols.length; i++) {
      for (let j = i + 1; j < cols.length; j++) {
        assert.ok(Math.abs(covariance(cols[i], cols[j])) < 0.1,
          'independent coordinates have covariance zero, not one');
      }
    }
    const lp = await ctx.getMeasure('lp');
    const oracle = -1.5 * Math.log(2 * Math.PI) - (0 + 1 + 4) / 2;
    assert.ok(Math.abs(lp.samples[0] - oracle) < 1e-12);
});

test('nested constructor wrappers preserve the same stochastic parameter', async () => {
  const { ctx } = makeMatCtx(`
z ~ Normal(0.5, 2.0)
q = Normal(z, 0.6)
r = normalize(q)
m = joint(u = joint(a = q, b = q), c = r)
`, { sampleCount: 4096, rootSeed: 303 });
  const m = await ctx.getMeasure('m');
  const z = (await ctx.getMeasure('z')).samples;
  const cols: Float64Array[] = [m.fields.u.fields.a.samples, m.fields.u.fields.b.samples, m.fields.c.samples];
  const noise = cols.map(xs => Float64Array.from(xs, (x, i) => x - z[i]));
  // Each coordinate is this same z plus independent N(0,.6) noise.
  for (let i = 0; i < cols.length; i++) {
    assert.ok(Math.abs(covariance(noise[i], noise[i]) - 0.36) < 0.06);
    for (let j = i + 1; j < cols.length; j++) {
      assert.ok(Math.abs(covariance(cols[i], cols[j]) - 4) < 0.6);
      assert.ok(Math.abs(covariance(noise[i], noise[j])) < 0.05);
    }
  }
});

test('a projected law is independent of another copy of its source law', async () => {
  const { ctx } = makeMatCtx(`
M = joint(a = Normal(0.0, 1.0), b = Normal(5.0, 1.0))
x ~ M
P = pushfwd(fn(get(_, "a")), lawof(x))
j = joint(x = lawof(x), p = P)
`, { sampleCount: 4096, rootSeed: 606 });
  const j = await ctx.getMeasure('j');
  assert.ok(Math.abs(covariance(j.fields.p.samples, j.fields.x.fields.a.samples)) < 0.1);
});

test('separate joint bindings have distinct streams with order-independent replay', async () => {
  const source = 'q = Normal(0.0, 1.0)\nj1 = joint(a = q, b = q)\nj2 = joint(a = q, b = q)';
  const first = makeMatCtx(source, { sampleCount: 256, rootSeed: 505 }).ctx;
  const a = await first.getMeasure('j1');
  const b = await first.getMeasure('j2');
  assert.ok(a.fields.a.samples.some((v: number, i: number) => v !== b.fields.a.samples[i]));
  const reverse = makeMatCtx(source, { sampleCount: 256, rootSeed: 505 }).ctx;
  const rb = await reverse.getMeasure('j2');
  const ra = await reverse.getMeasure('j1');
  assert.ok(ra.fields.a.samples.every((v: number, i: number) => v === a.fields.a.samples[i]));
  assert.ok(rb.fields.a.samples.every((v: number, i: number) => v === b.fields.a.samples[i]));
});
