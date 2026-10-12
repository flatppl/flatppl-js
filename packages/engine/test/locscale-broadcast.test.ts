'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('a Student row law scores the reified forward kernel at its supplied inputs', async () => {
  const source = `
a = elementof(reals)
b = elementof(reals)
c = elementof(posreals)
response(mu, nu, sigma) = locscale(StudentT(nu), mu, sigma)
y ~ response.([a, b], [5, 9], [c, -c])
K = kernelof(record(y = y), a = a, b = b, c = c)
`;
  const score = async (a: number, b: number, c: number) => {
    const { ctx } = makeMatCtx(source + `
r = logdensityof(K(${a}, ${b}, ${c}), record(y = [0, 0]))
`, { sampleCount: 1 });
    return (await ctx.getMeasure('r')).samples[0];
  };
  // Student-t normalizers from half-integer Gamma identities, independent of
  // the engine. The gradient differentiates that same analytic log density.
  const expected = Math.log(8 / (3 * Math.PI * Math.sqrt(5))) - 3 * Math.log1p(0.05)
    + Math.log(128 / (105 * Math.PI)) - 5 * Math.log1p(1 / 9) - 2 * Math.log(2);
  assert.ok(Math.abs(await score(1, 2, 2) - expected) < 1e-12);
  const point = [1, 2, 2], gradient = [-2 / 7, -0.5, -5 / 14], h = 1e-5;
  for (let i = 0; i < 3; i++) {
    const lo = point.slice(), hi = point.slice();
    lo[i] -= h; hi[i] += h;
    const actual = (await score(hi[0], hi[1], hi[2])
      - await score(lo[0], lo[1], lo[2])) / (2 * h);
    assert.ok(Math.abs(actual - gradient[i]) < 1e-8);
  }
});

test('row transforms redraw the held base while sharing a captured value draw', async () => {
  const { ctx } = makeMatCtx(`
base = Uniform(interval(0, 1))
shift ~ Normal(0, 2)
row(a, s) = locscale(base, a + shift, s)
y ~ row.([1, 3], [-2, 4])
`, { sampleCount: 1024, rootSeed: 117 });
  const shift = (await ctx.getMeasure('shift')).samples;
  const y = (await ctx.getMeasure('y')).value;
  assert.deepEqual(y.shape, [1024, 2]);
  const sums = [0, 0], squares = [0, 0];
  let cross = 0;
  for (let i = 0; i < shift.length; i++) {
    const u = [(y.data[2 * i] - shift[i] - 1) / -2,
      (y.data[2 * i + 1] - shift[i] - 3) / 4];
    for (let j = 0; j < 2; j++) {
      assert.ok(u[j] >= 0 && u[j] <= 1);
      sums[j] += u[j]; squares[j] += u[j] * u[j];
    }
    cross += u[0] * u[1];
  }
  for (let j = 0; j < 2; j++) {
    assert.ok(Math.abs(sums[j] / shift.length - 0.5) < 0.04);
    assert.ok(Math.abs(squares[j] / shift.length - 1 / 3) < 0.04);
  }
  assert.ok(Math.abs(cross / shift.length - sums[0] * sums[1] / shift.length ** 2) < 0.015);
});

test('bare location-scale broadcasts hold a discrete base and preserve each atom mass', async () => {
  const { ctx } = makeMatCtx(`
M = locscale.(Bernoulli(0.3), [0.25, 0.5], [-0.5, 2])
x ~ M
r = logdensityof(M, [-0.25, 0.5])
`, { sampleCount: 256, rootSeed: 117 });
  const r = (await ctx.getMeasure('r')).samples;
  assert.ok(Math.abs(r[0] - Math.log(0.3 * 0.7)) < 1e-12);
  const x = (await ctx.getMeasure('x')).value;
  assert.deepEqual(x.shape, [256, 2]);
  let first = 0, second = 0;
  for (let i = 0; i < 256; i++) {
    assert.ok(x.data[2 * i] === 0.25 || x.data[2 * i] === -0.25);
    assert.ok(x.data[2 * i + 1] === 0.5 || x.data[2 * i + 1] === 2.5);
    first += x.data[2 * i] === -0.25 ? 1 : 0;
    second += x.data[2 * i + 1] === 2.5 ? 1 : 0;
  }
  assert.ok(Math.abs(first / 256 - 0.3) < 0.1);
  assert.ok(Math.abs(second / 256 - 0.3) < 0.1);
});

test('a callback broadcast with no collection inputs keeps a scalar variate', async () => {
  const { ctx } = makeMatCtx(`
K(a, s) = locscale(Uniform(interval(0, 1)), a, s)
x ~ K.(1, 2)
r = logdensityof(lawof(x), 2)
`, { sampleCount: 128, rootSeed: 117 });
  const x = (await ctx.getMeasure('x')).value;
  assert.deepEqual(x.shape, [128]);
  assert.ok(x.data.every((v: number) => v >= 1 && v <= 3));
  assert.ok(Math.abs((await ctx.getMeasure('r')).samples[0] + Math.log(2)) < 1e-12);
});

test('explicit builtin row transforms preserve captured base parameters and mass', async () => {
  const source = `
shift = 7
scale = 3
locscale(m, a, b) = a + b
M = base.locscale.(Normal(shift, scale), [0, 0], 1)
x ~ M
r = logdensityof(M, [7, 7])
`;
  const { ctx } = makeMatCtx(source, { sampleCount: 512, rootSeed: 117 });
  assert.ok(Math.abs((await ctx.getMeasure('r')).samples[0]
    + Math.log(2 * Math.PI) + 2 * Math.log(3)) < 1e-12);
  const x = (await ctx.getMeasure('x')).value;
  assert.deepEqual(x.shape, [512, 2]);
  for (let j = 0; j < 2; j++) {
    let sum = 0;
    for (let i = 0; i < 512; i++) sum += x.data[2 * i + j];
    assert.ok(Math.abs(sum / 512 - 7) < 0.5);
  }
  // Qualifying the builtin must not hide the base's non-unit mass from draw.
  assert.throws(() => makeMatCtx(source.replace('Normal(shift, scale)',
    'weighted(2, Normal(shift, scale))')));
});

test('scalar row materialisation cannot silently truncate a vector variate', async () => {
  const { ctx } = makeMatCtx(`
M = iid(Normal(0, 1), 2)
K(a) = locscale(M, [a, a], eye(2))
x ~ K.([1, 2])
`, { sampleCount: 4, rootSeed: 117 });
  await assert.rejects(ctx.getMeasure('x'), /vector-variate/);
});
