'use strict';

// Spec §04: `relabel(1.0, ["x"])` is `record(x = 1.0)`. Spec §06: the keyword form
// `jointchain(a = M, b = K)` is `jointchain(relabel(M, ["a"]), relabel(K, ["b"]))`, and a
// record variate splats into the next step's inputs by field name. So over a scalar M the
// kernel K receives `record(a = x)` and binds its input `a` to x. The engine bound the
// whole record to K's single input instead, so `Normal(mu = record(a = x), ...)` scored and
// sampled NaN, while the positional form `jointchain(M, K)` was right.
//
// ORACLE: closed form. a ~ Normal(0, 1) and b | a ~ Normal(a, 0.5) give the joint density
// log N(a; 0, 1) + log N(b; a, 0.5), and b - a ~ Normal(0, 0.5), variance 0.25.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number, m = 0, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

// The spec-valid ways to reify the kernel; flatppl-rust accepts each of them.
const KERNELS = [
  'functionof(Normal(mu = _a_, sigma = 0.5), a = _a_)',
  'kernelof(draw(Normal(mu = _a_, sigma = 0.5)), a = _a_)',
];
const POINTS: Array<[number, number]> = [[0.3, 0.9], [-1.2, -0.7], [2.0, 2.0]];

for (const kernel of KERNELS) {
  test(`keyword jointchain over a scalar Normal scores the product density (${kernel.split('(')[0]})`, async () => {
    const src = `K = ${kernel}\nJ = jointchain(a = Normal(0, 1), b = K)\n`
      + POINTS.map(([a, b], i) => `s${i} = logdensityof(J, record(a = ${a}, b = ${b}))\n`).join('');
    const { ctx } = ctxFor(src, 1);
    for (let i = 0; i < POINTS.length; i++) {
      const [a, b] = POINTS[i];
      const got = (await ctx.getMeasure('s' + i)).samples[0];
      assert.ok(Math.abs(got - (logN(a) + logN(b, a, 0.5))) < 1e-12, `point ${i}: got ${got}`);
    }
  });
}

test('the positional form scores the same joint density', async () => {
  const { ctx } = ctxFor('K = functionof(Normal(mu = _a_, sigma = 0.5), a = _a_)\n'
    + 'J = jointchain(Normal(0, 1), K)\ns = logdensityof(J, [0.3, 0.9])\n', 1);
  const got = (await ctx.getMeasure('s')).samples[0];
  assert.ok(Math.abs(got - (logN(0.3) + logN(0.9, 0.3, 0.5))) < 1e-12);
});

test('draws from the keyword jointchain follow the kernel', async () => {
  const { ctx } = ctxFor('K = functionof(Normal(mu = _a_, sigma = 0.5), a = _a_)\n'
    + 'J = jointchain(a = Normal(0, 1), b = K)\nx ~ J\n', 20000);
  const m = await ctx.getMeasure('x');
  const a = m.fields.a.samples, b = m.fields.b.samples;
  let sumA = 0, sumD = 0, sumDD = 0;
  for (let i = 0; i < a.length; i++) {
    const d = b[i] - a[i];
    sumA += a[i]; sumD += d; sumDD += d * d;
  }
  const n = a.length, meanD = sumD / n;
  // Standard errors: 1 / sqrt(n) = 0.0071 for a, 0.5 / sqrt(n) = 0.0035 for b - a,
  // and 0.25 * sqrt(2 / n) = 0.0025 for the variance of b - a.
  assert.ok(Math.abs(sumA / n) < 0.04, `mean a ${sumA / n}`);
  assert.ok(Math.abs(meanD) < 0.02, `mean b - a ${meanD}`);
  assert.ok(Math.abs(sumDD / n - meanD ** 2 - 0.25) < 0.012, `var b - a ${sumDD / n - meanD ** 2}`);
});
