'use strict';

// Spec §04: `relabel(1.0, ["x"])` is `record(x = 1.0)`. Spec §06: the keyword form
// `jointchain(a = M, b = K)` calls `K(record(a = x))` by the ordinary calling convention (§04
// "Auto-splatting"), so over a scalar M a kernel with one input receives the record whole. A kernel
// that takes the record reads the field, `K(r) = Normal(r.a, 0.5)`. The engine used to bind the whole
// record to a real input, so `Normal(mu = record(a = x), ...)` scored and sampled NaN with no
// diagnostic; that is now a located static error. The positional form `jointchain(M, K)` feeds the
// scalar variate to K.
//
// ORACLE: closed form. a ~ Normal(0, 1) and b | a ~ Normal(a, 0.5) give the joint density
// log N(a; 0, 1) + log N(b; a, 0.5), and b - a ~ Normal(0, 0.5), variance 0.25.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const logN = (x: number, m = 0, s = 1) => -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;

// The ways to write a kernel that takes the record of the preceding variates.
const KERNELS = [
  'K(r) = Normal(mu = r.a, sigma = 0.5)',
  'r = elementof(cartprod(a = reals))\nK = functionof(Normal(mu = r.a, sigma = 0.5), r = r)',
];
const POINTS: Array<[number, number]> = [[0.3, 0.9], [-1.2, -0.7], [2.0, 2.0]];

for (const kernel of KERNELS) {
  test(`keyword jointchain over a scalar Normal scores the product density (${kernel.startsWith('K(') ? 'definition' : 'functionof'})`, async () => {
    const src = `${kernel}\nJ = jointchain(a = Normal(0, 1), b = K)\n`
      + POINTS.map(([a, b], i) => `s${i} = logdensityof(J, record(a = ${a}, b = ${b}))\n`).join('');
    const { ctx } = ctxFor(src, 1);
    for (let i = 0; i < POINTS.length; i++) {
      const [a, b] = POINTS[i];
      const got = (await ctx.getMeasure('s' + i)).samples[0];
      assert.ok(Math.abs(got - (logN(a) + logN(b, a, 0.5))) < 1e-12, `point ${i}: got ${got}`);
    }
  });
}

test('a one-input real kernel receiving the record is a located static error', () => {
  const { processSource } = require('../index.ts');
  // The input's type must be known: `elementof(reals)` declares it (a bare `_a_` placeholder is untyped).
  for (const kernel of ['functionof(Normal(mu = a, sigma = 0.5), a = a)',
    'kernelof(draw(Normal(mu = a, sigma = 0.5)), a = a)']) {
    const errs = processSource(`a = elementof(reals)\nK = ${kernel}\nJ = jointchain(a = Normal(0, 1), b = K)\n`)
      .diagnostics.filter((d: any) => d.severity === 'error');
    assert.equal(errs.length, 1, kernel);
    assert.match(errs[0].message, /record with fields a: real/);
    assert.equal(errs[0].loc.start.line, 2);
  }
});

test('the positional form scores the same joint density', async () => {
  const { ctx } = ctxFor('K = functionof(Normal(mu = _a_, sigma = 0.5), a = _a_)\n'
    + 'J = jointchain(Normal(0, 1), K)\ns = logdensityof(J, [0.3, 0.9])\n', 1);
  const got = (await ctx.getMeasure('s')).samples[0];
  assert.ok(Math.abs(got - (logN(0.3) + logN(0.9, 0.3, 0.5))) < 1e-12);
});

test('draws from the keyword jointchain follow the kernel', async () => {
  const { ctx } = ctxFor('K(r) = Normal(mu = r.a, sigma = 0.5)\n'
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
