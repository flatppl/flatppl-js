'use strict';

// A θ-dependent normalizer whose weight has a window that moves with a latent.
// crn-normalize's fixed 128-point box sample and leaf-mass-quad's fixed graded
// rule both fixed their nodes before θ was known, so a window between nodes was
// invisible: `1 + 100 · [x ∈ (c, c + 1e-4)]` over Lebesgue([0, 1]) scored
// log-density 0 against −0.00995, and over Normal(0, 1) the leaf rule scored
// log φ(0.8) as if the window were absent. Both now split at the breakpoints
// (piecewise-mass.ts), solving affine tests for them, or refuse where they cannot.
//
// Oracles: scipy.integrate.quad with points at the window (box), and the closed
// form log φ(x) + log w(x) − log(1 + 100 (Φ(c + w) − Φ(c))) from scipy.stats.norm
// (leaf).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

function model(win: string, base: string, point: string, c: number): string {
  return `c ~ Uniform(interval(0.1, 0.8))\nwin(x, c0) = ${win}\n`
    + `y ~ normalize(weighted(win, ${base}, c))\n`
    + `L = likelihoodof(kernelof(y, c = c), ${point})\nlp = logdensityof(L, record(c = ${c}))\n`;
}
async function score(src: string, name = 'lp', N = 1): Promise<number> {
  const { ctx } = ctxFor(src, N);
  const m = await ctx.getMeasure(name);
  return m.value ? m.value.data[0] : m.samples[0];
}
const spike = (w: number) => `1.0 + 100.0 * ifelse(x in interval(c0, c0 + ${w}), 1.0, 0.0)`;
const unit = 'Lebesgue(interval(0.0, 1.0))';

const CASES: Array<[string, string, string, string, number, number]> = [
  ['box, a 1e-4 window', spike(0.0001), unit, '0.8', 0.5, -0.009950330853166992],
  ['box, a 0.01 window', spike(0.01), unit, '0.8', 0.26, -0.6931471805599457],
  ['box, a point inside a 0.05 window', spike(0.05), unit, '0.32', 0.3, 2.823361047613205],
  ['box, a point inside a 1e-4 window', spike(0.0001), unit, '0.50005', 0.5, 4.605170185988093],
  ['box, a window partly outside', '1.0 + 100.0 * ifelse(x in interval(c0 + 0.55, c0 + 0.75), 1.0, 0.0)', unit, '0.1', 0.4, -Math.log(6)],
  ['box, a comparison', '1.0 + 9.0 * ifelse(x < c0, 1.0, 0.0)', unit, '0.9', 0.3, -Math.log(1 + 9 * 0.3)],
  ['box, a one-component vector variate', '1.0 + 100.0 * ifelse(x[1] in interval(c0, c0 + 0.01), 1.0, 0.0)',
    'Lebesgue(cartprod(interval(0.0, 1.0)))', '[0.8]', 0.26, -0.6931471805599457],
  ['Normal leaf, a 1e-4 window', spike(0.0001), 'Normal(0.0, 1.0)', '0.8', 0.5, -1.242452915768527],
  ['Normal leaf, a 0.01 window', spike(0.01), 'Normal(0.0, 1.0)', '0.8', 0.26, -1.5647657403141768],
  // Affine tests solved for their breakpoint: x < c0 / 2, and -c0 < x < c0.
  ['box, an affine comparison', '1.0 + 9.0 * ifelse(2.0 * x < c0, 1.0, 0.0)', unit, '0.9', 0.3, -0.8544153281560676],
  // Z = 1 + 100·0.01 + 50·0.02 = 3. The rule has one node per piece, so two
  // windows fit the node budget.
  ['box, two windows', '1.0 + 100.0 * ifelse(x in interval(c0, c0 + 0.01), 1.0, 0.0) '
    + '+ 50.0 * ifelse(x in interval(c0 + 0.5, c0 + 0.52), 1.0, 0.0)', unit, '0.95', 0.3, -Math.log(3)],
  ['box, a 1e-6 window', spike(0.000001), unit, '0.8', 0.123456, -9.999500033330833e-05],
  ['box, abs(x - c0) < 1e-5', '1.0 + 100.0 * ifelse(abs(x - c0) < 1.0e-5, 1.0, 0.0)', unit, '0.8', 0.4,
    -0.001998002662673056],
  ['Normal leaf, a 1e-5 window', spike(0.00001), 'Normal(0.0, 1.0)', '0.8', 0.7, -1.2392507373043737],
  ['Normal leaf, a point inside abs(x - c0) < 0.05', '1.0 + 100.0 * ifelse(abs(x - c0) < 0.05, 1.0, 0.0)',
    'Normal(0.0, 1.0)', '0.32', 0.3, 2.0737792998571054],
  ['Normal leaf, abs of the variate', '1.0 + 9.0 * ifelse(abs(x) < c0, 1.0, 0.0)', 'Normal(0.0, 1.0)', '0.9', 0.3,
    -2.462542263387726],
];
for (const [label, win, base, point, c, want] of CASES) {
  test(`latent window, ${label}: matches the scipy oracle`, async () => {
    const got = await score(model(win, base, point, c));
    assert.ok(Math.abs(got - want) < 1e-12, `${got} vs ${want}`);
  });
}

test('latent window, a log-weight over a Normal leaf', async () => {
  const src = 'c ~ Uniform(interval(0.1, 0.8))\nlw(x, c0) = ifelse(x in interval(c0, c0 + 0.01), log(101.0), 0.0)\n'
    + 'y ~ normalize(logweighted(lw, Normal(0.0, 1.0), c))\n'
    + 'L = likelihoodof(kernelof(y, c = c), 0.8)\nlp = logdensityof(L, record(c = 0.26))\n';
  const got = await score(src);
  assert.ok(Math.abs(got - -1.5647657403141768) < 1e-12, `${got}`);
});

test('latent window, the sampling route draws from the split normalizer', async () => {
  const src = 'c ~ Uniform(interval(0.1, 0.4))\nwin(x, c0) = ' + spike(0.01) + '\n'
    + `y ~ normalize(weighted(win, ${unit}, c))\n`;
  const { ctx } = ctxFor(src, 200);
  const y = await ctx.getMeasure('y');
  assert.equal(y.samples.length, 200);
  assert.ok(Array.from(y.samples as number[]).every((v) => v >= 0 && v <= 1));
});

for (const [label, win, base, point, why] of [
  ['a test with no breakpoint', '1.0 + 9.0 * ifelse(x * x < c0, 1.0, 0.0)', unit, '0.9',
    /a `lt` test on an expression that is not affine in the variate at line 2, column 33, which gives no breakpoint/],
  ['a step of the variate', '1.0 + floor(x / c0)', unit, '0.9', /`floor` of the variate, whose number of steps depends on θ/],
  // A narrow bump inside a piece falls between any fixed nodes: a 130-node rule
  // scored 0.1625 against the exact 0.0914, and its coarse-rule gate passed.
  ['a weight that varies with x between its breakpoints',
    'ifelse(x < c0, 1.0, 2.0) + 100.0 * exp(-0.5 * ((x - 0.37) / 0.0005)^2)', unit, '0.9',
    /varies with the variate at line 2, column \d+ between its breakpoints\. An x-varying weight inside a θ-dependent window has no exact rule/],
  ['a wider bump, which tripped the old gate as "infinite total mass"',
    'ifelse(x < c0, 1.0, 2.0) + 100.0 * exp(-0.5 * ((x - 0.37) / 0.003)^2)', unit, '0.9',
    /^(?![\s\S]*INFINITE)[\s\S]*varies with the variate/],
  ['a Normal leaf with a weight that varies with x inside its window',
    'ifelse(x in interval(c0, c0 + 0.01), exp(3.0 * x), 1.0)', 'Normal(0.0, 1.0)', '0.9', /varies with the variate at line 2/],
  ['a test on a 2-D variate', '1.0 + 9.0 * ifelse(x[1] < c0, 1.0, 0.0)',
    'Lebesgue(cartprod(interval(0.0, 1.0), interval(0.0, 1.0)))', '[0.9, 0.9]', /may jump in the 2-D variate/],
  ['three windows (more breakpoints than the network sorts)', '1.0 + ifelse(x in interval(c0, c0 + 0.01), 1.0, 0.0) '
    + '+ ifelse(x in interval(c0 + 0.1, c0 + 0.11), 1.0, 0.0) + ifelse(x in interval(c0 + 0.2, c0 + 0.21), 1.0, 0.0)',
    unit, '0.95', /6 breakpoints \(more than 4\)/],
  ['a Normal leaf with a test with no breakpoint', '1.0 + 9.0 * ifelse(mod(x, c0) < 0.1, 1.0, 0.0)', 'Normal(0.0, 1.0)', '0.9',
    /a `lt` test on an expression that is not affine in the variate .*, which gives no breakpoint/],
] as Array<[string, string, string, string, RegExp]>) {
  test(`latent window refuses ${label}`, async () => {
    const { ctx } = ctxFor(model(win, base, point, 0.3), 1);
    await assert.rejects(ctx.getMeasure('lp'), why);
  });
}
