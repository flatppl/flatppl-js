// An axis-free aggregate body subtree that reads a per-atom name, such as
// `exp(mu)` in `aggregate(sum, [], exp(mu) * v[.i])`, holds one value per
// atom. The lifter evaluated it with the single-point evaluator, which saw
// `mu` as its whole [N] column: `exp` of an array is NaN and `mu + 1.0`
// concatenated a string, so every atom came out NaN. A bare `mu` was right.
// Oracle: the closed form per atom from the materialised `mu` draws.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { inBothModes } = require('./_perf-helpers.ts');

const V = [1, 2, 3];
const CASES: Array<[string, (mu: number) => number[]]> = [
  ['aggregate(sum, [], exp(mu) * v[.i])', (mu) => [Math.exp(mu) * 6]],
  ['aggregate(sum, [], (mu + 1.0) * v[.i])', (mu) => [(mu + 1) * 6]],
  ['aggregate(sum, [], mu * v[.i])', (mu) => [mu * 6]],
  ['aggregate(sum, [.i], exp(mu) * v[.i])', (mu) => V.map((x) => Math.exp(mu) * x)],
];

inBothModes('a per-atom axis-free subtree in an aggregate body is evaluated per atom', 'aggregate', async () => {
  for (const [body, oracle] of CASES) {
    const src = `mu ~ Normal(0.0, 1.0)\nv = [${V.join('.0, ')}.0]\nz = ${body}\n`;
    const { ctx } = makeMatCtx(src, { sampleCount: 16 });
    const mu = Array.from((await ctx.getMeasure('mu')).samples as Float64Array);
    const z = (await ctx.getMeasure('z')).value;
    const width = z.data.length / mu.length;
    for (let a = 0; a < mu.length; a++) {
      const want = oracle(mu[a]);
      assert.equal(width, want.length, body);
      for (let j = 0; j < width; j++) {
        const got = z.data[a * width + j];
        assert.ok(Math.abs(got - want[j]) <= 1e-12 * Math.max(1, Math.abs(want[j])), `${body} atom ${a}[${j}]: ${got} vs ${want[j]}`);
      }
    }
  }
});
