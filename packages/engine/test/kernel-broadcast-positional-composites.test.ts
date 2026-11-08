// A composite kernel broadcast written the way a user writes it: the dot-call
// `K.(m)` passes its argument POSITIONALLY, and the kernel body writes its
// distributions positionally (`Normal(a, 0.1)`) or broadcasts them
// (`Normal.([a, a], 0.1)`). The joint, jointchain and nested-broadcast
// executors read only `d.kwargIRs` and the recognisers kept only a
// distribution's keyword arguments, so these failed ("'Normal' missing
// parameter 'mu'", "no derivation", "broadcast: first arg must be a
// function") while the keyword spellings worked.
// Oracle: per-cell moments of the denoted product measure.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const N = 20000;

async function cellMoments(src: string) {
  const { ctx } = makeMatCtx(src, { sampleCount: N });
  const v = (await ctx.getMeasure('X')).value;
  assert.equal(v.shape[0], N);
  const w = v.data.length / N;
  const mean: number[] = [], variance: number[] = [];
  for (let j = 0; j < w; j++) {
    let s = 0, q = 0;
    for (let i = 0; i < N; i++) { const x = v.data[i * w + j]; s += x; q += x * x; }
    mean.push(s / N); variance.push(q / N - (s / N) ** 2);
  }
  return { shape: v.shape, mean, variance };
}

// Each cell k of `m = [10, 20]` yields two coordinates; `want` gives the
// mean and variance of each, in row-major order.
const CASES: Array<[string, string, number[], number[]]> = [
  ['joint, positional dist and dot-call',
    'K(a) = joint(Normal(a, 0.1), Normal(a, 0.2))\nX ~ K.(m)\n',
    [10, 10, 20, 20], [0.01, 0.04, 0.01, 0.04]],
  ['joint, keyword dot-call',
    'K(a) = joint(Normal(a, 0.1), Normal(a, 0.2))\nX ~ K.(a = m)\n',
    [10, 10, 20, 20], [0.01, 0.04, 0.01, 0.04]],
  ['jointchain with an inline step',
    'K(a) = jointchain(Normal(a, 0.1), fn(Normal(_, 0.1)))\nX ~ K.(m)\n',
    [10, 10, 20, 20], [0.01, 0.02, 0.01, 0.02]],
  ['nested dot-broadcast over an array literal',
    'K(a) = Normal.([a, 2.0 * a], 0.1)\nX ~ K.(m)\n',
    [10, 20, 20, 40], [0.01, 0.01, 0.01, 0.01]],
  ['nested keyword broadcast over an array literal',
    'K(a) = broadcast(Normal, mu = [a, 2.0 * a], sigma = 0.1)\nX ~ K.(m)\n',
    [10, 20, 20, 40], [0.01, 0.01, 0.01, 0.01]],
];

for (const [label, body, wantMean, wantVar] of CASES) {
  test(`positional composite kernel broadcast: ${label}`, async () => {
    const got = await cellMoments('m = [10.0, 20.0]\n' + body);
    assert.deepEqual(got.shape, [N, 2, 2]);
    for (let j = 0; j < 4; j++) {
      // Standard errors at N = 20000: mean sqrt(v/N) <= 0.0014, variance v*sqrt(2/N) <= 4e-4.
      assert.ok(Math.abs(got.mean[j] - wantMean[j]) < 0.01, `${label} mean[${j}] ${got.mean[j]}`);
      assert.ok(Math.abs(got.variance[j] - wantVar[j]) < 6 * wantVar[j] * Math.sqrt(2 / N),
        `${label} var[${j}] ${got.variance[j]}`);
    }
  });
}
