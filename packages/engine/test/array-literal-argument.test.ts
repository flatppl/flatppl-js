// An array literal written as an argument (`rowstack([v, v])`, `sum([mu,
// 1.0])`) with stochastic entries is a value operand. `isEvaluable` refused
// every `vector` holding a ref, so such a binding got no derivation and
// failed "no derivation for 'A'" with no diagnostic, while the same literal
// bound to its own name first worked. Oracle: the stacked or summed draws.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const close = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));

test('rowstack and colstack of an array literal with stochastic entries', async () => {
  const src = 'v ~ iid(Normal(0.0, 1.0), 2)\nw ~ iid(Normal(0.0, 1.0), 2)\n'
    + 'R = rowstack([v, [1.0, 2.0]])\nC = colstack([v, 2.0 .* w])\n';
  const N = 5;
  const { ctx } = makeMatCtx(src, { sampleCount: N });
  const v = (await ctx.getMeasure('v')).value.data;
  const w = (await ctx.getMeasure('w')).value.data;
  const R = (await ctx.getMeasure('R')).value;
  const C = (await ctx.getMeasure('C')).value;
  assert.deepEqual(R.shape, [N, 2, 2]);
  assert.deepEqual(C.shape, [N, 2, 2]);
  for (let i = 0; i < N; i++) {
    const [v0, v1, w0, w1] = [v[2 * i], v[2 * i + 1], w[2 * i], w[2 * i + 1]];
    [v0, v1, 1, 2].forEach((x, k) => assert.ok(close(R.data[4 * i + k], x), `R atom ${i} cell ${k}`));
    [v0, 2 * w0, v1, 2 * w1].forEach((x, k) => assert.ok(close(C.data[4 * i + k], x), `C atom ${i} cell ${k}`));
  }
});

test('a reduction over an array literal with a stochastic entry', async () => {
  const { ctx } = makeMatCtx('mu ~ Normal(0.0, 1.0)\ns = sum([mu, 1.0])\n', { sampleCount: 5 });
  const mu = (await ctx.getMeasure('mu')).samples;
  const s = (await ctx.getMeasure('s')).samples;
  for (let i = 0; i < mu.length; i++) assert.ok(close(s[i], mu[i] + 1), `atom ${i}`);
});
