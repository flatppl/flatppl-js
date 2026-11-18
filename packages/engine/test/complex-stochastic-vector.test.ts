'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('stochastic vector literals retain both numeric planes', async () => {
  const { ctx } = makeMatCtx(`
t ~ Bernoulli(0.5)
v = [complex(1.0, t), im]
u = [t, im]
D = diagmat(v)
`, { sampleCount: 8, rootSeed: 117 });
  const [t, v, u, D] = await Promise.all(['t', 'v', 'u', 'D'].map(key => ctx.getMeasure(key)));
  assert.ok(t.samples.includes(0) && t.samples.includes(1));
  assert.deepEqual(v.value.shape, [8, 2]);
  assert.deepEqual(u.value.shape, [8, 2]);
  assert.deepEqual(D.value.shape, [8, 2, 2]);
  for (let i = 0; i < t.samples.length; i++) {
    const draw = t.samples[i];
    assert.deepEqual(Array.from(v.samples.subarray(2 * i, 2 * i + 2)), [1, 0]);
    assert.deepEqual(Array.from(v.imag.subarray(2 * i, 2 * i + 2)), [draw, 1]);
    assert.deepEqual(Array.from(u.samples.subarray(2 * i, 2 * i + 2)), [draw, 0]);
    assert.deepEqual(Array.from(u.imag.subarray(2 * i, 2 * i + 2)), [0, 1]);
    assert.deepEqual(Array.from(D.samples.subarray(4 * i, 4 * i + 4)), [1, 0, 0, 0]);
    assert.deepEqual(Array.from(D.imag.subarray(4 * i, 4 * i + 4)), [draw, 0, 0, 1]);
  }
});
