'use strict';

// §07 fill repeats a value, including complex scalar/array values; rowstack
// and colstack interpret the resulting vectors as rows and columns (§03).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

async function parts(source: string) {
  const { ctx } = ctxFor(source, 1);
  const m = await ctx.getMeasure('x');
  return { shape: m.intrinsicShape, re: Array.from(m.samples), im: Array.from(m.imag || []) };
}

test('fill repeats both parts of a complex scalar at the requested shape', async () => {
  assert.deepEqual(await parts('c = complex(1.0, -2.0)\nx = fill(c, [2, 3])\n'),
    { shape: [2, 3], re: [1, 1, 1, 1, 1, 1], im: [-2, -2, -2, -2, -2, -2] });
  assert.deepEqual(await parts('x = fill(2.0, 3)\n'),
    { shape: undefined, re: [2, 2, 2], im: [] });
});

test('rowstack and colstack retain the complex planes of nested fill', async () => {
  const source = 'v = conj.([complex(1.0, 2.0), complex(3.0, -4.0), complex(5.0, 6.0)])\nnested = fill(v, 2)\n';
  assert.deepEqual(await parts(source + 'x = rowstack(nested)\n'),
    { shape: [2, 3], re: [1, 3, 5, 1, 3, 5], im: [-2, 4, -6, -2, 4, -6] });
  assert.deepEqual(await parts(source + 'x = colstack(nested)\n'),
    { shape: [3, 2], re: [1, 1, 3, 3, 5, 5], im: [-2, -2, 4, 4, -6, -6] });
});

test('nested scalar fill becomes a complex matrix', async () => {
  assert.deepEqual(await parts('x = rowstack(fill(fill(complex(2.0, 3.0), 3), 2))\n'),
    { shape: [2, 3], re: [2, 2, 2, 2, 2, 2], im: [3, 3, 3, 3, 3, 3] });
});

test('a filled complex matrix multiplies each stochastic vector draw', async () => {
  const { ctx } = ctxFor('M = rowstack(fill(fill(complex(2.0, 3.0), 2), 2))\nv ~ iid(Normal(0.0, 1.0), 2)\nx = M * v\n', 4);
  const v = await ctx.getMeasure('v');
  const x = await ctx.getMeasure('x');
  for (let k = 0; k < 4; k++) {
    const sum = v.samples[2 * k] + v.samples[2 * k + 1];
    for (let j = 0; j < 2; j++) {
      assert.ok(Math.abs(x.samples[2 * k + j] - 2 * sum) < 1e-14);
      assert.ok(Math.abs(x.imag[2 * k + j] - 3 * sum) < 1e-14);
    }
  }
});
