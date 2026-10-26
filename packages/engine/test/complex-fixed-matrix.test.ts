'use strict';

// A fixed complex matrix lost its imaginary part on the way to the viewer:
// fixedValueToMeasure flattened only `data`, so `M * 1i` surfaced as a matrix
// of zeros. And `rowstack` / `colstack` of complex entries read `+{re, im}` as
// NaN, so a complex matrix written as a literal held NaN in every cell. Hand
// computed: M = [[1 + 2i, i], [3, 1 − i]], M · [1, 2] = [1 + 4i, 5 − 2i],
// M² = [[−3 + 7i, −1 + 2i], [6 + 3i, 0 + i]] (row-major), Mᴴ = [[1 − 2i, 3], [−i, 1 + i]].
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');

const M = 'rowstack([[complex(1.0, 2.0), complex(0.0, 1.0)], [complex(3.0, 0.0), complex(1.0, -1.0)]])';

async function parts(src: string, name: string) {
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure(name);
  return { dtype: m.dtype, shape: m.intrinsicShape, re: Array.from(m.samples), im: Array.from(m.imag || []) };
}

test('rowstack and colstack of complex entries build a complex matrix', async () => {
  assert.deepEqual(await parts(`M = ${M}\n`, 'M'),
    { dtype: 'complex', shape: [2, 2], re: [1, 0, 3, 1], im: [2, 1, 0, -1] });
  const cs = await parts('C = colstack([[complex(1.0, 2.0), complex(0.0, 1.0)], [complex(3.0, 0.0), complex(1.0, -1.0)]])\n', 'C');
  assert.deepEqual(cs, { dtype: 'complex', shape: [2, 2], re: [1, 3, 0, 1], im: [2, 0, 1, -1] });
});

test('a fixed complex matrix keeps both parts through to the measure', async () => {
  assert.deepEqual(await parts('B = rowstack([[1.0, 2.0], [3.0, 4.0]]) * complex(0.0, 1.0)\n', 'B'),
    { dtype: 'complex', shape: [2, 2], re: [0, 0, 0, 0], im: [1, 2, 3, 4] });
  assert.deepEqual(await parts(`M = ${M}\nP = M * M\n`, 'P'),
    { dtype: 'complex', shape: [2, 2], re: [-3, -1, 6, 0], im: [7, 2, 3, 1] });
  const a = await parts(`M = ${M}\nA = adjoint(M)\n`, 'A');
  assert.deepEqual(a.re, [1, 3, 0, 1]);
  assert.deepEqual(a.im.map((x: any) => x + 0), [-2, 0, -1, 1]);
});

test('a complex vector from a complex matrix keeps both parts', async () => {
  assert.deepEqual(await parts(`M = ${M}\nv = M * [1.0, 2.0]\n`, 'v'),
    { dtype: 'complex', shape: undefined, re: [1, 5], im: [4, -2] });
});
