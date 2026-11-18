'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const operations = `
A = rowstack([[complex(1.0, t), 2.0], [0.0, complex(1.0, -t)]])
v = [complex(1.0, t), complex(2.0, 1.0)]
rows = row_gram(A)
columns = col_gram(A)
outer = self_outer(v)
diagonal = diagmat(conj.(v))
total = trace(diagonal)
extracted = diag(adjoint(A))
crossed = cross([1.0, 0.0, 0.0] * complex(1.0, t), [0.0, complex(2.0, 1.0), 0.0])
quadratic = quadform(A, v)
`;

// §07 Linear algebra. The Gram entries come from row or column inner
// products. The off-diagonal outer entry is (1+it)(2-i).
function checkValues(values: any[], t: number, i: number) {
  const [rows, columns, outer, diagonal, total, extracted, crossed, quadratic] = values;
  const cells = (v: Float64Array) => Array.from(v.slice(4 * i, 4 * i + 4), x => x + 0);
  const diagonalFirst = 1 + t * t;
  assert.deepEqual(cells(rows.samples), [diagonalFirst + 4, 2, 2, diagonalFirst]);
  assert.deepEqual(cells(rows.imag), [0, 2 * t, -2 * t + 0, 0]);
  assert.deepEqual(cells(columns.samples), [diagonalFirst, 2, 2, diagonalFirst + 4]);
  assert.deepEqual(cells(columns.imag), [0, -2 * t + 0, 2 * t, 0]);
  assert.deepEqual(cells(outer.samples), [diagonalFirst, 2 + t, 2 + t, 5]);
  assert.deepEqual(cells(outer.imag), [0, 2 * t - 1, 1 - 2 * t, 0]);
  assert.deepEqual(cells(diagonal.samples), [1, 0, 0, 2]);
  assert.deepEqual(cells(diagonal.imag), [-t + 0, 0, 0, -1]);
  assert.equal(total.samples[i], 3);
  assert.equal(total.imag[i], -(t + 1));
  assert.deepEqual(Array.from(extracted.samples.slice(2 * i, 2 * i + 2)), [1, 1]);
  assert.deepEqual(Array.from(extracted.imag.slice(2 * i, 2 * i + 2), (x: number) => x + 0), [-t + 0, t]);
  assert.deepEqual(Array.from(crossed.samples.slice(3 * i, 3 * i + 3)), [0, 0, 2 - t]);
  assert.deepEqual(Array.from(crossed.imag.slice(3 * i, 3 * i + 3)), [0, 0, 1 + 2 * t]);
  // Expanding v† A v gives (10 + t² + 2t) + i(2 - 8t + t³).
  assert.equal(quadratic.samples[i], 10 + t * t + 2 * t);
  assert.equal(quadratic.imag[i], 2 - 8 * t + t * t * t);
}

test('complex Gram matrices and diagonal transforms preserve both planes', async () => {
  const { ctx } = makeMatCtx('t = 1.0\n' + operations, { sampleCount: 1 });
  const values = await Promise.all(
    ['rows', 'columns', 'outer', 'diagonal', 'total', 'extracted', 'crossed', 'quadratic']
      .map(n => ctx.getMeasure(n)));
  checkValues(values, 1, 0);
});

test('batched complex Gram matrices and diagonal transforms retain each atom', async () => {
  const source = operations.replace('v = [complex(1.0, t), complex(2.0, 1.0)]',
    'v = complex.([1.0, 2.0], [1.0, 0.0] * t + [0.0, 1.0])');
  const { ctx } = makeMatCtx('t ~ Bernoulli(0.5)\n' + source, { sampleCount: 8, rootSeed: 117 });
  const values = await Promise.all(
    ['rows', 'columns', 'outer', 'diagonal', 'total', 'extracted', 'crossed', 'quadratic']
      .map(n => ctx.getMeasure(n)));
  const t = (await ctx.getMeasure('t')).samples;
  assert.ok(t.some((x: number) => x === 0) && t.some((x: number) => x === 1));
  for (let i = 0; i < t.length; i++) {
    checkValues(values, t[i], i);
  }
});
