const test = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// §04 Multi-axis aggregation: inner axes have their own lexical scope.
// Check the full optimizer/materialiser path, not just aggregate evaluation.
for (const innerAxes of [['row', 'col'], ['i', 'j']]) {
  test(`nested matrix reductions preserve ${innerAxes.join('/')} axes`, async () => {
    const [row, col] = innerAxes;
    const { ctx } = makeMatCtx(`
      f(x) = sum(aggregate(sum, [.col], rowstack([
        [1.0, 2.0],
        aggregate(sum, [.${col}],
          (rowstack([[3.0, 4.0], [5.0, 6.0]]) .* x)[.${row}, .${col}])
      ])[.row, .col]))
      result = f.([1.0, 2.0])
    `, { sampleCount: 1 });
    const result = await ctx.getMeasure('result');
    // Column sums are [1 + 8x, 2 + 10x].
    assert.deepEqual(Array.from(result.value.data), [21, 39]);
  });
}

const cases = [
  ['independent scalar sums', 'aggregate(sum, [], a[.i] * aggregate(sum, [], b[.j]))', 21],
  ['independent scalar means', 'aggregate(mean, [], a[.i] * aggregate(mean, [], b[.j]))', 5.25],
  ['negated scalar sum', 'aggregate(sum, [], a[.i] * -aggregate(sum, [], b[.j]))', -21],
  ['shadowed reduced axis', 'aggregate(sum, [], a[.i] * aggregate(sum, [], b[.i]))', 21],
  ['additive context', 'aggregate(sum, [], a[.i] + aggregate(sum, [], b[.j]))', 17],
  ['nonlinear context', 'aggregate(sum, [], a[.i] * (aggregate(sum, [], b[.j]))^2)', 147],
] as const;

for (const [name, expression, expected] of cases) {
  test(`nested aggregate: ${name}`, async () => {
    const { ctx } = makeMatCtx(`a = [1.0, 2.0]\nb = [3.0, 4.0]\nresult = ${expression}`,
      { sampleCount: 1 });
    const result = await ctx.getMeasure('result');
    assert.equal(result.value.data[0], expected);
  });
}
