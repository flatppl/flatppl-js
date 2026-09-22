'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// Each loop rebinds its inputs. Repeated pure expressions may share work
// within one iteration, but must never reuse the previous iteration's value.
const cases = [
  {
    name: 'reduce',
    source: `
f = (a, b) -> max(a + b, a + b)
g = x -> reduce(f, [x, 2.0, 3.0])
result = broadcast(g, [1.0, 2.0])
`,
    expected: [6, 7],
  },
  {
    name: 'scan',
    source: `
f = (a, b) -> (a + b) + (a + b)
g = x -> sum(scan(f, x, [1.0, 2.0, 3.0]))
result = broadcast(g, [0.0, 1.0])
`,
    expected: [32, 46],
  },
  {
    name: 'filter',
    source: `
f = x -> (x + 1.0) > (x + 1.0) / 2.0
g = y -> lengthof(filter(f, [-2.0, y, 2.0]))
result = broadcast(g, [0.0, 1.0])
`,
    expected: [2, 2],
  },
];

for (const { name, source, expected } of cases) {
  test(`broadcast keeps ${name} iterations in distinct frames`, async () => {
    const { ctx } = makeMatCtx(source);
    const result = await ctx.getMeasure('result');
    assert.deepEqual(Array.from(result.value.data), expected);
  });
}
