'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { toJS } = require('./_value-helpers.ts');

// §03 tuples bundle objects, including functions; §11 call heads may be
// expressions. Materialization must preserve both the callable and its inputs.
test('a callable projected from a tuple applies to its numeric parameters', async () => {
  const { ctx } = makeMatCtx(`
rhs(t, y, p) = p * y
call(q) = q[1](1, 2, q[2])
result = call((rhs, 0.1))
`);
  assert.equal((await ctx.getMeasure('result')).samples[0], 0.2);
});

test('nested scans carry a callable and numeric parameter through each step', async () => {
  const { ctx } = makeMatCtx(`
rhs(t, y, p) = p * y
inner(a, x, p) = a + p[1](x, a, p[2])
outer(a, x, p) = scan(inner, a, [x, x], p)[2]
solve(f, y, p) = scan(outer, y, [1, 2], (f, p))
result = solve(rhs, 1, 0.1)
`);
  const result = toJS((await ctx.getMeasure('result')).value);
  // Each inner step multiplies its state by 1.1; each outer step does two.
  for (let i = 0; i < result.length; i++) {
    assert.ok(Math.abs(result[i] - 1.1 ** (2 * (i + 1))) < 1e-14);
  }
  assert.equal(result.length, 2);
});

test('a nested scan call retains its callable argument before building its tuple', async () => {
  const { ctx } = makeMatCtx(`
rhs(t, y, p) = p * y
step(a, x, p) = a + p[1](x, a, p[2])
solve(f, y, p) = scan(step, y, [1, 2], (f, p))[2]
result = solve(rhs, solve(f = rhs, y = 1, p = 0.1), 0.1)
`);
  // Two successive two-step solves apply four factors of 1.1.
  assert.ok(Math.abs((await ctx.getMeasure('result')).samples[0] - 1.1 ** 4) < 1e-14);
});
