// A density op inside a function body whose variate reads the function's own
// parameter (`[x, x]`). `_checkDensityShapes` re-inferred the variate with
// EMPTY scopes, so `x` was an unbound %local and the valid program got two
// spurious hard errors. Oracle: the error set, and the score 2·log φ(0.5).
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const errors = (src: string) => processSource(src).diagnostics
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);

test('a density op in a function body checks its variate in the body scope', async () => {
  const ok = 'f(x) = logdensityof(iid(Normal(0.0, 1.0), 2), [x, x])\nz = f(0.5)\n';
  assert.deepEqual(errors(ok), []);
  const { ctx } = makeMatCtx(ok, { sampleCount: 2 });
  const m = await ctx.getMeasure('z');
  const got = m.value ? m.value.data[0] : m.samples[0];
  const want = 2 * (-0.5 * Math.log(2 * Math.PI) - 0.125);
  assert.ok(Math.abs(got - want) < 1e-12, `${got} vs ${want}`);
  // A real mismatch in the body is still caught.
  const bad = errors('f(x) = logdensityof(iid(Normal(0.0, 1.0), 3), [x, x])\nz = f(0.5)\n');
  assert.ok(bad.some((m: string) => /iid expects 3 elements, got variate with leading axis 2/.test(m)), bad.join('; '));
});
