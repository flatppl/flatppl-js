// `joint(iid(D, 2), iid(D, 3))` at a length-5 vector: each iid takes the next
// n entries. The static check required the WHOLE leading axis to equal n and
// raised a false hard error; the runtime walker scored it fine. Oracle:
// 5·log φ(0) and the remaining shape errors.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const M = 'M = joint(iid(Normal(0.0, 1.0), 2), iid(Normal(0.0, 1.0), 3))\n';
const errors = (src: string) => processSource(src).diagnostics
  .filter((d: any) => d.severity === 'error').map((d: any) => d.message);

test('an iid inside a positional joint consumes n entries of the leading axis', async () => {
  const src = M + 'lp = logdensityof(M, [0.0, 0.0, 0.0, 0.0, 0.0])\n';
  assert.deepEqual(errors(src), []);
  const { ctx } = makeMatCtx(src, { sampleCount: 2 });
  const m = await ctx.getMeasure('lp');
  const got = m.value ? m.value.data[0] : m.samples[0];
  assert.ok(Math.abs(got - (-2.5 * Math.log(2 * Math.PI))) < 1e-12, String(got));
  assert.ok(errors(M + 'lp = logdensityof(M, [0.0, 0.0, 0.0, 0.0])\n')
    .some((e: string) => /iid expects 3 elements, got variate with leading axis 2/.test(e)));
  assert.ok(errors(M + 'lp = logdensityof(M, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])\n')
    .some((e: string) => /iid expects 3 elements, got variate with leading axis 4/.test(e)));
});
