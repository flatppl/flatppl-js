// The outer-product and matmul aggregate specialisers returned nested JS
// arrays for plain-array operands, while the broadcast-reduce default (the
// correctness oracle) returns a rank-2 Value, so toggling the `aggregate`
// optimisation changed the result TYPE. Oracle: the default lowering's Value.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');
const lowerMod = require('../lower.ts');
const perf = require('../perf-config.ts');

const ev = (src: string, env: any) =>
  sampler.evaluateExpr(lowerMod.lowerExpr(processSource(src).bindings.get('M').node.value), env);

test('aggregate specialisers return the same rank-2 Value as the default lowering', () => {
  const cases: Array<[string, any]> = [
    ['M = aggregate(sum, [.i, .j], u[.i] * v[.j])', { u: [1, 2, 3], v: [10, 20] }],
    ['M = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])', { A: [[1, 2], [3, 4]], B: [[5, 6], [7, 8]] }],
  ];
  const prev = perf.getOptimization('aggregate');
  try {
    for (const [src, env] of cases) {
      perf.setOptimization('aggregate', false);
      const want = ev(src, env);
      perf.setOptimization('aggregate', true);
      const got = ev(src, env);
      assert.ok(!Array.isArray(got), `${src}: nested array`);
      assert.deepEqual(got.shape, want.shape, src);
      assert.deepEqual(Array.from(got.data), Array.from(want.data), src);
    }
  } finally {
    perf.setOptimization('aggregate', prev);
  }
});
