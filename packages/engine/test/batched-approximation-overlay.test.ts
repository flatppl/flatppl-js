// `_batchedApproximation` evaluated polynomial / stepwise coefficients against
// baseEnv only. An overlay value (an env-threaded observation) was ignored,
// and per-atom coefficients were read from a stale baseEnv entry of the same
// name. The module's rule is overlay > refArrays > baseEnv.
// Oracle: the polynomial c0 + c1·x evaluated by hand.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const sampler = require('../sampler.ts');

const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const POLY = { kind: 'call', op: 'polynomial', kwargs: { coefficients: ref('c'), x: ref('x') } };
const x = new Float64Array([0, 1, 2]);
const data = (v: any) => Array.from(v.data || v);

test('batched polynomial coefficients follow overlay > refArrays > baseEnv', () => {
  assert.deepEqual(data(sampler.evaluateExprN(POLY, { x }, 3, { c: [1, 2] }, { overlay: { c: [3, 4] } })),
    [3, 7, 11], 'overlay shadows baseEnv');
  const cAtoms = { shape: [3, 2], data: new Float64Array([0, 1, 1, 1, 2, 1]) };
  assert.deepEqual(data(sampler.evaluateExprN(POLY, { x, c: cAtoms }, 3, { c: [100, 100] }, {})),
    [0, 2, 4], 'per-atom coefficients shadow baseEnv');
});
