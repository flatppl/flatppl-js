'use strict';

// Two prior-transform paths for a map of an iid vector latent.
//
// * The affine registry forward multiplies a shared scale L by the atom-batched
//   base z. From shape alone a shared [1, 1] L at N = 1 reads as batched, so the
//   forward states the atom axes instead of guessing them.
// * A map without a registry bijection is evaluated on the realised base. The
//   base is bound as a Value: an array-domain op such as `neg` (§07) evaluates a
//   Value whole and gave null for a plain JS array.
//
// Expected values are closed forms: x ~ Normal(1, 2) realises 1 + 2·Φ⁻¹(u), so at
// u = Φ(1) every coordinate is 3.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const valueOps = require('../value-ops.ts');
const valueLib = require('../value.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildPriorTransform } = require('../prior-transform.ts');

const U = 0.8413447460685429;   // Φ(1)

const model = (M: string) => `
x ~ ${M}
prior = lawof(record(x = x))
y ~ Normal(x[1], 1.0)
K = kernelof(record(y = y), x = x)
L = likelihoodof(K, record(y = 0.0))
posterior = bayesupdate(L, prior)
`;

function realise(M: string): number[] {
  const { ctx } = ctxFor(model(M), 10);
  const pt = buildPriorTransform(ctx, ctx.derivations.posterior);
  return Array.from(pt.transform(Float64Array.from({ length: pt.dim }, () => U)).x as Float64Array);
}

test('mulN with stated atom axes scales a batched base by a shared diagonal', () => {
  // N = 1: L = diag(2) has shape [1, 1], the same as the batched z.
  const L1 = valueLib.diagMatrix([2]);
  const z1 = { shape: [1, 1], data: Float64Array.of(3) };
  assert.deepEqual(Array.from(valueOps.mulN(L1, z1, 1, [false, true]).data), [6]);
  // N = 4, D = 2: each atom's row is scaled by diag(2, 5).
  const L4 = valueLib.diagMatrix([2, 5]);
  const z4 = { shape: [4, 2], data: Float64Array.of(1, 1, 2, 2, 3, 3, 4, 4) };
  const y4 = valueOps.mulN(L4, z4, 4, [false, true]);
  assert.deepEqual(y4.shape, [4, 2]);
  assert.deepEqual(Array.from(y4.data), [2, 5, 4, 10, 6, 15, 8, 20]);
});

test('the affine registry route realises a length-1 and a length-3 latent', () => {
  // 2·x with x = 3 per coordinate.
  assert.deepEqual(realise('pushfwd(x -> 2.0 * x, iid(Normal(1.0, 2.0), 1))'), [6]);
  assert.deepEqual(realise('pushfwd(x -> 2.0 * x, iid(Normal(1.0, 2.0), 3))'), [6, 6, 6]);
});

test('neg of an iid vector realises every coordinate negated', () => {
  for (const map of ['neg', 'fn(neg(_))', 'x -> -x']) {
    assert.deepEqual(realise(`pushfwd(${map}, iid(Normal(1.0, 2.0), 1))`), [-3], map);
    assert.deepEqual(realise(`pushfwd(${map}, iid(Normal(1.0, 2.0), 3))`), [-3, -3, -3], map);
  }
});
