// packages/engine/test/kernel-batch-equivalence.test.ts
// Batching the emcee/MH proposal scoring (one logPosteriorBatch call per
// half/sweep instead of nWalkers scalar logPosterior calls) must not change the
// draws: the proposal + accept randomness is consumed in the same prng order, so
// the result is bit-for-bit identical to the scalar path. This pins that.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { runMcmc } = require('../mcmc-driver.ts');
const { makeEmceeKernel } = require('../emcee-kernel.ts');
const { mhKernel } = require('../mh-kernel.ts');

// Correlated 2-D gaussian target. `batch` toggles whether the model view offers
// logPosteriorBatch (present → kernels take the batched path).
function makeMV(batch: boolean) {
  const dim = 2, names = ['x', 'y'];
  const lp = (v: Float64Array) => -0.5 * (v[0] * v[0] + v[1] * v[1]) - 0.3 * v[0] * v[1];
  const mv: any = { dim, names, constrainAll: (v: Float64Array) => ({ x: v[0], y: v[1] }), logPosterior: lp };
  if (batch) mv.logPosteriorBatch = (ys: Float64Array[]) => Float64Array.from(ys.map(lp));
  return mv;
}

test('emcee batched scorer is bit-identical to the scalar path', () => {
  const opts = { nWalkers: 8, warmup: 200, draws: 200, seed: 9 };
  const scalar = runMcmc(makeMV(false), makeEmceeKernel(), opts);
  const batched = runMcmc(makeMV(true), makeEmceeKernel(), opts);
  assert.deepEqual(Array.from(batched.drawsByName.x), Array.from(scalar.drawsByName.x));
  assert.deepEqual(Array.from(batched.drawsByName.y), Array.from(scalar.drawsByName.y));
  assert.equal(batched.diagnostics.acceptRate, scalar.diagnostics.acceptRate);
});

test('MH batched scorer (sample phase) is bit-identical to the scalar path', () => {
  const opts = { nWalkers: 6, warmup: 300, draws: 300, seed: 7 };
  const scalar = runMcmc(makeMV(false), mhKernel, opts);
  const batched = runMcmc(makeMV(true), mhKernel, opts);
  assert.deepEqual(Array.from(batched.drawsByName.x), Array.from(scalar.drawsByName.x));
  assert.deepEqual(Array.from(batched.drawsByName.y), Array.from(scalar.drawsByName.y));
  assert.equal(batched.diagnostics.acceptRate, scalar.diagnostics.acceptRate);
});
