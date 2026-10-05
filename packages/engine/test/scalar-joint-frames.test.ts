'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const density = require('../density.ts');
const { inBothModes } = require('./_perf-helpers.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const lit = (value: any) => ({ kind: 'lit', value });
const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const call = (op: string, args: any[]) => ({ kind: 'call', op, args });
const normal = (mu: any) => call('Normal', [mu, lit(1)]);
const logpdf = (x: number, mu: number) => -0.5 * Math.log(2 * Math.PI) - 0.5 * (x - mu) ** 2;

inBothModes('constant scalar joint scores broadcast across atoms', 'density.scalarJoint', () => {
  const ir = call('joint', [normal(lit(0)), normal(ref('s0'))]);
  const actual = density.logDensityN(ir, [0.5, 1], {}, 3, {});
  for (const lp of actual) assert.ok(Math.abs(lp - logpdf(0.5, 0) - logpdf(1, 0.5)) < 1e-12);
});

inBothModes('scalar joint keeps structural distribution arguments', 'density.scalarJoint', () => {
  const ir = call('joint', [call('Uniform', [{ kind: 'const', name: 'unitinterval' }]), normal(lit(0))]);
  assert.ok(Math.abs(density.logDensity(ir, [0.2, 0.3], {}) - logpdf(0.3, 0)) < 1e-12);
});

inBothModes('scalar joint reads current state and per-atom inputs', 'density.scalarJoint', () => {
  const mean = call('add', [ref('s0'), ref('delta')]);
  const leaf = normal(call('add', [mean, mean]));
  const ir = call('joint', [leaf, leaf, leaf]);
  const refs = { s0: Float64Array.of(10, 20), delta: Float64Array.of(0.25, -0.5) };
  for (const xs of [[1, 2, 3], [0, -1, 4]]) {
    const result = density.logDensityConsumeN(ir, [...xs, 99], refs, 2, {});
    for (let i = 0; i < 2; i++) {
      const expected = logpdf(xs[0], 2 * (refs.s0[i] + refs.delta[i]))
        + logpdf(xs[1], 2 * (xs[0] + refs.delta[i]))
        + logpdf(xs[2], 2 * (xs[0] + refs.delta[i]));
      assert.ok(Math.abs(result.logps[i] - expected) < 1e-10);
    }
    assert.deepEqual(result.rest, [99]);
  }
});

inBothModes('nested scalar joints keep the enclosing observed state', 'density.scalarJoint', () => {
  const inner = call('joint', [normal(ref('s0')), normal(ref('s0'))]);
  const ir = { kind: 'call', op: 'joint', fields: [
    { name: 's0', value: normal(lit(0)) },
    { name: 'a', value: inner },
    { name: 'b', value: inner },
    { name: 'tail', value: normal(ref('s0')) },
  ] };
  const actual = density.logDensity(ir, { s0: 5, a: [1, 2], b: [-1, 0], tail: 6 }, {});
  const expected = logpdf(5, 0) + logpdf(1, 5) + logpdf(2, 1)
    + logpdf(-1, 5) + logpdf(0, -1) + logpdf(6, 5);
  assert.ok(Math.abs(actual - expected) < 1e-12);
});

inBothModes('scalar joint recovers after a parameter throws', 'density.scalarJoint', () => {
  const mean = call('checked', [ref('s0'), call('gt', [ref('ok'), lit(0)])]);
  const ir = call('joint', [normal(lit(0)), normal(mean)]);
  assert.throws(() => density.logDensity(ir, [1, 2], { ok: -1 }));
  const actual = density.logDensity(ir, [3, 4], { ok: 1 });
  assert.ok(Math.abs(actual - logpdf(3, 0) - logpdf(4, 3)) < 1e-12);
});

test('kscan updates exogenous inputs in a nested compiled scan', async () => {
  const { ctx } = makeMatCtx(`
__mc_index = 8
step = (state, x) -> Bernoulli(ifelse(sum(scan((a, b) -> (a + b) + (a + b), state, [x, 1])) > __mc_index, 1, 0))
traj = kscan(step, 0, [2, 0, 2])
`, { sampleCount: 4 });
  const result = await ctx.getMeasure('traj');
  // The nested sum is 6 * state + 6 * x + 2: 14, 8, 14.
  assert.deepEqual(Array.from(result.samples), Array(4).fill([1, 0, 1]).flat());
});
