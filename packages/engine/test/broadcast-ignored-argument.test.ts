'use strict';

// A broadcast keeps its collection arguments' shape whatever the callable's
// body reads. §04 "Broadcasting": `broadcast(f, …)` maps f elementwise over the
// arrays, so `(x -> 0.0).(v)` is an array of zeros as long as `v`. The
// dissolver used to rewrite a broadcast into its body with the arguments
// substituted, and a body that ignores an argument then lost that argument's
// axes: `(x -> 0.0).(v)` became the scalar `0.0`, and a kernel that ignores its
// argument became a broadcast over no collection at all.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const { toJS } = require('./_value-helpers.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function fixedValues(src: string) {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), []);
  return buildDerivations(r.bindings).fixedValues;
}

test('a body that ignores its argument keeps the operand shape', () => {
  const fv = fixedValues(`
v = [1.0, 2.0, 3.0]
f = x -> 0.0
c = 3.0
a = (x -> 0.0).(v)
b = f.(v)
h = (x -> c).(v)
s = sum((x -> 0.0).([1.0, 2.0, 3.0]))
`);
  assert.deepEqual(toJS(fv.get('a')), [0, 0, 0]);
  assert.deepEqual(toJS(fv.get('b')), [0, 0, 0]);
  assert.deepEqual(toJS(fv.get('h')), [3, 3, 3]);
  assert.equal(toJS(fv.get('s')), 0);
});

test('a body that reads some operands keeps the shape of all of them', () => {
  const fv = fixedValues(`
v = [1.0, 2.0, 3.0]
w = [4.0, 5.0, 6.0]
first = ((a, b) -> a).(v, w)
none = ((a, b) -> 1.0).(v, w)
held = ((a, b) -> b).(v, 2.0)
`);
  assert.deepEqual(toJS(fv.get('first')), [1, 2, 3]);
  assert.deepEqual(toJS(fv.get('none')), [1, 1, 1]);
  assert.deepEqual(toJS(fv.get('held')), [2, 2, 2]);
});

test('a broadcast over scalars stays a single call', () => {
  const fv = fixedValues(`
z = (x -> 0.0).(2.0)
`);
  assert.equal(toJS(fv.get('z')), 0);
});

test('the zero volume term of a Dirichlet under neg scores the bare law', () => {
  // The determiniser's text for `pushfwd(neg, Dirichlet([2, 3, 4]))` at
  // [-0.2, -0.3, -0.5]. scipy 1.18.1 `dirichlet.logpdf([0.2, 0.3, 0.5],
  // [2, 3, 4])` gives 2.0228711901914433.
  const fv = fixedValues(`
lp = sub(
  builtin_logdensityof(Dirichlet, record(alpha = [2.0, 3.0, 4.0]), .-__shared_2),
  sum((x -> 0.0).(__shared_2)),
)
__shared_2 = [-0.2, -0.3, -0.5]
`);
  assert.ok(Math.abs(toJS(fv.get('lp')) - 2.0228711901914433) < 1e-12);
});

test('a kernel that ignores its argument is a product over the operand', async () => {
  // Three standard Normals at 0: 3 * log(1 / sqrt(2 pi)).
  const expected = -1.5 * Math.log(2 * Math.PI);
  for (const kernel of ['K(x) = Normal(0.0, 1.0)', 'K = x -> Normal(mu = 0.0, sigma = 1.0)']) {
    const { ctx } = ctxFor(`
${kernel}
y ~ K.([1.0, 2.0, 3.0])
score = logdensityof(lawof(y), [0.0, 0.0, 0.0])
`, 2);
    const score = await ctx.getMeasure('score');
    assert.ok(Math.abs((score.samples ?? score.value.data)[0] - expected) < 1e-12, kernel);
    const y = await ctx.getMeasure('y');
    assert.deepEqual(y.value.shape, [2, 3], kernel);
  }
});
