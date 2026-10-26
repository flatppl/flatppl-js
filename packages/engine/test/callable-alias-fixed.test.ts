'use strict';

// A binding that names a callable (`g = f`, or `f_b = common.f_b` from a loaded
// module) types as that callable, but carries the producer tag of a plain call.
// The fixed-value pass therefore evaluated it as a value, failed on the
// unbound function reference, and reported two spurious errors: "failed to
// evaluate" and "produced no value … engine gap". The committed example
// bayesian_inference_3 showed both in the playground with no edit. A callable
// has no fixed value, and calls through the alias were always right.
//
// bayesian_inference_4 showed a second spurious error of the same kind: the
// `likelihoodof` that a `restrict` expansion lifts into an anonymous binding was
// tagged 'call', not 'likelihood', and drew "produced no value".
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

function buildDiagnostics(src: string, opts?: any) {
  const proc = processSource(src, opts);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  return orchestrator.buildDerivations(proc.linkedBindings, { moduleRegistry: proc.linkedModuleRegistry })
    .diagnostics.map((d: any) => d.message);
}

for (const [label, src] of [
  ['an fn', 'f = fn(abs(_) * _)\ng = f\nb = g(-1.0, 2.0)\n'],
  ['a function definition', 'f(x, y) = abs(x) * y\ng = f\nb = g(-1.0, 2.0)\n'],
  ['a kernelof', 't = elementof(reals)\ny ~ Normal(mu = t, sigma = 1.0)\nk = kernelof(y, t = t)\nkk = k\n'],
]) {
  test(`an alias of ${label} draws no fixed-value diagnostic`, () => {
    assert.deepEqual(buildDiagnostics(src), []);
  });
}

test('an alias of a loaded module function draws none, and calls through it evaluate', async () => {
  const opts = { bundle: { sources: { 'lib.flatppl': 'f_b = fn(abs(_) * _)\n' } } };
  const src = 'm = load_module("lib.flatppl")\nf_b = m.f_b\nb = f_b(-1.5, 2.0)\n';
  assert.deepEqual(buildDiagnostics(src, opts), []);
  const { ctx } = ctxFor(src, 1, opts);
  const b = await ctx.getMeasure('b');
  assert.equal(b.value.data[0], 3);
});

test('the likelihood a restrict expansion lifts draws no fixed-value diagnostic', () => {
  const src = 'theta ~ Normal(mu = 0.0, sigma = 1.0)\nobs ~ iid(Normal(mu = theta, sigma = 1.0), 3)\n'
    + 'joint_model = lawof(record(theta = theta, obs = obs))\n'
    + 'Pi_post = restrict(joint_model, record(obs = [0.1, 0.2, 0.3]))\n';
  assert.deepEqual(buildDiagnostics(src), []);
});
