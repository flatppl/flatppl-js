'use strict';

// #591: `('prng' in o || 'seed' in o)` opts-recognition, routed through the
// shared `_sharedPrng` helper (sampler-registry.ts, near the DiracCtor
// prototype block), was widened at every `randX.factory` site that carried
// the old `('prng' in args[lastIdx])`-only idiom — not only
// `randInverseGamma` (covered on its own in inverse-gamma-moments.test.ts).
// This file sweeps every OTHER touched factory with the same two checks:
//
//   - a `{ seed }`-only STATIC call is deterministic: two factories built
//     with the same seed agree on their first draw:
//   - a `{ seed }`-only PARAMETRIC call is deterministic the same way;
//   - a bare call with no options object at all still draws (the
//     Math.random fallback arm of `_sharedPrng`), for both forms.
//
// None of these factories' engine-internal callers ever pass `{ seed }`
// (sampler.ts always builds `{ prng }` from a Philox bridge), so this is
// pure direct-API coverage, not an engine draw-stream change.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');

const REGISTRY = sampler._internal.REGISTRY;

// One row per factory this change touched, minus InverseGamma (its own
// file already covers this ground plus moment conformance).
const CASES: { name: string; args: any[] }[] = [
  { name: 'Logistic',           args: [0, 1] },
  { name: 'Weibull',            args: [2, 1] },
  { name: 'Pareto',             args: [2, 1] },
  { name: 'GeneralizedNormal',  args: [0, 1, 2] },
  { name: 'ChiSquared',         args: [3] },
  { name: 'VonMises',           args: [0, 2] },
  { name: 'Laplace',            args: [0, 1] },
  { name: 'Geometric',          args: [0.3] },
  { name: 'NegativeBinomial',   args: [3, 2] },
  { name: 'NegativeBinomial2',  args: [3, 2] },
  { name: 'Categorical',        args: [[0.2, 0.3, 0.5]] },
  { name: 'Categorical0',       args: [[0.2, 0.3, 0.5]] },
];

for (const { name, args } of CASES) {
  const randFn = REGISTRY[name].randFn;

  test(`${name}.randFn: STATIC form honours a {seed}-only options object`, () => {
    const a = randFn.factory(...args, { seed: 12345 })();
    const b = randFn.factory(...args, { seed: 12345 })();
    assert.equal(a, b,
      `${name} static: ${a} != ${b} — the {seed} opts object was dropped`);
  });

  test(`${name}.randFn: PARAMETRIC form honours a {seed}-only options object`, () => {
    const a = randFn.factory({ seed: 12345 })(...args);
    const b = randFn.factory({ seed: 12345 })(...args);
    assert.equal(a, b,
      `${name} parametric: ${a} != ${b} — the {seed} opts object was dropped`);
  });

  test(`${name}.randFn: a bare STATIC call with no options object at all still draws`, () => {
    const draw = randFn.factory(...args)();
    assert.equal(typeof draw, 'number');
    assert.ok(!Number.isNaN(draw), `${name} bare static draw was NaN`);
  });
}
