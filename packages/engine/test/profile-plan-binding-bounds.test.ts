// `resolveAxisBaseSet` / `resolveDrawSupport` called `parseSetIR` without the
// bindings, so interval bounds given as constant bindings (`interval(lo,
// hi)`) did not fold and the axis set degraded to `empirical`.
// Oracle: the interval [0.5, 2].
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const profilePlan = require('../profile-plan.ts');

test('an elementof and a Uniform support with constant-binding bounds resolve to the interval', () => {
  const r = processSource('lo = 0.5\nhi = 2.0\nsig = elementof(interval(lo, hi))\n'
    + 'mu ~ Normal(0.0, sig)\nnu ~ Uniform(support = interval(lo, hi))\n');
  const b = orchestrator.buildDerivations(r.bindings).bindings;
  for (const name of ['sig', 'nu']) {
    assert.deepEqual(profilePlan.resolveAxisBaseSet({ kind: 'binding', name }, b),
      { kind: 'interval', lo: 0.5, hi: 2 }, name);
  }
});
