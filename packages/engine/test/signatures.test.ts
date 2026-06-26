'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');

function sigOf(src: string, name: string) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  return orchestrator.signatureOf(name, built.bindings);
}

test('signatureOf: joint_likelihood unions term inputs + carries terms', () => {
  const src = `
n = elementof(cartpow(posreals, 2))
m = functionof(Poisson.(n))
L1 = likelihoodof(m, [5.0, 6.0])
L2 = likelihoodof(m, [4.0, 7.0])
J = joint_likelihood(L1, L2)
`;
  const sig = sigOf(src, 'J');
  assert.ok(sig, 'joint signature must not be null');
  assert.equal(sig.kind, 'likelihood');
  assert.equal(sig.inputs.length, 1);            // both terms share `n` → unioned to one
  assert.equal(sig.inputs[0].paramName, 'n');
  assert.equal(sig.terms.length, 2);
  assert.ok(sig.terms.every((t: any) => t.body && t.obsIR), 'each term carries body + obsIR');
  // axes come straight from the unioned inputs
  const axes = orchestrator.distributeAxes(sig);
  assert.deepEqual(axes.map((a: any) => a.key), ['n[1]', 'n[2]']);
});
