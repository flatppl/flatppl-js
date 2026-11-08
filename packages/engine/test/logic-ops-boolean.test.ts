// §07 `land` / `lor` return booleans. They returned an operand, so a boolean
// read from a vector (stored as 0/1) came back as the number 0 or 1, unlike
// `lnot` / `lxor`. Oracle: the truth table, with boolean results.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');

test('land and lor of booleans read from a vector are booleans', () => {
  const r = processSource('v = [false, true]\n'
    + 'a = land(v[1], true)\nb = lor(v[2], false)\nc = land(true, v[2])\nd = lor(false, v[1])\n');
  const fv = orchestrator.buildDerivations(r.bindings).fixedValues;
  assert.deepEqual(['a', 'b', 'c', 'd'].map((n) => fv.get(n)), [false, true, true, false]);
});
