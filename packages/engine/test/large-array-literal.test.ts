'use strict';

// A determinized model inlines its data as literals. An array literal longer
// than the JS argument-count limit must still evaluate: posteriorDB diamonds
// inlines a 5000 x 25 design matrix as one 125000-element `array([...], ...)`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');

test('a 200000-element array literal evaluates', () => {
  const n = 200000;
  const src = `v = [${new Array(n).fill('1.0').join(', ')}]\ns = sum(v)\nc = cat(v, [2.0])\nt = sum(c)\n`;
  const r = processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const { fixedValues } = orchestrator.buildDerivations(r.bindings);
  assert.equal(fixedValues.get('s'), n);
  assert.equal(fixedValues.get('t'), n + 2);
});
