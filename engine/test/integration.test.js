'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { processSource, computeSubDAG } = require('../index');

// Integration tests run against bundled flatppl source files copied from
// the flatppl-examples and statsmodel-rosetta-stone sibling repos. Update
// the copies in `fixtures/` when the originals change.

const FIXTURES_DIR = path.join(__dirname, 'fixtures');

const PARSE_FIXTURES = [
  'bayesian_inference_1.flatppl',
  'bayesian_inference_2.flatppl',
  'flatppl-uncorrelated_background-ma-auxm.flatppl',
  'flatppl-uncorrelated_background-ma-priors.flatppl',
  'flatppl-uncorrelated_background-draws-auxm.flatppl',
  'flatppl-uncorrelated_background-draws-priors.flatppl',
];

for (const name of PARSE_FIXTURES) {
  test(`integration: ${name} parses without errors`, () => {
    const src = fs.readFileSync(path.join(FIXTURES_DIR, name), 'utf8');
    const { diagnostics } = processSource(src);
    const errors = diagnostics.filter(d => d.severity === 'error');
    if (errors.length > 0) {
      for (const e of errors) {
        console.error(`  line ${e.loc.start.line + 1}: ${e.message}`);
      }
    }
    assert.equal(errors.length, 0, `${errors.length} errors in ${name}`);
  });
}

test('integration: bayesian_inference_1 forward_kernel DAG includes correct ancestors', () => {
  const src = fs.readFileSync(path.join(FIXTURES_DIR, 'bayesian_inference_1.flatppl'), 'utf8');
  const { bindings } = processSource(src);
  const dag = computeSubDAG(bindings, 'forward_kernel');
  const ids = new Set(dag.nodes.map(n => n.id));
  // Must include kernel target (obs), boundaries (theta1, theta2),
  // and the deterministic chain (a, b, f_a, f_b)
  for (const id of ['forward_kernel', 'obs', 'a', 'b', 'f_a', 'f_b', 'theta1', 'theta2']) {
    assert.ok(ids.has(id), `missing ${id}`);
  }
  assert.equal(dag.nodes.find(n => n.id === 'theta1').isBoundary, true);
  assert.equal(dag.nodes.find(n => n.id === 'theta2').isBoundary, true);
});
