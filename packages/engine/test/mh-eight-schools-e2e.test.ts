'use strict';
// mh-eight-schools-e2e.test.ts — Gate B: end-to-end MH posterior for eight-schools.
//
// Calls materialiseMeasure('posterior', ctx, {backend:'mh', …}) on the
// eight-schools hierarchical model (10-dimensional: mu, tau, theta[0..7]).
// Asserts: no throw, diagnostics present, acceptRate > 0, mu rHat finite.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor }    = require('./density/regression-baseline.test.ts');
const { materialiser } = require('..');

const FIXTURE = path.join(__dirname, 'fixtures/baseline/eight-schools.flatppl');

test('eight-schools: mh backend returns finite draws and diagnostics', async () => {
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { ctx } = ctxFor(src, 1);

  const m = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'mh', chains: 4, warmup: 200, draws: 200, seed: 1,
  });

  assert.ok(m, 'materialiseMeasure returned a result');
  assert.ok(m.diagnostics, 'result has diagnostics');

  const { acceptRate, perParam } = m.diagnostics;
  assert.ok(
    typeof acceptRate === 'number' && acceptRate > 0,
    `acceptRate ${acceptRate} should be > 0`,
  );
  assert.ok(
    perParam && typeof perParam === 'object',
    'perParam present',
  );

  // Check that mu coordinate is present and finite
  assert.ok('mu' in perParam, 'perParam has mu');
  const { rHat } = perParam.mu;
  assert.ok(Number.isFinite(rHat), `mu rHat ${rHat} is finite`);

  // Check that at least the tau coordinate is present
  assert.ok('tau' in perParam, 'perParam has tau');

  // Check that theta vector coordinates are present
  assert.ok('theta[0]' in perParam, 'perParam has theta[0]');
  assert.ok('theta[7]' in perParam, 'perParam has theta[7]');
  assert.ok(
    Number.isFinite(perParam['theta[0]'].rHat),
    `theta[0] rHat ${perParam['theta[0]'].rHat} is finite`,
  );

  // Check that the record measure has fields
  assert.ok(m.fields && typeof m.fields === 'object', 'result has fields (record measure)');
  assert.ok('mu' in m.fields, 'fields has mu');
});
