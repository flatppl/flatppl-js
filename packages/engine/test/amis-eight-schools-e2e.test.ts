'use strict';
// amis-eight-schools-e2e.test.ts — end-to-end EAMIS posterior for eight-schools.
//
// Calls materialiseMeasure('posterior', ctx, {backend:'amis', …}) on the
// eight-schools hierarchical model (10-dimensional: mu, tau, theta[0..7]).
// Asserts: no throw, IS diagnostics present (method/ess/K), record fields, and
// a recovery sanity check on mu (the classic eight-schools posterior mean for
// mu sits well inside [-5, 20] — a wide band that still catches a broken chain).

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor }    = require('./density/regression-baseline.test.ts');
const { materialiser } = require('..');

const FIXTURE = path.join(__dirname, 'fixtures/baseline/eight-schools.flatppl');

function mean(measure: any) {
  const s = measure.samples || (measure.value && measure.value.data);
  let m = 0; for (let i = 0; i < s.length; i++) m += s[i];
  return m / s.length;
}

test('eight-schools: amis backend returns finite draws and IS diagnostics', async () => {
  // A realistic prior pool (the proposal is initialised from prior draws, so a
  // single-atom pool would force the degenerate-init fallback rather than
  // exercising the real adaptation path).
  const src = fs.readFileSync(FIXTURE, 'utf8');
  const { ctx } = ctxFor(src, 2000);

  const m = await materialiser.materialiseMeasure('posterior', ctx, {
    backend: 'amis', amisIters: 30, amisSamples: 300, seed: 1,
  });

  assert.ok(m, 'materialiseMeasure returned a result');
  assert.ok(m.diagnostics, 'result has diagnostics');

  const dg = m.diagnostics;
  assert.equal(dg.method, 'amis', 'diagnostics tagged amis');
  assert.ok(typeof dg.ess === 'number' && dg.ess > 0, `ess ${dg.ess} > 0`);
  assert.ok(typeof dg.K === 'number' && dg.K >= 1, `K ${dg.K} >= 1`);
  assert.ok(dg.perParam && 'mu' in dg.perParam, 'perParam has mu');

  // Record measure with the prior's fields (drop-in shape).
  assert.ok(m.fields && typeof m.fields === 'object', 'result has fields (record measure)');
  assert.ok('mu' in m.fields, 'fields has mu');
  assert.ok('tau' in m.fields, 'fields has tau');
  assert.ok('theta' in m.fields, 'fields has theta');

  // Recovery: the eight-schools posterior mean for mu is ≈ 4.4 (textbook);
  // assert a band that catches a broken/collapsed chain but tolerates MC error.
  // tau (a positive scale) recovers a moderate positive value.
  const muMean = mean(m.fields.mu);
  assert.ok(Number.isFinite(muMean) && muMean > 1 && muMean < 9, `mu mean ${muMean} ≈ 4.4`);
  const tauMean = mean(m.fields.tau);
  assert.ok(Number.isFinite(tauMean) && tauMean > 1 && tauMean < 12, `tau mean ${tauMean} moderate positive`);
});
