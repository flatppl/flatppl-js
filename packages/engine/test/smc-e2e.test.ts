'use strict';
// smc-e2e.test.ts — end-to-end SMC posterior through materialiseMeasure on the
// hierarchical models AMIS collapsed on, plus a check that the prior/likelihood
// scorer split sums to the whole posterior.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor }       = require('./density/regression-baseline.test.ts');
const { materialiser } = require('..');
const MV               = require('../model-view.ts');
const rng              = require('../rng.ts');
const sampler          = require('../sampler.ts');

const FIX = path.join(__dirname, 'fixtures/baseline');
function mean(m: any) { const s = m.samples || (m.value && m.value.data); let x = 0; for (let i = 0; i < s.length; i++) x += s[i]; return x / s.length; }

test('eight-schools: smc backend recovers mu, finite evidence', async () => {
  const src = fs.readFileSync(path.join(FIX, 'eight-schools.flatppl'), 'utf8');
  const m = await materialiser.materialiseMeasure('posterior', ctxFor(src, 2000).ctx,
    { backend: 'smc', smcSteps: 16, smcCESS: 0.8, seed: 1 });
  const dg = m.diagnostics;
  assert.equal(dg.method, 'smc');
  assert.ok(Number.isFinite(dg.logZ), `logZ ${dg.logZ} finite`);
  assert.ok(dg.rungs >= 2, `rungs ${dg.rungs} >= 2`);
  assert.ok('mu' in m.fields && 'tau' in m.fields && 'theta' in m.fields, 'fields present');
  // mu is the well-determined parameter (≈4.4); tau's posterior is diffuse, so
  // only assert it stays positive.
  const muMean = mean(m.fields.mu);
  assert.ok(Number.isFinite(muMean) && muMean > 0 && muMean < 9, `mu mean ${muMean} ≈ 4.4`);
  assert.ok(mean(m.fields.tau) > 0, 'tau positive');
});

test('partial-pooling: smc backend recovers phi (where AMIS collapsed)', async () => {
  const src = fs.readFileSync(path.join(FIX, 'partial-pooling.flatppl'), 'utf8');
  const m = await materialiser.materialiseMeasure('posterior', ctxFor(src, 2000).ctx,
    { backend: 'smc', smcSteps: 16, smcCESS: 0.8, seed: 1 });
  assert.equal(m.diagnostics.method, 'smc');
  const phi = mean(m.fields.phi);
  // MH/numpyro put phi ≈ 0.31; AMIS collapsed to ~0.22. SMC should land near 0.31.
  assert.ok(phi > 0.25 && phi < 0.37, `phi ${phi} ≈ 0.31`);
});

test('logPriorLikBatch split sums to the full log-posterior', async () => {
  const src = fs.readFileSync(path.join(FIX, 'eight-schools.flatppl'), 'utf8');
  const ctx = ctxFor(src, 2000).ctx;
  const dv = ctx.lookupDerivation ? ctx.lookupDerivation('posterior') : ctx.derivations.posterior;
  const mv = await MV.buildModelViewFromCtx(ctx, dv);
  const k = rng.keyFromSeed(2);
  const prng = sampler.makePhiloxPrngAdapter(rng.stateFromKey(k[0], k[1]));
  const ys = mv.initFromPrior(16, prng);
  const { prior, lik } = mv.logPriorLikBatch(ys);
  const full = mv.logPosteriorBatch(ys);
  let maxDiff = 0;
  for (let i = 0; i < ys.length; i++) {
    if (!Number.isFinite(full[i])) continue;
    maxDiff = Math.max(maxDiff, Math.abs((prior[i] + lik[i]) - full[i]));
  }
  assert.ok(maxDiff < 1e-9, `prior+lik == logPosterior (max diff ${maxDiff})`);
});
