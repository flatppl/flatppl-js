'use strict';

// H3 conformance — `joint_likelihood` through an ALIAS CHAIN, plus the
// surviving-node diagnostic (spec §06 sec:joint_likelihood).
//
// Before the H3 fix, lift.inlineJointLikelihoodLift only resolved ONE
// level of ref from the bayesupdate's likelihood arg to the
// joint_likelihood. The natural alias chain
//   L = M;  M = joint_likelihood(L1, L2);  post = bayesupdate(L, prior)
// broke it: the rewrite bailed, the joint_likelihood node survived with
// no derivation kind, and materialise threw `no derivation for
// 'posterior'` with NO diagnostic (silent failure).
//
// This file pins:
//   1. The two-level alias desugars and gives the SAME posterior (atom
//      values + logWeights to 1e-9) as the explicit nested form
//      bayesupdate(L2, bayesupdate(L1, prior)).
//   2. A deliberately-unsupported SURVIVING joint_likelihood (a bare
//      `logdensityof(joint_likelihood(L1, L2), pt)`) yields a CLEAR
//      diagnostic — not the opaque `no derivation` failure.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { makeCtxFactory } = require('./_measure-helpers.ts');

const SAMPLE_COUNT = 2048;
const ROOT_SEED    = 0x70C7A11D;

const makeCtx = makeCtxFactory({ sampleCount: SAMPLE_COUNT, rootSeed: ROOT_SEED });

// Same conjugate model as joint-likelihood.test.ts: two likelihoods over
// a shared parameter mu, in the record-shaped form the bayesupdate
// classifier recognises.
const MODEL = `
mu = draw(Normal(mu = 0.0, sigma = 2.0))
prior = lawof(record(mu = mu))
obs_dist1 = joint(y = Normal(mu = mu, sigma = 1.0))
obs_dist2 = joint(z = Normal(mu = mu, sigma = 0.5))
K1 = functionof(obs_dist1, mu = mu)
K2 = functionof(obs_dist2, mu = mu)
L1 = likelihoodof(K1, record(y = 1.5))
L2 = likelihoodof(K2, record(z = 3.2))
`;

test('two-level alias joint_likelihood desugars away (no joint_likelihood node survives)', () => {
  const { built } = makeCtx(MODEL + `
M = joint_likelihood(L1, L2)
L = M
posterior = bayesupdate(L, prior)
`);
  const pb = built.bindings.get('posterior');
  assert.ok(pb, 'posterior binding exists');
  const ir = pb.ir || (pb.node && pb.node.value);
  assert.ok(JSON.stringify(ir).indexOf('joint_likelihood') === -1,
    'joint_likelihood must be desugared through the alias chain at lift time');
  const der = built.derivations['posterior'];
  assert.ok(der, 'posterior has a derivation');
  assert.equal(der.kind, 'bayesupdate', 'posterior folds to a bayesupdate');
});

test('two-level alias joint_likelihood posterior ≡ nested single-likelihood bayesupdates', async () => {
  const aliased = makeCtx(MODEL + `
M = joint_likelihood(L1, L2)
L = M
posterior = bayesupdate(L, prior)
`);
  const nested = makeCtx(MODEL + `
posterior = bayesupdate(L2, bayesupdate(L1, prior))
`);

  const pA = await aliased.ctx.getMeasure('posterior');
  const pN = await nested.ctx.getMeasure('posterior');

  const muA = pA.fields.mu.samples;
  const muN = pN.fields.mu.samples;
  assert.equal(muA.length, muN.length, 'same atom count');
  for (let i = 0; i < muA.length; i++) {
    assert.ok(Math.abs(muA[i] - muN[i]) < 1e-9,
      `atom ${i}: aliased mu ${muA[i]} != nested ${muN[i]}`);
  }
  const wA = pA.logWeights;
  const wN = pN.logWeights;
  assert.ok(wA && wN, 'both posteriors carry logWeights');
  for (let i = 0; i < wA.length; i++) {
    assert.ok(Math.abs(wA[i] - wN[i]) < 1e-9,
      `atom ${i}: aliased logWeight ${wA[i]} != nested ${wN[i]}`);
  }
});

test('bare logdensityof(joint_likelihood(...)) yields a CLEAR diagnostic, not `no derivation`', () => {
  // joint_likelihood scored directly (not via bayesupdate) is an
  // unsupported form. It must fail with an explicit joint_likelihood
  // diagnostic at lift time, NOT the opaque `no derivation for '...'`
  // from the materialiser.
  let err: any = null;
  try {
    makeCtx(MODEL + `
lp = logdensityof(joint_likelihood(L1, L2), record(y = 1.5))
`);
  } catch (e) {
    err = e;
  }
  assert.ok(err, 'an unsupported surviving joint_likelihood must throw');
  const msg = String(err && err.message);
  assert.ok(/joint_likelihood/.test(msg),
    `diagnostic must mention joint_likelihood, got: ${msg}`);
  assert.ok(!/no derivation/.test(msg),
    `diagnostic must NOT be the opaque 'no derivation' failure, got: ${msg}`);
});

test('M1: scoring a NAMED-REF joint_likelihood directly (J = joint_likelihood(...); logdensityof(J, ...)) is rejected loudly', () => {
  // joint_likelihood's only supported consumption site is bayesupdate.
  // The sibling test above pins the INLINE shape
  // (logdensityof(joint_likelihood(L1, L2), ...)). This pins the distinct
  // NAMED-REF chain: a `J = joint_likelihood(...)` binding consumed by a
  // bare `logdensityof(J, ...)`. That follows a different lift path (J's
  // binding must resolve back to a surviving joint_likelihood node), so it
  // gets its own pin. Direct scoring needs a likelihood-object derivation
  // (follow-up); until then it MUST be reported loudly, naming both
  // joint_likelihood and its only supported site, bayesupdate.
  let err: any = null;
  try {
    makeCtx(MODEL + `
J = joint_likelihood(L1, L2)
s = logdensityof(J, record(y = 1.5))
`);
  } catch (e) {
    err = e;
  }
  assert.ok(err, 'directly scoring a named-ref joint_likelihood must throw');
  const msg = String(err && err.message);
  assert.ok(/joint_likelihood/.test(msg),
    `rejection must mention joint_likelihood, got: ${msg}`);
  assert.ok(/bayesupdate/.test(msg),
    `rejection must point to the only supported site (bayesupdate), got: ${msg}`);
});
