'use strict';

// Tests for `joint_likelihood(L1, L2, ...)` (spec §06 sec:joint_likelihood):
// combine ≥2 likelihoods by multiplying densities / summing log-densities.
//
// Implemented as a lift-time structural rewrite (lift.inlineJointLikelihoodLift):
// at its documented consumption site `bayesupdate(joint_likelihood(L1,…,Ln), prior)`
// it folds into nested single-likelihood bayesupdates
//   bayesupdate(Ln, … bayesupdate(L1, prior) …),
// reusing the already-implemented bayesupdate path — no new derivation /
// density / materialiser surface.
//
// Posterior equivalence pinned here:
//   bayesupdate(joint_likelihood(L1, L2), prior)
//     ≡  bayesupdate(L2, bayesupdate(L1, prior))
// We compare the posterior empirical measures' per-atom values + log-
// weights: the joint-likelihood posterior must reweight the prior by the
// SUM of the two component log-likelihoods, atom-for-atom (same prior
// atoms + same seed → identical numbers).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SAMPLE_COUNT = 2048;
const ROOT_SEED    = 0x70C7A11D;

function makeCtx(source: any, opts?: any) {
  opts = opts || {};
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: opts.sampleCount != null ? opts.sampleCount : SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return { ctx, lifted, built };
}

// Two likelihoods over a single shared parameter mu, in the record-shaped
// conjugate form the bayesupdate classifier recognises (mirrors the
// Normal-Normal pattern in closed-form-measure-algebra.test.ts):
//   prior = lawof(record(mu = mu)),  mu ~ Normal(0, 2)
//   K1 = functionof(joint(y = Normal(mu, 1.0)),   mu = mu),  obs1 = {y: 1.5}
//   K2 = functionof(joint(z = Normal(mu, 0.5)),   mu = mu),  obs2 = {z: 3.2}
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

test('joint_likelihood desugars away (no joint_likelihood node survives lift)', () => {
  const { built } = makeCtx(MODEL + `
posterior = bayesupdate(joint_likelihood(L1, L2), prior)
`);
  const pb = built.bindings.get('posterior');
  assert.ok(pb, 'posterior binding exists');
  const ir = pb.ir || (pb.node && pb.node.value);
  assert.ok(JSON.stringify(ir).indexOf('joint_likelihood') === -1,
    'joint_likelihood must be desugared at lift time');
  // The rewritten posterior is a bayesupdate (the outer fold step).
  const der = built.derivations['posterior'];
  assert.ok(der, 'posterior has a derivation');
  assert.equal(der.kind, 'bayesupdate', 'posterior folds to a bayesupdate');
});

test('joint_likelihood posterior ≡ nested single-likelihood bayesupdates', async () => {
  const joint = makeCtx(MODEL + `
posterior = bayesupdate(joint_likelihood(L1, L2), prior)
`);
  const nested = makeCtx(MODEL + `
posterior = bayesupdate(L2, bayesupdate(L1, prior))
`);

  const pJ = await joint.ctx.getMeasure('posterior');
  const pN = await nested.ctx.getMeasure('posterior');

  const muJ = pJ.fields.mu.samples;
  const muN = pN.fields.mu.samples;
  // Same prior atoms (same seed) → posterior atom values must match.
  assert.equal(muJ.length, muN.length, 'same atom count');
  for (let i = 0; i < muJ.length; i++) {
    assert.ok(Math.abs(muJ[i] - muN[i]) < 1e-9,
      `atom ${i}: joint mu ${muJ[i]} != nested ${muN[i]}`);
  }
  // And the reweighting (logWeights) must match atom-for-atom.
  const wJ = pJ.logWeights;
  const wN = pN.logWeights;
  assert.ok(wJ && wN, 'both posteriors carry logWeights');
  for (let i = 0; i < wJ.length; i++) {
    assert.ok(Math.abs(wJ[i] - wN[i]) < 1e-9,
      `atom ${i}: joint logWeight ${wJ[i]} != nested ${wN[i]}`);
  }
});

test('joint_likelihood reweights prior by the SUM of component log-likelihoods', async () => {
  // The joint posterior's logWeight at each prior atom mu must equal
  // logpdf(Normal(mu,1), 1.5) + logpdf(Normal(mu,0.5), 3.2) up to a
  // common additive constant (bayesupdate is unnormalized). We compare
  // pairwise DIFFERENCES between atoms, which cancel the constant.
  const joint = makeCtx(MODEL + `
posterior = bayesupdate(joint_likelihood(L1, L2), prior)
`);
  const p = await joint.ctx.getMeasure('posterior');
  const mu = p.fields.mu.samples;
  const w = p.logWeights;
  assert.ok(w, 'posterior carries logWeights');

  const nLogpdf = (x: number, m: number, s: number) =>
    -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
  const analyticLogL = (muVal: number) =>
    nLogpdf(1.5, muVal, 1.0) + nLogpdf(3.2, muVal, 0.5);

  let checks = 0;
  for (let i = 1; i < Math.min(mu.length, 50); i++) {
    const dEngine = w[i] - w[0];
    const dAnalytic = analyticLogL(mu[i]) - analyticLogL(mu[0]);
    assert.ok(Math.abs(dEngine - dAnalytic) < 1e-6,
      `atom ${i}: Δlogw engine ${dEngine} != analytic ${dAnalytic}`);
    checks++;
  }
  assert.ok(checks > 0, 'made at least one comparison');
});
