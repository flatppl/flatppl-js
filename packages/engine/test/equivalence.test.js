'use strict';

// =====================================================================
// Variate-style ≡ Measure-Algebra-style equivalence tests
// =====================================================================
//
// FlatPPL supports two surface styles for expressing the same model:
//
//   - Variate style: `theta ~ M1; y ~ M2(theta); joint_model =
//     lawof(record(theta = theta, y = y))`. Uses stochastic-node
//     bindings (`~`) and `lawof` to materialise the joint as a
//     measure.
//
//   - Measure-algebra style: `prior = lawof(record(theta = draw(M1)));
//     forward = functionof(joint(y = M2), theta = theta);
//     joint_model = jointchain(prior, forward)`. Uses measure-algebra
//     combinators directly.
//
// These should produce IDENTICAL distributions: same per-atom marginals,
// same closed-form log-densities at any point. This file pins that
// contract so refactors don't drift one style relative to the other.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker');

const SAMPLE_COUNT = 1024;
const ROOT_SEED    = 12345;

function makeCtx(source) {
  const lifted = processSource(source);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg) => Promise.resolve(worker.handle(msg)),
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

function variance(arr) {
  const m = mean(arr);
  let s = 0;
  for (let i = 0; i < arr.length; i++) {
    const d = arr[i] - m;
    s += d * d;
  }
  return s / arr.length;
}

// =====================================================================
// Model A: hierarchical Normal-Normal
//
//   theta ~ Normal(0, 1)
//   y     ~ Normal(theta, 1)
//
// Equivalent shapes:
//   variate: theta = draw(Normal(...)); y = draw(Normal(theta, 1));
//            joint_model = lawof(record(theta = theta, y = y))
//   MA:      prior = lawof(record(theta = draw(Normal(...))))
//            forward = functionof(joint(y = Normal(theta, 1)), theta=theta)
//            joint_model = jointchain(prior, forward)
//
// Statistical predictions (analytic):
//   E[theta] = 0; Var[theta] = 1
//   E[y]     = 0; Var[y]     = 2   (marginal — Normal(0, √2))
//   logp(theta=0, y=0) = -½log(2π) + -½log(2π) = -log(2π) ≈ -1.8379
//
// We pin both per-axis marginal sample statistics (within MC error)
// AND the analytic log-density.

const VARIATE_NORMAL_NORMAL = `
theta = draw(Normal(mu = 0.0, sigma = 1.0))
y     = draw(Normal(mu = theta, sigma = 1.0))
joint_model = lawof(record(theta = theta, y = y))
lp    = logdensityof(joint_model, record(theta = 0.0, y = 0.0))
`;

const MA_NORMAL_NORMAL = `
theta = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta = theta))
obs_dist = joint(y = Normal(mu = theta, sigma = 1.0))
forward_kernel = functionof(obs_dist, theta = theta)
joint_model = jointchain(prior, forward_kernel)
lp    = logdensityof(joint_model, record(theta = 0.0, y = 0.0))
`;

test('equivalence: variate-style and MA-style hierarchical Normal-Normal classify identically', async () => {
  const varCtx = makeCtx(VARIATE_NORMAL_NORMAL);
  const maCtx  = makeCtx(MA_NORMAL_NORMAL);
  // BEHAVIOURAL equivalence: the two authoring styles must produce the
  // SAME model. Pre-consolidation this was proxied by both classifying
  // to the same internal kind='record' (variate-style via an
  // alias-chase, MA-style via the inlineChainOps rewrite). Under the
  // first-class consolidation those internal kinds legitimately differ
  // (variate-style lawof(record) → record/alias; MA-style jointchain →
  // kind:'jointchain') — what must hold is OBSERVABLE equivalence: the
  // materialised measure has the same record shape. Numeric agreement
  // (sample stats, closed-form density) is pinned by the other
  // equivalence tests in this file, which pass under first-class.
  const vm = await varCtx.getMeasure('joint_model');
  const mm = await maCtx.getMeasure('joint_model');
  assert.ok(vm.fields && mm.fields,
    'both styles materialise as a record measure');
  assert.deepEqual(Object.keys(vm.fields).sort(), ['theta', 'y']);
  assert.deepEqual(Object.keys(mm.fields).sort(), ['theta', 'y']);
});

test('equivalence: per-axis marginal sample statistics match within MC error', async () => {
  const varCtx = makeCtx(VARIATE_NORMAL_NORMAL);
  const maCtx  = makeCtx(MA_NORMAL_NORMAL);
  const vm = await varCtx.getMeasure('joint_model');
  const mm = await maCtx.getMeasure('joint_model');
  // theta marginal: standard Normal. E=0, Var=1.
  const v_theta = vm.fields.theta.samples;
  const m_theta = mm.fields.theta.samples;
  assert.ok(Math.abs(mean(v_theta) - 0) < 0.15, 'variate theta mean ≈ 0, got ' + mean(v_theta));
  assert.ok(Math.abs(mean(m_theta) - 0) < 0.15, 'MA theta mean ≈ 0, got '      + mean(m_theta));
  assert.ok(Math.abs(variance(v_theta) - 1) < 0.2, 'variate theta var ≈ 1, got ' + variance(v_theta));
  assert.ok(Math.abs(variance(m_theta) - 1) < 0.2, 'MA theta var ≈ 1, got '      + variance(m_theta));
  // y marginal: Normal(0, √2). E=0, Var=2.
  const v_y = vm.fields.y.samples;
  const m_y = mm.fields.y.samples;
  assert.ok(Math.abs(mean(v_y) - 0) < 0.2, 'variate y mean ≈ 0, got ' + mean(v_y));
  assert.ok(Math.abs(mean(m_y) - 0) < 0.2, 'MA y mean ≈ 0, got '      + mean(m_y));
  assert.ok(Math.abs(variance(v_y) - 2) < 0.4, 'variate y var ≈ 2, got ' + variance(v_y));
  assert.ok(Math.abs(variance(m_y) - 2) < 0.4, 'MA y var ≈ 2, got '      + variance(m_y));
});

test('equivalence: closed-form joint logdensity matches the analytic value AND each other', async () => {
  const varCtx = makeCtx(VARIATE_NORMAL_NORMAL);
  const maCtx  = makeCtx(MA_NORMAL_NORMAL);
  const v_lp = await varCtx.getMeasure('lp');
  const m_lp = await maCtx.getMeasure('lp');
  // Analytic: logpdf_N(0; 0, 1) + logpdf_N(0; 0, 1) = -log(2π)
  const expected = -Math.log(2 * Math.PI);
  // Both styles should give the same closed-form scalar at every
  // prior atom (the observation pins θ and y, so per-atom variation
  // collapses).
  assert.ok(Math.abs(v_lp.samples[0] - expected) < 1e-10,
    'variate-style logp should be -log(2π), got ' + v_lp.samples[0]);
  assert.ok(Math.abs(m_lp.samples[0] - expected) < 1e-10,
    'MA-style logp should be -log(2π), got ' + m_lp.samples[0]);
  assert.ok(Math.abs(v_lp.samples[0] - m_lp.samples[0]) < 1e-12,
    'variate and MA styles should agree exactly on closed-form density');
});

// =====================================================================
// Model B: 3-level chain
//
//   theta1 ~ Normal(0, 1)
//   theta2 ~ Normal(theta1, 1)
//   y      ~ Normal(theta2, 1)
//
// Closed-form joint density at (theta1, theta2, y) = (0, 0, 0):
//   -3 × ½ log(2π)
//
// Exercises N-ary jointchain (via the step-wise fold) AND the
// source-key env-threading fix for closed-form density on chains
// where field name ≠ source binding name.

const VARIATE_3LEVEL = `
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
theta2 = draw(Normal(mu = theta1, sigma = 1.0))
y      = draw(Normal(mu = theta2, sigma = 1.0))
joint_model = lawof(record(theta1 = theta1, theta2 = theta2, y = y))
lp = logdensityof(joint_model, record(theta1 = 0.0, theta2 = 0.0, y = 0.0))
`;

const MA_3LEVEL = `
theta1 = draw(Normal(mu = 0.0, sigma = 1.0))
prior = lawof(record(theta1 = theta1))
theta2 = draw(Normal(mu = theta1, sigma = 1.0))
obs_dist1 = joint(theta2 = Normal(mu = theta1, sigma = 1.0))
obs_dist2 = joint(y = Normal(mu = theta2, sigma = 1.0))
K1 = functionof(obs_dist1, theta1 = theta1)
K2 = functionof(obs_dist2, theta1 = theta1, theta2 = theta2)
joint_model = jointchain(prior, K1, K2)
lp = logdensityof(joint_model, record(theta1 = 0.0, theta2 = 0.0, y = 0.0))
`;

test('equivalence: 3-level model — joint density agrees across styles AND with analytic', async () => {
  const varCtx = makeCtx(VARIATE_3LEVEL);
  const maCtx  = makeCtx(MA_3LEVEL);
  const v_lp = await varCtx.getMeasure('lp');
  const m_lp = await maCtx.getMeasure('lp');
  const expected = -1.5 * Math.log(2 * Math.PI);
  assert.ok(Math.abs(v_lp.samples[0] - expected) < 1e-10,
    'variate 3-level logp should be -1.5·log(2π), got ' + v_lp.samples[0]);
  assert.ok(Math.abs(m_lp.samples[0] - expected) < 1e-10,
    'MA-style 3-level logp (N-ary jointchain) should be -1.5·log(2π), got '
    + m_lp.samples[0]);
  assert.ok(Math.abs(v_lp.samples[0] - m_lp.samples[0]) < 1e-12,
    'variate and N-ary MA styles should agree exactly on the 3-level joint density');
});

test('equivalence: 3-level model — y marginal variance matches Normal(0, √3)', async () => {
  // y = theta2 + ε3, theta2 = theta1 + ε2, theta1 = ε1, εi ~ N(0,1) iid.
  // So y is N(0, 3), variance = 3.
  const varCtx = makeCtx(VARIATE_3LEVEL);
  const maCtx  = makeCtx(MA_3LEVEL);
  const vm = await varCtx.getMeasure('joint_model');
  const mm = await maCtx.getMeasure('joint_model');
  const v_y = vm.fields.y.samples;
  const m_y = mm.fields.y.samples;
  assert.ok(Math.abs(variance(v_y) - 3) < 0.5,
    'variate y marginal var ≈ 3 (Normal(0, √3)), got ' + variance(v_y));
  assert.ok(Math.abs(variance(m_y) - 3) < 0.5,
    'MA y marginal var ≈ 3 (Normal(0, √3)), got ' + variance(m_y));
});
