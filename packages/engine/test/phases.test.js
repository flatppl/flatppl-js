'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, computePhases } = require('../index');
const { computePhasesForScope } = require('../analyzer');
const { computeSubDAG } = require('../dag');

function phasesOf(src) {
  const { bindings } = processSource(src);
  const result = {};
  for (const [name, b] of bindings) result[name] = b.phase;
  return result;
}

// --- Per-spec direct rules ---

test('phase: draw is stochastic', () => {
  const p = phasesOf('x = draw(Normal(mu = 0, sigma = 1))\n');
  assert.equal(p.x, 'stochastic');
});

test('phase: elementof is parameterized', () => {
  const p = phasesOf('x = elementof(reals)\n');
  assert.equal(p.x, 'parameterized');
});

test('phase: external is fixed', () => {
  const p = phasesOf('x = external(reals)\n');
  assert.equal(p.x, 'fixed');
});

test('phase: literal is fixed', () => {
  const p = phasesOf('x = 1.5\narr = [1, 2, 3]\n');
  assert.equal(p.x, 'fixed');
  assert.equal(p.arr, 'fixed');
});

// --- Propagation through ancestors ---

test('phase: propagates stochastic through deterministic operations', () => {
  // a = f(theta1) where theta1 is draw → a is stochastic
  const p = phasesOf(`
theta1 = draw(Normal(mu = 0, sigma = 1))
a = 2 * theta1 + 5
b = a + 1
`);
  assert.equal(p.theta1, 'stochastic');
  assert.equal(p.a, 'stochastic');
  assert.equal(p.b, 'stochastic');
});

test('phase: parameterized propagates through deterministic ops', () => {
  const p = phasesOf(`
mu_p = elementof(reals)
a = mu_p + 1
b = a * 2
`);
  assert.equal(p.mu_p, 'parameterized');
  assert.equal(p.a, 'parameterized');
  assert.equal(p.b, 'parameterized');
});

test('phase: stochastic dominates over parameterized', () => {
  const p = phasesOf(`
mu_p = elementof(reals)
theta1 = draw(Normal(mu = mu_p, sigma = 1))
a = mu_p + theta1
`);
  assert.equal(p.theta1, 'stochastic');
  assert.equal(p.a, 'stochastic');
});

test('phase: parameterized dominates over fixed', () => {
  const p = phasesOf(`
n = external(integers)
mu_p = elementof(reals)
a = mu_p + n
`);
  assert.equal(p.n, 'fixed');
  assert.equal(p.mu_p, 'parameterized');
  assert.equal(p.a, 'parameterized');
});

// --- Bayesian inference example ---

test('phase: bayesian_inference_2 fixture has correct phases', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'bayesian_inference_2.flatppl'), 'utf8');
  const p = phasesOf(src);

  // Stochastic chain
  assert.equal(p.theta1, 'stochastic');
  assert.equal(p.theta2, 'stochastic');
  assert.equal(p.a, 'stochastic', "a depends on theta1 (stochastic)");
  assert.equal(p.b, 'stochastic', "b depends on theta1, theta2 (stochastic)");
  assert.equal(p.obs, 'stochastic');

  // joint_model = lawof(record(...)). Per spec §sec:lawof line 309-314,
  // lawof absorbs stochasticity into the reified measure — the result
  // is fixed unless an elementof remains in the ancestor closure. Here
  // theta1/theta2/obs are all draws of literal-kwarg distributions
  // (no elementof anywhere), so joint_model is fixed.
  assert.equal(p.joint_model, 'fixed');

  // observed_data is a literal
  assert.equal(p.observed_data, 'fixed');
});

// --- computePhases as a standalone function ---

test('phase: computePhases works on already-built bindings', () => {
  const { bindings } = processSource(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
y = 2 * x
`);
  const phases = computePhases(bindings);
  assert.equal(phases.get('mu'), 'parameterized');
  assert.equal(phases.get('x'), 'stochastic');
  assert.equal(phases.get('y'), 'stochastic');
});

// --- computePhasesForScope: scope-local phase under boundaries ---

test('phase: computePhasesForScope cuts the chain at boundary names', () => {
  // Globally beta1 is stochastic (depends on draw via theta1).
  // With theta1 declared as a boundary input, beta1's phase walk
  // stops at theta1 → 'parameterized', so beta1 itself reads as
  // 'parameterized' too.
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
beta1 = 2 * theta1
`);
  const global = computePhases(bindings);
  assert.equal(global.get('theta1'), 'stochastic');
  assert.equal(global.get('beta1'),  'stochastic');

  const scoped = computePhasesForScope(bindings, new Set(['theta1']));
  assert.equal(scoped.get('theta1'), 'parameterized');
  assert.equal(scoped.get('beta1'),  'parameterized');
});

test('phase: computePhasesForScope with empty boundaries === computePhases', () => {
  const { bindings } = processSource(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
y = 2 * x
`);
  const a = computePhases(bindings);
  const b = computePhasesForScope(bindings, new Set());
  for (const k of a.keys()) assert.equal(b.get(k), a.get(k));
});

// --- DAG: scope-local phase override applied to in-bubble nodes ---

test('phase: DAG nodes inside a kernel bubble carry scope-local phase', () => {
  // forward_kernel's body has theta1/theta2 as boundary inputs.
  // Inside the bubble, both they and beta1 (= 2*theta2) read as
  // 'parameterized'; outside, they're stochastic.
  const { bindings } = processSource(`
theta1 = draw(Normal(mu = 0, sigma = 1))
theta2 = draw(Exponential(rate = 1))
beta1  = 2 * theta2
obs    = draw(Normal(mu = beta1, sigma = 1))
fk     = functionof(obs, theta1 = theta1, theta2 = theta2)
`);
  const dag = computeSubDAG(bindings, 'fk');
  const byId = new Map(dag.nodes.map(n => [n.id, n]));
  // Inside the kernel: theta2 and beta1 cut by the boundary.
  assert.equal(byId.get('theta2').phase, 'parameterized');
  assert.equal(byId.get('beta1').phase,  'parameterized');
  // Global view of the same bindings still has them stochastic.
  const global = computePhases(bindings);
  assert.equal(global.get('theta2'), 'stochastic');
  assert.equal(global.get('beta1'),  'stochastic');
});
