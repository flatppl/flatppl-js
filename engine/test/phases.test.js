'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, computePhases } = require('../index');

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

  // joint_model has stochastic ancestors so its phase is stochastic per spec.
  assert.equal(p.joint_model, 'stochastic');

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
