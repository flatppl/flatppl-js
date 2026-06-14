'use strict';

// Domain-contract checks (spec §08 parameter domains; engine-concepts
// §17.3 valueset consumer). A static diagnostic on top of the landed
// value-set inference: a distribution parameter whose value set is
// PROVABLY disjoint from its required domain is an error. Strictly
// conservative — only proven violations fire; a `reals` / `unknown` /
// `deferred` parameter passes.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

const errs = (src: any) =>
  processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
const violates = (src: any) => errs(src).some((m: any) => /provably outside/.test(m));

// =====================================================================
// Fires on a proven violation
// =====================================================================

test('domain-contract: negative scale parameter is flagged', () => {
  assert.ok(violates('x = draw(Normal(mu = 0.0, sigma = -1.0))'));
  assert.ok(violates('x = draw(LogNormal(mu = 0.0, sigma = -2.5))'));
  assert.ok(violates('x = draw(Cauchy(location = 0.0, scale = -1.0))'));
});

test('domain-contract: zero violates a strictly-positive parameter', () => {
  // posreals excludes 0 — a degenerate `sigma = 0.0` is caught.
  assert.ok(violates('x = draw(Normal(mu = 0.0, sigma = 0.0))'));
  assert.ok(violates('x = draw(Exponential(rate = 0.0))'));
});

test('domain-contract: negative shape/rate/concentration flagged', () => {
  assert.ok(violates('x = draw(Beta(alpha = -2.0, beta = 3.0))'));
  assert.ok(violates('x = draw(Beta(alpha = 2.0, beta = -3.0))'));
  assert.ok(violates('x = draw(Gamma(shape = -1.0, rate = 2.0))'));
  assert.ok(violates('x = draw(Gamma(shape = 1.0, rate = -2.0))'));
  assert.ok(violates('x = draw(InverseGamma(shape = -1.0, scale = 2.0))'));
  assert.ok(violates('x = draw(Weibull(shape = -1.0, scale = 2.0))'));
  assert.ok(violates('x = draw(StudentT(nu = -3.0))'));
});

test('domain-contract: probability outside [0,1] is flagged', () => {
  assert.ok(violates('x = draw(Bernoulli(p = 1.5))'));
  assert.ok(violates('x = draw(Bernoulli(p = -0.1))'));
  assert.ok(violates('x = draw(Geometric(p = 2.0))'));
  assert.ok(violates('n = external(posintegers)\nx = draw(Binomial(n = n, p = 1.2))'));
});

test('domain-contract: positional argument form is also checked', () => {
  assert.ok(violates('x = draw(Normal(0.0, -1.0))'));
  assert.ok(violates('x = draw(Beta(-1.0, 2.0))'));
});

// =====================================================================
// Does NOT fire — valid params and unprovable cases stay clean
// =====================================================================

test('domain-contract: valid parameters produce no error', () => {
  assert.ok(!violates('x = draw(Normal(mu = 0.0, sigma = 1.0))'));
  assert.ok(!violates('x = draw(Beta(alpha = 2.0, beta = 3.0))'));
  assert.ok(!violates('x = draw(Gamma(shape = 2.0, rate = 0.5))'));
  assert.ok(!violates('x = draw(Bernoulli(p = 0.3))'));
  assert.ok(!violates('x = draw(Exponential(rate = 1.0))'));
});

test('domain-contract: a parameterized/unknown-sign parameter is not flagged', () => {
  // `reals` straddles 0 — not provably outside posreals, so it passes
  // (conservative: this is a runtime concern, not a static one).
  assert.ok(!violates('s = elementof(reals)\nx = draw(Normal(mu = 0.0, sigma = s))'));
  assert.ok(!violates('s = elementof(posreals)\nx = draw(Normal(mu = 0.0, sigma = s))'));
});

test('domain-contract: unconstrained location parameters never flag', () => {
  // mu / location are real-domain — a negative value is perfectly valid.
  assert.ok(!violates('x = draw(Normal(mu = -5.0, sigma = 1.0))'));
  assert.ok(!violates('x = draw(Cauchy(location = -3.0, scale = 1.0))'));
});

test('domain-contract: a fixed-phase positive binding passes', () => {
  assert.ok(!violates('sd = 2.0\nx = draw(Normal(mu = 0.0, sigma = sd))'));
});
