'use strict';

// End-to-end tests for engine/typeinfer.js — runs the full
// parse → analyze pipeline (which now includes type inference) and
// asserts inferred types and diagnostics on representative sources.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, types: T } = require('..');

function infer(src) {
  const { bindings, diagnostics } = processSource(src);
  const errors = diagnostics.filter(d => d.severity === 'error');
  return { bindings, errors };
}
function typeOf(bindings, name) {
  return bindings.get(name).inferredType;
}

// =====================================================================
// Distributions and basic literals
// =====================================================================

test('distributions: Normal kwargs accept integer literals via promotion', () => {
  // §sec:valuetypes: integer literals satisfy the real-typed kwargs
  // through the canonical embedding integers ⊂ reals. No diagnostic.
  const { bindings, errors } = infer(`m = Normal(mu = 0, sigma = 1)`);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'm'), T.measure(T.REAL)));
});

test('distributions: discrete distributions return integer / boolean measures', () => {
  const { bindings, errors } = infer(`
    b = Bernoulli(p = 0.5)
    p = Poisson(rate = 2)
    bn = Binomial(n = 10, p = 0.3)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'b'),  T.measure(T.BOOLEAN)));
  assert.ok(T.equal(typeOf(bindings, 'p'),  T.measure(T.INTEGER)));
  assert.ok(T.equal(typeOf(bindings, 'bn'), T.measure(T.INTEGER)));
});

test('literals: lexical form decides integer vs real', () => {
  const { bindings, errors } = infer(`
    i = 42
    r = 3.14
    s = "hello"
    b = true
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'i'), T.INTEGER));
  assert.ok(T.equal(typeOf(bindings, 'r'), T.REAL));
  assert.ok(T.equal(typeOf(bindings, 's'), T.STRING));
  assert.ok(T.equal(typeOf(bindings, 'b'), T.BOOLEAN));
});

test('literals: array literal unifies element types and records length', () => {
  const { bindings, errors } = infer(`xs = [1.0, 2.0, 3.0]`);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'xs'), T.array(1, [3], T.REAL)));
});

test('literals: array of integer literals stays integer', () => {
  const { bindings } = infer(`xs = [1, 2, 3]`);
  assert.ok(T.equal(typeOf(bindings, 'xs'), T.array(1, [3], T.INTEGER)));
});

// =====================================================================
// Variates and law extraction
// =====================================================================

test('draw: extracts the value type from a measure', () => {
  const { bindings, errors } = infer(`
    m = Normal(mu = 0, sigma = 1)
    x = draw(m)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'x'), T.REAL));
});

test('lawof: lifts a value type back into a measure', () => {
  const { bindings, errors } = infer(`
    m = Normal(mu = 0, sigma = 1)
    x = draw(m)
    y = 2 * x
    y_dist = lawof(y)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'y_dist'), T.measure(T.REAL)));
});

// =====================================================================
// Measure-algebra type errors — the user's reported cases
// =====================================================================

test('weighted(measure, measure): structurally invalid → diagnostic', () => {
  // The user's invalid1_dist case. arg 0 must be a value but theta_dist
  // is a measure → should produce a clear error pointing at the bad arg.
  const { errors } = infer(`
    theta1_dist = Normal(mu = 0, sigma = 1)
    theta2_dist = Exponential(rate = 1)
    invalid = weighted(theta2_dist, theta1_dist)
  `);
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /weighted: arg 1 expects real, got measure over real/);
  assert.equal(errors[0].severity, 'error');
});

test('weighted(value, value): structurally invalid → diagnostic', () => {
  // The user's invalid2_dist case. arg 1 must be a measure but theta1
  // (a draw) is a real value → diagnostic on the wrong arg.
  const { errors } = infer(`
    theta1_dist = Normal(mu = 0, sigma = 1)
    theta1 = draw(theta1_dist)
    theta2_dist = Exponential(rate = 1)
    theta2 = draw(theta2_dist)
    invalid = weighted(theta2, theta1)
  `);
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /weighted: arg 2 expects measure, got real/);
});

test('weighted(value, measure): valid — no diagnostic, infers measure<real>', () => {
  const { bindings, errors } = infer(`
    theta1_dist = Normal(mu = 0, sigma = 1)
    theta2_dist = Exponential(rate = 1)
    theta2 = draw(theta2_dist)
    valid = weighted(theta2, theta1_dist)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'valid'), T.measure(T.REAL)));
});

// =====================================================================
// Diagnostic locations
// =====================================================================

test('diagnostic locations point at the offending argument, not the whole call', () => {
  const src = 'theta_dist = Normal(mu = 0, sigma = 1)\nbad = weighted(theta_dist, theta_dist)\n';
  const { errors } = infer(src);
  assert.equal(errors.length, 1);
  // Line 2; column should be inside the call's first arg (after
  // "weighted("), not at the beginning of "bad".
  assert.equal(errors[0].loc.start.line, 1);   // 0-based: line 2 = index 1
  assert.ok(errors[0].loc.start.col > 0);
});

// =====================================================================
// Cycles
// =====================================================================

test('cyclic bindings: inference falls back to %failed without diverging', () => {
  // Direct cycle. The analyzer surfaces undefined-name warnings; we
  // just want type inference to terminate and the offending binding
  // to carry a failed type.
  const { bindings } = infer(`
    a = b
    b = a
  `);
  // Both end up failed; we don't insist on any specific message.
  assert.equal(typeOf(bindings, 'a').kind, 'failed');
  assert.equal(typeOf(bindings, 'b').kind, 'failed');
});

// =====================================================================
// Composite types
// =====================================================================

test('record: produces record<…> with field types from kwargs', () => {
  const { bindings } = infer(`
    r = record(x = 1.0, y = 2)
  `);
  const t = typeOf(bindings, 'r');
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.x, T.REAL));
  assert.ok(T.equal(t.fields.y, T.INTEGER));
});

test('joint: produces measure<record<…>> from measure-typed kwargs', () => {
  const { bindings, errors } = infer(`
    a = Normal(mu = 0, sigma = 1)
    b = Exponential(rate = 1)
    j = joint(x = a, y = b)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'j');
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'record');
  assert.ok(T.equal(t.domain.fields.x, T.REAL));
  assert.ok(T.equal(t.domain.fields.y, T.REAL));
});

test('joint: a value-typed kwarg is a structural error', () => {
  const { errors } = infer(`
    a = Normal(mu = 0, sigma = 1)
    x = draw(a)
    j = joint(p = a, q = x)
  `);
  assert.ok(errors.some(e => /joint kwarg "q" expects a measure/.test(e.message)));
});

// =====================================================================
// elementof + set constructors
// =====================================================================

test('elementof: bare set name → structural value type', () => {
  const { bindings, errors } = infer(`
    a = elementof(reals)
    b = elementof(integers)
    c = elementof(booleans)
    d = elementof(posreals)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'a'), T.REAL));
  assert.ok(T.equal(typeOf(bindings, 'b'), T.INTEGER));
  assert.ok(T.equal(typeOf(bindings, 'c'), T.BOOLEAN));
  assert.ok(T.equal(typeOf(bindings, 'd'), T.REAL));   // refinement → real
});

test('elementof(cartpow(S, n, …)): array shape and element type', () => {
  const { bindings, errors } = infer(`
    a = elementof(cartpow(reals, 3))
    b = elementof(cartpow(posreals, 2, 4))
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'a'), T.array(1, [3], T.REAL)));
  assert.ok(T.equal(typeOf(bindings, 'b'), T.array(2, [2, 4], T.REAL)));
});

test('elementof(cartprod): kwargs form → record', () => {
  const { bindings, errors } = infer(`
    p = elementof(cartprod(x = reals, y = integers))
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'p');
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.x, T.REAL));
  assert.ok(T.equal(t.fields.y, T.INTEGER));
});
