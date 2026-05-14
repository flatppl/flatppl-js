'use strict';

// Spec §07 Approximation functions: polynomial, bernstein, stepwise.
// All three are pure value-typed functions over fixed-phase
// coefficient / edge arrays plus a scalar evaluation point. The
// sampler's evaluateCall dispatches them via dedicated kwarg cases
// rather than ARITH_OPS (which only handles positional args).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler');

function lit(v)   { return { kind: 'lit', value: v }; }
function vec(...vs) { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function call(op, kwargs) { return { kind: 'call', op, kwargs }; }

// =====================================================================
// polynomial: Σ a_i · x^i (power-series basis)
// =====================================================================

test('polynomial: constant coefficient ⇒ constant value', () => {
  // [3] at any x → 3
  const ir = call('polynomial', { coefficients: vec(3), x: lit(7) });
  assert.equal(sampler.evaluateExpr(ir, {}), 3);
});

test('polynomial: linear: 1 + 2x at x = 5 ⇒ 11', () => {
  const ir = call('polynomial', { coefficients: vec(1, 2), x: lit(5) });
  assert.equal(sampler.evaluateExpr(ir, {}), 11);
});

test('polynomial: quadratic: 2 + 3x + 4x^2 at x = 2 ⇒ 24', () => {
  const ir = call('polynomial', { coefficients: vec(2, 3, 4), x: lit(2) });
  assert.equal(sampler.evaluateExpr(ir, {}), 24);
});

test('polynomial: empty coefficients ⇒ 0', () => {
  const ir = call('polynomial', { coefficients: vec(), x: lit(5) });
  assert.equal(sampler.evaluateExpr(ir, {}), 0);
});

// =====================================================================
// bernstein: Σ a_k · C(n, k) · x^k · (1-x)^{n-k}  on [0, 1]
// =====================================================================

test('bernstein: equal coefficients ⇒ constant value (partition-of-unity)', () => {
  // All a_k = 1 ⇒ f(x) = Σ B_{n,k}(x) ≡ 1 by the partition-of-unity
  // property of Bernstein polynomials. Holds for any x ∈ [0, 1].
  const ir = call('bernstein', { coefficients: vec(1, 1, 1, 1), x: lit(0.5) });
  assert.ok(Math.abs(sampler.evaluateExpr(ir, {}) - 1) < 1e-12);
});

test('bernstein: at x=0 ⇒ coefficients[0]', () => {
  const ir = call('bernstein', { coefficients: vec(2, 5, 7), x: lit(0) });
  assert.equal(sampler.evaluateExpr(ir, {}), 2);
});

test('bernstein: at x=1 ⇒ coefficients[n] (last coefficient)', () => {
  // The 1-x = 0 short-circuit branch should return coeffs[n] verbatim
  // — the analytic limit value.
  const ir = call('bernstein', { coefficients: vec(2, 5, 7), x: lit(1) });
  assert.equal(sampler.evaluateExpr(ir, {}), 7);
});

test('bernstein: linear coefficients ⇒ linear interpolation', () => {
  // a_k = k/n ⇒ f(x) ≡ x by the endpoint-interpolation property.
  const n = 4;
  const coeffs = Array.from({ length: n + 1 }, (_, k) => k / n);
  const ir = call('bernstein',
    { coefficients: vec(...coeffs), x: lit(0.3) });
  const v = sampler.evaluateExpr(ir, {});
  assert.ok(Math.abs(v - 0.3) < 1e-12, 'expected ≈ x = 0.3, got ' + v);
});

// =====================================================================
// stepwise: piecewise constant over edge-defined bins
// =====================================================================

test('stepwise: returns the correct bin value', () => {
  // edges [0, 1, 2, 3], values [10, 20, 30]:
  //   x=0.5 → 10, x=1.5 → 20, x=2.5 → 30
  const ir15 = call('stepwise', {
    edges: vec(0, 1, 2, 3), values: vec(10, 20, 30), x: lit(1.5),
  });
  assert.equal(sampler.evaluateExpr(ir15, {}), 20);
  const ir25 = call('stepwise', {
    edges: vec(0, 1, 2, 3), values: vec(10, 20, 30), x: lit(2.5),
  });
  assert.equal(sampler.evaluateExpr(ir25, {}), 30);
});

test('stepwise: left-closed / right-open semantics for interior bins', () => {
  // x exactly at edges[1] = 1 should land in bin 1 (not bin 0).
  const ir = call('stepwise', {
    edges: vec(0, 1, 2), values: vec(10, 20), x: lit(1),
  });
  assert.equal(sampler.evaluateExpr(ir, {}), 20);
});

test('stepwise: right edge is inclusive for the LAST bin', () => {
  // x exactly at the upper boundary should land in the last bin.
  const ir = call('stepwise', {
    edges: vec(0, 1, 2), values: vec(10, 20), x: lit(2),
  });
  assert.equal(sampler.evaluateExpr(ir, {}), 20);
});

test('stepwise: out-of-range x ⇒ NaN', () => {
  const irLo = call('stepwise', {
    edges: vec(0, 1, 2), values: vec(10, 20), x: lit(-0.5),
  });
  assert.ok(Number.isNaN(sampler.evaluateExpr(irLo, {})));
  const irHi = call('stepwise', {
    edges: vec(0, 1, 2), values: vec(10, 20), x: lit(2.5),
  });
  assert.ok(Number.isNaN(sampler.evaluateExpr(irHi, {})));
});

test('stepwise: edges length must equal values length + 1', () => {
  const irBad = call('stepwise', {
    edges: vec(0, 1, 2),   // 3 edges
    values: vec(10, 20, 30), // 3 values — mismatch
    x: lit(0.5),
  });
  assert.throws(() => sampler.evaluateExpr(irBad, {}),
    /edges length must equal values length/);
});
