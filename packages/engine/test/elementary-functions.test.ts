'use strict';

// Spec §07 Elementary functions: gamma family, link functions, binary
// min/max. All scalar→scalar, dispatched through sampler.ARITH_OPS.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');

function lit(v: any)        { return { kind: 'lit', value: v }; }
function call(op: any, ...args: any[]) { return { kind: 'call', op, args }; }
const ev = (ir: any) => sampler.evaluateExpr(ir, {});

// =====================================================================
// gamma / loggamma
// =====================================================================

test('gamma: gamma(n) = (n-1)! for small positive integers', () => {
  assert.equal(ev(call('gamma', lit(1))), 1);
  assert.equal(ev(call('gamma', lit(2))), 1);
  assert.equal(ev(call('gamma', lit(3))), 2);
  assert.equal(ev(call('gamma', lit(4))), 6);
  assert.equal(ev(call('gamma', lit(5))), 24);
});

test('gamma: gamma(0.5) = sqrt(pi)', () => {
  const v = ev(call('gamma', lit(0.5)));
  assert.ok(Math.abs(v - Math.sqrt(Math.PI)) < 1e-12);
});

test('loggamma: loggamma(n) = log((n-1)!) for small n', () => {
  assert.equal(ev(call('loggamma', lit(1))), 0);
  assert.equal(ev(call('loggamma', lit(2))), 0);
  // gamma(5) = 24, log(24) ≈ 3.178
  assert.ok(Math.abs(ev(call('loggamma', lit(5))) - Math.log(24)) < 1e-12);
});

// =====================================================================
// logit / invlogit — symmetric inverse-pair
// =====================================================================

test('logit: logit(0.5) = 0; logit at extremes is signed infinity', () => {
  assert.equal(ev(call('logit', lit(0.5))), 0);
  assert.equal(ev(call('logit', lit(0))), -Infinity);
  assert.equal(ev(call('logit', lit(1))), Infinity);
});

test('invlogit: invlogit(0) = 0.5; saturates to 0 and 1 at ±∞', () => {
  assert.equal(ev(call('invlogit', lit(0))), 0.5);
  assert.ok(Math.abs(ev(call('invlogit', lit(10))) - 1) < 1e-4);
  assert.ok(Math.abs(ev(call('invlogit', lit(-10))) - 0) < 1e-4);
});

test('invlogit ∘ logit = identity on (0, 1)', () => {
  for (const p of [0.05, 0.25, 0.5, 0.75, 0.95]) {
    const round = ev(call('invlogit', call('logit', lit(p))));
    assert.ok(Math.abs(round - p) < 1e-12, 'roundtrip at p=' + p);
  }
});

// =====================================================================
// probit / invprobit — standard-normal quantile/CDF, symmetric inverse-pair
// =====================================================================

test('probit: probit(0.5) = 0', () => {
  assert.ok(Math.abs(ev(call('probit', lit(0.5)))) < 1e-12);
});

test('invprobit: invprobit(0) = 0.5; standard-normal CDF', () => {
  assert.ok(Math.abs(ev(call('invprobit', lit(0))) - 0.5) < 1e-12);
});

test('invprobit(1.96) ≈ 0.975 (standard-normal upper-tail landmark)', () => {
  const v = ev(call('invprobit', lit(1.96)));
  assert.ok(Math.abs(v - 0.975) < 1e-3);
});

test('invprobit ∘ probit = identity on (0, 1)', () => {
  for (const p of [0.05, 0.25, 0.5, 0.75, 0.95]) {
    const round = ev(call('invprobit', call('probit', lit(p))));
    assert.ok(Math.abs(round - p) < 1e-10, 'roundtrip at p=' + p);
  }
});

// =====================================================================
// binary min/max
// =====================================================================

test('min: binary minimum', () => {
  assert.equal(ev(call('min', lit(3), lit(5))), 3);
  assert.equal(ev(call('min', lit(-7), lit(-2))), -7);
});

test('max: binary maximum', () => {
  assert.equal(ev(call('max', lit(3), lit(5))), 5);
  assert.equal(ev(call('max', lit(-7), lit(-2))), -2);
});
