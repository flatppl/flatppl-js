'use strict';

// =====================================================================
// density.js — log-density-with-consume/rest primitive.
// =====================================================================
//
// These tests exercise the per-IR-kind dispatch directly via
// hand-built IRs. End-to-end source-to-density coverage is the
// materialiser's job (matLogdensityof tests); here we pin the
// primitive's numeric correctness and its empty-rest invariant.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const density = require('../density');

// Convenience IR constructors — keeps the asserts focused on the
// density math rather than IR plumbing.
function lit(v)        { return { kind: 'lit', value: v }; }
function refSelf(name) { return { kind: 'ref', ns: 'self', name }; }
function Normal(mu, sigma) {
  return { kind: 'call', op: 'Normal',
    kwargs: { mu: lit(mu), sigma: lit(sigma) } };
}
function Exponential(rate) {
  return { kind: 'call', op: 'Exponential', kwargs: { rate: lit(rate) } };
}
function callOp(op, args, fields) {
  const ir = { kind: 'call', op };
  if (args)   ir.args   = args;
  if (fields) ir.fields = fields;
  return ir;
}

const LOG_TWO_PI = Math.log(2 * Math.PI);
const STD_NORMAL_LOGP_AT_ZERO = -0.5 * LOG_TWO_PI;

// =====================================================================
// Scalar leaf — Normal, Exponential
// =====================================================================

test('density: standard Normal at 0 matches -log(sqrt(2π))', () => {
  const logp = density.logDensity(Normal(0, 1), 0, {});
  assert.ok(Math.abs(logp - STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

test('density: Normal(2, 3) at 5 matches stdlib logpdf form', () => {
  // logpdf = -log(σ√(2π)) - (x-μ)²/(2σ²)
  const mu = 2, sigma = 3, x = 5;
  const expected = -Math.log(sigma) - 0.5 * LOG_TWO_PI
    - (x - mu) * (x - mu) / (2 * sigma * sigma);
  const logp = density.logDensity(Normal(mu, sigma), x, {});
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: scalar leaf consumes one entry from a vector head', () => {
  const r = density.logDensityConsume(Normal(0, 1), [0.0, 1.0, 2.0], {});
  assert.equal(r.rest && r.rest.length, 2);
  assert.equal(r.rest[0], 1.0);
});

test('density: scalar leaf with bare number consumes fully', () => {
  const r = density.logDensityConsume(Normal(0, 1), 0.0, {});
  assert.equal(r.rest, null);
});

// =====================================================================
// weighted / logweighted
// =====================================================================

test('density: weighted(0.5, Normal) at 0 adds log(0.5)', () => {
  const ir = callOp('weighted', [lit(0.5), Normal(0, 1)]);
  const logp = density.logDensity(ir, 0, {});
  assert.ok(Math.abs(logp - (STD_NORMAL_LOGP_AT_ZERO + Math.log(0.5))) < 1e-12);
});

test('density: weighted(0, M) → -Infinity', () => {
  const ir = callOp('weighted', [lit(0), Normal(0, 1)]);
  const logp = density.logDensity(ir, 0, {});
  assert.equal(logp, -Infinity);
});

test('density: logweighted(-1, Normal) at 0 adds -1 directly (no log call)', () => {
  const ir = callOp('logweighted', [lit(-1), Normal(0, 1)]);
  const logp = density.logDensity(ir, 0, {});
  assert.ok(Math.abs(logp - (STD_NORMAL_LOGP_AT_ZERO - 1)) < 1e-12);
});

// =====================================================================
// truncate(M, S) — indicator over S
// =====================================================================

test('density: truncate(Normal, posreals) at +0.5 keeps base density', () => {
  const ir = callOp('truncate', [Normal(0, 1), { kind: 'const', name: 'posreals' }]);
  const opts = { parseSet: (setIR) => ({ kind: 'posreals' }) };
  const logp = density.logDensity(ir, 0.5, {}, opts);
  const expected = -0.5 * LOG_TWO_PI - 0.5 * 0.5 / 2;
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: truncate(Normal, posreals) at -0.5 returns -Infinity', () => {
  const ir = callOp('truncate', [Normal(0, 1), { kind: 'const', name: 'posreals' }]);
  const opts = { parseSet: (setIR) => ({ kind: 'posreals' }) };
  const logp = density.logDensity(ir, -0.5, {}, opts);
  assert.equal(logp, -Infinity);
});

// =====================================================================
// record / kwarg-joint
// =====================================================================

test('density: joint(a=N(0,1), b=N(0,1)) at {a:0,b:1} sums field logps', () => {
  const ir = callOp('joint', null, [
    { name: 'a', value: Normal(0, 1) },
    { name: 'b', value: Normal(0, 1) },
  ]);
  const logp = density.logDensity(ir, { a: 0, b: 1 }, {});
  const expected = STD_NORMAL_LOGP_AT_ZERO + (-0.5 * LOG_TWO_PI - 0.5);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: kwarg-joint with missing field throws', () => {
  // Consuming 'a' empties the record; the next iteration can't find
  // 'b' because there's nothing left to consume from. Either error
  // shape is a clear shape-mismatch signal.
  const ir = callOp('joint', null, [
    { name: 'a', value: Normal(0, 1) },
    { name: 'b', value: Normal(0, 1) },
  ]);
  assert.throws(() => density.logDensity(ir, { a: 0 }, {}),
    /missing field 'b'|non-record value|exhausted/);
});

test('density: kwarg-joint with extra field surfaces as leftover rest', () => {
  const ir = callOp('joint', null, [
    { name: 'a', value: Normal(0, 1) },
  ]);
  assert.throws(() => density.logDensity(ir, { a: 0, extra: 99 }, {}),
    /unconsumed leftover/);
});

// =====================================================================
// Positional joint — consume in declared order
// =====================================================================

test('density: positional joint(N, N) at [0, 1] = sum of components', () => {
  const ir = callOp('joint', [Normal(0, 1), Normal(0, 1)]);
  const logp = density.logDensity(ir, [0, 1], {});
  const expected = STD_NORMAL_LOGP_AT_ZERO + (-0.5 * LOG_TWO_PI - 0.5);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: positional joint with mixed leaves consumes correctly', () => {
  // joint(Normal, Exponential) at [0.0, 2.0]
  const ir = callOp('joint', [Normal(0, 1), Exponential(1)]);
  const logp = density.logDensity(ir, [0.0, 2.0], {});
  // logpdf_Normal(0;0,1) + logpdf_Exp(2;1) = -log√(2π) + (log 1 − 2)
  const expected = STD_NORMAL_LOGP_AT_ZERO + (0 - 2);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: positional joint with leftover vector entries throws', () => {
  const ir = callOp('joint', [Normal(0, 1), Normal(0, 1)]);
  assert.throws(() => density.logDensity(ir, [0, 1, 999], {}),
    /unconsumed leftover/);
});

// =====================================================================
// iid(M, n) — n copies of M's footprint
// =====================================================================

test('density: iid(Normal, 3) at [0, 0, 0] = 3 × logp(0)', () => {
  const ir = callOp('iid', [Normal(0, 1), lit(3)]);
  const logp = density.logDensity(ir, [0, 0, 0], {});
  assert.ok(Math.abs(logp - 3 * STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

test('density: iid count mismatch surfaces as leftover', () => {
  const ir = callOp('iid', [Normal(0, 1), lit(3)]);
  assert.throws(() => density.logDensity(ir, [0, 0], {}),
    /exhausted/);
  assert.throws(() => density.logDensity(ir, [0, 0, 0, 0], {}),
    /unconsumed leftover/);
});

// =====================================================================
// Composition — iid + joint, weighted + iid
// =====================================================================

test('density: joint(iid(N, 2), N) at [0, 0, 1] consumes 2 then 1', () => {
  const ir = callOp('joint', [
    callOp('iid', [Normal(0, 1), lit(2)]),
    Normal(0, 1),
  ]);
  const logp = density.logDensity(ir, [0, 0, 1], {});
  const expected = 2 * STD_NORMAL_LOGP_AT_ZERO
    + (-0.5 * LOG_TWO_PI - 0.5);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

test('density: weighted(c, iid(N, n)) propagates log(c) once', () => {
  const ir = callOp('weighted',
    [lit(0.25), callOp('iid', [Normal(0, 1), lit(2)])]);
  const logp = density.logDensity(ir, [0, 0], {});
  const expected = 2 * STD_NORMAL_LOGP_AT_ZERO + Math.log(0.25);
  assert.ok(Math.abs(logp - expected) < 1e-12);
});

// =====================================================================
// Measure refs — resolveMeasureRef callback
// =====================================================================

test('density: measure ref dispatch via resolveMeasureRef opt', () => {
  const ir = callOp('joint', [refSelf('Mref'), Normal(0, 1)]);
  const opts = {
    resolveMeasureRef: (name) => name === 'Mref' ? Normal(0, 1) : null,
  };
  const logp = density.logDensity(ir, [0, 0], {}, opts);
  assert.ok(Math.abs(logp - 2 * STD_NORMAL_LOGP_AT_ZERO) < 1e-12);
});

test('density: missing resolveMeasureRef opt with ref throws clearly', () => {
  const ir = callOp('joint', [refSelf('Mref'), Normal(0, 1)]);
  assert.throws(() => density.logDensity(ir, [0, 0], {}),
    /without resolveMeasureRef/);
});
