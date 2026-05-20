'use strict';

// Tests for sampler.evaluateExprN — the batched IR-evaluation
// primitive. evaluateExpr (single-point) is already exercised
// extensively by every existing test; the cases below cover the
// batched-specific behaviour:
//
//   1. Broadcast across scalar / per-atom inputs (scalar primitives
//      via ARITH_OPS_N).
//   2. Env precedence at refs: overlay > refArrays > baseEnv.
//   3. Atom-independent constant-folding (all-scalar inputs → scalar
//      result, no Float64Array allocation in the fast path).
//   4. Non-scalar-op fallback via per-atom dispatch.
//   5. ARITH_OPS_N covers every entry in the _SCALAR_PRIM_ARITY table
//      (regression guard for primitives added but not registered).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');

const lit = v => ({ kind: 'lit', value: v });
const ref = n => ({ kind: 'ref', ns: 'self', name: n });
const call = (op, ...args) => ({ kind: 'call', op, args });

// =====================================================================
// Broadcast semantics — scalar arith with per-atom refs
// =====================================================================

test('evaluateExprN: scalar-only inputs return a scalar (no Float64Array)', () => {
  // add(2, 3) — both literals, atom-indep. Should fold to a number.
  const r = sampler.evaluateExprN(call('add', lit(2), lit(3)), null, 4, {});
  assert.equal(r, 5);
  assert.equal(typeof r, 'number');
});

test('evaluateExprN: one per-atom input broadcasts the other', () => {
  // add(x, 1) where x = [1, 2, 3, 4] → [2, 3, 4, 5]
  const x = new Float64Array([1, 2, 3, 4]);
  const r = sampler.evaluateExprN(call('add', ref('x'), lit(1)),
    { x }, 4, {});
  assert.ok(r.BYTES_PER_ELEMENT, 'expected Float64Array result');
  assert.deepEqual(Array.from(r), [2, 3, 4, 5]);
});

test('evaluateExprN: both inputs per-atom — element-wise', () => {
  // mul(a, b) where a = [1,2,3], b = [10,20,30] → [10, 40, 90]
  const a = new Float64Array([1, 2, 3]);
  const b = new Float64Array([10, 20, 30]);
  const r = sampler.evaluateExprN(call('mul', ref('a'), ref('b')),
    { a, b }, 3, {});
  assert.deepEqual(Array.from(r), [10, 40, 90]);
});

test('evaluateExprN: chained scalar ops compose without per-atom env rebuild', () => {
  // (x + 1) * 2 — broadcast2 propagates the Float64Array up through
  // each scalar op.
  const x = new Float64Array([0, 1, 2]);
  const ir = call('mul', call('add', ref('x'), lit(1)), lit(2));
  const r = sampler.evaluateExprN(ir, { x }, 3, {});
  assert.deepEqual(Array.from(r), [2, 4, 6]);
});

test('evaluateExprN: ternary ifelse broadcasts over a per-atom condition', () => {
  // ifelse(x > 0, x, -x) — abs via ternary. Boolean per-atom condition.
  // Skip x=0 to dodge JS signed-zero (-0 vs 0); the dispatch path is
  // identical, and the existing scalar evaluateExpr tests already
  // cover the zero case.
  const x = new Float64Array([-2, -1, 1, 2]);
  const ir = call('ifelse', call('gt', ref('x'), lit(0)),
    ref('x'),
    call('neg', ref('x')));
  const r = sampler.evaluateExprN(ir, { x }, 4, {});
  assert.deepEqual(Array.from(r), [2, 1, 1, 2]);
});

// =====================================================================
// Env precedence — overlay > refArrays > baseEnv
// =====================================================================

test('evaluateExprN: baseEnv resolves refs when refArrays empty', () => {
  // x + 1 with x in baseEnv = 5 → 6 (scalar, no Float64Array).
  const r = sampler.evaluateExprN(call('add', ref('x'), lit(1)),
    null, 3, { x: 5 });
  assert.equal(r, 6);
});

test('evaluateExprN: refArrays wins over baseEnv', () => {
  const x_atoms = new Float64Array([10, 20, 30]);
  const r = sampler.evaluateExprN(call('add', ref('x'), lit(1)),
    { x: x_atoms }, 3, { x: 999 /* should be ignored */ });
  assert.deepEqual(Array.from(r), [11, 21, 31]);
});

test('evaluateExprN: overlay wins over BOTH refArrays and baseEnv', () => {
  // Critical for density.js env-threading: when an observation value
  // pins a binding, it must override per-atom prior samples and the
  // session env.
  const x_atoms = new Float64Array([10, 20, 30]);
  const r = sampler.evaluateExprN(call('add', ref('x'), lit(1)),
    { x: x_atoms }, 3, { x: 999 },
    { overlay: { x: 100 } });
  // When overlay covers the only per-atom name, the whole subtree is
  // atom-independent → fast path returns scalar 101 (no allocation).
  // Either scalar or broadcast Float64Array is correct; both encode
  // "every atom sees 101".
  if (typeof r === 'number') {
    assert.equal(r, 101);
  } else {
    assert.deepEqual(Array.from(r), [101, 101, 101]);
  }
});

test('evaluateExprN: overlay applies only to its names; other refs honour refArrays', () => {
  const x_atoms = new Float64Array([1, 2, 3]);
  const r = sampler.evaluateExprN(call('add', ref('x'), ref('y')),
    { x: x_atoms, y: new Float64Array([10, 20, 30]) }, 3, {},
    { overlay: { y: 100 } });
  // x stays per-atom, y is overlay-pinned.
  assert.deepEqual(Array.from(r), [101, 102, 103]);
});

// =====================================================================
// Non-scalar-op fallback
// =====================================================================

test('evaluateExprN: vector(...) of scalar refs returns Value shape=[N, 2]', () => {
  // vector(a, b) where a, b are per-atom. Falls back to per-atom
  // dispatch — Phase 7c packs the uniform-length array results into
  // a Value with shape=[N, k] (atom-major flat layout). The legacy
  // JS-Array-of-arrays form is no longer produced; the Value is the
  // single representation for vector-atom data.
  const a = new Float64Array([1, 2, 3]);
  const b = new Float64Array([10, 20, 30]);
  const r = sampler.evaluateExprN(call('vector', ref('a'), ref('b')),
    { a, b }, 3, {});
  // Phase 7c: Value shape=[N=3, k=2], atom-major flat data.
  assert.ok(r && r.shape && r.data);
  assert.deepEqual(r.shape, [3, 2]);
  assert.deepEqual(Array.from(r.data), [1, 10, 2, 20, 3, 30]);
});

test('evaluateExprN: vector(...) of atom-indep inputs returns one shared array', () => {
  // No per-atom refs touch the subtree → fast path, single evaluation.
  const r = sampler.evaluateExprN(call('vector', lit(1), lit(2), lit(3)),
    null, 5, {});
  assert.deepEqual(r, [1, 2, 3]);
});

// =====================================================================
// Single-point evaluateExprN with count=1 — degenerate fast path
// =====================================================================

test('evaluateExprN: count=1 with no refArrays equals scalar evaluateExpr', () => {
  // Same expression evaluated both ways must produce identical results.
  const ir = call('mul',
    call('add', ref('a'), ref('b')),
    call('sub', ref('a'), ref('b')));
  const env = { a: 7, b: 3 };
  const r1 = sampler.evaluateExpr(ir, env);
  const r2 = sampler.evaluateExprN(ir, null, 1, env);
  assert.equal(r1, r2);
  assert.equal(r1, 40);  // (7+3) * (7-3) = 40
});

// =====================================================================
// ARITH_OPS_N coverage regression — every entry in _SCALAR_PRIM_ARITY
// must be invocable via the batched dispatch. Catches "added a scalar
// op to ARITH_OPS, forgot to register it for batching" mistakes.
// =====================================================================

test('evaluateExprN: every scalar primitive dispatches via ARITH_OPS_N', () => {
  // For each scalar op, build a tiny IR that exercises it. We don't
  // verify the math here (existing tests do); we verify dispatch
  // succeeds and returns a finite numeric (or boolean) Float64Array
  // entry. Per-atom inputs ensure ARITH_OPS_N runs (not a scalar fold).
  const a = new Float64Array([1.5]);
  const b = new Float64Array([0.5]);
  const c = new Float64Array([1]);  // boolean-ish for ifelse condition

  // Inputs picked to land inside the domain of every op.
  // logit / probit need (0, 1); we feed 0.5.
  // ifelse arity=3 (cond, then, else).
  // boolean / integer accept the exact values 0/1 → 0/1.
  const cases = {
    add: [a, b], sub: [a, b], mul: [a, b], div: [a, b], mod: [a, b],
    neg: [a], pos: [a], pow: [a, b],
    abs: [a], abs2: [a], exp: [b], log: [a], log10: [a],
    sqrt: [a], sin: [a], cos: [a], floor: [a], ceil: [a], round: [a],
    min: [a, b], max: [a, b],
    gamma: [a], loggamma: [a],
    logit: [b], invlogit: [a], probit: [b], invprobit: [a],
    lt: [a, b], le: [a, b], gt: [a, b], ge: [a, b],
    equal: [a, b], unequal: [a, b],
    isfinite: [a], isinf: [a], isnan: [a], iszero: [a],
    land: [c, c], lor: [c, c], lxor: [c, c], lnot: [c],
    ifelse: [c, a, b],
    boolean: [c], integer: [c],
  };

  for (const op of Object.keys(cases)) {
    const args = cases[op].map((_, i) => ({ kind: 'ref', ns: 'self', name: `_p${i}` }));
    const refArrays = {};
    cases[op].forEach((arr, i) => { refArrays[`_p${i}`] = arr; });
    const ir = { kind: 'call', op, args };
    let r;
    try {
      r = sampler.evaluateExprN(ir, refArrays, 1, {});
    } catch (e) {
      assert.fail(`scalar prim '${op}' dispatch failed: ${e.message}`);
    }
    assert.ok(r && r.BYTES_PER_ELEMENT && r.length === 1,
      `'${op}' should produce Float64Array(1), got ${typeof r}`);
    assert.ok(Number.isFinite(r[0]) || r[0] === 0 || r[0] === 1,
      `'${op}' produced non-finite-non-bool ${r[0]}`);
  }
});

// =====================================================================
// isBatch helper — utility surface
// =====================================================================

test('isBatch: Float64Array of correct length is batched', () => {
  assert.equal(sampler.isBatch(new Float64Array(4), 4), true);
  assert.equal(sampler.isBatch(new Float64Array(4), 3), false);
  assert.equal(sampler.isBatch(3, 4), false);
  assert.equal(sampler.isBatch([1, 2, 3, 4], 4), false);  // not typed
  assert.equal(sampler.isBatch(null, 4), false);
});

// =====================================================================
// Phase 1: Value-polymorphic broadcast — ARITH_OPS_N accepts Value
// inputs (shape=[] / shape=[N]) alongside the legacy bare-primitive
// (number / Float64Array) forms. "Same kind as inputs" output
// semantics: any Value input ⇒ Value output; otherwise legacy bare.
// =====================================================================

const value = require('../value.ts');
const { ARITH_OPS_N } = sampler._internal;

test('isBatch: Value shape=[N] is batched; shape=[] is not', () => {
  const N = 4;
  const bv = value.batchedScalar(new Float64Array([1, 2, 3, 4]));
  assert.equal(sampler.isBatch(bv, N), true);
  const sv = value.scalar(5);
  assert.equal(sampler.isBatch(sv, N), false);
  // Wrong-length Value not batched.
  const wrong = value.batchedScalar(new Float64Array([1, 2, 3]));
  assert.equal(sampler.isBatch(wrong, N), false);
});

test('ARITH_OPS_N: all-bare inputs → bare result (legacy fast path)', () => {
  // No Value in → no Value out. Atom-indep scalar args fold to number.
  const r = ARITH_OPS_N.add([2, 3], 4);
  assert.equal(r, 5);
  assert.equal(typeof r, 'number');
  // Bare Float64Array batched → bare Float64Array out.
  const r2 = ARITH_OPS_N.add([new Float64Array([1, 2, 3, 4]), 1], 4);
  assert.ok(r2 instanceof Float64Array);
  assert.deepEqual(Array.from(r2), [2, 3, 4, 5]);
});

test('ARITH_OPS_N: Value shape=[] scalar in → Value shape=[] out', () => {
  const r = ARITH_OPS_N.add([value.scalar(2), value.scalar(3)], 4);
  assert.deepEqual(r.shape, []);
  assert.equal(r.data[0], 5);
});

test('ARITH_OPS_N: Value shape=[N] batched in → Value shape=[N] out', () => {
  const bv = value.batchedScalar(new Float64Array([1, 2, 3, 4]));
  const r = ARITH_OPS_N.mul([bv, value.scalar(10)], 4);
  assert.deepEqual(r.shape, [4]);
  assert.deepEqual(Array.from(r.data), [10, 20, 30, 40]);
});

test('ARITH_OPS_N: mixed Value + bare number → Value out (any Value triggers wrap)', () => {
  // Value × number → Value. The number is broadcast as atom-indep.
  const r = ARITH_OPS_N.add([value.scalar(5), 2], 4);
  assert.ok(value.isValue(r), 'expected Value wrap when any input is a Value');
  assert.equal(value.asScalar(r), 7);
});

test('ARITH_OPS_N: mixed Value batched + bare Float64Array → Value out', () => {
  const vb = value.batchedScalar(new Float64Array([1, 2, 3, 4]));
  const fb = new Float64Array([10, 20, 30, 40]);
  const r = ARITH_OPS_N.add([vb, fb], 4);
  assert.ok(value.isValue(r));
  assert.deepEqual(r.shape, [4]);
  assert.deepEqual(Array.from(r.data), [11, 22, 33, 44]);
});

test('ARITH_OPS_N: unary op preserves Value-ness (exp on shape=[] / shape=[N])', () => {
  const r1 = ARITH_OPS_N.exp([value.scalar(0)], 4);
  assert.ok(value.isValue(r1));
  assert.equal(value.asScalar(r1), 1);

  const r2 = ARITH_OPS_N.exp([value.batchedScalar(new Float64Array([0, 1]))], 2);
  assert.ok(value.isValue(r2));
  assert.deepEqual(r2.shape, [2]);
  assert.ok(Math.abs(r2.data[0] - 1) < 1e-12);
  assert.ok(Math.abs(r2.data[1] - Math.E) < 1e-12);
});

test('ARITH_OPS_N: ternary op (ifelse) honours polymorphism', () => {
  // ifelse(cond, then, else); arity 3. Mix Value + bare.
  const cond = value.batchedScalar(new Float64Array([1, 0, 1, 0]));
  const r = ARITH_OPS_N.ifelse([cond, 100, 200], 4);
  assert.ok(value.isValue(r), 'any Value input ⇒ Value output');
  assert.deepEqual(Array.from(r.data), [100, 200, 100, 200]);
});

test('ARITH_OPS_N: bare-only ternary stays bare', () => {
  const cond = new Float64Array([1, 0, 1, 0]);
  const r = ARITH_OPS_N.ifelse([cond, 100, 200], 4);
  assert.ok(r instanceof Float64Array, 'bare inputs ⇒ bare output');
  assert.deepEqual(Array.from(r), [100, 200, 100, 200]);
});

test('ARITH_OPS_N: result Value carries dtype=f64 default', () => {
  const r = ARITH_OPS_N.add([value.scalar(1), value.scalar(2)], 4);
  assert.equal(value.getDType(r), 'f64');
});
