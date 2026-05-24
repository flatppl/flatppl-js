'use strict';

// Spec §06: iid(M, size) — `size` is a positive integer (1-D length)
// or a vector of positive integers (multi-axis shape). Spec example:
// `iid(Normal(mu = 0, sigma = 1), [3, 3])` produces 3×3 matrices.
//
// Three engine entry points handle `iid` size:
//   1. classifyIid  (derivations) — unpacks `vector(...)` → dims[]
//   2. inferIid     (typeinfer)   — unpacks `vector(...)` → shape
//   3. matIid       (materialiser)— takes dims[] from the derivation
//   4. walkIid      (traceeval)   — must compute prod(size) draws
//                                    and reshape into nested array
//   5. walkIid      (density)     — must compute prod(size) for the
//                                    reduce loop over the footprint
//
// (1)-(3) already handle the vector form. (4) and (5) were
// scalar-only until fixed alongside this test file.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const sampler = require('../sampler.ts');
const traceeval = require('../traceeval.ts');
const rng = require('../rng.ts');
const { processSource } = require('../index.ts');

function makeIid(sizeIR: any) {
  return {
    kind: 'call', op: 'iid',
    args: [
      { kind: 'call', op: 'Normal',
        kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } } },
      sizeIR,
    ],
  };
}
function vec(...vs: number[]) {
  return { kind: 'call', op: 'vector',
    args: vs.map((v: number) => ({ kind: 'lit', value: v })) };
}
function lit(v: number) { return { kind: 'lit', value: v }; }

// ---------------------------------------------------------------------
// traceeval.walk (used by `rand`) — multi-axis shape
// ---------------------------------------------------------------------

test('iid multi-axis: traceeval walk produces a nested array of the requested shape', () => {
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = traceeval.walk(state, makeIid(vec(3, 3)), {}, {});
  assert.ok(Array.isArray(r.value), 'value is an array');
  assert.equal(r.value.length, 3, 'outer length 3');
  for (let i = 0; i < 3; i++) {
    assert.ok(Array.isArray(r.value[i]) || (r.value[i] && r.value[i].length === 3),
      `row ${i} is a length-3 inner sequence`);
    for (let j = 0; j < 3; j++) {
      assert.equal(typeof r.value[i][j], 'number', `entry [${i}][${j}] is a number`);
    }
  }
});

test('iid multi-axis: rank-3 shape produces 3D nested array', () => {
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = traceeval.walk(state, makeIid(vec(2, 3, 4)), {}, {});
  assert.equal(r.value.length, 2);
  assert.equal(r.value[0].length, 3);
  assert.equal(r.value[0][0].length, 4);
});

test('iid multi-axis: scalar size still produces a flat array (back-compat)', () => {
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = traceeval.walk(state, makeIid(lit(5)), {}, {});
  assert.equal(r.value.length, 5);
  assert.equal(typeof r.value[0], 'number',
    'scalar-size form remains flat (not wrapped in an extra dim)');
});

test('iid multi-axis: zero-prod size produces an empty result', () => {
  const state = rng.stateFromKey([1, 2, 3, 4, 5, 6, 7, 8]);
  const r = traceeval.walk(state, makeIid(vec(0, 3)), {}, {});
  // prod = 0 → no draws. The reshape walks the outer 0 first and
  // returns []; deeper structure isn't materialised.
  assert.equal(r.value.length, 0);
});

test('iid multi-axis: end-to-end `rand(state, iid(N, [3, 3]))` parses + analyses', () => {
  const src = `
rstate = rnginit([1,2,3,4])
A, _ = rand(rstate, iid(Normal(0,1), [3, 3]))
`;
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, 'parses cleanly: ' +
    errs.map((d: any) => d.message).join('; '));
});

// ---------------------------------------------------------------------
// density.walk — multi-axis size in iid logdensity
// ---------------------------------------------------------------------

test('iid multi-axis: density(iid(N, [3, 3]), x) reduces over prod(size) cells', () => {
  // Sum-of-logpdf over 9 standard-normal cells, all zero. The
  // consume/rest density walker operates on the flattened
  // footprint, so the value here is a 9-element array — what the
  // walker would receive after the surrounding measure-handler
  // (matLogdensityof / matBayesupdate / etc.) flattens the
  // user-facing nested input.
  const density = require('../density.ts');
  const ir = makeIid(vec(3, 3));
  const x = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  // 9 × log(1/sqrt(2π)) ≈ 9 × -0.9189385332
  const expected = 9 * Math.log(1 / Math.sqrt(2 * Math.PI));
  const got = density.logDensity(ir, x, {}, {});
  assert.ok(Math.abs(got - expected) < 1e-10,
    `expected ~${expected}, got ${got}`);
});

// ---------------------------------------------------------------------
// inferIid — multi-axis shape is preserved in the inferred type
// ---------------------------------------------------------------------

test('iid multi-axis: inferred type carries the full rank + per-axis lengths', () => {
  const ctx = processSource('M = iid(Normal(0,1), [3, 3])\n');
  const m = ctx.loweredModule.bindings.get('M');
  assert.ok(m && m.inferredType, 'inferred type present');
  const t = m.inferredType;
  // measure(array(2, [3, 3], real))
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array');
  assert.equal(t.domain.rank, 2);
  assert.equal(t.domain.shape[0], 3);
  assert.equal(t.domain.shape[1], 3);
});

// ---------------------------------------------------------------------
// cartpow — sister rule from spec §03 sets
// ---------------------------------------------------------------------

test('cartpow multi-axis: cartpow(reals, [3, 3]) type-infers as array(2, [3,3], real)', () => {
  // Use it via Lebesgue(support = cartpow(...)) so the engine flows
  // the set's value type through to a measure binding.
  const ctx = processSource(
    'M = Lebesgue(support = cartpow(reals, [3, 3]))\n');
  const m = ctx.loweredModule.bindings.get('M');
  assert.ok(m && m.inferredType);
  const t = m.inferredType;
  assert.equal(t.kind, 'measure', 'measure type');
  assert.equal(t.domain.kind, 'array');
  assert.equal(t.domain.rank, 2);
  assert.equal(t.domain.shape[0], 3);
  assert.equal(t.domain.shape[1], 3);
});

test('cartpow scalar still infers as rank-1 with that length', () => {
  const ctx = processSource(
    'M = Lebesgue(support = cartpow(reals, 5))\n');
  const m = ctx.loweredModule.bindings.get('M');
  const t = m.inferredType;
  assert.equal(t.domain.rank, 1);
  assert.equal(t.domain.shape[0], 5);
});
