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

// =====================================================================
// bincounts: count data points falling into bins
// =====================================================================

test('bincounts: simple 4-bin example', () => {
  // edges [0, 2.5, 5, 7.5, 10] → 4 bins.
  // data [1.0, 3.0, 4.9, 5.0, 9.9] →
  //   bin 0 [0, 2.5): 1.0 ⇒ 1
  //   bin 1 [2.5, 5.0): 3.0, 4.9 ⇒ 2
  //   bin 2 [5.0, 7.5): 5.0 ⇒ 1
  //   bin 3 [7.5, 10.0]: 9.9 ⇒ 1
  const ir = call('bincounts', {
    bins: vec(0, 2.5, 5, 7.5, 10),
    data: vec(1.0, 3.0, 4.9, 5.0, 9.9),
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [1, 2, 1, 1]);
});

test('bincounts: last-bin upper boundary is inclusive', () => {
  // x exactly at the right edge of the last bin falls into it.
  const ir = call('bincounts', {
    bins: vec(0, 1, 2),
    data: vec(2.0),  // last edge
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [0, 1]);
});

test('bincounts: interior bin upper boundary is exclusive (next bin owns it)', () => {
  // x exactly at the right edge of an INTERIOR bin should fall into
  // the next bin, not this one.
  const ir = call('bincounts', {
    bins: vec(0, 1, 2),
    data: vec(1.0),  // interior edge
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [0, 1]);
});

test('bincounts: out-of-range data is ignored', () => {
  const ir = call('bincounts', {
    bins: vec(0, 1, 2),
    data: vec(-1.0, 0.5, 1.5, 3.0),  // -1 and 3 fall outside
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [1, 1]);
});

test('bincounts: empty data ⇒ all zeros', () => {
  const ir = call('bincounts', {
    bins: vec(0, 1, 2, 3),
    data: vec(),
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [0, 0, 0]);
});

test('bincounts: rejects multi-dimensional binning (record bins)', () => {
  // First-arg bins as a record (multi-D) — defer per spec §07.
  // Constructed by hand here since the lower-level evaluator
  // doesn't have a record-of-vectors literal at this test layer.
  const ir = call('bincounts', {
    bins: { kind: 'call', op: 'record', fields: [
      { name: 'a', value: vec(0, 1) },
      { name: 'b', value: vec(0, 1) },
    ]},
    data: vec(0.5),
  });
  assert.throws(() => sampler.evaluateExpr(ir, {}),
    /multi-dimensional binning/);
});

// =====================================================================
// selectbins: keep counts for bins whose interval intersects region
// =====================================================================

function interval(lo, hi) {
  return { kind: 'call', op: 'interval', args: [lit(lo), lit(hi)] };
}
function constSet(name) { return { kind: 'const', name }; }

test('selectbins: interior region keeps only intersecting bins', () => {
  // edges [0, 1, 2, 3, 4] → bins [0,1] [1,2] [2,3] [3,4]
  // counts [10, 20, 30, 40]
  // region [1.5, 3.5] intersects bins 1, 2, 3 (in 1-based) → keeps [20, 30, 40]
  // (bin 0 [0,1] doesn't intersect [1.5, 3.5])
  const ir = call('selectbins', {
    edges: vec(0, 1, 2, 3, 4),
    region: interval(1.5, 3.5),
    counts: vec(10, 20, 30, 40),
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [20, 30, 40]);
});

test('selectbins: region covering all bins keeps everything', () => {
  const ir = call('selectbins', {
    edges: vec(0, 1, 2, 3),
    region: interval(-10, 10),
    counts: vec(5, 6, 7),
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [5, 6, 7]);
});

test('selectbins: region disjoint from all bins ⇒ empty result', () => {
  const ir = call('selectbins', {
    edges: vec(0, 1, 2),
    region: interval(10, 20),
    counts: vec(1, 2),
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), []);
});

test('selectbins: bin grazing region boundary counts as intersecting', () => {
  // Bin [1, 2] touches region [2, 3] at exactly 2 → bin is kept
  // per the inclusive ≤/≥ semantics.
  const ir = call('selectbins', {
    edges: vec(0, 1, 2, 3),
    region: interval(2, 3),
    counts: vec(10, 20, 30),
  });
  // Bins [0,1] [1,2] [2,3] vs region [2,3]: bin 0 misses, bins 1, 2 touch.
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [20, 30]);
});

test('selectbins: posreals region drops only bins entirely on the negative side', () => {
  // edges [-2, -1, 0, 1, 2] → bins [-2,-1] [-1,0] [0,1] [1,2]
  // region posreals ≡ [0, ∞]
  // bin [-2,-1] strictly negative → drop
  // bin [-1, 0] right edge at 0 → touches region → keep
  // bin [0, 1] → keep
  // bin [1, 2] → keep
  const ir = call('selectbins', {
    edges: vec(-2, -1, 0, 1, 2),
    region: constSet('posreals'),
    counts: vec(1, 2, 3, 4),
  });
  assert.deepEqual(sampler.evaluateExpr(ir, {}), [2, 3, 4]);
});

test('selectbins: edges length must equal counts length + 1', () => {
  const irBad = call('selectbins', {
    edges: vec(0, 1, 2),   // 3 edges
    region: interval(0, 5),
    counts: vec(1, 2, 3),  // 3 counts — mismatch (expected 2)
  });
  assert.throws(() => sampler.evaluateExpr(irBad, {}),
    /edges length must equal counts length/);
});

test('selectbins: unsupported region shape surfaces a clear error', () => {
  const irBad = call('selectbins', {
    edges: vec(0, 1, 2),
    region: { kind: 'lit', value: 42 },   // not a set IR
    counts: vec(1, 2),
  });
  assert.throws(() => sampler.evaluateExpr(irBad, {}),
    /unsupported region shape/);
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
