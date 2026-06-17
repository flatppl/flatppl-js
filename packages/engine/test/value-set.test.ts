'use strict';

// =====================================================================
// value-set.test.ts — static value-set inference (spec §11 third %meta
// slot; engine-concepts §17.3 valueset domain)
// =====================================================================
//
// Pins typeinfer's `fillValuesets` (the producer catalogue + the
// natural-extent fallback) and the `value-set.ts` lattice. The expected
// sets are the CROSS-ENGINE oracle: they mirror flatppl-rust's
// `flatppl-infer` golden `valueset_producers_and_simplex_chain`
// (crates/infer/tests/golden.rs), so the two engines agree at the
// `%meta` value-set layer.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const VS = require('../value-set.ts');

// Value set (rendered) of binding `name`'s outermost node.
function vsOf(src: string, name: string): string {
  const r = processSource(src);
  const b = r.loweredModule.bindings.get(name);
  return VS.toSexpr(b && b.rhs && b.rhs.meta && b.rhs.meta.valueset);
}

// =====================================================================
// 1. Producer catalogue (spec §08 supports + §07 normalization fns)
// =====================================================================

test('valueset: distribution supports (§08 Domain/Support column)', () => {
  assert.equal(vsOf('m = Normal(mu = 0.0, sigma = 1.0)', 'm'), 'reals');
  assert.equal(vsOf('m = LogNormal(mu = 0.0, sigma = 1.0)', 'm'), 'posreals');
  assert.equal(vsOf('m = Exponential(rate = 1.0)', 'm'), 'nonnegreals');
  assert.equal(vsOf('m = Beta(alpha = 1.0, beta = 1.0)', 'm'), 'unitinterval');
  assert.equal(vsOf('m = Bernoulli(p = 0.5)', 'm'), 'booleans');
  assert.equal(vsOf('m = Categorical(p = [0.3, 0.7])', 'm'), 'posintegers');
  assert.equal(vsOf('m = Poisson(rate = 3.0)', 'm'), 'nonnegintegers');
  assert.equal(vsOf('m = Dirichlet(alpha = [1.0, 1.0, 1.0])', 'm'), '(stdsimplex 3)');
  assert.equal(
    vsOf('mu = [0.0, 0.0]\ncov = rowstack([[1.0, 0.0], [0.0, 1.0]])\nm = MvNormal(mu = mu, cov = cov)', 'm'),
    '(cartpow reals 2)');
});

test('valueset: elementof / reference-measure supports / draw', () => {
  assert.equal(vsOf('m = elementof(posreals)', 'm'), 'posreals');
  assert.equal(vsOf('m = elementof(interval(0.0, 5.0))', 'm'), '(interval 0.0 5.0)');
  assert.equal(vsOf('m = Lebesgue(support = reals)', 'm'), 'reals');
  assert.equal(vsOf('m = Counting(support = nonnegintegers)', 'm'), 'nonnegintegers');
  // A draw lands in the measure's support (golden: Dirichlet draw → simplex).
  assert.equal(vsOf('x ~ Dirichlet(alpha = [1.0, 1.0, 1.0])', 'x'), '(stdsimplex 3)');
});

test('valueset: §07 normalization / range-constrained functions', () => {
  assert.equal(vsOf('z = softmax([0.0, 1.0])', 'z'), '(stdsimplex 2)');
  // l1unit lands on the simplex only for a provably-nonnegative vector.
  assert.equal(vsOf('w = [3.0, 1.0]\np = l1unit(w)', 'p'), '(stdsimplex 2)');
  assert.equal(vsOf('x = elementof(reals)\ny = exp(x)', 'y'), 'posreals');
  assert.equal(vsOf('x = elementof(reals)\ny = abs(x)', 'y'), 'nonnegreals');
  assert.equal(vsOf('x = elementof(reals)\ny = invlogit(x)', 'y'), 'unitinterval');
});

test('valueset: truncate restricts to the truncation set; reweighting preserves', () => {
  assert.equal(
    vsOf('m = truncate(Normal(mu = 0.0, sigma = 1.0), interval(0.0, 1.0))', 'm'),
    '(interval 0.0 1.0)');
  // weighted/normalize never grow the support.
  assert.equal(vsOf('m = normalize(Beta(alpha = 1.0, beta = 1.0))', 'm'), 'unitinterval');
  assert.equal(vsOf('m = weighted(2.0, Exponential(rate = 1.0))', 'm'), 'nonnegreals');
});

// =====================================================================
// 2. Natural-extent fallback (every value-typed node ≥ its type's extent)
// =====================================================================

test('valueset: natural-extent fallback for non-producer nodes', () => {
  // `add` is not a producer; it falls back to the type's natural extent.
  assert.equal(vsOf('x = elementof(reals)\ny = x + 1.0', 'y'), 'reals');
  assert.equal(vsOf('n = elementof(integers)\nm = n + 1', 'm'), 'integers');
});

// =====================================================================
// 3. The value-set.ts lattice (mirrors Rust ValueSet::{is_bounded,
//    subset_of, natural_of})
// =====================================================================

test('valueset lattice: isBounded', () => {
  assert.equal(VS.isBounded(VS.UNITINTERVAL), true);
  assert.equal(VS.isBounded(VS.BOOLEANS), true);
  assert.equal(VS.isBounded(VS.stdsimplex(3)), true);
  assert.equal(VS.isBounded(VS.interval(0, 1)), true);
  assert.equal(VS.isBounded(VS.interval(0, Infinity)), false);
  assert.equal(VS.isBounded(VS.REALS), false);
  assert.equal(VS.isBounded(VS.INTEGERS), false);
  assert.equal(VS.isBounded(VS.cartpow(VS.UNITINTERVAL, 3)), true);
  assert.equal(VS.isBounded(VS.cartpow(VS.REALS, 3)), false);
  assert.equal(VS.isBounded(VS.cartpow(VS.UNITINTERVAL, '%dynamic')), null);
  assert.equal(VS.isBounded(VS.UNKNOWN), null);
});

test('valueset lattice: subsetOf (conservative)', () => {
  assert.equal(VS.subsetOf(VS.POSREALS, VS.REALS), true);
  assert.equal(VS.subsetOf(VS.UNITINTERVAL, VS.NONNEGREALS), true);
  assert.equal(VS.subsetOf(VS.POSINTEGERS, VS.INTEGERS), true);
  assert.equal(VS.subsetOf(VS.interval(0, 1), VS.NONNEGREALS), true);
  assert.equal(VS.subsetOf(VS.interval(-1, 1), VS.NONNEGREALS), false);
  assert.equal(VS.subsetOf(VS.stdsimplex(3), VS.cartpow(VS.UNITINTERVAL, 3)), true);
  assert.equal(VS.subsetOf(VS.stdsimplex(3), VS.cartpow(VS.NONNEGREALS, '%dynamic')), true);
  // Unproven (not disproven) → false.
  assert.equal(VS.subsetOf(VS.REALS, VS.POSREALS), false);
  assert.equal(VS.subsetOf(VS.UNKNOWN, VS.REALS), false);
});

test('valueset lattice: naturalOf', () => {
  const T = require('../types.ts');
  assert.equal(VS.naturalOf(T.REAL), 'reals');
  assert.equal(VS.naturalOf(T.INTEGER), 'integers');
  assert.deepEqual(VS.naturalOf(T.array(1, [3], T.REAL)), VS.cartpow('reals', 3));
  assert.deepEqual(VS.naturalOf(T.measure(T.REAL)), 'reals');
  assert.equal(VS.naturalOf(T.funcType([], T.REAL)), 'unknown');   // callable, not a value
});

// =====================================================================
// 4. FlatPIR third-%meta-slot emission + cross-engine parity
// =====================================================================

test('valueset: toSexpr emits the third %meta slot (golden parity)', () => {
  // flatppl-rust golden `valueset_producers_and_simplex_chain`: a
  // Dirichlet draw + softmax both land on the simplex; a Categorical's
  // support is posintegers.
  const r = processSource('x ~ Dirichlet(alpha = [1.0, 1.0, 1.0])\n'
    + 'z = softmax([0.0, 1.0])\nc = Categorical(p = x)');
  const out = require('../pir-sexpr.ts').toSexpr(r.loweredModule, { meta: true })
    .replace(/\s+/g, ' ');
  assert.ok(out.includes('%stochastic (stdsimplex 3)'),
    'Dirichlet draw → (stdsimplex 3): ' + out);
  // softmax matches the Rust golden byte-for-byte on BOTH slots — the
  // type (length-preserving → concrete 2) and the value-set. The grouped
  // wrapper goes around the whole softmax call (spec §11).
  assert.ok(out.includes('(%meta ((%array 1 (2) (%scalar real)) %fixed (stdsimplex 2)) (softmax'),
    'softmax → (%array 1 (2) real) %fixed (stdsimplex 2): ' + out);
  assert.ok(/%stochastic posintegers\) \(Categorical/.test(out),
    'Categorical support → posintegers: ' + out);
});
