'use strict';

// Spec §07 Norms and normalization: l1norm, l2norm, linfnorm, l1unit,
// l2unit, logsumexp, softmax, logsoftmax. All pure vector→scalar or
// vector→vector reductions dispatched through ARITH_OPS.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const valueLib = require('../value.ts');
const { processSource, types: T } = require('..');
const { toJS } = require('./_value-helpers.ts');

function lit(v: any)        { return { kind: 'lit', value: v }; }
function vec(...vs: any[])    { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function call(op: any, v: any)   { return { kind: 'call', op, args: [v] }; }
const evRaw = (ir: any) => sampler.evaluateExpr(ir, {});
const ev = (ir: any) => toJS(evRaw(ir));

// Surface `complex(re, im)` / vector-of-complex IR, for §07's
// `real/complex vectors` domain. `cvec` builds the same IR the parser
// lowers `[complex(3.0, 4.0), complex(1.0, 0.0)]` to, so these go
// through the ordinary evaluation path — no hand-built Values.
function cx(re: number, im: number) {
  return { kind: 'call', op: 'complex', args: [lit(re), lit(im)] };
}
function cvec(...zs: any[]) { return { kind: 'call', op: 'vector', args: zs }; }
// toJS reads only a complex Value's real buffer, so complex results are
// compared through readComplex, which also applies the conjugation sign.
function parts(v: any) {
  const z = valueLib.readComplex(v);
  return { re: Array.from(z.re), im: Array.from(z.im) };
}
// Errors from a full parse → analyze run, for the static-domain checks.
function errorsFor(src: string) {
  return processSource(src).diagnostics
    .filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

function arrClose(a: any, b: any, tol?: any) {
  tol = tol == null ? 1e-12 : tol;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (!(Math.abs(a[i] - b[i]) <= tol)) return false;
  }
  return true;
}

// =====================================================================
// l1norm / l2norm
// =====================================================================

test('l1norm: Σ|v_i|', () => {
  assert.equal(ev(call('l1norm', vec(1, -2, 3, -4))), 10);
});

test('l2norm: classic 3-4-5 right triangle', () => {
  assert.equal(ev(call('l2norm', vec(3, 4))), 5);
});

test('l1norm / l2norm on empty vector ⇒ 0', () => {
  assert.equal(ev(call('l1norm', vec())), 0);
  assert.equal(ev(call('l2norm', vec())), 0);
});

// =====================================================================
// linfnorm — max_i |v_i|
// =====================================================================

test('linfnorm: largest magnitude wins, sign ignored', () => {
  assert.equal(ev(call('linfnorm', vec(-3, 2, -7))), 7);
  assert.equal(ev(call('linfnorm', vec(1, 1, 1))), 1);
});

test('linfnorm on empty vector ⇒ 0 (same convention as l1norm / l2norm)', () => {
  // Oracle: LinearAlgebra.norm(Float64[], Inf) == 0.0.
  assert.equal(ev(call('linfnorm', vec())), 0);
});

// =====================================================================
// §07's `real/complex vectors` domain
//
// Oracles, derived three independent ways for v = [3+4i, 1]:
// by hand from §07's formulas (Σ|v_i| = 5+1 = 6; √Σ|v_i|² = √26;
// max|v_i| = 5; v/‖v‖), cross-checked against LinearAlgebra.norm in
// Julia and numpy.linalg.norm — all three agree bit-for-bit.
// =====================================================================

test('l1norm / l2norm / linfnorm on a complex vector reduce over the moduli (§07 domain)', () => {
  const v = () => cvec(cx(3, 4), cx(1, 0));
  assert.equal(ev(call('l1norm', v())), 6);
  assert.equal(ev(call('l2norm', v())), 5.0990195135927845);   // √26
  assert.equal(ev(call('linfnorm', v())), 5);
});

test('the complex domain is REACHABLE from source — no diagnostic (§07 domain)', () => {
  // The guard for the defect this closed: the norms used to type their
  // argument `array(1, …, REAL)`, so every call below was a static error
  // and the runtime complex branches were dead code.
  const decl = 'v = [complex(3.0, 4.0), complex(1.0, 0.0)]\n';
  for (const op of ['l1norm', 'l2norm', 'linfnorm', 'l1unit', 'l2unit']) {
    assert.deepEqual(errorsFor(decl + 'n = ' + op + '(v)'), [],
      op + ' must accept a complex vector');
  }
  // The result types: a norm is real-valued whatever the input; a unit
  // vector keeps its argument's element type (§07 divides by a REAL norm).
  const { bindings } = processSource(decl + 'a = l2norm(v)\nb = l2unit(v)');
  assert.ok(T.equal(bindings.get('a').inferredType, T.REAL));
  assert.ok(T.equal(bindings.get('b').inferredType, T.array(1, [2], T.COMPLEX)));
});

test('logsumexp / softmax / logsoftmax still reject a complex vector (§07 real vectors)', () => {
  // §07 gives these three the domain `real vectors`, not `real/complex
  // vectors` — widening the norms must not widen them.
  const decl = 'v = [complex(3.0, 4.0), complex(1.0, 0.0)]\n';
  for (const op of ['logsumexp', 'softmax', 'logsoftmax']) {
    const errs = errorsFor(decl + 'n = ' + op + '(v)');
    assert.equal(errs.length, 1, op + ' must reject a complex vector');
    assert.match(errs[0], /expects array of real, got array of complex/);
  }
});

test('l1unit / l2unit on a complex vector divide by the real norm (§07 domain)', () => {
  // v / ‖v‖₁ = [(3+4i)/6, 1/6]; v / ‖v‖₂ = [(3+4i)/√26, 1/√26].
  const v = () => cvec(cx(3, 4), cx(1, 0));
  assert.deepEqual(parts(evRaw(call('l1unit', v()))), {
    re: [0.5, 0.16666666666666666],
    im: [0.6666666666666666, 0],
  });
  assert.deepEqual(parts(evRaw(call('l2unit', v()))), {
    re: [0.5883484054145521, 0.19611613513818404],
    im: [0.7844645405527362, 0],
  });
});

test('a mixed real/complex vector literal norms over the moduli', () => {
  // `[complex(3.0, 4.0), 1.0]` — the real entry promotes (§03: reals
  // embed into complexes), so the result matches [3+4i, 1] exactly.
  const v = () => cvec(cx(3, 4), lit(1.0));
  assert.equal(ev(call('l1norm', v())), 6);
  assert.equal(ev(call('l2norm', v())), 5.0990195135927845);
});

test('a complex vector holding only real values agrees with the real path', () => {
  // Oracle: norm([1.5+0i, -2+0i], p) == norm([1.5, -2], p) for p ∈ {1,2,∞}.
  const cv = () => cvec(cx(1.5, 0), cx(-2, 0));
  const rv = () => vec(1.5, -2);
  for (const op of ['l1norm', 'l2norm', 'linfnorm']) {
    assert.equal(ev(call(op, cv())), ev(call(op, rv())), op);
  }
  assert.equal(ev(call('l1norm', cv())), 3.5);
  assert.equal(ev(call('l2norm', cv())), 2.5);
  assert.equal(ev(call('linfnorm', cv())), 2);
  // The unit form keeps the complex element type, with a zero imaginary part.
  assert.deepEqual(parts(evRaw(call('l2unit', cv()))), { re: [0.6, -0.8], im: [0, 0] });
});

test('the norms accept the planar complex Value as well as the vector-of-complex form', () => {
  // The engine carries two complex representations (value.ts "Complex
  // Values"): the planar Value `dtype: 'complex'`, and the JS array of
  // scalar {re, im} that `vector(...)` falls back to — which is what the
  // surface tests above exercise. A norm reading only one still drops the
  // imaginary half on the other, so both are pinned.
  const v = valueLib.complexValue(Float64Array.from([3, 1]), Float64Array.from([4, 0]), [2]);
  const OPS = sampler._internal.ARITH_OPS;
  assert.equal(OPS.l1norm(v), 6);
  assert.equal(OPS.l2norm(v), 5.0990195135927845);
  assert.equal(OPS.linfnorm(v), 5);
  // Conjugation is a lazy tag flip, so the norms must read the LOGICAL
  // imaginary part: |conj(z)| = |z| leaves all three norms unchanged, and
  // l2unit's imaginary part flips sign.
  const c = valueLib.conjugate(v);
  assert.equal(OPS.l1norm(c), 6);
  assert.equal(OPS.l2norm(c), 5.0990195135927845);
  assert.equal(OPS.linfnorm(c), 5);
  // conj(1 + 0i) is 1 − 0i, so the second imaginary part is a signed zero.
  assert.deepEqual(parts(OPS.l2unit(c)), {
    re: [0.5883484054145521, 0.19611613513818404],
    im: [-0.7844645405527362, -0],
  });
});

test('an empty complex vector norms to 0 and units to the empty vector', () => {
  // Same empty-input convention as the real path (§07 "Empty inputs").
  const e = valueLib.complexValue(new Float64Array(0), new Float64Array(0), [0]);
  const OPS = sampler._internal.ARITH_OPS;
  assert.equal(OPS.l1norm(e), 0);
  assert.equal(OPS.l2norm(e), 0);
  assert.equal(OPS.linfnorm(e), 0);
  assert.deepEqual(OPS.l1unit(e).shape, [0]);
  assert.deepEqual(OPS.l2unit(e).shape, [0]);
});

test('a zero-norm complex vector has no unit form', () => {
  const z = valueLib.complexValue(Float64Array.from([0, 0]), Float64Array.from([0, 0]), [2]);
  const OPS = sampler._internal.ARITH_OPS;
  assert.throws(() => OPS.l1unit(z), /zero-norm/);
  assert.throws(() => OPS.l2unit(z), /zero-norm/);
});

// =====================================================================
// l1unit / l2unit
// =====================================================================

test('l1unit: sums to 1 (absolute) — uniform probability vector', () => {
  const r = ev(call('l1unit', vec(2, 2, 2, 2)));
  assert.ok(arrClose(r, [0.25, 0.25, 0.25, 0.25]));
});

test('l2unit: norm of result is 1', () => {
  const r = ev(call('l2unit', vec(3, 4)));
  assert.ok(Math.abs(Math.hypot(...r) - 1) < 1e-12);
  // Direction preserved: [3, 4] / 5 = [0.6, 0.8]
  assert.ok(arrClose(r, [0.6, 0.8]));
});

test('l1unit / l2unit on zero-norm vector throws', () => {
  assert.throws(() => ev(call('l1unit', vec(0, 0, 0))), /zero-norm/);
  assert.throws(() => ev(call('l2unit', vec(0, 0, 0))), /zero-norm/);
});

// Empty input — no quotient is ever evaluated, so the result is
// vacuously the empty vector (empty-array ruling, flatppl-dev/
// empty-arrays-ruling.md), not the zero-norm error. Was a throw.
test('l1unit / l2unit on empty vector ⇒ the empty vector, not the zero-norm error', () => {
  assert.deepEqual(ev(call('l1unit', vec())), []);
  assert.deepEqual(ev(call('l2unit', vec())), []);
});

// =====================================================================
// logsumexp — numerically stable log Σ exp
// =====================================================================

test('logsumexp: log Σ exp on small uniform vector', () => {
  // logsumexp([0, 0, 0]) = log(3)
  assert.ok(Math.abs(ev(call('logsumexp', vec(0, 0, 0))) - Math.log(3)) < 1e-12);
});

test('logsumexp: numerically stable at large entries', () => {
  // Direct exp(1000) would overflow; logsumexp must give 1000 + log(1).
  assert.ok(Math.abs(ev(call('logsumexp', vec(1000))) - 1000) < 1e-12);
  // [1000, 1000, 1000]: result = 1000 + log(3).
  assert.ok(Math.abs(ev(call('logsumexp', vec(1000, 1000, 1000))) -
    (1000 + Math.log(3))) < 1e-12);
});

test('logsumexp: empty vector ⇒ -Infinity', () => {
  assert.equal(ev(call('logsumexp', vec())), -Infinity);
});

// =====================================================================
// softmax / logsoftmax
// =====================================================================

test('softmax: uniform input ⇒ uniform output', () => {
  const r = ev(call('softmax', vec(0, 0, 0)));
  assert.ok(arrClose(r, [1/3, 1/3, 1/3], 1e-12));
});

test('softmax: sums to 1', () => {
  const r = ev(call('softmax', vec(1.0, 2.0, 3.0, 0.5)));
  const sum = r.reduce((s: any, v: any) => s + v, 0);
  assert.ok(Math.abs(sum - 1) < 1e-12);
});

test('softmax: shift-invariance — adding a constant to all entries gives the same output', () => {
  const r1 = ev(call('softmax', vec(1, 2, 3)));
  const r2 = ev(call('softmax', vec(101, 102, 103)));
  assert.ok(arrClose(r1, r2, 1e-12));
});

test('logsoftmax: exp ∘ logsoftmax = softmax', () => {
  const v = vec(0.5, -1.0, 2.0, 0.25);
  const ls = ev(call('logsoftmax', v));
  const s  = ev(call('softmax', v));
  const expLs = ls.map((x: any) => Math.exp(x));
  assert.ok(arrClose(expLs, s, 1e-12));
});

test('logsoftmax: each entry equals v_i − logsumexp(v)', () => {
  const v   = vec(1.0, 2.0, 3.0);
  const ls  = ev(call('logsoftmax', v));
  const lse = ev(call('logsumexp', v));
  const expected = [1 - lse, 2 - lse, 3 - lse];
  assert.ok(arrClose(ls, expected, 1e-12));
});

// Empty input — softmax / logsoftmax give the empty vector (empty-array
// ruling), already conformant; pinned here so a future change is
// deliberate.
test('softmax([]) / logsoftmax([]) ⇒ the empty vector', () => {
  assert.deepEqual(ev(call('softmax', vec())), []);
  assert.deepEqual(ev(call('logsoftmax', vec())), []);
});
