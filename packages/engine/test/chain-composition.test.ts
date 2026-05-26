'use strict';

// Unit tests for `inferChainComposition` (engine-concepts §19.4) and
// its helper `_matchChainBoundary`. Pure helper — no IR walking — so
// tests construct funcType / kernelType / record types directly and
// assert the composed result.
//
// Phase 1 covers mode='func' (fchain). mode='kchain-marginal' /
// 'jointchain-retain' are placeholders here; Phase 2 will populate.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const T = require('..').types;
const densityPrims = require('../density-prims.ts');

const { inferChainComposition } = densityPrims;
const { _matchChainBoundary } = densityPrims._internal;

// Convenience builders for the test fixtures.
function fn(inputs: Array<{name: string; type: any}>, result: any) {
  return T.funcType(inputs, result);
}
function step(type: any, name?: string, loc?: any) {
  return { type, name, loc };
}

test('inferChainComposition: empty step list fails', () => {
  const r = inferChainComposition([], 'func');
  assert.equal(r.resultType.kind, 'failed');
});

test('inferChainComposition: single-step fchain ≡ the step', () => {
  // fchain(f) ≡ f per spec §04 (left-associative; identity at N=1).
  const f = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const r = inferChainComposition([step(f, 'f')], 'func');
  assert.deepEqual(r.resultType, f);
  assert.deepEqual(r.diagnostics, []);
});

test('inferChainComposition: two-step scalar→scalar composition', () => {
  // f1: real → real; f2: real → real ⇒ fchain ≡ real → real.
  const f1 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const f2 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const r = inferChainComposition([step(f1, 'f1'), step(f2, 'f2')], 'func');
  assert.equal(r.resultType.kind, 'function');
  assert.equal(r.resultType.inputs.length, 1);
  assert.equal(r.resultType.inputs[0].name, 'x');
  assert.deepEqual(r.resultType.inputs[0].type, T.REAL);
  assert.deepEqual(r.resultType.result, T.REAL);
  assert.deepEqual(r.diagnostics, []);
});

test('inferChainComposition: three-step scalar chain', () => {
  // f1: real → integer; f2: integer → real; f3: real → boolean.
  // Result: real → boolean.
  const f1 = fn([{ name: 'x', type: T.REAL }], T.INTEGER);
  const f2 = fn([{ name: 'x', type: T.INTEGER }], T.REAL);
  const f3 = fn([{ name: 'x', type: T.REAL }], T.BOOLEAN);
  const r = inferChainComposition(
    [step(f1, 'f1'), step(f2, 'f2'), step(f3, 'f3')], 'func');
  assert.equal(r.resultType.kind, 'function');
  assert.deepEqual(r.resultType.result, T.BOOLEAN);
  assert.deepEqual(r.diagnostics, []);
});

test('inferChainComposition: scalar promotion at boundary (integer→real)', () => {
  // f1: real → integer; f2: real → real. integer promotes to real
  // per the spec's promotion ladder (booleans ⊂ integers ⊂ reals).
  const f1 = fn([{ name: 'x', type: T.REAL }], T.INTEGER);
  const f2 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const r = inferChainComposition([step(f1), step(f2)], 'func');
  assert.equal(r.resultType.kind, 'function');
  assert.deepEqual(r.diagnostics, []);
});

test('inferChainComposition: type-mismatch diagnostic anchors at next step', () => {
  // f1: real → real; f2: integer → real. real does not promote to
  // integer; the boundary is a static mismatch.
  const loc1 = { start: { line: 1 }, end: { line: 1 } };
  const loc2 = { start: { line: 2 }, end: { line: 2 } };
  const f1 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const f2 = fn([{ name: 'x', type: T.INTEGER }], T.REAL);
  const r = inferChainComposition(
    [step(f1, 'f1', loc1), step(f2, 'f2', loc2)], 'func');
  assert.equal(r.resultType.kind, 'failed');
  assert.equal(r.diagnostics.length, 1);
  const d = r.diagnostics[0];
  assert.equal(d.severity, 'error');
  assert.match(d.message, /step boundary 0 → 1/);
  assert.match(d.message, /f1/);
  assert.match(d.message, /f2/);
  // Diagnostic points at the failing-step's loc, not the prior step's.
  assert.deepEqual(d.loc, loc2);
});

test('inferChainComposition: deferred step ⇒ deferred chain', () => {
  // A step typed as deferred / any propagates as deferred. typeinfer
  // trusts runtime for these.
  const f1 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const r = inferChainComposition(
    [step(f1, 'f1'), step(T.deferred(), 'f2')], 'func');
  assert.equal(r.resultType.kind, 'deferred');
});

test('inferChainComposition: failed step ⇒ failed chain', () => {
  const f1 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const r = inferChainComposition(
    [step(f1, 'f1'), step(T.failed('upstream'), 'f2')], 'func');
  assert.equal(r.resultType.kind, 'failed');
});

test('inferChainComposition: non-function step diagnostic', () => {
  // fchain expects function-typed steps. A measure-typed step is a
  // static error (clear diagnostic, not silent acceptance).
  const f1 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const r = inferChainComposition(
    [step(f1, 'f1'),
     step(T.measure(T.REAL), 'm', { start: { line: 3 } })], 'func');
  assert.equal(r.resultType.kind, 'failed');
  assert.equal(r.diagnostics.length, 1);
  assert.match(r.diagnostics[0].message, /not a function/);
  assert.match(r.diagnostics[0].message, /m/);
});

test('inferChainComposition: kernel modes reject funcType-only chains', () => {
  // Kernel modes require step 0 to be a measure or kernel, not a
  // function. Passing a funcType-only chain to a kernel mode is a
  // static type error — kernel chains are the measure-side analogue.
  const f1 = fn([{ name: 'x', type: T.REAL }], T.REAL);
  const r1 = inferChainComposition([step(f1)], 'kchain-marginal');
  const r2 = inferChainComposition([step(f1)], 'jointchain-retain');
  assert.equal(r1.resultType.kind, 'failed');
  assert.equal(r2.resultType.kind, 'failed');
  assert.match(r1.resultType.reason, /bad step 0/);
});

// =====================================================================
// Kernel modes (Phase 2): kchain-marginal + jointchain-retain
// =====================================================================

function kernel(inputs: Array<{name: string; type: any}>, variate: any) {
  return T.kernelType(inputs, T.measure(variate));
}

test('inferChainComposition: closed-first kchain ⇒ measureType result', () => {
  // M (measure<real>) → K1 (real → measure<integer>) ⇒ measure<integer>.
  const M  = T.measure(T.REAL);
  const K1 = kernel([{ name: 'x', type: T.REAL }], T.INTEGER);
  const r = inferChainComposition([step(M, 'M'), step(K1, 'K1')], 'kchain-marginal');
  assert.equal(r.resultType.kind, 'measure',
    'closed-first kchain (empty residual) collapses to measureType');
  assert.deepEqual(r.resultType.domain, T.INTEGER);
  assert.deepEqual(r.diagnostics, []);
});

test('inferChainComposition: kernel-first kchain ⇒ kernelType with residual inputs', () => {
  // K0 (theta: real → measure<real>) → K1 (real → measure<integer>)
  // ⇒ kernelType({theta: real}, measure<integer>). Residual = K0.inputs.
  const K0 = kernel([{ name: 'theta', type: T.REAL }], T.REAL);
  const K1 = kernel([{ name: 'x', type: T.REAL }], T.INTEGER);
  const r = inferChainComposition([step(K0, 'K0'), step(K1, 'K1')], 'kchain-marginal');
  assert.equal(r.resultType.kind, 'kernel',
    'kernel-first kchain keeps residual inputs as kernelType');
  assert.equal(r.resultType.inputs.length, 1);
  assert.equal(r.resultType.inputs[0].name, 'theta');
  assert.equal(r.resultType.result.kind, 'measure');
  assert.deepEqual(r.resultType.result.domain, T.INTEGER);
});

test('inferChainComposition: closed-first jointchain ⇒ measure of retained record/tuple', () => {
  // jointchain(M, K1) keyword form ⇒ measure<record({m: real, k: integer})>.
  const M  = T.measure(T.REAL);
  const K1 = kernel([{ name: 'x', type: T.REAL }], T.INTEGER);
  const r = inferChainComposition([step(M, 'M'), step(K1, 'K1')],
                                  'jointchain-retain', { labels: ['m', 'k'] });
  assert.equal(r.resultType.kind, 'measure');
  assert.equal(r.resultType.domain.kind, 'record');
  assert.deepEqual(r.resultType.domain.fields.m, T.REAL);
  assert.deepEqual(r.resultType.domain.fields.k, T.INTEGER);
});

test('inferChainComposition: positional jointchain ⇒ measure of tuple variate', () => {
  // Positional form (no labels) yields a tuple variate (the spec's
  // `cat(...)` of mixed-typed variates).
  const M  = T.measure(T.REAL);
  const K1 = kernel([{ name: 'x', type: T.REAL }], T.INTEGER);
  const r = inferChainComposition([step(M, 'M'), step(K1, 'K1')],
                                  'jointchain-retain');
  assert.equal(r.resultType.kind, 'measure');
  assert.equal(r.resultType.domain.kind, 'tuple');
  assert.equal(r.resultType.domain.elems.length, 2);
});

test('inferChainComposition: kernel-first jointchain ⇒ kernelType result', () => {
  // K0 (theta: real → measure<real>) → K1 (real → measure<integer>)
  // ⇒ kernelType({theta: real}, measure<record(s0: real, s1: integer)>).
  const K0 = kernel([{ name: 'theta', type: T.REAL }], T.REAL);
  const K1 = kernel([{ name: 'x', type: T.REAL }], T.INTEGER);
  const r = inferChainComposition([step(K0, 'K0'), step(K1, 'K1')],
                                  'jointchain-retain', { labels: ['s0', 's1'] });
  assert.equal(r.resultType.kind, 'kernel',
    'kernel-first jointchain stays a kernel');
  assert.equal(r.resultType.inputs.length, 1);
  assert.equal(r.resultType.inputs[0].name, 'theta');
  assert.equal(r.resultType.result.domain.kind, 'record');
  assert.deepEqual(r.resultType.result.domain.fields.s0, T.REAL);
  assert.deepEqual(r.resultType.result.domain.fields.s1, T.INTEGER);
});

test('inferChainComposition: kchain step boundary mismatch (variate ↛ next-step input)', () => {
  // M variate real → K1 expects integer-typed input. real doesn't
  // promote to integer; boundary mismatch.
  const M  = T.measure(T.REAL);
  const K1 = kernel([{ name: 'x', type: T.INTEGER }], T.REAL);
  const r = inferChainComposition([step(M, 'M'), step(K1, 'K1')], 'kchain-marginal');
  assert.equal(r.resultType.kind, 'failed');
  assert.match(r.diagnostics[0].message, /step boundary 0 → 1/);
});

test('inferChainComposition: kchain step boundary auto-splat record→kwargs', () => {
  // M variate record(a, b) → K1 inputs {a, b} via auto-splat.
  const M  = T.measure(T.record({ a: T.REAL, b: T.INTEGER }));
  const K1 = kernel([{ name: 'a', type: T.REAL },
                     { name: 'b', type: T.INTEGER }], T.REAL);
  const r = inferChainComposition([step(M, 'M'), step(K1, 'K1')], 'kchain-marginal');
  assert.equal(r.resultType.kind, 'measure');
  assert.deepEqual(r.resultType.domain, T.REAL);
});

test('inferChainComposition: kchain measure-typed step i≥1 accepted as nullary kernel', () => {
  // Per spec §06 line 86-91 (uniform kernel extension), a measure IS
  // a nullary kernel. So a measure-typed step at position i≥1 is
  // accepted as a closed-kernel step (no boundary match required —
  // the next step produces a measure regardless of the prior
  // variate). Strict non-nullary enforcement is intentionally
  // deferred (existing fixtures rely on the looseness).
  const M  = T.measure(T.REAL);
  const M1 = T.measure(T.REAL);   // nullary kernel = closed measure.
  const r = inferChainComposition([step(M, 'M'), step(M1, 'M1')], 'kchain-marginal');
  assert.equal(r.resultType.kind, 'measure',
    'closed-first kchain with measure step ≡ measure result');
});

test('inferChainComposition: kchain function-typed step i≥1 still rejected', () => {
  // A function step (value→value) is genuinely not a kernel; should
  // be rejected even with the kernel↔measure unification.
  const M = T.measure(T.REAL);
  const f = T.funcType([{ name: 'x', type: T.REAL }], T.REAL);
  const r = inferChainComposition([step(M, 'M'), step(f, 'f')], 'kchain-marginal');
  assert.equal(r.resultType.kind, 'failed');
  assert.match(r.diagnostics[0].message, /must be a kernel or measure/);
});

test('inferChainComposition: residual-empty collapse (spec §06 kernel↔measure)', () => {
  // Specifically pin the collapse: closed-first chain returns
  // measureType, not kernelType with empty inputs. This is the spec
  // §06 line 87-90 "kernel with empty interface IS a measure" — falls
  // out of the helper without a special case.
  const M  = T.measure(T.REAL);
  const K1 = kernel([{ name: 'x', type: T.REAL }], T.INTEGER);
  const r = inferChainComposition([step(M, 'M'), step(K1, 'K1')], 'kchain-marginal');
  assert.equal(r.resultType.kind, 'measure');
  assert.notEqual(r.resultType.kind, 'kernel');
});

test('inferChainComposition: auto-splat at multi-input boundary', () => {
  // f1 returns record(a: real, b: real); f2 has inputs {a: real, b: real}.
  // Spec §04 + §06 (recently added): record-shaped variate auto-splats
  // into a callable's keyword inputs by field name.
  const f1 = fn([{ name: 'x', type: T.REAL }],
                T.record({ a: T.REAL, b: T.REAL }));
  const f2 = fn([{ name: 'a', type: T.REAL },
                 { name: 'b', type: T.REAL }],
                T.REAL);
  const r = inferChainComposition([step(f1, 'f1'), step(f2, 'f2')], 'func');
  assert.equal(r.resultType.kind, 'function');
  assert.deepEqual(r.resultType.result, T.REAL);
  assert.deepEqual(r.diagnostics, []);
});

test('inferChainComposition: auto-splat with missing field is a diagnostic', () => {
  // f1 returns record(a: real); f2 expects {a: real, b: real}. f2's
  // input `b` has no matching field in f1's result.
  const f1 = fn([{ name: 'x', type: T.REAL }], T.record({ a: T.REAL }));
  const f2 = fn([{ name: 'a', type: T.REAL },
                 { name: 'b', type: T.REAL }], T.REAL);
  const r = inferChainComposition([step(f1, 'f1'), step(f2, 'f2')], 'func');
  assert.equal(r.resultType.kind, 'failed');
  assert.match(r.diagnostics[0].message, /missing field "b"/);
});

test('inferChainComposition: auto-splat with extra field is a diagnostic', () => {
  // f1 returns record(a, b, c); f2 expects {a, b}. f1's `c` field has
  // no matching f2 input — the auto-splat is strict (every field must
  // bind something).
  const f1 = fn([{ name: 'x', type: T.REAL }],
                T.record({ a: T.REAL, b: T.REAL, c: T.REAL }));
  const f2 = fn([{ name: 'a', type: T.REAL },
                 { name: 'b', type: T.REAL }], T.REAL);
  const r = inferChainComposition([step(f1, 'f1'), step(f2, 'f2')], 'func');
  assert.equal(r.resultType.kind, 'failed');
  assert.match(r.diagnostics[0].message, /extra field "c"/);
});

test('inferChainComposition: multi-input boundary needs record (positional rejected)', () => {
  // f1 returns array(real, [2]); f2 has two scalar inputs. Spec auto-
  // splat only applies for record → kwargs; an array→multi-input
  // step is a static error (the next step would need positional
  // inputs declared, but functionof-produced functions have keyword-
  // only inputs unless explicit boundaries — handled at typeinfer of
  // the function itself).
  const f1 = fn([{ name: 'x', type: T.REAL }],
                T.array(1, [2], T.REAL));
  const f2 = fn([{ name: 'a', type: T.REAL },
                 { name: 'b', type: T.REAL }], T.REAL);
  const r = inferChainComposition([step(f1, 'f1'), step(f2, 'f2')], 'func');
  assert.equal(r.resultType.kind, 'failed');
  assert.match(r.diagnostics[0].message,
    /multi-input step boundary requires a record-typed previous result/);
});

test('inferChainComposition: single-input boundary accepts non-record', () => {
  // f1: real → array(real, [2]); f2: array(real, [2]) → real.
  // Single-input boundary; the array type is fine — no auto-splat.
  const f1 = fn([{ name: 'x', type: T.REAL }], T.array(1, [2], T.REAL));
  const f2 = fn([{ name: 'v', type: T.array(1, [2], T.REAL) }], T.REAL);
  const r = inferChainComposition([step(f1, 'f1'), step(f2, 'f2')], 'func');
  assert.equal(r.resultType.kind, 'function');
  assert.deepEqual(r.resultType.result, T.REAL);
});

test('inferChainComposition: chain inherits step_0\'s inputs verbatim', () => {
  // The chain's input signature must be exactly step_0's inputs — not
  // freshly-constructed, not re-keyed.
  const f1 = fn([{ name: 'alpha', type: T.REAL },
                 { name: 'beta',  type: T.INTEGER }],
                T.record({ a: T.REAL, b: T.REAL }));
  const f2 = fn([{ name: 'a', type: T.REAL },
                 { name: 'b', type: T.REAL }], T.BOOLEAN);
  const r = inferChainComposition([step(f1), step(f2)], 'func');
  assert.equal(r.resultType.kind, 'function');
  assert.equal(r.resultType.inputs.length, 2);
  assert.equal(r.resultType.inputs[0].name, 'alpha');
  assert.deepEqual(r.resultType.inputs[0].type, T.REAL);
  assert.equal(r.resultType.inputs[1].name, 'beta');
  assert.deepEqual(r.resultType.inputs[1].type, T.INTEGER);
  assert.deepEqual(r.resultType.result, T.BOOLEAN);
});

// =====================================================================
// _matchChainBoundary — unit tests for the per-boundary matcher.
// =====================================================================

test('_matchChainBoundary: single-input positional success', () => {
  const m = _matchChainBoundary(T.REAL,
    [{ name: 'x', type: T.REAL }]);
  assert.equal(m.ok, true);
});

test('_matchChainBoundary: single-input positional failure', () => {
  const m = _matchChainBoundary(T.STRING,
    [{ name: 'x', type: T.REAL }]);
  assert.equal(m.ok, false);
  assert.match(m.reason, /cannot unify/);
});

test('_matchChainBoundary: multi-input record auto-splat success', () => {
  const m = _matchChainBoundary(
    T.record({ a: T.REAL, b: T.INTEGER }),
    [{ name: 'a', type: T.REAL }, { name: 'b', type: T.INTEGER }]);
  assert.equal(m.ok, true);
});

test('_matchChainBoundary: multi-input non-record rejected', () => {
  const m = _matchChainBoundary(
    T.array(1, [3], T.REAL),
    [{ name: 'a', type: T.REAL }, { name: 'b', type: T.REAL }]);
  assert.equal(m.ok, false);
  assert.match(m.reason, /requires a record-typed previous result/);
});
