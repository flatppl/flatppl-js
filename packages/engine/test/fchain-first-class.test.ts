'use strict';

// Integration tests for fchain as a first-class function-valued
// binding (engine-concepts §19, Phase 1). Standalone (unapplied)
// fchain bindings must:
//
//   1. Classify as a function-like binding (binding.type === 'fchain';
//      isFunctionLikeBinding(b) === true).
//   2. Carry an inferredType of funcType with the composed signature.
//   3. Be referenceable from function-arg positions of other ops
//      (broadcast, pushfwd, etc.).
//   4. Surface step-boundary type-mismatch as a diagnostic anchored
//      at the failing step.
//
// The applied form `fchain(f1, f2)(x)` is already covered by
// orchestrator.test.ts via the inlineFchain AST rewrite; we don't
// re-test that here.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');

function infer(src: string) {
  const r = processSource(src);
  return r;
}

test('fchain: standalone binding classifies as fchain', () => {
  const r = infer(`
f1   = fn(_ + 1.0)
f2   = fn(_ * 2.0)
pipe = fchain(f1, f2)
`);
  const pipe = r.bindings.get('pipe');
  assert.ok(pipe, 'pipe binding exists');
  assert.equal(pipe.type, 'fchain',
    'pipe should be classified as fchain (distinct producer tag)');
});

test('fchain: standalone binding inferredType is funcType with composed signature', () => {
  const r = infer(`
f1   = fn(_ + 1.0)
f2   = fn(_ * 2.0)
pipe = fchain(f1, f2)
`);
  const pipe = r.loweredModule.bindings.get('pipe');
  assert.ok(pipe.inferredType, 'pipe has inferredType');
  // pipe should be a funcType. The specific input/output shape depends
  // on how fn(...) lowers to functionof inputs — fn(_ + 1.0) has one
  // positional placeholder _arg1_; the chain composes them.
  assert.equal(pipe.inferredType.kind, 'function',
    'pipe.inferredType should be funcType (got ' + pipe.inferredType.kind + ')');
  assert.ok(Array.isArray(pipe.inferredType.inputs),
    'pipe.inferredType has inputs array');
  assert.ok(pipe.inferredType.result, 'pipe.inferredType has result');
});

test('fchain: single-step fchain inferredType ≡ the step', () => {
  // fchain(f1) ≡ f1 per spec — the helper returns step_0.type
  // verbatim.
  const r = infer(`
f1   = fn(_ + 1.0)
pipe = fchain(f1)
`);
  const pipe = r.loweredModule.bindings.get('pipe');
  const f1 = r.loweredModule.bindings.get('f1');
  assert.deepEqual(pipe.inferredType, f1.inferredType,
    'single-step fchain has the same type as the single function');
});

test('fchain: predicates recognise standalone fchain bindings', () => {
  const r = infer(`
f1   = fn(_ + 1.0)
f2   = fn(_ * 2.0)
pipe = fchain(f1, f2)
`);
  const pipe = r.bindings.get('pipe');
  const matShared = require('../materialiser-shared.ts');
  assert.equal(matShared.isFunctionLikeBinding(pipe), true,
    'isFunctionLikeBinding should accept fchain bindings');
  // Reconstruct a binding object with both producer tag and inferredType
  // — the analyzer-bindings map carries the type, but inferredType lives
  // on the loweredModule bindings. Pass a synthetic binding with both
  // for the predicate test.
  const lowered = r.loweredModule.bindings.get('pipe');
  const synthetic = Object.assign({}, pipe, { inferredType: lowered.inferredType });
  assert.equal(matShared.isCallableLayerBinding(synthetic), true,
    'isCallableLayerBinding should accept fchain bindings via inferredType.kind === "function"');
});

test('fchain: type-mismatch at step boundary emits diagnostic anchored at failing step', () => {
  // f2 returns a scalar (not a record); f1 has two named inputs.
  // Multi-input next-step boundary requires a record-typed previous
  // result for auto-splat — scalar→multi-input is a static mismatch
  // (helper's `_matchChainBoundary` enforces this).
  const r = infer(`
v_a ~ Normal(mu = 0.0, sigma = 1.0)
v_b ~ Normal(mu = 0.0, sigma = 1.0)
f1 = functionof(v_a + v_b, v_a = v_a, v_b = v_b)
f2 = functionof(v_a + 1.0, v_a = v_a)
pipe = fchain(f2, f1)
`);
  const stepDiag = r.diagnostics.find((d: any) =>
    /fchain.*step boundary/i.test(d.message));
  assert.ok(stepDiag, 'should have an fchain step-boundary diagnostic; got: '
    + JSON.stringify(r.diagnostics.map((d: any) => d.message)));
  // Diagnostic anchors at the failing step (`f1` in this fixture).
  assert.match(stepDiag.message, /f1/,
    'diagnostic should mention the failing step name');
});

test('fchain: non-function step emits clear diagnostic', () => {
  // Passing a non-function (a measure!) to fchain is a type error.
  const r = infer(`
y_dist = Normal(mu = 0.0, sigma = 1.0)
f1     = fn(_ + 1.0)
bad    = fchain(f1, y_dist)
`);
  const stepDiag = r.diagnostics.find((d: any) =>
    /fchain.*not a function/i.test(d.message));
  assert.ok(stepDiag, 'should have an fchain non-function-step diagnostic');
});

test('fchain: standalone binding is not in derivations map (function value)', () => {
  // fchain bindings are function values, not measures. They should
  // NOT have a derivation entry — that's a measure-algebra concept.
  // Today's behaviour stays: derivations[pipe] is undefined.
  const r = infer(`
f1   = fn(_ + 1.0)
f2   = fn(_ * 2.0)
pipe = fchain(f1, f2)
`);
  const built = orchestrator.buildDerivations(r.bindings);
  assert.ok(!('pipe' in built.derivations),
    'pipe should not have a derivation (it is a function value)');
});

test('fchain: empty fchain() is a clear error (spec forbids nullary)', () => {
  const r = infer(`
pipe = fchain()
`);
  const stepDiag = r.diagnostics.find((d: any) =>
    /fchain/i.test(d.message));
  assert.ok(stepDiag,
    'fchain() with no args should produce a diagnostic');
});

test('fchain: auto-splatting at multi-input boundary works end-to-end', () => {
  // f1 returns record(a, b); f2 has inputs {a, b}. Per spec §04 +
  // recently-added §06 auto-splatting note: the record auto-splats
  // into f2's keyword inputs by field name.
  const r = infer(`
x  = elementof(reals)
f1 = functionof(record(a = x, b = 2.0 * x), x = x)
a  = elementof(reals)
b  = elementof(reals)
f2 = functionof(a + b, a = a, b = b)
pipe = fchain(f1, f2)
`);
  const stepBoundaryDiag = r.diagnostics.find((d: any) =>
    /fchain.*step boundary/i.test(d.message));
  assert.equal(stepBoundaryDiag, undefined,
    'auto-splatting record → kwargs should compose without diagnostic; got: '
    + JSON.stringify(r.diagnostics.map((d: any) => d.message)));
  const pipe = r.loweredModule.bindings.get('pipe');
  assert.equal(pipe.inferredType.kind, 'function');
});
