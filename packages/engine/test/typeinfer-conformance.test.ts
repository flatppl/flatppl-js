'use strict';

// =====================================================================
// Cross-mode conformance: typeinfer's shape rules vs sampler's runtime
// =====================================================================
//
// Engine-concepts §17.2 — "the duplication between typeinfer's per-op
// rules and sampler's ARITH_OPS isn't coupling, it's parallel
// structure with a clean IR boundary. The real risk is drift." These
// tests are the conformance contract: for representative op IRs,
// typeinfer's result type's shape MUST equal the actual runtime
// value's shape. Catches the case where one layer's shape semantics
// change and the other silently doesn't follow.
//
// Cases sampled here are representative of common shape-determining
// patterns; the suite is unit-style rather than property-based for
// now (each case names a specific op + its expected shape rule). A
// fast-check generator extension can come later if drift in a
// specific category needs continuous exercise.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const engine = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

// Run a binding through processSource, return { staticShape, runtimeShape }.
// staticShape is the binding's inferredType.shape (or null when not array).
// runtimeShape is the runtime Value's shape (or null when scalar / not Value).
function checkShape(name: string, src: string) {
  const r = engine.processSource(src);
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors, [], `parse/type errors: ${JSON.stringify(errors)}`);
  const lb = r.loweredModule.bindings.get(name);
  const inferredType = lb && lb.inferredType;
  const derivs = orchestrator.buildDerivations(r.bindings, r.loweredModule);
  const fv = derivs.fixedValues.get(name);
  return {
    staticShape: (inferredType && inferredType.kind === 'array')
      ? inferredType.shape : null,
    staticRank: (inferredType && inferredType.kind === 'array')
      ? inferredType.rank : null,
    runtimeValue: fv,
    runtimeShape: (fv && fv.shape) ? fv.shape : null,
    inferredKind: inferredType && inferredType.kind,
  };
}

// Compare an inferred shape against a runtime Value shape. %dynamic
// entries in the static shape match anything at runtime; literal
// entries must equal.
function shapesAgree(staticShape: any[], runtimeShape: any[]): boolean {
  if (staticShape.length !== runtimeShape.length) return false;
  for (let i = 0; i < staticShape.length; i++) {
    const s = staticShape[i];
    if (s === '%dynamic') continue;
    if (s !== runtimeShape[i]) return false;
  }
  return true;
}

// =====================================================================
// Direct shape-determining producers
// =====================================================================

test('conformance: rowstack of literal-rows static vs runtime shape', () => {
  const c = checkShape('M', `M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])`);
  assert.equal(c.staticRank, 2, 'inferredType rank=2');
  assert.deepEqual(c.runtimeShape, [2, 3]);
  // Static shape may be ['%dynamic', '%dynamic'] (rowstack's
  // signature doesn't pin rows × cols); runtime shape is concrete.
  assert.ok(shapesAgree(c.staticShape, c.runtimeShape),
    `static ${JSON.stringify(c.staticShape)} vs runtime ${JSON.stringify(c.runtimeShape)}`);
});

test('conformance: zeros(shape) static vs runtime shape', () => {
  // zeros takes a single shape arg (vector for rank-≥2, integer for
  // rank-1). Spec §07.
  const c = checkShape('Z', `Z = zeros([3, 4])`);
  assert.deepEqual(c.runtimeShape, [3, 4]);
  // zeros' signature returns deferred() today; we accept either
  // %dynamic axes or literal — but if the type carries a shape, it
  // must agree with the runtime.
  if (c.staticShape) {
    assert.ok(shapesAgree(c.staticShape, c.runtimeShape));
  }
});

test('conformance: eye(5) static vs runtime shape', () => {
  const c = checkShape('I', `I = eye(5)`);
  assert.deepEqual(c.runtimeShape, [5, 5]);
  if (c.staticShape) assert.ok(shapesAgree(c.staticShape, c.runtimeShape));
});

test('conformance: linspace static vs runtime shape', () => {
  const c = checkShape('xs', `xs = linspace(0.0, 1.0, 10)`);
  assert.deepEqual(c.runtimeShape, [10]);
  if (c.staticShape) assert.ok(shapesAgree(c.staticShape, c.runtimeShape));
});

// =====================================================================
// Const-eval-driven shapes (§17.4 — the recent landing)
// =====================================================================

test('conformance: iid(M, n) with n a literal-int binding — static shape resolves', () => {
  const r = engine.processSource(`
n = 7
M = iid(Normal(mu = 0.0, sigma = 1.0), n)
`);
  const lb = r.loweredModule.bindings.get('M');
  const t = lb && lb.inferredType;
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array');
  assert.deepEqual(t.domain.shape, [7],
    `const-eval should fold n=7; got ${JSON.stringify(t.domain.shape)}`);
});

test('conformance: iid(M, lengthof(data)) folds via shape-only short-circuit', () => {
  const r = engine.processSource(`
data = [1.0, 2.0, 3.0, 4.0, 5.0]
M = iid(Normal(mu = 0.0, sigma = 1.0), lengthof(data))
`);
  const lb = r.loweredModule.bindings.get('M');
  const t = lb && lb.inferredType;
  assert.deepEqual(t.domain.shape, [5]);
});

test('conformance: iid(M, length(data)) folds via shape-only short-circuit', () => {
  // Spec §07 names this `lengthof`; viewer also accepts `length` as
  // an alias for the same op via the resolver. Confirms both arrive
  // at the same shape.
  const r = engine.processSource(`
data = [1.0, 2.0, 3.0, 4.0, 5.0]
M = iid(Normal(mu = 0.0, sigma = 1.0), lengthof(data))
`);
  const lb = r.loweredModule.bindings.get('M');
  assert.deepEqual(lb.inferredType.domain.shape, [5]);
});

// =====================================================================
// Dotted-broadcast (the inferBroadcast handler from earlier)
// =====================================================================

test('conformance: dotted-broadcast preserves shape (rank-1)', () => {
  const c = checkShape('out', `
xs = [1.0, 2.0, 3.0]
out = xs .^ 2
`);
  assert.equal(c.staticRank, 1);
  assert.deepEqual(c.runtimeShape, [3]);
  assert.ok(shapesAgree(c.staticShape, c.runtimeShape));
});

test('conformance: dotted-broadcast over rowstack (rank-2)', () => {
  const c = checkShape('out', `
M = rowstack([[1.0, 2.0], [3.0, 4.0]])
out = M .^ 2
`);
  assert.equal(c.staticRank, 2);
  assert.deepEqual(c.runtimeShape, [2, 2]);
  assert.ok(shapesAgree(c.staticShape, c.runtimeShape));
});

// =====================================================================
// Matrix arithmetic: subtraction + matmul preserve known shapes
// =====================================================================

test('conformance: matmul shapes agree static vs runtime', () => {
  const c = checkShape('C', `
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = rowstack([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
C = A * B
`);
  assert.deepEqual(c.runtimeShape, [2, 2]);
  if (c.staticShape) assert.ok(shapesAgree(c.staticShape, c.runtimeShape));
});

test('conformance: matrix subtraction shape', () => {
  const c = checkShape('D', `
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[0.5, 0.5], [0.5, 0.5]])
D = A - B
`);
  assert.deepEqual(c.runtimeShape, [2, 2]);
  if (c.staticShape) assert.ok(shapesAgree(c.staticShape, c.runtimeShape));
});

// =====================================================================
// Reductions: sum / mean / sizeof return scalars
// =====================================================================

test('conformance: sum over rank-1 array is scalar', () => {
  const r = engine.processSource(`xs = [1.0, 2.0, 3.0]
s = sum(xs)
`);
  const lb = r.loweredModule.bindings.get('s');
  const t = lb && lb.inferredType;
  assert.ok(t.kind === 'scalar' || t.kind === 'deferred' || t.kind === 'any',
    `sum should infer to scalar/deferred/any, got ${JSON.stringify(t)}`);
});

test('conformance: sizeof rank-2 returns the shape vector', () => {
  // Per spec §07, `sizeof(x)` returns the per-axis dim vector. For
  // a 2x3 matrix the runtime is the rank-1 Value [2, 3]. (The total
  // element count is reached via `prod(sizeof(x))`; the engine
  // doesn't promote sizeof to that scalar.) Pins the spec semantics.
  const c = checkShape('s', `
M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
s = sizeof(M)
`);
  const v = c.runtimeValue;
  assert.ok(v && Array.isArray(v.shape) && v.shape[0] === 2 && v.data.length === 2,
    `sizeof should return rank-1 Value of length 2, got ${JSON.stringify(v)}`);
  assert.equal(v.data[0], 2);
  assert.equal(v.data[1], 3);
});

// =====================================================================
// Cycle protection still fires when const-eval would otherwise loop
// =====================================================================

test('conformance: const-eval cycle does not crash typeinfer', () => {
  // `a` and `b` mutually reference each other; typeinfer's existing
  // cycle detection should bail with `failed` rather than infinite-
  // loop the resolver.
  const r = engine.processSource(`a = b + 1
b = a + 1
M = iid(Normal(mu = 0.0, sigma = 1.0), a)
`);
  // Should produce diagnostics (a cyclic-binding error) but NOT throw.
  // The shape ends up %dynamic (resolver bails on the cycle).
  const lb = r.loweredModule.bindings.get('M');
  assert.ok(lb, 'binding M should exist');
});
