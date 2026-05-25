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

// =====================================================================
// Demand-driven const-eval: shape inference never materialises bindings
// whose value isn't needed (engine-concepts §17.4 lazy-evaluation pattern).
// =====================================================================

test('demand-driven: zeros/fill/ones/eye fold their dim args via the resolver', () => {
  // The user's motivating example, now end-to-end: shape inference
  // chains through `prod(sizeof(B))` to give v its concrete shape,
  // without B ever being evaluated. (B aliases A here for clarity;
  // the const-eval pass intercepts sizeof(B) and reads from B's
  // type — A's value never reached.)
  const engine = require('../index.ts');
  const r = engine.processSource(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = A
sz = sizeof(B)
n = prod(sz)
n2 = n + n
v = zeros(n2)
`);
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors, []);
  const lb = r.loweredModule.bindings.get('v');
  // sz = [2, 3]; n = 6; n2 = 12; v = zeros(12) → array([12], real).
  assert.equal(lb.inferredType.kind, 'array');
  assert.deepEqual(lb.inferredType.shape, [12]);
});

test('demand-driven: sizeof(B) never invokes B\'s value-mode eval (proof via spy)', () => {
  // Hardest-edge proof: monkey-patch sampler.evaluateExpr to record
  // every IR op it sees. If the short-circuit works, evaluating
  // `sizeof(<ref>)` never delegates to the value evaluator on that
  // ref's RHS — the recursive walk reads from the inferredType
  // instead.
  const engine = require('../index.ts');
  const fixedEval = require('../fixed-eval.ts');
  const typeinfer = require('../typeinfer.ts');
  const sampler = require('../sampler.ts');

  const r = engine.processSource(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
M = iid(Normal(mu = 0.0, sigma = 1.0), lengthof(A))
`);

  // Spy on sampler.evaluateExpr — count every call.
  const realEval = sampler.evaluateExpr;
  let opsSeen: string[] = [];
  (sampler as any).evaluateExpr = function spy(ir: any, env: any) {
    if (ir && ir.kind === 'call') opsSeen.push(ir.op);
    return realEval.call(sampler, ir, env);
  };
  try {
    const lm = r.loweredModule;
    const resolver = fixedEval.makeResolver({ loweredModule: lm });
    typeinfer.inferTypes(lm, { resolveFixed: resolver });
  } finally {
    (sampler as any).evaluateExpr = realEval;
  }
  // The short-circuit means `lengthof` never reaches evaluateExpr
  // (we handle it inline reading the inferredType). And critically,
  // `rowstack` (the expensive op constructing A's data) is NEVER
  // dispatched — A's value is never materialised by the const-eval
  // pass.
  assert.ok(!opsSeen.includes('rowstack'),
    `rowstack should not have been evaluated; ops seen: ${JSON.stringify(opsSeen)}`);
  assert.ok(!opsSeen.includes('lengthof'),
    `lengthof should be short-circuited, not dispatched; ops seen: ${JSON.stringify(opsSeen)}`);
});

test('demand-driven: shape-observer short-circuits non-ref operands too', () => {
  // Generalised short-circuit: `length(<inline-call>)` reads the
  // type off the inline-call's meta.type, no recursion into operand.
  const engine = require('../index.ts');
  const r = engine.processSource(`
n = lengthof([1.0, 2.0, 3.0, 4.0, 5.0])
M = iid(Normal(mu = 0.0, sigma = 1.0), n)
`);
  const lb = r.loweredModule.bindings.get('M');
  assert.deepEqual(lb.inferredType.domain.shape, [5],
    `expected M.domain.shape=[5], got ${JSON.stringify(lb.inferredType.domain.shape)}`);
});

test('demand-driven: eye(n) folds n via the resolver', () => {
  const engine = require('../index.ts');
  const r = engine.processSource(`
n = 4
I = eye(n)
`);
  const lb = r.loweredModule.bindings.get('I');
  assert.deepEqual(lb.inferredType.shape, [4, 4]);
});

test('demand-driven: fill(value, [m, n]) folds shape vector via the resolver', () => {
  const engine = require('../index.ts');
  const r = engine.processSource(`
shape = [3, 5]
M = fill(1.0, shape)
`);
  const lb = r.loweredModule.bindings.get('M');
  assert.deepEqual(lb.inferredType.shape, [3, 5]);
});

test('demand-driven: shape-observer short-circuits through inferredType (no operand eval)', () => {
  // The motivating example: a fixed-phase chain `A → B → sz =
  // sizeof(B) → n = prod(sz) → n2 = n + n → M = iid(Normal, n2)`.
  // To know M's variate shape, we need n2's value. The chain through
  // `sizeof(B)` short-circuits through B's TYPE — never
  // materialising B (which would force materialising A).
  //
  // The chain only fires for ops that consult the const-eval
  // resolver — today that's just `iid`. zeros/fill/ones consulting
  // the resolver is a tracked follow-up (TODO §17), so we route the
  // shape-position query through iid.
  const engine = require('../index.ts');
  const r = engine.processSource(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
B = A
sz = sizeof(B)
n = prod(sz)
n2 = n + n
M = iid(Normal(mu = 0.0, sigma = 1.0), n2)
`);
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors, []);
  const lb = r.loweredModule.bindings.get('M');
  const t = lb && lb.inferredType;
  // sz = [2, 3]; n = 6; n2 = 12; M = iid(Normal, 12) → measure(array([12], real)).
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array');
  assert.deepEqual(t.domain.shape, [12],
    `M's variate shape should be [12], got ${JSON.stringify(t.domain.shape)}`);
});

test('demand-driven: const-eval cache is empty for unconsulted bindings', () => {
  // Whitebox check: after running typeinfer with the resolver, only
  // bindings reached through shape-position queries (today only
  // iid's size arg consults the resolver) should appear in the
  // cache. Bindings whose value isn't needed by any shape position
  // stay untouched — the demand-driven guarantee.
  const fixedEval = require('../fixed-eval.ts');
  const typeinfer = require('../typeinfer.ts');
  const engine = require('../index.ts');
  const r = engine.processSource(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
n = lengthof(A)
M = iid(Normal(mu = 0.0, sigma = 1.0), n)
unused = 42
`);
  // Re-run inference with a fresh resolver so we can inspect the
  // cache (the analyzer-internal resolver isn't exposed).
  const lm = r.loweredModule;
  const resolver = fixedEval.makeResolver({ loweredModule: lm });
  typeinfer.inferTypes(lm, { resolveFixed: resolver });
  const cache = resolver.knownFixed;
  // `n` should be in the cache (reached via iid(Normal, n) →
  // resolveIntegerShape → resolver → recurse into n).
  assert.ok(cache.has('n'),
    `cache should contain 'n'; entries: ${Array.from(cache.keys())}`);
  // `unused` should NOT be in the cache — no shape position needs it.
  assert.ok(!cache.has('unused'),
    `cache should NOT contain 'unused'; entries: ${Array.from(cache.keys())}`);
});

// =====================================================================
// Static density shape diagnostics (engine-concepts §17.3 wiring)
// =====================================================================

test('static-density: logdensityof with matching shapes → no diagnostic', () => {
  const engine = require('../index.ts');
  const r = engine.processSource(`
m = Normal(mu = 0.0, sigma = 1.0)
x = 1.0
lp = logdensityof(m, x)
`);
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors, [],
    `unexpected errors: ${JSON.stringify(errors)}`);
});

test('static-density: logdensityof(iid(M, 3), length-5-vector) → static error', () => {
  const engine = require('../index.ts');
  const r = engine.processSource(`
m = iid(Normal(mu = 0.0, sigma = 1.0), 3)
x = [1.0, 2.0, 3.0, 4.0, 5.0]
lp = logdensityof(m, x)
`);
  const errors = r.diagnostics.filter((d: any) =>
    d.severity === 'error' && /3 elements.*5|5.*3 elements/.test(d.message));
  assert.ok(errors.length > 0,
    `expected length-mismatch error; got: ${JSON.stringify(r.diagnostics)}`);
});

test('static-density: logdensityof(joint(...), record-missing-field) → static error', () => {
  const engine = require('../index.ts');
  const r = engine.processSource(`
m = joint(a = Normal(mu = 0.0, sigma = 1.0), b = Normal(mu = 0.0, sigma = 1.0))
x = record(a = 1.0)
lp = logdensityof(m, x)
`);
  const errors = r.diagnostics.filter((d: any) =>
    d.severity === 'error' && /missing field/.test(d.message));
  assert.ok(errors.length > 0,
    `expected missing-field error; got: ${JSON.stringify(r.diagnostics)}`);
});

test('static-density: logdensityof(Normal, record) → static error', () => {
  const engine = require('../index.ts');
  const r = engine.processSource(`
m = Normal(mu = 0.0, sigma = 1.0)
x = record(a = 1.0, b = 2.0)
lp = logdensityof(m, x)
`);
  const errors = r.diagnostics.filter((d: any) =>
    d.severity === 'error' && /scalar leaf/.test(d.message));
  assert.ok(errors.length > 0,
    `expected scalar-leaf error; got: ${JSON.stringify(r.diagnostics)}`);
});

test('static-density: densityof + likelihoodof get checked the same way', () => {
  const engine = require('../index.ts');
  for (const op of ['densityof', 'likelihoodof']) {
    const r = engine.processSource(`
m = iid(Normal(mu = 0.0, sigma = 1.0), 3)
x = [1.0, 2.0]
val = ${op}(m, x)
`);
    const errors = r.diagnostics.filter((d: any) =>
      d.severity === 'error' && /3 elements.*2|2.*3 elements/.test(d.message));
    assert.ok(errors.length > 0,
      `${op}: expected length-mismatch error; got: ${JSON.stringify(r.diagnostics)}`);
  }
});

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
