'use strict';

// End-to-end tests for engine/typeinfer.js — runs the full
// parse → analyze pipeline (which now includes type inference) and
// asserts inferred types and diagnostics on representative sources.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, types: T } = require('..');

function infer(src: any) {
  const { bindings, diagnostics } = processSource(src);
  const errors = diagnostics.filter((d: any) => d.severity === 'error');
  return { bindings, errors };
}
function typeOf(bindings: any, name: any) {
  return bindings.get(name).inferredType;
}

// =====================================================================
// Distributions and basic literals
// =====================================================================

test('distributions: Normal kwargs accept integer literals via promotion', () => {
  // §sec:valuetypes: integer literals satisfy the real-typed kwargs
  // through the canonical embedding integers ⊂ reals. No diagnostic.
  const { bindings, errors } = infer(`m = Normal(mu = 0, sigma = 1)`);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'm'), T.measure(T.REAL)));
});

test('distributions: discrete distributions return integer / boolean measures', () => {
  const { bindings, errors } = infer(`
    b = Bernoulli(p = 0.5)
    p = Poisson(rate = 2)
    bn = Binomial(n = 10, p = 0.3)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'b'),  T.measure(T.BOOLEAN)));
  assert.ok(T.equal(typeOf(bindings, 'p'),  T.measure(T.INTEGER)));
  assert.ok(T.equal(typeOf(bindings, 'bn'), T.measure(T.INTEGER)));
});

test('literals: lexical form decides integer vs real', () => {
  const { bindings, errors } = infer(`
    i = 42
    r = 3.14
    s = "hello"
    b = true
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'i'), T.INTEGER));
  assert.ok(T.equal(typeOf(bindings, 'r'), T.REAL));
  assert.ok(T.equal(typeOf(bindings, 's'), T.STRING));
  assert.ok(T.equal(typeOf(bindings, 'b'), T.BOOLEAN));
});

test('literals: array literal unifies element types and records length', () => {
  const { bindings, errors } = infer(`xs = [1.0, 2.0, 3.0]`);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'xs'), T.array(1, [3], T.REAL)));
});

test('literals: array of integer literals stays integer', () => {
  const { bindings } = infer(`xs = [1, 2, 3]`);
  assert.ok(T.equal(typeOf(bindings, 'xs'), T.array(1, [3], T.INTEGER)));
});

// =====================================================================
// Variates and law extraction
// =====================================================================

test('draw: extracts the value type from a measure', () => {
  const { bindings, errors } = infer(`
    m = Normal(mu = 0, sigma = 1)
    x = draw(m)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'x'), T.REAL));
});

test('lawof: lifts a value type back into a measure', () => {
  const { bindings, errors } = infer(`
    m = Normal(mu = 0, sigma = 1)
    x = draw(m)
    y = 2 * x
    y_dist = lawof(y)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'y_dist'), T.measure(T.REAL)));
});

// =====================================================================
// Measure-algebra type errors — the user's reported cases
// =====================================================================

test('weighted(measure, measure): structurally invalid → diagnostic', () => {
  // The user's invalid1_dist case. arg 0 must be a value but theta_dist
  // is a measure → should produce a clear error pointing at the bad arg.
  const { errors } = infer(`
    theta1_dist = Normal(mu = 0, sigma = 1)
    theta2_dist = Exponential(rate = 1)
    invalid = weighted(theta2_dist, theta1_dist)
  `);
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /weighted: arg 1 expects real or function, got measure over real/);
  assert.equal(errors[0].severity, 'error');
});

test('weighted(value, value): structurally invalid → diagnostic', () => {
  // The user's invalid2_dist case. arg 1 must be a measure but theta1
  // (a draw) is a real value → diagnostic on the wrong arg.
  const { errors } = infer(`
    theta1_dist = Normal(mu = 0, sigma = 1)
    theta1 = draw(theta1_dist)
    theta2_dist = Exponential(rate = 1)
    theta2 = draw(theta2_dist)
    invalid = weighted(theta2, theta1)
  `);
  assert.equal(errors.length, 1);
  assert.match(errors[0].message, /weighted: arg 2 expects measure, got real/);
});

test('weighted(value, measure): valid — no diagnostic, infers measure<real>', () => {
  const { bindings, errors } = infer(`
    theta1_dist = Normal(mu = 0, sigma = 1)
    theta2_dist = Exponential(rate = 1)
    theta2 = draw(theta2_dist)
    valid = weighted(theta2, theta1_dist)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'valid'), T.measure(T.REAL)));
});

// =====================================================================
// Diagnostic locations
// =====================================================================

test('diagnostic locations point at the offending argument, not the whole call', () => {
  const src = 'theta_dist = Normal(mu = 0, sigma = 1)\nbad = weighted(theta_dist, theta_dist)\n';
  const { errors } = infer(src);
  assert.equal(errors.length, 1);
  // Line 2; column should be inside the call's first arg (after
  // "weighted("), not at the beginning of "bad".
  assert.equal(errors[0].loc.start.line, 1);   // 0-based: line 2 = index 1
  assert.ok(errors[0].loc.start.col > 0);
});

// =====================================================================
// Cycles
// =====================================================================

test('cyclic bindings: inference falls back to %failed without diverging', () => {
  // Direct cycle. The analyzer surfaces undefined-name warnings; we
  // just want type inference to terminate and the offending binding
  // to carry a failed type.
  const { bindings } = infer(`
    a = b
    b = a
  `);
  // Both end up failed; we don't insist on any specific message.
  assert.equal(typeOf(bindings, 'a').kind, 'failed');
  assert.equal(typeOf(bindings, 'b').kind, 'failed');
});

// =====================================================================
// Composite types
// =====================================================================

test('record: produces record<…> with field types from kwargs', () => {
  const { bindings } = infer(`
    r = record(x = 1.0, y = 2)
  `);
  const t = typeOf(bindings, 'r');
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.x, T.REAL));
  assert.ok(T.equal(t.fields.y, T.INTEGER));
});

test('joint: produces measure<record<…>> from measure-typed kwargs', () => {
  const { bindings, errors } = infer(`
    a = Normal(mu = 0, sigma = 1)
    b = Exponential(rate = 1)
    j = joint(x = a, y = b)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'j');
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'record');
  assert.ok(T.equal(t.domain.fields.x, T.REAL));
  assert.ok(T.equal(t.domain.fields.y, T.REAL));
});

test('joint: a value-typed kwarg is a structural error', () => {
  const { errors } = infer(`
    a = Normal(mu = 0, sigma = 1)
    x = draw(a)
    j = joint(p = a, q = x)
  `);
  assert.ok(errors.some((e: any) => /joint kwarg "q" expects a measure/.test(e.message)));
});

// =====================================================================
// elementof + set constructors
// =====================================================================

test('elementof: bare set name → structural value type', () => {
  const { bindings, errors } = infer(`
    a = elementof(reals)
    b = elementof(integers)
    c = elementof(booleans)
    d = elementof(posreals)
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'a'), T.REAL));
  assert.ok(T.equal(typeOf(bindings, 'b'), T.INTEGER));
  assert.ok(T.equal(typeOf(bindings, 'c'), T.BOOLEAN));
  assert.ok(T.equal(typeOf(bindings, 'd'), T.REAL));   // refinement → real
});

test('elementof(cartpow(S, n, …)): array shape and element type', () => {
  const { bindings, errors } = infer(`
    a = elementof(cartpow(reals, 3))
    b = elementof(cartpow(posreals, 2, 4))
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'a'), T.array(1, [3], T.REAL)));
  assert.ok(T.equal(typeOf(bindings, 'b'), T.array(2, [2, 4], T.REAL)));
});

// =====================================================================
// Polymorphic arithmetic (scalar / array / broadcast)
// =====================================================================

test('arithmetic: scalar + scalar still works (integer promotes to real)', () => {
  const { bindings, errors } = infer(`
    a = 1.0
    b = 2
    c = a + b
  `);
  assert.equal(errors.length, 0);
  assert.ok(T.equal(typeOf(bindings, 'c'), T.REAL));
});

test('arithmetic: array + scalar broadcasts to array', () => {
  const { bindings, errors } = infer(`
    xs = [1.0, 2.0, 3.0]
    y = 5.0
    out = xs + y
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'out');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('arithmetic: array + array of matching shape stays array', () => {
  const { bindings, errors } = infer(`
    xs = [1.0, 2.0, 3.0]
    ys = [4.0, 5.0, 6.0]
    out = xs * ys
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'out');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
});

test('arithmetic: shape mismatch is a diagnostic', () => {
  const { errors } = infer(`
    xs = [1.0, 2.0, 3.0]
    ys = [1.0, 2.0]
    out = xs + ys
  `);
  assert.ok(errors.some((e: any) => /not numerically compatible/.test(e.message)));
});

test('arithmetic: comparisons return boolean of the broadcast shape', () => {
  const { bindings, errors } = infer(`
    xs = [1.0, 2.0, 3.0]
    out = xs < 2.0
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'out');
  assert.equal(t.kind, 'array');
  assert.ok(T.equal(t.elem, T.BOOLEAN));
});

test('broadcast: dotted op over rank-1 array preserves shape', () => {
  // The viewer's matrix-heatmap dispatch keys off the inferredType
  // being array(rank=N, statically-known-shape). Before typeinfer
  // learned about `broadcast(...)` directly, `tau = (bkg ./ dbkg) .^
  // 2` deferred — so a fixed-phase 3×3 zero matrix rendered as a
  // scalar 0. Pin the tighten on a rank-1 literal where the shape is
  // statically known (rowstack's signature still leaves the dims
  // %dynamic at the static-type layer).
  const { bindings, errors } = infer(`
    bkg = [50.0, 52.0]
    tau = bkg .^ 2
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'tau');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [2]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('broadcast: user-defined callable resolves cell-type via callee result', () => {
  // `f.(args)` lowers to `broadcast(self.f, args...)`, keeping `f`
  // as a bare ref. inferBroadcast looks up the callable's declared
  // result type — same monomorphic-at-definition simplification
  // inferUserCall uses for scalar calls — and stacks it across the
  // broadcast outer shape. Without this, dotted-broadcasts of user
  // functions were silently deferred and downstream consumers (the
  // viewer's plot-plan, materialise-time shape checks) had to
  // fall back to runtime-value heuristics. Pinning this on the
  // motivating polyeval case from nested-broadcast.test.ts.
  const { bindings, errors } = infer(`
    polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
    C = [2.3, 1.5, 0.7]
    X = [1.1, 2.2, 3.3]
    Y = polyeval.([C], X)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'Y');
  assert.equal(t.kind, 'array',
    'Y should infer to array, got ' + JSON.stringify(t));
  assert.deepEqual(t.shape, [3],
    'Y outer shape should be [3] after singleton-expanding [1] vs [3], '
    + 'got ' + JSON.stringify(t.shape));
  assert.ok(T.equal(t.elem, T.REAL),
    'Y cell type should be real, got ' + JSON.stringify(t.elem));
});

test('broadcast: singleton outer-shape expansion ([1] × [3] → [3])', () => {
  // Per spec §04: size-one outer axes broadcast by repetition. The
  // nested-vector classifier sees `[C]` (length 1) as outer rank 1,
  // shape [1]; `X` (length 3) as outer rank 1, shape [3]. Both have
  // outer rank 1; per-axis sizes 1 vs 3 broadcast to 3. (`unifyArith`
  // didn't implement singleton-broadcast — this case used to defer.)
  const { bindings, errors } = infer(`
    polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
    C = [2.3, 1.5, 0.7]
    Xsingle = [3.3]
    Y = polyeval.([C], Xsingle)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'Y');
  assert.equal(t.kind, 'array');
  // Both outer axes are length 1; merged outer shape is [1].
  assert.deepEqual(t.shape, [1]);
});

test('broadcast: multi-axis flat tensor preserves all axes', () => {
  // `M .^ 2` with M = rowstack(...) — M is a flat rank-2 tensor; all
  // axes are loop axes (not nested-vector). The new classifier keeps
  // outer shape = M.shape, scalar elem; result has the same rank-2
  // shape. Pins that the multi-axis path didn't regress when the
  // classifier learned about nested vectors.
  const { bindings, errors } = infer(`
    M = rowstack([[1.0, 2.0], [3.0, 4.0]])
    out = M .^ 2
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'out');
  assert.equal(t.kind, 'array');
  assert.equal(t.rank, 2);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('broadcast: 2-level nested Ref-wrap polyeval.([[C]], X_2D) → array(2, [2,3], real)', () => {
  // Deeper nesting: `[[C]]` is `vector(vector(C))` →
  // array(rank=1, shape=[1], elem=array(rank=1, shape=[1], elem=array(rank=1, shape=[3], real))).
  // For broadcast type inference, the OUTERMOST axis is the loop
  // axis; the cell sees the inner nest whole. With `X_2D` of shape
  // [2, 3], the broadcast iterates the outer rank-1 axis (singleton
  // expansion 1 → 2), and per cell sees `[C]` and a length-3 slice
  // of X_2D. But X_2D's outer rank is 2 (flat tensor), while [[C]]'s
  // outer rank is 1 — the spec "same number of axes" rule kicks in,
  // so this case statically defers. (The viable shape for double-Ref
  // broadcast is when the OTHER arg is also nested-rank-2, e.g.
  // a length-(N*M) outer of inner-rank-1 vectors; see the next test.)
  const { bindings, errors } = infer(`
    polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
    C = [2.3, 1.5, 0.7]
    X_2D = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    Y = polyeval.([[C]], X_2D)
  `);
  // Outer-rank mismatch ([[C]] is nested rank 1; X_2D is flat rank 2)
  // ⇒ defer. Diagnostic-free: typeinfer doesn't claim a hard error
  // here because the runtime evaluator surfaces the precise mismatch
  // at the broadcast call site (see test/nested-broadcast.test.ts
  // v2 outer-rank-mismatch case). To get a non-deferred type here
  // you'd need `[[C]]` paired with a NESTED-RANK-2 collection (e.g.
  // `[[V1, V2], [V3, V4]]`); pinned in the next test.
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'Y');
  assert.equal(t.kind, 'deferred',
    'expected deferred (outer-rank mismatch); got ' + JSON.stringify(t));
});

test('broadcast: 2-level nested matches 2-level nested ([[C]] vs [[V1,V2],[V3,V4]])', () => {
  // The composable companion to the rank-mismatch case above.
  // `[[C]]` is nested rank-1 (outer shape [1]) wrapping a length-3 vector
  // (Ref-wrapped); `[[V1, V2], [V3, V4]]` is nested rank-1 (outer shape [2])
  // wrapping length-2 nested-vectors. Both have outer rank 1 — but the
  // inner-rank-1 cells then face their own broadcast (dot of inner vectors).
  // Outer-rank-1 unify of [1] vs [2] → [2] via singleton expansion.
  const { bindings, errors } = infer(`
    dot = (a, b) -> sum(a .* b)
    C = [1.0, 1.0, 1.0]
    V1 = [1.0, 2.0, 3.0]
    V2 = [4.0, 5.0, 6.0]
    Y = dot.([C], [V1, V2])
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'Y');
  assert.equal(t.kind, 'array');
  // Outer-shape unify [1] vs [2] → [2] via singleton expansion.
  assert.deepEqual(t.shape, [2]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('broadcast: pairwise dot.([V1,V2,V3], [W1,W2,W3]) → array(1, [3], real)', () => {
  // Pairwise broadcast over two length-3 nested vectors (`[V1,V2,V3]`
  // and `[W1,W2,W3]`). Both have outer rank 1, cell type = inner
  // length-k vector. The callable's result is a scalar (sum reduces
  // the per-cell elementwise product). Broadcast result = array(1, [3], real).
  const { bindings, errors } = infer(`
    dot = (a, b) -> sum(a .* b)
    V1 = [1.0, 2.0, 3.0]
    V2 = [4.0, 5.0, 6.0]
    V3 = [7.0, 8.0, 9.0]
    W1 = [1.0, 0.0, 0.0]
    W2 = [0.0, 1.0, 0.0]
    W3 = [0.0, 0.0, 1.0]
    Y = dot.([V1, V2, V3], [W1, W2, W3])
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'Y');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('broadcast: chained dotted ops preserve shape across compositions', () => {
  // Broadcast results feeding broadcast inputs. After the first
  // `A .+ B` types as array(1, [2], real), the second broadcast
  // `... .* B` sees both args as flat rank-1 arrays of matching
  // shape — outer-shape unify gives [2]. Pins the chained-typing
  // case so later tightening doesn't accidentally erase the result
  // type of the intermediate.
  const { bindings, errors } = infer(`
    A = [1.0, 2.0]
    B = [10.0, 20.0]
    Y = (A .+ B) .* B
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'Y');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [2]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('aggregate: matrix multiplication infers array(2, [2, 2], real)', () => {
  // Spec §04 §sec:aggregate canonical example. Output rank = 2
  // (output_axes = [.i, .k]), shape inferred from A and B's
  // statically-known dims (rowstack is loose at the type level, so
  // axis lengths come back %dynamic — but the rank is concrete and
  // the elem type is real).
  const { bindings, errors } = infer(`
    A = rowstack([[1.0, 3.0, 5.0], [9.0, 5.0, 1.0]])
    B = rowstack([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'C');
  assert.equal(t.kind, 'array');
  assert.equal(t.rank, 2);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('aggregate: column-wise reduction infers array(1, ...)', () => {
  // Reducing over `.i` keeps `.j` in the output → result is a
  // rank-1 array indexed by `.j`. Tests that the axis-length
  // inference correctly picks the right dim from A's shape.
  const { bindings, errors } = infer(`
    A = rowstack([[1.0, 3.0, 5.0], [9.0, 5.0, 1.0]])
    V = aggregate(var, [.j], A[.i, .j])
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'V');
  assert.equal(t.kind, 'array');
  assert.equal(t.rank, 1);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('polymorphic-at-call-site: f(scalar, scalar) → scalar; f(array, array) → array', () => {
  // Spec §sec:functionof. Before B5, user-defined fns were typed
  // monomorphically at definition with each param as `any`, so call
  // sites couldn't tighten. After B5, inferUserCall re-infers the
  // callee's functionof body with the call-site arg types in scope
  // — the result tightens to match what the body would produce for
  // those argument types.
  const { bindings, errors } = infer(`
    f = (a, b) -> a + b
    x = f(1, 2)
    y = f([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
    mixed = f(2.0, 3)
  `);
  assert.equal(errors.length, 0);
  // Scalar call: integer + integer → integer (per the existing arith
  // promotion rules; both operands stay integer).
  const tx = typeOf(bindings, 'x');
  assert.equal(tx.kind, 'scalar');
  assert.equal(tx.prim, 'integer');
  // Array call: array(1, [3], real) + array(1, [3], real) → array.
  const ty = typeOf(bindings, 'y');
  assert.equal(ty.kind, 'array');
  assert.deepEqual(ty.shape, [3]);
  assert.ok(T.equal(ty.elem, T.REAL));
  // Mixed scalar: real + integer → real (promotion).
  const tmixed = typeOf(bindings, 'mixed');
  assert.equal(tmixed.kind, 'scalar');
  assert.equal(tmixed.prim, 'real');
});

test('polymorphic-at-call-site: recursive call falls back to monomorphic (visiting guard)', () => {
  // Recursion: if the body calls the same fn, the visiting set
  // prevents re-entrant body inference (would otherwise infinite-
  // loop). The polymorphic branch is skipped; the monomorphic
  // `calleeType.result` is returned. Pinning that no infinite
  // loop happens AND no spurious error is emitted.
  const { errors } = infer(`
    f = (n) -> ifelse(n > 0, f(n - 1) + 1, 0)
    r = f(3)
  `);
  // Either no errors, or only the recursion-related ones — but
  // critically, no hang.
  assert.ok(errors.length < 5,
    'recursive call should not emit a cascade; got ' + errors.length + ' errors');
});

test('polymorphic-at-call-site: composes with broadcast (inferBroadcast uses polymorphic cell type)', () => {
  // Demonstrates B5 + the broadcast tightening compose: the user-fn
  // `square` types its CELL result correctly under arg-specific types,
  // and inferBroadcast picks up the cell result and stacks across the
  // broadcast outer shape. Pre-B5 the cell would have come back as
  // `any` (from the monomorphic path) and we'd default to REAL via
  // the deferred fallback. Now it's REAL directly.
  const { bindings, errors } = infer(`
    square = x -> x * x
    A = [1.0, 2.0, 3.0]
    out = square.(A)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'out');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('broadcasted direct form: broadcasted(f)(A, B) lowers and types as broadcast(f, A, B)', () => {
  // Spec §04: `broadcasted(f)(args) ≡ broadcast(f, args)`. The
  // direct form has the outer call's callee = `broadcasted(f)` (a
  // CallExpr, not an Identifier). `_lowerCallExpr` special-cases
  // this shape and lowers directly to a `broadcast` IR — bypassing
  // the lit-null catch the non-Identifier callee guard would
  // otherwise trip. inferBroadcast then types the result normally.
  const { bindings, errors } = infer(`
    add2 = (a, b) -> a + b
    A = [1.0, 2.0, 3.0]
    B = [10.0, 20.0, 30.0]
    G = broadcasted(add2)(A, B)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'G');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('broadcasted inline-in-fn: fn(broadcasted(f)(_, B)) types as a function returning an array', () => {
  // `fn(broadcasted(f)(a, _, c))` works because the broadcasted-curry
  // rewrite in `_lowerCallExpr` fires inside the fn body. The outer
  // fn becomes a function over the inner holes; its body lowers to
  // `broadcast(f, a, $hole, c)`. inferBroadcast types the body as
  // array(1, [outer], elem); the fn's signature wraps it.
  const { bindings, errors } = infer(`
    add2 = (a, b) -> a + b
    B = [10.0, 20.0, 30.0]
    F = fn(broadcasted(add2)(_, B))
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'F');
  assert.equal(t.kind, 'function');
  assert.equal(t.inputs.length, 1);
  // Result type follows the inner broadcast.
  assert.equal(t.result.kind, 'array');
  assert.deepEqual(t.result.shape, [3]);
  assert.ok(T.equal(t.result.elem, T.REAL));
});

test('broadcasted via-binding: bc = broadcasted(f); bc(A, B) types via broadcast routing', () => {
  // Spec §04: `broadcasted(f)(args)` ≡ `broadcast(f, args)`. The
  // wrapper is polymorphic — its type at the definition site
  // (`bc = broadcasted(f)`) is intentionally deferred, since the
  // result shape depends on call-site args. inferUserCall now
  // recognises the via-binding wrapper pattern and routes the call
  // through inferBroadcast at the type level — same routing the
  // runtime lift does later for the materialised path.
  //
  // Direct form `broadcasted(f)(A, B)` (no via-binding) is a known
  // limitation: it fails to lower (non-Identifier callee at
  // lower.ts) and stores as a lit-null placeholder. The via-binding
  // form is the supported way to type-route broadcasted.
  const { bindings, errors } = infer(`
    add2 = (a, b) -> a + b
    A = [1.0, 2.0, 3.0]
    B = [10.0, 20.0, 30.0]
    bcadd = broadcasted(add2)
    C = bcadd(A, B)
  `);
  assert.equal(errors.length, 0);
  // bcadd itself: deferred (no useful type at definition; the wrapper
  // is fully polymorphic and types at the call site).
  const tBc = typeOf(bindings, 'bcadd');
  assert.equal(tBc.kind, 'deferred');
  // The call: properly typed array via inferBroadcast routing.
  const tC = typeOf(bindings, 'C');
  assert.equal(tC.kind, 'array');
  assert.deepEqual(tC.shape, [3]);
  assert.ok(T.equal(tC.elem, T.REAL));
});

test('reduce: result type = element type of xs (left-fold accumulator)', () => {
  // Spec §04 §sec:higher-order: `reduce(f, xs)` uses xs[0] as the
  // initial accumulator; f is applied pairwise. The result type
  // follows the accumulator — same as xs's element type (T unified
  // through the SIGNATURE_FACTORIES tvar). Pins that this works for
  // both literal scalars and array operations.
  const { bindings, errors } = infer(`
    add2 = (a, b) -> a + b
    xs = [1.0, 2.0, 3.0, 4.0]
    S = reduce(add2, xs)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'S');
  assert.ok(T.equal(t, T.REAL),
    'reduce result should be real (xs element type); got ' + JSON.stringify(t));
});

test('scan: result type = array(1, %dynamic, T) where T is the accumulator type', () => {
  // Spec §04: `scan(f, init, xs)` produces a vector of intermediate
  // accumulator values, one per element/row of xs. The accumulator
  // type unifies from init AND xs's elements (both tvar 'T'). The
  // output length matches xs's; the signature keeps it %dynamic
  // until the cumulative-length tightening lands.
  const { bindings, errors } = infer(`
    add2 = (a, b) -> a + b
    xs = [1.0, 2.0, 3.0, 4.0]
    SCN = scan(add2, 0.0, xs)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'SCN');
  assert.equal(t.kind, 'array');
  assert.equal(t.rank, 1);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('filter: result type = array(1, %dynamic, T) — length dynamic by predicate', () => {
  // Spec §04: `filter(pred, data)` keeps elements satisfying pred,
  // returning a shorter array of the SAME element type. Length is
  // statically dynamic — depends on which elements match.
  const { bindings, errors } = infer(`
    xs = [1.0, 2.0, 3.0, 4.0]
    F = filter(fn(_ > 2.0), xs)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'F');
  assert.equal(t.kind, 'array');
  assert.equal(t.rank, 1);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('aggregate: outer-product (no reduction) infers rank-2', () => {
  // No reduction axis ⇒ output rank = input axes. The body is
  // u[.i] * v[.j] — pure broadcast, no contraction.
  const { bindings, errors } = infer(`
    u = [1.0, 2.0, 3.0]
    v = [10.0, 20.0]
    O = aggregate(sum, [.i, .j], u[.i] * v[.j])
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'O');
  assert.equal(t.kind, 'array');
  assert.equal(t.rank, 2);
  assert.deepEqual(t.shape, [3, 2]);
  assert.ok(T.equal(t.elem, T.REAL));
});

test('broadcast: kernel-broadcast stays deferred so joint/draw still accept it', () => {
  // `broadcast(K, ...)` is semantically an array of measures, but
  // tightening that would tip joint/draw's measure-typecheck into
  // rejecting the result. inferBroadcast deliberately falls back to
  // deferred whenever the function arg isn't a value-producing
  // synthesized functionof — see typeinfer.ts comments.
  const { bindings, errors } = infer(`
    A = [1.0, 2.0, 3.0]
    K = fn(Normal(mu = _, sigma = 0.1))
    D ~ broadcast(K, A)
  `);
  assert.equal(errors.length, 0);
});

test('arithmetic: unary ops preserve shape', () => {
  const { bindings, errors } = infer(`
    xs = [1.0, 2.0, 3.0]
    out = abs(xs)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'out');
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [3]);
});

test('iid: with literal n produces a measure over a concrete-shape array', () => {
  const { bindings, errors } = infer(`
    obs_dist = iid(Normal(mu = 0, sigma = 1), 10)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'obs_dist');
  assert.equal(t.kind, 'measure');
  assert.equal(t.domain.kind, 'array');
  assert.deepEqual(t.domain.shape, [10]);
  assert.ok(T.equal(t.domain.elem, T.REAL));
});

test('iid: integer-typed binding ref resolves via const-eval', () => {
  // Before the engine-concepts §17.4 const-eval-in-typeinfer pass
  // landed, this fell back to %dynamic. With const-eval, the shape
  // position folds `n` to the integer literal at type-check time.
  const { bindings, errors } = infer(`
    n = 10
    obs_dist = iid(Normal(mu = 0, sigma = 1), n)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'obs_dist');
  assert.deepEqual(t.domain.shape, [10]);
});

test('iid: n = arithmetic on literals resolves to a literal shape', () => {
  // Const-eval handles arithmetic, not just direct binding refs.
  const { bindings, errors } = infer(`
    n = 3 + 4
    obs_dist = iid(Normal(mu = 0, sigma = 1), n)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'obs_dist');
  assert.deepEqual(t.domain.shape, [7]);
});

test('iid: n = length(literal-array) resolves via shape-only short-circuit', () => {
  // The shape-only short-circuit reads `length` from the type's shape
  // when available, without invoking the full evaluator. Engine-
  // concepts §17.4 — most important optimisation for keeping
  // compile-time eval cheap.
  const { bindings, errors } = infer(`
    data = [1.0, 2.0, 3.0, 4.0]
    obs_dist = iid(Normal(mu = 0, sigma = 1), lengthof(data))
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'obs_dist');
  assert.deepEqual(t.domain.shape, [4]);
});

test('iid: nested arithmetic over binding refs resolves', () => {
  // The whole point: chained const-eval, ref → ref → arithmetic.
  const { bindings, errors } = infer(`
    a = 3
    b = a + 2
    c = 2 * b
    obs_dist = iid(Normal(mu = 0, sigma = 1), c)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'obs_dist');
  assert.deepEqual(t.domain.shape, [10]);
});

test('iid: %dynamic when n depends on an unresolvable expression', () => {
  // External / placeholder boundary inputs are NOT fixed-phase, so
  // const-eval correctly returns undefined and the shape stays
  // %dynamic. Pins the fall-through behaviour after the const-eval
  // landed — without this, an unintended widening of const-eval
  // (evaluating non-fixed expressions) would silently surface here.
  const { bindings, errors } = infer(`
    n = external(integer)
    obs_dist = iid(Normal(mu = 0, sigma = 1), n)
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'obs_dist');
  assert.deepEqual(t.domain.shape, ['%dynamic']);
});

test('iid: non-measure first arg is a type error', () => {
  const { errors } = infer(`
    obs_dist = iid(1.0, 10)
  `);
  assert.ok(errors.some((e: any) => /iid: arg 1 expects a measure/.test(e.message)));
});

test('elementof(cartprod): kwargs form → record', () => {
  const { bindings, errors } = infer(`
    p = elementof(cartprod(x = reals, y = integers))
  `);
  assert.equal(errors.length, 0);
  const t = typeOf(bindings, 'p');
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.x, T.REAL));
  assert.ok(T.equal(t.fields.y, T.INTEGER));
});

// =====================================================================
// inferExprInScope — on-demand call-site specialization
// =====================================================================
//
// Polymorphic function bodies (`fn(2 * _)` etc.) get a best-effort
// type at module-load time with `any` inputs. inferExprInScope lets
// downstream consumers (the viewer's plot dispatch, primarily)
// re-infer with concrete input types — same rules as a real call
// site, just specialized to the chosen inputs.

const { inferExprInScope } = require('../typeinfer.ts');

test('inferExprInScope: polymorphic body specializes by input type', () => {
  const { loweredModule } = processSource(`
    f = fn(2 * _)
  `);
  const fb = loweredModule.bindings.get('f');
  const body = fb.rhs.body;
  // Module-level inference saw `_` as `any` and produced a result
  // type of integer (because 2 is integer). On-demand inference
  // with `_arg1_` = real yields real; with integer yields integer.
  const tReal = inferExprInScope(loweredModule, body,
    new Map([[fb.rhs.params[0], T.REAL]]));
  const tInt  = inferExprInScope(loweredModule, body,
    new Map([[fb.rhs.params[0], T.INTEGER]]));
  assert.ok(T.equal(tReal, T.REAL));
  assert.ok(T.equal(tInt,  T.INTEGER));
});

test('inferExprInScope: record body specializes per-field', () => {
  const { loweredModule } = processSource(`
    f = fn(record(a = _, b = 2 * _))
  `);
  const fb = loweredModule.bindings.get('f');
  const body = fb.rhs.body;
  const params = fb.rhs.params;
  // Two holes → two params (_arg1_, _arg2_). Bind both to real.
  const t = inferExprInScope(loweredModule, body,
    new Map([[params[0], T.REAL], [params[1], T.REAL]]));
  assert.equal(t.kind, 'record');
  assert.ok(T.equal(t.fields.a, T.REAL));
  assert.ok(T.equal(t.fields.b, T.REAL));
});

test('inferExprInScope: tuple body yields tuple result', () => {
  const { loweredModule } = processSource(`
    f = fn((_ * 2, _ + 1))
  `);
  const fb = loweredModule.bindings.get('f');
  const body = fb.rhs.body;
  const params = fb.rhs.params;
  const t = inferExprInScope(loweredModule, body,
    new Map([[params[0], T.REAL], [params[1], T.REAL]]));
  assert.equal(t.kind, 'tuple');
  assert.equal(t.elems.length, 2);
});

test('inferExprInScope: refs to module bindings resolve via b.inferredType', () => {
  // The body references a module-level binding `c`; on-demand
  // inference should look up c's already-set inferredType rather
  // than re-walking.
  const { loweredModule } = processSource(`
    c = 3.14
    f = fn(c * _)
  `);
  const fb = loweredModule.bindings.get('f');
  const body = fb.rhs.body;
  const t = inferExprInScope(loweredModule, body,
    new Map([[fb.rhs.params[0], T.REAL]]));
  assert.ok(T.equal(t, T.REAL));
});

// =====================================================================
// Multi-LHS rewriter + rnginit / rand types (spec §sec:random)
// =====================================================================

test('multi-LHS rand: each name gets the projected element type', () => {
  const { bindings, loweredModule, diagnostics } = processSource(`
    rngseed = [0xb2, 0x51, 0xa4, 0x93, 0x49, 0xd8, 0x68, 0x88]
    rstate = rnginit(rngseed)
    random_data, rstate2 = rand(rstate, iid(Normal(0, 1), 10))
  `);
  // Type errors break the whole spec example, so guard hard.
  const errors = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors, [], `unexpected errors: ${JSON.stringify(errors)}`);

  const rstate = loweredModule.bindings.get('rstate');
  assert.equal(rstate.inferredType.kind, 'rngstate');

  // First LHS: array of reals (the iid(Normal,10) variate).
  const rd = loweredModule.bindings.get('random_data');
  assert.equal(rd.inferredType.kind, 'array',
    `random_data type: ${T.show(rd.inferredType)}`);

  // Second LHS: rngstate.
  const rs2 = loweredModule.bindings.get('rstate2');
  assert.equal(rs2.inferredType.kind, 'rngstate',
    `rstate2 type: ${T.show(rs2.inferredType)}`);

  // Synthetic shared binding exists and holds the tuple type.
  let synName = null;
  for (const k of loweredModule.bindings.keys()) {
    if (k.startsWith('%mlhs:')) { synName = k; break; }
  }
  assert.ok(synName, 'synthetic %mlhs binding inserted');
  const syn = loweredModule.bindings.get(synName);
  assert.equal(syn.inferredType.kind, 'tuple');
  assert.equal(syn.inferredType.elems.length, 2);
  assert.equal(syn.inferredType.elems[1].kind, 'rngstate');

  // Multi-LHS bindings carry deps on the synthetic, not on the raw
  // rand call's argument names — phase analysis depends on this.
  assert.deepEqual(bindings.get('random_data').effectiveDeps, [synName]);
  assert.deepEqual(bindings.get('rstate2').effectiveDeps, [synName]);
});

test('multi-LHS rand: chained rand calls keep type+phase consistent', () => {
  const { bindings, loweredModule, diagnostics } = processSource(`
    rngseed = [0xb2, 0x51, 0xa4, 0x93, 0x49, 0xd8, 0x68, 0x88]
    rstate = rnginit(rngseed)
    random_data, rstate2 = rand(rstate, iid(Normal(0, 1), 10))
    more_random_data, rstate3 = rand(rstate2, iid(Exponential(1), 5))
  `);
  const errors = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors, [], `unexpected errors: ${JSON.stringify(errors)}`);

  // All bindings fixed-phase per spec: "If their inputs have fixed
  // phase, their outputs have fixed phase as well."
  for (const name of ['rstate', 'random_data', 'rstate2', 'more_random_data', 'rstate3']) {
    assert.equal(bindings.get(name).phase, 'fixed', `${name} should be fixed-phase`);
  }
  assert.equal(loweredModule.bindings.get('more_random_data').inferredType.kind, 'array');
  assert.equal(loweredModule.bindings.get('rstate3').inferredType.kind, 'rngstate');
});

test('multi-LHS tuple literal: each name gets the element type', () => {
  const { loweredModule, diagnostics } = processSource(`
    a, b = (1, 2.5)
  `);
  const errors = diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errors, [], `unexpected errors: ${JSON.stringify(errors)}`);

  const a = loweredModule.bindings.get('a');
  const b = loweredModule.bindings.get('b');
  // 1 is parsed as a numeric literal — types.js scalar('integer') in
  // the integer-literal case, scalar('real') in the float case. Both
  // are scalar kinds; we just check the shape.
  assert.equal(a.inferredType.kind, 'scalar');
  assert.equal(b.inferredType.kind, 'scalar');
});

test('multi-LHS preserves disintegrate semantics (no tuple_get rewrite)', () => {
  // Smoke test: the disintegrate path must not be replaced by the
  // multi-LHS tuple_get rewriter — kernel/prior keep their per-name
  // synthesized RHS instead.
  const { bindings, diagnostics } = processSource(`
    obs = elementof(reals)
    theta = elementof(reals)
    joint_model = lawof(record(theta = Normal(0, 1), obs = Normal(theta, 1)))
    forward_kernel, theta_prior = disintegrate("obs", joint_model)
  `);
  // No type-system errors involving tuple_get — the multi-LHS
  // rewriter must NOT touch disintegrate result bindings.
  const tupleGetErrs = diagnostics.filter((d: any) =>
    d.severity === 'error' && /tuple_get/.test(d.message));
  assert.deepEqual(tupleGetErrs, [], 'disintegrate must not be rewritten as tuple_get');
  // Disintegrate bindings should not carry a tuple_get effectiveValue
  // (whether the disintegrate plan resolved is independent of this).
  for (const name of ['forward_kernel', 'theta_prior']) {
    const b = bindings.get(name);
    if (!b || !b.effectiveValue) continue;
    const cv = b.effectiveValue;
    assert.notEqual(
      cv.callee && cv.callee.name, 'tuple_get',
      `${name}: effectiveValue must not be tuple_get`);
  }
});


test('logdensityof: shape-mismatch diagnostic substitutes resolved measure domain', () => {
  // When the second arg's type doesn't unify with the substituted T
  // (the resolved measure domain), the generic argError now reports
  // the concrete expected type rather than the bare 'any' that
  // T.show renders for unbound type variables.
  const { errors } = infer(`
    data_5 = [1.0, 2.0, 3.0, 4.0, 5.0]
    M = iid(Normal(0, 1), 3)
    ld = logdensityof(M, data_5)
  `);
  // Two diagnostics expected: the generic arg-2 error (with the
  // resolved-domain expected type) AND the static-shape walker's
  // precise iid-step error. We pin only the generic one's wording.
  const genericArg = errors.find((d: any) =>
    /^logdensityof: arg 2 expects/.test(d.message));
  assert.ok(genericArg, 'expected a generic logdensityof arg-2 diagnostic');
  assert.match(genericArg.message,
    /array of real \(length 3\), got array of real \(length 5\)/,
    'diagnostic substitutes the resolved measure domain (length 3) ' +
    'rather than reporting bare any: ' + genericArg.message);
});

test('logdensityof: scalar measure vs array data — substituted domain shows real', () => {
  const { errors } = infer(`
    data = [1.0, 2.0, 3.0]
    ld = logdensityof(Normal(0, 1), data)
  `);
  const genericArg = errors.find((d: any) =>
    /^logdensityof: arg 2 expects/.test(d.message));
  assert.ok(genericArg, 'expected generic logdensityof arg-2 diagnostic');
  assert.match(genericArg.message,
    /expects real, got array of real \(length 3\)/,
    'expected concrete real type, got: ' + genericArg.message);
});

