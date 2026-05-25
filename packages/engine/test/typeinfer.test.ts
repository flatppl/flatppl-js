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
