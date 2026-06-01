'use strict';

// =====================================================================
// metricsum.test.ts — pins the metric-aware Einstein summation surface
// =====================================================================
//
// Spec §04 §sec:metricsum: `metricsum(metric, output_axes, expr)` and
// its shorthand `metric: result[output_indices...] := expr` extend
// `aggregate(sum, ...)` with explicit upper/lower (contravariant/
// covariant) index tracking. The lift pass (lift.ts inlineMetricsumLift)
// rewrites every metricsum() call to a sum-`aggregate(...)` with
// metric / inv(metric) factor insertions per the spec lowering rule:
//
//   - Every `.X_` (lower) body axis becomes a fresh internal axis +
//     a `inv(metric)[.internal, .X]` factor (lowers a stored upper
//     component via the metric).
//   - Every `.X_` (lower) output axis becomes a fresh `.X_up` output
//     axis + a `metric[.X, .X_up]` factor (raises the body-computed
//     lower component back to all-upper canonical storage).
//
// Tests cover:
//   1. Surface forms (direct call vs shorthand, variance markers).
//   2. Parse-time static checks (paired upper/lower-twice, output
//      variance match, bare-neutral-axis ban, kwargs rejected, etc.).
//   3. Lift-pass IR shape (metricsum → aggregate(sum, ...) + synthetic
//      `__g_down` binding).
//   4. Numerical correctness vs hand-computed values for a Minkowski
//      (1+1)D metric — including the spec's Lorentz-composition
//      example.
//   5. Equivalence to plain aggregate under identity metric.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, orchestrator } = require('..');

function readFixture(name: string): string {
  return fs.readFileSync(path.join(__dirname, 'fixtures', name), 'utf-8');
}

function errors(src: string): any[] {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

function valueAt(v: any, ...idx: number[]): number {
  // Walk the engine's Value shape for both representations the
  // materialiser returns:
  //  - Flat Float64Array Value: { shape: [...], data: Float64Array }.
  //    For these we compute the row-major flat offset from the full
  //    index tuple in one shot.
  //  - Nested object-of-objects (legacy): `[{0: row0, 1: row1, …}, …]`
  //    or plain JS arrays. For these we descend one axis per index.
  if (v && typeof v === 'object' && v.data
      && typeof v.data.length === 'number' && Array.isArray(v.shape)) {
    // Flat Float64Array — compute row-major offset = sum(i_k * stride_k)
    // with strides = [prod(shape[k+1:]), …, 1]. Empty idx → scalar.
    if (idx.length === 0) return v.shape.length === 0 ? v.data[0] : v.data[0];
    let offset = 0;
    let stride = 1;
    for (let k = v.shape.length - 1; k >= 0; k--) {
      offset += (idx[k] || 0) * stride;
      stride *= v.shape[k];
    }
    return v.data[offset];
  }
  // Nested-object form.
  let cur = v;
  for (const i of idx) {
    if (cur && typeof cur === 'object') cur = cur[i];
    else return cur;
  }
  return cur;
}

function approxEq(got: number, want: number, eps = 1e-9): boolean {
  return Math.abs(got - want) < eps;
}

// =====================================================================
// 1. Surface forms — direct call + shorthand + variance markers
// =====================================================================

test('metricsum: direct call form parses cleanly', () => {
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
norm = metricsum(g, [], p[.mu^] * p[.mu_])
`;
  assert.equal(errors(src).length, 0);
});

test('metricsum: `g: r[axes] := body` shorthand parses cleanly', () => {
  // Per spec §05: `metric: result[output_indices] := expr` desugars
  // to `result = metricsum(metric, [output_indices], expr)`. The
  // parser-level lookahead distinguishes this from the AggregateBinding
  // shape (`Name [` vs `Name : Name [`).
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: norm[] := p[.mu^] * p[.mu_]
`;
  assert.equal(errors(src).length, 0);
});

test('metricsum: shorthand desugars to metricsum(metric, [...], body)', () => {
  // The shorthand and direct-call forms produce structurally identical
  // post-parse IR (modulo source loc info). We verify by comparing the
  // bound binding's RHS shape — both should have `op === 'metricsum'`
  // and matching outputAxes shape.
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: short[.mu^] := p[.mu^]
long = metricsum(g, [.mu^], p[.mu^])
`;
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `parse errors: ${errs.map((d: any) => d.message).join('; ')}`);
  // Both should classify as metricsum at lift-time and rewrite to
  // aggregate. Post-lift the user-binding's RHS is the rewritten
  // aggregate call; we test by checking the materialized values match
  // (they reference the same expression, so the numerical result is
  // identical).
  const built = orchestrator.buildDerivations(ctx.bindings);
  const vShort = built.fixedValues.get('short');
  const vLong  = built.fixedValues.get('long');
  // Both should be [3, 2] — a rank-1 vector storing p^μ with no
  // contraction (.mu^ output matches .mu^ in body, no lower indices).
  assert.ok(vShort && vLong, 'both bindings materialise to fixed values');
  assert.equal(valueAt(vShort, 0), 3);
  assert.equal(valueAt(vShort, 1), 2);
  assert.equal(valueAt(vLong, 0), 3);
  assert.equal(valueAt(vLong, 1), 2);
});

test('metricsum: variance-marked axes parse as AxisRef with variance field', () => {
  // The lexer-level details (`.mu^` tokenises as DOT + IDENT('mu') +
  // CARET; `.mu_` tokenises as DOT + IDENT('mu_'); both are
  // disambiguated in `parseAxisFromTokens`) are tested implicitly by
  // every other metricsum test. Here we pin the AST representation:
  // an output `.mu^` becomes `AxisRef(name='mu', variance='upper')`,
  // a body `.nu_` becomes `AxisRef(name='nu', variance='lower')`.
  // We use distinct axis names for the output and the contracted body
  // axes so the analyzer doesn't fire the "output index also
  // contracted" diagnostic.
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
A = [[1.0, 2.0], [3.0, 4.0]]
p = [3.0, 2.0]
g: r[.mu^] := A[.mu^, .nu_] * p[.nu^]
`;
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `parse errors: ${errs.map((d: any) => d.message).join('; ')}`);
  const b = ctx.bindings.get('r');
  assert.ok(b, 'r binding exists');
  const ast = b.node.value;
  assert.equal(ast.type, 'CallExpr');
  assert.equal(ast.callee.name, 'metricsum');
  const outAxesAst = ast.args[1];
  assert.equal(outAxesAst.type, 'ArrayLiteral');
  assert.equal(outAxesAst.elements.length, 1);
  assert.equal(outAxesAst.elements[0].type, 'AxisRef');
  assert.equal(outAxesAst.elements[0].name, 'mu');
  assert.equal(outAxesAst.elements[0].variance, 'upper');
  // The body's `A[.mu^, .nu_]` IndexExpr should carry AxisRefs with
  // variance markers captured too. Walk down to the inner AxisRefs.
  const body = ast.args[2];
  // body = BinaryExpr(*, IndexExpr(A, [.mu^, .nu_]), IndexExpr(p, [.nu^]))
  assert.equal(body.type, 'BinaryExpr');
  const leftIdx = body.left;
  assert.equal(leftIdx.type, 'IndexExpr');
  assert.equal(leftIdx.indices[0].variance, 'upper');
  assert.equal(leftIdx.indices[1].variance, 'lower');
});

// =====================================================================
// 2. Static checks — spec §04 §sec:metricsum "Static checks"
// =====================================================================

test('metricsum: bare-neutral axis inside metricsum is a static error', () => {
  // Spec: "bare neutral aggregate axes (`.i` without a variance
  // marker) are not allowed inside `metricsum`."
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: r[.mu^] := p[.mu]
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /bare-neutral axis '\.mu'/.test(d.message)),
    `expected bare-neutral diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: contracted axis without paired upper/lower is a static error', () => {
  // Spec: "every repeated non-output index in `expr` must occur
  // exactly twice — once upper and once lower". Two upper occurrences
  // of `.nu^` is not a valid contraction (would be `delta^{nu nu}` with
  // no metric inserted — semantically ambiguous).
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
A = [[1.0, 0.0], [0.0, 1.0]]
B = [[1.0, 0.0], [0.0, 1.0]]
g: r[.mu^] := A[.mu^, .nu^] * B[.nu^, .rho_]
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /must appear exactly twice/.test(d.message)),
    `expected paired-variance diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: output index appearing with opposite variance in body is a static error', () => {
  // Spec: "every output index must occur in `expr` with the same
  // variance and may not also be contracted." Output `.mu^` (upper) +
  // body `B[.mu_]` (lower) requests contracting the output index,
  // which spec forbids.
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: r[.mu^] := p[.mu_]
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /opposite variance/.test(d.message)),
    `expected opposite-variance diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: output index missing from body is a static error', () => {
  // Spec: every output axis must occur in expr.
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: r[.nu^] := p[.mu^] * p[.mu_]
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /does not appear in expr/.test(d.message)),
    `expected output-not-in-expr diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: duplicate output axis is a static error', () => {
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: r[.mu^, .mu^] := p[.mu^]
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /duplicate axis '\.mu'/.test(d.message)),
    `expected duplicate-axis diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: kwargs rejected (positional-only per spec)', () => {
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
r = metricsum(metric = g, output_axes = [.mu^], expr = p[.mu^])
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /positional arguments only/.test(d.message)),
    `expected kwargs-rejected diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: wrong arg count is a static error', () => {
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
r = metricsum(g, [.mu^])
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /takes exactly three arguments/.test(d.message)),
    `expected wrong-arg-count diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('metricsum: variance-marked axis outside metricsum is a static error', () => {
  // Per analyzer's scope-check, an AxisRef with variance marker can
  // only appear inside aggregate or metricsum. Outside, even a plain
  // axis (no variance) is illegal — this test pins the rule.
  const src = `
A = [[1.0, 2.0], [3.0, 4.0]]
x = A[.mu^]
`;
  const ds = errors(src);
  assert.ok(ds.some((d: any) =>
    /may only appear inside aggregate.*metricsum/.test(d.message)),
    `expected axis-scope diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

// =====================================================================
// 3. Lift pass — metricsum rewrites to aggregate(sum, ...) + __g_down
// =====================================================================

test('metricsum: lift rewrites metricsum() to aggregate(sum, ...)', () => {
  // After buildDerivations runs liftInlineSubexpressions, the user's
  // metricsum-bound binding has its AST mutated in place: the call
  // becomes `aggregate(sum, [stripped_axes], wrapped_body)`. Pin
  // this shape so downstream consumers know they never see a raw
  // metricsum call.
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: norm[] := p[.mu^] * p[.mu_]
`;
  const ctx = processSource(src);
  assert.equal(ctx.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  const built = orchestrator.buildDerivations(ctx.bindings);
  const b = built.bindings.get('norm');
  assert.ok(b, 'norm binding exists post-lift');
  // The classifier produces a derivation kind for the rewritten
  // aggregate. The binding's RHS AST head is now `aggregate`.
  const rhs = b.node.value;
  assert.equal(rhs.type, 'CallExpr');
  assert.equal(rhs.callee.name, 'aggregate', 'metricsum was rewritten to aggregate');
});

test('metricsum: lift hoists __g_down = inv(metric) when body has any lower-variance axis', () => {
  // The lift emits a synthetic `__g_down_N = inv(metric)` binding so
  // every lower-variance body access can multiply by the same shared
  // inv-metric matrix. We pin its presence + its `inv(...)` RHS shape.
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: norm[] := p[.mu^] * p[.mu_]
`;
  const ctx = processSource(src);
  const built = orchestrator.buildDerivations(ctx.bindings);
  // Walk the bindings map looking for any synthetic name starting
  // with __g_down_.
  let gDown: any = null;
  for (const [n, b] of built.bindings) {
    if (n.startsWith('__g_down_')) { gDown = b; break; }
  }
  assert.ok(gDown, '__g_down synthetic binding emitted');
  const rhs = gDown.node.value;
  assert.equal(rhs.type, 'CallExpr');
  assert.equal(rhs.callee.name, 'inv');
  // The inv() call's single arg is an Identifier ref to the user's
  // metric binding (or a hoisted synthetic when the metric arg was
  // a non-Identifier expression — the test above only exercises the
  // Identifier case).
  assert.equal(rhs.args[0].type, 'Identifier');
  assert.equal(rhs.args[0].name, 'g');
});

test('metricsum: lift skips __g_down hoist when body has no lower-variance axis', () => {
  // If the user writes metricsum but every axis is upper, the rewrite
  // still produces an aggregate (with stripped markers) but no inv-
  // metric is needed. We pin the no-emit case.
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: r[.mu^] := p[.mu^]
`;
  const ctx = processSource(src);
  const built = orchestrator.buildDerivations(ctx.bindings);
  let gDownCount = 0;
  for (const [n] of built.bindings) {
    if (n.startsWith('__g_down_')) gDownCount++;
  }
  assert.equal(gDownCount, 0, 'no __g_down synthetic binding for all-upper body');
});

// =====================================================================
// 4. Numerical correctness — load the metricsum-tensor.flatppl fixture
// =====================================================================

const fixtureSrc = readFixture('metricsum-tensor.flatppl');
const fixtureCtx = processSource(fixtureSrc);
const fixtureBuilt = orchestrator.buildDerivations(fixtureCtx.bindings);

test('metricsum-tensor.flatppl: fixture parses with no diagnostics', () => {
  const errs = fixtureCtx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `parse errors: ${errs.map((d: any) => d.message).join('; ')}`);
});

test('metricsum-tensor: norm_pp = p^μ p_μ = 9 - 4 = 5 (Minkowski p·p)', () => {
  // Spec §sec:metricsum norm example: `p[.μ^] * p[.μ_]` with output
  // `[]` produces the scalar Minkowski invariant (p^0)² - (p^1)².
  // For p = [3, 2]: result = 9 - 4 = 5.
  const v = fixtureBuilt.fixedValues.get('norm_pp');
  assert.ok(approxEq(valueAt(v), 5),
    `norm_pp = ${valueAt(v)}, want 5`);
});

test('metricsum-tensor: inner_pq = p^μ q_μ = 3 - 8 = -5 (Minkowski cross product)', () => {
  // p^μ q_μ = p^0 q_0 + p^1 q_1. With Minkowski metric:
  // q_0 = g_{0,α} q^α = q^0 = 1; q_1 = g_{1,α} q^α = -q^1 = -4.
  // p^0 q_0 + p^1 q_1 = 3*1 + 2*(-4) = -5.
  const v = fixtureBuilt.fixedValues.get('inner_pq');
  assert.ok(approxEq(valueAt(v), -5),
    `inner_pq = ${valueAt(v)}, want -5`);
});

test('metricsum-tensor: lower_q = M_id^μ_ν q^ν = q_μ raised = [1, -4]', () => {
  // M_id is identity in upper-upper storage, so its mixed-variance
  // form is M_id^μ_ν = g_{μν}. Result M_id^μ_ν q^ν = g_{μν} q^ν = q_μ
  // raised back to all-upper canonical storage = q with spatial sign
  // flipped. For q = [1, 4]: result = [1, -4].
  const v = fixtureBuilt.fixedValues.get('lower_q');
  assert.ok(approxEq(valueAt(v, 0), 1),  `lower_q[0] = ${valueAt(v, 0)}, want 1`);
  assert.ok(approxEq(valueAt(v, 1), -4), `lower_q[1] = ${valueAt(v, 1)}, want -4`);
});

test('metricsum-tensor: outer_pq = p outer q metric-independent = [[3, 12], [2, 8]]', () => {
  // Invariance test: `g: outer_pq[.μ^, .ν_] := p[.μ^] * q[.ν_]`
  // factors the metric raise (from output .ν_) and the metric lower
  // (from body q[.ν_]) into a single δ^σ_ν contraction. Result is
  // p^μ q^ν stored upper-upper = plain outer product.
  // For p = [3, 2], q = [1, 4]: outer = [[3, 12], [2, 8]].
  const v = fixtureBuilt.fixedValues.get('outer_pq');
  const expected = [[3, 12], [2, 8]];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      assert.ok(approxEq(valueAt(v, i, j), expected[i][j]),
        `outer_pq[${i}, ${j}] = ${valueAt(v, i, j)}, want ${expected[i][j]}`);
    }
  }
});

test('metricsum-tensor: L_compose (spec example, identity factors) = identity', () => {
  // Spec example: `g: L[.μ^, .ρ_] := L1[.μ^, .ν_] * L2[.ν^, .σ_] * L3[.σ^, .ρ_]`
  // with L1 = L2 = L3 = identity upper-upper. Each factor evaluates
  // to g_{μν} in mixed-variance form; the composition collapses to
  // δ^μ_ρ; raised to all-upper canonical via .ρ_ → .ρ_up = identity.
  // (See the fixture's docblock for the full derivation.)
  const v = fixtureBuilt.fixedValues.get('L_compose');
  const expected = [[1, 0], [0, 1]];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      assert.ok(approxEq(valueAt(v, i, j), expected[i][j]),
        `L_compose[${i}, ${j}] = ${valueAt(v, i, j)}, want ${expected[i][j]}`);
    }
  }
});

test('metricsum-tensor: AB_eye (identity metric) = plain matmul [[19, 22], [43, 50]]', () => {
  // Sanity check (spec equivalence): `metricsum(eye(n), ...)` is
  // equivalent to `aggregate(sum, ...)` with bare axes. The lift
  // still emits inv(eye) factors but they're identity contractions
  // at runtime, so the result matches plain matrix multiplication.
  const v = fixtureBuilt.fixedValues.get('AB_eye');
  const expected = [[19, 22], [43, 50]];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      assert.ok(approxEq(valueAt(v, i, j), expected[i][j]),
        `AB_eye[${i}, ${j}] = ${valueAt(v, i, j)}, want ${expected[i][j]}`);
    }
  }
});

// =====================================================================
// 5. Equivalence to aggregate under identity metric
// =====================================================================

test('metricsum: PIR roundtrip emits + parses (%uaxis name) / (%laxis name)', () => {
  // Spec §11 FlatPIR: variance-marked axes lower to `(%uaxis <name>)` /
  // `(%laxis <name>)` distinct from `(%axis <name>)`. pir-sexpr's
  // _exprToSexpr emits these per `variance` field; readUAxisForm /
  // readLAxisForm reconstruct them. Roundtrip stability ensures the
  // PIR text form survives a tooling crossing without losing variance.
  const lowerMod = require('../lower.ts');
  const pir      = require('../pir-sexpr.ts');
  const src = `
g = [[1.0, 0.0], [0.0, -1.0]]
p = [3.0, 2.0]
g: r[.mu^, .rho_] := p[.mu^] * p[.rho_]
`;
  const ctx = processSource(src);
  assert.equal(ctx.diagnostics.filter((d: any) => d.severity === 'error').length, 0);
  // Lower the user binding's RHS to PIR JSON.
  const b = ctx.bindings.get('r');
  const ir = lowerMod.lowerExpr(b.node.value);
  // Sanity-check JSON has axis IR with variance field.
  const outAxes = ir.args[1].args;
  assert.equal(outAxes[0].kind, 'axis');
  assert.equal(outAxes[0].variance, 'upper');
  assert.equal(outAxes[1].variance, 'lower');
  // Emit the full module sexpr, then re-parse.
  const sexprText = pir.toSexpr(ctx.loweredModule, { indent: false });
  assert.ok(sexprText.includes('(%uaxis mu)'),
    `expected (%uaxis mu) in sexpr; got: ${sexprText}`);
  assert.ok(sexprText.includes('(%laxis rho)'),
    `expected (%laxis rho) in sexpr; got: ${sexprText}`);
  const parsed = pir.fromSexpr(sexprText);
  const errs = parsed.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `sexpr parse errors: ${errs.map((d: any) => d.message).join('; ')}`);
  const rBinding = parsed.module.bindings.get('r');
  assert.ok(rBinding, 'r binding survives roundtrip');
  const roundTrippedAxes = rBinding.rhs.args[1].args;
  assert.equal(roundTrippedAxes[0].kind, 'axis');
  assert.equal(roundTrippedAxes[0].name, 'mu');
  assert.equal(roundTrippedAxes[0].variance, 'upper');
  assert.equal(roundTrippedAxes[1].kind, 'axis');
  assert.equal(roundTrippedAxes[1].name, 'rho');
  assert.equal(roundTrippedAxes[1].variance, 'lower');
});

test('metricsum: identity metric matches plain aggregate(sum, ...)', () => {
  // Spec §sec:metricsum "Equivalence under identity metric":
  // `metricsum(eye(n), ...)` is equivalent to an `aggregate(sum, ...)`
  // with co-/contravariant axis names replaced by aggregate axis
  // names. The lift's inv(eye)/eye factors collapse to identity
  // contractions, so the runtime result must match the plain
  // aggregate exactly.
  // ms uses metricsum via the `eye2:` shorthand prefix; plain is the
  // analogous AggregateBinding without variance markers. Both should
  // produce the same A @ B = [[19, 22], [43, 50]].
  const src = `
eye2 = [[1.0, 0.0], [0.0, 1.0]]
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
eye2: ms[.i^, .k^] := A[.i^, .j_] * B[.j^, .k^]
plain[.i, .k]   := A[.i, .j]   * B[.j, .k]
`;
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `parse errors: ${errs.map((d: any) => d.message).join('; ')}`);
  const built = orchestrator.buildDerivations(ctx.bindings);
  const ms    = built.fixedValues.get('ms');
  const plain = built.fixedValues.get('plain');
  // Both should equal A @ B = [[19, 22], [43, 50]].
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 2; j++) {
      const m = valueAt(ms,    i, j);
      const p = valueAt(plain, i, j);
      assert.ok(approxEq(m, p),
        `eye-metric metricsum vs plain aggregate disagree at [${i}, ${j}]: ms=${m}, plain=${p}`);
    }
  }
});
