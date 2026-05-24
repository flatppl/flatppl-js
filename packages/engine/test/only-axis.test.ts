'use strict';

// Spec §07 "Singleton-axis indexing with `only`" (flatppl-design
// `42b103e`). `only` (or its `!` shorthand) selects the unique
// element of an axis of length 1; a runtime error fires if the axis
// has any other length.
//
// Surface forms:
//   B[.i, !]          # ! shorthand (only inside `[...]`)
//   get(B, .i, only)  # only keyword
//
// Both lower to {kind:'const', name:'only'} in the IR, which the
// `get` evaluator dispatches via the dedicated ONLY sentinel.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');
const lowerMod = require('../lower.ts');
const { inBothModes } = require('./_perf-helpers.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

function evalRHS(src: string, binding: string, env: any) {
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `source must parse cleanly: ${errs.map((d: any) => d.message).join('; ')}`);
  // Evaluate every binding in declaration order, threading the env
  // forward so later bindings can reference earlier ones. Stop after
  // evaluating the target binding.
  const workEnv = Object.assign({}, env);
  for (const stmt of ctx.ast.body) {
    if (stmt.type !== 'AssignStatement') continue;
    const ir = lowerMod.lowerExpr(stmt.value);
    const v = sampler.evaluateExpr(ir, workEnv);
    if (stmt.names.length === 1) {
      workEnv[stmt.names[0].name] = v;
    } else {
      // Tuple-decomposition: project each LHS via tuple_get.
      for (let i = 0; i < stmt.names.length; i++) {
        workEnv[stmt.names[i].name] = Array.isArray(v) ? v[i] : undefined;
      }
    }
    if (workEnv[binding] !== undefined) return workEnv[binding];
  }
  throw new Error(`binding ${binding} not found`);
}

// ---------------------------------------------------------------------
// Surface forms parse cleanly
// ---------------------------------------------------------------------

test('only: `!` shorthand parses', () => {
  assert.equal(errors('x = B[!]\n').length, 0);
});

test('only: `only` keyword parses', () => {
  assert.equal(errors('x = get(B, only)\n').length, 0);
});

test('only: mixed with other selectors parses', () => {
  assert.equal(errors('x = B[1, !, 2]\n').length, 0);
  assert.equal(errors('x = B[:, !]\n').length, 0);
});

test('only: lowers to {kind:const, name:only}', () => {
  const ctx = processSource('x = B[!]\n');
  const ir = ctx.bindings.get('x').node.value;
  // AST: IndexExpr with one SliceOnly index.
  assert.equal(ir.type, 'IndexExpr');
  assert.equal(ir.indices[0].type, 'SliceOnly');
  // Lowered: get call with const 'only' selector.
  const lowered = lowerMod.lowerExpr(ir);
  assert.equal(lowered.kind, 'call');
  assert.equal(lowered.op, 'get');
  assert.equal(lowered.args[1].kind, 'const');
  assert.equal(lowered.args[1].name, 'only');
});

// ---------------------------------------------------------------------
// Runtime: only returns the single element of a length-1 axis
// ---------------------------------------------------------------------

test('only: extracts the unique element of a length-1 vector', () => {
  const v = [42];
  assert.equal(evalRHS('x = v[!]', 'x', { v }), 42);
});

test('only: works alongside an integer index (mixed)', () => {
  // B shape [3, 1]: pick row 2, then the unique column element.
  const B = [[10], [20], [30]];
  assert.equal(evalRHS('x = B[2, !]', 'x', { B }), 20);
});

test('only: works alongside `all` (slice)', () => {
  // B shape [3, 1]: keep all rows, drop the singleton column dim.
  const B = [[10], [20], [30]];
  // B[:, !] = [10, 20, 30]
  const got = evalRHS('x = B[:, !]', 'x', { B });
  assert.deepEqual(got, [10, 20, 30]);
});

test('only: all-`!` indexing of a fully singleton array yields the scalar', () => {
  // Shape [1, 1, 1] → after !,!,! → the unique scalar.
  const A = [[[7]]];
  assert.equal(evalRHS('x = A[!, !, !]', 'x', { A }), 7);
});

test('only: `only` keyword behaves identically to `!`', () => {
  const v = [99];
  assert.equal(evalRHS('x = v[!]', 'x', { v }), 99);
  assert.equal(evalRHS('x = get(v, only)', 'x', { v }), 99);
});

// ---------------------------------------------------------------------
// Error path: length ≠ 1 → runtime error
// ---------------------------------------------------------------------

test('only: runtime error when indexed axis has length > 1', () => {
  const v = [1, 2, 3];
  assert.throws(
    () => evalRHS('x = v[!]', 'x', { v }),
    /requires the indexed axis to have length 1, got length 3/);
});

test('only: runtime error when indexed axis is empty', () => {
  const v: number[] = [];
  assert.throws(
    () => evalRHS('x = v[!]', 'x', { v }),
    /requires the indexed axis to have length 1, got length 0/);
});

// ---------------------------------------------------------------------
// Disambiguation: `!` outside an index list is still logical NOT
// ---------------------------------------------------------------------

test('only: `!` followed by an expression remains logical NOT', () => {
  // `!a` is lnot(a). Inside an index list this only kicks in when
  // the BANG is followed by a non-{COMMA, RBRACKET} token.
  assert.equal(errors('a = true\nx = !a\n').length, 0);
  const ctx = processSource('a = true\nx = !a\n');
  // x's RHS is `lnot(a)` per the parser's BANG handling.
  const xRhs = ctx.bindings.get('x').node.value;
  // CallExpr to lnot.
  assert.equal(xRhs.type, 'CallExpr');
  assert.equal(xRhs.callee.name, 'lnot');
});

test('only: `!` followed by an expression inside `[...]` is still logical NOT', () => {
  // xs[!a] indexes xs by lnot(a) — semantically degenerate but
  // syntactically valid; the BANG isn't followed by a `,` or `]`.
  assert.equal(errors('a = true\nxs = [1, 2]\nx = xs[!a]\n').length, 0);
});

// ---------------------------------------------------------------------
// Reserved-word: `all` and `only` cannot be bindings
// ---------------------------------------------------------------------

test('only: `only` is reserved at binding position', () => {
  const errs = errors('only = 1\n');
  assert.ok(errs.some((d: any) => /reserved/.test(d.message)),
    `expected reserved-name error, got: ${errs.map((d: any) => d.message).join(', ')}`);
});

test('only: `all` is reserved at binding position', () => {
  const errs = errors('all = 1\n');
  assert.ok(errs.some((d: any) => /reserved/.test(d.message)));
});

// ---------------------------------------------------------------------
// Spec §07 addaxes inverse property
// ---------------------------------------------------------------------

test('only: shape [3,1] via [:, !] returns the rank-1 view', () => {
  // Spec §07 addaxes inverse property — a shape [n, 1] array indexed
  // by [:, !] yields a length-n vector. We construct the [3,1] array
  // directly (the engine's `addaxes` produces a Value shape-object
  // that the evaluateExpr nested-array path doesn't unwrap; that's a
  // separate concern from `only` semantics).
  const B = [[1], [2], [3]];
  const got = evalRHS('r = B[:, !]', 'r', { B });
  assert.deepEqual(got, [1, 2, 3]);
});

// ---------------------------------------------------------------------
// aggregate + only: the spec §04 broadcast-equivalence example
// ---------------------------------------------------------------------

test('only: aggregate + ! matches broadcast for singleton-axis pattern (spec §04)', () => {
  // Spec §04 "Relationship to broadcasting":
  //   aggregate(any_f_reduction, [.i, .j], A[.i, .j] * B[.i, !])
  //   ≡ broadcast((a, b) -> a * b, A, B)
  // where B has a trailing singleton dim. We use a pre-shaped B
  // (shape [2,1]) directly.
  const A = [[1, 2, 3], [4, 5, 6]];      // 2x3
  const B = [[10], [20]];                // 2x1
  const got = evalRHS(
    'out = aggregate(sum, [.i, .j], A[.i, .j] * B[.i, !])',
    'out', { A, B });
  // Expected element-wise: A row i times B[i, 0]:
  //   [[10, 20, 30], [80, 100, 120]]
  assert.equal(got[0][0], 10);
  assert.equal(got[0][1], 20);
  assert.equal(got[0][2], 30);
  assert.equal(got[1][0], 80);
  assert.equal(got[1][1], 100);
  assert.equal(got[1][2], 120);
});

test('only: aggregate runtime error if a non-singleton dim is indexed with `!`', () => {
  // B shape [2, 1]: `!` on first dim is invalid (length 2, not 1).
  const B = [[10], [20]];
  assert.throws(
    () => evalRHS(
      'bad = aggregate(sum, [.j], B[!, .j])',
      'bad', { B }),
    /requires the indexed axis to have length 1/);
});

// ---------------------------------------------------------------------
// Broadcast ↔ aggregate equivalence corpus
//
// `aggregate` is the explicit-axes counterpart of `broadcast`. When
// every output axis appears (in expr) at exactly the same position in
// every collection input, aggregate produces broadcast's result.
// Equivalences pinned here:
//
//   element-wise unary  : broadcast(f, A) ≡ aggregate(sum,[.i],f(A[.i]))
//   element-wise binary : broadcast(f, A, B) ≡ aggregate(.., A[.i]*B[.i])
//   outer product       : broadcast outer (via addaxes) ≡ A[.i]*B[.j]
//   singleton broadcast : aggregate with `!` on the singleton dim
//
// The pattern matters for two reasons:
//   1. It serves as a correctness oracle for `aggregate` — the
//      pre-existing broadcast path is well-tested.
//   2. It documents the user-level semantic translation, so future
//      AGGREGATE_PATTERNS specialisers can rewrite recognised
//      aggregate shapes to broadcast (or BLAS) calls equivalently.
// ---------------------------------------------------------------------

inBothModes('broadcast/aggregate equivalence: element-wise binary scalar op',
  'aggregate', () => {
  // broadcast((a,b) -> a*b, A, B) for length-n vectors A, B
  // ≡ aggregate(sum, [.i], A[.i] * B[.i])
  const A = [1, 2, 3, 4];
  const B = [10, 20, 30, 40];
  const got = evalRHS(
    'out = aggregate(sum, [.i], A[.i] * B[.i])',
    'out', { A, B });
  // Expected: [10, 40, 90, 160]
  assert.deepEqual([...got], [10, 40, 90, 160]);
});

inBothModes('broadcast/aggregate equivalence: element-wise unary',
  'aggregate', () => {
  // broadcast(x -> x*x, A) ≡ aggregate(sum, [.i], A[.i] * A[.i])
  const A = [1, 2, 3, 4];
  const got = evalRHS(
    'out = aggregate(sum, [.i], A[.i] * A[.i])',
    'out', { A });
  assert.deepEqual([...got], [1, 4, 9, 16]);
});

inBothModes('broadcast/aggregate equivalence: outer product via independent axes',
  'aggregate', () => {
  // No broadcasting trick needed — independent axes naturally give
  // the outer product. `aggregate(sum, [.i, .j], A[.i] * B[.j])`
  // ≡ A * Bᵀ (outer product).
  const A = [1, 2, 3];   // length 3
  const B = [10, 20];    // length 2
  const got = evalRHS(
    'out = aggregate(sum, [.i, .j], A[.i] * B[.j])',
    'out', { A, B });
  // Expected: [[10,20],[20,40],[30,60]]
  assert.equal(got[0][0], 10);
  assert.equal(got[0][1], 20);
  assert.equal(got[1][0], 20);
  assert.equal(got[1][1], 40);
  assert.equal(got[2][0], 30);
  assert.equal(got[2][1], 60);
});

inBothModes('broadcast/aggregate equivalence: row-vector broadcast (singleton on first axis)',
  'aggregate', () => {
  // A shape [3, 4]; b shape [1, 4]; broadcast row-wise.
  //   broadcast((a,b) -> a + b, A, B) where B = addaxes(v, 1, 0)
  //   ≡ aggregate(sum, [.i, .j], A[.i, .j] + B[!, .j])
  const A = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]];
  const B = [[100, 200, 300, 400]];   // shape [1, 4]
  const got = evalRHS(
    'out = aggregate(sum, [.i, .j], A[.i, .j] + B[!, .j])',
    'out', { A, B });
  assert.equal(got[0][0], 101);
  assert.equal(got[1][2], 307);
  assert.equal(got[2][3], 412);
});

inBothModes('broadcast/aggregate equivalence: scalar-Value broadcast (rank-1 length-1)',
  'aggregate', () => {
  // A scalar wrapped as a shape-[1] vector broadcasts uniformly.
  //   s shape [1]; broadcast((a,s) -> a*s, A, s) ≡ A * s[0]
  //   aggregate variant: aggregate(sum, [.i, .j], A[.i, .j] * s[!])
  const A = [[1, 2], [3, 4]];
  const s = [7];
  const got = evalRHS(
    'out = aggregate(sum, [.i, .j], A[.i, .j] * s[!])',
    'out', { A, s });
  // Expected: 7 * A
  assert.equal(got[0][0], 7);
  assert.equal(got[0][1], 14);
  assert.equal(got[1][0], 21);
  assert.equal(got[1][1], 28);
});

inBothModes('broadcast/aggregate equivalence: doubly-singleton broadcast',
  'aggregate', () => {
  // A shape [2, 3]; scalar wrapped as [1, 1]; broadcast every element.
  //   aggregate(sum, [.i, .j], A[.i, .j] + s[!, !])
  const A = [[1, 2, 3], [4, 5, 6]];
  const s = [[10]];   // shape [1, 1]
  const got = evalRHS(
    'out = aggregate(sum, [.i, .j], A[.i, .j] + s[!, !])',
    'out', { A, s });
  assert.equal(got[0][0], 11);
  assert.equal(got[1][2], 16);
});

inBothModes('broadcast/aggregate equivalence: bilinear form (matrix-vec-vec contraction)',
  'aggregate', () => {
  // Quadratic form xᵀ A y, fully reduced:
  //   q = aggregate(sum, [.dummy], x[.i] * A[.i, .j] * y[.j] * vec[.dummy])
  // requires at least one output axis (spec). Use a length-1 dummy
  // dimension via `!`-style: split into two steps.
  //   tmp[.j] := x[.i] * A[.i, .j]    # vector
  //   q[.k]  := tmp[.j] * y[.j] * one[.k]   # collapse j, wrap in k
  // For comparison:
  const x = [1, 2];
  const A = [[3, 4], [5, 6]];
  const y = [7, 8];
  // tmp[j] = Σ_i x[i] A[i,j]; tmp = [1*3+2*5, 1*4+2*6] = [13, 16]
  const tmpSrc = 'tmp = aggregate(sum, [.j], x[.i] * A[.i, .j])';
  const tmp = evalRHS(tmpSrc, 'tmp', { x, A });
  assert.deepEqual([...tmp], [13, 16]);
  // q[k] = Σ_j tmp[j] * y[j] * one[k]; with one = [1] → q = [13*7 + 16*8] = [91 + 128] = [219]
  const one = [1];
  const qSrc = 'q = aggregate(sum, [.k], tmp[.j] * y[.j] * one[.k])';
  const q = evalRHS(qSrc, 'q', { tmp, y, one });
  assert.equal(q[0], 219);
});

inBothModes('broadcast/aggregate equivalence: explicit reduction (sum over axis)',
  'aggregate', () => {
  // sum(A, axis=0): aggregate(sum, [.j], A[.i, .j]) — reduces over .i.
  const A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
  // Column sums: [12, 15, 18]
  const got = evalRHS('out = aggregate(sum, [.j], A[.i, .j])', 'out', { A });
  assert.deepEqual([...got], [12, 15, 18]);
});

// ---------------------------------------------------------------------
// Static-shape `only` check (typeinfer)
//
// When the indexed array's shape is statically known and the
// `only`-indexed dim has length ≠ 1, the analyzer emits a static
// error BEFORE the evaluator runs. The check piggybacks on the new
// inferGet shape-inference path which flattens nested array types.
// ---------------------------------------------------------------------

test('static only: error when literal array dim is statically not 1', () => {
  const errs = errors('v = [1.0, 2.0, 3.0]\nx = v[!]\n');
  assert.ok(errs.some((d: any) =>
    /'only' selector requires the indexed axis to have length 1, got length 3/.test(d.message)),
    `expected static-only diagnostic, got: ${errs.map((d: any) => d.message).join('; ')}`);
});

test('static only: error on the 2-D shape mismatch', () => {
  // A is shape [2, 3]; `!` on dim 0 (length 2) errors statically.
  const errs = errors('A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\nx = A[!, 1]\n');
  assert.ok(errs.some((d: any) =>
    /'only' selector.*length 2/.test(d.message)));
});

test('static only: no error on a length-1 literal array', () => {
  // Singleton arrays pass.
  assert.equal(errors('v = [42.0]\nx = v[!]\n').length, 0);
});

test('static only: no error when shape is dynamic/deferred', () => {
  // Distribution-typed arrays (or any deferred-shape source) should
  // NOT trigger the static check — falls back to runtime.
  // iid(Normal, n) produces a measure over arrays; binding to a
  // draw makes the variate dynamic-shape.
  const src = `
n = elementof(posintegers)
xs ~ iid(Normal(mu = 0, sigma = 1), n)
y = xs[!]
`;
  // Per spec the analyzer may still warn at runtime (the indexed
  // dim is n ≥ 2 in practice), but it can't statically prove it
  // — verify no STATIC `only` diagnostic fires here.
  const diags = errors(src);
  assert.ok(!diags.some((d: any) =>
    /'only' selector requires/.test(d.message)),
    'no static only diagnostic for dynamic-shape source');
});

test('static only: aggregate body picks up the static check', () => {
  // Inside aggregate, the `only` selector is validated against the
  // indexed array's known shape — same static error.
  const errs = errors(`
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
bad = aggregate(sum, [.j], A[!, .j])
`);
  assert.ok(errs.some((d: any) =>
    /'only' selector.*length 2/.test(d.message)));
});

test('static aggregate shape: output shape is the output_axes lengths', () => {
  // C[.i, .k] := A[.i, .j] * B[.j, .k]
  //   A shape [2, 3], B shape [3, 4] → C shape [2, 4]
  // Verify the inferred type of C carries this exact shape.
  const ctx = processSource(`
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
C[.i, .k] := A[.i, .j] * B[.j, .k]
`);
  // processSource's loweredModule has already been inferred; each
  // binding carries `.inferredType` from the module-level pass.
  const cb = ctx.loweredModule.bindings.get('C');
  assert.ok(cb && cb.inferredType, 'C has an inferredType');
  const cT = cb.inferredType;
  assert.equal(cT.kind, 'array', `expected array type, got ${cT.kind}`);
  assert.equal(cT.rank, 2);
  assert.equal(cT.shape[0], 2, `axis .i length: ${cT.shape[0]}`);
  assert.equal(cT.shape[1], 4, `axis .k length: ${cT.shape[1]}`);
});

inBothModes('broadcast/aggregate equivalence: := shorthand matrix multiplication',
  'aggregate', () => {
  // C[.i,.k] := A[.i,.j] * B[.j,.k]   — classic matmul.
  const A = [[1, 2], [3, 4]];
  const B = [[5, 6], [7, 8]];
  // C = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
  const got = evalRHS('C[.i, .k] := A[.i, .j] * B[.j, .k]', 'C', { A, B });
  assert.equal(got[0][0], 19);
  assert.equal(got[0][1], 22);
  assert.equal(got[1][0], 43);
  assert.equal(got[1][1], 50);
});
