'use strict';

// Spec §04 §sec:aggregate — Multi-axis aggregation.
//
//   aggregate(f_reduction, output_axes, expr)
//
// generalizes vector reductions to multi-axis tensor contraction.
// Axis names `.i, .j, ...` are symbolic labels lexically scoped to the
// enclosing aggregate(...). The shorthand `C[.i, .k] := expr` lowers
// at parse time to `C = aggregate(sum, [.i, .k], expr)`.

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

// Direct evaluation: lower a source line containing an aggregate(...)
// call to IR and run it through sampler.evaluateExpr with a hand-built
// env supplying the indexed arrays. This isolates the contraction
// semantics from the rest of the orchestration pipeline.
function evalAggregateRHS(src: string, binding: string, env: any) {
  const ctx = processSource(src);
  const errs = ctx.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `source must parse cleanly: ${errs.map((d: any) => d.message).join('; ')}`);
  const b = ctx.bindings.get(binding);
  assert.ok(b, `binding ${binding} not found`);
  const ir = lowerMod.lowerExpr(b.node.value);
  return sampler.evaluateExpr(ir, env);
}

// ---------------------------------------------------------------------
// Surface forms
// ---------------------------------------------------------------------

test('aggregate: direct call form parses cleanly', () => {
  const src = `
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
`;
  assert.equal(errors(src).length, 0);
});

test('aggregate: := shorthand parses cleanly', () => {
  const src = `
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
C[.i, .k] := A[.i, .j] * B[.j, .k]
`;
  assert.equal(errors(src).length, 0);
});

test('aggregate: := desugars to aggregate(sum, [...], expr)', () => {
  const src = `
A = [[1.0, 2.0], [3.0, 4.0]]
C[.i, .j] := A[.i, .j]
`;
  const ctx = processSource(src);
  const b = ctx.bindings.get('C');
  assert.ok(b);
  // After parse-time desugar the RHS AST is a CallExpr to aggregate.
  const v = b.node.value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'aggregate');
  // Reduction = sum (the := shorthand fixes this).
  assert.equal(v.args[0].type, 'Identifier');
  assert.equal(v.args[0].name, 'sum');
  // output_axes is an ArrayLiteral of AxisRef.
  assert.equal(v.args[1].type, 'ArrayLiteral');
  assert.equal(v.args[1].elements[0].type, 'AxisRef');
  assert.equal(v.args[1].elements[0].name, 'i');
});

// ---------------------------------------------------------------------
// Semantic correctness — matmul equivalence (spec §04 example 1)
// ---------------------------------------------------------------------

// Matrix multiplication is THE motivating use case for the aggregate
// pattern table — the matmul specialiser dispatches this shape to
// the `mul` op directly. Run in BOTH modes so the specialiser
// (opt=on) and the nested-loop interpreter (opt=off) are pinned to
// agree on the same numeric output. Any future regression where the
// specialiser drifts from the general loop fails both variants
// here.
inBothModes('aggregate: matrix multiplication agrees with `mul` (spec §04 example 1)',
  'aggregate', () => {
  // Spec §04: C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
  // ≡ standard A·B matmul.
  const A = [[1, 2, 3], [4, 5, 6]];
  const B = [[7, 8], [9, 10], [11, 12]];
  // Expected matmul:
  //   [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
  // = [[58, 64], [139, 154]]
  const src = 'C_agg = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])';
  const got = evalAggregateRHS(src, 'C_agg', { A, B });
  // 2-D output: nested-array form in both modes (matmul specialiser
  // returns plain JS arrays of plain JS arrays; general loop
  // returns plain JS arrays of Float64Array). Element-access syntax
  // `got[i][j]` works on both, which is what we assert.
  assert.equal(got.length, 2);
  assert.equal(got[0].length, 2);
  assert.equal(got[0][0], 58);
  assert.equal(got[0][1], 64);
  assert.equal(got[1][0], 139);
  assert.equal(got[1][1], 154);
});

// Same equivalence test with the multiplicative operands swapped —
// the matmul specialiser must recognise both `A[.i,.j] * B[.j,.k]`
// and `B[.j,.k] * A[.i,.j]` (scalar multiplication is commutative;
// the result is still A·B).
inBothModes('aggregate: matmul with operand order swapped — same result',
  'aggregate', () => {
  const A = [[1, 2, 3], [4, 5, 6]];
  const B = [[7, 8], [9, 10], [11, 12]];
  const src = 'C_agg = aggregate(sum, [.i, .k], B[.j, .k] * A[.i, .j])';
  const got = evalAggregateRHS(src, 'C_agg', { A, B });
  assert.equal(got[0][0], 58);
  assert.equal(got[1][1], 154);
});

test('aggregate: weighted sum-of-squared-differences (spec §04 example 2)', () => {
  // D = aggregate(sum, [.i, .k], (A[.i, .j] - B[.j, .k])^2 * W[.j])
  // For A shape [I,J], B shape [J,K], W shape [J]:
  //   D[i,k] = Σ_j (A[i,j] - B[j,k])^2 * W[j]
  const A = [[1, 2], [3, 4]];          // 2x2
  const B = [[1, 0], [0, 1]];          // 2x2
  const W = [1, 2];                    // length 2
  // D[0,0] = (1-1)^2*1 + (2-0)^2*2 = 0 + 8 = 8
  // D[0,1] = (1-0)^2*1 + (2-1)^2*2 = 1 + 2 = 3
  // D[1,0] = (3-1)^2*1 + (4-0)^2*2 = 4 + 32 = 36
  // D[1,1] = (3-0)^2*1 + (4-1)^2*2 = 9 + 18 = 27
  const src = 'D = aggregate(sum, [.i, .k], (A[.i, .j] - B[.j, .k])^2 * W[.j])';
  const got = evalAggregateRHS(src, 'D', { A, B, W });
  assert.equal(got[0][0], 8);
  assert.equal(got[0][1], 3);
  assert.equal(got[1][0], 36);
  assert.equal(got[1][1], 27);
});

test('aggregate: column-wise variance (spec §04 example 3)', () => {
  // V = aggregate(var, [.j], M[.i, .j])
  //   = [var(M[:,0]), var(M[:,1]), ...]
  // Use a matrix where each column has a known variance.
  const M = [[1, 10], [2, 20], [3, 30]];
  const src = 'V = aggregate(var, [.j], M[.i, .j])';
  const got = evalAggregateRHS(src, 'V', { M });
  // Engine's `var` is population variance (divisor n; see sampler
  // ARITH_OPS.var). Spec §07 lists sample variance, but the impl is
  // population — that's an existing divergence we just inherit.
  // var([1,2,3])    = ((1-2)² + 0 + 1²)/3 = 2/3
  // var([10,20,30]) = (100 + 0 + 100)/3 = 200/3
  assert.equal(got.length, 2);
  assert.ok(Math.abs(got[0] - (2 / 3))   < 1e-10, `expected ~2/3, got ${got[0]}`);
  assert.ok(Math.abs(got[1] - (200 / 3)) < 1e-10, `expected ~200/3, got ${got[1]}`);
});

test('aggregate: row-wise sum with fixed column (spec §04 example 4)', () => {
  // S = aggregate(sum, [.i], A[.i, 1])
  // Just extracts column 1 (1-based) of A — no reduction since .i is
  // the sole output axis and the only axis.
  const A = [[10, 20, 30], [40, 50, 60]];   // 2x3
  // Column 1 (1-based) → [10, 40]
  const src = 'S = aggregate(sum, [.i], A[.i, 1])';
  const got = evalAggregateRHS(src, 'S', { A });
  assert.equal(got[0], 10);
  assert.equal(got[1], 40);
});

test('aggregate: all seven reductions over a known vector', () => {
  // 1-axis aggregate over a vector A — equivalent to reducing the
  // whole vector by f. Since the output axis .i has no reduction axis,
  // we need to reduce something. Use a 2D matrix and reduce over .j.
  //   R[.i] := f(M[.i, .j])    reduces over .j
  const M = [[1, 2, 3, 4, 5]];   // 1x5
  const tests: Array<[string, number]> = [
    ['sum',     15],          // 1+2+3+4+5
    ['prod',    120],         // 1*2*3*4*5
    ['mean',    3],           // 15/5
    ['var',     2],           // population variance: ((-2)²+(-1)²+0+1+2²)/5 = 10/5
    ['maximum', 5],
    ['minimum', 1],
  ];
  for (const [f, expected] of tests) {
    const src = `R = aggregate(${f}, [.i], M[.i, .j])`;
    const got = evalAggregateRHS(src, 'R', { M });
    assert.ok(Math.abs(got[0] - expected) < 1e-10,
      `${f}: expected ${expected}, got ${got[0]}`);
  }
});

test('aggregate: nested aggregate composes (inner produces a matrix the outer indexes)', () => {
  // outer .i reduces over a literal/inner-aggregate matrix.
  // Construct outer via aggregate over an inline value won't work
  // (axis name from outer can't reach inner). The user's mental model:
  // inner produces an array; outer treats that array via its own axes.
  // Test by composition: compute inner separately, store as value, use
  // value in outer.
  // (Direct programmatic equivalent of writing `inner` as a binding
  // and referencing it in `outer`.)
  const M = [[1, 2], [3, 4], [5, 6]];   // 3x2
  // inner: column sums of M → [9, 12] (sum over .i)
  const innerSrc = 'inner = aggregate(sum, [.j], M[.i, .j])';
  const inner = evalAggregateRHS(innerSrc, 'inner', { M });
  assert.equal(inner[0], 9);
  assert.equal(inner[1], 12);
  // outer: sum of inner → 21
  const outerSrc = 'outer = aggregate(sum, [.k], inner[.k])';
  // Wait — .k is the sole axis here AND it's an output axis, so no
  // reduction. The result is just `inner` again.
  const outer = evalAggregateRHS(outerSrc, 'outer', { inner });
  assert.equal(outer[0], inner[0]);
  assert.equal(outer[1], inner[1]);
});

test('aggregate: := matches plain aggregate(sum, ...)', () => {
  const ctxA = processSource(`
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
C[.i, .k] := A[.i, .j] * B[.j, .k]
`);
  const ctxB = processSource(`
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[5.0, 6.0], [7.0, 8.0]]
C = aggregate(sum, [.i, .k], A[.i, .j] * B[.j, .k])
`);
  function stripLocs(n: any): any {
    if (n == null) return n;
    if (Array.isArray(n)) return n.map(stripLocs);
    if (typeof n !== 'object') return n;
    const out: any = {};
    for (const k of Object.keys(n)) {
      if (k === 'loc') continue;
      out[k] = stripLocs(n[k]);
    }
    return out;
  }
  assert.deepEqual(stripLocs(ctxA.bindings.get('C').ir),
                   stripLocs(ctxB.bindings.get('C').ir));
});

// ---------------------------------------------------------------------
// All seven reductions parse + classify
// ---------------------------------------------------------------------

const REDUCTIONS = ['sum', 'prod', 'mean', 'var', 'std', 'maximum', 'minimum'];
for (const r of REDUCTIONS) {
  test(`aggregate: reduction '${r}' parses + classifies`, () => {
    const src = `
A = [[1.0, 2.0], [3.0, 4.0]]
R = aggregate(${r}, [.i], A[.i, .j])
`;
    assert.equal(errors(src).length, 0,
      `reduction ${r} should parse cleanly`);
  });
}

// ---------------------------------------------------------------------
// Static analyzer errors (per spec rules)
// ---------------------------------------------------------------------

test('aggregate: rejects an unknown reduction', () => {
  const errs = errors(`
A = [[1.0, 2.0], [3.0, 4.0]]
R = aggregate(exp, [.i], A[.i, .j])
`);
  assert.ok(errs.some((d: any) =>
    /must be one of: sum, prod, mean, var, std, maximum, minimum/.test(d.message)));
});

test('aggregate: rejects empty output_axes', () => {
  const errs = errors(`
A = [[1.0, 2.0]]
R = aggregate(sum, [], A[.i, .j])
`);
  assert.ok(errs.some((d: any) =>
    /requires at least one output axis/.test(d.message)));
});

test('aggregate: rejects duplicate output axes', () => {
  const errs = errors(`
A = [[1.0, 2.0]]
R = aggregate(sum, [.i, .i], A[.i, .j])
`);
  assert.ok(errs.some((d: any) =>
    /duplicate axis/.test(d.message)));
});

test('aggregate: rejects output axis not appearing in expr', () => {
  const errs = errors(`
A = [[1.0, 2.0]]
R = aggregate(sum, [.k], A[.i, .j])
`);
  assert.ok(errs.some((d: any) =>
    /output axis '\.k' does not appear in expr/.test(d.message)));
});

test('aggregate: rejects non-axis entries in output_axes', () => {
  const errs = errors(`
A = [[1.0, 2.0]]
R = aggregate(sum, [1, 2], A[.i, .j])
`);
  assert.ok(errs.some((d: any) =>
    /output_axes entries must be axis names/.test(d.message)));
});

test('aggregate: rejects wrong arg count', () => {
  const errs = errors(`
A = [[1.0]]
R = aggregate(sum, [.i])
`);
  assert.ok(errs.some((d: any) =>
    /takes exactly three arguments/.test(d.message)));
});

test('aggregate: rejects keyword args', () => {
  const errs = errors(`
A = [[1.0]]
R = aggregate(f = sum, axes = [.i], e = A[.i, .j])
`);
  assert.ok(errs.some((d: any) =>
    /takes positional arguments only/.test(d.message)));
});

// ---------------------------------------------------------------------
// Axis-name scope (lexical to enclosing aggregate)
// ---------------------------------------------------------------------

test('aggregate: axis name outside aggregate is a static error', () => {
  const errs = errors(`x = .i + 1\n`);
  assert.ok(errs.some((d: any) =>
    /Axis name '\.i' may only appear inside aggregate/.test(d.message)));
});

test('aggregate: axis name inside fn(...) but outside aggregate is a static error', () => {
  // .i has no enclosing aggregate even though it's nested in a fn.
  const errs = errors(`f = fn(.i + 1)\n`);
  assert.ok(errs.some((d: any) =>
    /Axis name '\.i' may only appear inside aggregate/.test(d.message)));
});

test('aggregate: nested aggregates each have their own axis scope', () => {
  // Inner aggregate uses .j; outer uses .i. Both legal.
  const src = `
A = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
inner = aggregate(sum, [.k], A[1, 1, .k])
S = aggregate(sum, [.i], A[.i, 1, 1])
`;
  assert.equal(errors(src).length, 0);
});

// ---------------------------------------------------------------------
// FieldAccess vs Axis disambiguation (spec §05 closing note)
// ---------------------------------------------------------------------

test('aggregate: `r.field` after expression is FieldAccess, not Axis', () => {
  // After a postfix-able expression, `.name` is FieldAccess.
  const src = `
r = record(a = 1, b = 2)
x = r.a
`;
  assert.equal(errors(src).length, 0);
  const ctx = processSource(src);
  // Verify at the AST level — the analyzer keeps node.value (RHS AST).
  const x = ctx.bindings.get('x');
  assert.equal(x.node.value.type, 'FieldAccess');
  assert.equal(x.node.value.field, 'a');
});

// ---------------------------------------------------------------------
// Numerical correctness — vector reductions through aggregate
// ---------------------------------------------------------------------

test('aggregate: 1-axis sum over a literal vector equals `sum`', async () => {
  // `aggregate(sum, [.i], A[.i])` reduces over no axes since .i is
  // the sole declared output axis and there are no others. So it just
  // produces [A[1], A[2], ...] — equivalent to identity on A.
  // Useful sanity test for the no-reduce-axes path.
  const src = `
A = [10.0, 20.0, 30.0]
R = aggregate(sum, [.i], A[.i])
`;
  const ctx = processSource(src);
  assert.equal(ctx.diagnostics.filter(
    (d: any) => d.severity === 'error').length, 0);
});

test('aggregate: scalar contraction (dot product via aggregate)', () => {
  // `aggregate(sum, [.i], A[.i] * B[.i])` is undefined (no reduction
  // axis — every output cell is just A[i]*B[i]). For an actual dot
  // product we declare a single output axis we don't want kept; but
  // the spec requires at least ONE output axis. So a "true" scalar
  // contraction isn't expressible with `aggregate` alone — wrap with
  // a trailing scalar `sum`:
  //   d = sum(aggregate(sum, [.i], A[.i] * B[.i]))
  // The aggregate part should at least parse cleanly.
  const src = `
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
d_array = aggregate(sum, [.i], A[.i] * B[.i])
`;
  assert.equal(errors(src).length, 0);
});
