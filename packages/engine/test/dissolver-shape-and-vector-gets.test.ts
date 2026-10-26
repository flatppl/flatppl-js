'use strict';

// Two dissolver rewrites read a sub-expression's type wrongly.
//
// `_resolveExprType` gave every dissolve-safe call the type of its operand,
// but transpose, diagmat, row_gram, col_gram and a two-array `mul` change
// the shape. The shape fold then wrote that operand shape into
// `lengthof` / `sizeof` as a literal: `sizeof(transpose(M))` was [2, 3].
//
// `_wrapVectorLeaves` treated every `get` as a scalar leaf. A get that
// returns a vector (`vv[1]`, `M[:, 2]`, `rec.a`) never got the `.j` reduction
// axis, so `broadcast(fn(sum(vv[1] .* _)), xs)` evaluated to nulls.
//
// ORACLE: §07 shapes and sums worked by hand. M is 2x3, v has length 3,
// vv = [[1, 2], [3, 4]], xs = [1, 10, 100]:
//   transpose(M), adjoint(M): 3x2       diagmat(v): 3x3
//   row_gram(M) = M M^T: 2x2            col_gram(M) = M^T M: 3x3
//   M * transpose(M): 2x2
//   sum(vv[1] .* x) = 3x   sum(vv[2] * x) = 7x   sum(M2[:, 2] .* x) = 6x
//   sum(vv[1] .* 3 .* x) = 9x   sum(c .* x .+ c[2]) = 11x + 12 with c = [5, 6]
//   get0 is 0-based (§07): get0(vv, 0) = [1, 2], get0(c, 1) = 6,
//   get0(M3, all, 1) = [2, 5], get0(M3, 1, [0, 2]) = [4, 6] with
//   M3 = [[1, 2, 3], [4, 5, 6]]

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

async function valueOf(src: string, name = 't', sampleCount = 1): Promise<any[]> {
  const { ctx } = makeMatCtx(src, { sampleCount });
  const m = await ctx.getMeasure(name);
  if (m.value && m.value.data) return Array.from(m.value.data);
  if (m.samples) return Array.from(m.samples);
  return [m.value];
}

const SHAPE_BASE = `M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])\nv = [1.0, 2.0, 3.0]\n`;
const SHAPES: [string, number[]][] = [
  ['lengthof(transpose(M))', [3]],
  ['sizeof(transpose(M))', [3, 2]],
  ['sizeof(adjoint(M))', [3, 2]],
  ['sizeof(diagmat(v))', [3, 3]],
  ['sizeof(row_gram(M))', [2, 2]],
  ['sizeof(col_gram(M))', [3, 3]],
  ['lengthof(col_gram(M))', [3]],
  ['sizeof(M * transpose(M))', [2, 2]],
  ['sizeof(M + M)', [2, 3]],
  ['sizeof(2.0 * M)', [2, 3]],
  ['indicesof(cumsum(v))', [1, 2, 3]],
];

for (const [expr, want] of SHAPES) {
  test(`shape fold: ${expr} = [${want}]`, async () => {
    assert.deepEqual(await valueOf(`${SHAPE_BASE}t = ${expr}\n`), want);
  });
}

const GET_BASE = `vv = [[1.0, 2.0], [3.0, 4.0]]
M2 = rowstack([[1.0, 2.0], [3.0, 4.0]])
M3 = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
rec = record(a = [1.0, 2.0], b = 3.0)
c = [5.0, 6.0]
k = 2
xs = [1.0, 10.0, 100.0]
`;
const GETS: [string, number[]][] = [
  ['broadcast(fn(sum(vv[1] .* _)), xs)', [3, 30, 300]],
  ['broadcast(fn(sum(vv[2] * _)), xs)', [7, 70, 700]],
  ['broadcast(fn(sum(vv[k] .* _)), xs)', [7, 70, 700]],
  ['broadcast(fn(sum(M2[1] .* _)), xs)', [3, 30, 300]],
  ['broadcast(fn(sum(M2[1, :] .* _)), xs)', [3, 30, 300]],
  ['broadcast(fn(sum(M2[:, 2] .* _)), xs)', [6, 60, 600]],
  ['broadcast(fn(sum(rec.a .* _)), xs)', [3, 30, 300]],
  ['broadcast(fn(sum(vv[1] .* rec.b .* _)), xs)', [9, 90, 900]],
  ['broadcast(fn(sum(c .* _ .+ c[k])), xs)', [23, 122, 1112]],
  ['broadcast(fn(sum(M2[2, [1, 2]] .* _)), xs)', [7, 70, 700]],
  ['broadcast(fn(sum(get0(vv, 0) .* _)), xs)', [3, 30, 300]],
  ['broadcast(fn(sum(c .* _ .+ get0(c, 1))), xs)', [23, 122, 1112]],
  ['broadcast(fn(sum(get0(M3, all, 1) .* _)), xs)', [7, 70, 700]],
  ['broadcast(fn(sum(get0(M3, 1, [0, 2]) .* _)), xs)', [10, 100, 1000]],
];

for (const [expr, want] of GETS) {
  test(`vector get in a fused reduction: ${expr}`, async () => {
    const { built } = makeMatCtx(`${GET_BASE}t = ${expr}\n`, { sampleCount: 1 });
    assert.equal(built.bindings.get('t').ir.op, 'aggregate', 'the broadcast fuses');
    assert.deepEqual(await valueOf(`${GET_BASE}t = ${expr}\n`), want);
  });
}

test('vector get in a fused reduction over a stochastic argument', async () => {
  const src = `${GET_BASE}z ~ iid(Normal(0, 1), 3)\nys = xs .+ 0.0 .* z\nt = broadcast(fn(sum(vv[1] .* _)), ys)\n`;
  assert.deepEqual(await valueOf(src, 't', 2), [3, 30, 300, 3, 30, 300]);
});

// An index form the rank resolver does not know leaves the broadcast to the
// runtime path, which still gives the right sum (M2[2, ix] = [3, 4]).
test('a get with an unresolved index form is not fused and stays correct', async () => {
  const src = `${GET_BASE}ix = [1, 2]\nt = broadcast(fn(sum(M2[2, ix] .* _)), xs)\n`;
  const { built } = makeMatCtx(src, { sampleCount: 1 });
  assert.equal(built.bindings.get('t').ir.op, 'broadcast');
  assert.deepEqual(await valueOf(src), [7, 70, 700]);
});

// A literal index outside the static shape is not fused, so the runtime
// reports its bounds error.
for (const [expr, msg] of [
  ['broadcast(fn(sum(vv[3] .* _)), xs)', /get index 3 out of bounds for length 2/],
  ['broadcast(fn(sum(M3[3] .* _)), xs)', /get index 3 out of bounds for length 2/],
  ['broadcast(fn(sum(M3[1, [1, 4]] .* _)), xs)', /get index 4 out of bounds for length 3/],
  ['broadcast(fn(sum(get0(vv, 2) .* _)), xs)', /get0 index 2 out of bounds for length 2/],
] as const) {
  test(`out-of-range index keeps the bounds error: ${expr}`, async () => {
    const src = `${GET_BASE}t = ${expr}\n`;
    const { built } = makeMatCtx(src, { sampleCount: 1 });
    assert.equal(built.bindings.get('t').ir.op, 'broadcast');
    await assert.rejects(valueOf(src), msg);
  });
}

// An index vector must hold scalars before its length is an axis length.
test('a nested index vector is not fused', () => {
  const { built } = makeMatCtx(`${GET_BASE}t = broadcast(fn(sum(c[[[1, 2], [2, 1]]] .* _)), xs)\n`, { sampleCount: 1 });
  assert.equal(built.bindings.get('t').ir.op, 'broadcast');
});
