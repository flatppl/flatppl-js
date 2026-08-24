'use strict';

// =====================================================================
// broadcast-collection-domain-heads.test.ts
// =====================================================================
//
// A broadcast hands its head one ELEMENT (spec §04 "Broadcasting"), so a §07
// head whose Domains cell admits only collections has no meaning over a
// SCALAR element — §03 defines no rank-0 array. The reasoning, the 53-head
// table across six §07 tables, and the two carve-outs live in
// collection-domain-heads.ts.
//
// What this file pins, in three groups:
//
//   1. The refusal. Every head in the table is a located static error over a
//      scalar cell, citing its §07 table and its Domains cell. At BASE every
//      one of these compiled clean, and 20 of them produced a NUMBER:
//      `sum.(v)` returned the operand unchanged, `lany.([true, false, true])`
//      returned `[false, false, false]`, `sum.(M)` returned the matrix
//      unchanged, `transpose.(v)` returned the whole transposed vector with
//      the broadcast wrapper discarded. Fifteen of the heads type-checked
//      because `types.ts` declares their argument slot `any()` — the type AST
//      cannot say "array of any rank" — which is why the gate keys on §07's
//      domain and not on the signature.
//
//   2. The carve-outs still type. `min` / `max` (§07 "Elementary functions",
//      Domains `reals`) and `cat` (§07 "Array and table operations", Domains
//      "scalars, vectors, or records") admit a scalar, so the rule does not
//      reach them, and neither does it reach an ordinary elementwise head.
//
//   3. The nested case is LEGAL and its numbers are right. §03: "Vectors of
//      vectors are not interpreted as matrices implicitly" — so a vector of
//      vectors has ONE axis and the head receives an inner VECTOR, which IS
//      in `sum`'s domain. At BASE `sum.(vv)`, `mean.(vv)`, `prod.(vv)`,
//      `cumsum.(vv)`, `cumprod.(vv)` and `cummin.(vv)` all returned wrong
//      numbers here: the dissolver rewrote the broadcast to a bare
//      whole-value call (`cumsum(vv)` — one scan across the entire flat
//      buffer) or fused it to `aggregate(sum, [.atom], get(vv, .atom))`, a
//      reduction over one element. Every expected value below is from
//      numpy / scipy, not from this engine.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const {
  COLLECTION_DOMAIN_HEADS, AGGREGATE_ELIGIBLE_HEADS,
} = require('../collection-domain-heads.ts');

function errorsOf(src: string): string[] {
  const r = processSource(src);
  return (r.diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

// Compile, assert clean, and return the fixed value of `y`.
function valueOfY(src: string): any {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [],
    'expected a clean compile, got: ' + JSON.stringify(errs));
  return buildDerivations(r.bindings).fixedValues.get('y');
}

// Flatten a Value / nested Value to a plain number list, so an expectation can
// be written as the mathematical answer without committing to a storage shape.
function flatten(v: any, out: number[] = []): number[] {
  if (typeof v === 'number') { out.push(v); return out; }
  if (typeof v === 'boolean') { out.push(v ? 1 : 0); return out; }
  if (v instanceof Float64Array || v instanceof Int32Array) {
    for (const x of v) out.push(x);
    return out;
  }
  if (Array.isArray(v)) { for (const x of v) flatten(x, out); return out; }
  if (v && v.data) return flatten(v.data, out);
  throw new Error('flatten: unhandled value ' + JSON.stringify(v));
}

function assertClose(actual: any, expected: number[], what: string) {
  const got = flatten(actual);
  assert.equal(got.length, expected.length, what + ': length');
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(got[i] - expected[i]) < 1e-12,
      what + ': entry ' + i + ' is ' + got[i] + ', expected ' + expected[i]);
  }
}

// The operand each head's first argument is spelled over. All are flat
// collections of SCALARS, so every cell is a scalar and every head refuses.
const V = 'v = [1.0, 2.0, 3.0, 4.0]';
const BOOLS = 'v = [true, false, true]';
const BOOLEAN_HEADS = new Set(['lany', 'lall']);
// Arity beyond the first argument, for the four multi-argument rows whose
// dotted call would otherwise be an arity error before the domain gate runs.
const EXTRA_ARGS: Record<string, string> = {
  quantile: '0.5',
  linsolve: 'v',
  cross: 'v',
  diag: '0',
  quadform: 'v',
  tile: '2',
  splitblocks: '2',
  partition: '2',
  addaxes: '1, 0',
  bandedmat: '2',
};

// =====================================================================
// 1. The refusal — all 53 heads, over a scalar cell
// =====================================================================

test('the table names 53 heads across the six §07 tables', () => {
  assert.equal(COLLECTION_DOMAIN_HEADS.size, 53);
  const bySection: Record<string, number> = {};
  for (const [, row] of COLLECTION_DOMAIN_HEADS) {
    bySection[row.section] = (bySection[row.section] || 0) + 1;
  }
  assert.deepEqual(bySection, {
    'Reductions': 13,
    'Boolean reductions': 2,
    'Norms and normalization': 8,
    'Cumulative operations': 4,
    'Linear algebra': 16,
    'Array and table operations': 10,
  });
});

test('every Domains cell is §07\'s, quoted', () => {
  // Spot-checked against flatppl-design/docs/07-functions.md at 4a11afd. The
  // whole cell is carried even for a multi-argument row, though only the
  // FIRST argument's domain drives the refusal.
  const spot: Record<string, string> = {
    sum: 'real/complex arrays',
    var: 'real arrays',
    quantile: 'real arrays, `interval(0, 1)`',
    lengthof: 'vectors, tables',
    lany: 'boolean arrays',
    l2norm: 'real/complex vectors',
    logsumexp: 'real vectors',
    cumsum: 'vectors',
    cummax: 'real vectors',
    transpose: 'vectors, matrices',
    det: 'square matrices',
    qr: 'm x n, m >= n matrices',
    cross: 'real or complex vectors with `lengthof(a) == lengthof(b) == 3`',
    diag: 'matrices, integer',
    rowstack: 'vector of equal-length vectors',
    addaxes: 'array, non-negative integer, non-negative integer',
  };
  for (const head in spot) {
    assert.equal(COLLECTION_DOMAIN_HEADS.get(head).domains, spot[head], head);
  }
});

for (const [head, row] of COLLECTION_DOMAIN_HEADS) {
  test(`${head}.(v) over a scalar cell is a static error citing §07 "${row.section}"`, () => {
    const decl = BOOLEAN_HEADS.has(head) ? BOOLS : V;
    const extra = EXTRA_ARGS[head] ? ', ' + EXTRA_ARGS[head] : '';
    const errs = errorsOf(`${decl}\ny = ${head}.(v${extra})\n`);
    assert.equal(errs.length, 1, head + ': expected exactly one error, got '
      + JSON.stringify(errs));
    const msg = errs[0];
    assert.ok(msg.startsWith(head + ': a broadcast applies its head to one ELEMENT'),
      head + ': message shape — ' + msg);
    assert.ok(msg.includes('spec §07 "' + row.section + '"'),
      head + ': cites its §07 table — ' + msg);
    assert.ok(msg.includes(row.domains), head + ': quotes its Domains cell — ' + msg);
    assert.ok(msg.includes('spec §03 defines no rank-0 array'),
      head + ': names why a scalar is not an array — ' + msg);
  });
}

test('the diagnostic is attached to the broadcast call site', () => {
  // Diagnostic lines are 0-based here, so the second source line is line 1.
  // At BASE this program compiled clean and `y` evaluated to `[1.0, 2.0]`.
  const r = processSource('v = [1.0, 2.0]\ny = sum.(v)\n');
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 1);
  assert.ok(errs[0].loc, 'carries a loc');
  assert.equal(errs[0].loc.start.line, 1);
  assert.equal(errs[0].loc.end.line, 1);
  // The span covers `sum.(v)`, not the whole binding.
  assert.equal(errs[0].loc.start.col, 4);
});

test('a rank-2 operand refuses too — its cells are scalars, not rows', () => {
  // §04 requires all axes of an array argument to be loop axes, so a [2, 3]
  // matrix has TWO axes and therefore scalar cells. At BASE `sum.(M)`
  // returned the matrix unchanged and `maximum.(M)` a matrix of nulls.
  const M = 'M = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])';
  for (const head of ['sum', 'mean', 'maximum', 'l2norm', 'cumsum']) {
    const errs = errorsOf(`${M}\ny = ${head}.(M)\n`);
    assert.equal(errs.length, 1, head);
    assert.ok(errs[0].includes('spec §07'), head + ': ' + errs[0]);
  }
});

test('the remedy offers `aggregate` for exactly §04\'s ten eligible reductions', () => {
  // §04 "Multi-axis aggregation": "The eligible built-ins are `sum`, `prod`,
  // `mean`, `var`, `std`, `maximum`, `minimum`, `median`, `lany` and `lall`."
  // §07 "Cumulative operations" says the scans "are not eligible reductions
  // for multi-axis aggregation", so offering it there would send the reader
  // into a second refusal.
  assert.equal(AGGREGATE_ELIGIBLE_HEADS.size, 10);
  for (const [head] of COLLECTION_DOMAIN_HEADS) {
    const decl = BOOLEAN_HEADS.has(head) ? BOOLS : V;
    const extra = EXTRA_ARGS[head] ? ', ' + EXTRA_ARGS[head] : '';
    const msg = errorsOf(`${decl}\ny = ${head}.(v${extra})\n`)[0];
    assert.equal(msg.includes('aggregate('), AGGREGATE_ELIGIBLE_HEADS.has(head),
      head + ': aggregate remedy offered iff §04 admits it — ' + msg);
  }
});

test('quantile\'s remedy is the two-argument bare form', () => {
  // `quantile(v)` would be an arity error, so the remedy names both arguments.
  const msg = errorsOf(`${V}\ny = quantile.(v, 0.5)\n`)[0];
  assert.ok(msg.includes('`quantile(xs, p)`'), msg);
  assert.ok(!msg.includes('aggregate('), 'quantile is not one of §04\'s ten');
});

test('every spelling of the same call refuses', () => {
  // §04: "`f.(<args>)` lowers to `broadcast(f, <args>)`", and
  // "`broadcasted(f)(args) ≡ broadcast(f, args)`". All three are one call, so
  // the refusal cannot be spelled around. At BASE each returned `[1.0, 2.0]`.
  for (const call of ['sum.(v)', 'broadcast(sum, v)', 'broadcasted(sum)(v)']) {
    const errs = errorsOf(`v = [1.0, 2.0]\ny = ${call}\n`);
    assert.equal(errs.length, 1, call + ': ' + JSON.stringify(errs));
    assert.ok(errs[0].startsWith('sum: a broadcast'), call + ': ' + errs[0]);
  }
});

test('a user-fn head wrapping the reduction refuses too', () => {
  // `tot = fn(sum(_))` puts a `self` ref in the head slot, so the head name
  // is a binding rather than a builtin. One level of inlining resolves it.
  // At BASE `tot.(v)` returned `[1.0, 2.0, 3.0]` — the operand unchanged.
  const errs = errorsOf('v = [1.0, 2.0, 3.0]\ntot = fn(sum(_))\ny = tot.(v)\n');
  assert.equal(errs.length, 1, JSON.stringify(errs));
  assert.ok(errs[0].startsWith('sum: a broadcast'), errs[0]);
  // The same head over a nested operand is legal and right.
  assertClose(
    valueOfY('vv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\n'
      + 'tot = fn(sum(_))\ny = tot.(vv)\n'),
    [6, 15], 'tot.(vv)');
  // A user fn that is NOT a collection-domain head is untouched.
  assertClose(
    valueOfY('v = [1.0, 2.0]\nw = [3.0, 4.0]\nmySum = (a, b) -> a + b\ny = mySum.(v, w)\n'),
    [4, 6], 'mySum.(v, w)');
});

test('a COMPOUND broadcast body is outside the rule — a known gap, not a silent pass', () => {
  // The gate resolves the head only when the body is the head applied to the
  // head's own params, in order. `fn(sum(_) + 1.0)` has `add` at its
  // outermost node, so the inner `sum` over a scalar cell is NOT refused —
  // the same scope the Rust rule has, where such a body is a `functionof`
  // head its own rule does not see either.
  //
  // This is the bare-call domain gap in per-cell clothing: `sum(<scalar>)`
  // types clean anywhere, dotted or not, because `types.ts` declares the slot
  // `any()`. Unchanged from BASE, where this program also evaluated to
  // `[1.0, 1.0]`. Pinned so closing it is a deliberate act; tracked as a LEFT
  // item in flatppl-dev/TODO-flatppl-js.md.
  const src = 'v = [1.0, 2.0]\ny = fn(sum(_) + 1.0).(v)\n';
  assert.deepEqual(errorsOf(src), []);
  assertClose(valueOfY(src), [1, 1], 'compound body (still wrong, still pinned)');
  // The bare calls it rests on produce NUMBERS, not merely a clean type, which
  // is the measured half of that card.
  assert.equal(valueOfY('y = sum(2.0)\n'), 0, 'sum(<scalar>) = 0');
  assert.equal(valueOfY('y = lany(true)\n'), false, 'lany(<scalar>) = false');
  // `transpose(<scalar>)` is `inferTransposeAdjoint`'s silent `T.failed` with no
  // diagnostic — closed on the broadcast path, still open for a bare call.
  assert.deepEqual(errorsOf('y = transpose(2.0)\n'), []);
});

test('a scalar in the FIRST argument refuses even when a later argument is a collection', () => {
  // Only the first argument's domain drives the refusal, and `quantile`'s
  // first argument is the data. Reversed, the constant lands in the data
  // slot and the call is out of domain regardless of the operand.
  const errs = errorsOf(
    'vv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]\ny = quantile.(0.5, vv)\n');
  assert.equal(errs.length, 1);
  assert.ok(errs[0].startsWith('quantile: a broadcast'), errs[0]);
});

// =====================================================================
// 2. The carve-outs and the elementwise heads still type
// =====================================================================

test('min / max admit a scalar per element (§07 "Elementary functions", `reals`)', () => {
  assertClose(valueOfY(`${V}\ny = max.(v, 0.0)\n`), [1, 2, 3, 4], 'max.(v, 0.0)');
  assertClose(valueOfY(`${V}\ny = min.(v, 2.5)\n`), [1, 2, 2.5, 2.5], 'min.(v, 2.5)');
});

test('cat admits a scalar per element and is not refused', () => {
  // §07 "Array and table operations" gives `cat` the cell "scalars, vectors,
  // or records", and its entry states: "`cat(scalar1, scalar2, ...)` with all
  // scalar arguments produces a vector of those scalars." So a per-element
  // `cat` of two scalars is well-formed where a per-element `sum` of one
  // scalar is not.
  assert.deepEqual(errorsOf(`${V}\ny = cat.(v, v)\n`), []);
});

test('ordinary elementwise heads are untouched', () => {
  assertClose(valueOfY('v = [0.0, 1.0]\ny = exp.(v)\n'),
    [1, Math.E], 'exp.(v)');
  assertClose(valueOfY('v = [1.0, 2.0]\nw = [10.0, 20.0]\ny = v .+ w\n'),
    [11, 22], 'v .+ w');
});

test('the bare call is unaffected — only the broadcast is refused', () => {
  assert.equal(valueOfY(`${V}\ny = sum(v)\n`), 10);
  assertClose(valueOfY(`${V}\ny = cumsum(v)\n`), [1, 3, 6, 10], 'cumsum(v)');
  assert.equal(valueOfY('v = [true, false]\ny = lany(v)\n'), true);
});

// =====================================================================
// 3. The nested case — typed, and numerically right
// =====================================================================

const VV = 'vv = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]';

// Expected values from numpy / scipy over [1, 2, 3] and [4, 5, 6]
// independently, NOT from this engine.
const NESTED_EXPECTED: Record<string, number[]> = {
  sum: [6, 15],
  mean: [2, 5],
  prod: [6, 120],
  var: [1, 1],                       // ddof = 1, spec §07's 1/(n-1)
  std: [1, 1],
  maximum: [3, 6],
  minimum: [1, 4],
  median: [2, 5],
  lengthof: [3, 3],
  sizeof: [3, 3],
  indicesof: [1, 2, 3, 1, 2, 3],
  indicesof0: [0, 1, 2, 0, 1, 2],
  cumsum: [1, 3, 6, 4, 9, 15],
  cumprod: [1, 2, 6, 4, 20, 120],
  cummax: [1, 2, 3, 4, 5, 6],
  cummin: [1, 1, 1, 4, 4, 4],
  l1norm: [6, 15],
  l2norm: [3.7416573867739413, 8.774964387392123],
  linfnorm: [3, 6],
  l1unit: [
    0.16666666666666666, 0.3333333333333333, 0.5,
    0.26666666666666666, 0.3333333333333333, 0.4,
  ],
  l2unit: [
    0.2672612419124244, 0.5345224838248488, 0.8017837257372732,
    0.45584230583855184, 0.5698028822981898, 0.6837634587578276,
  ],
  logsumexp: [3.4076059644443806, 6.407605964444381],
  softmax: [
    0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
    0.09003057317038046, 0.24472847105479764, 0.6652409557748218,
  ],
  logsoftmax: [
    -2.4076059644443806, -1.4076059644443806, -0.4076059644443806,
    -2.4076059644443806, -1.4076059644443806, -0.4076059644443806,
  ],
  reverse: [3, 2, 1, 6, 5, 4],
  transpose: [1, 2, 3, 4, 5, 6],     // a transposed vector, same storage
  adjoint: [1, 2, 3, 4, 5, 6],       // real operand: adjoint == transpose
  self_outer: [
    1, 2, 3, 2, 4, 6, 3, 6, 9,
    16, 20, 24, 20, 25, 30, 24, 30, 36,
  ],
  diagmat: [
    1, 0, 0, 0, 2, 0, 0, 0, 3,
    4, 0, 0, 0, 5, 0, 0, 0, 6,
  ],
};

for (const head in NESTED_EXPECTED) {
  test(`${head}.(vv) applies per inner vector`, () => {
    assertClose(valueOfY(`${VV}\ny = ${head}.(vv)\n`),
      NESTED_EXPECTED[head], head + '.(vv)');
  });
}

test('quantile.(vv, 0.5) applies per inner vector', () => {
  assertClose(valueOfY(`${VV}\ny = quantile.(vv, 0.5)\n`), [2, 5],
    'quantile.(vv, 0.5)');
});

test('the nested result carries the outer axis in its type', () => {
  const r = processSource(`${VV}\ny = sum.(vv)\n`);
  const t = r.loweredModule.bindings.get('y').inferredType;
  assert.equal(t.kind, 'array');
  assert.deepEqual(t.shape, [2]);
  assert.equal(t.elem.kind, 'scalar');
});

// A vector of MATRICES: two axes below the loop axis, so each cell is a
// matrix and the linear-algebra heads are in domain. At BASE `sum.(MM)`
// returned `[2, 4]` — the first entry of each matrix — and
// `transpose.(MM)` / `diag.(MM)` produced no value at all.
const MM = 'M1 = rowstack([[2.0, 1.0], [1.0, 3.0]])\n'
  + 'M2 = rowstack([[4.0, 0.0], [0.0, 5.0]])\n'
  + 'MM = [M1, M2]';

// numpy over [[2,1],[1,3]] and [[4,0],[0,5]] independently.
const MATRIX_CELL_EXPECTED: Record<string, number[]> = {
  sum: [7, 9],
  mean: [1.75, 2.25],
  maximum: [3, 5],
  det: [5, 20],
  logabsdet: [1.6094379124341005, 2.995732273553991],
  trace: [5, 9],
  inv: [0.6, -0.2, -0.2, 0.4, 0.25, 0, 0, 0.2],
  lower_cholesky: [
    1.4142135623730951, 0, 0.7071067811865475, 1.5811388300841898,
    2, 0, 0, 2.23606797749979,
  ],
  row_gram: [5, 5, 5, 10, 16, 0, 0, 25],
  col_gram: [5, 5, 5, 10, 16, 0, 0, 25],
  transpose: [2, 1, 1, 3, 4, 0, 0, 5],
  diag: [2, 3, 4, 5],
};

for (const head in MATRIX_CELL_EXPECTED) {
  test(`${head}.(MM) applies per inner matrix`, () => {
    assertClose(valueOfY(`${MM}\ny = ${head}.(MM)\n`),
      MATRIX_CELL_EXPECTED[head], head + '.(MM)');
  });
}

test('a matrix cell under a vector-domain head refuses on its signature', () => {
  // §07 gives `cumsum` "vectors" and `self_outer` "vectors"; a matrix cell is
  // out of domain either way. The refusal comes from the SIGNATURE, not the
  // domain gate — the gate keys on the head's FIRST argument being a scalar
  // (or, for the four collection-of-collections heads, a flat array), so this
  // is not the general "any out-of-domain cell" check its old name implied.
  for (const head of ['cumsum', 'self_outer']) {
    const errs = errorsOf(`${MM}\ny = ${head}.(MM)\n`);
    assert.equal(errs.length, 1, head + ': ' + JSON.stringify(errs));
    assert.ok(!errs[0].includes('a broadcast applies its head to one ELEMENT'),
      head + ': the signature caught it, not the domain gate — ' + errs[0]);
  }
});

test('a FLAT cell refuses under the four collection-of-collections heads', () => {
  // §07 gives `rowstack`/`colstack` "vector of equal-length vectors",
  // `joinblocks` "array of equal-shaped arrays", `blockdiagmat` "vector of
  // matrices". A `[3]` vector of SCALARS per cell is none of those, and §03
  // keeps the two shapes distinct. At BASE `rowstack.(vv)` and `colstack.(vv)`
  // both answered `{shape: [2, 0, 0], data: []}` — two empty 0x0 matrices.
  for (const head of ['rowstack', 'colstack', 'joinblocks', 'blockdiagmat']) {
    const errs = errorsOf(`${VV}\ny = ${head}.(vv)\n`);
    assert.equal(errs.length, 1, head + ': ' + JSON.stringify(errs));
    assert.ok(errs[0].startsWith(head + ': a broadcast'), head + ': ' + errs[0]);
    assert.ok(errs[0].includes('keeps a vector of vectors distinct from a flat array'),
      head + ': names why a flat cell is out of domain — ' + errs[0]);
  }
});

test('rowstack over an IN-DOMAIN nested cell still answers with empties', () => {
  // The gate refuses an out-of-domain FLAT cell; it does not fix `rowstack`
  // under a broadcast. `vvv`'s cells ARE vectors of vectors, so this is in
  // domain and legal, and it returns empty 0x0 blocks where the bare
  // `rowstack(vv)` returns the correct `[2, 3]`. Pre-existing and unchanged;
  // pinned so it is honest rather than silent, and carded.
  const legal = valueOfY('vvv = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]\n'
    + 'y = rowstack.(vvv)\n');
  assert.deepEqual(Array.from(legal.data), [], 'still empty — if this changes, drop the card');
  // The bare call on the same shape of input is correct, which is what makes
  // the broadcast path the defect.
  assertClose(valueOfY(`${VV}\ny = rowstack(vv)\n`), [1, 2, 3, 4, 5, 6], 'rowstack(vv)');
});

test('a held-constant first argument is described as held, not as an element', () => {
  // §04 "Non-collection inputs": a scalar constant is "not iterated over but
  // held constant while collection arguments are iterated over". Saying "the
  // elements here are real scalars" would describe the constant as if it were
  // a cell — the message-accuracy nit the Rust rule carries.
  const errs = errorsOf(`${VV}\ny = quantile.(0.5, vv)\n`);
  assert.equal(errs.length, 1);
  assert.ok(errs[0].includes('argument 1 here is real, held constant and never iterated'),
    errs[0]);
  // The iterated case still reads as elements.
  const el = errorsOf('v = [1.0, 2.0]\ny = sum.(v)\n')[0];
  assert.ok(el.includes('the elements here are real scalars'), el);
});

// =====================================================================
// The forward hazard
// =====================================================================

// A `broadcast(functionof(<op>(_arg1_)), vv)` IR, for the structural
// dissolver matcher.
function singleOpBroadcastIR(op: string): any {
  return {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof', params: ['_arg1_'],
        paramKwargs: ['arg1'],
        body: { kind: 'call', op, args: [{ kind: 'ref', ns: '%local', name: '_arg1_' }] },
      },
      { kind: 'ref', ns: 'self', name: 'vv' },
    ],
  };
}

// Synthetic bindings for the dissolver's structural matcher: a name → inferred
// type map with a resolved fixed phase, which is all `_argTypeAndPhase` reads.
function syntheticBindings(types: Record<string, any>): Map<string, any> {
  const m = new Map<string, any>();
  for (const n in types) m.set(n, { inferredType: types[n], phase: 'fixed', ir: null });
  return m;
}

function multiArgBroadcastIR(op: string, names: string[]): any {
  const params = names.map((_, i) => `_arg${i + 1}_`);
  return {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof', params,
        paramKwargs: params.map((_, i) => 'arg' + (i + 1)),
        body: {
          kind: 'call', op,
          args: params.map((p) => ({ kind: 'ref', ns: '%local', name: p })),
        },
      },
      ...names.map((n) => ({ kind: 'ref', ns: 'self', name: n })),
    ],
  };
}

test('a FLAT operand never dissolves, at any rank', () => {
  // `_collectionHeadSlicesPerCell` must not be satisfiable by a rank that
  // merely COINCIDES with `argRanks[k] + 1`. An earlier version summed nesting
  // depth, which cannot tell "one loop axis over rank-k cells" from "a genuine
  // flat rank-(k+1) array" — both come to k+1 — and so admitted `diagmat` over
  // a real `[2, 3]` and `det` over a `[2, 2, 2]`, whose cells are SCALARS and
  // which §04 makes a static error. Nothing reached a wrong number (the domain
  // gate or an op signature caught every route), but the condition was not the
  // proof it claimed to be, so it is now the two facts it needs: exactly ONE
  // outer axis, and a FLAT cell of exactly the op's logical rank.
  const dissolver = require('../dissolver.ts');
  const scalar = { kind: 'scalar', prim: 'real' };
  const arr = (rank: number, shape: any[], elem: any) =>
    ({ kind: 'array', rank, shape, elem });

  const nestedVec = arr(1, [2], arr(1, [3], scalar));           // [2] of [3]
  const nestedMat = arr(1, [2], arr(2, [2, 2], scalar));        // [2] of [2,2]
  const flatMat = arr(2, [2, 3], scalar);                       // a real [2,3]
  const flatMatDyn = arr(2, ['%dynamic', 3], scalar);
  const flatCube = arr(3, [2, 2, 2], scalar);                   // a real [2,2,2]
  const nestedNested = arr(1, [2], arr(1, [2], arr(1, [2], scalar)));
  const twoOuterAxes = arr(2, [2, 3], arr(1, [4], scalar));

  const dissolves = (op: string, argTypes: any[]) => {
    const names = argTypes.map((_, i) => 'x' + i);
    const types: Record<string, any> = {};
    argTypes.forEach((t, i) => { types['x' + i] = t; });
    return dissolver._tryDissolveSingleOp(
      multiArgBroadcastIR(op, names), syntheticBindings(types)) !== null;
  };

  // Sound: one nesting axis over a flat cell at the op's logical rank.
  assert.ok(dissolves('diagmat', [nestedVec]), 'diagmat over [2] of [3]');
  assert.ok(dissolves('self_outer', [nestedVec]), 'self_outer over [2] of [3]');
  assert.ok(dissolves('cross', [nestedVec, nestedVec]), 'cross over two [2] of [3]');
  for (const op of ['det', 'trace', 'inv', 'lower_cholesky', 'logabsdet',
    'row_gram', 'col_gram']) {
    assert.ok(dissolves(op, [nestedMat]), op + ' over [2] of [2,2]');
  }

  // Unsound, and previously admitted: a genuine flat array whose cells are
  // scalars, at the coincident rank.
  assert.ok(!dissolves('diagmat', [flatMat]), 'diagmat over a real [2,3]');
  assert.ok(!dissolves('self_outer', [flatMat]), 'self_outer over a real [2,3]');
  assert.ok(!dissolves('cross', [flatMat, flatMat]), 'cross over two real [2,3]');
  assert.ok(!dissolves('diagmat', [flatMatDyn]), 'diagmat over [%dynamic,3]');
  for (const op of ['det', 'trace', 'inv', 'lower_cholesky']) {
    assert.ok(!dissolves(op, [flatCube]), op + ' over a real [2,2,2]');
  }
  // §03: a vector of vectors is not a matrix, so it is not `det`'s cell either.
  assert.ok(!dissolves('det', [nestedNested]), 'det over [2] of ([2] of [2])');
  // More than one outer axis: the dispatcher's rank test fails and it would be
  // handed the whole buffer.
  assert.ok(!dissolves('diagmat', [twoOuterAxes]), 'diagmat over two outer axes');
  // A pointwise head is unaffected by any of this.
  assert.ok(dissolves('neg', [nestedVec]), 'neg still dissolves');
  assert.ok(dissolves('add', [flatMat, flatMat]), 'add over two real [2,3]');
});

test('the dissolver keeps its own half of the rule', () => {
  // Refuse-don't-mislower must not depend on inference having run. The
  // dissolver rewrites `broadcast(<head>, X)` to the bare `<head>(X)` only
  // where the whole-value call provably applies the head per outer slice, so
  // a head that reaches the legacy whole-value impl stays a broadcast.
  const dissolver = require('../dissolver.ts');
  // `bindings: null` is the structural-matcher path: no arg annotations, so
  // `_collectionHeadSlicesPerCell` can prove nothing and every
  // collection-domain head must refuse.
  for (const op of ['cumsum', 'sum', 'transpose', 'det', 'diag', 'reverse']) {
    assert.equal(dissolver._tryDissolveSingleOp(singleOpBroadcastIR(op), null), null,
      op + ': must not dissolve to a bare whole-value call');
  }
  // A genuinely pointwise head still dissolves on the same path.
  assert.ok(dissolver._tryDissolveSingleOp(singleOpBroadcastIR('neg'), null),
    'neg is elementwise at any rank and still dissolves');
});

test('the unlowered collection heads refuse on their signature, not the domain rule', () => {
  // §07 carries four more tables with a collection-domain FIRST argument that
  // are deliberately NOT in this table, mirroring the Rust head set:
  // Convolution (`conv`, `crosscorr`), Binning (`bincounts`), Approximation
  // functions (`polynomial`, `bernstein`, `stepwise`) and Array and table
  // generation (`array`). Each refuses today for an unrelated reason — an
  // arity error, or a signature slot that rejects a scalar — so there is no
  // live mislowering.
  //
  // The safety is therefore the SIGNATURE, not the domain reasoning. That is
  // fragile in exactly the way this whole wave was about: fifteen heads in
  // the table type-checked a scalar cell precisely because their slot is
  // `any()`. If one of these slots is ever loosened the same way, the dotted
  // mislowering opens — see the companion assertion below.
  const unlowered = [
    'conv', 'crosscorr', 'bincounts', 'polynomial', 'bernstein', 'stepwise',
    'array',
  ];
  for (const head of unlowered) {
    assert.ok(!COLLECTION_DOMAIN_HEADS.has(head),
      head + ' is now in the table — move it out of this test');
    const errs = errorsOf(`${V}\ny = ${head}.(v)\n`);
    assert.equal(errs.length, 1, head + ': ' + JSON.stringify(errs));
    assert.ok(!errs[0].includes('a broadcast applies its head to one ELEMENT'),
      head + ': refuses on the domain rule, so it belongs IN the table now — '
      + errs[0]);
  }
});

test('a per-cell transposed vector keeps its tag instead of becoming a matrix', () => {
  // §03: "transposed vectors are a distinct type in FlatPPL"; §07 "Linear
  // algebra": "The transpose of a vector is a transposed vector (see arrays),
  // not a single-row matrix." The inferred type says
  // `array of transposed vector`, so the value must not be a tag-less `[2, 3]`
  // matrix — a `t` tag on THAT would mean matrix transpose, a different claim.
  // The cells therefore stay a per-cell list, each carrying its own tag.
  for (const [head, tag] of [['transpose', 'T'], ['adjoint', 'A']] as const) {
    const v = valueOfY(`${VV}\ny = ${head}.(vv)\n`);
    assert.ok(Array.isArray(v), head + ': a per-cell list, not one stacked Value');
    assert.equal(v.length, 2);
    assert.equal(v[0].t, tag, head + ': cell 0 keeps its tag');
    assert.equal(v[1].t, tag, head + ': cell 1 keeps its tag');
    assert.deepEqual(Array.from(v[0].data), [1, 2, 3]);
    assert.deepEqual(Array.from(v[1].data), [4, 5, 6]);
  }
});

test('a complex cell keeps its imaginary half', () => {
  // The stacker built `{shape, data}` and dropped everything else, so a complex
  // per-cell result was presented as its real parts — a wrong VALUE, not a lost
  // tag. Exercised directly on the exported helper: the reachable source route
  // is masked by an upstream defect (below), which is the point of testing the
  // helper.
  const { tryStackBroadcastCells } = require('../broadcast-shape.ts');
  const cx = (re: number[], im: number[]) => ({
    shape: [re.length], data: Float64Array.from(re),
    im: Float64Array.from(im), dtype: 'complex',
  });
  const stacked = tryStackBroadcastCells([cx([1, 2], [3, 4]), cx([5, 6], [7, 8])], [2]);
  assert.deepEqual(stacked.shape, [2, 2]);
  assert.deepEqual(Array.from(stacked.data), [1, 2, 5, 6]);
  assert.deepEqual(Array.from(stacked.im), [3, 4, 7, 8]);
  assert.equal(stacked.dtype, 'complex');
  // A structured complex cell too: `densify` expands `im` alongside `data`.
  const diagCx = (re: number[], im: number[]) => ({
    shape: [2, 2], data: Float64Array.from(re),
    im: Float64Array.from(im), dtype: 'complex', struct: 2,
  });
  const s2 = tryStackBroadcastCells([diagCx([1, 2], [9, 8]), diagCx([3, 4], [7, 6])], [2]);
  assert.deepEqual(Array.from(s2.data), [1, 0, 0, 2, 3, 0, 0, 4]);
  assert.deepEqual(Array.from(s2.im), [9, 0, 0, 8, 7, 0, 0, 6]);
  // MIXED real and complex cells are not one Value; guessing the missing half
  // as zero would be a wrong number, so they refuse.
  assert.equal(
    tryStackBroadcastCells(
      [{ shape: [2], data: Float64Array.from([1, 2]) }, cx([5, 6], [7, 8])], [2]),
    null, 'mixed real/complex refuses');
});

test('a cell whose buffer is shorter than its shape refuses instead of zero-filling', () => {
  // `densify` expands the one packed form the engine produces (diag-stored).
  // Anything still short is a representation this consumer cannot read, and
  // copying it into the full stride is exactly what put a diagonal in the
  // first row.
  const { tryStackBroadcastCells } = require('../broadcast-shape.ts');
  const short = () => ({ shape: [2, 2], data: Float64Array.from([1, 2, 3]) });
  assert.equal(tryStackBroadcastCells([short(), short()], [2]), null);
});

test('the complex nested operand is honest, and its wrongness is upstream', () => {
  // The reviewer's four-line repro. `zz = [a, b]` over two complex vectors is
  // where the imaginary half is lost: the ARRAY LITERAL drops `im`, so `zz`
  // already carries real parts only. Pinned here so the day the literal is
  // fixed, these values change and this test says so.
  const src = 'a = complex.([1.0, 2.0], [3.0, 4.0])\n'
    + 'b = complex.([5.0, 6.0], [7.0, 8.0])\n'
    + 'zz = [a, b]\n';
  // `a` alone is complex and correct.
  const a = valueOfY(src + 'y = a\n');
  assert.deepEqual(Array.from(a.im), [3, 4], 'a keeps its imaginary half');
  // The literal drops it. THIS is the defect, carded in TODO-flatppl-js.md.
  const zz = valueOfY(src + 'y = zz\n');
  assert.equal(zz.im, undefined,
    'the array literal still drops `im` — if this now passes, update the card');
  // Given that input, `adjoint.(zz)` is an honest per-cell list of tagged
  // vectors rather than a tag-less matrix of real parts, which is what it was.
  const adj = valueOfY(src + 'y = adjoint.(zz)\n');
  assert.ok(Array.isArray(adj), 'a per-cell list');
  assert.equal(adj[0].t, 'A');
  assert.equal(adj[1].t, 'A');
});

test('stacking a structured cell densifies it first', () => {
  // value.ts's `struct` tag section states the contract: "Any consumer without
  // a structured fast-path calls `densify(v)` first". `tryStackBroadcastCells`
  // is such a consumer and did not, so a diagonal cell — shape [n, n] with
  // only n entries in `data` — had its diagonal copied into the first n slots
  // of an n*n block and the rest left zero.
  //
  // No dotted spelling reaches this today: `diagmat.(vv)` is the one head that
  // returns a structured cell and it dissolves to a per-slice call instead.
  // The unit test is the reachable one, and it keeps the contract honest for
  // whatever consumer arrives next.
  const { tryStackBroadcastCells } = require('../broadcast-shape.ts');
  // Two 3x3 diagonal cells, in the sparse representation `diagmat` returns.
  const diag = (d: number[]) =>
    ({ shape: [3, 3], data: Float64Array.from(d), struct: 2 });
  const stacked = tryStackBroadcastCells([diag([1, 2, 3]), diag([4, 5, 6])], [2]);
  assert.deepEqual(stacked.shape, [2, 3, 3]);
  assert.deepEqual(Array.from(stacked.data), [
    1, 0, 0, 0, 2, 0, 0, 0, 3,
    4, 0, 0, 0, 5, 0, 0, 0, 6,
  ]);
});

test('conv / crosscorr would still be dissolved — the hazard, pinned', () => {
  // `conv` and `crosscorr` sit in the dissolver's DISSOLVE_AT_ANY_RANK_OPS and
  // are NOT in the collection-domain table, so the dissolver still rewrites
  // `broadcast(conv, X)` to the bare `conv(X)`. Nothing reaches that today
  // because their signature slot rejects a scalar cell first.
  //
  // This assertion flips the moment either head joins the table — the
  // dissolution starts returning null and this test fails, which is the
  // prompt to delete it and move the head into the refusal group above. It
  // fails LOUDLY rather than leaving the hazard to a comment nobody reads.
  const dissolver = require('../dissolver.ts');
  for (const op of ['conv', 'crosscorr']) {
    assert.ok(!COLLECTION_DOMAIN_HEADS.has(op),
      op + ' joined the table — delete this test and move it above');
    const out = dissolver._tryDissolveSingleOp(singleOpBroadcastIR(op), null);
    assert.ok(out && out.op === op,
      op + ': the hazard is closed — update this test');
  }
});
