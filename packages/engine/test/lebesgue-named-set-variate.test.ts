'use strict';

// =====================================================================
// lebesgue-named-set-variate.test.ts — a set bound to a NAME types as
// its RHS (spec §03 sets + §04 name references + §06 Lebesgue)
// =====================================================================
//
// §03 "Cartesian product": `cartprod(S1, S2, …)` — "Each member is the
// `cat` of one element per component set … The resulting set is a set of
// arrays, not a set of tuples". Two scalar `interval` components therefore
// give a 2-element real array, so §06 `Lebesgue(support = S)` — "the
// canonical continuous reference measure on the support set `S`" — has a
// length-2 array variate.
//
// §04 makes a name reference an expression form: "Expressions are single or
// nested calls that bind expressions (literal or by name reference) to
// inputs of callables". So `square = cartprod(…)` + `Lebesgue(support =
// square)` denotes the SAME measure as the inline spelling and must carry
// the same type. Before this fix the named spelling fell back to the
// default SCALAR variate in both static readers (`typeinfer.setValueType`
// and `value-set.setExprValueset`), which made a correct 2-vector point
// raise "logdensityof: arg 2 expects real, got array of real (length 2)"
// while the model still evaluated correctly — the derivation reader had
// always seen the box, since `lift` inlines the set.
//
// §06 leaves the variate untouched through `weighted` ($d\nu = f \cdot dM$),
// `logweighted`, `normalize` ($M / Z$) and `truncate` ($\nu(A) = M(A \cap
// S)$), and states outright that "Measure algebra operations require their
// operands to share the same variate space (same type and dimension)".
//
// Cross-engine check (NOT the oracle — the spec above is): flatppl-rust
// `flatppl infer --level shape` gives BOTH spellings
// `(%measure (%domain (%array 1 (2) (%scalar real))) (%mass %finite))`,
// following the self-ref in `crates/infer/src/ops.rs` `set_element_type_at`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { processSource, computeSubDAG, orchestrator } = require('../index.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const T = require('../types.ts');
const VS = require('../value-set.ts');

const BOX2 = 'measure over array of real (length 2)';

function typesOf(src: string) {
  const r = processSource(src);
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error');
  const shown: Record<string, string> = {};
  for (const [k, b] of r.bindings) {
    if (b.inferredType) shown[k] = T.show(b.inferredType);
  }
  return { r, errors, shown };
}

function massOf(src: string, name: string) {
  const t = processSource(src).bindings.get(name).inferredType;
  return t.mass;
}

// =====================================================================
// 1. The named spelling matches the inline one
// =====================================================================

test('named cartprod set: Lebesgue(support = square) is a 2-vector variate', () => {
  const named = typesOf(`flatppl_compat = "0.1"
square = cartprod(interval(0.0, 1.0), interval(-1.0, 1.0))
L = Lebesgue(support = square)
`);
  const inline = typesOf(`flatppl_compat = "0.1"
L = Lebesgue(support = cartprod(interval(0.0, 1.0), interval(-1.0, 1.0)))
`);
  assert.deepEqual(named.errors, []);
  assert.deepEqual(inline.errors, []);
  assert.equal(named.shown.L, BOX2);
  assert.equal(inline.shown.L, BOX2, 'inline spelling unchanged');
});

test('named cartpow set: Lebesgue(support = cube) is a 3-vector variate', () => {
  const { errors, shown } = typesOf(`flatppl_compat = "0.1"
cube = cartpow(interval(0.0, 1.0), 3)
L = Lebesgue(support = cube)
`);
  assert.deepEqual(errors, []);
  assert.equal(shown.L, 'measure over array of real (length 3)');
});

test('a chained set alias resolves through every hop', () => {
  const { errors, shown } = typesOf(`flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))
alias = sq
alias2 = alias
L = Lebesgue(support = alias2)
`);
  assert.deepEqual(errors, []);
  assert.equal(shown.L, BOX2);
});

test('elementof and Counting read a named set too', () => {
  const { errors, shown } = typesOf(`flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))
grid = cartprod(integers, integers)
x = elementof(sq)
C = Counting(support = grid)
`);
  assert.deepEqual(errors, []);
  assert.equal(shown.x, 'array of real (length 2)');
  assert.equal(shown.C, 'measure over array of integer (length 2)');
});

// =====================================================================
// 2. §06 operators leave the variate alone
// =====================================================================

test('weighted / logweighted / normalize / truncate preserve the box variate', () => {
  const { errors, shown } = typesOf(`flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))
L = Lebesgue(support = sq)
w = weighted(2.0, L)
lw = logweighted(0.5, L)
n = normalize(L)
t = truncate(L, sq)
nt = normalize(truncate(L, sq))
wf(a, b) = a * b
wfn = weighted(wf, L)
`);
  assert.deepEqual(errors, []);
  for (const k of ['L', 'w', 'lw', 'n', 't', 'nt', 'wfn']) {
    assert.equal(shown[k], BOX2, k + ' keeps the 2-vector variate');
  }
});

// =====================================================================
// 3. The arity/type check on logdensityof now behaves
// =====================================================================

test('a 2-vector point against a named-set box measure is accepted', () => {
  const { errors, shown } = typesOf(`flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))
L = normalize(Lebesgue(support = sq))
d = logdensityof(L, [0.5, 0.25])
`);
  assert.deepEqual(errors, []);
  assert.equal(shown.d, 'real');
});

test('a SCALAR point against a box measure is rejected, naming the vector length', () => {
  const { errors } = typesOf(`flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))
L = normalize(Lebesgue(support = sq))
d = logdensityof(L, 0.5)
`);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].message,
    'logdensityof: arg 2 expects array of real (length 2), got real');
  assert.ok(errors[0].loc, 'the diagnostic is located');
});

test('a WRONG-LENGTH point against a box measure is rejected', () => {
  const { errors } = typesOf(`flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))
L = normalize(Lebesgue(support = sq))
d = logdensityof(L, [0.5, 0.25, 0.125])
`);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].message,
    'logdensityof: arg 2 expects array of real (length 2), got array of real (length 3)');
});

// =====================================================================
// 4. The SECOND reader — value-set.ts — follows the ref as well
// =====================================================================
//
// The §03 boundedness lattice drives the §06 Lebesgue mass rule, so a
// named bounded box must classify `%finite`, not `%unknown`. This is the
// other half of the dual-reader split: `setExprValueset` returned UNKNOWN
// for a ref to a user binding, exactly as `setValueType` returned the
// default scalar.

test('a named bounded box classifies %finite mass, like the inline spelling', () => {
  const named = `flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 2.0), interval(0.0, 3.0))
L = Lebesgue(support = sq)
`;
  const inline = `flatppl_compat = "0.1"
L = Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))
`;
  assert.equal(massOf(named, 'L'), 'finite');
  assert.equal(massOf(inline, 'L'), 'finite');
});

test('a named UNBOUNDED box classifies exactly as the inline one', () => {
  const named = `flatppl_compat = "0.1"
plane = cartprod(reals, reals)
L = Lebesgue(support = plane)
`;
  const inline = `flatppl_compat = "0.1"
L = Lebesgue(support = cartprod(reals, reals))
`;
  // Not `finite`: §06 leaves normalize/totalmass undefined over an
  // unbounded box, and the engine's class for that is `locallyfinite`.
  // The point of the row is PARITY with the inline spelling.
  assert.equal(massOf(named, 'L'), 'locallyfinite');
  assert.equal(massOf(inline, 'L'), 'locallyfinite');
});

test('the value set of a named-set support renders as the product set', () => {
  const r = processSource(`flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 1.0), interval(2.0, 4.0))
L = Lebesgue(support = sq)
x = elementof(sq)
`);
  const vsOf = (name: string) => {
    const b = r.loweredModule.bindings.get(name);
    return VS.toSexpr(b && b.rhs && b.rhs.meta && b.rhs.meta.valueset);
  };
  assert.equal(vsOf('L'), '(cartprod (interval 0.0 1.0) (interval 2.0 4.0))');
  assert.equal(vsOf('x'), '(cartprod (interval 0.0 1.0) (interval 2.0 4.0))');
});

test('a named interval reaches a distribution support (Uniform)', () => {
  const r = processSource(`flatppl_compat = "0.1"
win = interval(0.0, 5.0)
u = Uniform(support = win)
`);
  const b = r.loweredModule.bindings.get('u');
  assert.equal(VS.toSexpr(b.rhs.meta.valueset), '(interval 0.0 5.0)');
});

test('setExprValueset without a ref resolver leaves a named set UNKNOWN', () => {
  // The pure-library default: `derivations.ts` queries `distributionSupport`
  // with no LoweredModule in scope, so it cannot follow a ref and must not
  // guess. Third argument omitted deliberately.
  const ref = { kind: 'ref', ns: 'self', name: 'sq' };
  assert.equal(VS.setExprValueset(ref), VS.UNKNOWN);
  assert.equal(VS.setExprValueset(ref, undefined, () => null), VS.UNKNOWN);
});

// =====================================================================
// 5. Guards — a cycle and an over-long chain must not spin or throw
// =====================================================================

test('a cyclic set binding falls back instead of recursing forever', () => {
  const { shown } = typesOf(`flatppl_compat = "0.1"
a = b
b = a
L = Lebesgue(support = a)
`);
  // §06 default: the support could not be resolved, so the scalar default
  // stands. The point is that inference TERMINATES with a verdict.
  assert.equal(shown.L, 'measure over real');
});

test('a self-referential set binding falls back', () => {
  const { shown } = typesOf(`flatppl_compat = "0.1"
s = s
L = Lebesgue(support = s)
`);
  assert.equal(shown.L, 'measure over real');
});

test('a long surface alias chain collapses at lowering and still resolves', () => {
  // Alias resolution rewrites `s40 = s39 = … = s0` to a single ref at
  // lowering, so no surface spelling reaches value-set.ts's hop cap. This
  // row pins that the whole chain still types, whatever the lowering does
  // with it.
  const N = 40;
  let src = 'flatppl_compat = "0.1"\ns0 = cartprod(interval(0.0, 1.0), interval(0.0, 1.0))\n';
  for (let i = 1; i <= N; i++) src += `s${i} = s${i - 1}\n`;
  src += `L = Lebesgue(support = s${N})\n`;
  const { errors, shown } = typesOf(src);
  assert.deepEqual(errors, []);
  assert.equal(shown.L, BOX2);
  assert.equal(massOf(src, 'L'), 'finite');
});

test('setExprValueset gives up on a cyclic ref resolver rather than spinning', () => {
  // The hop cap is value-set.ts's ONLY cycle guard — typeinfer's own
  // in-progress set does not cover a call made through the library API, and
  // a resolver that keeps answering with a ref would otherwise recurse
  // forever. Exercised directly because the surface language cannot get
  // here (alias resolution collapses chains first).
  const selfRef = (name: string) => ({ kind: 'ref', ns: 'self', name });
  assert.equal(
    VS.setExprValueset(selfRef('a'), undefined, (n: string) => selfRef(n)),
    VS.UNKNOWN);
});

test('a support naming a non-set binding falls back to the default variate', () => {
  const { shown } = typesOf(`flatppl_compat = "0.1"
notaset = 3.0
L = Lebesgue(support = notaset)
`);
  assert.equal(shown.L, 'measure over real');
});

test('a support naming NO binding falls back and keeps the unbound-name error', () => {
  const r = processSource(`flatppl_compat = "0.1"
L = Lebesgue(support = nope)
`);
  assert.deepEqual(r.diagnostics.map((d: any) => d.message), ["Undefined variable 'nope'"]);
  assert.equal(T.show(r.bindings.get('L').inferredType), 'measure over real');
});

// =====================================================================
// 6. The reported fixture, end to end
// =====================================================================

test('dminus-to-3pi fixture: both measures type as a 2-vector variate', () => {
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'dminus-to-3pi-amplitude.flatppl'), 'utf8');
  const { errors, shown } = typesOf(src);
  assert.deepEqual(errors, [], 'the fixture types clean');
  assert.equal(shown.phase_space, BOX2);
  assert.equal(shown.amplitude_measure, BOX2);
});

test('dminus-to-3pi fixture: the reported logdensityof query no longer errors', () => {
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'dminus-to-3pi-amplitude.flatppl'), 'utf8')
    + '\nscore = logdensityof(amplitude_measure, [1.0, 0.35])\n';
  const { errors, shown } = typesOf(src);
  assert.deepEqual(errors, []);
  assert.equal(shown.score, 'real');
});

test('the viewer DAG bubble reads the vector variate for both measures', () => {
  // The bubble text is `types.show(b.inferredType)`, pre-rendered in
  // `dag.ts` so the webview does not duplicate the rendering. Both nodes
  // read "measure over real" before the fix.
  const src = fs.readFileSync(
    path.join(__dirname, 'fixtures', 'dminus-to-3pi-amplitude.flatppl'), 'utf8');
  const r = processSource(src);
  const { nodes } = computeSubDAG(r.bindings, 'amplitude_measure', {});
  const bubbleOf = (id: string) => {
    const n = nodes.find((node: any) => node.id === id);
    assert.ok(n, id + ' is in the DAG');
    return n.inferredType;
  };
  assert.equal(bubbleOf('phase_space'), BOX2);
  assert.equal(bubbleOf('amplitude_measure'), BOX2);
});

// =====================================================================
// 7. No number moved — closed-form oracle, not a cross-spelling compare
// =====================================================================
//
// The fix touches two STATIC readers only; the derivation reader already
// classified `lebesguebox` for the named spelling (`lift` inlines the set).
// Pinned against the closed form rather than against the inline spelling:
// `normalize(Lebesgue([0,2] × [0,3]))` has constant density 1 / area, so
// logdensityof = −log 6 at every interior point, and the box mass is
// 2 · 3 = 6 exactly.

const NAMED_BOX = `flatppl_compat = "0.1"
sq = cartprod(interval(0.0, 2.0), interval(0.0, 3.0))
L = Lebesgue(support = sq)
P = normalize(L)
`;

test('a named-set box still classifies as a box with the closed-form mass', () => {
  const proc = processSource(NAMED_BOX);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(proc.bindings);
  const d = built.derivations.L;
  assert.equal(d.kind, 'lebesguebox');
  assert.deepEqual(d.axes.map((a: any) => [a.lo, a.hi]), [[0, 2], [0, 3]]);
  // Box volume 2 · 3 = 6, so logTotalmass = log 6 exactly.
  assert.ok(Math.abs(d.logTotalmass - Math.log(6)) < 1e-15,
    'logTotalmass ' + d.logTotalmass + ' ≠ log 6');
});

test('a named-set box scores the uniform density −log 6 at an interior point', async () => {
  // `normalize(Lebesgue([0,2] × [0,3]))` is uniform on the box, so its
  // density is the constant 1/6 everywhere inside and
  // logdensityof = −log 6 = −1.791759469228055 at every interior point.
  // Closed form, not a cross-spelling comparison.
  const { ctx } = makeMatCtx(
    NAMED_BOX + 'd = logdensityof(P, [1.0, 1.0])\n', { sampleCount: 64, rootSeed: 11 });
  const scored = await ctx.getMeasure('d');
  assert.ok(Math.abs(scored.samples[0] + Math.log(6)) < 1e-12,
    'logdensityof ' + scored.samples[0] + ' ≠ −log 6 = ' + (-Math.log(6)));
});
