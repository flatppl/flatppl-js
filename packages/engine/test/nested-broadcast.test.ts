'use strict';

// Nested-vector broadcast semantics (spec §04 + design conversation
// 2026-05-27 on Ref-wrapped arguments).
//
// Background. The spec requires all collection inputs to a broadcast
// to have the SAME number of axes (no implicit axis insertion). The
// open question is what `[C]` MEANS when `C` is itself a vector: is
// the resulting object
//   (a) a 2-D matrix of shape [1, k] — in which case `polyeval.([C], X)`
//       is a rank mismatch and rejected by spec; or
//   (b) a length-1 nested-vector — i.e. a length-1 outer collection
//       whose only element is the rank-1 vector C — in which case
//       `polyeval.([C], X)` aligns rank-1 outer with rank-1 X and
//       broadcasts the singleton outer axis to X's length.
//
// FlatPPL chooses (b). Quoting the user: "[C] should NOT have shape
// [1,3] - nested vectors are NOT matrices in FlatPPL, semantically."
// Storage may still be flat for performance ("collapsed" nested
// broadcast), but the OUTER rank — the rank that broadcast iterates
// over — is the outer dimension count.
//
// This file pins the spec-relevant behaviour for callable broadcast
// with nested-vector arguments:
//
//   - Ref-wrap: `f.([C], X)` — hold C constant, iterate over X.
//   - Pairwise vector-arg broadcast: `f.([V1,V2,V3], [W1,W2,W3])`.
//   - Matrix held constant: `f.([M], V_batch)`.
//   - Singleton expansion + outer-rank alignment errors.
//   - Closure-over-scope as a held-constant alternative (no Ref).
//
// References from sibling ecosystems for the conceptual model:
//   - Julia: `Ref(C)` in a broadcast holds C as a 0-D container.
//   - JAX `vmap(in_axes=None)`: hold an argument constant.
//   - NumPy generalized ufuncs (gufuncs) with core signatures
//     `(n),(n)->()` express "iterate outer loop dims, pass inner
//     core dims whole".
//   - StableHLO `broadcast_in_dim` makes the loop / core split
//     explicit at the IR level.
//
// All four conventions agree on the v1 semantics this file pins:
// the outer collection axis is iterated, inner ranks are passed
// whole to the callable.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');
const { buildDerivations } = require('../orchestrator.ts');
const { toJS } = require('./_value-helpers.ts');

// Run a FlatPPL source, return the final fixedValues map. Asserts the
// program is error-free (any analyzer / typeinfer error fails the test
// immediately so we don't have to inspect downstream NaNs).
function run(src: string): Map<string, any> {
  const r = processSource(src);
  const errs = (r.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    'source compiled with errors: '
    + errs.map((e: any) => e.message).join(' | '));
  const ds = buildDerivations(r.bindings);
  return ds.fixedValues || new Map();
}

function get(values: Map<string, any>, name: string): any {
  assert.ok(values.has(name),
    'expected ' + name + ' to land in fixedValues (deterministic over '
    + 'fixed inputs); got keys [' + Array.from(values.keys()).join(', ') + ']');
  return toJS(values.get(name));
}

// =====================================================================
// A. Single-level Ref-wrap (motivating case + variants)
// =====================================================================

test('nested-broadcast: polyeval.([C], X) — Ref-wrap motivating case', () => {
  // The polyeval body uses `coeffs` as a vector (via indicesof0 and
  // elementwise mul) and `x` as a scalar (via dotted power). The Ref
  // wrap `[C]` holds C as a length-1 nested-vector whose only element
  // is the whole 3-vector C; X is a length-3 scalar vector. The outer
  // axes align (1 vs 3, singleton expansion), so the broadcast yields
  // a length-3 result where each cell is polyeval(C, X[k]).
  const v = run(`
polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
C = [2.3, 1.5, 0.7]
X = [1.1, 2.2, 3.3]
Y = polyeval.([C], X)
`);
  const Y = get(v, 'Y');
  // Reference values: 2.3 + 1.5*x + 0.7*x²
  const ref = [1.1, 2.2, 3.3].map((x) => 2.3 + 1.5 * x + 0.7 * x * x);
  assert.deepEqual(Y.length, 3, 'expected length-3 result');
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(Y[i] - ref[i]) < 1e-9,
      'Y[' + i + '] = ' + Y[i] + ', expected ' + ref[i]);
  }
});

test('nested-broadcast: scalar callable + Ref-wrapped vector — `len.([C])`', () => {
  // The simplest Ref-wrap: a callable that takes a vector arg and
  // returns its length. Outer axis = 1, single call. Validates the
  // dispatch path even when there's no second arg to align with.
  const v = run(`
len = xs -> lengthof(xs)
C = [10.0, 20.0, 30.0, 40.0]
N = len.([C])
`);
  // Outer axis = 1 (the Ref-wrap), result is a length-1 vector
  // containing lengthof(C) = 4.
  assert.deepEqual(get(v, 'N'), [4]);
});

test('nested-broadcast: held-constant via closure (no Ref) — comparison case', () => {
  // Spec §04 lists scalars, functions, kernels, measures, and
  // likelihoods as held-constant; closures-over-scope are NOT in that
  // list. So the only way to "hold a vector constant" in a broadcast
  // is the Ref-wrap idiom. This test pins the alternative: the
  // closure-over-scope form (C captured by mkfn) — there the broadcast
  // never sees C as an arg at all, so it's loop-invariant by virtue of
  // not being on the args list.
  const v = run(`
C = [2.3, 1.5, 0.7]
mkfn = x -> sum(C .* x .^ indicesof0(C))
X = [1.1, 2.2, 3.3]
Y = mkfn.(X)
`);
  const Y = get(v, 'Y');
  const ref = [1.1, 2.2, 3.3].map((x) => 2.3 + 1.5 * x + 0.7 * x * x);
  for (let i = 0; i < 3; i++) {
    assert.ok(Math.abs(Y[i] - ref[i]) < 1e-9,
      'closure form Y[' + i + '] = ' + Y[i] + ', expected ' + ref[i]);
  }
});

// =====================================================================
// B. Per-cell-different-vector arg (genuine nested-vector iteration)
// =====================================================================

test('nested-broadcast: dot.([V1,V2,V3], [W1,W2,W3]) — pairwise dot', () => {
  // Three different left-vectors paired with three different right-
  // vectors. Outer rank-1 alignment, length 3 each. Per cell the
  // callable receives the i-th INNER vector of each arg whole — same
  // semantics as a NumPy gufunc with core signature `(k),(k)->()`.
  const v = run(`
dot = (a, b) -> sum(a .* b)
V1 = [1.0, 2.0, 3.0]
V2 = [4.0, 5.0, 6.0]
V3 = [7.0, 8.0, 9.0]
W1 = [1.0, 0.0, 0.0]
W2 = [0.0, 1.0, 0.0]
W3 = [0.0, 0.0, 1.0]
Y = dot.([V1, V2, V3], [W1, W2, W3])
`);
  // Picks the diagonal entries: V1·W1=1, V2·W2=5, V3·W3=9.
  assert.deepEqual(get(v, 'Y'), [1, 5, 9]);
});

test('nested-broadcast: singleton expansion — [C] (len 1) vs [V1,V2,V3]', () => {
  // The Ref-wrap has outer length 1; the other arg has outer length 3.
  // Singleton expansion repeats C across the 3 outer positions.
  const v = run(`
dot = (a, b) -> sum(a .* b)
C = [1.0, 2.0, 3.0]
V1 = [1.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0]
V3 = [0.0, 0.0, 1.0]
Y = dot.([C], [V1, V2, V3])
`);
  // dot(C, V_k) just picks C[k]: [C[0], C[1], C[2]] = [1, 2, 3].
  assert.deepEqual(get(v, 'Y'), [1, 2, 3]);
});

// =====================================================================
// C. Higher-rank inner core (matrix held constant)
// =====================================================================

test('nested-broadcast: matrix held via Ref, vector-of-vectors batched', () => {
  // A more interesting gufunc-style case: signature `(m,n),(n)->(m)`.
  // The matrix M is held via [M] (a length-1 outer over an inner [2,3]);
  // V_batch is a length-3 outer over inner [3]s. The body computes
  // M·v per cell. Outer rank-1 alignment, singleton expansion of M.
  const v = run(`
matvec = (M, v) -> M * v
M = array([1.0, 0.0, 0.0,  0.0, 1.0, 0.0], [2, 3], [1, 2])
V1 = [10.0, 20.0, 30.0]
V2 = [40.0, 50.0, 60.0]
V3 = [70.0, 80.0, 90.0]
Y = matvec.([M], [V1, V2, V3])
`);
  // M is a 2×3 matrix selecting the first two coords of v. So per cell
  // M·V_k = V_k[0:2]. Y is a length-3 outer of length-2 inner vectors.
  // toJS yields a nested array.
  const Y = get(v, 'Y');
  assert.deepEqual(Y, [[10, 20], [40, 50], [70, 80]]);
});

// =====================================================================
// D. Error cases — outer-rank / outer-size validation
// =====================================================================

test('nested-broadcast: outer-size mismatch is rejected', () => {
  // Both args have outer rank 1, but sizes 2 vs 3 (neither is 1) —
  // singleton expansion does not apply, so this must NOT produce a
  // valid Y value. buildDerivations currently silently skips bindings
  // whose evaluator throws (orchestrator pattern at derivations.ts:
  // `try { ... } catch (_) { continue }`), so the runtime signal we
  // can observe is "Y didn't land in fixedValues". (Surfacing it as
  // an analyzer diagnostic would need static outer-shape inference —
  // a separate follow-up; v1 pins runtime detection.)
  const r = processSource(`
dot = (a, b) -> sum(a .* b)
V1 = [1.0, 2.0]
V2 = [3.0, 4.0]
W1 = [5.0]
W2 = [6.0]
W3 = [7.0]
Y = dot.([V1, V2], [W1, W2, W3])
`);
  const ds = buildDerivations(r.bindings);
  const fv = ds.fixedValues || new Map();
  assert.ok(!fv.has('Y'),
    'Y must not land in fixedValues when broadcast outer sizes are '
    + 'incompatible; got ' + JSON.stringify(fv.get('Y')));
});

test('nested-broadcast: ragged Ref-wrap arg is rejected', () => {
  // Nested-vector args must have a uniform inner shape (no ragged).
  // [V1, V2] where V1, V2 have different lengths must be rejected at
  // the vector(...) construction or at the broadcast call.
  let threw = false;
  try {
    run(`
dot = (a, b) -> sum(a .* b)
V1 = [1.0, 2.0]
V2 = [3.0, 4.0, 5.0]
Y = dot.([V1, V2], [[1.0,1.0], [1.0,1.0,1.0]])
`);
  } catch (_e: any) {
    threw = true;
  }
  assert.ok(threw, 'ragged nested-vector should have thrown');
});

// =====================================================================
// E. Deeper nesting (v2 — multi-axis nested-vector outer loops)
// =====================================================================
//
// The v1 classifier walks the full JS-array nesting depth and treats
// every level as an outer loop axis. So `[[C]]` is outerRank=2 with
// outerShape=[1, 1] (two singleton outer axes wrapping the rank-1
// vector C), and composes naturally with a flat 2-D collection arg.
//
// These tests pin the spec-relevant behaviour for >1 nesting levels.

test('nested-broadcast (v2): polyeval.([[C]], X_2D) — double Ref over 2-D X', () => {
  // Hold C across BOTH outer axes of a 2-D X. `[[C]]` has nesting
  // depth 2 (outer + inner JS-array layer) and innermost = C; its
  // outerShape is [1, 1]. X_2D has shape [2, 3] (a flat 2-D Value)
  // with outerRank=2 and outerShape=[2, 3]. Both have outerRank 2;
  // singleton expansion broadcasts [1, 1] → [2, 3]. Per cell:
  // polyeval(C, X_2D[i, j]).
  const v = run(`
polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
C = [2.3, 1.5, 0.7]
X_2D = array([1.0, 2.0, 3.0,  4.0, 5.0, 6.0], [2, 3], [1, 2])
Y = polyeval.([[C]], X_2D)
`);
  const Y = get(v, 'Y');
  // Expected: Y[i, j] = 2.3 + 1.5*X[i,j] + 0.7*X[i,j]^2.
  const xs = [[1, 2, 3], [4, 5, 6]];
  for (let i = 0; i < 2; i++) {
    for (let j = 0; j < 3; j++) {
      const x = xs[i][j];
      const expected = 2.3 + 1.5 * x + 0.7 * x * x;
      const got = Y[i][j];
      assert.ok(Math.abs(got - expected) < 1e-9,
        'Y[' + i + ',' + j + '] = ' + got + ', expected ' + expected);
    }
  }
});

test('nested-broadcast (v2): dot.([[C]], [[V1, V2], [V3, V4]]) — Ref over 2-D nest', () => {
  // Hold C across a 2-D nested-vector of inner vectors. Both args
  // have outerRank=2; the LHS has outerShape=[1, 1] (singleton in
  // both axes); the RHS has outerShape=[2, 2] with innermost rank-1
  // Values. Per cell: dot(C, V_ij).
  const v = run(`
dot = (a, b) -> sum(a .* b)
C = [1.0, 1.0, 1.0]
V1 = [1.0, 2.0, 3.0]
V2 = [4.0, 5.0, 6.0]
V3 = [7.0, 8.0, 9.0]
V4 = [10.0, 11.0, 12.0]
Y = dot.([[C]], [[V1, V2], [V3, V4]])
`);
  // dot(C=ones, V) = sum(V); expected matrix:
  //   [[ sum(V1), sum(V2) ], [ sum(V3), sum(V4) ]]
  //   = [[ 6, 15 ], [ 24, 33 ]]
  assert.deepEqual(get(v, 'Y'), [[6, 15], [24, 33]]);
});

test('nested-broadcast (v2): outer-rank mismatch ([C] vs X_2D) is rejected', () => {
  // Ref-wrap depth-1 vs flat 2-D ⇒ outerRanks 1 vs 2 — same-#-axes
  // rule fires (spec §04). Runtime detection ⇒ binding absent from
  // fixedValues (same convention as the v1 outer-size-mismatch case).
  const r = processSource(`
polyeval = (coeffs, x) -> sum(coeffs .* x .^ indicesof0(coeffs))
C = [2.3, 1.5, 0.7]
X_2D = array([1.0, 2.0, 3.0,  4.0, 5.0, 6.0], [2, 3], [1, 2])
Y = polyeval.([C], X_2D)
`);
  const ds = buildDerivations(r.bindings);
  const fv = ds.fixedValues || new Map();
  assert.ok(!fv.has('Y'),
    'outer-rank-1 [C] paired with outer-rank-2 X_2D must not produce '
    + 'a fixedValues entry; got ' + JSON.stringify(fv.get('Y')));
});

test('nested-broadcast (v2): rectangularity is enforced at every depth', () => {
  // `[[V1, V2], [V3]]` — rectangularity break at the inner level
  // (row 0 has length 2, row 1 has length 1). Must be rejected by
  // the classifier rather than producing a silently wrong result.
  const r = processSource(`
dot = (a, b) -> sum(a .* b)
C = [1.0, 1.0, 1.0]
V1 = [1.0, 2.0, 3.0]
V2 = [4.0, 5.0, 6.0]
V3 = [7.0, 8.0, 9.0]
Y = dot.([[C]], [[V1, V2], [V3]])
`);
  const ds = buildDerivations(r.bindings);
  const fv = ds.fixedValues || new Map();
  assert.ok(!fv.has('Y'),
    'ragged inner level must not produce a Y entry; got '
    + JSON.stringify(fv.get('Y')));
});
