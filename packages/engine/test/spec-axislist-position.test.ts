'use strict';

// =====================================================================
// spec-axislist-position.test.ts — pin spec §05's axis-list POSITION rule
// =====================================================================
//
// Spec §05 "Note on axis names":
//   "The grammar likewise admits `AxisList` as a `Primary`, but it is
//    legal only as the `output_axes` argument of an `aggregate` or
//    `metricsum` call and as the axis-list binder of an
//    `AggregateBinding` or `MetricsumBinding`; anywhere else it is a
//    static error. Unlike `ArrayLiteral`, `AxisList` may be empty:
//    `aggregate(sum, [], expr)` denotes full reduction to a scalar."
//
// Four legal positions, which the parser's own desugaring collapses to
// two by the time the analyzer runs — `C[.i] := e` becomes
// `aggregate(sum, [.i], e)` and `g: C[.mu^] := e` becomes
// `metricsum(g, [.mu^], e)`, so both binders arrive as argument 1 of the
// corresponding call.
//
// The rule is separate from §05's per-axis rule ("an axis name is legal
// only as an entry in `aggregate`'s `output_axes` axis list, as an index
// inside `[...]` within the body, or as a binder on the left-hand side
// of `:=`"). Two consequences make the axis-list rule bite where the
// per-axis rule cannot:
//
//   1. The EMPTY bracket. §05's `ArrayLiteral` production requires at
//      least one `Expression`, so `[]` can only be an `AxisList` — it
//      holds no axis name for the per-axis rule to catch. An empty
//      vector therefore has no literal spelling; `zeros(0)` spells one.
//   2. The OTHER argument slots of an aggregation. The per-axis rule is
//      scoped to the enclosing `aggregate(...)` / `metricsum(...)` as a
//      whole, so an axis name in the metric slot or the body slot looks
//      legal to it. Position is what distinguishes them.
//
// This file pins one legal and one illegal case per position class, and
// asserts the refusal is LOCATED and quotes §05.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function errors(src: string): any[] {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

/** The §05 axis-list-position refusals in `src`, in source order. */
function axisListRefusals(src: string): any[] {
  return errors(src).filter((d: any) => /is not legal here: an axis list/.test(d.message));
}

function assertAccepted(src: string) {
  const errs = errors(src);
  assert.equal(errs.length, 0,
    `expected no diagnostic; got: ${errs.map((d: any) => d.message).join('; ')}`);
}

/**
 * Assert exactly one §05 axis-list refusal, that it names the bracket as
 * written, and that it points at that bracket's own span.
 */
function assertRefusedAt(
  src: string, bracket: string, line: number, col: number,
) {
  const ds = axisListRefusals(src);
  assert.equal(ds.length, 1,
    `expected one axis-list refusal for ${JSON.stringify(src)}; got `
    + `${ds.length}: ${ds.map((d: any) => d.message).join('; ')}`);
  const d = ds[0];
  assert.match(d.message, new RegExp(`Axis list '${bracket.replace(/[[\]^.*+?$(){}|\\]/g, '\\$&')}'`));
  assert.ok(d.loc && d.loc.start, 'refusal carries a location');
  assert.equal(d.loc.start.line, line, `refusal line for ${bracket}`);
  assert.equal(d.loc.start.col, col, `refusal column for ${bracket}`);
}

// =====================================================================
// 1. The four legal positions stay legal — non-empty and empty
// =====================================================================

test('§05 legal position 1: aggregate output_axes', () => {
  assertAccepted(
    'A = rowstack([[1.0, 2.0], [3.0, 4.0]])\n'
    + 's = aggregate(sum, [.i], A[.i, .j])\n'
    + 'y ~ Normal(mu = s[1], sigma = 1)\n');
});

test('§05 legal position 1: aggregate output_axes may be empty', () => {
  // "Unlike `ArrayLiteral`, `AxisList` may be empty: `aggregate(sum, [],
  // expr)` denotes full reduction to a scalar."
  assertAccepted(
    'A = [1.0, 2.0, 3.0]\n'
    + 's = aggregate(sum, [], A[.i])\n'
    + 'y ~ Normal(mu = s, sigma = 1)\n');
});

test('§05 legal position 2: metricsum output_axes', () => {
  assertAccepted(
    'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\n'
    + 'V = [1.0, 2.0]\n'
    + 's = metricsum(g, [.mu^], V[.mu^])\n'
    + 'y ~ Normal(mu = s[1], sigma = 1)\n');
});

test('§05 legal position 2: metricsum output_axes may be empty', () => {
  assertAccepted(
    'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\n'
    + 'V = [1.0, 2.0]\n'
    + 's = metricsum(g, [], V[.mu^] * V[.mu_])\n'
    + 'y ~ Normal(mu = s, sigma = 1)\n');
});

test('§05 legal position 3: AggregateBinding axis-list binder', () => {
  assertAccepted(
    'A = rowstack([[1.0, 2.0], [3.0, 4.0]])\n'
    + 'C[.i] := A[.i, .j]\n'
    + 'y ~ Normal(mu = C[1], sigma = 1)\n');
});

test('§05 legal position 3: AggregateBinding binder may be empty', () => {
  assertAccepted(
    'A = [1.0, 2.0, 3.0]\n'
    + 'C[] := A[.i]\n'
    + 'y ~ Normal(mu = C, sigma = 1)\n');
});

test('§05 legal position 4: MetricsumBinding axis-list binder', () => {
  assertAccepted(
    'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\n'
    + 'V = [1.0, 2.0]\n'
    + 'g: C[.mu^] := V[.mu^]\n'
    + 'y ~ Normal(mu = C[1], sigma = 1)\n');
});

test('§05 legal position 4: MetricsumBinding binder may be empty', () => {
  assertAccepted(
    'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\n'
    + 'V = [1.0, 2.0]\n'
    + 'g: C[] := V[.mu^] * V[.mu_]\n'
    + 'y ~ Normal(mu = C, sigma = 1)\n');
});

// =====================================================================
// 2. Illegal positions — one per class, located and §05-quoting
// =====================================================================

test('§05: axis list as a binding right-hand side is refused', () => {
  assertRefusedAt('x = [.i, .j]\n', '[.i, .j]', 0, 4);
});

test('§05: axis list as a tilde-binding right-hand side is refused', () => {
  assertRefusedAt('y ~ [.i]\n', '[.i]', 0, 4);
});

test('§05: axis list as a function argument is refused', () => {
  assertRefusedAt('x = sum([.i, .j])\n', '[.i, .j]', 0, 8);
});

test('§05: axis list as an operator operand is refused', () => {
  assertRefusedAt('A = [1.0, 2.0]\nx = A + [.i]\n', '[.i]', 1, 8);
});

test('§05: axis list as a distribution parameter is refused', () => {
  assertRefusedAt('y ~ Normal(mu = [.i], sigma = 1)\n', '[.i]', 0, 16);
});

test('§05: axis list as a record field is refused', () => {
  assertRefusedAt('r = record(a = [.i, .j])\n', '[.i, .j]', 0, 15);
});

test('§05: axis list as a tuple element is refused', () => {
  assertRefusedAt('x = ([.i], 2.0)\n', '[.i]', 0, 5);
});

test('§05: axis list as a named-function body is refused', () => {
  assertRefusedAt('f(a) = [.i]\n', '[.i]', 0, 7);
});

test('§05: an indexed axis list is refused', () => {
  assertRefusedAt('x = [.i, .j][1]\n', '[.i, .j]', 0, 4);
});

test('§05: a negated axis list is refused', () => {
  assertRefusedAt('x = -[.i]\n', '[.i]', 0, 5);
});

test('§05: the refusal echoes both variance markers', () => {
  // §05's `VarianceMarker` is part of the axis, so the message has to
  // render `.mu^` and `.nu_` as the user spelled them.
  assertRefusedAt('x = [.mu_]\n', '[.mu_]', 0, 4);
  assertRefusedAt('x = [.mu^, .nu_]\n', '[.mu^, .nu_]', 0, 4);
});

// ---- the slots the per-axis rule cannot see (inside an aggregation) ----

test('§05: axis list in aggregate\'s f_reduction slot is refused', () => {
  // Argument 0, not 1 — the per-axis rule widens its scope to the whole
  // call, so only position tells this slot from output_axes.
  const ds = axisListRefusals('A = [1.0, 2.0]\ns = aggregate([.i], [.i], A[.i])\n');
  assert.equal(ds.length, 1, ds.map((d: any) => d.message).join('; '));
  assert.equal(ds[0].loc.start.line, 1);
  assert.equal(ds[0].loc.start.col, 14);
});

test('§05: axis list in aggregate\'s expr slot is refused', () => {
  assertRefusedAt('s = aggregate(sum, [.i], [.i])\n', '[.i]', 0, 25);
});

test('§05: axis list in metricsum\'s metric slot is refused', () => {
  assertRefusedAt(
    'V = [1.0, 2.0]\ns = metricsum([.mu], [.mu^], V[.mu^])\n', '[.mu]', 1, 14);
});

test('§05: axis list in metricsum\'s expr slot is refused', () => {
  const ds = axisListRefusals(
    'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\ns = metricsum(g, [.mu^], [.mu^])\n');
  assert.equal(ds.length, 1, ds.map((d: any) => d.message).join('; '));
  assert.match(ds[0].message, /Axis list '\[\.mu\^\]'/);
  assert.equal(ds[0].loc.start.line, 1);
  assert.equal(ds[0].loc.start.col, 25);
});

test('§05: axis list as an AggregateBinding body is refused', () => {
  // `C[.i] := [.i]` — the binder is legal, the body is not.
  assertRefusedAt('C[.i] := [.i]\n', '[.i]', 0, 9);
});

test('§05: axis list as a MetricsumBinding body is refused', () => {
  const ds = axisListRefusals(
    'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\ng: C[.mu^] := [.mu^]\n');
  assert.equal(ds.length, 1, ds.map((d: any) => d.message).join('; '));
  assert.equal(ds[0].loc.start.line, 1);
  assert.equal(ds[0].loc.start.col, 14);
});

// =====================================================================
// 3. The empty bracket — no legal reading outside the four positions
// =====================================================================

test('§05: an empty bracket as a binding right-hand side is refused', () => {
  // §05's `ArrayLiteral` needs an element, so `[]` is an `AxisList` and
  // nothing else. An empty vector has no literal spelling.
  assertRefusedAt('x = []\n', '[]', 0, 4);
});

test('§05: an empty bracket as a function argument is refused', () => {
  assertRefusedAt('x = sum([])\n', '[]', 0, 8);
});

test('§05: an empty bracket as a distribution parameter is refused', () => {
  assertRefusedAt('y ~ Normal(mu = [], sigma = 1)\n', '[]', 0, 16);
});

test('§05: an empty bracket nested inside a legal aggregation is refused', () => {
  // The aggregation's own output_axes `[.i]` is legal; the `[]` inside the
  // body is not, so position is resolved per slot, not per call.
  const src = 'A = [1.0, 2.0]\ns = aggregate(sum, [.i], A[.i] + sum([]))\n';
  const ds = axisListRefusals(src);
  assert.equal(ds.length, 1, ds.map((d: any) => d.message).join('; '));
  assert.match(ds[0].message, /Axis list '\[\]'/);
  assert.equal(ds[0].loc.start.line, 1);
  assert.equal(ds[0].loc.start.col, 37);
});

test('§05: zeros(0) spells the empty vector the literal cannot', () => {
  // §07 "Empty inputs" governs empty-vector behaviour, so empty vectors
  // are values — they just have no literal spelling.
  assertAccepted('xs = zeros(0)\nn = l1norm(xs)\n');
});

// =====================================================================
// 4. The refusal replaces the per-axis report rather than adding to it
// =====================================================================

test('§05: an illegal axis list is refused once, not once per axis', () => {
  // `[.i, .j, .k]` violates the per-axis rule three times over. The
  // axis-list refusal owns the whole bracket, so the user sees one error.
  const src = 'x = [.i, .j, .k]\n';
  assert.equal(axisListRefusals(src).length, 1);
  const perAxis = errors(src).filter((d: any) =>
    /may only appear inside aggregate/.test(d.message));
  assert.equal(perAxis.length, 0,
    `expected no per-axis reports; got ${perAxis.length}`);
});

test('§05: a bare axis outside an aggregation still gets the per-axis rule', () => {
  // The axis-list rule must not swallow the per-axis rule — `.i` here is
  // not in a bracket at all.
  const ds = errors('x = .i + 1\n');
  assert.ok(ds.some((d: any) => /Axis name '\.i' may only appear inside aggregate/.test(d.message)),
    `expected the per-axis diagnostic; got: ${ds.map((d: any) => d.message).join('; ')}`);
});

test('§05: an array literal of non-axis elements is untouched', () => {
  // Only an all-axis-name (or empty) bracket is an `AxisList`; an
  // ordinary vector literal must keep parsing as an `ArrayLiteral`.
  assertAccepted('x = [1.0, 2.0, 3.0]\nn = l1norm(x)\n');
  assert.equal(axisListRefusals('x = [1.0, 2.0]\n').length, 0);
});
