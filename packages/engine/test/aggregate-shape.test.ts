'use strict';

// =====================================================================
// aggregate-shape.test.ts — canonical form of an aggregate IR node
// =====================================================================
//
// Pins the contract in `aggregate-shape.ts`:
//   - typeinfer annotates every aggregate IR node with its canonical
//     form (`ir.meta.aggregateCanonical = {outAxes, reduceAxes,
//     canonicalAxes, axisLengths, fullyResolved}`).
//   - The runtime evaluators consume the annotation; the body walk
//     that previously lived in 3 places (single-point runtime
//     `_inferAggregateAxisLengths` + `_collectInScopeAxisNames`,
//     atom-batched `_inferAxisLengthsN`, typeinfer's `inferAggregate`)
//     collapses to one source.
//
// (P1 of the broadcast / aggregate / batching consolidation — see
// TODO-flatppl-js.md "In-flight: broadcast / aggregate / batching
// consolidation".)

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const aggregateShape = require('../aggregate-shape.ts');

function aggIRFromSource(src: string, name: string) {
  const ctx = processSource(src);
  const errs = (ctx.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0,
    `unexpected errors: ${JSON.stringify(errs)}`);
  const lb = ctx.loweredModule.bindings.get(name);
  assert.ok(lb, `binding ${name} not found`);
  return lb.rhs;
}

// =====================================================================
// 1. typeinfer populates the annotation
// =====================================================================

test('aggregate-shape: typeinfer annotates ir.meta.aggregateCanonical', () => {
  const ir = aggIRFromSource(`
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
B = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
C[.i, .k] := A[.i, .j] * B[.j, .k]
`, 'C');
  assert.ok(ir.meta && ir.meta.aggregateCanonical,
    'meta.aggregateCanonical populated by typeinfer');
  const c = ir.meta.aggregateCanonical;
  assert.deepEqual(c.outAxes, ['i', 'k']);
  assert.deepEqual(c.reduceAxes, ['j']);
  assert.deepEqual(c.canonicalAxes, ['i', 'k', 'j']);
  // A is [2, 3]; B is [3, 2]. Lengths resolved from typeinfer's
  // static type analysis (both literals → fully resolved).
  assert.equal(c.axisLengths.i, 2);
  assert.equal(c.axisLengths.k, 2);
  assert.equal(c.axisLengths.j, 3);
  assert.equal(c.fullyResolved, true);
});

// =====================================================================
// 2. Empty output_axes (full reduction) annotation
// =====================================================================

test('aggregate-shape: empty output_axes — reduceAxes covers all body axes', () => {
  const ir = aggIRFromSource(`
u = [1.0, 2.0, 3.0]
v = [4.0, 5.0, 6.0]
d[] := u[.i] * v[.i]
`, 'd');
  const c = ir.meta.aggregateCanonical;
  assert.deepEqual(c.outAxes, []);
  assert.deepEqual(c.reduceAxes, ['i']);
  assert.deepEqual(c.canonicalAxes, ['i']);
  assert.equal(c.axisLengths.i, 3);
});

// =====================================================================
// 3. Repeated axis (trace, einsum 'ii') — annotation collapses to one
//    occurrence of the axis, fully resolved.
// =====================================================================

test('aggregate-shape: repeated axis on a single source — one entry in canonical form', () => {
  const ir = aggIRFromSource(`
A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
tr[] := A[.i, .i]
`, 'tr');
  const c = ir.meta.aggregateCanonical;
  assert.deepEqual(c.outAxes, []);
  assert.deepEqual(c.reduceAxes, ['i']);
  // Both dims of A are length 3 — typeinfer records the first-seen.
  assert.equal(c.axisLengths.i, 3);
});

// =====================================================================
// 4. Nested aggregates: each has its own axis scope
// =====================================================================

test('aggregate-shape: nested aggregate does NOT leak its axes outward', () => {
  // Outer aggregate uses .i; inner uses .k. Outer's canonical form
  // should NOT include .k (the inner aggregate is opaque to the outer
  // axis walk).
  const ir = aggIRFromSource(`
A = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
inner = aggregate(sum, [.k], A[1, 1, .k])
outer[.i] := A[.i, 1, 1] + inner
`, 'outer');
  const c = ir.meta.aggregateCanonical;
  assert.deepEqual(c.outAxes, ['i']);
  assert.deepEqual(c.reduceAxes, []);
  assert.ok(!('k' in c.axisLengths),
    'inner aggregate axis .k must not leak into outer canonical form');
});

// =====================================================================
// 5. Pure-helper unit tests (collectAxesInScope, structuralCanonicalAxes)
// =====================================================================

test('aggregate-shape: collectAxesInScope returns axes in DFS first-occurrence order', () => {
  // Body shape: `A[.a, .b] * B[.c, .a]` — .a appears first.
  const ir = {
    kind: 'call', op: 'mul',
    args: [
      { kind: 'call', op: 'get', args: [
        { kind: 'ref', ns: 'self', name: 'A' },
        { kind: 'axis', name: 'a' },
        { kind: 'axis', name: 'b' },
      ]},
      { kind: 'call', op: 'get', args: [
        { kind: 'ref', ns: 'self', name: 'B' },
        { kind: 'axis', name: 'c' },
        { kind: 'axis', name: 'a' },
      ]},
    ],
  };
  const axes = aggregateShape.collectAxesInScope(ir);
  assert.deepEqual(axes, ['a', 'b', 'c']);
});

test('aggregate-shape: structuralCanonicalAxes derives reduceAxes by set difference', () => {
  const ir = {
    kind: 'call', op: 'aggregate',
    args: [
      { kind: 'ref', name: 'sum' },
      { kind: 'call', op: 'vector', args: [
        { kind: 'axis', name: 'i' },
      ]},
      { kind: 'call', op: 'get', args: [
        { kind: 'ref', ns: 'self', name: 'A' },
        { kind: 'axis', name: 'i' },
        { kind: 'axis', name: 'j' },
      ]},
    ],
  };
  const skel = aggregateShape.structuralCanonicalAxes(ir);
  assert.deepEqual(skel.outAxes, ['i']);
  assert.deepEqual(skel.reduceAxes, ['j']);
  assert.deepEqual(skel.canonicalAxes, ['i', 'j']);
});

test('aggregate-shape: structuralCanonicalAxes returns null for non-aggregate IR', () => {
  const ir = { kind: 'call', op: 'mul', args: [] };
  assert.equal(aggregateShape.structuralCanonicalAxes(ir), null);
});

// =====================================================================
// 6. Re-annotation: shape drift triggers recompute, same shape is idempotent
// =====================================================================

test('aggregate-shape: re-annotate same shape returns the cached object (identity)', () => {
  const ir = aggIRFromSource(`
A = [[1.0, 2.0], [3.0, 4.0]]
B = [[1.0, 0.0], [0.0, 1.0]]
C[.i, .k] := A[.i, .j] * B[.j, .k]
`, 'C');
  const a = aggregateShape.getCanonical(ir);
  const b = aggregateShape.getCanonical(ir);
  assert.equal(a, b, 're-annotation returns the same object by identity');
});
