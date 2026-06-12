'use strict';

// =====================================================================
// shape-fold.test.ts — shape-driven constant folding (§20.10)
// =====================================================================
//
// Pins the precomputation pass added in the dissolver that lifts
// spec §07 shape→value functions (`indicesof` / `indicesof0` /
// `sizeof` / `lengthof`) into literal IR when the input's shape is
// statically known. Architectural foundation for fusion (a) —
// broadcast-through-reductions — where the substituted body of a
// fused aggregate can then carry literal-vector index sources
// instead of recomputing them per cell.
//
// What's pinned:
//   1. Each of the four functions folds for a statically-shaped
//      array binding.
//   2. The cap (_SHAPE_FOLD_MAX_LEN) prevents IR-bloat for
//      pathologically large axes.
//   3. Dynamic shapes refuse the fold (the runtime path stays).
//   4. Nested folds work (the dissolver walks bottom-up so
//      `sizeof(<binding>)[0]` could pre-fold both ends, although
//      the engine doesn't fold get yet).
//   5. End-to-end: `Y = indicesof0(C)` for a literal C resolves to
//      the literal vector in the post-dissolve IR.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { _foldShapeCall, dissolveBindings } = require('../dissolver.ts');
const { processSource } = require('../index.ts');
const { buildDerivations } = require('../derivations.ts');

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

function mkBindings(entries: Array<[string, any]>) {
  const m = new Map();
  for (const [name, b] of entries) m.set(name, b);
  return m;
}

function arrayType(rank: number, shape: any[], prim = 'real') {
  return {
    kind: 'array',
    rank,
    shape,
    elem: { kind: 'scalar', prim },
  };
}

function ref(name: string) {
  return { kind: 'ref', ns: 'self', name };
}

// ---------------------------------------------------------------------
// 1. Each shape function folds for statically-shaped arrays
// ---------------------------------------------------------------------

test('indicesof0(X) folds to vector(lit_0…lit_{n-1}) for static shape [3]', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, [3]), phase: 'fixed' }],
  ]);
  const call = { kind: 'call', op: 'indicesof0', args: [ref('X')] };
  const folded = _foldShapeCall(call, bindings);
  assert.ok(folded, 'indicesof0 should fold');
  assert.equal(folded.kind, 'call');
  assert.equal(folded.op, 'vector');
  assert.equal(folded.args.length, 3);
  assert.deepEqual(folded.args.map((a: any) => a.value), [0, 1, 2]);
  for (const a of folded.args) assert.equal(a.numType, 'integer');
});

test('indicesof(X) folds to vector(lit_1…lit_n) for static shape [4]', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, [4]), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'indicesof', args: [ref('X')] }, bindings);
  assert.ok(folded, 'indicesof should fold');
  assert.equal(folded.args.length, 4);
  assert.deepEqual(folded.args.map((a: any) => a.value), [1, 2, 3, 4]);
});

test('lengthof(X) folds to literal integer n', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, [10]), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'lengthof', args: [ref('X')] }, bindings);
  assert.ok(folded, 'lengthof should fold');
  assert.equal(folded.kind, 'lit');
  assert.equal(folded.value, 10);
  assert.equal(folded.numType, 'integer');
});

test('sizeof(X) folds to literal vector of shape dims for rank-2 [3, 4]', () => {
  const bindings = mkBindings([
    ['M', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(2, [3, 4]), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'sizeof', args: [ref('M')] }, bindings);
  assert.ok(folded, 'sizeof should fold');
  assert.equal(folded.op, 'vector');
  assert.deepEqual(folded.args.map((a: any) => a.value), [3, 4]);
});

// ---------------------------------------------------------------------
// 2. Refusals: dynamic / unresolvable / unknown ops
// ---------------------------------------------------------------------

test('dynamic shape refuses fold (shape[0] === "%dynamic")', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, ['%dynamic']), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'indicesof0', args: [ref('X')] }, bindings);
  assert.equal(folded, null, 'dynamic shape stays on runtime path');
});

test('non-array type refuses fold', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: { kind: 'scalar', prim: 'real' }, phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'sizeof', args: [ref('X')] }, bindings);
  assert.equal(folded, null);
});

test('unrelated op (e.g. add) is not folded', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, [3]), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'add', args: [ref('X')] }, bindings);
  assert.equal(folded, null);
});

test('placeholder (%local) arg refuses fold', () => {
  // Inside a functionof body, `indicesof0(coeffs)` references a
  // %local. The pre-substitution fold can't see the actual array
  // shape; refuse cleanly. (Post-substitution fold happens when
  // fusion (a) sub-2 lands.)
  const folded = _foldShapeCall(
    { kind: 'call', op: 'indicesof0', args: [{ kind: 'ref', ns: '%local', name: 'coeffs' }] },
    new Map());
  assert.equal(folded, null);
});

test('non-call arg refuses fold (defensive)', () => {
  const folded = _foldShapeCall(
    { kind: 'call', op: 'indicesof0', args: [] }, new Map());
  assert.equal(folded, null);
});

// ---------------------------------------------------------------------
// 3. Length cap prevents IR bloat for pathological sizes
// ---------------------------------------------------------------------

test('length-cap: shape > 4096 refuses indicesof0 fold', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, [4097]), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'indicesof0', args: [ref('X')] }, bindings);
  assert.equal(folded, null, 'over-cap shapes stay on runtime path');
});

test('length-cap: shape == 4096 still folds (boundary)', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, [4096]), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'indicesof0', args: [ref('X')] }, bindings);
  assert.ok(folded, 'boundary shape folds');
  assert.equal(folded.args.length, 4096);
});

test('length-cap: lengthof is uncapped (returns one scalar regardless)', () => {
  const bindings = mkBindings([
    ['X', { ir: { kind: 'lit', value: 0 }, inferredType: arrayType(1, [99999]), phase: 'fixed' }],
  ]);
  const folded = _foldShapeCall(
    { kind: 'call', op: 'lengthof', args: [ref('X')] }, bindings);
  assert.ok(folded, 'lengthof folds regardless of size (one scalar)');
  assert.equal(folded.value, 99999);
});

// ---------------------------------------------------------------------
// 4. End-to-end: dissolveBindings runs the fold
// ---------------------------------------------------------------------

test('e2e: Y = indicesof0(C) folds to literal vector after dissolveBindings', () => {
  const proc = processSource(`
C = [2.3, 1.5, 0.7]
Y = indicesof0(C)
`);
  const out = buildDerivations(proc.bindings);
  const y = out.bindings.get('Y');
  assert.ok(y && y.ir);
  // After dissolution, Y's IR should be the folded vector literal.
  assert.equal(y.ir.kind, 'call');
  assert.equal(y.ir.op, 'vector');
  assert.equal(y.ir.args.length, 3);
  assert.deepEqual(y.ir.args.map((a: any) => a.value), [0, 1, 2]);
});

test('e2e: lengthof on a static [10]-vector binding folds to integer 10', () => {
  const proc = processSource(`
C = [2.3, 1.5, 0.7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
N = lengthof(C)
`);
  const out = buildDerivations(proc.bindings);
  const n = out.bindings.get('N');
  assert.ok(n && n.ir);
  assert.equal(n.ir.kind, 'lit');
  assert.equal(n.ir.value, 10);
  assert.equal(n.ir.numType, 'integer');
});

test('e2e: sizeof on a static rank-2 array folds to shape literal vector', () => {
  const proc = processSource(`
M = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
S = sizeof(M)
`);
  const out = buildDerivations(proc.bindings);
  const s = out.bindings.get('S');
  assert.ok(s && s.ir);
  assert.equal(s.ir.kind, 'call');
  assert.equal(s.ir.op, 'vector');
  assert.deepEqual(s.ir.args.map((a: any) => a.value), [2, 3]);
});

test('e2e: nested fold — indicesof0 of a [3]-vector inside an add stays correct', () => {
  // C is a static [3]-vector; indicesof0(C) folds to vector(0,1,2);
  // add(C, indicesof0(C)) becomes add(C, vector(0,1,2)) which then
  // has the same arithmetic semantics as before, just literal-driven.
  const proc = processSource(`
C = [2.3, 1.5, 0.7]
Y = C .+ indicesof0(C)
`);
  const out = buildDerivations(proc.bindings);
  const y = out.bindings.get('Y');
  assert.ok(y && y.ir);
  // After fold + dissolve, Y is add(C, vector(0,1,2)) — the
  // broadcast(add, ...) dissolved to a direct add call, and the
  // indicesof0 call folded.
  function findVec(ir: any): any {
    if (!ir || typeof ir !== 'object') return null;
    if (ir.kind === 'call' && ir.op === 'vector') return ir;
    for (const a of (ir.args || [])) {
      const f = findVec(a);
      if (f) return f;
    }
    return null;
  }
  const vec = findVec(y.ir);
  assert.ok(vec, 'folded literal vector should appear in Y\'s IR');
  assert.deepEqual(vec.args.map((a: any) => a.value), [0, 1, 2]);
});

// ---------------------------------------------------------------------
// 5. Idempotence: a second dissolve pass doesn't change the folded IR
// ---------------------------------------------------------------------

test('idempotent: dissolveBindings twice produces the same folded IR', () => {
  const proc = processSource(`
C = [2.3, 1.5, 0.7]
Y = indicesof0(C)
`);
  const out1 = buildDerivations(proc.bindings);
  const y1 = out1.bindings.get('Y');
  // Run dissolve a second time on the same map.
  dissolveBindings(out1.bindings);
  const y2 = out1.bindings.get('Y');
  assert.deepEqual(y1.ir, y2.ir);
});
