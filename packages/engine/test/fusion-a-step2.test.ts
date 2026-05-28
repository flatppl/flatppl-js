'use strict';

// =====================================================================
// fusion-a-step2.test.ts — broadcast → aggregate rewrite (fusion (a))
// =====================================================================
//
// Pins the MVP scope of `_tryDissolveBroadcastReduction` in
// dissolver.ts (engine-concepts §20.10.8 / TODO-flatppl-js fusion
// thread (a)). The rewrite recognises:
//
//   broadcast(<head>, args…)
//     where head = functionof((p1,…) -> R(<plain_elementwise>), p1,…)
//     R ∈ {sum, mean, prod}
//     <plain_elementwise> uses no nested higher-order ops
//
// and emits `aggregate(R, [.atom], <substituted_body>)` with rank-1
// leaves wrapped in `get(<leaf>, .j)`.
//
// Coverage gates:
//   - All broadcast args are fixed-phase (MVP gate; stochastic args
//     stay on the runtime path until fusion (b) lands).
//   - Inner expression has no nested broadcast / aggregate /
//     functionof / kernel_broadcast / measure-algebra ops.
//   - Reducer is sum / mean / prod (matches the §20.10.5 calibrated
//     set; var/std/min/max stay on the runtime path).
//
// What's pinned:
//   1. The rewrite fires structurally for the canonical pattern and
//      produces an aggregate IR with the expected output_axes /
//      reducer / leaf-wrapping.
//   2. Refusals: stochastic broadcast arg, nested broadcast in body,
//      non-matching reducer, kwarg form, head not a functionof.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { _tryDissolveBroadcastReduction } = require('../dissolver.ts');

function mkBindings(entries: Array<[string, any]>) {
  const m = new Map();
  for (const [name, b] of entries) m.set(name, b);
  return m;
}

function arrayType(rank: number, shape: any[], prim = 'real') {
  return {
    kind: 'array', rank, shape,
    elem: { kind: 'scalar', prim },
  };
}

function ref(name: string) { return { kind: 'ref', ns: 'self', name }; }
function loc(name: string) { return { kind: 'ref', ns: '%local', name }; }
function axis(name: string) { return { kind: 'axis', name }; }
function lit(v: number, t = 'real') { return { kind: 'lit', value: v, numType: t }; }
function vector(...args: any[]) { return { kind: 'call', op: 'vector', args }; }
function call(op: string, ...args: any[]) { return { kind: 'call', op, args }; }

// ---------------------------------------------------------------------
// Helpers to find structural elements of the produced aggregate
// ---------------------------------------------------------------------

function findGet(ir: any, srcMatch: (s: any) => boolean): any | null {
  if (!ir || typeof ir !== 'object') return null;
  if (ir.kind === 'call' && ir.op === 'get' && Array.isArray(ir.args)) {
    if (srcMatch(ir.args[0])) return ir;
  }
  for (const a of (ir.args || [])) {
    const f = findGet(a, srcMatch);
    if (f) return f;
  }
  return null;
}

// =====================================================================
// 1. Canonical fire: weighted-sum-like pattern
// =====================================================================

test('fusion (a) Step 2: weighted-sum dot product — broadcast → aggregate', () => {
  // f = (v, s) -> sum(v .* s) — but the MVP gate refuses bodies with
  // nested broadcasts (`.*` lowers to broadcast(mul, …)). So instead
  // we feed an IR where the body is a plain (mul …) — what a future
  // pre-dissolve step would produce.
  //
  //   fn body = sum(mul(<%local v>, <%local s>))
  //   broadcast(<fn>, [V] Ref-wrap, S broadcast-over)
  const V = ref('V');
  const S = ref('S');
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
    ['S', { ir: lit(0), inferredType: arrayType(1, [4]), phase: 'fixed' }],
  ]);
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v', 's'],
    body: call('sum', call('mul', loc('v'), loc('s'))),
  };
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [fn, vector(V), S],   // [V] Ref-wrap; S broadcast-over rank-1
  };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.ok(r, 'fusion (a) Step 2 should fire');
  assert.equal(r.kind, 'call');
  assert.equal(r.op, 'aggregate');
  // args[0] = ref to reducer 'sum'.
  assert.equal(r.args[0].name, 'sum');
  // args[1] = vector(axis 'atom') — single output axis.
  assert.equal(r.args[1].op, 'vector');
  assert.equal(r.args[1].args.length, 1);
  assert.equal(r.args[1].args[0].kind, 'axis');
  const atomName = r.args[1].args[0].name;
  // args[2] = body with get(V, .j) and get(S, .atom) somewhere.
  const body = r.args[2];
  const getV = findGet(body, (s) => s && s.kind === 'ref' && s.name === 'V');
  assert.ok(getV, 'V should be wrapped in get(.j)');
  const getS = findGet(body, (s) => s && s.kind === 'ref' && s.name === 'S');
  assert.ok(getS, 'S should be wrapped in get(.atom)');
  assert.equal(getS.args[1].name, atomName);
  // The .j axis should be different from .atom.
  const jName = getV.args[1].name;
  assert.notEqual(jName, atomName, 'reduction axis .j differs from .atom');
});

// =====================================================================
// 2. Reducer accepted set
// =====================================================================

test('fusion (a) Step 2: mean reducer accepted', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
    ['S', { ir: lit(0), inferredType: arrayType(1, [4]), phase: 'fixed' }],
  ]);
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v', 's'],
    body: call('mean', call('mul', loc('v'), loc('s'))),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, vector(ref('V')), ref('S')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.ok(r, 'mean accepted');
  assert.equal(r.args[0].name, 'mean');
});

test('fusion (a) Step 2: prod reducer accepted', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
    ['S', { ir: lit(0), inferredType: arrayType(1, [4]), phase: 'fixed' }],
  ]);
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v', 's'],
    body: call('prod', call('mul', loc('v'), loc('s'))),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, vector(ref('V')), ref('S')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.ok(r, 'prod accepted');
  assert.equal(r.args[0].name, 'prod');
});

test('fusion (a) Step 2: unsupported reducer (var) refused', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
    ['S', { ir: lit(0), inferredType: arrayType(1, [4]), phase: 'fixed' }],
  ]);
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v', 's'],
    body: call('var', call('mul', loc('v'), loc('s'))),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, vector(ref('V')), ref('S')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.equal(r, null);
});

// =====================================================================
// 3. Phase gate — stochastic broadcast arg refuses fusion
// =====================================================================

test('fusion (a) Step 2: stochastic broadcast arg refused (MVP phase gate)', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
    ['X', { ir: lit(0), inferredType: arrayType(1, [4]), phase: 'stochastic' }],
  ]);
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v', 'x'],
    body: call('sum', call('mul', loc('v'), loc('x'))),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, vector(ref('V')), ref('X')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.equal(r, null, 'stochastic args stay on runtime path');
});

// =====================================================================
// 4. Forbidden-op gate — nested broadcast in body refuses
// =====================================================================

test('fusion (a) Step 2: nested broadcast in body refused', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
    ['S', { ir: lit(0), inferredType: arrayType(1, [4]), phase: 'fixed' }],
  ]);
  // Body has a nested broadcast(mul, ...) — the polyeval-like surface.
  // MVP refuses this; pre-dissolution would have to flatten the dotted
  // binary first.
  const innerBroadcast = call('broadcast',
    { kind: 'call', op: 'functionof', params: ['a', 'b'], body: call('mul', loc('a'), loc('b')) },
    loc('v'),
    loc('s'));
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v', 's'],
    body: call('sum', innerBroadcast),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, vector(ref('V')), ref('S')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.equal(r, null, 'nested broadcast in body refuses');
});

test('fusion (a) Step 2: nested aggregate in body refused', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
  ]);
  const innerAgg = {
    kind: 'call', op: 'aggregate',
    args: [ref('sum'), vector(axis('k')), call('mul', loc('v'), axis('k'))],
  };
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v'],
    body: call('sum', innerAgg),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, vector(ref('V'))] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.equal(r, null);
});

// =====================================================================
// 5. Structural refusals — non-functionof head, kwarg form, etc.
// =====================================================================

test('fusion (a) Step 2: non-functionof head refused', () => {
  const bindings = new Map();
  const bcIR = { kind: 'call', op: 'broadcast',
    args: [ref('not_a_fn'), ref('V')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (a) Step 2: kwarg form refused', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
  ]);
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v'],
    body: call('sum', loc('v')),
  };
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [fn], kwargs: { v: ref('V') },
  };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.equal(r, null);
});

test('fusion (a) Step 2: body not a reducer call refused', () => {
  const bindings = mkBindings([
    ['V', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
  ]);
  // body is just `loc('v')` not `sum(loc('v'))`.
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v'],
    body: loc('v'),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, ref('V')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.equal(r, null);
});

// =====================================================================
// 6. Single-param reducer (closure-form mkfn-like, body is plain)
// =====================================================================

test('fusion (a) Step 2: single-param fn with closed-over self-ref in body', () => {
  // fn = (v) -> sum(mul(self.C, v))  — C is closed-over, V broadcast-over
  // Body has no placeholders for C; just %local('v') and ref('C').
  const bindings = mkBindings([
    ['C', { ir: lit(0), inferredType: arrayType(1, [3]), phase: 'fixed' }],
    ['V', { ir: lit(0), inferredType: arrayType(1, [4]), phase: 'fixed' }],
  ]);
  const fn = {
    kind: 'call', op: 'functionof',
    params: ['v'],
    body: call('sum', call('mul', ref('C'), loc('v'))),
  };
  const bcIR = { kind: 'call', op: 'broadcast', args: [fn, ref('V')] };
  const r = _tryDissolveBroadcastReduction(bcIR, bindings);
  assert.ok(r, 'fires for closure-form');
  assert.equal(r.op, 'aggregate');
  // C should be wrapped in get(C, .j); V (broadcast-over) in get(V, .atom).
  const body = r.args[2];
  const getC = findGet(body, (s) => s && s.kind === 'ref' && s.name === 'C');
  const getV = findGet(body, (s) => s && s.kind === 'ref' && s.name === 'V');
  assert.ok(getC && getV);
});
