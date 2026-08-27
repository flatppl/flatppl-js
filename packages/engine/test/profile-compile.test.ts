'use strict';

// Tests for sampler-profile-compile.ts — the single-point compiler
// `worker.profileN` mode 'function' runs a profile body through.
//
// The contract is equality with the interpreter, so most of this file is
// "compile it, interpret it, compare". Coverage:
//   - every node kind the compiler specialises, plus the opaque fallback
//   - error text, which must stay byte-identical (the viewer's NaN gaps
//     come from these throws)
//   - structural identity: `loc` ignored, `-0` distinct from `0`, kwargs
//     and ops distinguished, equal structures in distinct objects shared
//   - purity and free-axis, the two gates on memoisation
//   - the memo's rewrap rule, which is what keeps a cached object from
//     being shared between two occurrences
//   - the per-point generation and the env-object guard
//   - the aggregate re-entry seam, end to end from source

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const PC = require('../sampler-profile-compile.ts');
const valueLib = require('../value.ts');
const { createWorkerHandler } = require('../worker.ts');
const FlatPPLEngine = require('../index.ts');

function synthLoc() {
  return { start: { line: -1, col: -1 }, end: { line: -1, col: -1 }, synthetic: true };
}
function lit(v: any) { return { kind: 'lit', value: v, loc: synthLoc() }; }
function ref(name: string) { return { kind: 'ref', ns: '%local', name, loc: synthLoc() }; }
function call(op: string, args: any[], kwargs?: any) {
  const n: any = { kind: 'call', op, args, loc: synthLoc() };
  if (kwargs) n.kwargs = kwargs;
  return n;
}

// Compile `ir`, then compare the compiled and interpreted results at each
// env in `envs`. Values are compared bit-exactly with NaN == NaN; a throw
// must happen on both sides with the same message.
function assertSameAsInterpreter(ir: any, envs: any[], label: string) {
  const prog = sampler.compileProfileBody(ir);
  for (let i = 0; i < envs.length; i++) {
    const envI = Object.assign({}, envs[i]);
    const envC = Object.assign({}, envs[i]);
    if (prog.bodyEval !== undefined) envC.__bodyEval = prog.bodyEval;
    let vi: any, vc: any, ei: any = null, ec: any = null;
    try { vi = sampler.evaluateExpr(ir, envI); } catch (e) { ei = e; }
    prog.nextPoint();
    try { vc = prog.evalPoint(envC); } catch (e) { ec = e; }
    const where = label + ' env#' + i;
    assert.equal(ec === null, ei === null,
      where + ': throw disagreement — interpreted ' +
      (ei ? '"' + ei.message + '"' : 'ok') + ', compiled ' +
      (ec ? '"' + ec.message + '"' : 'ok'));
    if (ei !== null) {
      assert.equal(ec.message, ei.message, where + ': error text differs');
      continue;
    }
    if (typeof vi === 'number' && Number.isNaN(vi)) {
      assert.ok(typeof vc === 'number' && Number.isNaN(vc), where + ': expected NaN');
      continue;
    }
    assert.deepEqual(vc, vi, where + ': value differs');
  }
  return prog;
}

// =====================================================================
// Node kinds
// =====================================================================

test('profile-compile: every specialised node kind matches the interpreter', () => {
  const envs = [{ x: 0.5, y: 2.0 }, { x: -3.25, y: 0.0 }, { x: 1e-9, y: 1e9 }];
  const cases: [string, any][] = [
    ['lit', lit(2.5)],
    ['const pi', { kind: 'const', name: 'pi', loc: synthLoc() }],
    ['const im', { kind: 'const', name: 'im', loc: synthLoc() }],
    ['ref', ref('x')],
    ['arity 1', call('exp', [ref('x')])],
    ['arity 2', call('add', [ref('x'), ref('y')])],
    ['arity 3', call('ifelse', [call('lt', [ref('x'), lit(0)]), ref('y'), ref('x')])],
    ['nested', call('mul', [call('sqrt', [call('abs', [ref('x')])]), call('log1p', [ref('y')])])],
    ['opaque tuple_get', call('tuple_get', [call('tuple', [ref('x'), ref('y')]), lit(1)])],
    ['opaque get_field', call('get_field',
      [{ kind: 'call', op: 'record', fields: [{ name: 'a', value: ref('x') }], loc: synthLoc() },
        lit('a')])],
    ['opaque fixed', call('fixed', [ref('x')])],
    ['kwargs ignored on the arith arm', call('add', [ref('x'), ref('y')], { extra: lit(1) })],
  ];
  for (const [label, ir] of cases) assertSameAsInterpreter(ir, envs, label);
});

test('profile-compile: complex arithmetic matches the interpreter', () => {
  const im = { kind: 'const', name: 'im', loc: synthLoc() };
  const z = call('add', [ref('x'), call('mul', [ref('y'), im])]);
  const ir = call('add', [call('abs2', [z]), call('real', [call('exp', [z])])]);
  assertSameAsInterpreter(ir, [{ x: 0.5, y: 1.5 }, { x: -2, y: 0.25 }], 'complex');
});

test('profile-compile: an unbound ref throws the interpreter\'s message', () => {
  assertSameAsInterpreter(call('add', [ref('x'), ref('missing')]),
    [{ x: 1 }], 'unbound');
  let msg = '';
  try {
    const prog = sampler.compileProfileBody(call('add', [ref('a'), ref('b')]));
    prog.evalPoint({ a: 1 });
  } catch (e: any) { msg = e.message; }
  assert.match(msg, /unbound %local reference 'b'/);
});

test('profile-compile: a free axis node throws the interpreter\'s message', () => {
  const ir = call('add', [{ kind: 'axis', name: 'i', loc: synthLoc() }, lit(1)]);
  assertSameAsInterpreter(ir, [{}], 'free axis');
  let msg = '';
  try { sampler.compileProfileBody(ir).evalPoint({}); } catch (e: any) { msg = e.message; }
  assert.match(msg, /axis '\.i' is not in scope/);
});

test('profile-compile: an unsupported node kind falls back to the interpreter', () => {
  const ir = { kind: 'nonesuch', loc: synthLoc() };
  let msg = '';
  try { sampler.compileProfileBody(ir).evalPoint({}); } catch (e: any) { msg = e.message; }
  assert.match(msg, /unsupported IR node kind 'nonesuch'/);
});

// =====================================================================
// Structural identity
// =====================================================================

test('profile-compile: structurally equal nodes in distinct objects share an id', () => {
  const An = PC.newAnalysis();
  const a = call('mul', [ref('x'), ref('x')]);
  const b = call('mul', [ref('x'), ref('x')]);
  assert.equal(PC._idOf(a, An), PC._idOf(b, An));
});

test('profile-compile: `loc` does not affect the id', () => {
  const An = PC.newAnalysis();
  const a = { kind: 'lit', value: 3, loc: { start: { line: 1, col: 1 } } };
  const b = { kind: 'lit', value: 3, loc: { start: { line: 99, col: 4 } } };
  assert.equal(PC._idOf(a, An), PC._idOf(b, An));
});

test('profile-compile: the id separates ops, kwargs, arg order and -0', () => {
  const An = PC.newAnalysis();
  const base = PC._idOf(call('add', [ref('x'), ref('y')]), An);
  assert.notEqual(PC._idOf(call('sub', [ref('x'), ref('y')]), An), base);
  assert.notEqual(PC._idOf(call('add', [ref('y'), ref('x')]), An), base);
  assert.notEqual(PC._idOf(call('add', [ref('x'), ref('y')], { k: lit(1) }), An), base);
  // -0 and 0 are not interchangeable in a literal: 1/-0 is -Infinity.
  assert.notEqual(PC._idOf(lit(-0), An), PC._idOf(lit(0), An));
  assert.notEqual(PC._idOf(lit(1), An), PC._idOf(lit('1'), An));
});

test('profile-compile: purity and free-axis gates', () => {
  const An = PC.newAnalysis();
  const pureId = PC._idOf(call('mul', [ref('x'), ref('x')]), An);
  assert.equal(An.pure[pureId], true);
  assert.equal(An.freeAxis[pureId], false);

  const randId = PC._idOf(call('add', [call('rand', []), lit(1)]), An);
  assert.equal(An.pure[randId], false, 'a subtree containing rand is impure');

  const targeted: any = { kind: 'call', target: { ns: 'self', name: 'f' },
    args: [ref('x')], loc: synthLoc() };
  const targetId = PC._idOf(call('neg', [targeted]), An);
  assert.equal(An.pure[targetId], false, 'a targeted call is opaque, so impure');

  const axisId = PC._idOf(call('neg', [{ kind: 'axis', name: 'i', loc: synthLoc() }]), An);
  assert.equal(An.freeAxis[axisId], true, 'a bare axis node is free');

  const aggId = PC._idOf(
    call('aggregate', [ref('sum'), lit(0), { kind: 'axis', name: 'i', loc: synthLoc() }]), An);
  assert.equal(An.freeAxis[aggId], false, 'aggregate binds the axes in its body');
});

// =====================================================================
// The memo
// =====================================================================

test('profile-compile: only a number, a boolean, a scalar complex or a Value is cached', () => {
  assert.equal(PC._cacheKind(1.5), PC._CACHE_PRIM);
  assert.equal(PC._cacheKind(true), PC._CACHE_PRIM);
  assert.equal(PC._cacheKind({ re: 1, im: 2 }), PC._CACHE_COMPLEX);
  assert.equal(PC._cacheKind(valueLib.vector([1, 2, 3])), PC._CACHE_VALUE);
  assert.equal(PC._cacheKind({ re: 1, im: 2, extra: 3 }), PC._CACHE_NONE);
  assert.equal(PC._cacheKind({ a: 1 }), PC._CACHE_NONE, 'a record is not cached');
  assert.equal(PC._cacheKind([1, 2]), PC._CACHE_NONE, 'a tuple is not cached');
  assert.equal(PC._cacheKind(null), PC._CACHE_NONE);
  assert.equal(PC._cacheKind('s'), PC._CACHE_NONE);
});

test('profile-compile: a cached object is handed out as a fresh wrapper', () => {
  const z = { re: 1, im: -2 };
  const zw = PC._rewrap(PC._CACHE_COMPLEX, z);
  assert.notEqual(zw, z, 'a cached complex must not be shared by reference');
  assert.deepEqual(zw, z);

  const v = valueLib.vector([1, 2, 3]);
  const vw = PC._rewrap(PC._CACHE_VALUE, v);
  assert.notEqual(vw, v, 'a cached Value must not be shared by reference');
  assert.notEqual(vw.shape, v.shape, 'the shape array is copied');
  assert.deepEqual(vw.shape, v.shape);
  // The data buffer IS shared — the engine's own convention for a
  // non-copying result, and nothing writes into a buffer it received.
  assert.equal(vw.data, v.data);

  assert.equal(PC._rewrap(PC._CACHE_PRIM, 4), 4);
});

test('profile-compile: a repeated pure subtree gets one slot', () => {
  const inner = () => call('sqrt', [call('add', [ref('x'), lit(1)])]);
  const prog = sampler.compileProfileBody(call('mul', [inner(), inner()]));
  // Two slots, not one: `add(x, 1)` repeats as well as the `sqrt` around
  // it. The inner slot never pays off once the outer one hits, and it
  // costs one array write on the miss.
  assert.equal(prog.stats().slots, 2);
  assert.equal(prog.stats().compiled, true);
});

test('profile-compile: a subtree that occurs once gets no slot', () => {
  const prog = sampler.compileProfileBody(
    call('mul', [call('sqrt', [ref('x')]), call('exp', [ref('y')])]));
  assert.equal(prog.stats().slots, 0);
});

test('profile-compile: nextPoint invalidates the memo', () => {
  const inner = () => call('mul', [ref('x'), ref('x')]);
  const ir = call('add', [inner(), inner()]);
  const prog = sampler.compileProfileBody(ir);
  const env: any = { x: 3 };
  prog.nextPoint();
  assert.equal(prog.evalPoint(env), 18);
  env.x = 5;
  prog.nextPoint();
  assert.equal(prog.evalPoint(env), 50, 'a new point must re-read the env');
});

test('profile-compile: a different env object recomputes within one point', () => {
  // The env-object guard is what makes slot sharing safe across trees
  // without a scope analysis. Two envs, one generation, two answers.
  const inner = () => call('mul', [ref('x'), ref('x')]);
  const prog = sampler.compileProfileBody(call('add', [inner(), inner()]));
  prog.nextPoint();
  assert.equal(prog.evalPoint({ x: 3 }), 18);
  assert.equal(prog.evalPoint({ x: 5 }), 50);
});

test('profile-compile: an impure repeated subtree is not memoised', () => {
  // Two `rand()` draws must stay two draws. `rand` advances RNG state, so
  // collapsing them would change the value AND the stream.
  const An = PC.newAnalysis();
  const ir = call('add', [call('rand', []), call('rand', [])]);
  PC._analyse(ir, An);
  const id = An.byNode.get(ir.args[0]);
  assert.equal(An.pure[id], false);
  const prog = sampler.compileProfileBody(ir);
  assert.equal(prog.stats().slots, 0, 'no slot for an impure subtree');
});

// =====================================================================
// End to end
// =====================================================================

test('profile-compile: worker profileN function mode matches the interpreter', () => {
  // f(x) = sqrt(x+1) * sqrt(x+1) + sqrt(x+1) — a body whose repeated
  // subtree the compiler memoises.
  const inner = () => call('sqrt', [call('add', [ref('x'), lit(1)])]);
  const ir = call('add', [call('mul', [inner(), inner()]), inner()]);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'profileN', ir, sweepName: 'x', range: [0, 8],
    count: 5, mode: 'function' });
  assert.equal(r.type, 'samples');
  for (let i = 0; i < 5; i++) {
    const x = i * 2;
    const want = sampler.evaluateExpr(ir, { x });
    assert.equal(r.samples[i], want, 'point ' + i);
  }
});

test('profile-compile: profileN turns a per-point throw into a NaN gap', () => {
  // `checked(value, condition)` throws when the condition is false, so
  // this body throws exactly at the negative points.
  const ir = call('checked', [ref('x'), call('ge', [ref('x'), lit(0)])]);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const r = w.handle({ type: 'profileN', ir, sweepName: 'x', range: [-1, 1],
    count: 3, mode: 'function' });
  assert.equal(r.type, 'samples');
  assert.ok(Number.isNaN(r.samples[0]), 'x = -1 throws and becomes NaN');
  assert.equal(r.samples[1], 0);
  assert.equal(r.samples[2], 1);
});

test('profile-compile: an aggregate body re-enters through the compiler', () => {
  // `metricsum` lowers to a sum-aggregate, so this body evaluates through
  // sampler-aggregate's re-entry. `y[1]` appears twice, which is what the
  // env.__bodyEval seam exists to let the compiler share.
  const source = `flatppl_compat = "0.1"
eye2 = rowstack([[1.0, 0.0], [0.0, 1.0]])
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
t = elementof(interval(1.0, 4.0))
u = [t, t]
eye2: y[.i^] := A[.i^, .j_] * u[.j^]
s = y[1] + y[2] + y[1]
f = functionof(s, t = t)
`;
  const res = FlatPPLEngine.processSource(source, { path: '/synthetic/aggseam.flatppl' });
  const errs = (res.diagnostics || []).filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), [], 'fixture must analyse clean');
  const built = FlatPPLEngine.orchestrator.buildDerivations(res.bindings);
  const sig = FlatPPLEngine.orchestrator.signatureOf('f', built.bindings);
  assert.equal(sig.kind, 'function');
  const node = FlatPPLEngine.clm.lowerMeasure(sig.body,
    { derivations: built.derivations, bindings: built.bindings, fixedValues: built.fixedValues },
    { boundaries: {}, freeInputs: ['t'] });
  assert.ok(node && node.body, 'expected a lowered body');

  const sessionEnv: any = {};
  for (const n of FlatPPLEngine.orchestrator.collectSelfRefs(node.body)) {
    if (built.fixedValues.has(n)) sessionEnv[n] = built.fixedValues.get(n);
  }
  // A^i_j u^j for u = [t, t] is [3t, 7t]; y[1] + y[2] + y[1] = 13t.
  assertSameAsInterpreter(node.body,
    [Object.assign({}, sessionEnv, { t: 1 }),
      Object.assign({}, sessionEnv, { t: 2.5 })], 'aggregate seam');

  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  if (Object.keys(sessionEnv).length > 0) {
    w.handle({ type: 'setEnv', env: sessionEnv, merge: true });
  }
  const r = w.handle({ type: 'profileN', ir: node.body, sweepName: 't',
    range: [1, 4], count: 4, mode: 'function', fixedEnv: {} });
  assert.equal(r.type, 'samples');
  assert.deepEqual(Array.from(r.samples), [13, 26, 39, 52]);
});

test('profile-compile: a Value-valued repeated subtree stays correct', () => {
  // The memo's Value branch: `add(v, v)` returns a Value and appears
  // twice, so the second occurrence reads the cache and must still see
  // its own wrapper.
  const inner = () => call('add', [ref('v'), ref('v')]);
  const ir = call('add', [inner(), inner()]);
  const prog = sampler.compileProfileBody(ir);
  assert.equal(prog.stats().slots, 1);
  const env = { v: valueLib.vector([1, 2, 3]) };
  prog.nextPoint();
  const got = prog.evalPoint(env);
  assert.deepEqual(Array.from(got.data), [4, 8, 12]);
  assert.deepEqual(sampler.evaluateExpr(ir, env), got);
});

// =====================================================================
// Edge arms
// =====================================================================

test('profile-compile: an inherited field is not part of the signature', () => {
  // The signature walk asks for OWN fields only. A node carrying an
  // enumerable inherited key must sign the same as one without it.
  const An = PC.newAnalysis();
  const plain = { kind: 'lit', value: 7 };
  const inherited: any = Object.create({ sneaky: 'ignored' });
  inherited.kind = 'lit';
  inherited.value = 7;
  assert.equal((inherited as any).sneaky, 'ignored', 'the inherited key is visible');
  assert.equal(PC._idOf(inherited, An), PC._idOf(plain, An));
});

test('profile-compile: the tally counts call occurrences, shared objects included', () => {
  const An = PC.newAnalysis();
  const sharedRef = ref('x');
  const sharedCall = call('mul', [sharedRef, sharedRef]);
  const ir = call('add', [sharedCall, sharedCall]);
  PC._analyse(ir, An);
  assert.equal(An.callCounts[An.byNode.get(sharedRef)], 0,
    'a ref is never tallied as a call');
  assert.equal(An.callCounts[An.byNode.get(sharedCall)], 2,
    'the same call object reached twice counts twice');
  assert.equal(An.callCounts[An.byNode.get(ir)], 1);
});

test('profile-compile: a hash collision is resolved by comparing the tuples', () => {
  // Collisions are rare enough that the corpus never produces one, so
  // seed the bucket the next intern will land in with two decoys: one of
  // a different length, one of the same length with a different element.
  const An = PC.newAnalysis();
  const parts = [-2, 0, 5];
  const h = PC._hashParts(parts);
  const decoyA = PC._intern(An, [-2, 0], true, false);
  const decoyB = PC._intern(An, [-2, 0, 6], true, false);
  An.buckets.set(h, [decoyA, decoyB]);
  assert.equal(PC._samePartsAs(An, decoyA, parts), false, 'length differs');
  assert.equal(PC._samePartsAs(An, decoyB, parts), false, 'element differs');
  const fresh = PC._intern(An, parts, true, false);
  assert.notEqual(fresh, decoyA);
  assert.notEqual(fresh, decoyB);
  // And an exact re-intern now finds it in the same bucket.
  assert.equal(PC._intern(An, [-2, 0, 5], true, false), fresh);
});

test('profile-compile: an axis in scope reads through __axisEnv', () => {
  // Nothing in the engine populates `__axisEnv` today — the aggregate
  // lowering consumes axis labels structurally — but the compiled arm
  // must still match the interpreter's if anything ever does.
  const ir = call('add', [{ kind: 'axis', name: 'i', loc: synthLoc() }, lit(1)]);
  const env = { __axisEnv: { i: 4 } };
  assert.equal(sampler.evaluateExpr(ir, env), 5);
  const prog = sampler.compileProfileBody(ir);
  prog.nextPoint();
  assert.equal(prog.evalPoint(env), 5);
});

test('profile-compile: a targeted call is opaque', () => {
  // A `self`-target call resolves its body from the env at run time, so
  // the compiler hands the whole node to the interpreter.
  const ir: any = { kind: 'call', target: { ns: 'self', name: 'nosuchfn' },
    args: [ref('x')], loc: synthLoc() };
  assertSameAsInterpreter(ir, [{ x: 1 }], 'targeted call');
  const An = PC.newAnalysis();
  PC._analyse(ir, An);
  assert.equal(An.pure[An.byNode.get(ir)], false);
});

test('profile-compile: an aggregate node compiles to the interpreter', () => {
  // The `aggregate` name is in ARITH_OPS but dispatches as a higher-order
  // op, so the compiler must not treat it as an arithmetic call.
  assert.equal('aggregate' in sampler._internal.ARITH_OPS, false,
    'aggregate must stay out of ARITH_OPS — the opaque arm relies on it');
  const ir = call('neg', [call('aggregate',
    [ref('sum'), lit(0), { kind: 'axis', name: 'i', loc: synthLoc() }])]);
  assertSameAsInterpreter(ir, [{ sum: 1 }], 'aggregate opaque');
});

test('profile-compile: a call with no args is handled like the interpreter', () => {
  // `rand` has no ARITH_OPS entry, so this takes the opaque arm and the
  // interpreter owns both the RNG and the error.
  const ir: any = { kind: 'call', op: 'rand', loc: synthLoc() };
  assertSameAsInterpreter(ir, [{}], 'arity 0');
});

test('profile-compile: an undeclared op of arity 3 reaches the variadic arm', () => {
  // `linspace` is in ARITH_OPS but has no ops.ts declaration, so it takes
  // the spread call rather than ops.dispatch.
  assert.equal(require('../ops.ts').isDeclared('linspace'), false);
  const ir = call('linspace', [lit(0), ref('x'), lit(5)]);
  assertSameAsInterpreter(ir, [{ x: 4 }, { x: 1 }], 'undeclared variadic');
});

test('profile-compile: a memoised subtree returning an uncacheable value is not cached', () => {
  // `get_field` of a missing key yields undefined — a repeated, pure,
  // axis-free subtree that gets a slot but nothing the memo will store.
  const gf = () => call('get_field',
    [{ kind: 'call', op: 'record', fields: [{ name: 'a', value: ref('x') }], loc: synthLoc() },
      lit('missing')]);
  const ir = call('add', [gf(), gf()]);
  const prog = sampler.compileProfileBody(ir);
  assert.equal(prog.stats().slots, 1, 'the repeated get_field still takes a slot');
  assertSameAsInterpreter(ir, [{ x: 1 }], 'uncacheable memo');
});

test('profile-compile: bodyEval passes a non-node straight to the interpreter', () => {
  const prog = sampler.compileProfileBody(call('add', [ref('x'), ref('x')]));
  assert.notEqual(prog.bodyEval, undefined);
  assert.throws(() => prog.bodyEval(5, {}), /unsupported IR node kind/);
});

test('profile-compile: a non-node argument falls back to the interpreter', () => {
  const ir: any = { kind: 'call', op: 'add', args: [ref('x'), 7], loc: synthLoc() };
  assertSameAsInterpreter(ir, [{ x: 1 }], 'raw number argument');
});

test('profile-compile: an ARITH_OPS call with no args matches the interpreter', () => {
  // Malformed IR, but `evaluateCall`'s arithmetic arm tolerates a missing
  // `args` and so must the compiler — it reaches the variadic arm with
  // zero arguments.
  const ir: any = { kind: 'call', op: 'add', loc: synthLoc() };
  assertSameAsInterpreter(ir, [{}], 'declared, no args');
  const undeclared: any = { kind: 'call', op: 'linspace', loc: synthLoc() };
  assertSameAsInterpreter(undeclared, [{}], 'undeclared, no args');
});
