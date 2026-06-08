'use strict';

// Tests for pir-sexpr.ts user-defined-call + %params emission (spec §11).
//
// Two FlatPIR-grammar invariants are pinned here:
//   1. Calls to user-defined callables use `(%call (%ref <ns> <name>) args…)`,
//      NOT a bare `(undefined …)` head (the old behaviour, where the printer
//      emitted `e.op` which is undefined for user calls).
//   2. `functionof`/`kernelof` parameter lists are the NESTED form
//      `(%params (x y z))`, not a flat `(%params x y z)`.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const pirSexpr = require('../pir-sexpr.ts');

function buildModule(src: any) {
  return processSource(src).loweredModule;
}

test('toSexpr: user-defined call emits (%call (%ref self <name>) …)', () => {
  const mod = buildModule(`f = functionof(_x_ + 1.0, x = _x_)\ng = f(2.0)`);
  const out = pirSexpr.toSexpr(mod);
  // The call to user-defined `f` must use the %call/%ref form.
  assert.match(out, /\(%bind g \(%call \(%ref self f\) 2\)\)/);
  // Regression guard: the head must not be the literal `undefined`.
  assert.doesNotMatch(out, /\(undefined/);
});

test('toSexpr: builtin call keeps its bare-symbol head', () => {
  // `add` is a built-in, so the head stays bare (no %call wrapper).
  const mod = buildModule(`x = 1\ny = 2\nz = x + y`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%bind z \(add \(%ref self x\) \(%ref self y\)\)\)/);
});

test('toSexpr: functionof params use nested (%params (…)) form', () => {
  const mod = buildModule(`f = functionof(_x_ + 1.0, x = _x_)`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%params \(_x_\)\)/);
  // Regression guard: must not be the flat list `(%params _x_)`.
  assert.doesNotMatch(out, /\(%params _x_\)/);
});

test('fromSexpr: %call round-trips to a user-call target shape', () => {
  const src = `(%module
    (%public f g)
    (%bind f (functionof (%params (_x_)) (add (%ref %local _x_) 1)))
    (%bind g (%call (%ref self f) 2)))`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics, []);
  const g = module.bindings.get('g');
  assert.ok(g, 'binding g should exist');
  assert.equal(g.rhs.kind, 'call');
  assert.deepEqual(g.rhs.target, { ns: 'self', name: 'f' });
  // No spurious `op` on a user call.
  assert.equal(g.rhs.op, undefined);
});

test('fromSexpr: nested %params reconstructs the parameter list', () => {
  const src = `(%module
    (%public f)
    (%bind f (functionof (%params (a b _x_)) (add (%ref %local a) (%ref %local _x_)))))`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics, []);
  const f = module.bindings.get('f');
  assert.deepEqual(f.rhs.params, ['a', 'b', '_x_']);
});

test('round-trip: user call + functionof params survive toSexpr ∘ fromSexpr', () => {
  const mod = buildModule(`f = functionof(_x_ + 1.0, x = _x_)\ng = f(2.0)`);
  const sexp1 = pirSexpr.toSexpr(mod);
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(sexp1);
  assert.deepEqual(diagnostics, []);
  const sexp2 = pirSexpr.toSexpr(mod2);
  assert.equal(sexp1, sexp2);
});

test('fromSexpr: %call absorbs a kwarg part', () => {
  const src = `(%module
    (%public f g)
    (%bind f (functionof (%params (_x_)) (add (%ref %local _x_) 1)))
    (%bind g (%call (%ref self f) (%kwarg x 2))))`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics, []);
  const g = module.bindings.get('g');
  assert.deepEqual(g.rhs.target, { ns: 'self', name: 'f' });
  assert.equal(g.rhs.kwargs.x.value, 2);
});

test('fromSexpr: legacy flat (%params a b) is tolerated', () => {
  // Pre-spec flat form (no inner list) still parses to the param list.
  const src = `(%module
    (%public f)
    (%bind f (functionof (%params a b _x_) (add (%ref %local a) (%ref %local _x_)))))`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(module.bindings.get('f').rhs.params, ['a', 'b', '_x_']);
});

test('fromSexpr: %call with a non-(%ref …) head reports a diagnostic', () => {
  const src = `(%module (%public g) (%bind g (%call (add 1 2) 3)))`;
  const { diagnostics } = pirSexpr.fromSexpr(src);
  assert.ok(diagnostics.some((d: any) => /%call head must be a \(%ref/.test(d.message)),
    `expected a %call-head diagnostic, got ${JSON.stringify(diagnostics)}`);
});

test('fromSexpr: %call with no (%ref …) head reports a diagnostic', () => {
  const src = `(%module (%public g) (%bind g (%call foo 3)))`;
  const { diagnostics } = pirSexpr.fromSexpr(src);
  assert.ok(diagnostics.some((d: any) => /%call expects a \(%ref/.test(d.message)),
    `expected a missing-head diagnostic, got ${JSON.stringify(diagnostics)}`);
});

test('fromSexpr: unclosed (%call …) reports a diagnostic', () => {
  const src = `(%module (%public g) (%bind g (%call (%ref self f) 2`;
  const { diagnostics } = pirSexpr.fromSexpr(src);
  assert.ok(diagnostics.some((d: any) => /unclosed \(%call/.test(d.message)),
    `expected an unclosed-%call diagnostic, got ${JSON.stringify(diagnostics)}`);
});

test('L7: a user-call target with no explicit namespace emits the default `self` ref', () => {
  // Build the module IR directly: a `call` node whose target omits `ns`.
  // (fromSexpr always sets ns, so this default arm is only reachable by
  // constructing the IR — e.g. a lowering that did not record a namespace.)
  const mod = {
    publicSet: new Set(['g']),
    bindings: new Map([
      ['g', { rhs: { kind: 'call', target: { name: 'f' }, args: [{ kind: 'lit', value: 2 }] } }],
    ]),
  };
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%call \(%ref self f\) 2\)/,
    `a target without ns must default to self, got: ${out}`);
});
