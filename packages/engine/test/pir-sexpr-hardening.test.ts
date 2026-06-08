'use strict';

// Hardening tests for pir-sexpr.ts (spec §11 FlatPIR reader).
//
// Three robustness fixes are pinned here:
//   L4    — `%call` head namespace validation: `%local` is a parameter,
//           not a callable, so it must be rejected as a `%call` head;
//           `self` and module-alias heads still parse & round-trip.
//   NIT-1 — kwargs / name-keyed maps must not surface a crafted
//           `__proto__` key (prototype-pollution / Object.keys leak).
//   NIT-2 — the recursive-descent reader caps nesting depth and emits a
//           diagnostic rather than overflowing the stack.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const pirSexpr = require('../pir-sexpr.ts');

// ---- L4: %call head namespace validation --------------------------------

test('fromSexpr: %call head pointing at %local reports a diagnostic', () => {
  // `%local x` is a parameter reference, not a callable head.
  const src = `(%module (%public g) (%bind g (%call (%ref %local x) 2)))`;
  const { diagnostics } = pirSexpr.fromSexpr(src);
  assert.ok(
    diagnostics.some((d: any) => /%local/.test(d.message) && /%call/.test(d.message)),
    `expected a %local-head diagnostic, got ${JSON.stringify(diagnostics)}`,
  );
});

test('fromSexpr: %call with a self head parses cleanly', () => {
  const src = `(%module
    (%public f g)
    (%bind f (functionof (%params (_x_)) (add (%ref %local _x_) 1)))
    (%bind g (%call (%ref self f) 2)))`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics, []);
  const g = module.bindings.get('g');
  assert.deepEqual(g.rhs.target, { ns: 'self', name: 'f' });
});

test('fromSexpr: %call with a module-alias head parses cleanly', () => {
  const src = `(%module (%public g) (%bind g (%call (%ref othermod f) 2)))`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics, []);
  const g = module.bindings.get('g');
  assert.deepEqual(g.rhs.target, { ns: 'othermod', name: 'f' });
});

test('fromSexpr: module-alias %call head round-trips', () => {
  const src = `(%module (%public g) (%bind g (%call (%ref othermod f) 2)))`;
  const { module } = pirSexpr.fromSexpr(src);
  const out = pirSexpr.toSexpr(module);
  assert.match(out, /\(%call \(%ref othermod f\) 2\)/);
});

// ---- NIT-1: prototype-pollution-safe kwargs map -------------------------

test('fromSexpr: %kwarg __proto__ does not pollute the kwargs map', () => {
  const src = `(%module
    (%public f g)
    (%bind f (functionof (%params (_x_)) (add (%ref %local _x_) 1)))
    (%bind g (%call (%ref self f) (%kwarg __proto__ 1))))`;
  const { module } = pirSexpr.fromSexpr(src);
  const g = module.bindings.get('g');
  const kwargs = g.rhs.kwargs;
  // The crafted key must be an OWN, enumerable key holding the value —
  // not a write through to Object.prototype.
  assert.equal(Object.getPrototypeOf({}).__proto__nonsense, undefined);
  // The kwargs map carries the value safely under the literal name.
  assert.equal(kwargs.__proto__.value, 1);
  // And it does not leak onto a fresh plain object's prototype chain.
  assert.equal(({} as any).__proto__poison, undefined);
});

// ---- NIT-2: recursion depth cap -----------------------------------------

test('fromSexpr: pathologically deep nesting yields a depth diagnostic', () => {
  const depth = 50000;
  const src = '(%module (%public g) (%bind g '
    + '(add '.repeat(depth) + '1' + ')'.repeat(depth) + '))';
  let result: any;
  assert.doesNotThrow(() => {
    result = pirSexpr.fromSexpr(src);
  }, 'deeply nested input must not throw a RangeError');
  assert.ok(
    result.diagnostics.some((d: any) => /nesting too deep/.test(d.message)),
    `expected a depth diagnostic, got ${JSON.stringify(result.diagnostics.slice(0, 3))}`,
  );
});

test('fromSexpr: moderate nesting below the cap parses without a depth diagnostic', () => {
  const depth = 100;
  const src = '(%module (%public g) (%bind g '
    + '(add '.repeat(depth) + '1' + ')'.repeat(depth) + '))';
  const { diagnostics } = pirSexpr.fromSexpr(src);
  assert.ok(
    !diagnostics.some((d: any) => /nesting too deep/.test(d.message)),
    `unexpected depth diagnostic at depth ${depth}: ${JSON.stringify(diagnostics)}`,
  );
});
