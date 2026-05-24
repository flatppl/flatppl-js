'use strict';

// Tests for pir-sexpr.ts — FlatPIR S-expression printer + reader.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const pirSexpr = require('../pir-sexpr.ts');

function buildModule(src: any) {
  return processSource(src).loweredModule;
}

// ---------------------------------------------------------------------
// Printer
// ---------------------------------------------------------------------

test('toSexpr: trivial scalar binding', () => {
  const mod = buildModule(`x = 3.14`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%module/);
  assert.match(out, /\(%bind x 3\.14\)/);
});

test('toSexpr: %public list includes user names', () => {
  const mod = buildModule(`x = 1\ny = 2`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%public x y\)/);
});

test('toSexpr: identifier ref → (%ref self <name>)', () => {
  const mod = buildModule(`x = 1\ny = x`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%bind y \(%ref self x\)\)/);
});

test('toSexpr: binary op call', () => {
  const mod = buildModule(`x = 1\ny = 2\nz = x + y`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%bind z \(add \(%ref self x\) \(%ref self y\)\)\)/);
});

test('toSexpr: kwarg call form', () => {
  const mod = buildModule(`m = Normal(mu = 0, sigma = 1)`);
  const out = pirSexpr.toSexpr(mod);
  // both kwargs in some order
  assert.match(out, /\(Normal /);
  assert.match(out, /\(%kwarg mu 0\)/);
  assert.match(out, /\(%kwarg sigma 1\)/);
});

test('toSexpr: array literal → vector', () => {
  const mod = buildModule(`v = [1.0, 2.0, 3.0]`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(vector 1 2 3\)/);
});

test('toSexpr: record literal → record with %field entries', () => {
  const mod = buildModule(`r = record(mu = 0.0, sigma = 1.0)`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(record /);
  // Fields are positional-ordered structural entries
  assert.match(out, /\(%field mu 0\)/);
  assert.match(out, /\(%field sigma 1\)/);
});

// ---------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------

test('fromSexpr: simple module round-trip preserves bindings', () => {
  const src = `
(%module
  (%public x y)
  (%bind x 3.14)
  (%bind y (add (%ref self x) 1.0)))
`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.ok(module, 'should produce a module');
  assert.deepEqual(Array.from(module.publicSet), ['x', 'y']);
  assert.equal(module.bindings.size, 2);
  const yBind = module.bindings.get('y');
  assert.equal(yBind.rhs.kind, 'call');
  assert.equal(yBind.rhs.op, 'add');
  assert.equal(yBind.rhs.args[0].kind, 'ref');
  assert.equal(yBind.rhs.args[0].ns, 'self');
  assert.equal(yBind.rhs.args[0].name, 'x');
});

test('fromSexpr: comments and whitespace are tolerated', () => {
  const src = `
; A comment up top
(%module
  (%bind x 1)   ; inline
  ; comment between
  (%bind y (add (%ref self x) 1)))
`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal(module.bindings.size, 2);
});

test('fromSexpr: kwargs reconstruct call kwargs', () => {
  const src = `
(%module
  (%bind m (Normal (%kwarg mu 0) (%kwarg sigma 1))))
`;
  const { module } = pirSexpr.fromSexpr(src);
  const mBind = module.bindings.get('m');
  assert.equal(mBind.rhs.kind, 'call');
  assert.equal(mBind.rhs.op, 'Normal');
  assert.equal(mBind.rhs.kwargs.mu.value, 0);
  assert.equal(mBind.rhs.kwargs.sigma.value, 1);
});

test('fromSexpr: %field entries reconstruct record fields', () => {
  const src = `
(%module
  (%bind r (record (%field mu 0) (%field sigma 1))))
`;
  const { module } = pirSexpr.fromSexpr(src);
  const rBind = module.bindings.get('r');
  assert.equal(rBind.rhs.op, 'record');
  assert.equal(rBind.rhs.fields.length, 2);
  assert.equal(rBind.rhs.fields[0].name, 'mu');
  assert.equal(rBind.rhs.fields[1].name, 'sigma');
});

// ---------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------

test('round-trip: simple program → S-expr → module → S-expr is stable', () => {
  const src = `
x = 1
y = 2
z = x + y
m = Normal(mu = x, sigma = 1.0)
`;
  const mod = buildModule(src);
  const sexp1 = pirSexpr.toSexpr(mod);
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(sexp1);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const sexp2 = pirSexpr.toSexpr(mod2);
  // We don't insist on byte-identical output (kwarg ordering, blank
  // lines may shift), but the binding names should match and the
  // re-emitted text should be parseable and produce the same bindings.
  assert.equal(mod2.bindings.size, mod.bindings.size);
  for (const name of mod.bindings.keys()) {
    assert.ok(mod2.bindings.has(name), `binding ${name} missing after round-trip`);
  }
  // sanity: second pass produces a valid module too
  const { diagnostics: d2 } = pirSexpr.fromSexpr(sexp2);
  assert.deepEqual(d2.filter((d: any) => d.severity === 'error'), []);
});
