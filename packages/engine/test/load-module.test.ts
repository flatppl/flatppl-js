'use strict';

// Spec §04 (Module composition) + §11 (FlatPIR cross-module) — engine-level
// `load_module` behaviour: IR shape, bundle compilation, cross-module type
// inference, and (later stages) materialisation. The host I/O is mocked by
// passing pre-resolved sources via `opts.bundle.sources` (the engine stays
// synchronous; async fetch is the host's job).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, pirSexpr } = require('../index.ts');

// ---------------------------------------------------------------------
// IR shape: substitutions lower to an ARRAY of {name, value} (the spec
// §11 `(%assign ...)` ordered-entry shape), NOT a plain object. Both
// `ir-walk` (child descent) and `pir-sexpr` (writer + reader) already
// assume the array; lower.ts was the outlier emitting an object, which
// made substitution IRs invisible to every IR walker and crashed toSexpr.
// ---------------------------------------------------------------------

test('load_module substitutions lower to an assigns ARRAY', () => {
  const r = processSource('a = elementof(reals)\nm = load_module("h.flatppl", mu = a)');
  const b = r.loweredModule.bindings.get('m');
  assert.equal(b.rhs.op, 'load_module');
  assert.ok(Array.isArray(b.rhs.assigns),
    'assigns must be an array of {name, value} (spec §11 %assign entries)');
  assert.equal(b.rhs.assigns.length, 1);
  assert.equal(b.rhs.assigns[0].name, 'mu');
  assert.ok(b.rhs.assigns[0].value && b.rhs.assigns[0].value.kind,
    'each assign entry carries a value IR node');
});

test('toSexpr round-trips load_module assigns without crashing', () => {
  // Was: "e.assigns is not iterable" on the object shape (TODO §11).
  const r = processSource('a = elementof(reals)\nm = load_module("h.flatppl", mu = a)');
  const sexpr = pirSexpr.toSexpr(r.loweredModule);
  assert.match(sexpr, /%assign mu/);
});

// ---------------------------------------------------------------------
// Module type (spec §11 `%module`): a `load_module(...)` /
// `standard_module(...)` binding is a module reference, not a value —
// it carries no value set and is not callable / measure-typed.
// ---------------------------------------------------------------------

test('a load_module binding types as `module`', () => {
  const r = processSource('m = load_module("h.flatppl")');
  const lb = r.loweredModule.bindings.get('m');
  assert.equal(lb.inferredType.kind, 'module');
});

test('a standard_module binding types as `module`', () => {
  const r = processSource('s = standard_module("polynomials", "0.1")');
  const lb = r.loweredModule.bindings.get('s');
  assert.equal(lb.inferredType.kind, 'module');
});

// ---------------------------------------------------------------------
// Bundle compilation (spec §04 Module composition): the host pre-resolves
// each `.flatppl` dependency into `opts.bundle.sources` (keyed by resolved
// path); the engine recursively compiles each referenced dep into its own
// LoweredModule (modules do NOT flatten — spec §11). Resolution failures
// (missing source, cycle) surface as diagnostics, not crashes.
// ---------------------------------------------------------------------

function errs(diags: any[]) { return diags.filter((d: any) => d.severity === 'error'); }

test('a load_module dependency is compiled into its own module', () => {
  const r = processSource('m = load_module("helpers.flatppl")',
    { bundle: { sources: { 'helpers.flatppl': 'x = 1\ny = x + 2' } } });
  assert.ok(r.modules instanceof Map, 'result carries a `modules` registry (Map)');
  const dep = r.modules.get('helpers.flatppl');
  assert.ok(dep, 'helpers.flatppl is compiled and registered by resolved path');
  assert.ok(dep.loweredModule.bindings.has('x'), 'dep binding x compiled');
  assert.ok(dep.loweredModule.bindings.has('y'), 'dep binding y compiled');
  assert.equal(errs(r.diagnostics).length, 0, 'no errors for a well-formed bundle');
});

test('the load_module binding records its resolved path in the module registry', () => {
  const r = processSource('m = load_module("helpers.flatppl")',
    { bundle: { sources: { 'helpers.flatppl': 'x = 1' } } });
  const reg = r.loweredModule.moduleRegistry || {};
  assert.ok(reg.m, 'alias m registered');
  assert.equal(reg.m.kind, 'load_module');
  assert.equal(reg.m.path, 'helpers.flatppl');
});

test('a missing module source is a clear error on the load_module call', () => {
  const r = processSource('m = load_module("nope.flatppl")', { bundle: { sources: {} } });
  assert.ok(errs(r.diagnostics).some((d) => /nope\.flatppl/.test(d.message)
    && /not found|could not (be )?resolve/i.test(d.message)),
  'missing dep source surfaces a diagnostic naming the path');
});

test('a module cycle is detected and reported, not infinite-looped', () => {
  const r = processSource('a = load_module("a.flatppl")', {
    bundle: { sources: {
      'a.flatppl': 'b = load_module("b.flatppl")',
      'b.flatppl': 'a = load_module("a.flatppl")',
    } },
  });
  // entry → a → b → a : the back-edge to a is a cycle.
  const all = [r.diagnostics].concat(
    [...r.modules.values()].map((m: any) => m.diagnostics));
  assert.ok(all.flat().some((d: any) => d.severity === 'error' && /cycle/i.test(d.message)),
    'cycle reported (and compilation terminates)');
});

test('relative dependency paths resolve against the importer directory', () => {
  const r = processSource('m = load_module("../shared/h.flatppl")', {
    path: 'demo/model.flatppl',
    bundle: { sources: { 'shared/h.flatppl': 'z = 5' } },
  });
  const dep = r.modules.get('shared/h.flatppl');
  assert.ok(dep && dep.loweredModule.bindings.has('z'),
    'dependency resolved relative to the importer dir');
  assert.equal(errs(r.diagnostics).length, 0);
});

test('single-file source: modules registry is an empty Map', () => {
  const r = processSource('x = 1');
  assert.ok(r.modules instanceof Map && r.modules.size === 0);
});
