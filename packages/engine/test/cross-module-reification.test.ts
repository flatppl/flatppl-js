'use strict';

// Spec §04 "Reification and module scope" (design commit 2a558dd):
// `functionof` / `kernelof` reify within the current module ONLY. A
// parameterized value reached through a loaded-module reference cannot
// become an input — neither by the automatic trace nor as an explicit
// boundary node (directly or via a pure alias) — so such a reification is a
// STATIC error. A loaded module's callables and fixed values MAY be used in
// the reified DAG (applied, or referenced and closed over).
//
// The host I/O is mocked via `opts.bundle.sources` (engine stays sync).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

function errs(r: any) {
  return (r.diagnostics || []).filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}
function compile(main: string, sources: Record<string, string>) {
  return processSource(main, { bundle: { sources } });
}
// A cross-module-reification diagnostic: mentions the module-boundary reify rule.
function xmodReifErr(r: any): string | undefined {
  return errs(r).find((m: string) => /reif/i.test(m) && /module/i.test(m));
}

const MOD = 'theta = elementof(reals)\nshifted = theta + 1.0\nc = 5.0\ng = fn(_ + 1.0)';

// ---------------------------------------------------------------------
// VIOLATIONS — must be a static error
// ---------------------------------------------------------------------

test('auto-trace reaching a cross-module elementof is a static error', () => {
  const r = compile('m = load_module("mod.flatppl")\nf = functionof(m.theta * 2.0)',
    { 'mod.flatppl': MOD });
  assert.ok(xmodReifErr(r), 'expected a cross-module reification error, got: ' + errs(r).join(' | '));
});

test('auto-trace reaching a cross-module DERIVED parameterized value is a static error', () => {
  const r = compile('m = load_module("mod.flatppl")\nf = functionof(m.shifted * 2.0)',
    { 'mod.flatppl': MOD });
  assert.ok(xmodReifErr(r), 'shifted is parameterized (theta+1); got: ' + errs(r).join(' | '));
});

// NB a bare `functionof(…, p = m.theta)` FieldAccess boundary is rejected at
// lowering (a boundary must be an identifier or placeholder), so the canonical
// way to name a cross-module value as a boundary is via a local alias — that is
// the boundary case the rule targets (covered below).

test('auto-trace reaching a cross-module param via a LOCAL ALIAS is a static error', () => {
  const r = compile('m = load_module("mod.flatppl")\na = m.theta\nf = functionof(a * 2.0)',
    { 'mod.flatppl': MOD });
  assert.ok(xmodReifErr(r), 'a aliases m.theta; got: ' + errs(r).join(' | '));
});

test('a boundary that pure-aliases a cross-module param is a static error', () => {
  const r = compile('m = load_module("mod.flatppl")\na = m.theta\nf = functionof(a * 2.0, p = a)',
    { 'mod.flatppl': MOD });
  assert.ok(xmodReifErr(r), 'boundary p = a, a = m.theta; got: ' + errs(r).join(' | '));
});

test('kernelof with a cross-module parameterized boundary (via alias) is a static error', () => {
  // kernelof(x, mu = mu_a) lowers to functionof(lawof(x), mu = mu_a); the
  // boundary mu_a aliases the cross-module m.theta.
  const r = compile(
    'm = load_module("mod.flatppl")\nmu_a = m.theta\nx ~ Normal(mu = mu_a, sigma = 1.0)\n'
    + 'K = kernelof(x, mu = mu_a)',
    { 'mod.flatppl': MOD });
  assert.ok(xmodReifErr(r), 'kernelof boundary mu = mu_a (= m.theta); got: ' + errs(r).join(' | '));
});

test('auto-trace where a cross-module param is a transitive ancestor via a local derived node', () => {
  const r = compile(
    'm = load_module("mod.flatppl")\na = elementof(reals)\ny = m.theta * a\nf = functionof(y)',
    { 'mod.flatppl': MOD });
  assert.ok(xmodReifErr(r), 'y = m.theta * a, auto-trace reaches m.theta; got: ' + errs(r).join(' | '));
});

// ---------------------------------------------------------------------
// ALLOWED — cross-module callables / fixed values / cut-off boundaries
// ---------------------------------------------------------------------

test('closing over a cross-module FIXED value is allowed', () => {
  const r = compile('a = elementof(reals)\nm = load_module("mod.flatppl")\nf = functionof(a * m.c)',
    { 'mod.flatppl': MOD });
  assert.equal(xmodReifErr(r), undefined, 'm.c is fixed; got: ' + errs(r).join(' | '));
});

test('applying a cross-module CALLABLE is allowed', () => {
  const r = compile('a = elementof(reals)\nm = load_module("mod.flatppl")\nf = functionof(m.g(a))',
    { 'mod.flatppl': MOD });
  assert.equal(xmodReifErr(r), undefined, 'm.g is a callable; got: ' + errs(r).join(' | '));
});

test('a boundary at a LOCAL DERIVED node cuts off the cross-module dependency (allowed)', () => {
  // y = m.theta * 2.0 is a current-module node; the boundary at y makes y the
  // input and cuts off m.theta upstream — m.theta never becomes an input.
  const r = compile(
    'm = load_module("mod.flatppl")\ny = m.theta * 2.0\nf = functionof(y * 3.0, p = y)',
    { 'mod.flatppl': MOD });
  assert.equal(xmodReifErr(r), undefined, 'boundary at derived y cuts m.theta; got: ' + errs(r).join(' | '));
});

test('same-module reification is unaffected', () => {
  const r = compile('theta = elementof(reals)\nf = functionof(theta * 2.0)', {});
  assert.equal(xmodReifErr(r), undefined, 'no module involved; got: ' + errs(r).join(' | '));
});

test('a bundle-less compile does not flag cross-module reification (no dep to resolve)', () => {
  // Editor-lint tolerance: without a bundle the dep is unresolved, so the
  // phase can't be known — no spurious error (mirrors the load_module lint rule).
  const r = processSource('m = load_module("mod.flatppl")\nf = functionof(m.theta * 2.0)');
  assert.equal(xmodReifErr(r), undefined, 'no bundle → no cross-module resolution; got: ' + errs(r).join(' | '));
});
