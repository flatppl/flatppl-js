'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function errors(src: any) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

// --- Hole (_) ---

test('hole: valid inside fn(...)', () => {
  assert.equal(errors('f = fn(_ * 2)\n').length, 0);
});

test('hole: multiple holes inside fn(...) all valid', () => {
  assert.equal(errors('f = fn(_ + _ * _)\n').length, 0);
});

test('hole: error when used outside fn', () => {
  const errs = errors('x = _ + 1\n');
  assert.ok(errs.some((d: any) => /Hole.*fn/.test(d.message)));
});

test('hole: error when used inside functionof', () => {
  // _ is hole-only-in-fn; placeholder _x_ is for functionof
  const errs = errors('f = functionof(_ * 2, x = _x_)\n');
  assert.ok(errs.some((d: any) => /Hole.*fn/.test(d.message)));
});

test('hole: error when used inside kernelof', () => {
  const errs = errors('m = kernelof(_, x = _x_)\n');
  assert.ok(errs.some((d: any) => /Hole.*fn/.test(d.message)));
});

test('hole: nested fn — inner fn redefines the hole scope', () => {
  // fn(...) inside fn(...) — both _ are valid
  assert.equal(errors('f = fn(fn(_ + 1)(_))\n').length, 0);
});

// --- Placeholder (_name_) ---

test('placeholder: valid inside functionof', () => {
  assert.equal(errors('f = functionof(_par_ * 2, par = _par_)\n').length, 0);
});

test('placeholder: valid inside kernelof', () => {
  assert.equal(errors('m = kernelof(_x_, x = _x_)\n').length, 0);
});

test('placeholder: error when used outside reification', () => {
  const errs = errors('x = _par_ * 2\n');
  assert.ok(errs.some((d: any) => /Placeholder.*functionof.*kernelof/.test(d.message)));
});

test('placeholder: error when used inside fn', () => {
  // fn allows holes only, not placeholders.
  const errs = errors('f = fn(_par_ + _)\n');
  assert.ok(errs.some((d: any) => /Placeholder.*functionof.*kernelof/.test(d.message)));
});

// --- Placeholder scoping (§04 "Placeholders and holes", scoping rule) ---
//
// "The scope of a placeholder is the nearest enclosing `functionof` or
// `kernelof`. The same placeholder name may appear in different scopes
// without conflict … A placeholder in an inner `functionof` or `kernelof`
// **must** be bound there".

test('placeholder: §04 licenses the same name in two nested scopes', () => {
  // The spec's own example, verbatim.
  assert.equal(errors(
    'b = 2.0\nsome_value = 3.0\n'
    + 'f = functionof(functionof(_a_ * b, a = _a_)(some_value) + _a_, a = _a_)\n',
  ).length, 0);
});

test('placeholder: §04\'s DISALLOWED nested example is refused', () => {
  // "A placeholder in an inner `functionof` or `kernelof` **must** be bound
  // there, so this code is invalid" — §04, quoted verbatim below. The inner
  // reification binds only `a`, so its body's `_c_` resolves to nothing.
  const errs = errors(
    'b = 2.0\nsome_value = 3.0\n'
    + 'f = functionof(functionof(_a_ * b + _c_, a = _a_)(some_value) + _d_, '
    + 'c = _c_, d = _d_)\n',
  );
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /Placeholder '_c_' is not bound by the nearest/);
  assert.match(errs[0].message, /§04/);
});

test('placeholder: §04\'s aggregate-inside-functionof example is accepted', () => {
  // "`aggregate` composes cleanly with `functionof` as the namespace of axis
  // names is local to the enclosing `aggregate` and the namespace of
  // placeholders is local to the enclosing `functionof`" — the axis scope
  // must not cut the body off from the reification's placeholders.
  assert.equal(errors(
    'mymatmul = functionof(\n'
    + '    aggregate(sum, [.i, .k], _A_[.i, .j] * _B_[.j, .k]),\n'
    + '    A = _A_, B = _B_\n)\n',
  ).length, 0);
});

test('placeholder: a curried lambda keeps the outer arg in scope', () => {
  // §04's lambda rule rewrites `x -> y -> x + y` to nested `functionof`s
  // whose inner body reads `_x_`. The nesting is the rewrite's, so the
  // scoping rule above does not bite.
  assert.equal(errors('f = x -> y -> x + y\n').length, 0);
  assert.equal(errors('f = a -> (a -> a + 1)\n').length, 0);
});

test('placeholder: a USER-written inner functionof still must bind it', () => {
  const errs = errors('f = functionof(functionof(_x_ + 1, z = _z_), x = _x_)\n');
  assert.ok(errs.some((d: any) => /'_x_' is not bound by the nearest/.test(d.message)));
});

// --- LHS underscore is fine (not a hole or placeholder per parser) ---

test('decomposition with bare _ on LHS is not flagged', () => {
  // The LHS _ is a Name (discard binding), not a hole reference
  assert.equal(errors('value, _ = (1, 2)\n').length, 0);
});

// --- Real-world examples should still pass ---

test('bayesian_inference_2 fixture parses cleanly', () => {
  const fs = require('node:fs');
  const path = require('node:path');
  const file = path.join(__dirname, 'fixtures', 'bayesian_inference_2.flatppl');
  const src = fs.readFileSync(file, 'utf8');
  assert.equal(errors(src).length, 0);
});
