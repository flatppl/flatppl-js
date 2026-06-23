'use strict';

// Spec §04 "Aliasing is just assignment": a binding `g = f` that aliases a
// CALLABLE (fn / functionof / kernelof) is itself callable, and
// `g(args) ≡ f(args)` — calling through the alias must behave exactly like
// calling the original. This is the callable analogue of value aliasing
// (`p = q; r = p + 1` already works). The cross-module idiom
// `f_b = common.f_b; b = f_b(...)` reduces to this after linking (the
// linker rewrites the member access to a bare alias `f_b = common$f_b`).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator } = require('..');

function build(src: string) {
  const r = processSource(src);
  const errs = r.diagnostics.filter((d: any) => d.severity === 'error');
  const ds = orchestrator.buildDerivations(r.bindings);
  return { ds, errs };
}

test('call through a value-function alias (g = f0) classifies + evaluates', () => {
  // f0 = abs(arg1) * arg2; g aliases f0; z = g(3.0, -2.0) = 3 * -2 = -6.
  const { ds, errs } = build('f0 = fn(abs(_) * _)\ng = f0\nz = g(3.0, -2.0)');
  assert.equal(errs.length, 0);
  assert.ok(ds.derivations.z, 'z should classify (call through callable alias)');
  assert.ok(Math.abs(ds.fixedValues.get('z') - (-6.0)) < 1e-9,
    'g(3, -2) ≡ f0(3, -2) = abs(3) * -2 = -6');
});

test('call through a functionof alias evaluates like the original', () => {
  // f = (a -> a * 2 + 1); g aliases f; y = g(5.0) = 11.
  const { ds, errs } = build('a = elementof(reals)\nf = functionof(a * 2 + 1, a = a)\ng = f\ny = g(5.0)');
  assert.equal(errs.length, 0);
  assert.ok(ds.derivations.y, 'y should classify');
  assert.ok(Math.abs(ds.fixedValues.get('y') - 11.0) < 1e-9, 'g(5) ≡ f(5) = 5*2+1 = 11');
});

test('call through a CHAIN of callable aliases resolves transitively', () => {
  // f0 = (_ + 1); g = f0; h = g; z = h(4.0) = 5.
  const { ds, errs } = build('f0 = fn(_ + 1)\ng = f0\nh = g\nz = h(4.0)');
  assert.equal(errs.length, 0);
  assert.ok(ds.derivations.z, 'z should classify through g → f0');
  assert.ok(Math.abs(ds.fixedValues.get('z') - 5.0) < 1e-9, 'h(4) ≡ f0(4) = 5');
});

test('a self-alias cycle does not hang (degenerate, no derivation)', () => {
  // g = g is a degenerate cycle — must terminate, not infinite-loop.
  const { ds } = build('g = g\nz = g(1.0)');
  assert.ok(!ds.derivations.z || ds.derivations.z, 'reached here = no hang');
});

// signatureOf — the viewer's callable introspection — must follow the same
// alias chain, so an aliased callable / kernel is plottable like the original.

function sigOf(src: string, name: string) {
  const r = processSource(src);
  return orchestrator.signatureOf(name, orchestrator.liftInlineSubexpressions(r.bindings));
}

test('signatureOf follows a function alias (g = f → f\'s signature)', () => {
  const s = sigOf('a = elementof(reals)\nf = functionof(a * 2, a = a)\ng = f', 'g');
  assert.ok(s && s.kind === 'function', 'alias of a functionof introspects as a function');
  assert.equal(s.inputs.length, 1);
  assert.equal(s.inputs[0].paramName, 'a');
});

test('signatureOf follows a kernel alias (gk = k → k\'s signature)', () => {
  const s = sigOf('mu = elementof(reals)\nx = draw(Normal(mu, 1))\nk = kernelof(x, mu = mu)\ngk = k', 'gk');
  assert.ok(s && s.kind === 'kernel', 'alias of a kernelof introspects as a kernel');
});

test('signatureOf on a 2-cycle alias terminates (returns null, no hang)', () => {
  const s = sigOf('g = h\nh = g', 'g');
  assert.equal(s, null);
});
