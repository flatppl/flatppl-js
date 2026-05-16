'use strict';

// Tilde bindings (spec §05): `x ~ M` lowers to `x = draw(M)`.
// Allowed in FlatPPL and FlatPPJ; rejected in FlatPPY.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');

function parseOK(src, opts) {
  const r = processSource(src, opts);
  // The minimal tilde snippets in these tests reference undefined
  // names (M, Normal, …) just to keep them short. Ignore analyzer
  // "Undefined variable" warnings — they aren't the parse-level
  // failures these tests are checking for.
  const errors = r.diagnostics.filter(d =>
    d.severity === 'error'
    || (d.severity === 'warning' && !/Undefined variable/.test(d.message)));
  assert.deepEqual(errors, [], 'expected no parse errors, got: '
    + JSON.stringify(errors));
  return r;
}

function firstBindingValue(r, name) {
  const b = r.bindings.get(name);
  assert.ok(b, `expected binding ${name}`);
  return b.node.value;
}

// ---------------------------------------------------------------------
// FlatPPL: tilde bindings lower to draw(...)
// ---------------------------------------------------------------------

test('tilde: FlatPPL `x ~ M` lowers to draw(M)', () => {
  const r = parseOK('x ~ M', { variant: 'flatppl' });
  const v = firstBindingValue(r, 'x');
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'draw');
  assert.equal(v.args.length, 1);
  assert.equal(v.args[0].type, 'Identifier');
  assert.equal(v.args[0].name, 'M');
});

test('tilde: FlatPPL `x ~ Normal(mu = 0, sigma = 1)` equivalent to draw(...)', () => {
  const rTilde = parseOK('x ~ Normal(mu = 0, sigma = 1)', { variant: 'flatppl' });
  const rDraw  = parseOK('x = draw(Normal(mu = 0, sigma = 1))', { variant: 'flatppl' });
  // The two trees agree structurally on the binding's RHS (modulo
  // synthetic CallExpr's outer loc, which is irrelevant downstream).
  const vTilde = firstBindingValue(rTilde, 'x');
  const vDraw  = firstBindingValue(rDraw,  'x');
  assert.equal(vTilde.callee.name, vDraw.callee.name);
  assert.equal(vTilde.args.length, vDraw.args.length);
});

test('tilde: FlatPPL decomposition `a, b ~ M`', () => {
  const r = parseOK('a, b ~ M', { variant: 'flatppl' });
  // analyze() emits one binding per LHS name (decomposition lowering);
  // both share the same wrapped draw(M) RHS.
  const a = r.bindings.get('a');
  const b = r.bindings.get('b');
  assert.ok(a && b);
  // Each binding's node.value is the draw(M) call; the per-name
  // projection happens in analyzer.lower / pir.
});

test('tilde: FlatPPL `_, rs ~ rand(...)` discards first', () => {
  const r = parseOK('_, rs ~ rand(state, m)', { variant: 'flatppl' });
  // Should parse cleanly even though spec leans on `=` here; the
  // tilde form is still grammatically valid.
  assert.ok(r.bindings.has('rs'));
});

// ---------------------------------------------------------------------
// Path-detected source still accepts tilde (canonical FlatPPL)
// ---------------------------------------------------------------------

test('tilde: path-detected .flatppl accepts tilde', () => {
  parseOK('x ~ M', { path: 'foo.flatppl' });
});
