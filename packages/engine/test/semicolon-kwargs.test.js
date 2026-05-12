'use strict';

// FlatPPJ semicolon kwargs (spec §05):
//   f(x; a = 1, b = 2)  ≡  f(x, a = 1, b = 2)
//   f(; a = 1)          ≡  f(a = 1)
// Allowed in FlatPPJ only; FlatPPL and FlatPPY produce a diagnostic.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');

function parseOK(src, opts) {
  const r = processSource(src, opts);
  const errors = r.diagnostics.filter(d =>
    d.severity === 'error'
    || (d.severity === 'warning' && !/Undefined variable/.test(d.message)));
  assert.deepEqual(errors, [], 'expected no parse errors, got: '
    + JSON.stringify(errors));
  return r;
}

function rhs(r) { return r.bindings.get('x').node.value; }

// ---------------------------------------------------------------------
// FlatPPJ accepts ; with the expected lowering
// ---------------------------------------------------------------------

test('semi: FlatPPJ `f(; a = 1)` ≡ `f(a = 1)`', () => {
  const a = rhs(parseOK('x = f(; a = 1)',  { variant: 'flatppj' }));
  const b = rhs(parseOK('x = f(a = 1)',    { variant: 'flatppj' }));
  assert.equal(a.type, 'CallExpr');
  assert.equal(a.callee.name, 'f');
  assert.equal(a.args.length, 1);
  assert.equal(a.args[0].type, 'KeywordArg');
  assert.equal(a.args[0].name, 'a');
  // Same arg shape as the comma form
  assert.equal(a.args.length, b.args.length);
});

test('semi: FlatPPJ `f(x; a = 1, b = 2)` mixes positional + kwargs', () => {
  const v = rhs(parseOK('x = f(p; a = 1, b = 2)', { variant: 'flatppj' }));
  assert.equal(v.args.length, 3);
  assert.equal(v.args[0].type, 'Identifier');
  assert.equal(v.args[0].name, 'p');
  assert.equal(v.args[1].type, 'KeywordArg');
  assert.equal(v.args[1].name, 'a');
  assert.equal(v.args[2].type, 'KeywordArg');
  assert.equal(v.args[2].name, 'b');
});

test('semi: FlatPPJ `f(p, q; a = 1)` allows multiple positional before `;`', () => {
  const v = rhs(parseOK('x = f(p, q; a = 1)', { variant: 'flatppj' }));
  assert.equal(v.args.length, 3);
  assert.equal(v.args[0].name, 'p');
  assert.equal(v.args[1].name, 'q');
  assert.equal(v.args[2].type, 'KeywordArg');
});

test('semi: FlatPPJ kwarg form parses across normal call sites', () => {
  // Real spec example: tilde + ; combo
  const v = rhs(parseOK('x ~ Normal(;mu = 0, sigma = 1)', { variant: 'flatppj' }));
  // x ~ M lowers to draw(M); the outer is the draw call, inner is
  // Normal with two kwargs.
  assert.equal(v.callee.name, 'draw');
  const inner = v.args[0];
  assert.equal(inner.callee.name, 'Normal');
  assert.equal(inner.args.length, 2);
});

// ---------------------------------------------------------------------
// FlatPPL and FlatPPY reject `;`
// ---------------------------------------------------------------------

test('semi: FlatPPL rejects `;` with a clear diagnostic', () => {
  const r = processSource('x = f(; a = 1)', { variant: 'flatppl' });
  const errors = r.diagnostics.filter(d => d.severity === 'error');
  assert.ok(errors.length >= 1);
  assert.match(errors[0].message, /';' is not allowed in flatppl/);
});

test('semi: FlatPPY rejects `;` with a clear diagnostic', () => {
  const r = processSource('x = f(p; a = 1)', { variant: 'flatppy' });
  const errors = r.diagnostics.filter(d => d.severity === 'error');
  assert.ok(errors.length >= 1);
  assert.match(errors[0].message, /';' is not allowed in flatppy/);
});

test('semi: FlatPPL `f(p, a = 1)` (comma form) still works', () => {
  const v = rhs(parseOK('x = f(p, a = 1)', { variant: 'flatppl' }));
  assert.equal(v.args.length, 2);
  assert.equal(v.args[0].name, 'p');
  assert.equal(v.args[1].type, 'KeywordArg');
});

// ---------------------------------------------------------------------
// Positional after kwarg via `;` is still a positional-after-kwarg error
// ---------------------------------------------------------------------

test('semi: FlatPPJ `f(; a = 1, p)` flags positional after kwarg', () => {
  const r = processSource('x = f(; a = 1, p)', { variant: 'flatppj' });
  const errors = r.diagnostics.filter(d => d.severity === 'error');
  assert.ok(errors.length >= 1);
  assert.match(errors[0].message,
    /Expected keyword argument after `;`|Positional argument cannot follow keyword argument/);
});
