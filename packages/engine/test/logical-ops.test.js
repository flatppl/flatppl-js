'use strict';

// Logical operators (spec §05):
//   FlatPPL/FlatPPJ — `&&`, `||`, `!`        lower to land/lor/lnot
//   FlatPPY         — `and`, `or`, `not`     lower to land/lor/lnot
// Precedence:
//   FlatPPL/FlatPPJ — `!` is Unary  → binds tighter than Comparison
//   FlatPPY         — `not` is above Comparison (Python rule)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

function parseRHS(src, opts) {
  const r = processSource(src, opts);
  const errors = r.diagnostics.filter(d =>
    d.severity === 'error'
    || (d.severity === 'warning' && !/Undefined variable/.test(d.message)));
  assert.deepEqual(errors, [], 'expected no parse errors, got: '
    + JSON.stringify(errors));
  return r.bindings.get('x').node.value;
}

// ---------------------------------------------------------------------
// FlatPPL: && / || / !
// ---------------------------------------------------------------------

test('logical: FlatPPL `a && b` → land(a, b)', () => {
  const v = parseRHS('x = a && b', { variant: 'flatppl' });
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args.length, 2);
  assert.equal(v.args[0].name, 'a');
  assert.equal(v.args[1].name, 'b');
});

test('logical: FlatPPL `a || b` → lor(a, b)', () => {
  const v = parseRHS('x = a || b', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'lor');
});

test('logical: FlatPPL `!a` → lnot(a)', () => {
  const v = parseRHS('x = !a', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'lnot');
  assert.equal(v.args.length, 1);
  assert.equal(v.args[0].name, 'a');
});

test('logical: FlatPPL `!!a` → lnot(lnot(a))', () => {
  const v = parseRHS('x = !!a', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'lnot');
  assert.equal(v.args[0].callee.name, 'lnot');
  assert.equal(v.args[0].args[0].name, 'a');
});

test('logical: FlatPPL `!a < b` parses as `(!a) < b`', () => {
  // ! is Unary → binds tighter than Comparison.
  const v = parseRHS('x = !a < b', { variant: 'flatppl' });
  assert.equal(v.type, 'BinaryExpr');
  assert.equal(v.op, '<');
  assert.equal(v.left.type, 'CallExpr');
  assert.equal(v.left.callee.name, 'lnot');
});

test('logical: FlatPPL `a || b && c` — && binds tighter than ||', () => {
  // a || (b && c) — Or → And → Comparison
  const v = parseRHS('x = a || b && c', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'lor');
  assert.equal(v.args[0].name, 'a');
  assert.equal(v.args[1].type, 'CallExpr');
  assert.equal(v.args[1].callee.name, 'land');
});

test('logical: FlatPPL `a && b && c` is left-associative', () => {
  // (a && b) && c
  const v = parseRHS('x = a && b && c', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args[0].type, 'CallExpr');
  assert.equal(v.args[0].callee.name, 'land');
  assert.equal(v.args[0].args[0].name, 'a');
  assert.equal(v.args[0].args[1].name, 'b');
  assert.equal(v.args[1].name, 'c');
});
