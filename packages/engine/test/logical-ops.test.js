'use strict';

// Logical operators (spec §05):
//   FlatPPL/FlatPPJ — `&&`, `||`, `!`        lower to land/lor/lnot
//   FlatPPY         — `and`, `or`, `not`     lower to land/lor/lnot
// Precedence:
//   FlatPPL/FlatPPJ — `!` is Unary  → binds tighter than Comparison
//   FlatPPY         — `not` is above Comparison (Python rule)

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');

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

// ---------------------------------------------------------------------
// FlatPPJ: same as FlatPPL
// ---------------------------------------------------------------------

test('logical: FlatPPJ `a && b` → land(a, b)', () => {
  const v = parseRHS('x = a && b', { variant: 'flatppj' });
  assert.equal(v.callee.name, 'land');
});

test('logical: FlatPPJ `!a` → lnot(a)', () => {
  const v = parseRHS('x = !a', { variant: 'flatppj' });
  assert.equal(v.callee.name, 'lnot');
});

// ---------------------------------------------------------------------
// FlatPPY: and / or / not
// ---------------------------------------------------------------------

test('logical: FlatPPY `a and b` → land(a, b)', () => {
  const v = parseRHS('x = a and b', { variant: 'flatppy' });
  assert.equal(v.callee.name, 'land');
});

test('logical: FlatPPY `a or b` → lor(a, b)', () => {
  const v = parseRHS('x = a or b', { variant: 'flatppy' });
  assert.equal(v.callee.name, 'lor');
});

test('logical: FlatPPY `not a` → lnot(a)', () => {
  const v = parseRHS('x = not a', { variant: 'flatppy' });
  assert.equal(v.callee.name, 'lnot');
  assert.equal(v.args[0].name, 'a');
});

test('logical: FlatPPY `not a < b` parses as `not (a < b)` (Python rule)', () => {
  // FlatPPY's `not` is above Comparison.
  const v = parseRHS('x = not a < b', { variant: 'flatppy' });
  assert.equal(v.callee.name, 'lnot');
  assert.equal(v.args[0].type, 'BinaryExpr');
  assert.equal(v.args[0].op, '<');
});

test('logical: FlatPPY `not not a` → lnot(lnot(a))', () => {
  const v = parseRHS('x = not not a', { variant: 'flatppy' });
  assert.equal(v.callee.name, 'lnot');
  assert.equal(v.args[0].callee.name, 'lnot');
});

test('logical: FlatPPY `a or b and c` — `and` binds tighter than `or`', () => {
  const v = parseRHS('x = a or b and c', { variant: 'flatppy' });
  assert.equal(v.callee.name, 'lor');
  assert.equal(v.args[1].callee.name, 'land');
});

test('logical: FlatPPY `a and not b` → land(a, lnot(b))', () => {
  const v = parseRHS('x = a and not b', { variant: 'flatppy' });
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args[0].name, 'a');
  assert.equal(v.args[1].callee.name, 'lnot');
  assert.equal(v.args[1].args[0].name, 'b');
});

// ---------------------------------------------------------------------
// Cross-variant equivalence at the AST level
// ---------------------------------------------------------------------

test('logical: FlatPPL `!a` and FlatPPY `not a` produce identical lnot calls', () => {
  const vL = parseRHS('x = !a',    { variant: 'flatppl' });
  const vY = parseRHS('x = not a', { variant: 'flatppy' });
  assert.equal(vL.callee.name, vY.callee.name);
  assert.equal(vL.args[0].name, vY.args[0].name);
});

test('logical: FlatPPL `a && b` and FlatPPY `a and b` produce identical land calls', () => {
  const vL = parseRHS('x = a && b',  { variant: 'flatppl' });
  const vY = parseRHS('x = a and b', { variant: 'flatppy' });
  assert.equal(vL.callee.name, vY.callee.name);
  assert.equal(vL.args[0].name, vY.args[0].name);
  assert.equal(vL.args[1].name, vY.args[1].name);
});
