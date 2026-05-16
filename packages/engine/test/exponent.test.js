'use strict';

// `^` exponentiation (spec §05): right-associative, binds tighter
// than unary `-`, lowers to `pow(base, exponent)`. FlatPPL/FlatPPJ
// accept it; FlatPPY does not.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');

function parseRHS(src, opts) {
  const r = processSource(src, opts);
  // Filter analyzer "undefined variable" warnings — these snippets
  // reference bare identifiers just to keep them short.
  const errors = r.diagnostics.filter(d =>
    d.severity === 'error'
    || (d.severity === 'warning' && !/Undefined variable/.test(d.message)));
  assert.deepEqual(errors, [], 'expected no parse errors, got: '
    + JSON.stringify(errors));
  return r.bindings.get('x').node.value;
}

test('exp: FlatPPL `x = a ^ b` lowers to pow(a, b)', () => {
  const v = parseRHS('x = a ^ b', { variant: 'flatppl' });
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'pow');
  assert.equal(v.args.length, 2);
  assert.equal(v.args[0].name, 'a');
  assert.equal(v.args[1].name, 'b');
});

test('exp: FlatPPL `x = a ^ b ^ c` is right-associative', () => {
  // a ^ (b ^ c) = pow(a, pow(b, c))
  const v = parseRHS('x = a ^ b ^ c', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'pow');
  assert.equal(v.args[0].name, 'a');
  assert.equal(v.args[1].type, 'CallExpr');
  assert.equal(v.args[1].callee.name, 'pow');
  assert.equal(v.args[1].args[0].name, 'b');
  assert.equal(v.args[1].args[1].name, 'c');
});

test('exp: FlatPPL `x = -a ^ 2` parses as -(a^2)', () => {
  // Spec: ^ binds tighter than unary `-`. So `-a ^ 2` = `UnaryExpr(-, pow(a, 2))`.
  const v = parseRHS('x = -a ^ 2', { variant: 'flatppl' });
  assert.equal(v.type, 'UnaryExpr');
  assert.equal(v.op, '-');
  assert.equal(v.operand.type, 'CallExpr');
  assert.equal(v.operand.callee.name, 'pow');
  assert.equal(v.operand.args[0].name, 'a');
  assert.equal(v.operand.args[1].value, 2);
});

test('exp: FlatPPL `x = (a + b) ^ 2` respects parens', () => {
  const v = parseRHS('x = (a + b) ^ 2', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'pow');
  assert.equal(v.args[0].type, 'BinaryExpr');
  assert.equal(v.args[0].op, '+');
});

test('exp: FlatPPL `x = a * b ^ c` — ^ binds tighter than *', () => {
  // a * (b ^ c) — multiplicative parses ^ as the right operand
  const v = parseRHS('x = a * b ^ c', { variant: 'flatppl' });
  assert.equal(v.type, 'BinaryExpr');
  assert.equal(v.op, '*');
  assert.equal(v.left.name, 'a');
  assert.equal(v.right.type, 'CallExpr');
  assert.equal(v.right.callee.name, 'pow');
});

test('exp: FlatPPL pow(a, b) parses identically to a ^ b at AST level', () => {
  const fromOp = parseRHS('x = a ^ b', { variant: 'flatppl' });
  const fromFn = parseRHS('x = pow(a, b)', { variant: 'flatppl' });
  assert.equal(fromOp.callee.name, fromFn.callee.name);
  assert.equal(fromOp.args.length, fromFn.args.length);
  assert.equal(fromOp.args[0].name, fromFn.args[0].name);
  assert.equal(fromOp.args[1].name, fromFn.args[1].name);
});
