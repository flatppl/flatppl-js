'use strict';

// Chained comparisons (spec §05): `a < b <= c` lowers to
// `land(a < b, b <= c)`. All three variants support chaining.
// `in` is a comparison operator; it can participate in chains.

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
// Single comparison stays a plain BinaryExpr (no land wrapping)
// ---------------------------------------------------------------------

test('chain: FlatPPL `a < b` stays a plain BinaryExpr', () => {
  const v = parseRHS('x = a < b', { variant: 'flatppl' });
  assert.equal(v.type, 'BinaryExpr');
  assert.equal(v.op, '<');
});

// ---------------------------------------------------------------------
// FlatPPL: chained → land(...)
// ---------------------------------------------------------------------

test('chain: FlatPPL `a < b <= c` → land(a<b, b<=c)', () => {
  const v = parseRHS('x = a < b <= c', { variant: 'flatppl' });
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args.length, 2);
  assert.equal(v.args[0].type, 'BinaryExpr');
  assert.equal(v.args[0].op, '<');
  assert.equal(v.args[0].left.name, 'a');
  assert.equal(v.args[0].right.name, 'b');
  assert.equal(v.args[1].type, 'BinaryExpr');
  assert.equal(v.args[1].op, '<=');
  assert.equal(v.args[1].left.name, 'b');
  assert.equal(v.args[1].right.name, 'c');
});

test('chain: FlatPPL `a < b <= c < d` is left-associative', () => {
  // land(land(a<b, b<=c), c<d)
  const v = parseRHS('x = a < b <= c < d', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args[0].type, 'CallExpr');
  assert.equal(v.args[0].callee.name, 'land');
  assert.equal(v.args[0].args[0].op, '<');
  assert.equal(v.args[0].args[1].op, '<=');
  assert.equal(v.args[1].op, '<');
  assert.equal(v.args[1].left.name, 'c');
  assert.equal(v.args[1].right.name, 'd');
});

test('chain: FlatPPL mixed operators `1 < a == b != 0`', () => {
  // land(land(1<a, a==b), b!=0)
  const v = parseRHS('x = 1 < a == b != 0', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args[0].callee.name, 'land');
  assert.equal(v.args[0].args[0].op, '<');
  assert.equal(v.args[0].args[1].op, '==');
  assert.equal(v.args[1].op, '!=');
});

// ---------------------------------------------------------------------
// `in` operator
// ---------------------------------------------------------------------

test('chain: FlatPPL `x in S` is a BinaryExpr with op "in"', () => {
  const v = parseRHS('x = a in S', { variant: 'flatppl' });
  assert.equal(v.type, 'BinaryExpr');
  assert.equal(v.op, 'in');
  assert.equal(v.left.name, 'a');
  assert.equal(v.right.name, 'S');
});

test('chain: `a < b in S` chains: land(a<b, b in S)', () => {
  const v = parseRHS('x = a < b in S', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args[0].op, '<');
  assert.equal(v.args[1].op, 'in');
  assert.equal(v.args[1].right.name, 'S');
});

// ---------------------------------------------------------------------
// Interaction with logical operators
// ---------------------------------------------------------------------

test('chain: FlatPPL `a < b <= c && d` — chain has tighter precedence', () => {
  // land(land(a<b, b<=c), d) — the outer && wraps the chain result
  const v = parseRHS('x = a < b <= c && d', { variant: 'flatppl' });
  // The outermost is `&&` (also land) — but the chain itself is a
  // land call. So we get land(land(a<b, b<=c), d).
  assert.equal(v.callee.name, 'land');
  assert.equal(v.args[0].callee.name, 'land');  // the inner chain
  assert.equal(v.args[1].name, 'd');
});

test('chain: `!(a < b < c)` — `!` wraps the chain', () => {
  // `!` is above Comparison → lnot(land(a<b, b<c))
  const v = parseRHS('x = !(a < b < c)', { variant: 'flatppl' });
  assert.equal(v.callee.name, 'lnot');
  assert.equal(v.args[0].callee.name, 'land');
});
