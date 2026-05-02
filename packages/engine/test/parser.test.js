'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { tokenize } = require('../tokenizer');
const { parse } = require('../parser');

function parseSrc(src) {
  return parse(tokenize(src).tokens);
}

test('parser: empty file', () => {
  const { ast, diagnostics } = parseSrc('');
  assert.equal(ast.body.length, 0);
  assert.equal(diagnostics.length, 0);
});

test('parser: comment-only file', () => {
  const { ast, diagnostics } = parseSrc('# only a comment\n# another\n');
  assert.equal(ast.body.length, 0);
  assert.equal(diagnostics.length, 0);
});

test('parser: blank lines between statements', () => {
  const { ast, diagnostics } = parseSrc('a = 1\n\n\nb = 2\n');
  assert.equal(ast.body.length, 2);
  assert.equal(diagnostics.length, 0);
});

test('parser: simple binding', () => {
  const { ast } = parseSrc('x = 42');
  assert.equal(ast.body.length, 1);
  const stmt = ast.body[0];
  assert.equal(stmt.type, 'AssignStatement');
  assert.equal(stmt.names.length, 1);
  assert.equal(stmt.names[0].name, 'x');
  assert.equal(stmt.value.type, 'NumberLiteral');
  assert.equal(stmt.value.value, 42);
});

test('parser: decomposition', () => {
  const { ast } = parseSrc('a, b, c = something');
  const stmt = ast.body[0];
  assert.deepEqual(stmt.names.map(n => n.name), ['a', 'b', 'c']);
});

test('parser: decomposition with bare _', () => {
  const { ast, diagnostics } = parseSrc('value, _ = f()');
  assert.equal(diagnostics.length, 0);
  const stmt = ast.body[0];
  assert.deepEqual(stmt.names.map(n => n.name), ['value', '_']);
});

test('parser: bare _ as LHS', () => {
  const { ast, diagnostics } = parseSrc('_ = f()');
  assert.equal(diagnostics.length, 0);
  assert.equal(ast.body[0].names[0].name, '_');
});

test('parser: arithmetic precedence', () => {
  // a + b * c parses as a + (b * c)
  const { ast } = parseSrc('x = a + b * c');
  const v = ast.body[0].value;
  assert.equal(v.type, 'BinaryExpr');
  assert.equal(v.op, '+');
  assert.equal(v.right.type, 'BinaryExpr');
  assert.equal(v.right.op, '*');
});

test('parser: left-associative addition', () => {
  // a + b + c parses as (a + b) + c
  const { ast } = parseSrc('x = a + b + c');
  const v = ast.body[0].value;
  assert.equal(v.type, 'BinaryExpr');
  assert.equal(v.op, '+');
  assert.equal(v.left.type, 'BinaryExpr');
  assert.equal(v.left.op, '+');
});

test('parser: unary minus', () => {
  const { ast } = parseSrc('x = -5');
  const v = ast.body[0].value;
  assert.equal(v.type, 'UnaryExpr');
  assert.equal(v.op, '-');
  assert.equal(v.operand.type, 'NumberLiteral');
});

test('parser: function call with positional args', () => {
  const { ast } = parseSrc('x = f(a, b, c)');
  const v = ast.body[0].value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.args.length, 3);
  for (const a of v.args) assert.equal(a.type, 'Identifier');
});

test('parser: function call with keyword args', () => {
  const { ast } = parseSrc('x = f(a = 1, b = 2)');
  const v = ast.body[0].value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.args.length, 2);
  assert.equal(v.args[0].type, 'KeywordArg');
  assert.equal(v.args[0].name, 'a');
});

test('parser: function call with mixed args', () => {
  const { ast, diagnostics } = parseSrc('x = f(a, b = 2)');
  assert.equal(diagnostics.length, 0);
  const v = ast.body[0].value;
  assert.equal(v.args[0].type, 'Identifier');
  assert.equal(v.args[1].type, 'KeywordArg');
});

test('parser: positional after kwarg is an error', () => {
  const { diagnostics } = parseSrc('x = f(a = 1, b)');
  assert.ok(diagnostics.some(d =>
    d.severity === 'error' && /Positional.*keyword/.test(d.message)));
});

test('parser: indexing', () => {
  const { ast } = parseSrc('x = A[1, 2]');
  const v = ast.body[0].value;
  assert.equal(v.type, 'IndexExpr');
  assert.equal(v.indices.length, 2);
});

test('parser: slice with :', () => {
  const { ast } = parseSrc('x = A[:, 3]');
  const v = ast.body[0].value;
  assert.equal(v.type, 'IndexExpr');
  assert.equal(v.indices[0].type, 'SliceAll');
});

test('parser: field access', () => {
  const { ast } = parseSrc('x = r.field');
  const v = ast.body[0].value;
  assert.equal(v.type, 'FieldAccess');
  assert.equal(v.field, 'field');
});

test('parser: chained postfix (module.func(args))', () => {
  const { ast, diagnostics } = parseSrc('x = m.func(y)');
  assert.equal(diagnostics.length, 0);
  const v = ast.body[0].value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.type, 'FieldAccess');
});

test('parser: array literal', () => {
  const { ast } = parseSrc('x = [1, 2, 3]');
  const v = ast.body[0].value;
  assert.equal(v.type, 'ArrayLiteral');
  assert.equal(v.elements.length, 3);
});

test('parser: empty array literal', () => {
  const { ast, diagnostics } = parseSrc('x = []');
  assert.equal(diagnostics.length, 0);
  assert.equal(ast.body[0].value.type, 'ArrayLiteral');
  assert.equal(ast.body[0].value.elements.length, 0);
});

test('parser: array literal with trailing comma', () => {
  const { ast, diagnostics } = parseSrc('x = [1, 2,]');
  assert.equal(diagnostics.length, 0);
  assert.equal(ast.body[0].value.elements.length, 2);
});

test('parser: tuple literal', () => {
  const { ast, diagnostics } = parseSrc('x = (a, b, c)');
  assert.equal(diagnostics.length, 0);
  const v = ast.body[0].value;
  assert.equal(v.type, 'TupleLiteral');
  assert.equal(v.elements.length, 3);
});

test('parser: parenthesised expression (single value)', () => {
  const { ast } = parseSrc('x = (a)');
  // Should NOT be a TupleLiteral — just the inner expression
  assert.equal(ast.body[0].value.type, 'Identifier');
});

test('parser: single-element tuple (a,) is an error', () => {
  const { diagnostics } = parseSrc('x = (a,)');
  assert.ok(diagnostics.some(d =>
    d.severity === 'error' && /at least two/.test(d.message)));
});

test('parser: comparison (single)', () => {
  const { ast } = parseSrc('x = a < b');
  const v = ast.body[0].value;
  assert.equal(v.type, 'BinaryExpr');
  assert.equal(v.op, '<');
});

test('parser: chained comparison rejected', () => {
  // grammar disallows a < b < c
  const { diagnostics } = parseSrc('x = a < b < c');
  assert.ok(diagnostics.length > 0);
});

test('parser: hole and placeholder in expression', () => {
  const { ast } = parseSrc('f = fn(_ * _)');
  const v = ast.body[0].value;
  assert.equal(v.type, 'CallExpr');
  const inner = v.args[0]; // _ * _
  assert.equal(inner.left.type, 'Hole');
  assert.equal(inner.right.type, 'Hole');
});

test('parser: location info on binding', () => {
  const { ast } = parseSrc('x = 42');
  const stmt = ast.body[0];
  assert.equal(stmt.loc.start.line, 0);
  assert.equal(stmt.loc.start.col, 0);
  assert.equal(stmt.names[0].loc.start.col, 0);
  assert.equal(stmt.names[0].loc.end.col, 1);
});

test('parser: error recovery skips to next line', () => {
  // First line has an unparseable RHS; second line should still parse.
  const { ast, diagnostics } = parseSrc('x = ?\ny = 1');
  assert.ok(diagnostics.length > 0);
  // The y=1 statement should still appear
  assert.ok(ast.body.some(s => s.type === 'AssignStatement'
    && s.names[0].name === 'y'));
});
