'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { tokenize, T } = require('../tokenizer');

function tokenTypes(src) {
  return tokenize(src).tokens.filter(t => t.type !== T.EOF).map(t => t.type);
}
function tokenValues(src) {
  return tokenize(src).tokens.filter(t => t.type !== T.EOF).map(t => t.value);
}

test('tokenizer: empty input', () => {
  const { tokens, diagnostics } = tokenize('');
  assert.equal(tokens.length, 1);
  assert.equal(tokens[0].type, T.EOF);
  assert.equal(diagnostics.length, 0);
});

test('tokenizer: simple assignment', () => {
  assert.deepEqual(
    tokenTypes('x = 42'),
    [T.IDENT, T.EQUALS, T.NUMBER]
  );
});

test('tokenizer: integer literals', () => {
  assert.deepEqual(tokenValues('0 1 42 1_000_000'), ['0', '1', '42', '1_000_000']);
});

test('tokenizer: hex literals', () => {
  assert.deepEqual(tokenValues('0x0 0xF7 0x3e 0xFF_3a 0xABCDEF'),
    ['0x0', '0xF7', '0x3e', '0xFF_3a', '0xABCDEF']);
});

test('tokenizer: hex 0x with no digits is a diagnostic', () => {
  const { diagnostics } = tokenize('x = 0x\n');
  assert.equal(diagnostics.length, 1);
  assert.match(diagnostics[0].message, /Hex literal requires/);
});

test('tokenizer: float literals', () => {
  assert.deepEqual(tokenValues('3.14 0.5 1.0 1.45e7 1e7 .5 1.'),
    ['3.14', '0.5', '1.0', '1.45e7', '1e7', '.5', '1.']);
});

test('tokenizer: float with negative exponent', () => {
  assert.deepEqual(tokenValues('1.5e-2 1e+10'), ['1.5e-2', '1e+10']);
});

test('tokenizer: identifier vs hole vs placeholder', () => {
  const tt = tokenTypes('x _ _x_ _name_ foo_bar');
  assert.deepEqual(tt, [T.IDENT, T.HOLE, T.PLACEHOLDER, T.PLACEHOLDER, T.IDENT]);
});

test('tokenizer: placeholder yields inner name', () => {
  const tokens = tokenize('_par_').tokens;
  assert.equal(tokens[0].type, T.PLACEHOLDER);
  assert.equal(tokens[0].value, 'par');
});

test('tokenizer: operators', () => {
  assert.deepEqual(tokenTypes('+ - * / == != < > <= >='),
    [T.PLUS, T.MINUS, T.STAR, T.SLASH, T.EQEQ, T.NEQ, T.LT, T.GT, T.LTE, T.GTE]);
});

test('tokenizer: punctuation', () => {
  assert.deepEqual(tokenTypes('( ) [ ] , . :'),
    [T.LPAREN, T.RPAREN, T.LBRACKET, T.RBRACKET, T.COMMA, T.DOT, T.COLON]);
});

test('tokenizer: string literal with escapes', () => {
  const { tokens, diagnostics } = tokenize('s = "a\\nb\\tc\\\\d\\"e"');
  const stringTok = tokens.find(t => t.type === T.STRING);
  assert.equal(stringTok.value, 'a\nb\tc\\d"e');
  assert.equal(diagnostics.length, 0);
});

test('tokenizer: string with \\r and \\0 escapes', () => {
  const { tokens, diagnostics } = tokenize('s = "a\\rb\\0c"');
  const stringTok = tokens.find(t => t.type === T.STRING);
  assert.equal(stringTok.value, 'a\rb\0c');
  assert.equal(diagnostics.length, 0);
});

test('tokenizer: invalid escape sequence is a diagnostic', () => {
  const { diagnostics } = tokenize('s = "\\q"');
  assert.equal(diagnostics.length, 1);
  assert.match(diagnostics[0].message, /Invalid escape/);
});

test('tokenizer: unterminated string is a diagnostic', () => {
  const { diagnostics } = tokenize('s = "abc');
  assert.ok(diagnostics.some(d => /Unterminated/.test(d.message)));
});

test('tokenizer: comment', () => {
  const { tokens } = tokenize('# this is a comment\nx = 1');
  assert.equal(tokens[0].type, T.COMMENT);
  assert.equal(tokens[0].value, '# this is a comment');
});

test('tokenizer: implicit line continuation in parens', () => {
  // Newline inside parens is suppressed
  const tt = tokenTypes('f(\n  a,\n  b\n)');
  assert.deepEqual(tt, [T.IDENT, T.LPAREN, T.IDENT, T.COMMA, T.IDENT, T.RPAREN]);
});

test('tokenizer: implicit line continuation in brackets', () => {
  const tt = tokenTypes('a = [\n  1,\n  2\n]');
  assert.deepEqual(tt,
    [T.IDENT, T.EQUALS, T.LBRACKET, T.NUMBER, T.COMMA, T.NUMBER, T.RBRACKET]);
});

test('tokenizer: NEWLINE between statements at depth 0', () => {
  const tt = tokenTypes('a = 1\nb = 2');
  assert.deepEqual(tt,
    [T.IDENT, T.EQUALS, T.NUMBER, T.NEWLINE, T.IDENT, T.EQUALS, T.NUMBER]);
});

test('tokenizer: source locations are 0-based', () => {
  const { tokens } = tokenize('xy = 1');
  assert.equal(tokens[0].loc.start.line, 0);
  assert.equal(tokens[0].loc.start.col, 0);
  assert.equal(tokens[0].loc.end.col, 2);
});

test('tokenizer: location after newline', () => {
  const { tokens } = tokenize('a = 1\nbc = 2');
  const bcTok = tokens.find(t => t.type === T.IDENT && t.value === 'bc');
  assert.equal(bcTok.loc.start.line, 1);
  assert.equal(bcTok.loc.start.col, 0);
});
