'use strict';

// =====================================================================
// Spec §05 Statement separation — a trailing operator continues a line
// =====================================================================
//
// §05 "Statement separation": "At paren/bracket depth 0, a newline is
// likewise treated as whitespace when the line's last token is a
// `ContinuationOp`, so the statement continues on the next line that
// carries a token. A trailing line comment, and any blank or
// comment-only lines in between, do not end the continuation. A `^` or
// `_` immediately following an axis name is that axis's variance
// marker, not a `ContinuationOp`."
//
// §05 EBNF `ContinuationOp`: "Every infix binary operator, the lambda
// arrow, and the binding operators. Only a TRAILING occurrence
// continues a line; a line beginning with an operator starts a new
// statement, and is a parse error unless that operator is unary."

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { T, tokenize } = require('../tokenizer.ts');
const { parse } = require('../parser.ts');
const { processSource } = require('..');

function parseSrc(src: string) {
  return parse(tokenize(src).tokens);
}
function errors(r: any): any[] {
  return r.diagnostics.filter((d: any) => d.severity === 'error');
}
function newlineCount(src: string): number {
  return tokenize(src).tokens
    .filter((t: any) => t.type === T.NEWLINE).length;
}

// ---------------------------------------------------------------------
// One case per ContinuationOp family
// ---------------------------------------------------------------------

test('continuation: AddOp joins two lines', () => {
  const r = parseSrc('mu = 1.0 +\n  2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: MulOp joins two lines', () => {
  const r = parseSrc('mu = 2.0 *\n  3.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: CompOp joins two lines', () => {
  const r = parseSrc('flag = 1.0 <\n  2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: `in` joins two lines', () => {
  // `in` is a CompOp spelled as a keyword, so it lexes as an IDENT.
  const r = parseSrc('flag = 1.0 in\n  [1.0, 2.0]\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: `^` joins two lines', () => {
  const r = parseSrc('mu = 2.0 ^\n  3.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: a dotted operator joins two lines', () => {
  const r = parseSrc('v = [1.0, 2.0] .+\n  [3.0, 4.0]\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: `&&` joins two lines', () => {
  const r = parseSrc('flag = true &&\n  false\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: `->` joins two lines', () => {
  const r = parseSrc('f = x ->\n  x * 2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: `=` joins two lines', () => {
  const r = parseSrc('mu =\n  1.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: `~` joins two lines', () => {
  const r = parseSrc('mu ~\n  Normal(0.0, 1.0)\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: `:=` joins two lines', () => {
  const r = parseSrc('x = [1.0, 2.0]\ns[.i] :=\n  x[.i] * 2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 2);
});

test('continuation: metricsum `:` joins two lines', () => {
  // The MetricsumBinding lookahead needs `IDENT COLON IDENT LBRACKET`
  // adjacent in the token stream, so the suppressed NEWLINE is what
  // makes this shape reachable across lines.
  const src = 'g = rowstack([[1.0, 0.0], [0.0, -1.0]])\n'
    + 'p = [3.0, 2.0]\n'
    + 'g:\n  norm[] := p[.mu^] * p[.mu_]\n';
  const r = parseSrc(src);
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 3);
});

// ---------------------------------------------------------------------
// What may sit between the operator and the next operand
// ---------------------------------------------------------------------

test('continuation: blank lines inside a join', () => {
  const r = parseSrc('mu = 1.0 +\n\n\n  2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: trailing line comment inside a join', () => {
  const r = parseSrc('mu = 1.0 +  # partial sum\n  2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: comment-only lines inside a join', () => {
  const r = parseSrc('mu = 1.0 +  # first\n  # second\n\n  # third\n  2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: a `###` block inside a join', () => {
  const r = parseSrc('mu = 1.0 +\n###\nstill the same statement\n###\n  2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('continuation: a joined statement emits no NEWLINE at the join', () => {
  assert.equal(newlineCount('mu = 1.0 +\n  2.0\n'), 1);
  assert.equal(newlineCount('mu = 1.0 +  # c\n\n  2.0\n'), 1);
  assert.equal(newlineCount('mu = 1.0\n  2.0\n'), 2);
});

// ---------------------------------------------------------------------
// Where the rule stops
// ---------------------------------------------------------------------

test('continuation: a leading operator does not continue the line', () => {
  // §05 EBNF ContinuationOp: only a TRAILING occurrence continues.
  const r = parseSrc('mu = 1.0\n  * 2.0\n');
  assert.ok(errors(r).length > 0);
});

test('continuation: a leading unary operator starts a new statement', () => {
  // `- 2.0` is a well-formed statement on its own, so the two lines
  // stay two statements rather than joining into `1.0 - 2.0`.
  const r = parseSrc('mu = 1.0\n- 2.0\n');
  assert.equal(r.ast.body.length, 2);
});

test('continuation: `;` still separates after a trailing operator', () => {
  // §05 names a newline as the thing a trailing operator absorbs. A
  // `;` stays a hard statement separator, so the operator is dangling.
  const r = parseSrc('mu = 1.0 +; 2.0\n');
  assert.ok(errors(r).length > 0);
});

test('continuation: an unlexable character ends the line', () => {
  // `?` emits a diagnostic and no token, so the `=` before it would
  // otherwise still be the line's last token and would swallow `y = 1`.
  const r = parseSrc('x = ?\ny = 1\n');
  assert.ok(errors(r).length > 0);
  assert.ok(r.ast.body.some((s: any) => s.type === 'AssignStatement'
    && s.names[0].name === 'y'));
});

test('continuation: axis variance `^` is not a ContinuationOp', () => {
  // Parse-level only: `.mu^` outside an aggregation is a static error,
  // which is exactly why the statement must not swallow the next line.
  assert.equal(newlineCount('x = .mu^\ny = 1.0\n'), 2);
});

test('continuation: axis variance `_` is not a ContinuationOp', () => {
  // The lower marker lexes as part of the IDENT, so it can never look
  // like an operator.
  const { tokens } = tokenize('x = .mu_\ny = 1.0\n');
  assert.equal(tokens.filter((t: any) => t.type === T.NEWLINE).length, 2);
  assert.equal(tokens.some((t: any) => t.type === T.CARET), false);
});

test('continuation: `^` after a field access is exponentiation', () => {
  // `.b` here is FieldAccess, not an axis, so the `^` is the operator
  // and the line continues.
  const r = parseSrc('y = rec.b ^\n  2.0\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

// ---------------------------------------------------------------------
// Diagnostics still point at the right physical line
// ---------------------------------------------------------------------

test('continuation: a diagnostic across a join keeps its source line', () => {
  const src = 'mu = 1.0 +\n  no_such_name\n';
  const errs = processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /no_such_name/);
  assert.equal(errs[0].loc.start.line, 1);
  assert.equal(errs[0].loc.start.col, 2);
});

test('continuation: a fixed-phase diagnostic across a join keeps its line', () => {
  // A `~` binding is random, so using it in a fixed-phase position is
  // a phase error. The location must be the operand's own line.
  const src = 'mu ~ Normal(0.0, 1.0)\n'
    + 'n =\n'
    + '  mu\n'
    + 'v = zeros(n)\n';
  const errs = processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
  assert.equal(errs.length, 1);
  assert.match(errs[0].message, /fixed-phase/);
  assert.deepEqual(errs[0].loc.start, { line: 3, col: 10 });
});
