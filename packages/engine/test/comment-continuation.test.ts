'use strict';

// =====================================================================
// Spec §05 Comments + Statement separation — comments are transparent
// inside a continued expression
// =====================================================================
//
// §05 "Comments": "`#` starts a line comment; `###` alone on a line
// opens a block comment closed by a matching `###` alone on a line.
// Both forms are discarded by the parser."
//
// §05 "Statement separation": "Newlines inside an unclosed `(` or `[`
// (paren/bracket depth > 0) are treated as whitespace (implicit line
// continuation)".
//
// Together: a comment may sit anywhere a newline may sit, including
// mid-expression at bracket depth > 0.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { tokenize } = require('../tokenizer.ts');
const { parse } = require('../parser.ts');

function parseSrc(src: any) {
  return parse(tokenize(src).tokens);
}
function errors(r: any) {
  return r.diagnostics.filter((d: any) => d.severity === 'error');
}

test('comment: line comment inside parens is whitespace', () => {
  const r = parseSrc('mu = (1.0 +  # partial sum\n  2.0)\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('comment: trailing comment inside a bracket list', () => {
  const r = parseSrc('v = [1.0,  # first\n  2.0,  # second\n  3.0]\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('comment: comment inside call arguments', () => {
  const r = parseSrc('x = exp(  # why not\n  1.0\n)\n');
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('comment: `###` block comment inside a continued expression', () => {
  const r = parseSrc(`mu = (1.0 +
###
A block comment at depth 1.
###
  2.0)
`);
  assert.deepEqual(errors(r), []);
  assert.equal(r.ast.body.length, 1);
});

test('comment: block fence inside brackets is not a line comment', () => {
  // A degraded fence would swallow only its own line, leaving the body
  // text and the closing fence as garbage tokens.
  const { tokens } = tokenize(`v = [1.0,
  ###
  1.0 + junk +
  ###
  2.0]
`);
  const kinds = tokens.map((t: any) => t.type).join(' ');
  assert.equal(kinds.includes('COMMENT'), false);
  const r = parse(tokens);
  assert.deepEqual(errors(r), []);
});

test('comment: `### Section ###` is a line comment, not an opening fence', () => {
  // §05 EBNF: an opening fence is `HWS* "###" HWS* Newline`. A
  // decorative banner must not swallow the rest of the file.
  const outer = parseSrc('### Section ###\na = 1\nb = 2\n');
  assert.deepEqual(errors(outer), []);
  assert.equal(outer.ast.body.length, 2);
  const inner = parseSrc('v = [1.0,\n  ### Section ###\n  2.0]\n');
  assert.deepEqual(errors(inner), []);
  assert.equal(inner.ast.body.length, 1);
});

test('comment: trailing operator still does not continue a line', () => {
  // §05 continuation is bracket-depth-driven only; a dangling operator
  // is an error with or without a comment after it.
  const bare = parseSrc('mu = 1.0 +\n  2.0\n');
  assert.ok(errors(bare).length > 0);
  const commented = parseSrc('mu = 1.0 +  # dangling\n  2.0\n');
  assert.ok(errors(commented).length > 0);
});

test('comment: statement-level comments still reach ast.comments', () => {
  const r = parseSrc('# leading\na = 1  # trailing\n');
  assert.deepEqual(errors(r), []);
  assert.deepEqual(r.ast.comments.map((c: any) => c.text),
    ['# leading', '# trailing']);
});

test('comment: doc-comment attachment survives comment filtering', () => {
  const r = parseSrc('# plain\n% Prior mean.\nmu = 0.0\n');
  assert.deepEqual(errors(r), []);
  assert.deepEqual(r.ast.body[0].doc,
    { markup: 'md', lines: ['Prior mean.'] });
});
