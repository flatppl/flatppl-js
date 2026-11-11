'use strict';

// =====================================================================
// Spec §05 Statements + Comments — end-to-end via processSource.
// =====================================================================
//
// Lexical-only changes (no AST/IR surface change), but the round-trip
// through parse → analyse should produce the same module shape that
// newline-separated source produces.
//
// Spec commits:
//   - flatppl-design 7cf8705 (semicolon as statement separator)
//   - flatppl-design d543b79 (block-comment `###` form)

const { test } = require('node:test');
const assert = require('node:assert/strict');
const engine = require('../index.ts');

function errors(r: any) {
  return r.diagnostics.filter((d: any) => d.severity === 'error');
}

// =====================================================================
// Semicolon as statement separator
// =====================================================================

test('§05: `;` separates statements at top level', () => {
  const r = engine.processSource(`a = 1; b = 2; c = a + b`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('a'));
  assert.ok(r.bindings.has('b'));
  assert.ok(r.bindings.has('c'));
});

test('§05: `;` and newline produce the same module', () => {
  const a = engine.processSource(`a = 1; b = 2`);
  const b = engine.processSource(`a = 1\nb = 2`);
  assert.deepEqual(errors(a), []);
  assert.deepEqual(errors(b), []);
  // Same bindings, same RHS shapes.
  for (const n of ['a', 'b']) {
    assert.ok(a.bindings.has(n) && b.bindings.has(n));
  }
});

test('§05: trailing `;` is harmless', () => {
  const r = engine.processSource(`a = 1;`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('a'));
});

test('§05: multiple consecutive separators collapse', () => {
  const r = engine.processSource(`a = 1;;\n;b = 2`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('a') && r.bindings.has('b'));
});

test('§05: `;` inside parens is FlatPPJ kwargs (not a statement separator)', () => {
  // The kwargs-split `;` only applies under the FlatPPJ variant; in
  // FlatPPL it produces a diagnostic that we don't gate on here. The
  // important thing is the `;` does NOT silently break parsing into
  // two top-level statements when it's inside a call.
  const r = engine.processSource(`a = 1
b = 2
f(a, b) = a + b
x = f(a, b)`);
  assert.deepEqual(errors(r), []);
});

// =====================================================================
// `###` block comments
// =====================================================================

test('§05: `###` block between statements is transparent', () => {
  const r = engine.processSource(`a = 1
###
multi-line
block comment
###
b = 2
`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('a') && r.bindings.has('b'));
});

test('§05: `###` block with leading whitespace on fences', () => {
  const r = engine.processSource(`a = 1
    ###
    indented body
    ###
b = 2`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('a') && r.bindings.has('b'));
});

test('§05: `###` in mid-line is just a line comment, ending at the newline', () => {
  const r = engine.processSource(`a = 1 ### still a line comment\nb = 2`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('a') && r.bindings.has('b'));
});

// =====================================================================
// Combined: a `;` inside a comment is comment text
// =====================================================================

test('§05: a `;` inside a `#` line comment does not end it', () => {
  const r = engine.processSource(`a = 1.0  # one; b = 2.0`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('a'));
  assert.ok(!r.bindings.has('b'), 'b = 2.0 is comment text');
});

test('§05: a comment-only line with a `;` parses', () => {
  const r = engine.processSource(`# one; two\nx = 1`);
  assert.deepEqual(errors(r), []);
  assert.deepEqual([...r.bindings.keys()].filter((n: string) => !n.startsWith('__')), ['x']);
});

test('§05: a `;` inside a `###` block comment is comment text', () => {
  const r = engine.processSource(`###\nc = 1; d = 2\n###\nx = 1`);
  assert.deepEqual(errors(r), []);
  assert.ok(r.bindings.has('x'));
  assert.ok(!r.bindings.has('c') && !r.bindings.has('d'));
});

test('§05: comment-only file with `###` block parses cleanly', () => {
  const r = engine.processSource(`###
just a block
###`);
  assert.deepEqual(errors(r), []);
  assert.equal(r.bindings.size, 0);
});
