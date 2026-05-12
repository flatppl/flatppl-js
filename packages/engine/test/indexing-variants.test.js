'use strict';

// Per-variant indexing behavior (spec §05):
//   FlatPPL/FlatPPJ — `xs[i]` lowers to `get(xs, i)`, 1-based.
//   FlatPPY         — `xs[i]` lowers to `get0(xs, i)`, 0-based.
//
// The AST's IndexExpr carries the chosen lowering op so downstream
// passes don't need to know about variants.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');

function errors(src, opts) {
  return processSource(src, opts).diagnostics.filter(d => d.severity === 'error');
}

function indexExpr(src, opts) {
  const r = processSource(src, opts);
  // Walk to the RHS of `x = ...[...]`.
  return r.bindings.get('x').node.value;
}

// ---------------------------------------------------------------------
// AST tagging: IndexExpr's indexOp field follows the variant
// ---------------------------------------------------------------------

test('idx: FlatPPL `xs[i]` tags IndexExpr with indexOp=get', () => {
  const v = indexExpr('xs = [1, 2]\nx = xs[1]', { variant: 'flatppl' });
  assert.equal(v.type, 'IndexExpr');
  assert.equal(v.indexOp, 'get');
});

test('idx: FlatPPJ `xs[i]` tags IndexExpr with indexOp=get', () => {
  const v = indexExpr('xs = [1, 2]\nx = xs[1]', { variant: 'flatppj' });
  assert.equal(v.indexOp, 'get');
});

test('idx: FlatPPY `xs[i]` tags IndexExpr with indexOp=get0', () => {
  const v = indexExpr('xs = [1, 2]\nx = xs[0]', { variant: 'flatppy' });
  assert.equal(v.type, 'IndexExpr');
  assert.equal(v.indexOp, 'get0');
});

// ---------------------------------------------------------------------
// Index validation: 1-based vs 0-based
// ---------------------------------------------------------------------

test('idx: FlatPPL rejects `xs[0]` (1-based)', () => {
  const errs = errors('xs = [1, 2]\ny = xs[0]', { variant: 'flatppl' });
  assert.ok(errs.length >= 1);
  assert.match(errs[0].message, /1-based/);
});

test('idx: FlatPPY accepts `xs[0]` (0-based)', () => {
  assert.deepEqual(errors('xs = [1, 2]\ny = xs[0]', { variant: 'flatppy' }), []);
});

test('idx: FlatPPL accepts `xs[1]`', () => {
  assert.deepEqual(errors('xs = [1, 2]\ny = xs[1]', { variant: 'flatppl' }), []);
});

test('idx: FlatPPY accepts `xs[1]` (still inside bounds)', () => {
  // The validator only checks the negative/zero literal cases;
  // out-of-bounds for non-literals isn't a static error in either
  // variant.
  assert.deepEqual(errors('xs = [1, 2]\ny = xs[1]', { variant: 'flatppy' }), []);
});

test('idx: both variants reject `xs[-1]`', () => {
  assert.ok(errors('xs = [1, 2]\ny = xs[-1]', { variant: 'flatppl' }).length >= 1);
  assert.ok(errors('xs = [1, 2]\ny = xs[-1]', { variant: 'flatppy' }).length >= 1);
});

// ---------------------------------------------------------------------
// Runtime-expression indices: not checked statically
// ---------------------------------------------------------------------

test('idx: FlatPPY `xs[i]` with runtime i is allowed', () => {
  const errs = errors('xs = [1, 2]\ni = elementof(nonnegintegers)\ny = xs[i]',
    { variant: 'flatppy' });
  assert.deepEqual(errs, []);
});

// ---------------------------------------------------------------------
// `get0` is recognized as a builtin name
// ---------------------------------------------------------------------

test('idx: get0 is a known builtin in all variants', () => {
  const { isKnownName } = require('../builtins');
  assert.equal(isKnownName('get0'), true);
});
