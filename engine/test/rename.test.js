'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const {
  processSource, planRename, isValidBindingName, isValidPlaceholderText,
} = require('../index');

function plan(src, line, col) {
  const { ast, bindings } = processSource(src);
  return planRename(ast, bindings, line, col);
}

// --- Validation helpers ---

test('isValidBindingName accepts public names', () => {
  assert.ok(isValidBindingName('x'));
  assert.ok(isValidBindingName('theta1'));
  assert.ok(isValidBindingName('forward_kernel'));
});

test('isValidBindingName accepts private names', () => {
  assert.ok(isValidBindingName('_tmp'));
  assert.ok(isValidBindingName('_private_helper'));
});

test('isValidBindingName accepts auto-generated names', () => {
  assert.ok(isValidBindingName('__internal'));
  assert.ok(isValidBindingName('__0xabc'));
});

test('isValidBindingName rejects bare _', () => {
  assert.equal(isValidBindingName('_'), false);
});

test('isValidBindingName rejects placeholder pattern', () => {
  assert.equal(isValidBindingName('_par_'), false);
  assert.equal(isValidBindingName('_x_'), false);
});

test('isValidBindingName rejects reserved names', () => {
  assert.equal(isValidBindingName('self'), false);
  assert.equal(isValidBindingName('base'), false);
});

test('isValidBindingName rejects non-identifiers', () => {
  assert.equal(isValidBindingName(''), false);
  assert.equal(isValidBindingName('1abc'), false);
  assert.equal(isValidBindingName('a.b'), false);
  assert.equal(isValidBindingName(' x'), false);
});

test('isValidPlaceholderText accepts _name_ patterns', () => {
  assert.ok(isValidPlaceholderText('_par_'));
  assert.ok(isValidPlaceholderText('_x1_'));
  assert.ok(isValidPlaceholderText('_alpha_beta_'));
});

test('isValidPlaceholderText rejects non-placeholder text', () => {
  assert.equal(isValidPlaceholderText('par'), false);
  assert.equal(isValidPlaceholderText('_par'), false);
  assert.equal(isValidPlaceholderText('par_'), false);
  assert.equal(isValidPlaceholderText('_'), false);
});

// --- planRename: binding renames ---

test('planRename returns null for empty file', () => {
  assert.equal(plan('', 0, 0), null);
});

test('planRename returns null for cursor in whitespace', () => {
  assert.equal(plan('x = 1\n', 0, 4), null);
});

test('planRename for cursor on LHS binding', () => {
  // 'x = 1' — cursor at col 0 on 'x'
  const result = plan('x = 1\n', 0, 0);
  assert.ok(result);
  assert.equal(result.kind, 'binding');
  assert.equal(result.oldName, 'x');
  // Single binding, single occurrence on LHS
  assert.equal(result.locs.length, 1);
});

test('planRename for cursor on identifier reference in RHS', () => {
  const src = 'a = 1\nb = a + 2\n';
  // 'b = a + 2' — 'a' is at col 4 of line 1
  const result = plan(src, 1, 4);
  assert.ok(result);
  assert.equal(result.kind, 'binding');
  assert.equal(result.oldName, 'a');
  // Two locs: LHS in line 0, reference in line 1
  assert.equal(result.locs.length, 2);
});

test('planRename: bindings with multiple references are collected', () => {
  const src = 'a = 1\nb = a + a * a\nc = a\n';
  const result = plan(src, 0, 0); // cursor on 'a' definition
  assert.equal(result.locs.length, 5); // 1 def + 4 refs
});

test('planRename: cursor on bare _ (LHS) returns null', () => {
  const src = '_, x = (1, 2)\n';
  const result = plan(src, 0, 0); // cursor on '_'
  assert.equal(result, null);
});

test('planRename: only the matching name in a decomposition', () => {
  const src = 'forward_kernel, prior = (1, 2)\n';
  // Cursor on 'prior' (col 17)
  const result = plan(src, 0, 17);
  assert.ok(result);
  assert.equal(result.kind, 'binding');
  assert.equal(result.oldName, 'prior');
  // Only the 'prior' name location, not 'forward_kernel'
  assert.equal(result.locs.length, 1);
});

test('planRename: keyword arg names are NOT treated as references', () => {
  // 'mu = a' — the keyword 'mu' isn't a reference. Renaming 'a' should
  // only touch the identifier 'a', not the kwarg name.
  const src = 'a = 1\nx = Normal(mu = a, sigma = 1)\n';
  const result = plan(src, 0, 0); // on 'a' def
  // Two locs: 'a' on LHS and the 'a' inside Normal call
  assert.equal(result.locs.length, 2);
});

test('planRename: field access dot-suffix is NOT renamed', () => {
  // r.x — the 'x' here is a field name, not an identifier reference.
  const src = 'r = 1\ny = r.x\n';
  const result = plan(src, 0, 0); // on 'r' def
  assert.equal(result.locs.length, 2); // r def + r ref in 'r.x'
});

// --- planRename: placeholder renames ---

test('planRename for cursor on a placeholder', () => {
  // f = functionof(_par_ * 2, par = _par_)
  // Find col of first '_par_' (in body)
  const src = 'f = functionof(_par_ * 2, par = _par_)\n';
  // _par_ is at col 15-19 (indices)
  const result = plan(src, 0, 16);
  assert.ok(result);
  assert.equal(result.kind, 'placeholder');
  assert.equal(result.oldName, 'par');
  // Two locs: in body and as kwarg value
  assert.equal(result.locs.length, 2);
});

test('planRename for placeholder in lawof', () => {
  const src = 'm = lawof(_x_, x = _x_)\n';
  // First _x_ is at col 10-12
  const result = plan(src, 0, 11);
  assert.ok(result);
  assert.equal(result.kind, 'placeholder');
  assert.equal(result.oldName, 'x');
  assert.equal(result.locs.length, 2);
});

test('planRename: nested placeholders in different scopes are independent', () => {
  // Outer functionof has _par_; inner functionof has _par_ — different scope.
  // Renaming the outer _par_ should NOT touch the inner ones.
  const src = `outer = functionof(_par_ + functionof(_par_ * 2, par = _par_)(b), par = _par_)\n`;
  // First _par_ (outer) is right after 'functionof(' — col 19-23
  const result = plan(src, 0, 20);
  assert.ok(result);
  assert.equal(result.kind, 'placeholder');
  assert.equal(result.oldName, 'par');
  // Should find exactly the outer scope's _par_ occurrences:
  // - in body: `_par_ + functionof(...)(b)` — first one
  // - as kwarg value at end: `, par = _par_`
  // Inner scope's two _par_ should NOT be included.
  assert.equal(result.locs.length, 2);
});

test('planRename: cursor on inner placeholder in nested scope finds inner only', () => {
  const src = `outer = functionof(_par_ + functionof(_par_ * 2, par = _par_)(b), par = _par_)\n`;
  // Inner _par_ (in functionof(_par_ * 2, ...)) is around col 38
  const result = plan(src, 0, 39);
  assert.ok(result);
  assert.equal(result.kind, 'placeholder');
  // 2 inner _par_ occurrences (body + kwarg value)
  assert.equal(result.locs.length, 2);
});
