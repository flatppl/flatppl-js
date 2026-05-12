'use strict';

// Reserved-name enforcement at the binding LHS (spec §04) + variant-
// aware boolean-literal parsing (spec §05).
//
// Reserved at the binding LHS:
//   - FlatPPL:  and, or, not, True, False
//   - FlatPPY:  and, or, not, True, False, true, false
//   - FlatPPJ:  same as FlatPPL
//
// Boolean spelling:
//   - FlatPPL/FlatPPJ: `true` / `false`
//   - FlatPPY:         `True` / `False`

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');

function errors(src, opts) {
  return processSource(src, opts).diagnostics.filter(d => d.severity === 'error');
}

function rhs(src, opts) {
  return processSource(src, opts).bindings.get('x').node.value;
}

// ---------------------------------------------------------------------
// Reserved names
// ---------------------------------------------------------------------

test('reserved: FlatPPL rejects `and = 1`', () => {
  const errs = errors('and = 1', { variant: 'flatppl' });
  assert.ok(errs.length >= 1);
  assert.match(errs[0].message, /'and' is a reserved name in flatppl/);
});

test('reserved: FlatPPL rejects `or = 1`, `not = 1`, `True = 1`, `False = 1`', () => {
  for (const name of ['or', 'not', 'True', 'False']) {
    const errs = errors(`${name} = 1`, { variant: 'flatppl' });
    assert.ok(errs.length >= 1, `${name} should be rejected`);
    assert.match(errs[0].message, new RegExp(`'${name}' is a reserved name`));
  }
});

test('reserved: FlatPPL allows `true = 1` (lowercase true/false are bool literals, '
  + 'but they parse as BoolLiteral on the RHS only; on LHS they collide with '
  + 'the canonical boolean spelling and would shadow it — but the spec lists '
  + 'them as ordinary identifiers in FlatPPL, so binding is allowed)', () => {
  // Per spec §04 only `and`/`or`/`not`/`True`/`False` are reserved
  // in FlatPPL — `true`/`false` are not, even though they're the
  // boolean keywords. The parser still emits a BoolLiteral on the
  // RHS, but on the LHS they're plain names.
  assert.deepEqual(errors('true = 1', { variant: 'flatppl' }), []);
});

test('reserved: FlatPPY rejects `True`, `False`, `and`, `or`, `not`, `true`, `false`', () => {
  for (const name of ['True', 'False', 'and', 'or', 'not', 'true', 'false']) {
    const errs = errors(`${name} = 1`, { variant: 'flatppy' });
    assert.ok(errs.length >= 1, `${name} should be rejected in FlatPPY`);
  }
});

test('reserved: FlatPPJ behaves like FlatPPL', () => {
  for (const name of ['and', 'or', 'not', 'True', 'False']) {
    const errs = errors(`${name} = 1`, { variant: 'flatppj' });
    assert.ok(errs.length >= 1);
  }
  assert.deepEqual(errors('true = 1', { variant: 'flatppj' }), []);
});

test('reserved: decomposition LHS also rejects reserved names', () => {
  const errs = errors('a, and = (1, 2)', { variant: 'flatppl' });
  assert.ok(errs.length >= 1);
});

test('reserved: bare _ is always allowed', () => {
  // Discard pattern.
  assert.deepEqual(errors('_ = 1', { variant: 'flatppl' }), []);
  assert.deepEqual(errors('_ = 1', { variant: 'flatppy' }), []);
});

// ---------------------------------------------------------------------
// Boolean literals (spelling per variant)
// ---------------------------------------------------------------------

test('bool: FlatPPL `x = true` → BoolLiteral(true)', () => {
  const v = rhs('x = true', { variant: 'flatppl' });
  assert.equal(v.type, 'BoolLiteral');
  assert.equal(v.value, true);
});

test('bool: FlatPPL `x = false` → BoolLiteral(false)', () => {
  const v = rhs('x = false', { variant: 'flatppl' });
  assert.equal(v.type, 'BoolLiteral');
  assert.equal(v.value, false);
});

test('bool: FlatPPL `x = True` is NOT a boolean — falls to identifier', () => {
  const r = processSource('x = True', { variant: 'flatppl' });
  const v = r.bindings.get('x').node.value;
  assert.notEqual(v.type, 'BoolLiteral');
  // Analyzer flags it as undefined since `True` isn't a known name
  // in FlatPPL.
  const warns = r.diagnostics.filter(d =>
    d.severity === 'warning' && /Undefined variable 'True'/.test(d.message));
  assert.ok(warns.length >= 1);
});

test('bool: FlatPPY `x = True` → BoolLiteral(true)', () => {
  const v = rhs('x = True', { variant: 'flatppy' });
  assert.equal(v.type, 'BoolLiteral');
  assert.equal(v.value, true);
});

test('bool: FlatPPY `x = False` → BoolLiteral(false)', () => {
  const v = rhs('x = False', { variant: 'flatppy' });
  assert.equal(v.type, 'BoolLiteral');
  assert.equal(v.value, false);
});

test('bool: FlatPPY `x = true` is NOT a boolean — falls to identifier', () => {
  // FlatPPY's boolean spellings are True/False; bare `true` here
  // is an identifier ref (and reserved at the LHS).
  const r = processSource('x = true', { variant: 'flatppy' });
  const v = r.bindings.get('x').node.value;
  assert.notEqual(v.type, 'BoolLiteral');
});

test('bool: FlatPPJ `x = true` → BoolLiteral(true)', () => {
  const v = rhs('x = true', { variant: 'flatppj' });
  assert.equal(v.type, 'BoolLiteral');
  assert.equal(v.value, true);
});

// ---------------------------------------------------------------------
// Cross-variant: FlatPPL `true` and FlatPPY `True` are AST-equivalent
// ---------------------------------------------------------------------

test('bool: FlatPPL `true` and FlatPPY `True` produce identical BoolLiterals', () => {
  const vL = rhs('x = true', { variant: 'flatppl' });
  const vY = rhs('x = True', { variant: 'flatppy' });
  assert.equal(vL.type, vY.type);
  assert.equal(vL.value, vY.value);
});
