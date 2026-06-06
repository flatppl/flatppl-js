'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const C = require('../sampler-eval-compile.ts');

const ref = (name) => ({ kind: 'ref', ns: 'self', name });
const lit = (v) => ({ kind: 'lit', value: v });
const call = (op, ...args) => ({ kind: 'call', op, args });

test('_hasPerAtomRef: true when a per-atom name appears', () => {
  const ir = call('add', ref('x'), call('mul', ref('pars'), lit(2)));
  assert.equal(C._hasPerAtomRef(ir, new Set(['x'])), true);
  assert.equal(C._hasPerAtomRef(ir, new Set(['nope'])), false);
});

test('_hasPerAtomRef: nested under a non-compile op still detected', () => {
  const ir = call('get_field', ref('x'), lit('a'));
  assert.equal(C._hasPerAtomRef(ir, new Set(['x'])), true);
});

test('_COMPILE_ARITY: has the scalar op set, excludes structural ops', () => {
  assert.equal(C._COMPILE_ARITY.add, 2);
  assert.equal(C._COMPILE_ARITY.exp, 1);
  assert.equal(C._COMPILE_ARITY.ifelse, 3);
  assert.equal(C._COMPILE_ARITY.tuple, undefined);
  assert.equal(C._COMPILE_ARITY.get_field, undefined);
  assert.equal(C._COMPILE_ARITY.aggregate, undefined);
});
