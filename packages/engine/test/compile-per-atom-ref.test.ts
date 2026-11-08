'use strict';
// The compiler folds a subtree to one constant when no per-atom ref sits under
// it. The check walked only call args, kwargs and fields, so a per-atom ref in
// a callee, a lambda body or a select branch read as atom-independent.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { _hasPerAtomRef } = require('../sampler-eval-compile.ts');

const x = { kind: 'ref', ns: 'self', name: 'x' };
const lit = (value: number) => ({ kind: 'lit', value });
const perAtom = new Set(['x']);

test('a per-atom ref is found in every IR child position', () => {
  const nodes: Record<string, any> = {
    'call args': { kind: 'call', op: 'add', args: [x, lit(1)] },
    'call kwargs': { kind: 'call', op: 'polynomial', kwargs: { x } },
    callee: { kind: 'call', callee: x, args: [lit(1)] },
    'lambda body': { kind: 'lambda', params: ['_'], body: { kind: 'call', op: 'mul', args: [{ kind: 'ref', ns: 'param', name: '_' }, x] } },
    'lambda as argument': { kind: 'call', op: 'broadcast', args: [{ kind: 'lambda', params: ['_'], body: x }, lit(1)] },
    'select branches': { kind: 'select', branches: [x, lit(0)] },
  };
  for (const [where, ir] of Object.entries(nodes)) assert.equal(_hasPerAtomRef(ir, perAtom), true, where);
});

test('an atom-independent subtree stays foldable', () => {
  const y = { kind: 'ref', ns: 'self', name: 'y' };
  assert.equal(_hasPerAtomRef({ kind: 'call', op: 'add', args: [y, lit(1)] }, perAtom), false);
  assert.equal(_hasPerAtomRef({ kind: 'lambda', params: ['_'], body: y }, perAtom), false);
});
