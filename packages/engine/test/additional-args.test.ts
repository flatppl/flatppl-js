'use strict';

// The §04 "Additional arguments" binding rule, stated once (additional-args.ts) and tested on
// plain labels: which element binds which of the callable's remaining inputs.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { bindAdditionalArguments } = require('../additional-args.ts');

const f = ['x', 'p', 'q'];   // x is supplied by the operation

test('positional arguments bind the remaining inputs in order', () => {
  assert.deepEqual(bindAdditionalArguments({ declared: f, leading: 1, positional: ['a', 'b'], keywords: {} }),
    { rest: ['a', 'b'] });
  // Without declared names the arguments pass through as given.
  assert.deepEqual(bindAdditionalArguments({ declared: null, leading: 1, positional: ['a'], keywords: {} }),
    { rest: ['a'] });
});

test('named arguments bind by input name, in the callable\'s own order', () => {
  assert.deepEqual(bindAdditionalArguments({ declared: f, leading: 1, positional: [], keywords: { q: 'B', p: 'A' } }),
    { rest: ['A', 'B'] });
});

test('named arguments need the callable\'s names, and each name binds exactly one input', () => {
  const err = (a: any) => (bindAdditionalArguments(a) as any).error;
  assert.match(err({ declared: null, leading: 1, positional: [], keywords: { p: 'A' } }), /input names are not known/);
  assert.match(err({ declared: f, leading: 1, positional: [], keywords: { p: 'A' } }), /input "q" is not bound/);
  assert.match(err({ declared: f, leading: 1, positional: [], keywords: { p: 'A', q: 'B', r: 'C' } }),
    /no remaining input named "r"/);
  // The operation's own input cannot be bound again by name.
  assert.match(err({ declared: f, leading: 1, positional: [], keywords: { x: 'A', p: 'B', q: 'C' } }),
    /no remaining input named "x"/);
});

test('positional and named arguments do not mix', () => {
  assert.match((bindAdditionalArguments({ declared: f, leading: 1, positional: ['a'], keywords: { q: 'B' } }) as any).error,
    /all positional or all named/);
});

test('a sole record or table binds whole to one remaining input and splats by name to two or more', () => {
  const fields = { p: 'P', q: 'Q' };
  // Two remaining inputs: the fields bind by name, whatever their order in the record.
  assert.deepEqual(bindAdditionalArguments({ declared: f, leading: 1, positional: ['rec'], keywords: {},
    soleFields: { q: 'Q', p: 'P' } }), { rest: ['P', 'Q'] });
  // One remaining input: the record is that input.
  assert.deepEqual(bindAdditionalArguments({ declared: ['x', 'p'], leading: 1, positional: ['rec'], keywords: {},
    soleFields: fields }), { rest: ['rec'] });
  // Splatting a record with a field the callable lacks is an error.
  assert.match((bindAdditionalArguments({ declared: f, leading: 1, positional: ['rec'], keywords: {},
    soleFields: { p: 'P', z: 'Z' } }) as any).error, /no remaining input named "z"/);
  // Two arguments are never a splat.
  assert.deepEqual(bindAdditionalArguments({ declared: f, leading: 1, positional: ['r1', 'r2'], keywords: {},
    soleFields: fields }), { rest: ['r1', 'r2'] });
});
