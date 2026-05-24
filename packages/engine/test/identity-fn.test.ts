'use strict';

// Tests for spec §07 `identity(x)` — the trivial first-class T → T
// function. Useful as a placeholder argument (e.g. pushfwd(identity, M))
// and during interactive editing.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');

function errs(src: any) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

test('identity: scalar real binding parses + analyzes clean', () => {
  assert.equal(errs(`
x = 3.14
y = identity(x)
`).length, 0);
});

test('identity: array binding parses + analyzes clean', () => {
  assert.equal(errs(`
arr = [1, 2, 3]
arr2 = identity(arr)
`).length, 0);
});

test('identity: polymorphic — accepts measure-typed input too', () => {
  // identity is value-typed in spec §07 but the type system uses tvar('T')
  // so the analyzer wouldn't object to a measure either. The strict
  // value-vs-measure split is enforced by classifier rules where it
  // matters (e.g. inputs to draw / lawof). Smoke check: no diagnostic.
  assert.equal(errs(`
m = Normal(mu = 0, sigma = 1)
y = identity(m)
`).length, 0);
});
