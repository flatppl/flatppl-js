'use strict';

// Spec §07 / flatppl-design commit 12eec8c: `cat(scalar1, scalar2, ...)`
// with all-scalar arguments produces a vector. Equivalent to
// `vector(scalar1, ...)`. At the parser level there's nothing
// special to do — `cat` accepts a variadic positional arg list
// like any other builtin. The full runtime / type-inference path
// for `cat` is still TBD (tracked in TODO-flatppl-js.md under §07);
// these tests are a regression guard so a future restrictive
// typing doesn't accidentally block the scalar form.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index.ts');

function parseOK(src, opts) {
  const r = processSource(src, opts);
  const errors = r.diagnostics.filter(d => d.severity === 'error');
  assert.deepEqual(errors, [], 'expected no parse errors, got: '
    + JSON.stringify(errors));
  return r;
}

test('cat: FlatPPL `x = cat(1, 2, 3)` parses cleanly', () => {
  const r = parseOK('x = cat(1, 2, 3)', { variant: 'flatppl' });
  const v = r.bindings.get('x').node.value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'cat');
  assert.equal(v.args.length, 3);
});

test('cat: mixed scalar / vector args still parse (semantic check is downstream)', () => {
  // The spec rejects mixed-kind cats as a runtime/type error; the
  // parser doesn't know the kind of each arg. This test pins the
  // parser-level behavior: accept any call shape.
  parseOK('x = cat(1, [2, 3])', { variant: 'flatppl' });
});
