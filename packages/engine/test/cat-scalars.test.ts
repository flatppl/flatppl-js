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

function parseOK(src: any, opts: any) {
  const r = processSource(src, opts);
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error');
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

test('cat: mixed scalar / vector args are a static type error (§07)', () => {
  // §07: "concatenation of a mix of value types … is not permitted." The
  // parser accepts the call shape (it doesn't know arg kinds), but type
  // inference now rejects the mix STATICALLY via the shared catShapeType rule —
  // no longer deferred to runtime (which used to produce `[1, null]`).
  const r = processSource('x = cat(1, [2, 3])', { variant: 'flatppl' });
  const errors = r.diagnostics.filter((d: any) => d.severity === 'error');
  assert.ok(errors.some((e: any) => /mix of value kinds|all scalars, all vectors/.test(e.message)),
    'expected a mixed-kind cat static error; got ' + JSON.stringify(errors));
});
