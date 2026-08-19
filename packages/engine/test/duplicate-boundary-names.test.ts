'use strict';

// Spec §04 "Specifying reification boundaries":
//
//   "Boundary input specification is all-or-none: either every reified input
//    is specified explicitly, or none is. Boundary input names must be
//    distinct — a repeated name is a static error, which likewise forbids a
//    lambda or named function from repeating an argument name."
//
// The parser desugars `f(a, b) = e` and `(a, b) -> e` to
// `functionof(e', a = _a_, b = _b_)`, so every surface spelling reaches the
// analyzer as a boundary kwarg list and one check covers them all. The
// hand-written FlatPIR `%specinputs` list is a separate ingest path with its
// own check in pir-sexpr.ts.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { fromSexpr } = require('../pir-sexpr.ts');

function dupErrors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error'
      && /Duplicate (argument|boundary input) name/.test(d.message));
}

function errors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

// --- The repeated-name spellings ------------------------------------------

const REPEATED: Array<[string, string, string]> = [
  // [label, source, expected message fragment]
  ['function-definition sugar',
    'f(a, a) = a * 2.0\n',
    "Duplicate argument name 'a'"],
  ['multi-arg lambda',
    'g = (a, a) -> a * 2.0\n',
    "Duplicate argument name 'a'"],
  ['explicit functionof boundary',
    'x ~ Normal(mu = 0.0, sigma = 1.0)\nz = 2.0 * x\n'
      + 'k = functionof(z, p = x, p = x)\n',
    "Duplicate boundary input name 'p'"],
  ['explicit kernelof boundary',
    'x ~ Normal(mu = 0.0, sigma = 1.0)\nK = kernelof(x, p = x, p = x)\n',
    "Duplicate boundary input name 'p'"],
  // A reification nested in an argument, not at the top of the RHS — the
  // reason the check walks the whole expression.
  ['functionof nested in pushfwd',
    'x ~ Normal(mu = 0.0, sigma = 1.0)\n'
      + 'M = pushfwd(functionof(2.0 * x, p = x, p = x), lawof(x))\n',
    "Duplicate boundary input name 'p'"],
];

for (const [label, src, fragment] of REPEATED) {
  test(`${label}: a repeated boundary name is a static error`, () => {
    const errs = dupErrors(src);
    assert.equal(errs.length, 1,
      `expected one duplicate-name error, got ${errs.length}:\n`
        + JSON.stringify(errs, null, 2));
    assert.ok(errs[0].message.includes(fragment),
      `message must name the repeat; got: ${errs[0].message}`);
    assert.ok(errs[0].message.includes('spec §04'),
      `message must cite the rule; got: ${errs[0].message}`);
    assert.ok(errs[0].loc && errs[0].loc.start,
      'the diagnostic must be located');
  });
}

// A name repeated three times reports each repeat, so the user sees every
// offending token rather than only the first.
test('a name repeated three times reports both repeats', () => {
  assert.equal(dupErrors('f(a, a, a) = a * 2.0\n').length, 2);
});

// --- Distinct-name controls ----------------------------------------------

const DISTINCT: Array<[string, string]> = [
  ['function-definition sugar', 'f(a, b) = a * b\n'],
  ['multi-arg lambda', 'g = (a, b) -> a * b\n'],
  ['single-arg lambda', 'g = a -> a * 2.0\n'],
  ['explicit functionof boundary',
    'x ~ Normal(mu = 0.0, sigma = 1.0)\nz = 2.0 * x\n'
      + 'k = functionof(z, p = x)\n'],
  ['explicit kernelof boundary',
    'x ~ Normal(mu = 0.0, sigma = 1.0)\nK = kernelof(x, p = x)\n'],
];

for (const [label, src] of DISTINCT) {
  test(`${label}: distinct names analyse clean`, () => {
    assert.deepEqual(errors(src), []);
  });
}

// A boundary name may shadow the node it binds (spec §04: "The function
// argument names do not have to differ from the boundary node names"), and
// the same name under two SEPARATE reifications is not a repeat.
test('a boundary name may reuse a node name, and repeat across reifications', () => {
  assert.deepEqual(errors(
    'x ~ Normal(mu = 0.0, sigma = 1.0)\ny ~ Normal(mu = 0.0, sigma = 1.0)\n'
    + 'h = functionof(2.0 * x, x = x)\n'
    + 'j = functionof(3.0 * y, x = y)\n'), []);
});

// --- Hand-written FlatPIR ------------------------------------------------

test('FlatPIR %specinputs refuses a repeated boundary name', () => {
  const src = '(%module (%bind x (external (Real)))'
    + ' (%bind f (functionof (%ref self x) %specinputs'
    + ' ((p (%ref self x)) (p (%ref self x)))))'
    + ' (%public x f))';
  const errs = (fromSexpr(src).diagnostics || []).filter(
    (d: any) => /duplicate boundary input name/.test(d.message || ''));
  assert.equal(errs.length, 1,
    `expected one duplicate-name error, got: ${JSON.stringify(errs)}`);
  assert.ok(errs[0].message.includes('spec §04'),
    `message must cite the rule; got: ${errs[0].message}`);
});

test('FlatPIR %specinputs accepts distinct boundary names', () => {
  const src = '(%module (%bind x (external (Real))) (%bind y (external (Real)))'
    + ' (%bind f (functionof (%ref self x) %specinputs'
    + ' ((p (%ref self x)) (q (%ref self y)))))'
    + ' (%public x y f))';
  const errs = (fromSexpr(src).diagnostics || []).filter(
    (d: any) => /duplicate boundary input name/.test(d.message || ''));
  assert.deepEqual(errs, []);
});
