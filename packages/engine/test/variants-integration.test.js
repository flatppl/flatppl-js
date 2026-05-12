'use strict';

// End-to-end variant integration: the same Bayesian model written
// in each of FlatPPL, FlatPPY, and FlatPPJ should produce
// equivalent binding maps (same names, structurally-equivalent
// RHS IR after lowering). Per-feature parsing tests live in the
// individual feature test files; this file is the cross-feature
// smoke test that proves the variant story holds end-to-end.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('../index');

// A small model that exercises:
//   - tilde bindings vs explicit draw(...)
//   - logical operators (&& / and)
//   - boolean literals (true / True)
//   - chained comparisons
//   - record / cartprod preset shapes
//   - FlatPPJ semicolon kwargs in one place
// Each variant version below is written in its idiomatic form;
// all three should lower to the same binding shape.

const FLATPPL_SRC = `
mu = elementof(reals)
sigma = elementof(posreals)
x ~ Normal(mu = mu, sigma = sigma)
y = x ^ 2
positive = x > 0 && y > 0
nominal = record(mu = 0.0, sigma = fixed(1.0))
domain  = cartprod(mu = interval(-3, 3), sigma = interval(0, 5))
flag = true
`;

const FLATPPY_SRC = `
mu = elementof(reals)
sigma = elementof(posreals)
x = draw(Normal(mu = mu, sigma = sigma))
y = pow(x, 2)
positive = x > 0 and y > 0
nominal = record(mu = 0.0, sigma = fixed(1.0))
domain  = cartprod(mu = interval(-3, 3), sigma = interval(0, 5))
flag = True
`;

const FLATPPJ_SRC = `
mu = elementof(reals)
sigma = elementof(posreals)
x ~ Normal(;mu = mu, sigma = sigma)
y = x ^ 2
positive = x > 0 && y > 0
nominal = record(mu = 0.0, sigma = fixed(1.0))
domain  = cartprod(mu = interval(-3, 3), sigma = interval(0, 5))
flag = true
`;

function processNoErrors(src, opts) {
  const r = processSource(src, opts);
  const errors = r.diagnostics.filter(d => d.severity === 'error');
  assert.deepEqual(errors, [], `Errors in ${opts.variant}: `
    + JSON.stringify(errors));
  return r;
}

test('integration: all three variants process the model without errors', () => {
  processNoErrors(FLATPPL_SRC, { variant: 'flatppl' });
  processNoErrors(FLATPPY_SRC, { variant: 'flatppy' });
  processNoErrors(FLATPPJ_SRC, { variant: 'flatppj' });
});

test('integration: all three produce the same binding names', () => {
  const expected = ['mu', 'sigma', 'x', 'y', 'positive', 'nominal',
                    'domain', 'flag'];
  for (const variant of ['flatppl', 'flatppy', 'flatppj']) {
    const src = { flatppl: FLATPPL_SRC, flatppy: FLATPPY_SRC, flatppj: FLATPPJ_SRC }[variant];
    const r = processNoErrors(src, { variant });
    assert.deepEqual(Array.from(r.bindings.keys()), expected,
      `binding names mismatch in ${variant}`);
  }
});

test('integration: tilde and draw(...) produce structurally equal x bindings', () => {
  const rL = processNoErrors(FLATPPL_SRC, { variant: 'flatppl' });
  const rY = processNoErrors(FLATPPY_SRC, { variant: 'flatppy' });
  const rJ = processNoErrors(FLATPPJ_SRC, { variant: 'flatppj' });
  // All three x bindings end up as `draw(Normal(...))`.
  for (const r of [rL, rY, rJ]) {
    const v = r.bindings.get('x').node.value;
    assert.equal(v.type, 'CallExpr');
    assert.equal(v.callee.name, 'draw');
    assert.equal(v.args[0].callee.name, 'Normal');
  }
});

test('integration: ^ and pow(...) produce identical y bindings', () => {
  const yL = processNoErrors(FLATPPL_SRC, { variant: 'flatppl' }).bindings.get('y').node.value;
  const yY = processNoErrors(FLATPPY_SRC, { variant: 'flatppy' }).bindings.get('y').node.value;
  assert.equal(yL.callee.name, 'pow');
  assert.equal(yY.callee.name, 'pow');
  assert.equal(yL.args[0].name, yY.args[0].name);
  assert.equal(yL.args[1].value, yY.args[1].value);
});

test('integration: && and `and` produce identical land calls', () => {
  const pL = processNoErrors(FLATPPL_SRC, { variant: 'flatppl' }).bindings.get('positive').node.value;
  const pY = processNoErrors(FLATPPY_SRC, { variant: 'flatppy' }).bindings.get('positive').node.value;
  assert.equal(pL.callee.name, 'land');
  assert.equal(pY.callee.name, 'land');
  // Both args are comparisons (x > 0, y > 0)
  assert.equal(pL.args[0].op, '>');
  assert.equal(pY.args[0].op, '>');
});

test('integration: true and True produce identical BoolLiterals', () => {
  const fL = processNoErrors(FLATPPL_SRC, { variant: 'flatppl' }).bindings.get('flag').node.value;
  const fY = processNoErrors(FLATPPY_SRC, { variant: 'flatppy' }).bindings.get('flag').node.value;
  assert.equal(fL.type, 'BoolLiteral');
  assert.equal(fY.type, 'BoolLiteral');
  assert.equal(fL.value, true);
  assert.equal(fY.value, true);
});

test('integration: path-detected variants give the same result as explicit ones', () => {
  const fromPath = processNoErrors(FLATPPL_SRC, { path: 'foo.flatppl' });
  const explicit = processNoErrors(FLATPPL_SRC, { variant: 'flatppl' });
  assert.deepEqual(
    Array.from(fromPath.bindings.keys()),
    Array.from(explicit.bindings.keys()));
});

test('integration: cross-variant rejection — FlatPPL src under FlatPPY fails', () => {
  // FLATPPL_SRC uses `~`, `^`, `&&`, `true`. Under FlatPPY each is
  // a parse error.
  const r = processSource(FLATPPL_SRC, { variant: 'flatppy' });
  const errors = r.diagnostics.filter(d => d.severity === 'error');
  assert.ok(errors.length >= 1,
    'expected FlatPPL source to fail under FlatPPY parsing');
});

test('integration: cross-variant rejection — FlatPPY src under FlatPPL fails', () => {
  // FLATPPY_SRC uses `True`, `and`. Under FlatPPL `True` is reserved
  // and `and` is reserved + not a logical operator.
  const r = processSource(FLATPPY_SRC, { variant: 'flatppl' });
  const errors = r.diagnostics.filter(d => d.severity === 'error');
  assert.ok(errors.length >= 1,
    'expected FlatPPY source to fail under FlatPPL parsing');
});
