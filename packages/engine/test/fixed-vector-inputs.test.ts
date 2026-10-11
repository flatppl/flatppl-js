'use strict';

// Fixed-data validation is separate from §03 coordinate-measure membership.
// The equality bound allows source-precision roundoff without changing data.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, value: valueLib } = require('../index.ts');

function fixedInputs(source: string) {
  const parsed = processSource(source);
  assert.deepEqual(parsed.diagnostics, []);
  return orchestrator.buildDerivations(parsed.linkedBindings).fixedValues;
}

test('nested module substitutions validate zero-sum data with the supplied size', () => {
  const sources = {
    'outer.flatppl': 'n=external(posintegers)\nx=external(cartpow(reals,n))\n'
      + 'm=load_module("inner.flatppl",n=n,x=x)\ny=m.x',
    'inner.flatppl': 'n=external(posintegers)\nS=zerosum(n)\nx=external(S)',
  };
  const load = (x: string, n = 3) => processSource(
    `m=load_module("outer.flatppl",n=${n},x=${x})\ny=m.y`, { bundle: { sources } });
  const valid = load('[0.1,0.2,-0.3]');
  assert.deepEqual(valid.diagnostics, []);
  const values = orchestrator.buildDerivations(valid.linkedBindings).fixedValues;
  assert.deepEqual(Array.from(values.get('y').data), [0.1, 0.2, -0.3]);
  assert.ok(load('[0.1,0.2,-0.2]').diagnostics.some((d: any) => d.severity === 'error'));
  assert.ok(load('[0.1,0.2,-0.3]', 4).diagnostics.some((d: any) => d.severity === 'error'));
});

test('fixed simplex and ordered inputs keep exact inequalities', () => {
  const values = fixedInputs('p=external(stdsimplex(3))\no=external(ordered(3))');
  const p = valueLib.asValue(new Float32Array([0.1, 0.2, 0.7]));
  values.set('p', p);
  assert.equal(values.get('p'), p);
  assert.deepEqual(Array.from(p.data), Array.from(new Float32Array([0.1, 0.2, 0.7])));
  assert.throws(() => values.set('p', [-Number.EPSILON, 0.5, 0.5]));
  assert.throws(() => values.set('p', [0.1, 0.2, 0.6]));
  const ordered = [0, Number.MIN_VALUE, 1];
  values.set('o', ordered);
  assert.equal(values.get('o'), ordered);
  assert.throws(() => values.set('o', [0, 0, 1]));
});

test('fixed inputs use source f32 precision and preserve their values', () => {
  const values = fixedInputs('x=external(zerosum(3))\ns=sum(x)');
  const x = new Float32Array([0.1, 0.2, -0.3]);
  const original = Array.from(x);
  // Exact sum of these three binary32 values: -2^-27. It exceeds the
  // binary64 bound but fits 8*3*(2^-24)*sum(abs(x)).
  values.set('x', x);
  assert.equal(values.get('x'), x);
  assert.equal(values.get('s'), -(2 ** -27));
  assert.throws(() => values.set('x', Float64Array.from(x)));
  // 2^-18 exceeds 8*3*(2^-24)*(2 + 2^-18), about 2.86e-6.
  assert.throws(() => values.set('x', new Float32Array([1, -1, 2 ** -18])));
  assert.equal(values.get('x'), x);
  assert.deepEqual(Array.from(x), original);
});

test('fixed sum validation handles zero and finite values near overflow', () => {
  const values = fixedInputs('x=external(zerosum(4))');
  const large = [1e308, 1e308, -1e308, -1e308];
  values.set('x', large);
  assert.equal(values.get('x'), large);
  assert.throws(() => values.set('x', [1e308, 1e308, 0, 0]));
  assert.throws(() => values.set('x', [Infinity, 0, 0, -Infinity]));
  values.set('x', [0, 0, 0, 0]);
  assert.deepEqual(values.get('x'), [0, 0, 0, 0]);
});

test('forwarded fixed input constraints survive until a host value is supplied', () => {
  const parsed = processSource('x=external(cartpow(reals,3))\n'
    + 'm=load_module("inner.flatppl",x=x)\ny=sum(m.x)', {
    bundle: { sources: { 'inner.flatppl': 'x=external(zerosum(3))' } },
  });
  assert.deepEqual(parsed.diagnostics, []);
  const values = orchestrator.buildDerivations(parsed.linkedBindings).fixedValues;
  values.set('x', [0.1, 0.2, -0.2]);
  assert.throws(() => values.get('y'));
  values.set('x', [0.1, 0.2, -0.3]);
  assert.equal(values.get('y'), 0.1 + 0.2 - 0.3);
});
