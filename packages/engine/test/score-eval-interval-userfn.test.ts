'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const sampler = require('../sampler.ts');

test('evaluateExpr: interval(a,b) evaluates to a finite value-set', () => {
  const ir = { kind: 'call', op: 'interval',
    args: [ { kind: 'lit', value: 0.1 }, { kind: 'lit', value: 20 } ] };
  assert.deepEqual(sampler.evaluateExpr(ir, {}), { kind: 'interval', a: 0.1, b: 20 });
});

test('evaluateExpr: interval feeds Uniform support (density reads {a,b})', () => {
  // The evaluated value-set has .a/.b, which density-prims Uniform consumes.
  const v = sampler.evaluateExpr(
    { kind: 'call', op: 'interval', args: [ { kind: 'lit', value: 2 }, { kind: 'lit', value: 5 } ] }, {});
  assert.equal(v.a, 2); assert.equal(v.b, 5);
});

test('evaluateExpr: interval with non-finite / degenerate bounds throws (not the UNSUPPORTED phrase)', () => {
  for (const [a, b] of [[5, 5], [7, 2], [0, Infinity]]) {
    assert.throws(
      () => sampler.evaluateExpr({ kind: 'call', op: 'interval',
        args: [ { kind: 'lit', value: a }, { kind: 'lit', value: b } ] }, {}),
      (e: any) => /interval/.test(e.message) && !/not evaluable in sampler context/.test(e.message));
  }
});
