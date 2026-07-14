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

test('evaluateExpr: user-function call (self target) binds args + returns record', () => {
  // fn f(s, r) => record(shape = s * 2, rate = r)
  const fnBody = { kind: 'call', op: 'record', fields: [
    { name: 'shape', value: { kind: 'call', op: 'mul',
        args: [ { kind: 'ref', ns: '%local', name: 's' }, { kind: 'lit', value: 2 } ] } },
    { name: 'rate', value: { kind: 'ref', ns: '%local', name: 'r' } },
  ] };
  const env = { __resolveFnBody: (n: string) => n === 'f' ? { body: fnBody, params: ['s', 'r'] } : null };
  const callIR = { kind: 'call', target: { ns: 'self', name: 'f' },
    args: [ { kind: 'lit', value: 2.0 }, { kind: 'lit', value: 1.0 } ] };
  assert.deepEqual(sampler.evaluateExpr(callIR, env), { shape: 4.0, rate: 1.0 });
  // …and field access over the call result (the #261 shape: fn(...).shape)
  const fieldIR = { kind: 'call', op: 'get_field', args: [ callIR, { kind: 'lit', value: 'shape' } ] };
  assert.equal(sampler.evaluateExpr(fieldIR, env), 4.0);
});

test('evaluateExpr: user-function call binds kwargs by param name', () => {
  const fnBody = { kind: 'ref', ns: '%local', name: 'x' };
  const env = { __resolveFnBody: (n: string) => n === 'g' ? { body: fnBody, params: ['x'] } : null };
  const callIR = { kind: 'call', target: { ns: 'self', name: 'g' },
    args: [], kwargs: { x: { kind: 'lit', value: 42 } } };
  assert.equal(sampler.evaluateExpr(callIR, env), 42);
});

test('evaluateExpr: unresolvable self-target call still errors clearly', () => {
  const callIR = { kind: 'call', target: { ns: 'self', name: 'nope' }, args: [] };
  assert.throws(() => sampler.evaluateExpr(callIR, { __resolveFnBody: () => null }),
    /not evaluable in sampler context/);
});

test('evaluateExpr: user-fn call maps kwargs surface→internal (fn(_) sugar, keyword call)', () => {
  // fn(_) sugar: params ['_arg1_'], paramKwargs ['arg1']; body = _arg1_ * 2.
  const fnBody = { kind: 'call', op: 'mul',
    args: [ { kind: 'ref', ns: '%local', name: '_arg1_' }, { kind: 'lit', value: 2 } ] };
  const env = { __resolveFnBody: (n: string) =>
    n === 'h' ? { body: fnBody, params: ['_arg1_'], paramKwargs: ['arg1'] } : null };
  // Positional call binds correctly.
  assert.equal(sampler.evaluateExpr(
    { kind: 'call', target: { ns: 'self', name: 'h' }, args: [ { kind: 'lit', value: 5 } ] }, env), 10);
  // Keyword call by the SURFACE name (arg1) must bind the internal param (_arg1_).
  assert.equal(sampler.evaluateExpr(
    { kind: 'call', target: { ns: 'self', name: 'h' }, args: [],
      kwargs: { arg1: { kind: 'lit', value: 5 } } }, env), 10);
});
