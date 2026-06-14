'use strict';

// Tests for `checked(value, condition)` — spec §07 value-preserving
// assertion. Returns `value` unchanged when `condition` (a fixed-phase
// boolean, evaluated at load/inference time) is true, else a static
// error. Three contract halves:
//   - TYPE pass-through (result type == value type)          [typeinfer]
//   - condition is a boolean                                 [typeinfer]
//   - condition is fixed-phase                               [analyzer]
//   - runtime value-preserving guard                         [sampler]

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');

function process(src: any) {
  return processSource(src);
}
const errs = (r: any) => r.diagnostics.filter((d: any) => d.severity === 'error');

// =====================================================================
// Type pass-through (typeinfer)
// =====================================================================

test('checked: result type is identical to value (real)', () => {
  const r = process(`
v = 3.0
n = checked(value = v, condition = lt(1.0, 2.0))
`);
  const lb = r.loweredModule.bindings.get('n');
  const t = lb && lb.inferredType;
  assert.ok(t && t.kind === 'scalar' && t.prim === 'real',
    `expected scalar real, got ${JSON.stringify(t)}`);
  assert.equal(errs(r).length, 0, JSON.stringify(errs(r)));
});

test('checked: positional value + kwarg condition types through (integer)', () => {
  const r = process(`
k = 5
n = checked(k, condition = equal(2, 2))
`);
  const lb = r.loweredModule.bindings.get('n');
  const t = lb && lb.inferredType;
  assert.ok(t && t.kind === 'scalar' && t.prim === 'integer',
    `expected scalar integer, got ${JSON.stringify(t)}`);
  assert.equal(errs(r).length, 0, JSON.stringify(errs(r)));
});

test('checked: array value types through', () => {
  const r = process(`
xs = external(reals)
data = [1.0, 2.0, 3.0]
n = checked(value = data, condition = equal(lengthof(data), 3))
`);
  const lb = r.loweredModule.bindings.get('n');
  const t = lb && lb.inferredType;
  assert.ok(t && t.kind === 'array', `expected array, got ${JSON.stringify(t)}`);
});

// =====================================================================
// Condition must be a boolean (typeinfer)
// =====================================================================

test('checked: real condition is a type error', () => {
  const r = process(`
n = checked(value = 1.0, condition = 2.0)
`);
  assert.ok(errs(r).some((d: any) => /condition must be a boolean/.test(d.message)),
    JSON.stringify(errs(r)));
});

// =====================================================================
// Condition must be fixed-phase (analyzer — the phase authority)
// =====================================================================

test('checked: stochastic condition is a phase error', () => {
  const r = process(`
v = external(reals)
x = draw(Normal(mu = 0.0, sigma = 1.0))
n = checked(value = v, condition = gt(x, 0.0))
`);
  assert.ok(errs(r).some((d: any) => /condition must be fixed-phase/.test(d.message)),
    JSON.stringify(errs(r)));
});

test('checked: parameterized condition is a phase error', () => {
  const r = process(`
mu = elementof(reals)
v = external(reals)
n = checked(value = v, condition = gt(mu, 0.0))
`);
  assert.ok(errs(r).some((d: any) => /condition must be fixed-phase/.test(d.message)),
    JSON.stringify(errs(r)));
});

test('checked: nested inside an expression is still phase-checked', () => {
  const r = process(`
mu = elementof(reals)
v = external(reals)
y = 2.0 * checked(v, condition = gt(mu, 0.0))
`);
  assert.ok(errs(r).some((d: any) => /condition must be fixed-phase/.test(d.message)),
    JSON.stringify(errs(r)));
});

test('checked: valid fixed-phase boolean condition is clean', () => {
  const r = process(`
v = external(reals)
n = checked(value = v, condition = equal(2, 2))
`);
  assert.equal(errs(r).length, 0, JSON.stringify(errs(r)));
});

// =====================================================================
// Runtime value-preserving guard (sampler.evaluateExpr)
// =====================================================================

const lit = (v: any) => ({ kind: 'lit', value: v });

test('checked: runtime returns value when condition is true', () => {
  const ir = { kind: 'call', op: 'checked', args: [lit(42)],
    kwargs: { condition: lit(true) } };
  assert.equal(sampler.evaluateExpr(ir, {}), 42);
});

test('checked: runtime throws when condition is false', () => {
  const ir = { kind: 'call', op: 'checked', args: [lit(42)],
    kwargs: { condition: lit(false) } };
  assert.throws(() => sampler.evaluateExpr(ir, {}), /assertion failed/);
});

test('checked: runtime with no condition is a plain pass-through', () => {
  const ir = { kind: 'call', op: 'checked', args: [lit(7)], kwargs: {} };
  assert.equal(sampler.evaluateExpr(ir, {}), 7);
});

test('checked: fully-positional runtime form', () => {
  const ir = { kind: 'call', op: 'checked', args: [lit(5), lit(true)] };
  assert.equal(sampler.evaluateExpr(ir, {}), 5);
});
