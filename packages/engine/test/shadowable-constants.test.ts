'use strict';

// Spec §05 "Note on reserved words" (flatppl-design #215): every built-in
// name is shadowable, including the numeric constants `im`, `pi`, `inf`. A
// module binding or a lambda/function argument wins over the built-in by
// ordinary name resolution. `self`, `base` and the grammar keywords stay
// refused.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter((d: any) => d.severity === 'error');
}

async function scalar(src: string, name: string) {
  const { ctx } = ctxFor(src, 16);
  return (await ctx.getMeasure(name)).samples[0];
}

test('shadowing: a module binding `pi = 3.0` is used by a later expression', async () => {
  const src = 'pi = 3.0\ny = pi * 2.0\n';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'y'), 6.0);
});

test('shadowing: the binding wins when it is defined after its use', async () => {
  const src = 'y = pi * 2.0\npi = 3.0\n';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'y'), 6.0);
});

test('shadowing: a bound `inf` is a binding reference, not the constant', async () => {
  const src = 'inf = 5.0\ny = inf + 1.0\n';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'y'), 6.0);
});

test('shadowing: a bound `inf` is not read as an interval bound', async () => {
  const src = 'inf = 2.0\nx = draw(Uniform(support = interval(0.0, inf)))\n'
    + 'lp = logdensityof(Uniform(support = interval(0.0, inf)), 1.0)\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await scalar(src, 'lp') - Math.log(0.5)) < 1e-12);
});

test('shadowing: a bound `pi` feeds sampling and density', async () => {
  const src = 'pi = 3.0\nx = draw(Normal(mu = pi, sigma = 1.0))\n'
    + 'lp = logdensityof(Normal(mu = pi, sigma = 1.0), 3.0)\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await scalar(src, 'lp') + 0.5 * Math.log(2 * Math.PI)) < 1e-12);
  const { ctx } = ctxFor(src, 4096);
  const x = (await ctx.getMeasure('x')).samples;
  let mean = 0;
  for (let i = 0; i < x.length; i++) mean += x[i] / x.length;
  assert.ok(Math.abs(mean - 3.0) < 0.1, `mean ${mean}`);
});

test('shadowing: a bound `im` is a binding reference', async () => {
  const src = 'im = 4.0\ny = im * 2.0\n';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'y'), 8.0);
});

test('shadowing: a lambda argument `inf` shadows the constant in the body only', async () => {
  const src = 'f = inf -> inf + 1.0\ny = f(2.0)\nz = inf\n';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'y'), 3.0);
  assert.equal(await scalar(src, 'z'), Infinity);
});

test('shadowing: a function-definition argument `pi` shadows the constant', async () => {
  const src = 'g(pi) = pi * 2.0\ny = g(5.0)\nz = pi\n';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'y'), 10.0);
  assert.ok(Math.abs(await scalar(src, 'z') - Math.PI) < 1e-15);
});

test('shadowing: an unshadowed `pi` still means π', async () => {
  const src = 'y = pi * 2.0\n';
  assert.deepEqual(errors(src), []);
  assert.ok(Math.abs(await scalar(src, 'y') - 2 * Math.PI) < 1e-15);
});

test('shadowing: a binding in one statement does not change an argument-free sibling', async () => {
  const src = 'f = pi -> pi + 1.0\ny = f(2.0)\nz = pi * 1.0\n';
  assert.deepEqual(errors(src), []);
  assert.equal(await scalar(src, 'y'), 3.0);
  assert.ok(Math.abs(await scalar(src, 'z') - Math.PI) < 1e-15);
});

test('shadowing: a bound `pi` lowers to a self reference, an unbound one to a constant', () => {
  const bound = processSource('pi = 3.0\ny = pi\n').loweredModule;
  const rhs = bound.bindings.get('y').rhs;
  assert.deepEqual({ kind: rhs.kind, ns: rhs.ns, name: rhs.name },
    { kind: 'ref', ns: 'self', name: 'pi' });
  const unbound = processSource('y = pi\n').loweredModule;
  assert.equal(unbound.bindings.get('y').rhs.kind, 'const');
});

// The engine refuses `self` and `base` as argument names only; it does not
// yet refuse them as module-binding names (existing tests bind `base`).
test('reserved: `self` and `base` are still refused as lambda and function arguments', () => {
  for (const name of ['self', 'base']) {
    assert.ok(errors(`f = ${name} -> ${name}`).length > 0, `lambda ${name}`);
    assert.ok(errors(`f(${name}) = ${name}`).length > 0, `def ${name}`);
  }
});

test('reserved: the keywords are still refused as bindings', () => {
  for (const name of ['in', 'true', 'false', 'all', 'only']) {
    assert.ok(errors(`${name} = 1.0`).length > 0, `${name} = 1.0`);
  }
});
