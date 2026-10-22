'use strict';

// Spec §05 "Note on records" and §04 "Reification to functions and kernels".
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { toSexpr } = require('../pir-sexpr.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const INPUTS = 'a = elementof(reals)\nb = elementof(reals)\ne = a + 2*b\n';

function pir(source: string) {
  const result = processSource(source);
  assert.deepEqual(result.diagnostics.filter((d: any) => d.severity === 'error'), []);
  return toSexpr(result.loweredModule);
}

test('parenthesized records retain singleton shape, field order and nested expressions', () => {
  const { built } = makeMatCtx(`
record(x) = x + 100
r = (b = (a = 3), a = (4 + 1),)
single = (a = 7)
tuple = (r.b.a, r.a)
`, { sampleCount: 1 });
  assert.deepEqual(built.fixedValues.get('r'), { b: { a: 3 }, a: 5 });
  assert.deepEqual(Object.keys(built.fixedValues.get('r')), ['b', 'a']);
  assert.deepEqual(built.fixedValues.get('single'), { a: 7 });
  assert.deepEqual(built.fixedValues.get('tuple'), [3, 5]);
});

test('bare and keyword function boundaries preserve written input order', () => {
  const { built } = makeMatCtx(INPUTS + `
f = functionof(e, a, b)
g = functionof(e, q = b, a = a)
h = functionof(e, b, a)
p = functionof(a + 2*_x_, a, _x_)
v = (f(3, 5), g(5, 3), h(5, 3), p(3, 5))
`, { sampleCount: 1 });
  assert.deepEqual(built.fixedValues.get('v'), [13, 13, 13, 13]);
});

test('bare and keyword kernel boundaries score the declared conditional law', async () => {
  const { ctx } = makeMatCtx(INPUTS + `
y ~ Normal(e, 2)
K = kernelof(y, a, b)
L = kernelof(y, q = b, a = a)
M = kernelof(y, b, a)
lp = logdensityof(K(3, 5), 15)
lq = logdensityof(L(5, 3), 15)
lr = logdensityof(M(5, 3), 15)
`, { sampleCount: 1 });
  // -log(sqrt(2*pi)) - log(2) - (15-13)^2/(2*2^2).
  for (const name of ['lp', 'lq', 'lr']) {
    const m = await ctx.getMeasure(name);
    assert.ok(Math.abs((m.value?.data[0] ?? m.samples[0]) - (-2.112085713764618)) < 1e-12);
  }
});

test('record and boundary sugar lower to the existing operations', () => {
  assert.equal(pir('r = (a = 3, b = (c = 5))'),
    pir('r = record(a = 3, b = record(c = 5))'));
  assert.equal(pir(INPUTS + 'f = functionof(a + 2*_x_, a, _x_)'),
    pir(INPUTS + 'f = functionof(a + 2*_x_, a = a, x = _x_)'));
  assert.equal(pir(INPUTS + 'K = kernelof(draw(Normal(a + _shift_, 2)), _shift_, a)'),
    pir(INPUTS + 'K = kernelof(draw(Normal(a + _shift_, 2)), shift = _shift_, a = a)'));
  assert.equal(pir(INPUTS + 'y ~ Normal(e, 2)\nK = kernelof(y, a, b)'),
    pir(INPUTS + 'y ~ Normal(e, 2)\nK = kernelof(y, a = a, b = b)'));
});

test('boundary sugar preserves shadowed calls, explicit base access, lambdas and tuple values', () => {
  const { built } = makeMatCtx(INPUTS + `
functionof(x, y) = x - y
kernelof(x, y) = x + y
plain = (functionof(9, 4), kernelof(3, 5))
f = base.functionof(e, b, a)
value = f(5, 3)
t = (a, b)
g = base.functionof(t[1] + 2*t[2], pair = t)
tuplevalue = g((3, 5))
outer = (a, b) -> base.functionof(_a_ + 2*_b_, b, a)
lambdavalue = ((a, b) -> a + 2*b)(3, 5)
shadowcapture = ((x, y) -> functionof(y, y))(3, 5)
`, { sampleCount: 1 });
  assert.deepEqual(built.fixedValues.get('plain'), [5, 8]);
  assert.equal(built.fixedValues.get('value'), 13);
  assert.equal(built.fixedValues.get('tuplevalue'), 13);
  assert.equal(built.fixedValues.get('lambdavalue'), 13);
  assert.equal(built.fixedValues.get('shadowcapture'), 0);
  assert.equal(pir(INPUTS + 'outer = (a, b) -> base.functionof(_a_ + 2*_b_, b, a)'),
    pir(INPUTS + 'outer = (a, b) -> base.functionof(_a_ + 2*_b_, b = b, a = a)'));
});

test('boundary sugar does not admit grouped lists or relax ordinary calls', () => {
  for (const call of ['functionof(e, (a, b))', 'functionof(e, (a = a, b = b))',
    'kernelof(y, (a, b))', 'kernelof(y, (a = a, b = b))',
    'functionof(e, a + b)', 'functionof(e, (a))', 'f(a = 3, b)',
    'functionof(a = 3, b)', 'ordinary(a = 3, b)']) {
    const result = processSource(INPUTS + `y ~ Normal(e, 2)\nf(x, y) = x + y\nordinary(x, y) = x + y\nz = ${call}`);
    assert.ok(result.diagnostics.some((d: any) => d.severity === 'error'), call);
  }
  const shadowed = processSource('functionof(x, y) = x + y\nb = 5\nz = functionof(x = 3, b)');
  assert.ok(shadowed.diagnostics.some((d: any) => d.severity === 'error'));
});

test('boundary lists refuse mixing bare and keyword entries', () => {
  for (const call of ['functionof(e, b, p = a)', 'functionof(e, q = b, a)',
    'kernelof(y, b, p = a)', 'base.functionof(e, b, p = a)', 'functionof(e + _x_, a, x = _x_)',
    'functionof(e + _x_, _x_, b = b)']) {
    const result = processSource(INPUTS + `y ~ Normal(e, 2)\nz = ${call}`);
    assert.ok(result.diagnostics.some((d: any) => d.severity === 'error'
      && d.message.includes('mixing the two forms')), call);
  }
});

test('a bare placeholder must derive a legal, distinct input name', () => {
  for (const placeholder of ['_in_', '_all_', '_only_', '_true_', '_false_']) {
    const result = processSource(INPUTS + `z = functionof(a + ${placeholder}, a, ${placeholder})`);
    assert.ok(result.diagnostics.some((d: any) => d.severity === 'error'
      && d.message.includes('reserved word')), placeholder);
  }
  const duplicate = processSource('x = elementof(reals)\nz = functionof(x + _x_, x, _x_)');
  assert.ok(duplicate.diagnostics.some((d: any) => d.severity === 'error'
    && d.message.includes("Duplicate argument name 'x'")));
});

test('a bare placeholder may derive self or base, which are barred only as module bindings', () => {
  const bare = 'f = functionof(_self_ + _base_, _self_, _base_)\nz = f(self = 2, base = 3)';
  const keyword = 'f = functionof(_self_ + _base_, self = _self_, base = _base_)\nz = f(self = 2, base = 3)';
  assert.equal(pir(bare), pir(keyword));
  const { built } = makeMatCtx(bare, { sampleCount: 1 });
  assert.equal(built.fixedValues.get('z'), 5);
});
