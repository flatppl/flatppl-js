'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index');

function process(src) {
  return processSource(src);
}

test('analyzer: classifies stochastic/deterministic/input', () => {
  const { bindings } = process(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
y = 2 * x
`);
  assert.equal(bindings.get('mu').type, 'input');
  assert.equal(bindings.get('x').type, 'stochastic');
  assert.equal(bindings.get('y').type, 'deterministic');
});

test('analyzer: classifies external as input', () => {
  const { bindings } = process('n = external(posintegers)\n');
  assert.equal(bindings.get('n').type, 'input');
});

test('analyzer: classifies lawof, functionof, fn', () => {
  const { bindings } = process(`
a = elementof(reals)
b = a * 2
m = lawof(b)
f = functionof(b)
g = fn(_ * 2)
`);
  assert.equal(bindings.get('m').type, 'lawof');
  assert.equal(bindings.get('f').type, 'functionof');
  assert.equal(bindings.get('g').type, 'fn');
});

test('analyzer: classifies likelihood and bayesupdate', () => {
  const { bindings } = process(`
a = elementof(reals)
m = Normal(mu = a, sigma = 1)
L = likelihoodof(m, 2.5)
post = bayesupdate(L, m)
`);
  assert.equal(bindings.get('L').type, 'likelihood');
  assert.equal(bindings.get('post').type, 'bayesupdate');
});

test('analyzer: classifies module loaders', () => {
  const { bindings } = process(`
m1 = load_module("foo.flatppl")
m2 = standard_module("particle-physics", "0.1")
d = load_data(source = "x.csv", valueset = reals)
`);
  assert.equal(bindings.get('m1').type, 'module');
  assert.equal(bindings.get('m2').type, 'module');
  assert.equal(bindings.get('d').type, 'data');
});

test('analyzer: classifies literals', () => {
  const { bindings } = process(`
n = 42
arr = [1, 2, 3]
s = "hello"
b = true
`);
  assert.equal(bindings.get('n').type, 'literal');
  assert.equal(bindings.get('arr').type, 'literal');
  assert.equal(bindings.get('s').type, 'literal');
  assert.equal(bindings.get('b').type, 'literal');
});

test('analyzer: dependency extraction', () => {
  const { bindings } = process(`
a = elementof(reals)
b = elementof(reals)
c = a + b
d = c * 2
`);
  assert.deepEqual(new Set(bindings.get('c').deps), new Set(['a', 'b']));
  assert.deepEqual(new Set(bindings.get('d').deps), new Set(['c']));
});

test('analyzer: keyword arg names are not deps', () => {
  // 'mu' here is a parameter name, not a reference to a bound var
  const { bindings } = process(`
val = elementof(reals)
m = Normal(mu = val, sigma = 1)
`);
  assert.deepEqual(bindings.get('m').deps, ['val']);
});

test('analyzer: callDeps tracks identifier in callable position', () => {
  const { bindings } = process(`
f = fn(_ * 2)
g = fn(_ + 1)
y = elementof(reals)
z = f(y)
`);
  assert.ok(bindings.get('z').callDeps.includes('f'));
  assert.ok(!bindings.get('z').callDeps.includes('y'));
});

test('analyzer: duplicate variable name is an error', () => {
  const { diagnostics } = process(`
x = 1
x = 2
`);
  assert.ok(diagnostics.some(d => /Duplicate/.test(d.message)));
});

test('analyzer: undefined variable warning', () => {
  const { diagnostics } = process(`
y = no_such_var + 1
`);
  assert.ok(diagnostics.some(d =>
    d.severity === 'warning' && /Undefined/.test(d.message)));
});

test('analyzer: builtin names are not flagged as undefined', () => {
  const { diagnostics } = process(`
mu = elementof(reals)
x = exp(mu)
`);
  assert.equal(diagnostics.filter(d => d.severity === 'warning').length, 0);
});

test('analyzer: lawof argument validation', () => {
  // No first argument
  const { diagnostics } = process('m = lawof(a = 1)\n');
  assert.ok(diagnostics.some(d => /functionof|lawof/i.test(d.message)));
});

test('analyzer: functionof requires keyword args after first', () => {
  const { diagnostics } = process(`
a = elementof(reals)
b = elementof(reals)
f = functionof(a + b, a, b)
`);
  // Args after the first must be keyword args
  assert.ok(diagnostics.some(d => /keyword/i.test(d.message)));
});

test('analyzer: builds symbols from bindings', () => {
  const { symbols } = process(`
a = elementof(reals)
m = lawof(a)
`);
  assert.equal(symbols.length, 2);
  assert.equal(symbols[0].name, 'a');
  assert.equal(symbols[0].type, 'input');
  assert.equal(symbols[1].name, 'm');
  assert.equal(symbols[1].type, 'lawof');
});
