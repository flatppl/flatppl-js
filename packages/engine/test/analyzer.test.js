'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function process(src) {
  return processSource(src);
}

test('analyzer: classifies draw/call/input', () => {
  const { bindings } = process(`
mu = elementof(reals)
x = draw(Normal(mu = mu, sigma = 1))
y = 2 * x
`);
  assert.equal(bindings.get('mu').type, 'input');
  assert.equal(bindings.get('x').type, 'draw');
  assert.equal(bindings.get('y').type, 'call');
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

// validateSpecialOperation completion coverage — each case wired in
// analyzer.ts. These are structural shape checks (positional vs
// keyword, fixed-vs-variadic argument count). The test bodies use
// inline source so each one stays self-contained and the failure
// message + offending source live in one place.

test('analyzer: weighted/logweighted/truncate/pushfwd take exactly 2 positional args', () => {
  for (const op of ['weighted', 'logweighted', 'truncate', 'pushfwd']) {
    const wrongCount = process(`m = ${op}(1)\n`);
    assert.ok(wrongCount.diagnostics.some(d => /takes exactly two/.test(d.message)), op + ' wrongCount');
    const withKwarg = process(`m = ${op}(a = 1, b = 2)\n`);
    assert.ok(withKwarg.diagnostics.some(d => /positional/i.test(d.message)), op + ' kwarg');
  }
});

test('analyzer: superpose / fchain require at least 2 positional args', () => {
  for (const op of ['superpose', 'fchain']) {
    const tooFew = process(`m = ${op}(a)\n`);
    assert.ok(tooFew.diagnostics.some(d => /at least two/.test(d.message)), op + ' tooFew');
  }
});

test('analyzer: joint/jointchain/kchain reject mixed positional+keyword', () => {
  for (const op of ['joint', 'jointchain', 'kchain']) {
    const mixed = process(`m = ${op}(a, b = c)\n`);
    assert.ok(mixed.diagnostics.some(d => /all positional or all keyword/.test(d.message)),
              op + ' mixed');
  }
});

test('analyzer: joint(x = M) (single-kwarg form) is valid', () => {
  const { diagnostics } = process(`
a = elementof(reals)
m = joint(obs = a)
`);
  assert.ok(!diagnostics.some(d => /joint\(\)/.test(d.message)));
});

test('analyzer: iid requires at least 2 positional args', () => {
  assert.ok(process(`m = iid(M)\n`).diagnostics.some(d => /at least two/.test(d.message)));
  assert.ok(process(`m = iid(M, n = 3)\n`).diagnostics.some(d => /positional/i.test(d.message)));
});

test('analyzer: bijection takes exactly 3 positional args', () => {
  assert.ok(process(`b = bijection(f, g)\n`).diagnostics.some(d => /takes exactly three/.test(d.message)));
  assert.ok(process(`b = bijection(f = a, g = b, h = c)\n`).diagnostics.some(d => /positional/i.test(d.message)));
});

test('analyzer: broadcasted takes exactly 1 positional arg', () => {
  assert.ok(process(`f = broadcasted(a, b)\n`).diagnostics.some(d => /takes exactly one/.test(d.message)));
});

test('analyzer: broadcast requires positional head + at least 1 more arg', () => {
  assert.ok(process(`x = broadcast(Normal)\n`).diagnostics.some(d => /at least two/.test(d.message)));
  assert.ok(process(`x = broadcast(f = Normal, mu = 1)\n`).diagnostics.some(d => /positional head/.test(d.message)));
});

test('analyzer: reduce / scan take exactly 3 positional args', () => {
  for (const op of ['reduce', 'scan']) {
    assert.ok(process(`y = ${op}(f, init)\n`).diagnostics.some(d => /takes exactly three/.test(d.message)),
              op + ' tooFew');
  }
});

test('analyzer: checked takes positional value + kwargs after', () => {
  assert.ok(process(`y = checked()\n`).diagnostics.some(d => /requires a value/.test(d.message)));
  assert.ok(process(`y = checked(x, y)\n`).diagnostics.some(d => /only keyword arguments/.test(d.message)));
});

test('analyzer: record / table require keyword args only', () => {
  for (const op of ['record', 'table']) {
    const pos = process(`r = ${op}(a, b)\n`);
    assert.ok(pos.diagnostics.some(d => /keyword arguments only/.test(d.message)), op + ' positional');
    const empty = process(`r = ${op}()\n`);
    assert.ok(empty.diagnostics.some(d => /at least one field/.test(d.message)), op + ' empty');
  }
});

test('analyzer: tuple / vector take positional args only', () => {
  for (const op of ['tuple', 'vector']) {
    assert.ok(process(`v = ${op}(a = 1)\n`).diagnostics.some(d => /positional/i.test(d.message)),
              op + ' kwarg');
  }
});

test('analyzer: disintegrate first arg must be string literal or array of string literals', () => {
  const okString = process(`a, b = disintegrate("obs", joint_model)\njoint_model = elementof(reals)\n`);
  assert.ok(!okString.diagnostics.some(d => /disintegrate/.test(d.message)));
  const okArray = process(`a, b = disintegrate(["o1", "o2"], joint_model)\njoint_model = elementof(reals)\n`);
  assert.ok(!okArray.diagnostics.some(d => /disintegrate/.test(d.message)));
  const badIdent = process(`a, b = disintegrate(some_var, m)\nsome_var = "obs"\nm = elementof(reals)\n`);
  assert.ok(badIdent.diagnostics.some(d => /string literal or an array of string literals/.test(d.message)));
});

test('analyzer: interval / cartpow / stdsimplex set constructors', () => {
  assert.ok(process(`s = interval(0)\n`).diagnostics.some(d => /takes exactly two/.test(d.message)));
  assert.ok(process(`s = cartpow(reals)\n`).diagnostics.some(d => /takes exactly two/.test(d.message)));
  assert.ok(process(`s = cartprod(reals)\n`).diagnostics.some(d => /at least two/.test(d.message)));
  assert.ok(process(`s = stdsimplex()\n`).diagnostics.some(d => /takes exactly one/.test(d.message)));
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
