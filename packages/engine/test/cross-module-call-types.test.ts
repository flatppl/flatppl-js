'use strict';

// §11 cross-module inference uses the imported callable and its concrete
// arguments. The importer's same-named bindings must not affect the result.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, pirSexpr } = require('..');
const T = require('../types.ts');

const library = 't = elementof(reals)\ny = t + 1.0\nf = functionof(y, t = t)';
function compile(body: string, source = library) {
  return processSource('m = load_module("lib.flatppl")\n' + body,
    { bundle: { sources: { 'lib.flatppl': source } } });
}
function errors(result: any): string[] {
  return result.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message);
}

test('a loaded callable returns its declared scalar type', () => {
  const r = compile('x = m.f(2.0)');
  assert.deepEqual(errors(r), []);
  assert.deepEqual(r.loweredModule.bindings.get('x').inferredType, T.REAL);
});

for (const call of ['m.f("bad")', 'm.f()', 'm.f(1.0, 2.0)', 'm.f(t=1.0, extra=2.0)', 'm.f(1.0, t=2.0)']) {
  test(`a loaded callable rejects invalid arguments: ${call}`, () => {
    assert.ok(errors(compile('x = ' + call)).some(m => /arg|expects/.test(m)));
  });
}

test('a direct imported call keeps module access checks', () => {
  assert.ok(errors(compile('x = m._f(2.0)', 't=elementof(reals)\ny=t+1.0\n_f=functionof(y,t=t)'))
    .some(m => /private/.test(m)));
  assert.ok(errors(compile('x = m.f(2.0)', 'f = 1.0')).some(m => /not callable/.test(m)));
});

test('a standard-module call checks the supported function signature', () => {
  const r = processSource('p = standard_module("polynomials", "0.1")\nx = p.chebyshev(2, 0.5)');
  assert.deepEqual(errors(r), []);
  assert.deepEqual(r.loweredModule.bindings.get('x').inferredType, T.REAL);
  const bad = processSource('p = standard_module("polynomials", "0.1")\nx = p.chebyshev("bad", 0.5)');
  assert.ok(errors(bad).some(m => /expects/.test(m)));
});

test('an imported polymorphic body uses each call argument in its own module scope', () => {
  const r = compile('offset = "caller"\nf = x -> "caller"\na = m.f(2.0)\nb = m.id([1.0, 2.0])',
    'offset = 1.0\nf = x -> x + offset\nid = x -> x');
  assert.deepEqual(errors(r), []);
  assert.deepEqual(r.loweredModule.bindings.get('a').inferredType, T.REAL);
  const b = r.loweredModule.bindings.get('b').inferredType;
  assert.equal(b.kind, 'array');
  assert.deepEqual(b.shape, [2]);
});

test('a renamed imported boundary rejects incompatible concrete arguments', () => {
  const source = 't=elementof(reals)\ny=t+1.0\nf=functionof(y,u=t)';
  assert.deepEqual(errors(compile('x=m.f(u=2.0)', source)), []);
  assert.ok(errors(compile('x=m.f(u="bad")', source)).some(m => /argument/.test(m)));
});

test('imported call inference retains dependency-owned fixed shapes', () => {
  const source = 'p=standard_module("polynomials","0.1")\n'
    + 'n=integer(p.chebyshev(2,2.0))\nf=x -> zeros(n) .+ x';
  const r = compile('n=99\nx=m.f(2.0)', source);
  assert.deepEqual(errors(r), []);
  assert.deepEqual(r.loweredModule.bindings.get('x').inferredType.shape, [7]);
});

test('call order does not specialize stored dependency annotations', () => {
  const source = 'f=x -> x .+ 1.0';
  const a = compile('a=m.f(2.0)\nb=m.f([1.0,2.0])', source);
  const b = compile('b=m.f([1.0,2.0])\na=m.f(2.0)', source);
  assert.deepEqual(errors(a), []);
  assert.deepEqual(errors(b), []);
  assert.equal(pirSexpr.toSexpr(a.modules.get('lib.flatppl').loweredModule, { meta: true }),
    pirSexpr.toSexpr(b.modules.get('lib.flatppl').loweredModule, { meta: true }));
  for (const r of [a, b]) {
    assert.deepEqual(r.loweredModule.bindings.get('a').inferredType, T.REAL);
    assert.deepEqual(r.loweredModule.bindings.get('b').inferredType.shape, [2]);
  }
});

test('recursive callable definitions report errors without a stack overflow', () => {
  for (const source of ['f=x -> f(x)', 'f=x -> g(x)\ng=x -> f(x)']) {
    const r = processSource(source);
    assert.ok(errors(r).length > 0);
  }
});

test('an imported call preserves the dependency mass annotations', () => {
  for (const source of ['offset=Normal(0.0,1.0)\nf=x -> logdensityof(offset,x)',
    'g=x -> Normal(x,1.0)\nf=x -> g(x)']) {
    const before = compile('', source);
    const after = compile('a=m.f(2.0)', source);
    assert.deepEqual(errors(after), []);
    assert.equal(pirSexpr.toSexpr(before.modules.get('lib.flatppl').loweredModule, { meta: true }),
      pirSexpr.toSexpr(after.modules.get('lib.flatppl').loweredModule, { meta: true }));
  }
});
