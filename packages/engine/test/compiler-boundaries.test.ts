'use strict';

// Public compiler boundaries from §§04, 05, and 11: names retain their scope,
// nested functions own their holes, and lexical recovery preserves later code.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, pirSexpr } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

function errors(result: any) {
  return result.diagnostics.filter((d: any) => d.severity === 'error');
}

function fixed(source: string, options?: any) {
  const result = processSource(source, options);
  assert.deepEqual(errors(result), []);
  return { result, values: orchestrator.buildDerivations(result.linkedBindings).fixedValues };
}

test('loaded self references retain their module instance', () => {
  const { values } = fixed('a=100\nload_module(x)=x+1\nn=load_module(2)\n'
    + 'm=base.load_module("m.flatppl")\nz=m.y', {
    bundle: { sources: { 'm.flatppl': 'a=3\ny=self.a+1' } },
  });
  assert.equal(values.get('z'), 4);
  assert.equal(values.get('n'), 3);
});

test('nested fn holes map a vector without capturing inner arguments', () => {
  const { values } = fixed('f=fn(broadcast(fn(_+1), _))\ny=f([2,3])');
  assert.deepEqual(Array.from(values.get('y').data), [3, 4]);
});

test('shadowed builtins preserve calls and values while operators keep their meaning', () => {
  const { result, values } = fixed('reals=[4,5]\nx=reals[1]\nadd(x)=x+10\ny=add(1)\nz=1+2\n'
    + 'pow(x)=x+10\np=pow(2)\nq=2^3\nv=[2,3].^2\ndraw(x)=x+1\nd=draw(2)\ns~Normal(0,1)');
  assert.deepEqual(['x', 'y', 'z', 'p', 'q', 'd'].map(n => values.get(n)), [4, 11, 3, 12, 8, 3]);
  assert.deepEqual(Array.from(values.get('v').data), [4, 9]);
  assert.equal(result.bindings.get('d').phase, 'fixed');
  assert.equal(result.bindings.get('s').phase, 'stochastic');
  const wire = pirSexpr.toSexpr(result.loweredModule);
  assert.match(wire, /\(%call \(%ref self add\) 1\)/);
  assert.match(wire, /\(%bind z \(add 1 2\)\)/);
});

test('self and base qualification select the requested namespace', () => {
  const { values } = fixed('exp(x)=x+10\na=self.exp(2)\nb=base.exp(1)\nc=base.pi\n'
    + 'g=base.exp\nd=g(1)\nfn=3\nf=base.fn(broadcast(base.fn(_+1),_))\ny=f([2,3])');
  assert.deepEqual(['a', 'b', 'c', 'd'].map(n => values.get(n)), [12, Math.E, Math.PI, Math.E]);
  assert.deepEqual(Array.from(values.get('y').data), [3, 4]);
});

test('syntax-defined builtin names do not capture module inputs', () => {
  const { result, values } = fixed('pow=elementof(reals)\nsum=elementof(reals)\n'
    + 'x=2^3\nv=[1,2]\na[]:=v[.i]\nb=base.aggregate(base.sum,[],v[.i])');
  assert.equal(result.bindings.get('x').phase, 'fixed');
  assert.equal(result.bindings.get('a').phase, 'fixed');
  assert.deepEqual([values.get('x'), values.get('a'), values.get('b')], [8, 3, 3]);
  assert.match(pirSexpr.toSexpr(result.loweredModule), /\(aggregate sum /);
  const local = fixed('f(reals)=reals+1\ng=reals->reals+2\nh(pow)=2^3+pow\n'
    + 'x=f(2)\ny=g(2)\nz=h(2)').values;
  assert.deepEqual(['x', 'y', 'z'].map(n => local.get(n)), [3, 4, 10]);
});

test('kernel pushforward permits projection from its mapped record variate', async () => {
  const { proc: result, ctx } = ctxFor('mu=elementof(reals)\ny~Dirac(value=mu)\n'
    + 'k=kernelof(y,mu=mu)\np=pushfwd(x -> record(value=x),k)\n'
    + 'a=p(2)\nv~a\nz=v.value', 4);
  assert.deepEqual(errors(result), []);
  assert.match(pirSexpr.toSexpr(result.loweredModule), /\(%bind z \(get \(%ref self v\) "value"\)\)/);
  assert.deepEqual(Array.from((await ctx.getMeasure('z')).samples), [2, 2, 2, 2]);
});

test('relabel permits projection from its named measure variate', async () => {
  const { proc: result, ctx } = ctxFor('m=relabel(Dirac(value=3),["x"])\ny~m\nz=y.x', 4);
  assert.deepEqual(errors(result), []);
  assert.match(pirSexpr.toSexpr(result.loweredModule), /\(%bind z \(get \(%ref self y\) "x"\)\)/);
  assert.deepEqual(Array.from((await ctx.getMeasure('z')).samples), [3, 3, 3, 3]);
});

test('Dirac positional and keyword calls preserve their variate domain', () => {
  const result = processSource('x~Dirac(2.5)\ny~Dirac(value=2.5)\na=x+1\nb=y+1\n'
    + 'v~Dirac([1,2])\nw=v.+1');
  assert.deepEqual(errors(result), []);
  assert.deepEqual(result.bindings.get('a').inferredType, { kind: 'scalar', prim: 'real' });
  assert.deepEqual(result.bindings.get('a').inferredType, result.bindings.get('b').inferredType);
  assert.deepEqual(result.bindings.get('w').inferredType.shape, [2]);
});

test('stacked integer observations preserve element type and column orientation', () => {
  const { result, values } = fixed('R=rowstack([[1,2,3],[4,5,6]])\n'
    + 'C=colstack([[1,2,3],[4,5,6]])\nlp=logdensityof(iid(Poisson(1),[2,3]),R)');
  assert.equal(result.bindings.get('R').inferredType.elem.prim, 'integer');
  assert.deepEqual(result.bindings.get('C').inferredType.shape, values.get('C').shape);
  assert.deepEqual(values.get('C').shape, [3, 2]);
});

test('kchain scoring demands an exact density while sampling stays available', () => {
  const source = 'M=kchain(Normal(0,1),fn(Normal(_^2,1)))';
  assert.deepEqual(errors(processSource(source + '\nx~M')), []);
  for (const score of ['logdensityof(M,0)', 'broadcast(logdensityof,M,[0,1])']) {
    const result = processSource(source + '\nlp=' + score);
    assert.ok(errors(result).some((d: any) => d.loc?.start.line === 1));
  }
  assert.deepEqual(errors(processSource('M=kchain(Normal(0,1),fn(Normal(_,1)))\nlp=logdensityof(M,0)')), []);
  assert.ok(errors(processSource('c=base.kchain\n'
    + 'M=c(Normal(0,1),fn(Normal(_^2,1)))\nlp=logdensityof(M,0)')).length > 0);
  const imported = processSource('m=load_module("m.flatppl")\nlp=logdensityof(m.M,0)', {
    bundle: { sources: { 'm.flatppl': 'c=base.kchain\nM=c(Normal(0,1),fn(Normal(_^2,1)))' } },
  });
  assert.ok(errors(imported).some((d: any) => /closed form or finite discrete enumeration/.test(d.message)));
});

test('comment closing fences require a blank tail', () => {
  const { result, values } = fixed('###\ncomment\n### text\nx=1\n###\n'
    + '%%%\nalpha\n%%% text\nbeta\n%%%\ny=2');
  assert.deepEqual([...result.bindings.keys()], ['y']);
  assert.equal(values.get('y'), 2);
  assert.deepEqual(result.loweredModule.bindings.get('y').doc.lines, ['alpha', '%%% text', 'beta']);
  assert.ok(errors(processSource('###\nunclosed')).length > 0);
});

test('a cycle reached through lawof stays a compiler diagnostic', () => {
  const result = processSource('a=b\nb=a\nc=lawof(a)\ny=7');
  assert.ok(errors(result).length > 0);
  assert.equal(orchestrator.buildDerivations(new Map([['y', result.bindings.get('y')]])).fixedValues.get('y'), 7);
});

test('checked requires its condition and preserves valid checked values', () => {
  assert.ok(errors(processSource('x=checked(7)')).length > 0);
  const { values } = fixed('x=checked(value=7, condition=true)\ny=checked(8, condition=true)');
  assert.deepEqual([values.get('x'), values.get('y')], [7, 8]);
});

test('reserved function parameters diagnose without discarding later bindings', () => {
  const result = processSource('f(true)=true\ny=7');
  assert.ok(errors(result).length > 0);
  assert.equal(orchestrator.buildDerivations(result.linkedBindings).fixedValues.get('y'), 7);
  assert.equal(fixed('f(x)=x\ny=f(2)').values.get('y'), 2);
});

test('CR and CRLF end comments and statements without changing string bytes', () => {
  const { result, values } = fixed('# comment\rx=1\r\n%%%\rline\r%%%\ry=x+2\rs="a\rb"');
  assert.equal(values.get('y'), 3);
  assert.equal(values.get('s'), 'a\rb');
  assert.deepEqual(result.loweredModule.bindings.get('y').doc.lines, ['line']);
});

test('incomplete exponents diagnose and preserve the next statement', () => {
  const result = processSource('x=1e+\ny=2e2');
  assert.ok(errors(result).length > 0);
  assert.equal(orchestrator.buildDerivations(result.linkedBindings).fixedValues.get('y'), 200);
});
