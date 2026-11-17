'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const T = require('../types.ts');
const stdmod = require('../standard-modules.ts');
const V = require('../value.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');

// A small external descriptor exercises optional interfaces without depending
// on any particular standard-module numerical algorithm.
const mat = T.array(2, ['%dynamic', '%dynamic'], T.COMPLEX);
stdmod.registerStandardModule({ name: 'optional-test', compat: '0.1', bindings: new Map<string, any>([
  ['add', { kind: 'function', sig: T.funcType([
    { name: 'x', type: T.REAL }, { name: 'y', type: T.REAL, fixed: true }, { name: 'b', type: T.REAL, optional: true },
    { name: 'c', type: T.REAL, optional: true },
  ], T.REAL), impl: (x: number, y: number, b = 2, c = 3) => x + y + b + c }],
  ['solve', { kind: 'function', sig: { ...T.funcType([
    { name: 'A', type: mat }, { name: 'b', type: T.any() },
  ], T.array(1, ['%dynamic'], T.COMPLEX)), rhsResult: { input: 'b' } },
  realResult: T.array(1, ['%dynamic'], T.REAL), impl: (_a: any, b: any) => V.asValue(b) }],
]) });
const HEAD = 'm=standard_module("optional-test","0.1")\n';

test('optional module inputs retain defaults through aliases, records and broadcasts', async () => {
  const { ctx } = makeMatCtx(HEAD + `
f=m.add
a=f(1,0)
b=f(x=1,y=0,c=10)
c=f(record(x=1,y=0,c=10))
d=f.([1,2],0)
e=f.(x=[1,2],y=0,c=10)
r=load_module("optional.flatppl")
g=r.add
h=g(1,0)
`, { sampleCount: 1, bundle: { sources: { 'optional.flatppl': HEAD + 'add=m.add' } } });
  const invalid = require('..').processSource(HEAD + 't=elementof(reals)\na=m.add(1,t)');
  assert.ok(invalid.diagnostics.some((d: any) => d.message.includes('argument "y" must be fixed')));
  for (const [name, expected] of Object.entries({ a: [6], b: [13], c: [13], d: [6, 7], e: [13, 14], h: [6] })) {
    assert.deepEqual(Array.from((await ctx.getMeasure(name)).samples), expected);
  }
});

test('module RHS refinement preserves matrix shape and numeric promotion through exports', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[1,0],[0,1]])
B=rowstack([[1,2],[3,4]])
f=m.solve
X=f(A,B)
y=linsolve(X,[1.0,2.0])
r=load_module("optional.flatppl")
z=r.solve(A,[1,2])
`, { sampleCount: 1, bundle: { sources: { 'optional.flatppl': HEAD + 'solve=m.solve' } } });
  assert.deepEqual(Array.from((await ctx.getMeasure('X')).samples), [1, 2, 3, 4]);
  assert.deepEqual(Array.from((await ctx.getMeasure('y')).samples), [0, .5]);
  assert.deepEqual(Array.from((await ctx.getMeasure('z')).samples), [1, 2]);
});
