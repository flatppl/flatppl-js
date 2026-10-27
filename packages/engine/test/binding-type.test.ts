'use strict';

// §04 input/module producers and §07 loaded data keep their consumer tags
// through lifting; the source AST remains available for source navigation.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('..');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('lifted input and data producers retain their source expressions', () => {
  const source = 'r=record(p=elementof(reals),n=external(posintegers),d=load_data("data.json",reals))';
  const parsed = processSource(source);
  assert.deepEqual(parsed.diagnostics, []);
  const original = parsed.bindings.get('r');
  const before = structuredClone(original.node);
  const lifted = orchestrator.liftInlineSubexpressions(parsed.bindings);
  const fields = lifted.get('r').node.value.args;
  for (const [i, tag] of ['input', 'input', 'data'].entries()) {
    const binding = lifted.get(fields[i].value.name);
    assert.equal(binding.type, tag);
    assert.deepEqual(binding.node.value, before.value.args[i].value);
    assert.deepEqual(binding.nameLoc, before.value.args[i].value.loc);
  }
  assert.deepEqual(original.node, before);
  assert.equal(lifted.get('r').rhs, original.rhs);
  assert.deepEqual(lifted.get('r').nameLoc, original.nameLoc);
});

test('module references and loaded data survive lift and evaluate with their own sources', async () => {
  const source = 'm=load_module("dep.flatppl")\ns=standard_module("polynomials","0.1")\nd=load_data("data.json",reals)\nscore=s.legendre(2,m.x)+d';
  const { ctx, lifted } = makeMatCtx(source, {
    path: 'model.flatppl', sampleCount: 1,
    bundle: { sources: { 'dep.flatppl': 'x=3', 'data.json': '7' } },
  });
  const bindings = orchestrator.liftInlineSubexpressions(lifted.bindings);
  for (const [name, tag] of [['m', 'module'], ['s', 'module'], ['d', 'data']]) {
    assert.equal(lifted.bindings.get(name).type, tag);
    assert.equal(bindings.get(name).type, tag);
    assert.equal(bindings.get(name).name, name);
    assert.deepEqual(bindings.get(name).nameLoc, lifted.bindings.get(name).nameLoc);
  }
  // P₂(3) + 7 = (3·3² − 1)/2 + 7.
  assert.equal((await ctx.getMeasure('score')).samples[0], 20);
});

test('shadowed producer names evaluate as ordinary user functions', async () => {
  const source = 'elementof(x)=x+1\nexternal(x)=x+2\nload_module(x)=x+3\nstandard_module(x)=x+4\nload_data(x)=x+5\nr=record(a=elementof(10),b=external(10),c=load_module(10),d=standard_module(10),e=load_data(10))';
  const { ctx, built } = makeMatCtx(source, { sampleCount: 1 });
  const fields = (await ctx.getMeasure('r')).fields;
  assert.deepEqual(Object.values(fields).map((m: any) => m.samples[0]), [11, 12, 13, 14, 15]);
  const names = built.bindings.get('r').node.value.args.map((a: any) => a.value.name);
  for (const name of names) assert.equal(built.bindings.get(name).type, 'call');
});
