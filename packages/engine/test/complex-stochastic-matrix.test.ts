'use strict';

// §03/§07: each live matrix value retains both coordinates, including atoms
// whose imaginary plane is zero. The oracle is its literal entry definition.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('live complex matrix cells pack zero and nonzero imaginary planes together', async () => {
  const { ctx } = makeMatCtx(`
t~Bernoulli(0.5)
A=rowstack([[complex(1.0,t),2.0],[0.0,3.0]])
C=conj.(A)
build=fn(record(M=rowstack([[complex(_,1.0),2.0],[0.0,3.0]])))
result=build(t)
M=result.M
n=lengthof(filter(fn(_<0),[1.0]))
E=fill(complex(t,1.0),[n,n])
`, { sampleCount: 64, rootSeed: 117 });
  const t = (await ctx.getMeasure('t')).samples;
  const a = await ctx.getMeasure('A'), c = await ctx.getMeasure('C'), m = await ctx.getMeasure('M');
  assert.deepEqual(a.value.shape, [64, 2, 2]);
  const empty = await ctx.getMeasure('E');
  assert.deepEqual(empty.value.shape, [64, 0, 0]);
  assert.equal(empty.samples.length, 0);
  for (let i = 0; i < t.length; i++) {
    assert.deepEqual(Array.from(a.samples.slice(4 * i, 4 * i + 4)), [1, 2, 0, 3]);
    assert.deepEqual(Array.from(a.imag.slice(4 * i, 4 * i + 4)), [t[i], 0, 0, 0]);
    assert.equal(c.imag[4 * i] + 0, -t[i] + 0);
    assert.equal(m.samples[4 * i], t[i]);
    assert.equal(m.imag[4 * i], 1);
  }
  assert.ok(t.some((x: number) => x === 0) && t.some((x: number) => x === 1));
});
