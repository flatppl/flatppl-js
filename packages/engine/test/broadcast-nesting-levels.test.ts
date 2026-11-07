'use strict';

// §04 broadcasting iterates the elements of the outermost array, and §03 keeps
// a vector of vectors distinct from a matrix. So a broadcast over
// vvv = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] iterates one axis and hands each
// vector of vectors whole. §07 gives splitblocks of a rank-r array a rank-r
// outer array, so its broadcast iterates r axes. A vector of two splitblocks
// values has one axis at its first level and two at its second.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator } = require('../index.ts');
const { ctxFor } = require('./_ctx-factory.ts');

function fixed(src: string, name = 'y'): any {
  const r = processSource(src);
  assert.deepEqual(r.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message), []);
  return orchestrator.buildDerivations(r.bindings).fixedValues.get(name);
}

const VVV = 'vvv = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]\n';
const SPLITS = 'b = splitblocks(rowstack([[1.0, 2.0], [3.0, 4.0]]), [1, 2])\n'
  + 'c = splitblocks(rowstack([[5.0, 6.0], [7.0, 8.0]]), [1, 2])\n';

test('rowstack.(vvv) stacks each vector of vectors', () => {
  const y = fixed(VVV + 'y = rowstack.(vvv)\n');
  assert.deepEqual(y.shape, [2, 2, 2]);
  assert.equal(y.outerRank, 1);
  assert.deepEqual(Array.from(y.data), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('sum.(vvv) sums each vector of vectors whole', () => {
  assert.deepEqual(Array.from(fixed(VVV + 'y = sum.(vvv)\n').data), [10, 26]);
});

test('a 2-D splitblocks broadcast still iterates both outer axes', () => {
  const y = fixed(SPLITS + 'y = sum.(b)\n');
  assert.deepEqual(y.shape, [2, 1]);
  assert.deepEqual(Array.from(y.data), [3, 7]);
});

test('a vector of two splitblocks values broadcasts over its two elements', () => {
  const v = fixed(SPLITS + 'y = [b, c]\n');
  assert.deepEqual(v.shape, [2, 2, 1, 1, 2]);
  assert.equal(v.outerRank, 3);
  assert.deepEqual(v.levelAxes, [1, 2]);
  const y = fixed(SPLITS + 'v = [b, c]\ny = joinblocks.(v)\n');
  assert.deepEqual(y.shape, [2, 2, 2]);
  assert.deepEqual(Array.from(y.data), [1, 2, 3, 4, 5, 6, 7, 8]);
});

test('a kernel broadcast over vvv hands each cell whole', async () => {
  // Each cell is a vector of two probability vectors; the kernel draws one
  // Categorical per inner vector.
  const src = 'pv = [[[0.2, 0.8], [0.5, 0.5]], [[0.9, 0.1], [0.3, 0.7]]]\n'
    + 'cat2(p) = Categorical.(p)\n'
    + 'lp = logdensityof(cat2.(pv), [[1, 2], [2, 2]])\n';
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('lp');
  const got = (m.value ? m.value.data : m.samples)[0];
  const want = Math.log(0.2) + Math.log(0.5) + Math.log(0.1) + Math.log(0.7);
  assert.ok(Math.abs(got - want) < 1e-12, `got ${got}, want ${want}`);
});
