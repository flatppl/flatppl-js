'use strict';

// Every measure's batch of array cells carries its atom axis as a nesting
// level of its own (value.batchOf), whichever handler built it. Before, the
// iid handlers tagged it by hand and the evaluate, Dirichlet and pushforward
// paths left it off, so the same [N, k] batch read as a vector per atom on one
// path and as an N x k matrix on another.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const valueLib = require('../value.ts');

async function batch(src: string, name: string, N = 8) {
  const { ctx } = ctxFor(src, N);
  return (await ctx.getMeasure(name)).value;
}

for (const [label, src, name, levels] of [
  ['an iid of scalars', 'x ~ iid(Normal(0.0, 1.0), 3)\n', 'x', [1]],
  ['a Dirichlet draw', 'x ~ Dirichlet([1.0, 2.0, 3.0])\n', 'x', [1]],
  ['an elementwise transform', 'x ~ Dirichlet([1.0, 2.0, 3.0])\ny = 2.0 .* x\n', 'y', [1]],
  ['a pushforward of a Dirichlet', 'x ~ pushfwd(fn(2.0 .* _), Dirichlet([1.0, 2.0, 3.0]))\n', 'x', [1]],
  ['an iid of vectors', 'x ~ iid(iid(Normal(0.0, 1.0), 2), 3)\n', 'x', [1, 1]],
] as [string, string, string, number[]][]) {
  test(`${label} batch carries the atom level`, async () => {
    const v = await batch(src, name);
    assert.equal(v.shape[0], 8);
    assert.deepEqual(valueLib.levelAxesOf(v), levels);
    // A slice returns one atom's cell, with the cell's own levels.
    assert.deepEqual(valueLib.levelAxesOf(valueLib._sliceRow(v, 0)), levels.slice(1));
  });
}

test('a batch of scalars stays untagged', async () => {
  const v = await batch('x ~ Normal(0.0, 1.0)\n', 'x');
  assert.deepEqual(v.shape, [8]);
  assert.equal(v.outerRank, undefined);
});

test('batchOf and cellLevelsOf', () => {
  const out = valueLib.batchOf({ shape: [4, 3], data: new Float64Array(12) }, []);
  assert.deepEqual(valueLib.levelAxesOf(out), [1]);
  assert.deepEqual(valueLib.cellLevelsOf({ shape: [3, 2], data: new Float64Array(6), outerRank: 1 }), [1]);
  assert.deepEqual(valueLib.cellLevelsOf([1, 2]), []);
  assert.equal(valueLib.cellLevelsOf(2.5), null);
  const tagged = { shape: [4, 3], data: new Float64Array(12), outerRank: 2 };
  assert.equal(valueLib.withBatchLevel(tagged).outerRank, 2, 'a tagged batch keeps its levels');
});
