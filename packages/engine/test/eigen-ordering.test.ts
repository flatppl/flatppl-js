'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('a small complex pair keeps its coordinates beside a large real eigenvalue', async () => {
  const { ctx } = makeMatCtx(`e=standard_module("ext-linear-algebra","0.1")
A=rowstack([[1e10,0.0,0.0],[0.0,0.0,-1e-8],[0.0,1e-8,0.0]])
f=e.eigen(A)
w=f.values
lo=e.eigmin(A)
hi=e.eigmax(A)`, { sampleCount: 1 });
  // The independent block spectrum is {1e10, +i·1e-8, -i·1e-8}.
  const w = await ctx.getMeasure('w');
  const pair = Array.from(w.imag).filter((x: number) => Math.abs(x) > 0).sort((a: number, b: number) => a - b);
  assert.equal(pair.length, 2);
  assert.ok(Math.abs(pair[0] / -1e-8 - 1) < 2e-13);
  assert.ok(Math.abs(pair[1] / 1e-8 - 1) < 2e-13);
  assert.ok(Array.from(w.samples).some((x: number) => x === 1e10));
  // These correctly computed complex values admit neither real ordering.
  await assert.rejects(ctx.getMeasure('lo'));
  await assert.rejects(ctx.getMeasure('hi'));
});
