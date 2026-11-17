'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const HEAD = 'e=standard_module("ext-linear-algebra","0.1")\n';
const close = (a: ArrayLike<number>, b: number[]) => {
  assert.equal(a.length, b.length);
  b.forEach((x, i) => assert.ok(Math.abs(a[i] - x) < 2e-13, `${i}: ${a[i]} vs ${x}`));
};

test('complex and mixed Kronecker calls retain matrix blocks and adjoint coordinates', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[complex(0,1),complex(2,0)],[complex(1,-1),complex(0,0)]])
B=rowstack([[complex(1,1),complex(3,0)]])
kron=e.kron
K=kron(A=A,B=B)
R=e.kron(rowstack([[1,2]]),B)
S=e.kron(A,rowstack([[1,2]]))
H=e.kron(adjoint(A),B)
realK=e.kron(rowstack([[1,0],[0,2]]),rowstack([[3]]))
x=linsolve(realK,[3.0,12.0])`, { sampleCount: 1 });
  // Hand multiplication of each A_ij into the whole B row.
  const k = await ctx.getMeasure('K');
  assert.deepEqual(k.intrinsicShape, [2, 4]);
  close(k.samples, [-1, 0, 2, 6, 2, 3, 0, 0]);
  close(k.imag, [1, 3, 2, 0, 0, -3, 0, 0]);
  close((await ctx.getMeasure('R')).samples, [1, 3, 2, 6]);
  close((await ctx.getMeasure('R')).imag, [1, 0, 2, 0]);
  close((await ctx.getMeasure('S')).samples, [0, 0, 2, 4, 1, 2, 0, 0]);
  close((await ctx.getMeasure('S')).imag, [1, 2, 0, 0, -1, -2, 0, 0]);
  close((await ctx.getMeasure('H')).samples, [1, 0, 0, 3, 2, 6, 0, 0]);
  close((await ctx.getMeasure('H')).imag, [-1, -3, 2, 3, 2, 0, 0, 0]);
  close((await ctx.getMeasure('x')).samples, [1, 2]);
});

test('Kronecker products consume each live complex matrix atom', async () => {
  const { ctx } = makeMatCtx(HEAD + `
t~Normal(0,2)
A=rowstack([[complex(t,1)]])
B=rowstack([[complex(1,2),complex(3,-1)]])
K=e.kron(A,B)`, { sampleCount: 32, rootSeed: 117 });
  const t = (await ctx.getMeasure('t')).samples, k = await ctx.getMeasure('K');
  assert.deepEqual(k.value.shape, [32, 1, 2]);
  for (let i = 0; i < t.length; i++) {
    close(k.samples.slice(2 * i, 2 * i + 2), [t[i] - 2, 3 * t[i] + 1]);
    close(k.imag.slice(2 * i, 2 * i + 2), [2 * t[i] + 1, 3 - t[i]]);
  }
});
