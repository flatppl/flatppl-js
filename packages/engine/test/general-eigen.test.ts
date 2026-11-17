'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const fixture = require('./fixtures/general-eigen.json');
const HEAD = 'e=standard_module("ext-linear-algebra","0.1")\n';

function match(re: ArrayLike<number>, im: ArrayLike<number>, expected: Array<[number, number]>, scale: number) {
  const remaining = expected.map((_, i) => i), order: number[] = [];
  for (let j = 0; j < re.length; j++) {
    let best = 0, error = Infinity;
    remaining.forEach((k, i) => {
      const delta = Math.hypot(re[j] - expected[k][0], im[j] - expected[k][1]);
      if (delta < error) { error = delta; best = i; }
    });
    assert.ok(error <= 2e-12 * scale, `eigenvalue ${j}: ${re[j]}+${im[j]}i, error ${error}`);
    order.push(remaining.splice(best, 1)[0]);
  }
  assert.equal(remaining.length, 0);
  return order;
}

test('general complex eigenpairs match independent SciPy values and vector directions', async () => {
  const entries = fixture.A.re.map((r: number, i: number) => `complex(${r},${fixture.A.im[i]})`);
  const rows = Array.from({ length: 3 }, (_, i) => `[${entries.slice(3 * i, 3 * i + 3)}]`);
  const { ctx } = makeMatCtx(HEAD + `A=rowstack([${rows}])
eig=e.eigen
f=eig(A=A)
w=f.values
V=f.vectors
g=e.eigen(adjoint(A))
wa=g.values`, { sampleCount: 1 });
  const w = await ctx.getMeasure('w'), v = await ctx.getMeasure('V');
  const expected = fixture.values.re.map((r: number, i: number) => [r, fixture.values.im[i]]);
  const order = match(w.samples, w.imag, expected, 5);
  for (let j = 0; j < 3; j++) {
    let re = 0, im = 0;
    for (let i = 0; i < 3; i++) {
      const a = i * 3 + order[j], b = i * 3 + j;
      re += fixture.vectors.re[a] * v.samples[b] + fixture.vectors.im[a] * v.imag[b];
      im += fixture.vectors.re[a] * v.imag[b] - fixture.vectors.im[a] * v.samples[b];
    }
    assert.ok(Math.abs(Math.hypot(re, im) - 1) < 2e-12, `vector ${j}`);
  }
  const adjoint = await ctx.getMeasure('wa');
  match(adjoint.samples, adjoint.imag, expected.map(([r, i]: number[]) => [r, -i]), 5);
});

test('live real matrices retain conjugate eigenvalues and right vectors at every atom', async () => {
  const { ctx } = makeMatCtx(HEAD + `
s~Uniform(interval(0.5,2.0))
A=rowstack([[0.0,-s],[s,0.0]])
f=e.eigen(A)
w=f.values
V=f.vectors`, { sampleCount: 32, rootSeed: 117 });
  const s = (await ctx.getMeasure('s')).samples, w = await ctx.getMeasure('w'), v = await ctx.getMeasure('V');
  assert.deepEqual(v.value.shape, [32, 2, 2]);
  for (let a = 0; a < s.length; a++) {
    match(w.samples.slice(2 * a, 2 * a + 2), w.imag.slice(2 * a, 2 * a + 2), [[0, s[a]], [0, -s[a]]], s[a]);
    for (let j = 0; j < 2; j++) {
      const first = 4 * a + j, second = first + 2, k = 2 * a + j;
      const re = (-w.samples[k] * v.samples[first] + w.imag[k] * v.imag[first]) / s[a];
      const im = (-w.samples[k] * v.imag[first] - w.imag[k] * v.samples[first]) / s[a];
      assert.ok(Math.hypot(v.samples[second] - re, v.imag[second] - im) < 2e-13);
    }
  }
});

test('general real eigenvalue extrema follow a dense similarity with known spectrum', async () => {
  // A=S diag(1,2,4) S⁻¹, S=[[1,1,1],[0,1,1],[1,0,1]].
  const { ctx } = makeMatCtx(HEAD + `A=rowstack([[-1,3,2],[-2,4,2],[-3,3,4]])
lo=e.eigmin(A)
hi=e.eigmax(A)`, { sampleCount: 1 });
  assert.ok(Math.abs((await ctx.getMeasure('lo')).samples[0] - 1) < 2e-13);
  assert.ok(Math.abs((await ctx.getMeasure('hi')).samples[0] - 4) < 2e-13);
});

test('cyclic permutation eigenvalues are roots of unity through stalled-shift fallback', async () => {
  const n = 5, rows = Array.from({ length: n }, (_, i) => `[${Array.from({ length: n }, (_, j) => +(j === (i + 1) % n))}]`);
  const { ctx } = makeMatCtx(HEAD + `A=rowstack([${rows}])
f=e.eigen(A)
w=f.values`, { sampleCount: 1 });
  const w = await ctx.getMeasure('w');
  match(w.samples, w.imag, Array.from({ length: n }, (_, k) => [Math.cos(2 * Math.PI * k / n), Math.sin(2 * Math.PI * k / n)]), 1);
});

test('defective and data-derived empty matrices retain their defined eigen records', async () => {
  const { ctx } = makeMatCtx(HEAD + `A=rowstack([[1,1,0],[0,1,1],[0,0,1]])
f=e.eigen(A)
w=f.values
V=f.vectors
n=lengthof(filter(fn(_<0),[1.0]))
empty=e.eigen(fill(0.0,[n,n]))
ew=empty.values`, { sampleCount: 1 });
  const w = await ctx.getMeasure('w'), v = await ctx.getMeasure('V');
  match(w.samples, w.imag, [[1, 0], [1, 0], [1, 0]], 1);
  // This Jordan block has only one independent right eigenvector: e1.
  for (let j = 0; j < 3; j++) {
    assert.ok(Math.abs(Math.hypot(v.samples[j], v.imag[j]) - 1) < 2e-13);
    for (let i = 1; i < 3; i++) assert.ok(Math.hypot(v.samples[3 * i + j], v.imag[3 * i + j]) < 2e-13);
  }
  assert.equal((await ctx.getMeasure('ew')).samples.length, 0);
});
