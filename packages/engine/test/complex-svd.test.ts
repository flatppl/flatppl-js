'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const HEAD = 'e=standard_module("ext-linear-algebra","0.1")\n';
const MATRIX = 'rowstack([[complex(1,2),complex(2,0)],[complex(3,0),complex(4,1)],[complex(5,0),complex(6,-1)]])';
const close = (a: ArrayLike<number>, b: number[], tolerance = 2e-12) => {
  assert.equal(a.length, b.length);
  b.forEach((x, i) => assert.ok(Math.abs(a[i] - x) < tolerance, `${i}: ${a[i]} vs ${x}`));
};
const shape = (m: any) => m.intrinsicShape || m.value.shape;

test('complex tall and wide SVD/least squares execute independent singular-value and solution oracles', async () => {
  const { ctx } = makeMatCtx(HEAD + `A=${MATRIX}
f=e.svd(A)
S=f.S
solve=e.lstsq
x=solve(A=A,b=[complex(-2,7),complex(-1,10.5),complex(4,17.5)])
W=adjoint(A)
g=e.svd(W)
U=g.U
V=g.V
wideS=g.S
y=e.lstsq(W,[complex(1,-1),complex(2,3)])`, { sampleCount: 1 });
  // scipy.linalg.svdvals and lstsq (SciPy1.18.1), independently computed.
  const singular = [9.690820544135475, 1.7572698089257215];
  close((await ctx.getMeasure('S')).samples, singular);
  close((await ctx.getMeasure('wideS')).samples, singular);
  close((await ctx.getMeasure('x')).samples, [1, -.5]);
  close((await ctx.getMeasure('x')).imag, [1, 2]);
  close((await ctx.getMeasure('y')).samples, [1.2551724137931035, -.6448275862068965, .28620689655172393]);
  close((await ctx.getMeasure('y')).imag, [.12413793103448426, .39310344827586174, .0413793103448274]);
  assert.deepEqual(shape(await ctx.getMeasure('U')), [2, 2]);
  assert.deepEqual(shape(await ctx.getMeasure('V')), [3, 2]);
});

test('real wide SVD retains economy shapes and real consumers', async () => {
  const { ctx } = makeMatCtx(HEAD + `A=rowstack([[3,0,0],[0,4,0]])
f=e.svd(A)
U=f.U
V=f.V
S=f.S
x=e.lstsq(A,[6.0,8.0])
tx=e.lstsq((A,[6.0,8.0]))
score=logdensityof(Normal(tx[1],1.0),2.0)
y=linsolve(U,U*[1.0,2.0])
r=e.rank(A)`, { sampleCount: 1 });
  close((await ctx.getMeasure('S')).samples, [4, 3]);
  close((await ctx.getMeasure('x')).samples, [2, 2, 0]);
  close((await ctx.getMeasure('tx')).samples, [2, 2, 0]);
  close((await ctx.getMeasure('score')).samples, [-Math.log(2 * Math.PI) / 2]);
  close((await ctx.getMeasure('y')).samples, [1, 2]);
  assert.deepEqual(shape(await ctx.getMeasure('U')), [2, 2]);
  assert.deepEqual(shape(await ctx.getMeasure('V')), [3, 2]);
  assert.equal((await ctx.getMeasure('r')).samples[0], 2);
});

test('complex rank deficiency retains unitary null columns and minimum-norm solutions', async () => {
  const { ctx } = makeMatCtx(HEAD + `A=rowstack([[complex(1,0),complex(0,-2)],[complex(0,1),complex(2,0)]])
f=e.svd(A)
S=f.S
GU=adjoint(f.U)*f.U
GV=adjoint(f.V)*f.V
x=e.lstsq(A,[complex(1,0),complex(0,1)])
r=e.rank(A)
tiny=e.svd(rowstack([[complex(1e200,0),complex(0,0)],[complex(0,0),complex(0,1e-200)]]))
TS=tiny.S`, { sampleCount: 1 });
  close((await ctx.getMeasure('S')).samples, [Math.sqrt(10), 0]);
  close((await ctx.getMeasure('x')).samples, [.2, 0]);
  close((await ctx.getMeasure('x')).imag, [0, .4]);
  assert.equal((await ctx.getMeasure('r')).samples[0], 1);
  for (const name of ['GU', 'GV']) {
    close((await ctx.getMeasure(name)).samples, [1, 0, 0, 1]);
    close((await ctx.getMeasure(name)).imag, [0, 0, 0, 0]);
  }
  const values = (await ctx.getMeasure('TS')).samples;
  assert.ok(Math.abs(values[0] / 1e200 - 1) < 2e-13);
  assert.ok(Math.abs(values[1] / 1e-200 - 1) < 2e-13);
});

test('zero and computed-empty complex matrices retain defined factors and minimum-norm zeros', async () => {
  const { ctx } = makeMatCtx(HEAD + `Z=fill(complex(0,0),[3,2])
f=e.svd(Z)
S=f.S
G=adjoint(f.U)*f.U
x=e.lstsq(Z,[complex(1,1),complex(2,-1),complex(3,0)])
r=e.rank(Z)
n=lengthof(filter(fn(_<0),[1.0]))
empty=fill(complex(0,0),[n,2])
g=e.svd(empty)
ES=g.S
y=e.lstsq(empty,fill(complex(0,0),n))`, { sampleCount: 1 });
  close((await ctx.getMeasure('S')).samples, [0, 0]);
  close((await ctx.getMeasure('G')).samples, [1, 0, 0, 1]);
  close((await ctx.getMeasure('G')).imag, [0, 0, 0, 0]);
  close((await ctx.getMeasure('x')).samples, [0, 0]);
  close((await ctx.getMeasure('y')).samples, [0, 0]);
  assert.equal((await ctx.getMeasure('r')).samples[0], 0);
  assert.equal((await ctx.getMeasure('ES')).samples.length, 0);
});

test('mixed real/complex least-squares inputs retain their independent coordinates', async () => {
  const { ctx } = makeMatCtx(HEAD + `
x=e.lstsq(rowstack([[1,0],[0,2],[0,0]]),[complex(1,1),complex(2,-1),complex(0,3)])
y=e.lstsq(rowstack([[complex(0,1),complex(0,0)],[complex(0,0),complex(0,2)]]),[2.0,4.0])`,
  { sampleCount: 1 });
  close((await ctx.getMeasure('x')).samples, [1, 1]);
  close((await ctx.getMeasure('x')).imag, [1, -.5]);
  close((await ctx.getMeasure('y')).samples, [0, 0]);
  close((await ctx.getMeasure('y')).imag, [-2, -2]);
});

test('live complex SVD and solves follow each atom including changes of rank', async () => {
  const { ctx } = makeMatCtx(HEAD + `t~Bernoulli(0.5)
A=rowstack([[complex(1,1),complex(0,0)],[complex(0,0),complex(t,0)],[complex(0,0),complex(0,0)]])
f=e.svd(A)
S=f.S
x=e.lstsq(A,[complex(1,1),complex(2,0),complex(0,0)])
r=e.rank(A)`, { sampleCount: 32, rootSeed: 117 });
  const t = (await ctx.getMeasure('t')).samples, s = (await ctx.getMeasure('S')).samples;
  const x = await ctx.getMeasure('x'), r = (await ctx.getMeasure('r')).samples;
  for (let i = 0; i < t.length; i++) {
    close(s.slice(2 * i, 2 * i + 2), [Math.SQRT2, t[i]]);
    close(x.samples.slice(2 * i, 2 * i + 2), [1, 2 * t[i]]);
    close(x.imag.slice(2 * i, 2 * i + 2), [0, 0]);
    assert.equal(r[i], 1 + t[i]);
  }
});
