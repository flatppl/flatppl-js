'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const HEAD = 'e=standard_module("ext-linear-algebra","0.1")\n';
const close = (a: ArrayLike<number>, b: number[], tolerance = 2e-12) => {
  assert.equal(a.length, b.length);
  b.forEach((x, i) => assert.ok(Math.abs(a[i] - x) < tolerance, `${i}: ${a[i]} vs ${x}`));
};

test('complex matrix exponential retains named calls, adjoints and independent SciPy coordinates', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[complex(1,2),complex(2,-1)],[complex(-0.5,3),complex(-1,0.2)]])
exponentiate=e.matexp
E=exponentiate(A=A)
H=e.matexp(adjoint(A))
R=e.matexp(rowstack([[0,0],[0,0]]))
x=linsolve(R,[2.0,3.0])`, { sampleCount: 1 });
  // scipy.linalg.expm (SciPy1.18.1), evaluated independently.
  const re = [-7.212609255998931, -.7526208579151789, -3.6307255913470784, -2.5716733957883156];
  const im = [1.6715339734250143, 3.8480202524356564, -3.906299145655223, .8213404232183272];
  close((await ctx.getMeasure('E')).samples, re);
  close((await ctx.getMeasure('E')).imag, im);
  close((await ctx.getMeasure('H')).samples, [re[0], re[2], re[1], re[3]]);
  close((await ctx.getMeasure('H')).imag, [-im[0], -im[2], -im[1], -im[3]]);
  close((await ctx.getMeasure('x')).samples, [2, 3]);
});

test('complex diagonal, nilpotent and skew-Hermitian exponentials follow closed forms', async () => {
  const { ctx } = makeMatCtx(HEAD + `
D=e.matexp(rowstack([[complex(1,0.5),complex(0,0)],[complex(0,0),complex(-2,-1)]]))
N=e.matexp(rowstack([[complex(0,0),complex(2,3)],[complex(0,0),complex(0,0)]]))
S=e.matexp(rowstack([[complex(0,0),complex(0,0.7)],[complex(0,0.7),complex(0,0)]]))
n=lengthof(filter(fn(_<0),[1.0]))
empty=e.matexp(fill(complex(0,0),[n,n]))`, { sampleCount: 1 });
  close((await ctx.getMeasure('D')).samples, [Math.exp(1) * Math.cos(.5), 0, 0, Math.exp(-2) * Math.cos(1)]);
  close((await ctx.getMeasure('D')).imag, [Math.exp(1) * Math.sin(.5), 0, 0, -Math.exp(-2) * Math.sin(1)]);
  close((await ctx.getMeasure('N')).samples, [1, 2, 0, 1]);
  close((await ctx.getMeasure('N')).imag, [0, 3, 0, 0]);
  close((await ctx.getMeasure('S')).samples, [Math.cos(.7), 0, 0, Math.cos(.7)]);
  close((await ctx.getMeasure('S')).imag, [0, Math.sin(.7), Math.sin(.7), 0]);
  assert.equal((await ctx.getMeasure('empty')).samples.length, 0);
});

test('complex matrix exponential evaluates every live scalar phase', async () => {
  const { ctx } = makeMatCtx(HEAD + `t~Normal(0,3)
A=rowstack([[complex(0,t),complex(0,0)],[complex(0,0),complex(0,-t)]])
E=e.matexp(A)`, { sampleCount: 64, rootSeed: 117 });
  const t = (await ctx.getMeasure('t')).samples, e = await ctx.getMeasure('E');
  for (let i = 0; i < t.length; i++) {
    close(e.samples.slice(4 * i, 4 * i + 4), [Math.cos(t[i]), 0, 0, Math.cos(t[i])]);
    close(e.imag.slice(4 * i, 4 * i + 4), [Math.sin(t[i]), 0, 0, -Math.sin(t[i])]);
  }
});
