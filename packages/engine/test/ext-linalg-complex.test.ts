'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const HEAD = 'e=standard_module("ext-linear-algebra","0.1")\n';

function close(actual: ArrayLike<number>, expected: number[], tolerance = 2e-13) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance,
    `${i}: ${actual[i]} vs ${expected[i]}`);
}

test('complex LU retains pivoted factors, named arguments and adjoint views', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[complex(1.0,1.0),complex(2.0,0.0)],[complex(2.0,0.0),complex(0.0,3.0)]])
lu=e.lu
f=lu(A=A)
L=f.L
U=f.U
g=e.lu(adjoint(A))
UL=g.U
residual=f.P*A-L*U
`, { sampleCount: 1 });
  // Hand elimination: swap rows, multiplier=(1+i)/2, U22=3.5-1.5i.
  close((await ctx.getMeasure('L')).samples, [1, 0, .5, 1]);
  close((await ctx.getMeasure('L')).imag, [0, 0, .5, 0]);
  close((await ctx.getMeasure('U')).samples, [2, 0, 0, 3.5]);
  close((await ctx.getMeasure('U')).imag, [0, 3, 0, -1.5]);
  close((await ctx.getMeasure('UL')).imag, [0, -3, 0, 1.5]);
  close((await ctx.getMeasure('residual')).samples, [0, 0, 0, 0]);
  close((await ctx.getMeasure('residual')).imag, [0, 0, 0, 0]);
});

test('complex LU uses each live matrix and its own magnitude pivot', async () => {
  const { ctx } = makeMatCtx(HEAD + `
t~Normal(0,2)
A=rowstack([[complex(1.0,t),complex(2.0,0.0)],[complex(2.0,0.0),complex(0.0,3.0)]])
f=e.lu(A)
L=f.L
U=f.U
`, { sampleCount: 64, rootSeed: 117 });
  const t = (await ctx.getMeasure('t')).samples;
  const l = await ctx.getMeasure('L'), u = await ctx.getMeasure('U');
  let swapped = 0;
  for (let i = 0; i < t.length; i++) {
    const x = t[i], pivot = Math.hypot(1, x) <= 2;
    const denominator = 1 + x * x;
    if (pivot) swapped++;
    const lr = pivot ? .5 : 2 / denominator, li = pivot ? x / 2 : -2 * x / denominator;
    close(l.samples.slice(4 * i, 4 * i + 4), [1, 0, lr, 1]);
    close(l.imag.slice(4 * i, 4 * i + 4), [0, 0, li, 0]);
    const ur = pivot ? 2 + 1.5 * x : -4 / denominator;
    const ui = pivot ? -1.5 : 3 + 4 * x / denominator;
    close([u.samples[4 * i + 3], u.imag[4 * i + 3]], [ur, ui]);
  }
  assert.ok(swapped > 0 && swapped < t.length);
});

test('integer and singular LU factors remain usable by real matrix consumers', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[0,1,2],[0,2,4],[0,3,7]])
f=e.lu(A)
R=f.P*A-f.L*f.U
x=linsolve(f.L,[1.0,2.0,3.0])
F=e.lu.([A,A])
y=linsolve(F[1].L,[1.0,2.0,3.0])
lower=fn(e.lu(_).L)
z=linsolve(lower(A),[1.0,2.0,3.0])
`, { sampleCount: 1 });
  close((await ctx.getMeasure('R')).samples, new Array(9).fill(0));
  close((await ctx.getMeasure('x')).samples, [1, 2, 5 / 3]);
  close((await ctx.getMeasure('y')).samples, [1, 2, 5 / 3]);
  close((await ctx.getMeasure('z')).samples, [1, 2, 5 / 3]);
});

test('complex singular LU continues past a zero column and retains later pivots', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[complex(0,0),complex(1,1),complex(2,0)],
  [complex(0,0),complex(2,0),complex(4,0)],
  [complex(0,0),complex(3,0),complex(0,7)]])
f=e.lu(A)
P=f.P
L=f.L
U=f.U`, { sampleCount: 1 });
  close((await ctx.getMeasure('P')).samples, [1, 0, 0, 0, 0, 1, 0, 1, 0]);
  close((await ctx.getMeasure('L')).samples, [1, 0, 0, 0, 1, 0, 0, 2 / 3, 1]);
  close((await ctx.getMeasure('U')).samples, [0, 1, 2, 0, 3, 0, 0, 0, 4]);
  close((await ctx.getMeasure('U')).imag, [0, 1, 0, 0, 0, 7, 0, 0, -14 / 3]);
});

test('re-exported numerical functions retain real result specialization', async () => {
  const { ctx } = makeMatCtx(`
m=load_module("factors.flatppl")
alias=m.lu
A=rowstack([[4.0,3.0],[2.0,1.0]])
f=m.lu(A)
g=alias(A)
x=linsolve(f.L,[1.0,2.0])
y=linsolve(g.L,[1.0,2.0])`, { sampleCount: 1, bundle: { sources: {
    'factors.flatppl': HEAD + 'lu=e.lu',
  } } });
  // L21=1/2, so forward substitution gives (1, 3/2).
  close((await ctx.getMeasure('x')).samples, [1, 1.5]);
  close((await ctx.getMeasure('y')).samples, [1, 1.5]);
});
