'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const ext = require('../ext-linalg.ts');
const solves = require('../ext-linalg-solves.ts');
const HEAD = 'e=standard_module("ext-linear-algebra","0.1")\n';

function close(actual: ArrayLike<number>, expected: number[], tolerance = 2e-13) {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) assert.ok(Math.abs(actual[i] - expected[i]) <= tolerance,
    `${i}: ${actual[i]} vs ${expected[i]}`);
}

function mat(rows: number[][]): any {
  return { shape: [rows.length, rows[0]?.length || 0], data: Float64Array.from(rows.flat()) };
}

test('triangular solves preserve RHS columns, selected triangle and implicit diagonal', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[2,999],[3,4]])
B=rowstack([[2,4],[11,18]])
f=e.solve_triangular
X=f(A,B)
Y=f(A=A,b=B,lower=true,unit_diagonal=true)
Z=f(A=transpose(A),b=B,lower=false)
`, { sampleCount: 1 });
  close((await ctx.getMeasure('X')).samples, [1, 2, 2, 3]);
  close((await ctx.getMeasure('Y')).samples, [2, 4, 5, 6]);
  close((await ctx.getMeasure('Z')).samples, [-3.125, -4.75, 2.75, 4.5]);
});

test('factor records and mixed complex RHS remain usable through aliases and exports', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[0,2],[1,3]])
B=rowstack([[complex(4,2),complex(8,4)],[complex(7,2),complex(14,4)]])
f=e.lu(A)
X=e.lu_solve(f,B)
m=load_module("solves.flatppl")
g=m.solve
Y=g(factors=f,b=B)
L=rowstack([[2,0],[complex(1,1),1]])
Z=e.cholesky_solve(L,[complex(4,0),complex(3,2)])
`, { sampleCount: 1, bundle: { sources: { 'solves.flatppl': HEAD + 'solve=e.lu_solve' } } });
  close((await ctx.getMeasure('X')).samples, [1, 2, 2, 4]);
  close((await ctx.getMeasure('X')).imag, [-1, -2, 1, 2]);
  close((await ctx.getMeasure('Y')).samples, [1, 2, 2, 4]);
  // L*y=b gives y=(2,1), Lᴴ*x=y gives x=(1/2+i/2,1).
  close((await ctx.getMeasure('Z')).samples, [.5, 1]);
  close((await ctx.getMeasure('Z')).imag, [.5, 0]);
});

test('factorized live solves reuse each atom with matrix RHS and real Cholesky factors', async () => {
  const { ctx } = makeMatCtx(HEAD + `
t~Normal(0,1)
A=rowstack([[2.0,0.0],[t,3.0]])
f=e.lu(A)
B=rowstack([[2.0,4.0],[t+3.0,2.0*t+6.0]])
X=e.lu_solve(f,B)
L=lower_cholesky(rowstack([[4.0,2.0],[2.0,5.0]]))
Y=e.cholesky_solve(L,[t,1.0])
`, { sampleCount: 24, rootSeed: 1771 });
  const t = (await ctx.getMeasure('t')).samples;
  const x = (await ctx.getMeasure('X')).samples, y = (await ctx.getMeasure('Y')).samples;
  for (let i = 0; i < t.length; i++) {
    close(x.slice(4 * i, 4 * i + 4), [1, 2, 1, 2]);
    close(y.slice(2 * i, 2 * i + 2), [(5 * t[i] - 2) / 16, (4 - 2 * t[i]) / 16]);
  }
});

test('Hermitian eigenpairs have ascending real values and canonical orthonormal columns', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[complex(2,0),complex(0,1)],[complex(0,-1),complex(2,0)]])
f=e.eigh(A)
w=f.values
V=f.vectors
h=e.eigvalsh(adjoint(A))
R=A*V-V*diagmat(w)
G=adjoint(V)*V-eye(2)
`, { sampleCount: 1 });
  close((await ctx.getMeasure('w')).samples, [1, 3]);
  close((await ctx.getMeasure('h')).samples, [1, 3]);
  close((await ctx.getMeasure('R')).samples, [0, 0, 0, 0]);
  close((await ctx.getMeasure('R')).imag, [0, 0, 0, 0]);
  close((await ctx.getMeasure('G')).samples, [0, 0, 0, 0]);
  close((await ctx.getMeasure('G')).imag, [0, 0, 0, 0]);
  const vectors = await ctx.getMeasure('V');
  for (let j = 0; j < 2; j++) {
    let pivot = 0;
    if (Math.hypot(vectors.samples[2 + j], vectors.imag[2 + j])
        > Math.hypot(vectors.samples[j], vectors.imag[j])) pivot = 2;
    assert.ok(vectors.samples[pivot + j] > 0);
    assert.equal(vectors.imag[pivot + j], 0);
  }
});

test('repeated Hermitian eigenspaces retain orthonormal bases at practical scales', () => {
  // I+z*zᴴ with z=(1,i,2) has the independently derived spectrum (1,1,7).
  for (const scale of [1e-100, 1, 1e100]) {
    const A = { shape: [3, 3], data: Float64Array.from([2, 0, 2, 0, 2, 0, 2, 0, 5], x => x * scale),
      im: Float64Array.from([0, -1, 0, 1, 0, 2, 0, -2, 0], x => x * scale) };
    const f = ext._eigh(A);
    close(Array.from(f.values.data, (x: number) => x / scale), [1, 1, 7], 2e-13);
    const V = f.vectors;
    for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {
      let real = 0, imag = 0;
      for (let i = 0; i < 3; i++) {
        real += V.data[i * 3 + p] * V.data[i * 3 + q] + V.im[i * 3 + p] * V.im[i * 3 + q];
        imag += V.data[i * 3 + p] * V.im[i * 3 + q] - V.im[i * 3 + p] * V.data[i * 3 + q];
      }
      close([real, imag], [p === q ? 1 : 0, 0]);
    }
  }
});

test('rank, pseudoinverse and least squares share strict absolute and relative cutoffs', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=diagmat([4.0,1.0,1e-17])
B=eye(3)
r=e.rank(A)
rrel=e.rank(A=A,rtol=0.25)
rabs=e.rank(A=A,rtol=0.0,atol=1.0)
p=e.pinv(A=A,rtol=0.25)
x=e.lstsq(A=A,b=B,rtol=0.0,atol=1.0)
c=e.cond(A)
`, { sampleCount: 1 });
  close((await ctx.getMeasure('r')).samples, [2]);
  close((await ctx.getMeasure('rrel')).samples, [1]);
  close((await ctx.getMeasure('rabs')).samples, [1]);
  close((await ctx.getMeasure('p')).samples, [.25, 0, 0, 0, 0, 0, 0, 0, 0]);
  close((await ctx.getMeasure('x')).samples, [.25, 0, 0, 0, 0, 0, 0, 0, 0]);
  close((await ctx.getMeasure('c')).samples, [4e17], 64);
});

test('wide rank deficient matrix solves give the Moore-Penrose minimum norm', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=rowstack([[1,2,3],[2,4,6]])
B=rowstack([[1,2],[2,4]])
P=e.pinv(A)
X=e.lstsq(A,B)
R=A*P*A-A
G=P*A*P-P
`, { sampleCount: 1 });
  close((await ctx.getMeasure('P')).samples, [1 / 70, 2 / 70, 2 / 70, 4 / 70, 3 / 70, 6 / 70]);
  close((await ctx.getMeasure('X')).samples, [1 / 14, 2 / 14, 2 / 14, 4 / 14, 3 / 14, 6 / 14]);
  close((await ctx.getMeasure('R')).samples, [0, 0, 0, 0, 0, 0]);
  close((await ctx.getMeasure('G')).samples, [0, 0, 0, 0, 0, 0]);
});

test('zero-column and empty cases preserve mathematically defined output shapes', () => {
  const A = mat([[1, 0], [0, 2]]), b = { shape: [2, 0], data: new Float64Array() };
  for (const result of [solves.solveTriangular(A, b), solves.luSolve(ext._lu(A), b),
    solves.choleskySolve(A, b), ext._lstsq(A, b)]) assert.deepEqual(result.shape, [2, 0]);
  const empty = { shape: [0, 3], data: new Float64Array() };
  assert.deepEqual(ext._pinv(empty).shape, [3, 0]);
  assert.deepEqual(ext._lstsq(empty, { shape: [0, 2], data: new Float64Array() }).shape, [3, 2]);
  assert.equal(ext._rank(empty), 0);
  assert.deepEqual(ext._eigh({ shape: [0, 0], data: new Float64Array() }).values.shape, [0]);
  assert.equal(ext._cond(mat([[0, 0], [0, 0]])), Infinity);
  assert.throws(() => ext._cond(empty), /nonempty/);
});

test('ordinary broadcasting preserves factor records, optional cutoffs and matrix RHS cells', async () => {
  const { ctx } = makeMatCtx(HEAD + `
A=diagmat([2.0,4.0])
B=eye(2)
X=e.solve_triangular.([A,A],[B])
F=e.lu.([A,A])
U=e.lu_solve.(F,[B])
P=e.pinv.([A,A])
L=e.lstsq.([A,A],[B])
R=e.rank.(A=[A,A],atol=2.0)
`, { sampleCount: 1 });
  for (const name of ['X', 'U', 'P', 'L']) {
    close((await ctx.getMeasure(name)).samples, [.5, 0, 0, .25, .5, 0, 0, .25]);
  }
  close((await ctx.getMeasure('R')).samples, [1, 1]);
});

test('singular factors fail while near-singular and Hermitian projection paths remain accurate', () => {
  const near = mat([[1, 0], [0, 1e-100]]);
  close(solves.solveTriangular(near, [1, 1e-100]).data, [1, 1]);
  close(solves.luSolve(ext._lu(near), [1, 1e-100]).data, [1, 1]);
  const singular = mat([[1, 0], [0, 0]]);
  assert.throws(() => solves.solveTriangular(singular, [1, 1]), /singular/);
  assert.throws(() => solves.luSolve(ext._lu(singular), [1, 1]), /singular/);
  assert.throws(() => solves.choleskySolve(singular, [1, 1]), /positive/);
  const factors = ext._lu(near);
  factors.P.data[3] = 0;
  assert.throws(() => solves.luSolve(factors, [1, 1]), /permutation/);
  assert.throws(() => ext._eigh(mat([[2, 1], [0, 2]])), /Hermitian/);
  // Accepted skew is projected, rather than selecting one input triangle.
  const perturbation = 16 * Number.EPSILON;
  const f = ext._eigh(mat([[2, 1 + perturbation], [1 - perturbation, 2]]));
  close(f.values.data, [1, 3]);
  const unused = mat([[1, NaN], [2, 3]]);
  close(solves.solveTriangular(unused, [1, 5]).data, [1, 1]);
  assert.throws(() => solves.solveTriangular(unused, [1, 5], false), /finite/);
  assert.throws(() => solves.solveTriangular(mat([[Infinity]]), [1]), /finite/);
  assert.throws(() => ext._eigh(mat([[NaN]])), /finite/);
  assert.throws(() => ext._rank(near, -2), /rtol/);
  assert.throws(() => ext._lstsq(near, [1, 1], -1, -1), /atol/);
});
