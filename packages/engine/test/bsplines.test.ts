'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const source = 'sp = standard_module("splines", "0.1")\n';
const close = (a: number, b: number, tolerance = 3e-12) =>
  assert.ok(Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(b)), `${a} != ${b}`);
function evaluate(expression: string): any {
  const result = processSource(source + 'r = ' + expression);
  assert.deepEqual(result.diagnostics.filter((d: any) => d.severity === 'error'), []);
  return sampler.evaluateExpr(result.loweredModule.bindings.get('r').rhs,
    { __moduleRegistry: result.loweredModule.moduleRegistry });
}
const rowClose = (a: Float64Array, b: number[]) => {
  assert.equal(a.length, b.length); a.forEach((v, i) => close(v, b[i]));
};

// A degree-2 clamped basis is the Bernstein basis ((1-x)^2,2x(1-x),x^2).
// Its first/second derivatives and their exact polynomial integrals provide
// an analytic oracle independent of the Cox-de Boor implementation.
test('B-spline values, derivatives, signed integrals and penalties match polynomials', () => {
  const t = '[0,0,0,1,1,1]';
  rowClose(evaluate(`sp.bspline_basis(.25,${t},2)`).data, [9 / 16, 3 / 8, 1 / 16]);
  rowClose(evaluate(`sp.bspline_derivative(.25,${t},2,1)`).data, [-1.5, 1, .5]);
  rowClose(evaluate(`sp.bspline_derivative(.25,${t},2)`).data, [-1.5, 1, .5]);
  rowClose(evaluate(`sp.bspline_derivative(.25,${t},2,2)`).data, [2, -4, 2]);
  rowClose(evaluate(`sp.bspline_derivative(.25,${t},2,3)`).data, [0, 0, 0]);
  rowClose(evaluate(`sp.bspline_integral(-1,2,${t},2)`).data, [1 / 3, 1 / 3, 1 / 3]);
  rowClose(evaluate(`sp.bspline_integral(1,0,${t},2)`).data, [-1 / 3, -1 / 3, -1 / 3]);
  rowClose(evaluate(`sp.bspline_integral(.5,.5,${t},2)`).data, [0, 0, 0]);
  rowClose(evaluate(`sp.bspline_penalty(${t},2,2)`).data, [4, -8, 4, -8, 16, -8, 4, -8, 4]);
  rowClose(evaluate(`sp.bspline_penalty(${t},2)`).data, [4, -8, 4, -8, 16, -8, 4, -8, 4]);
  rowClose(evaluate(`sp.bspline_penalty(${t},2,0)`).data,
    [1 / 5, 1 / 10, 1 / 30, 1 / 10, 2 / 15, 1 / 10, 1 / 30, 1 / 10, 1 / 5]);
  assert.throws(() => evaluate(`sp.bspline_penalty(${t},2,3)`), /order must not exceed/);
});

test('B-spline repeated knots and endpoint derivatives match independent SciPy values', () => {
  const t = '[0,0,0,0,.7,1.1,1.1,2,3,3,3,3]';
  rowClose(evaluate(`sp.bspline_basis(1.1,${t},3)`).data, [0, 0, 0, 9 / 13, 4 / 13, 0, 0, 0]);
  rowClose(evaluate(`sp.bspline_derivative(1.7,${t},3,1)`).data,
    [0, 0, 0, -.2564102564102565, -.8146885432203993, .7386888273314869, .33240997229916885, 0]);
  rowClose(evaluate(`sp.bspline_integral(.2,2.7,${t},3)`).data,
    [.04555393586005829, .21362130448401317, .26629701467363803, .49952774498229036,
      .5744390581717451, .4629771468144044, .3775587950138504, .06002500000000006]);
  const p = evaluate(`sp.bspline_penalty(${t},3,2)`);
  assert.deepEqual(p.shape, [8, 8]);
  rowClose(p.data.slice(24, 32), [7.083825265643446, -10.898192716374531, -38.14367450731085,
    84.76372112735746, -46.925289937125726, 2.841110874351872, 1.2784998934583423, 0]);
  rowClose(evaluate(`sp.bspline_basis(0,${t},3)`).data, [1, 0, 0, 0, 0, 0, 0, 0]);
  rowClose(evaluate(`sp.bspline_basis(3,${t},3)`).data, [0, 0, 0, 0, 0, 0, 0, 1]);
  rowClose(evaluate(`sp.bspline_derivative(3,${t},3,1)`).data, [0, 0, 0, 0, 0, 0, -3, 3]);
  rowClose(evaluate(`sp.bspline_basis(3.1,${t},3)`).data, Array(8).fill(0));
});

test('degree-zero jumps use the right interior interval and the left upper endpoint', () => {
  rowClose(evaluate('sp.bspline_basis(1,[0,1,2],0)').data, [0, 1]);
  rowClose(evaluate('sp.bspline_basis(2,[0,1,2],0)').data, [0, 1]);
  rowClose(evaluate('sp.bspline_derivative(1,[0,1,2],0,1)').data, [0, 0]);
  rowClose(evaluate('sp.bspline_penalty([0,1,2],0,0)').data, [1, 0, 0, 1]);
  // Full interior multiplicity permits a discontinuity. Ordinary derivatives
  // exclude its point mass; zero-width intervals still contribute nothing.
  rowClose(evaluate('sp.bspline_basis(1,[0,0,1,1,2,2],1)').data, [0, 0, 1, 0]);
  rowClose(evaluate('sp.bspline_basis(4,[0,1,2,3,4,5],1)').data, [0, 0, 0, 1]);
});

test('higher-degree basis partition and exact support integrals retain practical scales', () => {
  for (const degree of [3, 9, 21]) {
    const knots = [...Array(degree + 1).fill(-200), -70, 30, 100, ...Array(degree + 1).fill(300)];
    const literal = '[' + knots + ']';
    const b = evaluate(`sp.bspline_basis(17,${literal},${degree})`).data;
    close(b.reduce((a: number, v: number) => a + v, 0), 1);
    const d = evaluate(`sp.bspline_derivative(17,${literal},${degree},1)`).data;
    close(d.reduce((a: number, v: number) => a + v, 0), 0);
    const integral = evaluate(`sp.bspline_integral(-300,400,${literal},${degree})`).data;
    integral.forEach((v: number, i: number) => close(v, (knots[i + degree + 1] - knots[i]) / (degree + 1)));
  }
  for (const scale of [1e-8, 1e8]) {
    const t = `[0,0,0,${scale},${scale},${scale}]`;
    rowClose(evaluate(`sp.bspline_basis(${scale / 4},${t},2)`).data, [9 / 16, 3 / 8, 1 / 16]);
    close(evaluate(`sp.bspline_derivative(${scale / 4},${t},2)`).data[0] * scale, -1.5);
    close(evaluate(`sp.bspline_integral(0,${scale},${t},2)`).data[0] / scale, 1 / 3);
    close(evaluate(`sp.bspline_penalty(${t},2)`).data[0] * scale ** 3, 4);
  }
});

test('named aliases, broadcast and live knots preserve vector and matrix cells', async () => {
  const { ctx } = makeMatCtx(source + `
f = sp.bspline_basis
row(x) = f(degree = 2, knots = [0,0,0,1,1,1], x = x)
a = row.([0,.5,1])
x ~ Uniform(interval(0,1))
scale ~ Uniform(interval(1,2))
b = f(x * scale,[0,0,0,scale,scale,scale],2)
p = sp.bspline_penalty([0,0,0,scale,scale,scale],2,2)
`, { sampleCount: 16, rootSeed: 839 });
  const rows = (await ctx.getMeasure('a')).elems;
  [[1, 0, 0], [.25, .5, .25], [0, 0, 1]].forEach((r, i) => rowClose(rows[i].samples, r));
  const x = (await ctx.getMeasure('x')).samples, scale = (await ctx.getMeasure('scale')).samples;
  const b = (await ctx.getMeasure('b')).value, p = (await ctx.getMeasure('p')).value;
  assert.deepEqual(b.shape, [16, 3]); assert.deepEqual(p.shape, [16, 3, 3]);
  for (let i = 0; i < 16; i++) {
    rowClose(b.data.slice(i * 3, i * 3 + 3), [(1 - x[i]) ** 2, 2 * x[i] * (1 - x[i]), x[i] ** 2]);
    close(p.data[i * 9], 4 / scale[i] ** 3);
  }
});

test('basis derivatives match a numerical value gradient away from knot branches', () => {
  const t = '[0,0,0,0,.7,1.1,1.1,2,3,3,3,3]', h = 1e-5;
  const hi = evaluate(`sp.bspline_basis(${1.7 + h},${t},3)`).data;
  const lo = evaluate(`sp.bspline_basis(${1.7 - h},${t},3)`).data;
  const derivative = evaluate(`sp.bspline_derivative(1.7,${t},3,1)`).data;
  derivative.forEach((v: number, i: number) => close((hi[i] - lo[i]) / (2 * h), v, 2e-9));
  const integralHi = evaluate(`sp.bspline_integral(.2,${1.7 + h},${t},3)`).data;
  const integralLo = evaluate(`sp.bspline_integral(.2,${1.7 - h},${t},3)`).data;
  const basis = evaluate(`sp.bspline_basis(1.7,${t},3)`).data;
  basis.forEach((v: number, i: number) => close((integralHi[i] - integralLo[i]) / (2 * h), v, 2e-9));
});

test('B-spline domains reject degenerate knot vectors and non-L2 penalties', () => {
  for (const expression of ['sp.bspline_basis(0,[0,0,0,1,1],1)',
    'sp.bspline_basis(0,[0,0,2,1,3,3],2)', 'sp.bspline_basis(0,[0,0,0,0],1)',
    'sp.bspline_basis(0,[0,0,inf,inf],1)', 'sp.bspline_basis(inf,[0,0,1,1],1)',
    'sp.bspline_penalty([0,0,1,1,2,2],1,1)']) {
    assert.throws(() => evaluate(expression), /splines:/);
  }
});

test('B-spline degree and derivative order are fixed controls', () => {
  for (const expression of ['sp.bspline_basis(.2,[0,0,0,1,1,1],p)',
    'sp.bspline_derivative(.2,[0,0,0,1,1,1],2,p)']) {
    const result = processSource(source + 'p = elementof(nonnegintegers)\nr = ' + expression);
    assert.ok(result.diagnostics.some((d: any) => d.severity === 'error' && /fixed/.test(d.message)));
  }
});
