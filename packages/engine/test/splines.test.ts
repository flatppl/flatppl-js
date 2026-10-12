'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');

const moduleSource = 'sp = standard_module("splines", "0.1")\n';
const close = (actual: number, expected: number, tolerance = 1e-12) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} != ${expected}`);

test('thin-plate module call forms preserve radial and affine coordinates', async () => {
  const { ctx } = makeMatCtx(moduleSource + `
centers = [-1, 0, 1]
basis = sp.thinplate_basis
row(x) = basis(centers = centers, x = x)
B = row.([-2, -1, 0, 1, 2])
P = sp.thinplate_penalty(centers = centers)
b = sp.thinplate_basis(record(centers = [2, 2, -1], x = 1))
`, { sampleCount: 1 });
  const basis = (await ctx.getMeasure('B')).elems;
  assert.equal(basis.length, 5);
  const expected = [
    [1 / 12, 2 / 3, 9 / 4, 1, -2],
    [0, 1 / 12, 2 / 3, 1, -1],
    [1 / 12, 0, 1 / 12, 1, 0],
    [2 / 3, 1 / 12, 0, 1, 1],
    [9 / 4, 2 / 3, 1 / 12, 1, 2],
  ];
  expected.forEach((row, i) => {
    assert.deepEqual(basis[i].value.shape, [5]);
    row.forEach((value, j) => close(basis[i].samples[j], value));
  });
  const penalty = await ctx.getMeasure('P');
  assert.deepEqual(penalty.intrinsicShape, [5, 5]);
  const radial = [[0, 1 / 12, 2 / 3], [1 / 12, 0, 1 / 12], [2 / 3, 1 / 12, 0]];
  for (let i = 0; i < 5; i++) {
    for (let j = 0; j < 5; j++) close(penalty.samples[5 * i + j], i < 3 && j < 3 ? radial[i][j] : 0);
  }
  const duplicate = (await ctx.getMeasure('b')).samples;
  [1 / 12, 1 / 12, 2 / 3, 1, 1].forEach((value, i) => close(duplicate[i], value));
});

test('thin-plate values, derivatives and constrained roughness match the piecewise cubic', async () => {
  const score = async (x: number, centers = [-1, 0, 1]) => {
    const { ctx } = makeMatCtx(moduleSource + `
centers = [${centers}]
beta = [1, -2, 1, .25, -.5]
y = sum(beta .* sp.thinplate_basis(${x}, centers))
energy = quadform(sp.thinplate_penalty(centers), beta)
`, { sampleCount: 1 });
    return [(await ctx.getMeasure('y')).samples[0], (await ctx.getMeasure('energy')).samples[0]];
  };
  // The second derivative is 1-|x| on [-1,1], zero outside; its squared
  // integral is 2/3. Values and first derivatives follow piecewise integration.
  const values = [2.25, 1.25, 5 / 12, .25, .25], derivatives = [-1, -1, -.5, 0, 0];
  const h = 1e-5;
  for (let i = 0; i < 5; i++) {
    const x = i - 2, actual = await score(x);
    close(actual[0], values[i]);
    close(actual[1], 2 / 3);
    close(((await score(x + h))[0] - (await score(x - h))[0]) / (2 * h), derivatives[i], 1e-8);
  }
  for (let i = 0; i < 3; i++) {
    const lo = [-1, 0, 1], hi = lo.slice();
    lo[i] -= h; hi[i] += h;
    close(((await score(.5, hi))[0] - (await score(.5, lo))[0]) / (2 * h),
      [-.5625, .125, .0625][i], 1e-8);
  }
});

test('thin-plate functions retain each sampled scalar and complete centers vector', async () => {
  const { ctx } = makeMatCtx(moduleSource + `
x ~ Normal(0, 1)
centers ~ iid(Normal(0, 1), 3)
b = sp.thinplate_basis(x, centers)
p = sp.thinplate_penalty(centers)
`, { sampleCount: 32, rootSeed: 117 });
  const x = (await ctx.getMeasure('x')).samples;
  const centers = (await ctx.getMeasure('centers')).value;
  const basis = (await ctx.getMeasure('b')).value;
  const penalty = (await ctx.getMeasure('p')).value;
  assert.deepEqual(basis.shape, [32, 5]);
  assert.deepEqual(penalty.shape, [32, 5, 5]);
  for (let atom = 0; atom < 32; atom++) {
    for (let i = 0; i < 3; i++) {
      const c = centers.data[atom * 3 + i];
      close(basis.data[atom * 5 + i], Math.abs(x[atom] - c) ** 3 / 12);
      for (let j = 0; j < 3; j++) {
        close(penalty.data[atom * 25 + i * 5 + j],
          Math.abs(c - centers.data[atom * 3 + j]) ** 3 / 12);
      }
    }
    close(basis.data[atom * 5 + 3], 1);
    close(basis.data[atom * 5 + 4], x[atom]);
  }
});

test('thin-plate module rejects arguments outside its scalar and nonempty vector domains', () => {
  for (const call of ['sp.thinplate_basis([1,2], [0,1])',
    'sp.thinplate_basis(im, [0,1])', 'sp.thinplate_penalty(eye(2))',
    'sp.thinplate_penalty([[0,1],[2,3]])', 'sp.thinplate_penalty([0,im])']) {
    assert.ok(processSource(moduleSource + 'r = ' + call).diagnostics
      .some((d: any) => d.severity === 'error'), call);
  }
  for (const call of ['sp.thinplate_basis(0, [])', 'sp.thinplate_penalty([])']) {
    const result = processSource(moduleSource + 'r = ' + call);
    assert.throws(() => sampler.evaluateExpr(result.loweredModule.bindings.get('r').rhs,
      { __moduleRegistry: result.loweredModule.moduleRegistry }), /nonempty real vector/);
  }
});
