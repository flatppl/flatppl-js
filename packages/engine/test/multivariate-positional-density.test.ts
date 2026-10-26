'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const density = require('../density.ts');

async function score(expression: string): Promise<number> {
  const { proc, ctx } = ctxFor(`alpha = [2, 3]\nI = rowstack([[1, 0], [0, 1]])\nlp = ${expression}`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  return (await ctx.getMeasure('lp')).value.data[0];
}

// §04 binds positional parameters in §08's declared order. These closed-form
// values check that vector/matrix parameters stay whole and reach the density.
test('multivariate density accepts positional parameters through the public API', async () => {
  const cases: [string, string, number][] = [
    ['Dirichlet(alpha)', '[0.25, 0.75]', Math.log(12 * 0.25 * 0.75 ** 2)],
    ['Multinomial(3, [0.25, 0.75])', '[1, 2]', Math.log(3 * 0.25 * 0.75 ** 2)],
    ['MvNormal([0, 0], I)', '[0, 0]', -Math.log(2 * Math.PI)],
    ['Wishart(3, I)', 'I', -1 - Math.log(4 * Math.PI)],
    ['InverseWishart(3, I)', 'I', -1 - Math.log(4 * Math.PI)],
    ['LKJ(2, 1)', 'I', -Math.LN2],
    ['LKJCholesky(2, 1)', 'I', -Math.LN2],
  ];
  for (const [measure, point, expected] of cases) {
    const actual = await score(`logdensityof(${measure}, ${point})`);
    assert.ok(Math.abs(actual - expected) < 1e-12, `${measure}: ${actual} != ${expected}`);
  }
});

test('matrix density preserves analytic values for structured observations', async () => {
  // §08, n = 2 and nu = 3: the common normalizer is 4π. For the
  // inverse Wishart at diag(1, 2), |X| = 2 and tr(inv(X)) = 3/2.
  assert.ok(Math.abs(await score('logdensityof(Wishart(3, I), eye(2))')
    - (-1 - Math.log(4 * Math.PI))) < 1e-12);
  assert.ok(Math.abs(await score('logdensityof(InverseWishart(3, I), diagmat([1, 2]))')
    - (-3 * Math.LN2 - 0.75 - Math.log(4 * Math.PI))) < 1e-12);
});

test('Dirichlet call forms agree with the analytic density, including the FlatPDL primitive', async () => {
  for (const input of ['[2, 3]', 'alpha', 'record(alpha = alpha)']) {
    const expression = `builtin_logdensityof(Dirichlet, ${input}, [0.25, 0.75])`;
    assert.ok(Math.abs(await score(expression) - Math.log(1.6875)) < 1e-12);
  }
  assert.ok(Math.abs(await score('logdensityof(Dirichlet(alpha = alpha), [0.25, 0.75])')
    - Math.log(1.6875)) < 1e-12);
  assert.equal(await score('logdensityof(Dirichlet(alpha), [0.25, 0.5])'), -Infinity);
  // The simplex has one free coordinate, hence one factor of 2 in the volume.
  assert.ok(Math.abs(await score(
    'logdensityof(pushfwd(x -> 2 .* x .+ 3, Dirichlet([2, 3])), [3.5, 4.5])')
    - Math.log(1.6875 / 2)) < 1e-12);
});

test('positional multivariate parameters stay live across density atoms', () => {
  const ir = { kind: 'call', op: 'Dirichlet', args: [{ kind: 'ref', ns: 'self', name: 'alpha' }] };
  const alpha = { shape: [3, 2], data: new Float64Array([2, 3, 3, 2, 1, 1]) };
  const actual = density.logDensityN(ir, [0.25, 0.75], { alpha }, 3, {});
  const expected = [Math.log(12 * 0.25 * 0.75 ** 2), Math.log(12 * 0.25 ** 2 * 0.75), 0];
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 1e-12, `atom ${i}`);
  }
});

test('multivariate density refuses surplus positional arguments', async () => {
  const { proc, ctx } = ctxFor('lp = logdensityof(Dirichlet([2, 3], [4, 5]), [0.25, 0.75])', 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error').map((d: any) => d.message),
    ['Dirichlet expects at most 1 positional argument(s), got 2']);
  await assert.rejects(() => ctx.getMeasure('lp'), /Dirichlet: expected at most 1 positional arguments, got 2/);
});
