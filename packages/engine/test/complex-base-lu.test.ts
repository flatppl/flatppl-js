'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const source = `
A = rowstack([[0.0, complex(1.0, t)], [complex(2.0, -1.0), 1.0]])
b = complex.([3.0, 7.0] + [1.0, 0.0] * t, [-1.0, 2.0] + [3.0, 0.0] * t)
detA = det(A)
logDet = logabsdet(A)
inverse = inv(A)
solution = linsolve(A, b)
`;

function near(value: number, oracle: number) {
  assert.ok(Math.abs(value - oracle) <= 8 * Number.EPSILON * Math.max(1, Math.abs(oracle)),
    `${value} != ${oracle}`);
}

function checkParts(series: any, atom: number, expectedRe: number[], expectedIm: number[]) {
  const width = expectedRe.length;
  expectedRe.forEach((value, j) => near(series.samples[atom * width + j], value));
  expectedIm.forEach((value, j) => near(series.imag[atom * width + j], value));
}

for (const { name, binding, samples } of [
  { name: 'fixed', binding: 't = 1.0', samples: 1 },
  { name: 'sampled', binding: 't ~ Bernoulli(0.5)', samples: 8 },
]) {
  test(`base LU keeps complex determinant, inverse and right-hand side for ${name} matrices`, async () => {
    const { ctx } = makeMatCtx(binding + '\n' + source, { sampleCount: samples, rootSeed: 117 });
    const [determinant, logDeterminant, inverse, solution] = await Promise.all(
      ['detA', 'logDet', 'inverse', 'solution'].map(key => ctx.getMeasure(key)));
    const atoms = (await ctx.getMeasure('t')).samples;
    const atomCount = name === 'fixed' ? 1 : atoms.length;
    if (name === 'sampled') assert.ok(atoms.includes(0) && atoms.includes(1));
    for (let i = 0; i < atomCount; i++) {
      const t = atoms[i];
      // det([[0,a],[c,1]]) = -ac and A⁻¹ = [[-1/(ac),1/c],[1/a,0]].
      const realDet = -2 - t, imagDet = 1 - 2 * t;
      checkParts(determinant, i, [realDet], [imagDet]);
      near(logDeterminant.samples[i], Math.log(Math.hypot(realDet, imagDet)));
      const detSquare = realDet ** 2 + imagDet ** 2;
      checkParts(inverse, i, [realDet / detSquare, 0.4, 1 / (1 + t * t), 0],
        [-imagDet / detSquare, 0.2, -t / (1 + t * t), 0]);
      checkParts(solution, i, [1, 3], [2, -1]);
    }
  });
}

test('a real diagonal system retains a complex right-hand side', async () => {
  const { ctx } = makeMatCtx(`
A = diagmat([2.0, 4.0])
b = [complex(2.0, 6.0), complex(8.0, -4.0)]
solution = linsolve(A, b)
`, { sampleCount: 1 });
  checkParts(await ctx.getMeasure('solution'), 0, [1, 2], [3, -1]);
});

test('complex logabsdet stays finite when the determinant would overflow', async () => {
  const { ctx } = makeMatCtx(`
A = diagmat([complex(1e200, 1e200), complex(1e200, 0.0)])
logDet = logabsdet(A)
`, { sampleCount: 1 });
  const value = await ctx.getMeasure('logDet');
  near(value.samples[0], 400 * Math.log(10) + Math.log(2) / 2);
});
