'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const N = 256;
const logStudent = (x: number, scale = 1) =>
  Math.log(3 / 8) - 2.5 * Math.log1p(x * x / (4 * scale * scale)) - Math.log(scale);

// Independent Student-t4 CDF from integrating (3/8)*(1+x²/4)^(-5/2).
function checkDraws(measure: any, scales: number[], scalar: boolean) {
  assert.deepEqual(measure.value.shape, scalar ? [N] : [N, scales.length]);
  assert.ok(Math.abs(measure.logTotalmass) < 1e-12);
  for (let j = 0; j < scales.length; j++) {
    const pit = Array.from({ length: N }, (_, i) => {
      const x = measure.samples[i * scales.length + j] / scales[j];
      const u = x / Math.sqrt(x * x + 4);
      return 0.5 + (3 * u - u * u * u) / 4;
    }).sort((a, b) => a - b);
    for (let i = 0; i < N; i++) {
      assert.ok(Math.max((i + 1) / N - pit[i], pit[i] - i / N) < 0.13);
    }
  }
}

for (const [op, identity, positive] of [
  ['weighted', 1, 2], ['logweighted', 0, Math.log(2)],
] as Array<[string, number, number]>) {
  test(`${op} callbacks preserve named probability bases through every application shape`, async () => {
    const { ctx } = makeMatCtx(`
base = StudentT(4)
K(w) = normalize(${op}(w, base))
direct ~ K(${identity})
scalar ~ K.(${identity})
rows ~ K.([${identity}, ${identity}])
scaled ~ K.(${positive})
a = logdensityof(lawof(direct), 2)
b = logdensityof(lawof(scalar), 2)
c = logdensityof(lawof(rows), [2, 2])
`, { sampleCount: N, rootSeed: 117 });
    for (const [name, expected] of [['a', logStudent(2)], ['b', logStudent(2)],
      ['c', 2 * logStudent(2)]] as Array<[string, number]>) {
      assert.ok(Math.abs((await ctx.getMeasure(name)).samples[0] - expected) < 1e-12);
    }
    for (const name of ['direct', 'scalar', 'scaled']) {
      checkDraws(await ctx.getMeasure(name), [1], true);
    }
    checkDraws(await ctx.getMeasure('rows'), [1, 1], false);
  });
}

test('the live Student scale guard retains its normalized forward law', async () => {
  const guard = `
guard(s) = normalize(weighted(ifelse((s > 0) && (s < inf), 1, 0),
  locscale(StudentT(4), 0, ifelse((s > 0) && (s < inf), s, 1))))
`;
  for (const [application, scales, scalar] of [
    ['guard(1)', [1], true], ['guard.(2)', [2], true],
    ['guard.([1, 2])', [1, 2], false],
  ] as Array<[string, number[], boolean]>) {
    const { ctx } = makeMatCtx(guard + `
x ~ ${application}
r = logdensityof(lawof(x), ${scalar ? '2' : '[2, 2]'})
`, { sampleCount: N, rootSeed: 117 });
    const expected = scales.reduce((sum, scale) => sum + logStudent(2, scale), 0);
    assert.ok(Math.abs((await ctx.getMeasure('r')).samples[0] - expected) < 1e-12);
    checkDraws(await ctx.getMeasure('x'), scales, scalar);
  }
});
