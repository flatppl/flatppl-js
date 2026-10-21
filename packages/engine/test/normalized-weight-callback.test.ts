'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

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
  ['logweighted', 0, 1000], ['logweighted', 0, -1000],
] as Array<[string, number, number]>) {
  test(`${op}(${positive}) callbacks preserve normalized laws through every application shape`, async () => {
    const { ctx } = makeMatCtx(`
unit_base = StudentT(4)
K(w) = normalize(${op}(w, unit_base))
direct ~ K(${positive})
scalar ~ K.(${positive})
rows ~ K.([${identity}, ${positive}])
a = logdensityof(lawof(direct), 2)
b = logdensityof(lawof(scalar), 2)
c = logdensityof(lawof(rows), [2, 2])
lo = logdensityof(K(${positive - 1e-5}), 2)
hi = logdensityof(K(${positive + 1e-5}), 2)
`, { sampleCount: N, rootSeed: 117 });
    for (const [name, expected] of [['a', logStudent(2)], ['b', logStudent(2)],
      ['c', 2 * logStudent(2)]] as Array<[string, number]>) {
      assert.ok(Math.abs((await ctx.getMeasure(name)).samples[0] - expected) < 1e-12);
    }
    for (const name of ['direct', 'scalar']) {
      checkDraws(await ctx.getMeasure(name), [1], true);
    }
    checkDraws(await ctx.getMeasure('rows'), [1, 1], false);
    const lo = (await ctx.getMeasure('lo')).samples[0];
    const hi = (await ctx.getMeasure('hi')).samples[0];
    assert.ok(Math.abs((hi - lo) / 2e-5) < 1e-8);
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

for (const weight of ['weighted(v', 'logweighted(1000 + v']) {
  test(`normalized ${weight}) callbacks preserve each captured parameter and its prior`, async () => {
    const { ctx } = makeMatCtx(`
w ~ Uniform(interval(1, 5))
K(v) = normalize(${weight}, locscale(StudentT(4), 0, v)))
x ~ K.(w)
P = lawof(record(w = w, x = x))
r = logdensityof(P, record(w = 2, x = 2))
`, { sampleCount: N, rootSeed: 117 });
    assert.ok(Math.abs((await ctx.getMeasure('r')).samples[0]
      - (logStudent(2, 2) - Math.log(4))) < 1e-12);
    const joint = await ctx.getMeasure('P');
    const w = joint.fields.w.samples;
    const x = joint.fields.x.samples;
    const probabilities = Array.from({ length: N }, (_, i) =>
      joint.logWeights ? Math.exp(joint.logWeights[i]) : 1 / N);
    const total = probabilities.reduce((a, b) => a + b, 0);
    const mean = probabilities.reduce((sum, p, i) => sum + p * w[i], 0) / total;
    assert.ok(Math.abs(mean - 3) < 0.2, `captured prior mean ${mean}`);
    checkDraws({ samples: x.map((v: number, i: number) => v / w[i]),
      value: { shape: [N] }, logTotalmass: joint.logTotalmass }, [1], true);
  });
}

test('MCMC scoring uses the same scalar logweight normalizer', async () => {
  const { ctx } = makeMatCtx(`
w ~ Uniform(interval(-1001, 1001))
y ~ normalize(logweighted(w, StudentT(4)))
K = kernelof(y, w = w)
L = likelihoodof(K, 2)
P = bayesupdate(L, lawof(w))
`, { sampleCount: 16 });
  const d = Object.values(ctx.derivations).find((d: any) => d.kind === 'bayesupdate');
  const { likOf } = await buildLogPi(ctx, d);
  for (const w of [-1000, 1000]) assert.ok(Math.abs(likOf({ w }) - logStudent(2)) < 1e-12);
});

test('a valid normalized law still has zero density outside its support', async () => {
  const { ctx } = makeMatCtx(`
K(w) = normalize(logweighted(w, Uniform(interval(0, 1))))
x ~ K.([1000, -1000])
r = logdensityof(lawof(x), [-1, 0.5])
`, { sampleCount: 16 });
  assert.equal((await ctx.getMeasure('r')).samples[0], -Infinity);
});

test('undefined scalar normalizers cannot score or return a normalized ensemble', async () => {
  const guard = `guard(s) = normalize(weighted(ifelse((s > 0) && (s < inf), 1, 0),
    locscale(StudentT(4), 0, ifelse((s > 0) && (s < inf), s, 1))))`;
  for (const application of ['guard(0)', 'guard.(0)', 'guard.([1, 0])']) {
    const source = `${guard}\nx ~ ${application}\nr = logdensityof(lawof(x), ${
      application.includes('[') ? '[2, 2]' : '2'})`;
    for (const name of ['x', 'r']) {
      await assert.rejects(async () => makeMatCtx(source, { sampleCount: 16 }).ctx.getMeasure(name));
    }
  }
  await assert.rejects(async () => makeMatCtx(`
K(w) = normalize(weighted(w, joint(a = Normal(0, 1), b = StudentT(4))))
x ~ K(0)
`, { sampleCount: 16 }).ctx.getMeasure('x'));
  // Distinct weight-domain boundaries: negative/NaN ordinary weight and
  // infinite mass (including exp(+Infinity)); exp(-Infinity) has zero mass.
  for (const weight of ['weighted(-1', 'weighted(0/0', 'weighted(inf',
    'logweighted(inf', 'logweighted(-inf', 'logweighted(0/0']) {
    for (const name of ['x', 'r']) {
      await assert.rejects(async () => makeMatCtx(`
K(w) = normalize(${weight}, StudentT(4)))
x ~ K.(1)
r = logdensityof(lawof(x), 2)
`, { sampleCount: 16 }).ctx.getMeasure(name));
    }
  }
});
