'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { inBothModes } = require('./_perf-helpers.ts');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const density = require('../density.ts');

const near = (actual: number, expected: number, tolerance = 1e-10) => {
  if (!Number.isFinite(expected)) assert.equal(actual, expected);
  else assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
};
const factorial = (n: number): number => n < 2 ? 1 : n * factorial(n - 1);
const normal = (x: number, mu: number, sigma: number) => Math.log(Math.exp(-.5 * ((x - mu) / sigma) ** 2)
  / (sigma * Math.sqrt(2 * Math.PI)));
const binomial = (x: number, n: number, p: number) => {
  if (p === 0) return x === 0 ? 0 : -Infinity;
  if (p === 1) return x === n ? 0 : -Infinity;
  return Math.log(factorial(n) / factorial(x) / factorial(n - x) * p ** x * (1 - p) ** (n - x));
};

/** Exercise public posterior setup and scalar/batch likelihood evaluation.
 * Integer shape oracles use finite factorial products, not special functions. */
async function model(law: string, data: any[], iid = true) {
  const { ctx } = ctxFor(`
a ~ Normal(0, 1)
b ~ Normal(0, 1)
data = ${JSON.stringify(data)}
y ~ ${iid ? `iid(${law}, ${data.length})` : law}
P = lawof(record(a = a, b = b))
K = kernelof(record(y = y), a = a, b = b)
L = likelihoodof(K, record(y = data))
posterior = bayesupdate(L, P)
`, 1);
  return { ctx, scorer: await buildLogPi(ctx, ctx.derivations.posterior) };
}

const families = [
  { law: 'Normal(a, b)', data: [-.5, .25, 2], points: [{ a: 0, b: 1 }, { a: 1, b: 2 }],
    oracle: (x: number, p: any) => normal(x, p.a, p.b) },
  { law: 'Gamma(a, b)', data: [.25, 1, 3], points: [{ a: 2, b: 2 }, { a: 3, b: .5 }],
    oracle: (x: number, p: any) => Math.log(p.b ** p.a * x ** (p.a - 1) * Math.exp(-p.b * x) / factorial(p.a - 1)) },
  { law: 'Beta(a, b)', data: [.2, .5, .75], points: [{ a: 2, b: 3 }, { a: 3, b: 2 }],
    oracle: (x: number, p: any) => Math.log(factorial(p.a + p.b - 1) / factorial(p.a - 1)
      / factorial(p.b - 1) * x ** (p.a - 1) * (1 - x) ** (p.b - 1)) },
  { law: 'Poisson(a)', data: [0, 2, 5], points: [{ a: 2, b: 1 }, { a: 7, b: 1 }],
    oracle: (x: number, p: any) => Math.log(p.a ** x * Math.exp(-p.a) / factorial(x)) },
  { law: 'Binomial(6, a)', data: [0, 2, 6], points: [{ a: .2, b: 1 }, { a: .7, b: 1 }],
    oracle: (x: number, p: any) => binomial(x, 6, p.a) },
  { law: 'Exponential(a)', data: [0, .5, 3], points: [{ a: 2, b: 1 }, { a: .5, b: 1 }],
    oracle: (x: number, p: any) => Math.log(p.a * Math.exp(-p.a * x)) },
];

for (const family of families) inBothModes(`fixed IID ${family.law} keeps normalized live scalar/batch scores`,
  'density.fixedLikelihood', async () => {
    const { scorer } = await model(family.law, family.data);
    const points = [...family.points, family.points[0]];
    const expected = points.map(p => family.data.reduce((s, x) => s + family.oracle(x, p), 0));
    points.forEach((p, i) => near(scorer.likOf(p), expected[i]));
    scorer.priorLikBatch(points).lik.forEach((x: number, i: number) => near(x, expected[i]));
  });

const dotted = [
  { law: 'Normal.(a, [1, 1, 1])', family: families[0], point: { a: 1, b: 1 } },
  { law: 'Gamma.(a, [2, 2, 2])', family: families[1], point: { a: 3, b: 2 } },
  { law: 'Beta.(a, [3, 3, 3])', family: families[2], point: { a: 2, b: 3 } },
  { law: 'Poisson.([2, 2, 2])', family: families[3], point: { a: 2, b: 1 } },
  { law: 'Binomial.(n = [6, 6, 6], p = a)', family: families[4], point: { a: .7, b: 1 } },
  { law: 'Exponential.([2, 2, 2])', family: families[5], point: { a: 2, b: 1 } },
];
for (const { law, family, point } of dotted) inBothModes(`fixed splat ${law} retains its normalized plate score`,
  'density.fixedLikelihood', async () => {
    const { scorer } = await model(law, family.data, false);
    const expected = family.data.reduce((s, x) => s + family.oracle(x, point), 0);
    near(scorer.likOf(point), expected);
    scorer.priorLikBatch([point, point]).lik.forEach((x: number) => near(x, expected));
  });

inBothModes('fixed splat broadcasts keep singleton parameters live and varying vectors ordinary', 'density.fixedLikelihood', async () => {
  const singleton = await model('Normal.([0, 0, 0], [b])', [1, 2, 3], false);
  const varying = await model('Normal.([0, 1, 2], b)', [1, 2, 3], false);
  const repeated = await model('Normal.([a, a, a], b)', [1, 2, 3], false);
  const points = [{ a: 0, b: 1 }, { a: 1, b: 2 }, { a: 0, b: 1 }];
  for (const [scorer, expected] of [
    [singleton.scorer, (p: any) => [1, 2, 3].reduce((s, x) => s + normal(x, 0, p.b), 0)],
    [varying.scorer, (p: any) => 3 * normal(1, 0, p.b)],
    [repeated.scorer, (p: any) => [1, 2, 3].reduce((s, x) => s + normal(x, p.a, p.b), 0)],
  ] as const) {
    points.forEach(p => near(scorer.likOf(p), expected(p)));
    scorer.priorLikBatch(points).lik.forEach((x: number, i: number) => near(x, expected(points[i])));
  }
});

inBothModes('fixed Normal moments retain a small spread around a large center', 'density.fixedLikelihood', async () => {
  const center = 1e12, offsets = [0, .25, -.5];
  const { scorer } = await model('Normal(a, b)', offsets.map(x => center + x));
  for (const offset of [.125, -.25, .125]) {
    const expected = offsets.reduce((s, x) => s + normal(x, offset, .75), 0);
    near(scorer.likOf({ a: center + offset, b: .75 }), expected);
  }
});

inBothModes('fixed likelihood ownership survives host mutation and alternating scorers', 'density.fixedLikelihood', async () => {
  const first = await model('Normal(a, b)', [1, 2, 3]);
  const second = await model('Normal(a, b)', [4, 5, 6]);
  first.ctx.fixedValues.set('data', [1, 2, 3]);
  const captured = await buildLogPi(first.ctx, first.ctx.derivations.posterior);
  const source = first.ctx.fixedValues.get('data');
  const data = source.data ?? source;
  const point = { a: .25, b: 2 };
  const expected = [1, 2, 3].reduce((s, x) => s + normal(x, point.a, point.b), 0);
  near(captured.likOf(point), expected);
  data[0] = 9;
  const leaf = { kind: 'call', op: 'Normal', args: [{ kind: 'lit', value: point.a }, { kind: 'lit', value: point.b }] };
  const ir = { kind: 'call', op: 'iid', args: [leaf, { kind: 'lit', value: 3 }] };
  near(density.logDensityN(ir, source, null, 1, {})[0], [9, 2, 3].reduce((s, x) => s + normal(x, point.a, point.b), 0));
  near(second.scorer.likOf(point), [4, 5, 6].reduce((s, x) => s + normal(x, point.a, point.b), 0));
  near(captured.likOf(point), expected);
  const rebuilt = await buildLogPi(first.ctx, first.ctx.derivations.posterior);
  near(rebuilt.likOf(point), [9, 2, 3].reduce((s, x) => s + normal(x, point.a, point.b), 0));
});

inBothModes('owned record fields and flat IID suffixes retain their distinct regions and trials', 'density.fixedLikelihood', async () => {
  const { ctx } = ctxFor(`
a ~ Beta(2, 2)
u ~ iid(Binomial(4, a), 2)
v ~ iid(Binomial(6, a), 2)
w ~ joint(iid(Binomial(4, a), 2), iid(Binomial(6, a), 2))
P = lawof(record(a = a))
K = kernelof(record(u = u, v = v, w = w), a = a)
L = likelihoodof(K, record(u = [0, 4], v = [1, 5], w = [0, 4, 1, 5]))
posterior = bayesupdate(L, P)
`, 1);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  for (const a of [.2, .7, .2]) near(scorer.likOf({ a }),
    2 * (binomial(0, 4, a) + binomial(4, 4, a) + binomial(1, 6, a) + binomial(5, 6, a)));
});

inBothModes('fixed likelihoods keep boundary, invalid-parameter and empty-product semantics', 'density.fixedLikelihood', async () => {
  const cases = [
    { law: 'Gamma(a, b)', data: [0, 1], point: { a: 1, b: 2 }, expected: 2 * Math.log(2) - 2 },
    { law: 'Beta(a, b)', data: [0, 1], point: { a: 1, b: 1 }, expected: 0 },
    { law: 'Exponential(a)', data: [-1, 1], point: { a: 2, b: 1 }, expected: -Infinity },
    { law: 'Poisson(a)', data: [0, 0], point: { a: 0, b: 1 }, expected: 0 },
    { law: 'Poisson(a)', data: [0, 1], point: { a: 0, b: 1 }, expected: -Infinity },
    { law: 'Binomial(4, a)', data: [0, 0], point: { a: 0, b: 1 }, expected: 0 },
    { law: 'Binomial(4, a)', data: [4, 4], point: { a: 1, b: 1 }, expected: 0 },
    { law: 'Binomial(4, a)', data: [0, 1], point: { a: 0, b: 1 }, expected: -Infinity },
    { law: 'Binomial(4, a)', data: [3, 4], point: { a: 1, b: 1 }, expected: -Infinity },
    { law: 'Binomial(4, a)', data: [0, 5], point: { a: .5, b: 1 }, expected: -Infinity },
    { law: 'Binomial(4, a)', data: [true, false], point: { a: .5, b: 1 }, expected: -Math.log(64) },
    { law: 'Normal(a, b)', data: [0, 1], point: { a: 0, b: -1 }, expected: -Infinity },
    { law: 'Gamma(a, b)', data: [1, 2], point: { a: -1, b: 2 }, expected: -Infinity },
    { law: 'Beta(a, b)', data: [.2, .5], point: { a: 2, b: -1 }, expected: -Infinity },
    { law: 'Poisson(a)', data: [0, 1], point: { a: -1, b: 1 }, expected: -Infinity },
    { law: 'Binomial(4, a)', data: [0, 1], point: { a: 1.1, b: 1 }, expected: -Infinity },
    { law: 'Exponential(a)', data: [0, 1], point: { a: -1, b: 1 }, expected: -Infinity },
    { law: 'Normal(a, b)', data: [], point: { a: 0, b: 1 }, expected: 0 },
  ];
  for (const c of cases) {
    const { scorer } = await model(c.law, c.data);
    near(scorer.likOf(c.point), c.expected);
  }
});

inBothModes('runtime Binomial trials and Loader-scale Poisson retain their numeric oracle', 'density.fixedLikelihood', async () => {
  const runtime = await model('Binomial(b, a)', [1, 2]);
  for (const b of [4, 7, 4]) near(runtime.scorer.likOf({ a: .3, b }), binomial(1, b, .3) + binomial(2, b, .3));
  const poisson = await model('Poisson(a)', [1e12, 1e12]);
  // Stirling at its mean: next omitted term is below 1e-60 here.
  near(poisson.scorer.likOf({ a: 1e12, b: 1 }), -Math.log(2 * Math.PI * 1e12) - 1 / (6e12));
});

inBothModes('non-finite observations and overflowing summaries keep ordinary fallback arithmetic', 'density.fixedLikelihood', async () => {
  const initial = await model('Normal(a, b)', [0, 1]);
  const data = initial.ctx.fixedValues.get('data');
  (data.data ?? data)[0] = NaN;
  const nonfinite = await buildLogPi(initial.ctx, initial.ctx.derivations.posterior);
  near(nonfinite.likOf({ a: 0, b: 1 }), -Infinity);
  const overflow = await model('Exponential(a)', [1e308, 1e308]);
  // Each product rate*x remains finite even though the observation sum does not.
  near(overflow.scorer.likOf({ a: 1e-308, b: 1 }), 2 * Math.log(1e-308) - 2);
  near(overflow.scorer.likOf({ a: Infinity, b: 1 }), -Infinity);
  const normalOverflow = await model('Normal(a, b)', [1e200, -1e200]);
  near(normalOverflow.scorer.likOf({ a: 0, b: 1 }), -Infinity);
  const normalUnderflow = await model('Normal(a, b)', [1e-200, -1e-200]);
  near(normalUnderflow.scorer.likOf({ a: 0, b: 1 }), -Math.log(2 * Math.PI));
  const concentrated = await model('Gamma(a, b)', [1e6, 1e6]);
  near(concentrated.scorer.likOf({ a: 1e6, b: 1 }), -Math.log(2 * Math.PI * 1e6) - 1 / 6e6);
  const unsupported = await model('StudentT(a)', [0, 1]);
  near(unsupported.scorer.likOf({ a: 1, b: 1 }), -2 * Math.log(Math.PI) - Math.log(2));
});

inBothModes('IID logarithmic Binomial and Poisson tilts retain rare-event scores', 'density.fixedLikelihood', async () => {
  const bin = await model('normalize(logweighted(fn(a * _), Binomial(4, .2)))', [2, 3]);
  near(bin.scorer.likOf({ a: 1000, b: 1 }), Math.log(24) - 3000 + 3 * Math.log(4));
  near(bin.scorer.likOf({ a: -1000, b: 1 }), Math.log(24) - 5000 - 5 * Math.log(4));
  const poisson = await model('normalize(logweighted(fn(a * _), Poisson(1)))', [1, 2]);
  near(poisson.scorer.likOf({ a: -1000, b: 1 }), -3000 - Math.log(2));
});

inBothModes('IID mixtures and mixtures of IID retain different latent-choice laws', 'density.fixedLikelihood', async () => {
  const { ctx } = ctxFor(`
a ~ Normal(0, 1)
A = Normal(a, 1)
B = Normal(a + 2, 1)
independent ~ iid(superpose(weighted(.25, A), weighted(.75, B)), 2)
shared ~ superpose(weighted(.25, iid(A, 2)), weighted(.75, iid(B, 2)))
P = lawof(record(a = a))
K = kernelof(record(independent = independent, shared = shared), a = a)
L = likelihoodof(K, record(independent = [0, 2], shared = [0, 2]))
posterior = bayesupdate(L, P)
`, 1);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  for (const a of [0, 1, 0]) {
    const p = [0, 2].map(x => Math.exp(normal(x, a, 1)));
    const q = [0, 2].map(x => Math.exp(normal(x, a + 2, 1)));
    near(scorer.likOf({ a }), Math.log((.25 * p[0] + .75 * q[0]) * (.25 * p[1] + .75 * q[1]))
      + Math.log(.25 * p[0] * p[1] + .75 * q[0] * q[1]));
  }
});

inBothModes('changing inverse-transform observations retain their ordinary score and Jacobian', 'density.fixedLikelihood', async () => {
  const { ctx } = ctxFor(`
a ~ Exponential(1)
y ~ pushfwd(x -> a .* x .+ 1, iid(Normal(0, 1), 2))
P = lawof(record(a = a))
K = kernelof(record(y = y), a = a)
L = likelihoodof(K, record(y = [2, 3]))
posterior = bayesupdate(L, P)
`, 1);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  for (const a of [1, 2, 1]) near(scorer.likOf({ a }),
    normal(1 / a, 0, 1) + normal(2 / a, 0, 1) - 2 * Math.log(a));
});

inBothModes('prepared positional jointchain observations retain conditional scalar/batch values', 'density.fixedLikelihood', async () => {
  const { ctx } = ctxFor(`
a ~ Normal(0, 1)
K = x -> Normal(x, 1)
y ~ jointchain(Normal(a, 1), K)
P = lawof(record(a = a))
L = likelihoodof(kernelof(y, a = a), [1, 2])
posterior = bayesupdate(L, P)
`, 1);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  const points = [{ a: 0 }, { a: 1 }, { a: 0 }];
  const expected = points.map(({ a }) => normal(1, a, 1) + normal(2, 1, 1));
  points.forEach((p, i) => near(scorer.likOf(p), expected[i]));
  scorer.priorLikBatch(points).lik.forEach((x: number, i: number) => near(x, expected[i]));
});

test('density pipeline hooks cannot mutate a prepared likelihood snapshot', async () => {
  const { scorer } = await model('Normal(a, b)', [1, 2]);
  const other = await model('Normal(a, b)', [4, 5]);
  const point = { a: 0, b: 1 }, expected = normal(1, 0, 1) + normal(2, 0, 1);
  near(scorer.likOf(point), expected);
  let insideHook = false;
  density.registerDensityStage((ir: any, value: any, refs: any, count: number, opts: any, next: any) => {
    if (insideHook) return next(ir, value, refs, count, opts);
    insideHook = true;
    try { near(other.scorer.likOf(point), normal(4, 0, 1) + normal(5, 0, 1)); }
    finally { insideHook = false; }
    const vector = value.y.data ?? value.y;
    vector[0] = 7;
    return next(ir, value, refs, count, opts);
  });
  try { near(scorer.likOf(point), normal(7, 0, 1) + normal(2, 0, 1)); }
  finally { density._resetDensityPipeline(); }
  near(scorer.likOf(point), expected);
});
