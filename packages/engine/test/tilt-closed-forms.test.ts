'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { orchestrator, lower, materialiser, rng } = require('..');
const density = require('../density.ts');
const sampler = require('../sampler.ts');
const oracle = require('./fixtures/tilt/oracle.json');

const close = (actual: number, expected: number, tol = 1e-11) =>
  assert.ok(Math.abs(actual - expected) < tol, `${actual} != ${expected} (tol ${tol})`);

/** Check every entry against independent moments, with six Monte Carlo SEs. */
function checkMoments(data: Float64Array, mean: number[], covariance: number[][]) {
  const dimension = mean.length;
  const count = data.length / dimension;
  for (let j = 0; j < dimension; j++) {
    let sum = 0;
    for (let i = 0; i < count; i++) sum += data[i * dimension + j];
    close(sum / count, mean[j], 6 * Math.sqrt(covariance[j][j] / count));
    for (let k = 0; k < dimension; k++) {
      let product = 0, square = 0;
      for (let i = 0; i < count; i++) {
        const p = (data[i * dimension + j] - mean[j]) * (data[i * dimension + k] - mean[k]);
        product += p;
        square += p * p;
      }
      const observed = product / count;
      const se = Math.sqrt(Math.max(0, square / count - observed ** 2) / count);
      close(observed, covariance[j][k], 6 * se + 1e-12);
    }
  }
}

/** Composite Simpson quadrature; all callers use an even interval count. */
function simpson(f: (x: number) => number, lo: number, hi: number, intervals: number): number {
  const h = (hi - lo) / intervals;
  let total = f(lo) + f(hi);
  for (let i = 1; i < intervals; i++) total += (i % 2 === 0 ? 2 : 4) * f(lo + i * h);
  return total * h / 3;
}

test('ordered Exponential and Gamma scores retain inferred rate and shape', async () => {
  const x = JSON.stringify(oracle.ordered_gamma.point);
  const { ctx } = makeMatCtx(`
rate = elementof(posreals)
shape = elementof(posreals)
E = tilt(iid(Exponential(rate), 3), Lebesgue(ordered(3)))
G = tilt(iid(Gamma(shape, rate), 3), Lebesgue(ordered(3)))
es = logdensityof(E, ${x})
gs = logdensityof(G, ${x})
outside_score = logdensityof(E, [0.5, 0.1, 2])
ef = functionof(es, rate = rate)
gf = functionof(gs, shape = shape, rate = rate)
outside_fn = functionof(outside_score, rate = rate)
outside = outside_fn(1)
e = ef(1)
g = gf(2.5, 1)
elo = ef(0.99999)
ehi = ef(1.00001)
glo = gf(2.49999, 1)
ghi = gf(2.50001, 1)
grlo = gf(2.5, 0.99999)
grhi = gf(2.5, 1.00001)
`, { sampleCount: 4 });
  const score = async (name: string) => (await ctx.getMeasure(name)).samples[0];
  close(await score('e'), oracle.ordered_exponential.logdensity);
  close(await score('g'), oracle.ordered_gamma.logdensity);
  assert.equal(await score('outside'), -Infinity);
  close(((await score('ehi')) - (await score('elo'))) / 0.00002,
    oracle.ordered_exponential.rate_gradient, 2e-8);
  close(((await score('ghi')) - (await score('glo'))) / 0.00002,
    oracle.ordered_gamma.shape_gradient, 2e-8);
  close(((await score('grhi')) - (await score('grlo'))) / 0.00002,
    3 * oracle.ordered_gamma.shape - oracle.ordered_gamma.point.reduce((a: number, b: number) => a + b, 0), 2e-8);
});

test('ordered normalized composite scores preserve restriction and cancel constant weights', async () => {
  const { ctx } = makeMatCtx(`
D = normalize(logweighted(-2, truncate(Exponential(1), interval(0, 2))))
W = normalize(weighted(3, truncate(Exponential(1), interval(0, 2))))
A = tilt(iid(D, 3), Lebesgue(ordered(3)))
B = tilt(iid(W, 3), Lebesgue(ordered(3)))
a = logdensityof(A, [0.1, 0.5, 1])
b = logdensityof(B, [0.1, 0.5, 1])
outside = logdensityof(A, [0.1, 0.5, 3])
`, { sampleCount: 4 });
  const expected = Math.log(6) - 1.6 - 3 * Math.log(-Math.expm1(-2));
  close((await ctx.getMeasure('a')).samples[0], expected, 1e-9);
  close((await ctx.getMeasure('b')).samples[0], expected, 1e-9);
  assert.equal((await ctx.getMeasure('outside')).samples[0], -Infinity);
});

test('directly lowered normalized iid laws retain scores, sorted draws and RNG successors', async () => {
  const { ctx, lifted } = makeMatCtx(`
M = tilt(iid(normalize(Exponential(1)), 3), Lebesgue(ordered(3)))
B = iid(Exponential(1), 3)
`, { sampleCount: 4 });
  const ir = lower.lowerBinding(lifted.ast.body[0]);
  close(density.logDensity(ir, [0.1, 0.5, 2], {}), Math.log(6) - 2.6);
  const state = rng.seedFromBytes([1, 2, 3, 4]);
  const draw = sampler.walk(state, ir, {});
  const base = sampler.walk(state, lower.lowerBinding(lifted.ast.body[1]), {});
  assert.deepEqual(Array.from(draw.value), Array.from<number>(base.value).sort((a, b) => a - b));
  assert.deepEqual(draw.state, base.state);
  const measure = await materialiser.materialiseMeasureIR(ir, ctx);
  for (let i = 0; i < 4; i++) {
    const x = measure.value.data.subarray(3 * i, 3 * i + 3);
    assert.ok(x[0] > 0 && x[0] < x[1] && x[1] < x[2]);
    close(density.logDensity(ir, Array.from(x), {}), Math.log(6) - x.reduce((a: number, b: number) => a + b, 0));
  }
});

test('scalar and vector Normal parameters give identical explicit draws and RNG successors', async () => {
  const { ctx } = makeMatCtx(`
rng = rnginit([1, 2, 3, 4])
a, ar = rand(tilt(Normal.(1, [1, 2, 0.5]), Lebesgue(zerosum(3))), rng)
b, br = rand(tilt(Normal.([1, 1, 1], [1, 2, 0.5]), Lebesgue(zerosum(3))), rng)
after_a, _ = rand(Normal(0, 1), ar)
after_b, _ = rand(Normal(0, 1), br)
`, { sampleCount: 4 });
  assert.deepEqual((await ctx.getMeasure('a')).value.data, (await ctx.getMeasure('b')).value.data);
  assert.deepEqual((await ctx.getMeasure('after_a')).value.data, (await ctx.getMeasure('after_b')).value.data);
});

test('unequal zero-sum Normals match the chart oracle and all parameter derivatives', async () => {
  const { mu, sigma, point } = oracle.zerosum_normal;
  const lines: string[] = [];
  for (const [name, values] of [['mu', mu], ['sigma', sigma]] as [string, number[]][]) {
    for (let j = 0; j < 3; j++) {
      for (const [side, delta] of [['lo', -1e-5], ['hi', 1e-5]] as [string, number][]) {
        const perturbed = values.map((value, k) => value + (k === j ? delta : 0));
        lines.push(`${name}_${side}_${j} = f(${JSON.stringify(name === 'mu' ? perturbed : mu)}, ${JSON.stringify(name === 'sigma' ? perturbed : sigma)})`);
      }
    }
  }
  const { ctx } = makeMatCtx(`
mu = elementof(cartpow(reals, 3))
sigma = elementof(cartpow(posreals, 3))
M = tilt(Normal.(mu, sigma), Lebesgue(zerosum(3)))
score = logdensityof(M, ${JSON.stringify(point)})
f = functionof(score, mu = mu, sigma = sigma)
point_logdensity = f(${JSON.stringify(mu)}, ${JSON.stringify(sigma)})
${lines.join('\n')}
`, { sampleCount: 4 });
  const score = async (name: string) => (await ctx.getMeasure(name)).samples[0];
  close(await score('point_logdensity'), oracle.zerosum_normal.logdensity);
  close(await score('point_logdensity'), oracle.zerosum_normal.chart_logdensity);
  for (const parameter of ['mu', 'sigma']) {
    for (let j = 0; j < 3; j++) {
      close(((await score(`${parameter}_hi_${j}`)) - (await score(`${parameter}_lo_${j}`))) / 0.00002,
        oracle.zerosum_normal[`${parameter}_gradient`][j], 2e-8);
    }
  }
});

test('conditional Normal broadcasts preserve density and distinguish atom and collection axes', async () => {
  const { ctx } = makeMatCtx(`
A = tilt(Normal.(1, [1, 2, 0.5]), Lebesgue(zerosum(3)))
B = tilt(Normal.([1], [1, 2, 0.5]), Lebesgue(zerosum(3)))
C = tilt(Normal.([1, 1, 1], [1, 2, 0.5]), Lebesgue(zerosum(3)))
D = tilt(Normal.([1, -2, 0.5], 2), Lebesgue(zerosum(3)))
E = tilt(Normal.([1, -2, 0.5], [2]), Lebesgue(zerosum(3)))
F = tilt(Normal.([1, -2, 0.5], [2, 2, 2]), Lebesgue(zerosum(3)))
a = logdensityof(A, [0.3, -0.5, 0.2])
b = logdensityof(B, [0.3, -0.5, 0.2])
c = logdensityof(C, [0.3, -0.5, 0.2])
d = logdensityof(D, [0.3, -0.5, 0.2])
e = logdensityof(E, [0.3, -0.5, 0.2])
f = logdensityof(F, [0.3, -0.5, 0.2])
`, { sampleCount: 3 });
  const score = async (name: string) => (await ctx.getMeasure(name)).samples[0];
  close(await score('a'), await score('c'));
  close(await score('b'), await score('c'));
  close(await score('d'), await score('f'));
  close(await score('e'), await score('f'));
  for (const name of ['A', 'B', 'C', 'D', 'E', 'F']) {
    assert.deepEqual((await ctx.getMeasure(name)).value.shape, [3, 3]);
  }
});

test('a singleton conditional Normal scores zero and keeps the primitive RNG advance', async () => {
  const { ctx } = makeMatCtx(`
M = tilt(Normal.([1], [2]), Lebesgue(zerosum(1)))
score = logdensityof(M, [0])
rng = rnginit([1, 2, 3, 4])
x, xs = rand(M, rng)
z, zs = rand(iid(Normal(0, 1), 1), rng)
after_x, _ = rand(iid(Normal(0, 1), 4), xs)
after_z, _ = rand(iid(Normal(0, 1), 4), zs)
`, { sampleCount: 4 });
  close((await ctx.getMeasure('score')).samples[0], 0);
  assert.deepEqual(Array.from((await ctx.getMeasure('x')).value.data), [0]);
  assert.deepEqual((await ctx.getMeasure('after_x')).value.data,
    (await ctx.getMeasure('after_z')).value.data);
});

test('twenty thousand draws per law match all oracle moments and their support', async () => {
  const count = 20_000;
  const { ctx } = makeMatCtx(`
exponential ~ tilt(iid(Exponential(1), 3), Lebesgue(ordered(3)))
gamma ~ tilt(iid(Gamma(2.5, 1), 3), Lebesgue(ordered(3)))
normal ~ tilt(Normal.([1, -2, 0.5], [1, 2, 0.5]), Lebesgue(zerosum(3)))
`, { sampleCount: count, rootSeed: 1729 });
  for (const [name, expected] of [
    ['exponential', oracle.ordered_exponential],
    ['gamma', oracle.ordered_gamma],
    ['normal', oracle.zerosum_normal],
  ] as [string, any][]) {
    const { value } = await ctx.getMeasure(name);
    assert.deepEqual(value.shape, [count, 3]);
    for (let i = 0; i < count; i++) {
      const [a, b, c] = value.data.subarray(3 * i, 3 * i + 3);
      if (name === 'normal') close(a + b + c, 0, 1e-12);
      else assert.ok(a >= 0 && a < b && b < c);
    }
    checkMoments(value.data, expected.mean, expected.covariance);
  }
});

test('ordered and conditional sampling retain each source primitive continuation', async () => {
  const { ctx } = makeMatCtx(`
rng = rnginit([5, 6, 7, 8])
e, es = rand(iid(Exponential(1), 3), rng)
g, gs = rand(iid(Gamma(2.5, 1), 3), rng)
n, ns = rand(iid(Normal(0, 1), 3), rng)
oe, oes = rand(tilt(iid(Exponential(1), 3), Lebesgue(ordered(3))), rng)
og, ogs = rand(tilt(iid(Gamma(2.5, 1), 3), Lebesgue(ordered(3))), rng)
zn, zns = rand(tilt(Normal.([1, -2, 0.5], [1, 2, 0.5]), Lebesgue(zerosum(3))), rng)
after_e, _ = rand(iid(Normal(0, 1), 4), es)
after_g, _ = rand(iid(Normal(0, 1), 4), gs)
after_n, _ = rand(iid(Normal(0, 1), 4), ns)
after_oe, _ = rand(iid(Normal(0, 1), 4), oes)
after_og, _ = rand(iid(Normal(0, 1), 4), ogs)
after_zn, _ = rand(iid(Normal(0, 1), 4), zns)
`, { sampleCount: 4 });
  const data = async (name: string) => Array.from((await ctx.getMeasure(name)).value.data) as number[];
  assert.deepEqual(await data('oe'), (await data('e')).sort((a, b) => a - b));
  assert.deepEqual(await data('og'), (await data('g')).sort((a, b) => a - b));
  for (const [tilted, source] of [['oe', 'e'], ['og', 'g'], ['zn', 'n']]) {
    assert.deepEqual(await data(`after_${tilted}`), await data(`after_${source}`));
  }
});

test('the current n=2 densities integrate to the independent unit-mass oracle', () => {
  const { ctx } = makeMatCtx(`
E = tilt(iid(Exponential(1), 2), Lebesgue(ordered(2)))
G = tilt(iid(Gamma(2.5, 1), 2), Lebesgue(ordered(2)))
N = tilt(Normal.([1, -2], [1, 2]), Lebesgue(zerosum(2)))
`, { sampleCount: 4 });
  const expand = (name: string) => orchestrator.expandMeasure(name, {
    bindings: ctx.bindings, derivations: ctx.derivations, fixedValues: ctx.fixedValues,
  });
  const env = Object.fromEntries(ctx.fixedValues);
  for (const [name, expected] of [
    ['E', oracle.unit_mass_n2.ordered_exponential], ['G', oracle.unit_mass_n2.ordered_gamma],
  ] as [string, number][]) {
    const ir = expand(name);
    // x>=0 and gap>=0 have unit Jacobian to [x,x+gap]. At gap=0 use
    // the interior limit because strict ordering excludes only that boundary.
    // On this grid the independent Exponential/Gamma formulas differ from
    // SciPy's infinite-domain oracle by 6e-7/1.3e-5, including the cutoff tail.
    const mass = simpson(x => simpson(gap => {
      const step = gap === 0 ? 2 * Number.EPSILON * Math.max(1, x) : gap;
      return Math.exp(density.logDensity(ir, [x, x + step], env, {}));
    }, 0, 24, 480), 0, 24, 480);
    close(mass, expected, 2e-5);
  }
  const normal = expand('N');
  const mass = simpson(x => Math.exp(density.logDensity(normal, [x, -x], env, {})), -12, 12, 480);
  close(mass, oracle.unit_mass_n2.zerosum_normal, 1e-10);
});

test('a reified unequal-Normal prior keeps its inferred scale in scores and every draw', async () => {
  const expected = oracle.zerosum_normal;
  const count = 20_000;
  const { ctx } = makeMatCtx(`
scale ~ Exponential(1)
effects ~ tilt(Normal.([1, -2, 0.5], scale .* [1, 2, 0.5]), Lebesgue(zerosum(3)))
prior = lawof(record(scale = scale, effects = effects))
a = logdensityof(prior, record(scale = 2, effects = ${JSON.stringify(expected.point)}))
b = logdensityof(prior, record(scale = 3, effects = ${JSON.stringify(expected.point)}))
`, { sampleCount: count, rootSeed: 2718 });
  const quadratic = expected.point.reduce((sum: number, x: number, j: number) =>
    sum + ((x - expected.mean[j]) / expected.sigma[j]) ** 2, 0);
  for (const [name, scale] of [['a', 2], ['b', 3]] as [string, number][]) {
    close((await ctx.getMeasure(name)).samples[0],
      -scale + expected.logdensity - 2 * Math.log(scale) + quadratic / 2 * (1 - 1 / scale ** 2));
  }
  const prior = await ctx.getMeasure('prior');
  const scales = prior.fields.scale.samples;
  const effects = prior.fields.effects.value.data;
  const standardized = Float64Array.from(effects, (x: number, i: number) =>
    (x - expected.mean[i % 3]) / scales[Math.floor(i / 3)]);
  checkMoments(standardized, [0, 0, 0], expected.covariance);
});

test('unsupported sum laws, non-iid ordering and atomic ordering refuse scores and draws', async () => {
  for (const [source, support, diagnostic] of [
    ['iid(StudentT(3), 3)', 'zerosum', /tilt on zerosum.*closed-form sum law.*StudentT/],
    ['Normal.([0, 1, 2], 1)', 'ordered', /tilt on ordered.*requires iid.*broadcast.*unsupported/],
    ['iid(Dirac(0.5), 3)', 'ordered', /tilt on ordered.*continuous scalar law.*Dirac.*unsupported/],
  ] as [string, string, RegExp][]) {
    const { ctx } = makeMatCtx(`
M = tilt(${source}, Lebesgue(${support}(3)))
score = logdensityof(M, [0.1, 0.5, 2])
draw, _ = rand(M, rnginit(0))
`, { sampleCount: 4 });
    await assert.rejects(ctx.getMeasure('score'), diagnostic);
    await assert.rejects(ctx.getMeasure('M'), diagnostic);
    await assert.rejects(ctx.getMeasure('draw'), diagnostic);
  }
});

test('extreme common scales preserve the two-dimensional coordinate density', async () => {
  const { mu, sigma, point, logdensity } = oracle.zerosum_normal;
  for (const scale of [1e-200, 1e200]) {
    const scaled = (values: number[]) => JSON.stringify(values.map(value => value * scale));
    const { ctx } = makeMatCtx(`
M = tilt(Normal.(${scaled(mu)}, ${scaled(sigma)}), Lebesgue(zerosum(3)))
score = logdensityof(M, ${scaled(point)})
`, { sampleCount: 4 });
    close((await ctx.getMeasure('score')).samples[0], logdensity - 2 * Math.log(scale), 2e-10);
  }
});
