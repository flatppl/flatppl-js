'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

// Independent count PMFs and their normalized parameter derivatives.
function poisson(rate: number, count: number): [number, number[]] {
  let factorial = 0;
  for (let j = 1; j <= count; j++) factorial += Math.log(j);
  const mass = 1 - Math.exp(-rate);
  return [count * Math.log(rate) - rate - factorial - Math.log(mass),
    [count / rate - 1 - Math.exp(-rate) / mass, 0, 0]];
}

function nb(shape: number, rate: number, count: number): [number, number[]] {
  const q = rate / (1 + rate), mass = 1 - q ** shape;
  let coefficient = 0, shapeDerivative = 0;
  for (let j = 0; j < count; j++) {
    coefficient += Math.log((shape + j) / (j + 1));
    shapeDerivative += 1 / (shape + j);
  }
  return [coefficient + shape * Math.log(q) - count * Math.log1p(rate) - Math.log(mass),
    [0, shapeDerivative + Math.log(q) / mass,
      shape / (rate * (1 + rate) * mass) - count / (1 + rate)]];
}

const model = `
positive_poisson(r) = normalize(truncate(Poisson(r), interval(1, inf)))
positive_nb(a, b) = normalize(truncate(NegativeBinomial(alpha=a, beta=b), interval(1, inf)))
r ~ Exponential(1)
a ~ Gamma(2, 1)
b ~ Gamma(2, 1)
y ~ positive_poisson.([r, 2*r])
z ~ positive_nb.([a, a+1], b)
K = kernelof(record(y=y, z=z), r=r, a=a, b=b)
L = likelihoodof(K, record(y=[2,3], z=[2,3]))
P = bayesupdate(L, lawof(record(r=r, a=a, b=b)))
`;

function oracle(point: number[]): [number[], number[][]] {
  const [r, a, b] = point;
  const [p, dp] = poisson(r, 2), [p2, dp2] = poisson(2*r, 3);
  dp2[0] *= 2;
  const [n, dn] = nb(a, b, 2), [n2, dn2] = nb(a+1, b, 3);
  return [[p, n, p+p2+n+n2, p, p2, n, n2],
    [dp, dn, dp.map((x, j) => x+dp2[j]+dn[j]+dn2[j]), dp, dp2, dn, dn2]];
}

async function scores(point: number[]): Promise<number[]> {
  const [r, a, b] = point;
  const { ctx } = makeMatCtx(model + `
direct_p = logdensityof(positive_poisson(${r}), 2)
direct_n = logdensityof(positive_nb(${a}, ${b}), 2)
conditional = logdensityof(L, record(r=${r}, a=${a}, b=${b}))
pointwise_p = ((v, k) -> logdensityof(positive_poisson(v), k)).([${r}, ${2*r}], [2,3])
pointwise_n = ((v, k) -> logdensityof(positive_nb(v, ${b}), k)).([${a}, ${a+1}], [2,3])
`, { sampleCount: 1 });
  const result: number[] = [];
  for (const name of ['direct_p', 'direct_n', 'conditional', 'pointwise_p', 'pointwise_n']) {
    const value = await ctx.getMeasure(name);
    result.push(...value.samples);
  }
  return result;
}

test('positive-count densities retain exact live masses in direct, kernel and pointwise paths', async () => {
  for (const point of [[.01, 2, 50], [.2, .7, 2], [3, 2.3, .6]]) {
    const [expected, derivatives] = oracle(point);
    const actual = await scores(point);
    assert.equal(actual.length, 7);
    for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i]-expected[i]) < 1e-11);
    for (let j = 0; j < point.length; j++) {
      const step = 1e-6 * Math.max(.1, point[j]);
      const lo = point.slice(), hi = point.slice();
      lo[j] -= step; hi[j] += step;
      const lower = await scores(lo), upper = await scores(hi);
      for (let i = 0; i < actual.length; i++) {
        const derivative = (upper[i]-lower[i])/(2*step);
        assert.ok(Math.abs(derivative-derivatives[i][j]) < 1e-6 * Math.max(1, Math.abs(derivatives[i][j])));
      }
    }
  }
});

test('MCMC likelihoods use the same positive-count masses at each parameter point', async () => {
  const { ctx } = makeMatCtx(model, { sampleCount: 16 });
  const derivation = Object.values(ctx.derivations).find((d: any) => d.kind === 'bayesupdate');
  const { likOf } = await buildLogPi(ctx, derivation);
  for (const point of [[.01, 2, 50], [.2, .7, 2], [3, 2.3, .6]]) {
    const [r, a, b] = point;
    assert.ok(Math.abs(likOf({ r, a, b }) - oracle(point)[0][2]) < 1e-11);
  }
});

test('positive-count normalization distinguishes excluded counts from undefined laws', async () => {
  for (const base of ['Poisson(.2)', 'NegativeBinomial(2,3)']) {
    const { ctx } = makeMatCtx(`r=logdensityof(normalize(truncate(${base},interval(1,inf))),0)`, { sampleCount: 1 });
    assert.equal((await ctx.getMeasure('r')).samples[0], -Infinity);
  }
  for (const invalid of ['0', '-2', 'inf', '-inf']) {
    for (const base of ['Poisson(v)', 'NegativeBinomial(v,3)', 'NegativeBinomial(2,v)']) {
      for (const [call, point] of [[`K(${invalid})`, '2'], [`K.([${invalid},${invalid}])`, '[2,2]']]) {
        const { ctx } = makeMatCtx(`K(v)=normalize(truncate(${base},interval(1,inf)))
r=logdensityof(${call},${point})`, { sampleCount: 1 });
        await assert.rejects(() => ctx.getMeasure('r'), /normaliz|undefined|mass|derivation/i);
      }
    }
  }
});
