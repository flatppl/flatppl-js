'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const model = `
  s ~ Exponential(2)
  b ~ iid(normalize(truncate(Normal(0,1),interval(0,s))),2)
  prior = lawof(record(b=b,s=s))
  y ~ Normal(sum(b),s)
  K = kernelof(y,b=b,s=s)
  L = likelihoodof(K,1)
  posterior = bayesupdate(L,prior)
`;

test('dependent truncation scores its current bound in reified priors and posterior inference', async () => {
  const { ctx } = ctxFor(model, 3);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  // b=[.25,.5]. Prior = sum(log phi(b)) - 2 log(Phi(s)-1/2)
  // + log(2)-2s; posterior adds log N(1; .75,s). Phi(s)-1/2 was
  // independently integrated by its convergent exp(-x²/2) power series.
  for (const [s, prior, posterior] of [
    [1, -1.1512552321252576, -2.1014437653299303],
    [2, -3.8215497001447294, -5.4414479139093475],
  ]) {
    const { ctx: scored } = ctxFor(model + `
      lp = logdensityof(prior,record(b=[.25,.5],s=${s}))
      post_lp = logdensityof(posterior,record(b=[.25,.5],s=${s}))
    `, 3);
    for (const [name, expected] of [['lp', prior], ['post_lp', posterior]] as const) {
      const values = (await scored.getMeasure(name)).samples;
      for (const actual of values) assert.ok(Math.abs(actual - expected) < 1e-12);
    }
    const point = { b: [.25, .5], s };
    assert.ok(Math.abs(scorer.priorOf(point) - prior) < 1e-12);
    assert.ok(Math.abs(scorer.logPi(point) - posterior) < 1e-12);
  }
  const { ctx: outside } = ctxFor(model
    + '\nlp=logdensityof(prior,record(b=[.25,.5],s=.4))', 1);
  assert.equal((await outside.getMeasure('lp')).samples[0], -Infinity);
  assert.equal(scorer.priorOf({ b: [.25, .5], s: .4 }), -Infinity);
});

function weightedMean(values: number[], logWeights: Float64Array | null): number {
  const peak = logWeights ? Math.max(...logWeights) : 0;
  let sum = 0, mass = 0;
  for (let i = 0; i < values.length; i++) {
    const weight = logWeights ? Math.exp(logWeights[i] - peak) : 1;
    sum += weight * values[i]; mass += weight;
  }
  return sum / mass;
}

test('dependent bound sampling keeps conditional CDFs and shared prior weights', async () => {
  const { ctx } = ctxFor(`
    s ~ normalize(weighted(fn(2*_),Uniform(interval(.25,.75))))
    one ~ normalize(truncate(Beta(2,2),interval(0,s)))
    copies ~ iid(normalize(truncate(Beta(2,2),interval(0,s))),2)
    raw = iid(truncate(Beta(2,2),interval(0,s)),2)
    y ~ normalize(truncate(Beta(2,2),interval(0,s)))
    prior = lawof(s)
    K = kernelof(y,s=s)
    L = likelihoodof(K,.2)
    posterior = bayesupdate(L,prior)
  `, 2048);
  const bound = await ctx.getMeasure('s');
  const s: number[] = Array.from(bound.samples);
  const cdf = (x: number) => x * x * (3 - 2 * x);
  for (const [name, copies] of [['one', 1], ['copies', 2]] as const) {
    const sampled = await ctx.getMeasure(name);
    const pit = s.map((hi, i) => {
      let sum = 0;
      for (let j = 0; j < copies; j++) {
        const x = sampled.samples[i * copies + j];
        assert.ok(x >= 0 && x <= hi);
        sum += cdf(x) / cdf(hi);
      }
      return sum / copies;
    });
    // Probability integral transforms are Uniform(0,1). The bound's density
    // is proportional to s on [.25,.75], giving E[s]=13/24 rather than 1/2.
    assert.ok(Math.abs(weightedMean(pit, sampled.logWeights) - .5) < .025);
    assert.ok(Math.abs(weightedMean(s, sampled.logWeights) - 13 / 24) < .015);
    assert.ok(Math.abs(sampled.logTotalmass) < 1e-12);
  }
  const raw = await ctx.getMeasure('raw');
  const masses = s.map(hi => cdf(hi) ** 2);
  assert.ok(Math.abs(Math.exp(raw.logTotalmass) - weightedMean(masses, bound.logWeights)) < 1e-12);
  for (let i = 0; i < s.length; i++) {
    const relative = raw.logWeights[i] - raw.logWeights[0]
      - bound.logWeights[i] + bound.logWeights[0];
    assert.ok(Math.abs(relative - Math.log(masses[i] / masses[0])) < 1e-12);
  }
  const posterior = await ctx.getMeasure('posterior');
  for (let i = 0; i < s.length; i++) {
    const relative = posterior.logWeights[i] - posterior.logWeights[0]
      - bound.logWeights[i] + bound.logWeights[0];
    assert.ok(Math.abs(relative - Math.log(cdf(s[0]) / cdf(s[i]))) < 1e-12);
  }
});

test('indexed vector bounds preserve conditional laws, IID axes and prior weights', async () => {
  const N = 1024;
  const { ctx } = ctxFor(`
    s ~ iid(normalize(weighted(fn(2*_),Uniform(interval(.25,.75)))),2)
    one ~ normalize(truncate(Beta(2,2),interval(0,s[1])))
    copies ~ iid(normalize(truncate(Beta(2,2),interval(0,s[1]))),[2,2])
    raw = iid(truncate(Beta(2,2),interval(0,s[1])),[2,2])
  `, N);
  const parent = await ctx.getMeasure('s');
  const bounds = Array.from({ length: N }, (_, i) => parent.samples[2 * i]);
  const cdf = (x: number) => x * x * (3 - 2 * x);
  for (const [name, copies, shape] of [
    ['one', 1, [N]], ['copies', 4, [N, 2, 2]],
  ] as const) {
    const sampled = await ctx.getMeasure(name);
    assert.deepEqual(sampled.value.shape, shape);
    const pit = bounds.map((hi, i) => {
      let sum = 0;
      for (let j = 0; j < copies; j++) {
        const x = sampled.samples[i * copies + j];
        assert.ok(x >= 0 && x <= hi);
        sum += cdf(x) / cdf(hi);
      }
      const relative = sampled.logWeights[i] - sampled.logWeights[0];
      assert.ok(Math.abs(relative - parent.logWeights[i] + parent.logWeights[0]) < 1e-12);
      return sum / copies;
    });
    assert.ok(Math.abs(weightedMean(pit, sampled.logWeights) - .5) < .03);
  }
  const raw = await ctx.getMeasure('raw');
  const masses = bounds.map(hi => cdf(hi) ** 4);
  assert.ok(Math.abs(Math.exp(raw.logTotalmass) - weightedMean(masses, parent.logWeights)) < 1e-12);
  for (let i = 0; i < N; i++) {
    const relative = raw.logWeights[i] - raw.logWeights[0]
      - parent.logWeights[i] + parent.logWeights[0];
    assert.ok(Math.abs(relative - Math.log(masses[i] / masses[0])) < 1e-12);
  }
});

test('a truncated Normal reads its mean from the current vector atom', async () => {
  const N = 1024;
  const { ctx } = ctxFor(`
    s ~ iid(Uniform(interval(.25,.75)),2)
    x ~ normalize(truncate(Normal(s[2],1),interval(0,1)))
  `, N);
  const s = (await ctx.getMeasure('s')).samples;
  const x = (await ctx.getMeasure('x')).samples;
  // Integral of phi(t) from 0 to z by its convergent power series.
  // All arguments lie in [-.75,.75], so 20 terms suffice in double precision.
  const area = (z: number) => {
    let term = z, sum = z;
    for (let k = 1; k < 20; k++) {
      term *= -z * z / (2 * k);
      sum += term / (2 * k + 1);
    }
    return sum / Math.sqrt(2 * Math.PI);
  };
  const pit = Array.from(x, (value: number, i) => {
    const mu = s[2 * i + 1];
    assert.ok(value >= 0 && value <= 1);
    return (area(value - mu) - area(-mu)) / (area(1 - mu) - area(-mu));
  });
  assert.ok(Math.abs(weightedMean(pit, null) - .5) < .03);
});

test('parametric and rejection sampling retain vector parameters across IID repeats', async () => {
  const N = 2048;
  const { ctx } = ctxFor(`
    p ~ iid(Uniform(interval(.25,.75)),2)
    direct ~ Binomial(2,p[1])
    direct_copies ~ iid(Binomial(2,p[1]),2)
    one ~ iid(normalize(truncate(Binomial(2,p[1]),interval(0,1))),1)
    copies ~ iid(normalize(truncate(Binomial(2,p[1]),interval(0,1))),2)
  `, N);
  const p = (await ctx.getMeasure('p')).samples;
  for (const [name, copies] of [['direct', 1], ['direct_copies', 2]] as const) {
    const direct = (await ctx.getMeasure(name)).samples;
    const residuals = Array.from(direct, (x: number, i) => x - 2 * p[2 * Math.floor(i / copies)]);
    assert.ok(Math.abs(weightedMean(residuals, null)) < .035);
  }
  for (const [name, copies] of [['one', 1], ['copies', 2]] as const) {
    const sampled = await ctx.getMeasure(name);
    let residual = 0;
    for (let i = 0; i < N; i++) {
      // Binomial(2,p) conditioned on X<=1 has P(X=1)=2p/(1+p).
      const probability = 2 * p[2 * i] / (1 + p[2 * i]);
      for (let j = 0; j < copies; j++) {
        const x = sampled.samples[i * copies + j];
        assert.ok(x === 0 || x === 1);
        residual += x - probability;
      }
    }
    assert.ok(Math.abs(residual / (N * copies)) < .035);
  }
});
