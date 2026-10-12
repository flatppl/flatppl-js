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
