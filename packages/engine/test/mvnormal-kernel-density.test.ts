'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

const model = `
  a~Uniform(interval(1,4))
  covariance=a^2*eye(2)
  y~MvNormal([0.,0.],covariance)
  K=kernelof(y,a=a)
  L=likelihoodof(pars -> K(a = pars.a),[1.,2.])
`;
const logLikelihood = (a: number) => -Math.log(2 * Math.PI) - 2 * Math.log(a) - 5 / (2 * a * a);

test('MvNormal kernel scores a covariance derived from its supplied boundary', async () => {
  const { ctx } = makeMatCtx(model + `
    p=logdensityof(L,record(a=2.))
    q=logdensityof(L,record(a=3.))
  `, { sampleCount: 4 });
  for (const [name, a] of [['p', 2], ['q', 3]] as const) {
    for (const score of (await ctx.getMeasure(name)).samples) {
      assert.ok(Math.abs(score - logLikelihood(a)) < 1e-12);
    }
  }
});

test('MvNormal posterior weights use each retained prior atom covariance', async () => {
  const { ctx } = makeMatCtx(model + `
    prior=lawof(record(a=a))
    posterior=bayesupdate(L,prior)
  `, { sampleCount: 8 });
  const post = await ctx.getMeasure('posterior');
  const a = post.fields.a.samples;
  for (let i = 1; i < a.length; i++) {
    const actual = post.logWeights[i] - post.logWeights[0];
    const expected = logLikelihood(a[i]) - logLikelihood(a[0]);
    assert.ok(Math.abs(actual - expected) < 1e-12);
  }
});

test('MvNormal kernel inputs preserve vector means and matrix covariances', async () => {
  const { ctx } = makeMatCtx(`
    mu=elementof(cartpow(reals,2))
    cov=elementof(cartpow(reals,[2,2]))
    y~MvNormal(mu,cov)
    K=kernelof(y,mu=mu,cov=cov)
    L=likelihoodof(K,[1.,2.])
    p=logdensityof(L,record(mu=[1.,-1.],cov=rowstack([[4.,1.],[1.,2.]])))
  `, { sampleCount: 3 });
  // det(cov)=7; (y-mu)' inv(cov) (y-mu)=36/7.
  const expected = -Math.log(2 * Math.PI) - 0.5 * Math.log(7) - 18 / 7;
  for (const score of (await ctx.getMeasure('p')).samples) {
    assert.ok(Math.abs(score - expected) < 1e-12);
  }
});
