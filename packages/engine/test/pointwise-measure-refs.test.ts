'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');

test('reified pointwise scores expand held measures without sampling their parameters', async () => {
  const { ctx } = makeMatCtx(`
mu ~ Normal(0,1)
hu ~ Beta(1,1)
rates = exp.([0,.2] .+ mu)
data = [0,2]
count_rows = functionof(((rate,x) -> logdensityof(normalize(superpose(weighted(hu,Dirac(0)),weighted(1-hu,Poisson(rate)))),x)).(rates,data),mu=mu,hu=hu)
p = [.4,.3]
a = count_rows(mu=p[1],hu=p[2])
q = [-.7,.6]
b = count_rows(mu=q[1],hu=q[2])
c = count_rows(mu=mu,hu=p[2])
positive_score = functionof(logdensityof(normalize(superpose(weighted(hu,Dirac(0)),weighted(1-hu,Poisson(exp(mu))))),0)>0,mu=mu,hu=hu)
d = positive_score(mu=p[1],hu=p[2])
`, { sampleCount: 3 });
  for (const [name, mu, hu] of [['a',.4,.3],['b',-.7,.6]] as Array<[string,number,number]>) {
    const r0 = Math.exp(mu), r1 = Math.exp(mu+.2);
    const expected = [Math.log(hu+(1-hu)*Math.exp(-r0)),
      Math.log1p(-hu)-r1+2*Math.log(r1)-Math.log(2)];
    const actual = (await ctx.getMeasure(name)).samples;
    assert.equal(actual.length, expected.length);
    for (let i=0;i<expected.length;i++) assert.ok(Math.abs(actual[i]-expected[i])<1e-12);
  }
  assert.deepEqual(Array.from((await ctx.getMeasure('d')).samples), [0,0,0]);
  const parent = (await ctx.getMeasure('mu')).samples;
  const stochastic = await ctx.getMeasure('c');
  assert.deepEqual(stochastic.value.shape, [3,2]);
  for (let i=0;i<parent.length;i++) {
    const r0 = Math.exp(parent[i]), r1 = Math.exp(parent[i]+.2);
    assert.ok(Math.abs(stochastic.samples[2*i]-Math.log(.3+.7*Math.exp(-r0)))<1e-12);
    assert.ok(Math.abs(stochastic.samples[2*i+1]
      -(Math.log(.7)-r1+2*Math.log(r1)-Math.log(2)))<1e-12);
  }
});
