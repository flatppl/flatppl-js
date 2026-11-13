'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { makeMatCtx } = require('./_materialise-helpers.ts');
const { buildLogPi } = require('../mcmc-density.ts');

const factorialLog = (n: number) => Array.from({ length: n }, (_, i) => Math.log(i + 1))
  .reduce((a, b) => a + b, 0);
const likelihoodConstant = 12 * Math.log(2) - factorialLog(7) - factorialLog(12);

for (const shape of ['vector', 'record']) {
  for (const restriction of ['restrict', 'disintegrate']) {
    test(`${restriction} scores only the unobserved prior for a ${shape} observation`, async () => {
      const law = shape === 'vector' ? 'Poisson.([s, 2*s])'
        : 'joint(on=Poisson(s), off=Poisson(2*s))';
      const value = shape === 'vector' ? '[7,12]' : 'record(on=7,off=12)';
      const posterior = restriction === 'restrict'
        ? `posterior=restrict(model,record(y=${value}))`
        : `K,P=disintegrate(["y"],model)\nL=likelihoodof(K,record(y=${value}))\nposterior=bayesupdate(L,P)`;
      const { ctx } = makeMatCtx(`s~Gamma(2,1)\ny~${law}\nmodel=lawof(record(s=s,y=y))\n${posterior}`);
      const score = await buildLogPi(ctx, ctx.derivations.posterior);
      // Gamma(2,1) times the two Poisson masses gives Gamma(21,4), up to its evidence.
      for (const s of [1, 3.5]) {
        const expected = 20 * Math.log(s) - 4 * s + likelihoodConstant;
        assert.ok(Math.abs(score.logPi({ s }) - expected) < 1e-12);
        assert.ok(Math.abs(score.logPiBatch([{ s }])[0] - expected) < 1e-12);
      }
    });
  }
}

test('conditioning retains a genuine unobserved ancestor of a prior field', async () => {
  const { ctx } = makeMatCtx(`scale~Gamma(2,1)
s~Gamma(2,scale)
y~Poisson.([s,2*s])
model=lawof(record(s=s,y=y))
posterior=restrict(model,record(y=[7,12]))`);
  const score = await buildLogPi(ctx, ctx.derivations.posterior);
  for (const [scale, s] of [[2, 3], [0.5, 4]]) {
    const expected = 3 * Math.log(scale) - scale - scale * s
      + 20 * Math.log(s) - 3 * s + likelihoodConstant;
    assert.ok(Math.abs(score.logPi({ scale, s }) - expected) < 1e-12);
  }
});
