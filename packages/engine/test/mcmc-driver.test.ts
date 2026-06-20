// packages/engine/test/mcmc-driver.test.ts
const { test } = require('node:test');
const assert = require('node:assert/strict');
const MV = require('../model-view.ts');
const { runMcmc } = require('../mcmc-driver.ts');
const { mhKernel } = require('../mh-kernel.ts');

function logNormal(x: number, m: number, s: number) { return -0.5*Math.log(2*Math.PI) - Math.log(s) - 0.5*((x-m)/s)**2; }
function specNormalNormal(obs: number) {
  return {
    latents: [{ name:'mu', distOp:'Normal', params:{mu:0,sigma:10}, support:{kind:'real'}, discrete:false }],
    logLikelihood: (th: any) => logNormal(obs, th.mu, 1),
  };
}

test('runMcmc+mhKernel recovers Normal-Normal posterior; output shape correct', () => {
  const mv = MV.buildModelView(specNormalNormal(5.0));
  const r = runMcmc(mv, mhKernel, { nWalkers: 4, warmup: 1000, draws: 2000, seed: 7 });
  const draws = r.drawsByName.mu;
  assert.equal(draws.length, 4 * 2000);
  assert.equal(r.walkers.mu.length, 4);
  assert.equal(r.walkers.mu[0].length, 2000);
  let m = 0; for (const v of draws) m += v; m /= draws.length;
  let v = 0; for (const d of draws) v += (d-m)**2; v /= draws.length-1;
  const postVar = 1/(1+1/100), postMean = 5*postVar;
  assert.ok(Math.abs(m - postMean) < 0.1, `mean ${m} vs ${postMean}`);
  assert.ok(Math.abs(v - postVar) < 0.1, `var ${v} vs ${postVar}`);
  assert.ok(r.diagnostics.perParam.mu.rHat < 1.2, `rHat ${r.diagnostics.perParam.mu.rHat}`);
  assert.ok(r.diagnostics.acceptRate > 0.1 && r.diagnostics.acceptRate < 0.9);
});

test('runMcmc is deterministic for a fixed seed', () => {
  const a = runMcmc(MV.buildModelView(specNormalNormal(5.0)), mhKernel, { nWalkers:2, warmup:200, draws:200, seed:42 });
  const b = runMcmc(MV.buildModelView(specNormalNormal(5.0)), mhKernel, { nWalkers:2, warmup:200, draws:200, seed:42 });
  assert.deepEqual(Array.from(a.drawsByName.mu), Array.from(b.drawsByName.mu));
});
