// packages/engine/test/emcee-kernel.test.ts
const { test } = require('node:test');
const assert = require('node:assert/strict');
const MV = require('../model-view.ts');
const { runMcmc } = require('../mcmc-driver.ts');
const { makeEmceeKernel } = require('../emcee-kernel.ts');

function logNormal(x: number, m: number, s: number) { return -0.5*Math.log(2*Math.PI) - Math.log(s) - 0.5*((x-m)/s)**2; }

test('emcee recovers Normal-Normal posterior (1D)', () => {
  const mv = MV.buildModelView({
    latents: [{ name:'mu', distOp:'Normal', params:{mu:0,sigma:10}, support:{kind:'real'}, discrete:false }],
    logLikelihood: (th: any) => logNormal(5.0, th.mu, 1),
  });
  const r = runMcmc(mv, makeEmceeKernel(), { nWalkers: 8, warmup: 400, draws: 1000, seed: 3 });
  const draws = r.drawsByName.mu;
  let m = 0; for (const v of draws) m += v; m /= draws.length;
  const postVar = 1/(1+1/100), postMean = 5*postVar;
  assert.ok(Math.abs(m - postMean) < 0.12, `mean ${m} vs ${postMean}`);
  assert.ok(r.diagnostics.acceptRate > 0.1 && r.diagnostics.acceptRate < 0.95, `accept ${r.diagnostics.acceptRate}`);
});

// Affine-invariance payoff: a correlated 2D Gaussian. Two latents mu,nu with
// a target logπ = -0.5 * quadratic form with correlation 0.95. emcee should
// recover both marginal variances and the correlation without tuning.
test('emcee recovers a correlated 2D Gaussian (marginals + correlation)', () => {
  const rho = 0.95, s = 1;
  const inv00 = 1/(s*s*(1-rho*rho)), inv11 = inv00, inv01 = -rho/(s*s*(1-rho*rho));
  const mv = MV.buildModelView({
    latents: [
      { name:'x', distOp:'Normal', params:{mu:0,sigma:100}, support:{kind:'real'}, discrete:false },
      { name:'y', distOp:'Normal', params:{mu:0,sigma:100}, support:{kind:'real'}, discrete:false },
    ],
    // Likelihood IS the correlated bivariate normal core (flat-ish priors above).
    logLikelihood: (th: any) => -0.5*(inv00*th.x*th.x + 2*inv01*th.x*th.y + inv11*th.y*th.y),
  });
  const r = runMcmc(mv, makeEmceeKernel(), { nWalkers: 20, warmup: 1000, draws: 3000, seed: 11 });
  const xs = r.drawsByName.x, ys = r.drawsByName.y, n = xs.length;
  let mx=0,my=0; for (let i=0;i<n;i++){mx+=xs[i];my+=ys[i];} mx/=n; my/=n;
  let vx=0,vy=0,cxy=0;
  for (let i=0;i<n;i++){ const dx=xs[i]-mx, dy=ys[i]-my; vx+=dx*dx; vy+=dy*dy; cxy+=dx*dy; }
  vx/=n-1; vy/=n-1; cxy/=n-1;
  const corr = cxy/Math.sqrt(vx*vy);
  assert.ok(Math.abs(corr - rho) < 0.1, `recovered corr ${corr} vs ${rho}`);
  assert.ok(Math.abs(vx - 1) < 0.25 && Math.abs(vy - 1) < 0.25, `marginal vars ${vx},${vy} vs 1`);
});

test('emcee is deterministic for a fixed seed', () => {
  const spec = { latents:[{name:'mu',distOp:'Normal',params:{mu:0,sigma:10},support:{kind:'real'},discrete:false}], logLikelihood:(th: any)=>logNormal(5.0,th.mu,1) };
  const a = runMcmc(MV.buildModelView(spec), makeEmceeKernel(), { nWalkers:8, warmup:200, draws:200, seed:9 });
  const b = runMcmc(MV.buildModelView(spec), makeEmceeKernel(), { nWalkers:8, warmup:200, draws:200, seed:9 });
  assert.deepEqual(Array.from(a.drawsByName.mu), Array.from(b.drawsByName.mu));
});

test('emcee rejects odd or too-small walker counts', () => {
  const spec = { latents:[{name:'mu',distOp:'Normal',params:{mu:0,sigma:10},support:{kind:'real'},discrete:false}], logLikelihood:()=>0 };
  assert.throws(() => runMcmc(MV.buildModelView(spec), makeEmceeKernel(), { nWalkers:3, warmup:1, draws:1, seed:0 }), /even and >= 4/);
});
