// packages/engine/test/mh-sample.test.ts
const { test } = require('node:test');
const assert = require('node:assert/strict');
const MV = require('../model-view.ts');
const MH = require('../mh-sample.ts');

function logNormal(x: number, m: number, s: number) {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
}

// Conjugate: mu ~ N(0, 10^2); y ~ N(mu, 1), observe y=5.
// Posterior mu | y ~ N( (5/1)/(1/1 + 1/100), 1/(1 + 1/100) ).
test('mhSample recovers Normal-Normal conjugate posterior mean/var', () => {
  const spec = {
    latents: [{ name: 'mu', distOp: 'Normal', params: { mu: 0, sigma: 10 }, support: { kind: 'real' }, discrete: false }],
    logLikelihood: (th: any) => logNormal(5.0, th.mu, 1),
  };
  const mv = MV.buildModelView(spec);
  const post = MH.mhSample(mv, { chains: 4, warmup: 1000, draws: 2000, seed: 7 });
  const draws = post.drawsByName.mu;
  let m = 0; for (const v of draws) m += v; m /= draws.length;
  let v = 0; for (const d of draws) v += (d - m) ** 2; v /= draws.length - 1;
  const postVar = 1 / (1 + 1 / 100), postMean = 5 * postVar;
  assert.ok(Math.abs(m - postMean) < 0.1, `mean ${m} vs ${postMean}`);
  assert.ok(Math.abs(v - postVar) < 0.1, `var ${v} vs ${postVar}`);
  assert.ok(post.acceptRate > 0.1 && post.acceptRate < 0.9, `accept ${post.acceptRate}`);
});

test('mhSample is deterministic for a fixed seed', () => {
  const spec = {
    latents: [{ name: 'mu', distOp: 'Normal', params: { mu: 0, sigma: 10 }, support: { kind: 'real' }, discrete: false }],
    logLikelihood: (th: any) => logNormal(5.0, th.mu, 1),
  };
  const a = MH.mhSample(MV.buildModelView(spec), { chains: 2, warmup: 200, draws: 200, seed: 42 });
  const b = MH.mhSample(MV.buildModelView(spec), { chains: 2, warmup: 200, draws: 200, seed: 42 });
  assert.deepEqual(Array.from(a.drawsByName.mu), Array.from(b.drawsByName.mu));
});

// sigma ~ HalfNormal(sigma=2); y ~ Normal(0, sigma), observe y=1.5. Check MH stays
// in (0,inf) and posterior mean is finite + positive (exercises the positive transform).
test('mhSample respects positive support via the transform', () => {
  function logNormal2(x: number, m: number, s: number) {
    return -0.5 * Math.log(2 * Math.PI) - Math.log(s) - 0.5 * ((x - m) / s) ** 2;
  }
  const spec = {
    latents: [{ name: 'sigma', distOp: 'HalfNormal', params: { sigma: 2 }, support: { kind: 'positive' }, discrete: false }],
    logLikelihood: (th: any) => logNormal2(1.5, 0, th.sigma),
  };
  const mv = MV.buildModelView(spec);
  const post = MH.mhSample(mv, { chains: 4, warmup: 1000, draws: 2000, seed: 3 });
  const d = post.drawsByName.sigma;
  for (let i = 0; i < d.length; i++) assert.ok(d[i] > 0, `draw ${d[i]} not positive`);
  let m = 0; for (const v of d) m += v; m /= d.length;
  assert.ok(m > 0.3 && m < 5, `posterior mean sigma ${m} implausible`);
});
