'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

const SOURCE = `
cov = rowstack([[1.0, 0.5], [0.5, 1.0]])
prior = relabel(MvNormal(mu = [0.0, 0.0], cov = cov), ["x", "y"])
K = (x, y) -> Normal(mu = x, sigma = 1.0)
L = likelihoodof(K, 1.0)
posterior = bayesupdate(L, prior)
`;

const OPTIONS = {
  chains: 4, walkers: 12, warmup: 500, draws: 1500, seed: 1,
  amisSamples: 500, amisIters: 12, smcParticles: 4000, smcSteps: 12,
  nLive: 400, dlogz: 0.05, regionMetric: 'off',
};

// Gaussian conditioning: m = Σ[:,1]/2, V = Σ - Σ[:,1]Σ[1,:]/2.
// This gives m=[.5,.25], V=[[.5,.25],[.25,.875]].
for (const backend of ['is', 'mh', 'ram', 'slice', 'emcee', 'demcz', 'amis', 'smc', 'nested',
  'elliptical-slice-sampler']) {
  test(`${backend} preserves the joint law of a relabelled Gaussian prior`, async () => {
    const { ctx, proc } = ctxFor(SOURCE, 4000);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const m = await materialiser.materialiseMeasure('posterior', ctx, { ...OPTIONS, backend });
    assert.deepEqual(Object.keys(m.fields), ['x', 'y']);
    const values = [m.fields.x.samples, m.fields.y.samples];
    let total = 0;
    const mean = [0, 0], second = [[0, 0], [0, 0]];
    for (let i = 0; i < values[0].length; i++) {
      const w = m.logWeights ? Math.exp(m.logWeights[i]) : 1;
      total += w;
      for (let j = 0; j < 2; j++) {
        mean[j] += w * values[j][i];
        for (let k = 0; k < 2; k++) second[j][k] += w * values[j][i] * values[k][i];
      }
    }
    for (let j = 0; j < 2; j++) mean[j] /= total;
    const covariance = second.map((row, j) => row.map((v, k) => v / total - mean[j] * mean[k]));
    const wantMean = [0.5, 0.25], wantCov = [[0.5, 0.25], [0.25, 0.875]];
    const diagnostics = m.diagnostics?.perParam || {};
    const ess = Math.min(4000, m.n_eff ?? 4000, diagnostics.x?.essBulk ?? 4000, diagnostics.y?.essBulk ?? 4000);
    for (let j = 0; j < 2; j++) {
      assert.ok(Math.abs(mean[j] - wantMean[j]) < 7 * Math.sqrt(wantCov[j][j] / ess),
        `${backend}: mean[${j}]=${mean[j]}`);
      for (let k = 0; k < 2; k++) {
        const se = Math.sqrt((wantCov[j][k] ** 2 + wantCov[j][j] * wantCov[k][k]) / ess);
        assert.ok(Math.abs(covariance[j][k] - wantCov[j][k]) < 7 * se,
          `${backend}: covariance[${j},${k}]=${covariance[j][k]}`);
      }
    }
    if (m.diagnostics?.perParam) assert.deepEqual(Object.keys(m.diagnostics.perParam), ['x', 'y']);
    if (m.diagnostics?.rawChains) assert.deepEqual(Object.keys(m.diagnostics.rawChains), ['x', 'y']);
    if (backend === 'nested') {
      // Evidence is N(1;0,sqrt(2)), independent of the output field names.
      const logZ = -0.5 * Math.log(4 * Math.PI) - 0.25;
      assert.ok(Math.abs(m.logTotalmass - logZ) < 0.2);
    }
  });
}

for (const [label, source, want] of [
  ['source aliases and captured module names', SOURCE.replace('prior =', 'original =')
    .replace('K = (x, y) -> Normal(mu = x, sigma = 1.0)', `
prior = original
x = 20.0
a = elementof(reals)
b = elementof(reals)
model = Normal(mu = a + x, sigma = 1.0)
K = functionof(model, y = b, x = a)`)
    .replace('likelihoodof(K, 1.0)', 'likelihoodof(K, 21.0)'), [0.5, 0.25]],
  ['a parameter-dependent normalizer', SOURCE.replace('Normal(mu = x, sigma = 1.0)',
    'normalize(weighted(exp(x), Normal(mu = x, sigma = 1.0)))'), [0.5, 0.25]],
  ['nested placeholder scope', SOURCE.replace('Normal(mu = x, sigma = 1.0)',
    'logweighted(x -> -x*x, Normal(mu = x, sigma = 1.0))'), [0.5, 0.25]],
  ['captured bijection parameters', SOURCE.replace('Normal(mu = x, sigma = 1.0)',
    'MvNormal(mu = [x, y], cov = cov)').replace('likelihoodof(K, 1.0)',
    'likelihoodof(K, [1.0, 0.0])'), [0.5, 0.0]],
] as Array<[string, string, number[]]>) {
  test(`relabelled prior inference preserves ${label}`, async () => {
    const { ctx, proc } = ctxFor(source, 4000);
    assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
    const m = await materialiser.materialiseMeasure('posterior', ctx, { ...OPTIONS, backend: 'mh' });
    for (const [i, name] of ['x', 'y'].entries()) {
      const s = m.fields[name].samples;
      const mean = s.reduce((a: number, b: number) => a + b, 0) / s.length;
      const ess = Math.min(s.length, m.diagnostics.perParam[name].essBulk);
      assert.ok(Math.abs(mean - want[i]) < 7 / Math.sqrt(ess), `${name}: ${mean} vs ${want[i]}`);
    }
  });
}

for (const backend of ['mh', 'nested']) {
  test(`${backend} keeps the simplex chart of a relabelled Dirichlet prior`, async () => {
    const { ctx } = ctxFor(`
prior = relabel(Dirichlet(alpha = [2.0, 3.0, 4.0]), ["a", "b", "c"])
K = (a, b, c) -> Bernoulli(p = a)
L = likelihoodof(K, true)
posterior = bayesupdate(L, prior)
`, 4000);
    const m = await materialiser.materialiseMeasure('posterior', ctx, { ...OPTIONS, backend });
    // Conjugacy gives Dirichlet(3,3,4), including a+b+c=1 on every draw.
    const values = ['a', 'b', 'c'].map(name => m.fields[name].samples);
    for (let i = 0; i < 4000; i++) {
      assert.ok(Math.abs(values[0][i] + values[1][i] + values[2][i] - 1) < 1e-12);
    }
    for (let j = 0; j < 3; j++) {
      const mean = values[j].reduce((a: number, b: number) => a + b, 0) / 4000;
      assert.ok(Math.abs(mean - [0.3, 0.3, 0.4][j]) < 0.035);
    }
    if (backend === 'nested') assert.ok(Math.abs(m.logTotalmass - Math.log(2 / 9)) < 0.2);
  });
}
