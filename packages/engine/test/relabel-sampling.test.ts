'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { clm, materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');

// §04 "Interface adaptation": relabel changes the fields of each whole atom.
const SOURCE = `
n_sig = elementof(interval(0, 150))
sig_cov = rowstack([[1.0, 0.5], [0.5, 1.0]])
sig_shape = relabel(MvNormal(mu = [0.0, 0.0], cov = sig_cov), ["x", "y"])
bg_shape = joint(x = Uniform(support = interval(-5.0, 5.0)), y = Uniform(support = interval(-5.0, 5.0)))
w = n_sig / 150.0
point_dist = superpose(weighted(w, sig_shape), weighted(1 - w, bg_shape))
`;

test('a named relabel preserves each joint vector atom and its mass', async () => {
  const { ctx } = ctxFor(SOURCE.replace('sig_shape = relabel(MvNormal(mu = [0.0, 0.0], cov = sig_cov),',
    'sig = MvNormal(mu = [0.0, 0.0], cov = sig_cov)\nsig_shape = relabel(sig,'), 16);
  const raw = await ctx.getMeasure('sig');
  const named = await ctx.getMeasure('sig_shape');
  for (let i = 0; i < 16; i++) {
    assert.equal(named.fields.x.samples[i], raw.samples[2 * i]);
    assert.equal(named.fields.y.samples[i], raw.samples[2 * i + 1]);
  }
  assert.equal(named.logTotalmass, raw.logTotalmass);
});

test('relabel preserves the weight of each atom and the total mass once', async () => {
  const source = 'base = weighted(3.0, iid(Normal(0.0, 1.0), 2))\n'
    + 'named = relabel(base, ["x", "y"])\n';
  const { ctx } = ctxFor(source, 8);
  const base = await ctx.getMeasure('base');
  const named = await ctx.getMeasure('named');
  assert.equal(named.logWeights, base.logWeights);
  assert.equal(named.logTotalmass, Math.log(3));
  assert.equal(named.n_eff, base.n_eff);
  assert.equal(named.fields.x.logWeights, null);
  assert.equal(named.fields.y.logWeights, null);
});

test('the kernel plot samples and scores the relabelled Gaussian mixture', async () => {
  const N = 4096;
  for (const n_sig of [0, 75, 150]) {
    const { ctx } = ctxFor(SOURCE, N);
    const ir = clm.lowerMeasure('point_dist', ctx, { boundaries: { n_sig } });
    const m = await materialiser.materialiseMeasureIR(ir, ctx);
    const x = m.fields.x.samples, y = m.fields.y.samples;
    const w = n_sig / 150;
    const variance = w + (1 - w) * 25 / 3;
    const fourth = 3 * w + (1 - w) * 125;
    const average = (f: (i: number) => number) => {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += f(i);
      return sum / N;
    };
    for (const a of [x, y]) {
      assert.ok(Math.abs(average(i => a[i])) < 7 * Math.sqrt(variance / N));
      assert.ok(Math.abs(average(i => a[i] ** 2) - variance)
        < 7 * Math.sqrt((fourth - variance ** 2) / N));
    }
    const crossSecond = 1.5 * w + (1 - w) * (25 / 3) ** 2;
    assert.ok(Math.abs(average(i => x[i] * y[i]) - 0.5 * w)
      < 7 * Math.sqrt((crossSecond - (0.5 * w) ** 2) / N));
    for (const [px, py] of [[0, 0], [1, 2]]) {
      const source = SOURCE.replace('elementof(interval(0, 150))', `${n_sig}.0`)
        + `lp = logdensityof(point_dist, record(x = ${px}.0, y = ${py}.0))\n`;
      const score = await ctxFor(source, 1).ctx.getMeasure('lp');
      const normal = Math.exp(-(px ** 2 - px * py + py ** 2) / 1.5) / (Math.PI * Math.sqrt(3));
      const want = Math.log(w * normal + (1 - w) / 100);
      assert.ok(Math.abs(score.samples[0] - want) < 1e-12);
    }
  }
});

// One event at the origin gives p(w | event) ∝ b + (a-b)w on [0,1].
// Integrating that linear density gives both moments without an engine oracle.
const POSTERIOR = SOURCE + `
K = functionof(point_dist, n_sig = n_sig)
L = likelihoodof(K, record(x = 0.0, y = 0.0))
prior = joint(n_sig = Uniform(support = interval(0.0, 150.0)))
posterior = bayesupdate(L, prior)
`;

for (const backend of ['is', 'mh', 'ram', 'slice', 'emcee', 'demcz', 'amis', 'smc', 'nested',
  'elliptical-slice-sampler']) {
  test(`${backend} recovers the relabelled mixture's analytic signal-yield posterior`, async (t: import('node:test').TestContext) => {
    const { ctx } = ctxFor(POSTERIOR, 2000);
    const m = await materialiser.materialiseMeasure('posterior', ctx, {
      backend, chains: 4, walkers: 12, warmup: 300, draws: 1000, seed: 1,
      amisSamples: 400, amisIters: 8, smcParticles: 2000, smcSteps: 8,
      nLive: 200, dlogz: 0.1, regionMetric: 'off',
    });
    const s = m.fields.n_sig.samples;
    let total = 0, first = 0, second = 0;
    for (let i = 0; i < s.length; i++) {
      const weight = m.logWeights ? Math.exp(m.logWeights[i]) : 1;
      total += weight;
      first += weight * s[i];
      second += weight * s[i] ** 2;
    }
    const mean = first / total, sd = Math.sqrt(second / total - mean ** 2);
    const a = 1 / (Math.PI * Math.sqrt(3)), b = 0.01;
    const wantMean = 150 * (2 * a + b) / (3 * (a + b));
    const wantSD = Math.sqrt(150 ** 2 * (3 * a + b) / (6 * (a + b)) - wantMean ** 2);
    const ess = Math.min(s.length, m.n_eff ?? m.diagnostics?.perParam?.n_sig?.essBulk ?? s.length);
    const se = wantSD / Math.sqrt(ess);
    assert.ok(Math.abs(mean - wantMean) < 7 * se, `${backend}: mean ${mean} vs ${wantMean}`);
    assert.ok(Math.abs(sd - wantSD) < 7 * se, `${backend}: SD ${sd} vs ${wantSD}`);
    t.diagnostic(`seed=1, draws=${s.length}, mean=${mean.toFixed(3)}, MCSE≈${se.toFixed(3)}, SD=${sd.toFixed(3)}`);
  });
}
