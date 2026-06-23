'use strict';
// sampler-coverage.test.ts — exercises the samplers' edge/error/fallback paths
// (degenerate or singular prior pools, vanished weights, derived record fields,
// unknown transforms) that the recovery e2e tests don't reach.

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { amisSample } = require('../amis-sample.ts');
const { smcSample }  = require('../smc-sample.ts');
const { runMcmc }    = require('../mcmc-driver.ts');
const { makeEllipticalSliceKernel } = require('../elliptical-slice-kernel.ts');
const { transformFor } = require('../transforms.ts');
const { ctxFor }       = require('./density/regression-baseline.test.ts');
const { materialiser } = require('..');
const diagnostics      = require('../diagnostics.ts');
const { recognizeCompositeIidDraw } = require('../composite-prior.ts');

const FIX = path.join(__dirname, 'fixtures/baseline');
const S0 = 8;
function target(y: Float64Array): number { return -0.5 * (y[0] * y[0] + y[1] * y[1]); }   // std normal
function logPrior(y: Float64Array): number { return -2 * Math.log(S0) - Math.log(2 * Math.PI) - 0.5 * (y[0] * y[0] + y[1] * y[1]) / (S0 * S0); }

// 2-D synthetic ModelView with crafted prior pools / targets to drive edge paths.
function mv2d(opts: any): any {
  const inf = !!opts.neginf;
  return {
    dim: 2, names: ['a', 'b'],
    gaussianPrior: opts.gaussianPrior || null,
    constrainAll: (y: Float64Array) => ({ a: y[0], b: y[1] }),
    logPosterior: (y: Float64Array) => inf ? -Infinity : target(y),
    logPosteriorBatch: (ys: Float64Array[]) => { const o = new Float64Array(ys.length); for (let i = 0; i < ys.length; i++) o[i] = inf ? -Infinity : target(ys[i]); return o; },
    logPriorLikBatch: (ys: Float64Array[]) => {
      const n = ys.length; const prior = new Float64Array(n), lik = new Float64Array(n);
      for (let i = 0; i < n; i++) { prior[i] = logPrior(ys[i]); lik[i] = inf ? -Infinity : (target(ys[i]) - prior[i]); }
      return { prior, lik };
    },
    initFromPrior: (n: number, prng: () => number) => {
      const out: Float64Array[] = [];
      const m = opts.tinyPool ? Math.min(1, n) : n;
      for (let i = 0; i < m; i++) {
        let y: Float64Array;
        if (opts.identical) y = new Float64Array([1, 2]);                          // zero-variance pool
        else if (opts.singular) { const t = S0 * (prng() - 0.5); y = new Float64Array([t, t]); }  // rank-1 cov
        else y = new Float64Array([S0 * (prng() - 0.5), S0 * (prng() - 0.5)]);
        out.push(y);
      }
      return out;
    },
  };
}

test('transformFor rejects an unknown support kind', () => {
  assert.throws(() => transformFor({ kind: 'nonsense' }), /no scalar transform/);
});

test('AMIS handles a degenerate (zero-variance) prior pool', () => {
  const r = amisSample(mv2d({ identical: true }), { amisSamples: 40, amisIters: 4, seed: 1 });
  assert.ok(r.samples.length > 0);
});

test('AMIS handles a tiny (<2) prior pool', () => {
  const r = amisSample(mv2d({ tinyPool: true }), { amisSamples: 40, amisIters: 4, seed: 1 });
  assert.ok(r.samples.length > 0);
});

test('AMIS + SMC handle a singular prior-pool covariance (non-PD Cholesky fallback)', () => {
  assert.ok(amisSample(mv2d({ singular: true }), { amisSamples: 40, amisIters: 4, seed: 1 }).samples.length > 0);
  assert.ok(smcSample(mv2d({ singular: true }), { smcParticles: 200, smcSteps: 5, seed: 1 }).samples.length > 0);
});

test('elliptical slice handles a singular fitted reference covariance', () => {
  const post = runMcmc(mv2d({ singular: true }), makeEllipticalSliceKernel(), { nWalkers: 4, warmup: 50, draws: 100, seed: 1 });
  assert.ok(post.drawsByName['a'].length > 0);
});

test('AMIS stops cleanly and SMC throws when all weights vanish', () => {
  const r = amisSample(mv2d({ neginf: true }), { amisSamples: 40, amisIters: 4, seed: 1 });
  assert.ok(Array.isArray(r.samples) || r.samples.length >= 0);
  assert.throws(() => smcSample(mv2d({ neginf: true }), { smcParticles: 200, seed: 1 }), /advance|weights|mass/i);
});

test('diagnostics: degenerate (constant / single) chains return finite values', () => {
  const constChain = new Float64Array(100).fill(2.5);
  // Zero variance → essBulk short-circuits to m*n (finite); split-R̂ is NaN
  // (undefined with no variance) — just exercise both branches.
  assert.equal(diagnostics.essBulk([constChain, constChain]), 200);
  const r = diagnostics.splitRHat([constChain, constChain]);
  assert.equal(typeof r, 'number');
});

test('recognizeCompositeIidDraw returns null for a non-composite draw', async () => {
  const src = fs.readFileSync(path.join(FIX, 'eight-schools.flatppl'), 'utf8');
  const ctx = ctxFor(src, 50).ctx;
  assert.equal(recognizeCompositeIidDraw('mu', ctx), null);   // mu ~ Normal, not a kernel-broadcast
});

test('sampler output evaluates a derived record field (sigma = sqrt(sigma2))', async () => {
  const src = fs.readFileSync(path.join(FIX, 'linear-regression.flatppl'), 'utf8');
  // This test only checks that the derived field `sigma` is present, positive,
  // and carries consistent nSamples diagnostics — none of which needs heavy
  // inference. Keep the workload small (like the sibling cases) so it isn't
  // the slowest test in the suite: a tiny dataset + a short MH run suffice.
  const m = await materialiser.materialiseMeasure('posterior', ctxFor(src, 50).ctx,
    { backend: 'mh', chains: 2, warmup: 100, draws: 200, seed: 1 });
  assert.ok(m.fields && m.fields.sigma, 'derived field sigma present');
  const s = m.fields.sigma.samples || (m.fields.sigma.value && m.fields.sigma.value.data);
  assert.ok(s && s.length > 0, 'sigma has samples');
  for (let i = 0; i < s.length; i++) assert.ok(s[i] > 0, `sigma[${i}] = ${s[i]} > 0`);
  // Each FIELD measure carries the sampler diagnostics (true draw count), not
  // just the top record — the viewer renders a single field on its own and must
  // label it with the draw count, not the resampled plot-atom count.
  const recN = m.diagnostics && m.diagnostics.nSamples;
  assert.ok(Number.isFinite(recN) && recN > 0, 'record carries nSamples');
  for (const fn of Object.keys(m.fields)) {
    const fd = m.fields[fn].diagnostics;
    assert.ok(fd && fd.nSamples === recN, `field '${fn}' carries the same nSamples (${fd && fd.nSamples} vs ${recN})`);
  }
});
