'use strict';
// A `ksuperpose` mixture whose weight vector is a `Dirichlet` latent, sampled
// alongside a positive scalar. Two latents of different support kinds share one
// unconstrained vector, so this pins the block layout: the simplex latent takes
// K−1 coordinates and `sigma` takes one, for 3 unconstrained dimensions behind 4
// constrained output coordinates.
//
// Spec §08 "Dirichlet" puts the latent on `stdsimplex(3)`; §06 `ksuperpose` gives
// the mixture density as logsumexp over `log wᵢ + logdensityof(Normal(cᵢ, σ), y)`.
//
// ORACLE — an independent numpy/scipy quadrature, no engine code in the loop.
// Prior Dirichlet(2,2,2) × Gamma(shape 3, rate 4); likelihood the iid ksuperpose
// mixture over the 24 observations below. Integrated on a 320² × 400 midpoint
// grid over (w₁, w₂) ∈ stdsimplex(3) and σ ∈ [0.35, 2.35]; halving every axis
// moves each moment by under 2e-8, and the same integrand reproduces the
// testsuite's frozen log-density vector for corpora/coverage/spectral_lines at
// all five points to 7e-15.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { materialiser } = require('..');
const { ctxFor } = require('./_ctx-factory.ts');
const MV = require('../model-view.ts');

const MODEL = `
centers = [412.1, 415.7, 421.3]
y_obs = [
    416.9659, 412.0515, 421.551, 411.9125, 410.2642, 412.8685,
    410.7146, 421.7806, 420.9414, 411.5065, 420.6732, 420.4982,
    413.3433, 411.403, 412.5639, 411.3898, 412.7373, 421.1254,
    416.742, 421.9416, 412.5557, 415.002, 412.2671, 422.4918
]
N = lengthof(y_obs)
w ~ Dirichlet(alpha = [2.0, 2.0, 2.0])
sigma ~ Gamma(shape = 3.0, rate = 4.0)
line = normalize(ksuperpose(Normal, w)(mu = centers, sigma = sigma))
y ~ iid(line, N)
prior = lawof(record(w = w, sigma = sigma))
forward_kernel = kernelof(record(y = y), w = w, sigma = sigma)
L = likelihoodof(forward_kernel, record(y = y_obs))
posterior = bayesupdate(L, prior)
`;

const REF: Record<string, { mean: number; sd: number }> = {
  'w[0]':  { mean: 0.499805, sd: 0.090160 },
  'w[1]':  { mean: 0.166842, sd: 0.067448 },
  'w[2]':  { mean: 0.333353, sd: 0.084672 },
  sigma:   { mean: 0.847796, sd: 0.128144 },
};

const VIEWER: any = {
  chains: 4, walkers: null, warmup: 1000, draws: 4000, seed: 1,
  amisIters: 30, amisSamples: 300,
  smcParticles: 2000, smcSteps: 12, smcCESS: 0.7,
  nLive: 400, dlogz: 0.5, regionMetric: 'off',
};

// Per-coordinate draws under the viewer's coordinate names.
function coordsOf(m: any): Record<string, Float64Array> {
  const out: Record<string, Float64Array> = {};
  for (const nm of Object.keys(m.fields)) {
    const f = m.fields[nm];
    const data: Float64Array = f.samples || (f.value && f.value.data);
    const sh = (f.value && Array.isArray(f.value.shape)) ? f.value.shape : null;
    if (sh && sh.length === 2 && sh[1] > 1) {
      const N = sh[0], D = sh[1];
      for (let j = 0; j < D; j++) {
        const c = new Float64Array(N);
        for (let a = 0; a < N; a++) c[a] = data[a * D + j];
        out[`${nm}[${j}]`] = c;
      }
    } else {
      out[nm] = data;
    }
  }
  return out;
}

function meanSd(a: Float64Array) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  const mu = s / a.length;
  let v = 0;
  for (let i = 0; i < a.length; i++) v += (a[i] - mu) ** 2;
  return { mean: mu, sd: Math.sqrt(v / (a.length - 1)) };
}

test('mixed simplex + positive latents give 3 unconstrained dimensions over 4 output coordinates', async () => {
  const { ctx } = ctxFor(MODEL, 200);
  const mv = await MV.buildModelViewFromCtx(ctx, ctx.derivations['posterior']);
  assert.equal(mv.dim, 3, 'stdsimplex(3) contributes 2, sigma contributes 1');
  assert.deepEqual(mv.names, ['w[0]', 'w[1]', 'w[2]', 'sigma']);
  // The block offsets must not collide: perturbing sigma's coordinate alone
  // must leave w untouched, and vice versa.
  const y0 = new Float64Array([0.1, -0.2, 0.3]);
  const a = mv.constrainAll(y0);
  const y1 = Float64Array.from(y0); y1[2] += 1;
  const b = mv.constrainAll(y1);
  for (let k = 0; k < 3; k++) {
    assert.equal(a[`w[${k}]`], b[`w[${k}]`], `moving sigma changed w[${k}]`);
  }
  assert.ok(b.sigma > a.sigma, 'sigma should follow its own coordinate');
  const y2 = Float64Array.from(y0); y2[0] += 1;
  assert.equal(mv.constrainAll(y2).sigma, a.sigma, 'moving w changed sigma');
});

// Absolute margins on the weights (which live in [0,1]) and relative on sigma.
// The loosest observed per-sampler deviation is well inside these; the slack
// covers the seed-to-seed spread of the low-ESS backends.
const W_MEAN_TOL = 0.03;
const SD_REL_TOL = 0.20;
const SIGMA_MEAN_REL_TOL = 0.05;

// `slice` mixes best here (bulk ESS ~14.7k of 16k draws) but costs ~27 s at the
// viewer's budget, so it runs on a shorter chain. 500 draws × 4 chains still
// leaves ESS in the low thousands, far more than these margins need.
const SHORT: Record<string, any> = { slice: { warmup: 200, draws: 500 } };

for (const backend of ['mh', 'ram', 'slice', 'emcee', 'demcz', 'amis', 'smc', 'nested',
  'elliptical-slice-sampler']) {
  test(`backend:${backend} recovers the spectral-lines posterior moments (scipy quadrature)`, async () => {
    const { ctx } = ctxFor(MODEL, 4000);
    const m = await materialiser.materialiseMeasure('posterior', ctx,
      Object.assign({}, VIEWER, { backend }, SHORT[backend] || {}));
    const cs = coordsOf(m);
    for (const nm of Object.keys(REF)) {
      assert.ok(cs[nm], `missing coordinate '${nm}'; got ${Object.keys(cs).join(',')}`);
      const { mean, sd } = meanSd(cs[nm]);
      assert.ok(Number.isFinite(mean) && Number.isFinite(sd), `${nm} non-finite: ${mean} / ${sd}`);
      const r = REF[nm];
      const tol = nm === 'sigma' ? r.mean * SIGMA_MEAN_REL_TOL : W_MEAN_TOL;
      assert.ok(Math.abs(mean - r.mean) < tol,
        `E[${nm}] ${mean.toFixed(5)} vs reference ${r.mean.toFixed(5)} (tol ${tol.toFixed(5)})`);
      assert.ok(Math.abs(sd / r.sd - 1) < SD_REL_TOL,
        `sd[${nm}] ${sd.toFixed(5)} vs reference ${r.sd.toFixed(5)}`);
    }
    // Every draw stays on the simplex, and sigma stays positive.
    let sumMean = 0;
    for (let k = 0; k < 3; k++) {
      const c = cs[`w[${k}]`];
      let lo = Infinity, hi = -Infinity;
      for (const v of c) { if (v < lo) lo = v; if (v > hi) hi = v; }
      assert.ok(lo >= 0 && hi <= 1, `w[${k}] left [0,1]: [${lo}, ${hi}]`);
      sumMean += meanSd(c).mean;
    }
    assert.ok(Math.abs(sumMean - 1) < 1e-9, `Σ E[wᵢ] = ${sumMean}, want 1`);
    let sLo = Infinity;
    for (const v of cs.sigma) if (v < sLo) sLo = v;
    assert.ok(sLo > 0, `sigma reached ${sLo}`);
  });
}
