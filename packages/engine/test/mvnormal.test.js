'use strict';

// Tests for MvNormal — Phase 6 of the shape-explicit refactor.
// MvNormal is the first multivariate distribution wired through the
// new Value-aware materialiser + density paths. Samples are atom-
// batched n-vectors (shape=[N, n]); density is closed-form via
// the Cholesky factor of cov.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker');

const SAMPLE_COUNT = 4096;
const ROOT_SEED    = 0xCAFEBEEF;

function makeCtx(source) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: SAMPLE_COUNT,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

function vecMean(samples, n) {
  // Per-component mean across atom axis. samples is atom-major
  // shape=[N, n]; returns length-n array.
  const N = samples.length / n;
  const out = new Float64Array(n);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < n; j++) out[j] += samples[i * n + j];
  }
  for (let j = 0; j < n; j++) out[j] /= N;
  return out;
}

function vecCov(samples, mean, n) {
  // Sample covariance matrix from atom-batched vector samples.
  const N = samples.length / n;
  const out = new Float64Array(n * n);
  for (let i = 0; i < N; i++) {
    for (let a = 0; a < n; a++) {
      const da = samples[i * n + a] - mean[a];
      for (let b = 0; b < n; b++) {
        const db = samples[i * n + b] - mean[b];
        out[a * n + b] += da * db;
      }
    }
  }
  for (let i = 0; i < n * n; i++) out[i] /= (N - 1);
  return out;
}

// =====================================================================
// matMvNormal — sampling correctness via empirical moments
// =====================================================================

test('MvNormal: classifier recognises and produces vector atoms', async () => {
  const ctx = makeCtx(`
mu = [1.0, 2.0]
sigma = [[1.0, 0.0], [0.0, 1.0]]
m = MvNormal(mu = mu, cov = sigma)
`);
  const m = await ctx.getMeasure('m');
  // shape=[N, 2] via the new Value-aware path
  assert.ok(m.value, 'matMvNormal must produce .value');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 2]);
  // dims legacy field also populated
  assert.deepEqual(m.dims, [2]);
});

test('MvNormal: empirical mean ≈ mu (identity cov)', async () => {
  const ctx = makeCtx(`
mu = [3.0, -2.0]
sigma = [[1.0, 0.0], [0.0, 1.0]]
m = MvNormal(mu = mu, cov = sigma)
`);
  const m = await ctx.getMeasure('m');
  const meanHat = vecMean(m.samples, 2);
  // MC error ~ sigma_i / sqrt(N) = 1/sqrt(4096) ≈ 0.016. Tolerate 5×.
  assert.ok(Math.abs(meanHat[0] - 3.0) < 0.08, 'mean[0] off: ' + meanHat[0]);
  assert.ok(Math.abs(meanHat[1] - (-2.0)) < 0.08, 'mean[1] off: ' + meanHat[1]);
});

test('MvNormal: empirical covariance ≈ cov (non-trivial off-diag)', async () => {
  // cov = [[2, 1], [1, 3]]. Should hold approximately.
  const ctx = makeCtx(`
mu = [0.0, 0.0]
sigma = [[2.0, 1.0], [1.0, 3.0]]
m = MvNormal(mu = mu, cov = sigma)
`);
  const m = await ctx.getMeasure('m');
  const meanHat = vecMean(m.samples, 2);
  const covHat = vecCov(m.samples, meanHat, 2);
  // MC error on covariance: ~ sqrt(2 * sigma_ii * sigma_jj / N).
  // For sigma_max ≈ 3: tol ≈ sqrt(2*3*3/4096) ≈ 0.04. Allow 6× for
  // CI flakes.
  assert.ok(Math.abs(covHat[0] - 2.0) < 0.25, 'cov[0,0] off: ' + covHat[0]);
  assert.ok(Math.abs(covHat[3] - 3.0) < 0.30, 'cov[1,1] off: ' + covHat[3]);
  assert.ok(Math.abs(covHat[1] - 1.0) < 0.25, 'cov[0,1] off: ' + covHat[1]);
  assert.ok(Math.abs(covHat[2] - 1.0) < 0.25, 'cov[1,0] off: ' + covHat[2]);
});

test('MvNormal: error on non-PD cov', async () => {
  // [[1, 2], [2, 1]] has eigenvalues 3, -1; not positive definite.
  const ctx = makeCtx(`
mu = [0.0, 0.0]
sigma = [[1.0, 2.0], [2.0, 1.0]]
m = MvNormal(mu = mu, cov = sigma)
`);
  await assert.rejects(ctx.getMeasure('m'), /MvNormal:.*positive definite/i);
});

test('MvNormal: dim mismatch error mu vs cov', async () => {
  const ctx = makeCtx(`
mu = [0.0, 0.0]
sigma = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
m = MvNormal(mu = mu, cov = sigma)
`);
  await assert.rejects(ctx.getMeasure('m'),
    /MvNormal.*must be 2x2|3x3/);
});

// =====================================================================
// walkMvNormal — closed-form density
// =====================================================================
//
// Analytical reference for the test cases below. For x ~ N_d(mu, Σ):
//
//   log p(x) = -½ d log(2π) - ½ log|Σ| - ½ (x-mu)ᵀ Σ⁻¹ (x-mu)
//
// Two-dimensional case with cov = [[a, b], [b, c]]:
//   |Σ| = a·c - b²
//   Σ⁻¹ = (1/|Σ|) [[c, -b], [-b, a]]
//   Mahalanobis = (1/|Σ|) (c·(x0-mu0)² - 2b·(x0-mu0)(x1-mu1) + a·(x1-mu1)²)

function mvnormalLogpdf2D(x, mu, cov) {
  const [a, b, c] = [cov[0][0], cov[0][1], cov[1][1]];
  const det = a * c - b * b;
  const d0 = x[0] - mu[0], d1 = x[1] - mu[1];
  const mahal = (c * d0 * d0 - 2 * b * d0 * d1 + a * d1 * d1) / det;
  return -Math.log(2 * Math.PI) - 0.5 * Math.log(det) - 0.5 * mahal;
}

test('walkMvNormal: density at observation point matches closed form (identity cov)', async () => {
  // x = [1, 2], mu = [0, 0], cov = I → logpdf = -log(2π) - ½ (1+4) = -log(2π) - 2.5
  const ctx = makeCtx(`
mu = [0.0, 0.0]
sigma = [[1.0, 0.0], [0.0, 1.0]]
m = MvNormal(mu = mu, cov = sigma)
lp = logdensityof(m, [1.0, 2.0])
`);
  const lp = await ctx.getMeasure('lp');
  const expected = -Math.log(2 * Math.PI) - 2.5;
  for (let i = 0; i < lp.samples.length; i++) {
    assert.ok(Math.abs(lp.samples[i] - expected) < 1e-10,
      'atom ' + i + ': got ' + lp.samples[i] + ', expected ' + expected);
  }
});

test('walkMvNormal: density at multiple points (correlated cov)', async () => {
  // cov = [[2, 1], [1, 3]] (det = 5).
  const cases = [
    [[1.0, 1.0], [0.0, 0.0]],
    [[0.5, -0.5], [1.0, 2.0]],
    [[3.0, 4.0], [-1.0, 0.0]],
  ];
  for (const [x, mu] of cases) {
    const ctx = makeCtx(`
mu = [${mu[0]}, ${mu[1]}]
sigma = [[2.0, 1.0], [1.0, 3.0]]
m = MvNormal(mu = mu, cov = sigma)
lp = logdensityof(m, [${x[0]}, ${x[1]}])
`);
    const lp = await ctx.getMeasure('lp');
    const expected = mvnormalLogpdf2D(x, mu, [[2, 1], [1, 3]]);
    assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
      'x=' + JSON.stringify(x) + ' mu=' + JSON.stringify(mu)
      + ': got ' + lp.samples[0] + ', expected ' + expected);
  }
});

test('walkMvNormal: 3-D identity case', async () => {
  // x = [1, 1, 1], mu = 0, cov = I → logpdf = -1.5 log(2π) - 1.5
  const ctx = makeCtx(`
mu = [0.0, 0.0, 0.0]
sigma = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
m = MvNormal(mu = mu, cov = sigma)
lp = logdensityof(m, [1.0, 1.0, 1.0])
`);
  const lp = await ctx.getMeasure('lp');
  const expected = -1.5 * Math.log(2 * Math.PI) - 1.5;
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    'got ' + lp.samples[0] + ', expected ' + expected);
});

test('walkMvNormal: density consistent with marginal Normals when cov is diagonal', async () => {
  // Independent components with diag cov [[a, 0], [0, c]]:
  // log p_MvN(x; mu, diag(a,c)) = log φ(x0; mu0, a) + log φ(x1; mu1, c)
  const a = 2.0, c = 5.0;
  const x = [1.5, -0.5], mu = [0.0, 1.0];
  const ctx = makeCtx(`
mu = [${mu[0]}, ${mu[1]}]
sigma = [[${a}, 0.0], [0.0, ${c}]]
m = MvNormal(mu = mu, cov = sigma)
lp = logdensityof(m, [${x[0]}, ${x[1]}])
`);
  const lp = await ctx.getMeasure('lp');
  // Sum of independent 1-D normal logpdfs:
  function logN(x, mu, var_) {
    return -0.5 * Math.log(2 * Math.PI * var_) - (x - mu) * (x - mu) / (2 * var_);
  }
  const expected = logN(x[0], mu[0], a) + logN(x[1], mu[1], c);
  assert.ok(Math.abs(lp.samples[0] - expected) < 1e-10,
    'got ' + lp.samples[0] + ', expected ' + expected);
});
