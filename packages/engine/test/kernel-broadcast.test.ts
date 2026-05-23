'use strict';

// Generic stochastic kernel-broadcast (spec §04): broadcast(Dist, …) →
// array-valued independent-product measure. v1 = distribution-
// constructor kernel, 1-D collection args + held-constant scalars,
// sampling (closed-form logdensity deferred, like matIid).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function materialise(src: any, target: any, sampleCount: any) {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 4242 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: any) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: sampleCount || 4000,
    rootSeed: 4242,
  };
  return ctx.getMeasure(target);
}

function colMean(m: any, K: any, j: any, N: any) {
  let s = 0;
  for (let i = 0; i < N; i++) s += m.samples[i * K + j];
  return s / N;
}

test('broadcast(Normal, means, sigmas) → [N,K] array measure', async () => {
  const N = 4000;
  const m = await materialise(
    'means = [0.0, 10.0, 100.0]\n' +
    'sigmas = [0.001, 0.001, 0.001]\n' +
    'x ~ broadcast(Normal, means, sigmas)\n', 'x', N);
  assert.equal(m.shape, 'array');
  assert.deepEqual(m.dims, [3]);
  assert.equal(m.samples.length, N * 3);
  assert.equal(m.logTotalmass, 0, 'product of probability measures');
  assert.equal(m.n_eff, N);
  assert.ok(m.value && m.value.shape[0] === N && m.value.shape[1] === 3);
  assert.ok(Math.abs(colMean(m, 3, 0, N) - 0) < 0.05);
  assert.ok(Math.abs(colMean(m, 3, 1, N) - 10) < 0.05);
  assert.ok(Math.abs(colMean(m, 3, 2, N) - 100) < 0.05);
});

test('broadcast: independence — per-element params, not shared', async () => {
  // Distinct, tiny sigmas; each column tracks its own mean tightly.
  const N = 4000;
  const m = await materialise(
    'mu = [-5.0, 5.0]\n' +
    's  = [0.002, 0.002]\n' +
    'v ~ broadcast(Normal, mu, s)\n', 'v', N);
  assert.ok(colMean(m, 2, 0, N) < -4.9 && colMean(m, 2, 1, N) > 4.9);
});

test('broadcast: kwargs form binds by parameter name', async () => {
  const N = 3000;
  const m = await materialise(
    'a = [1.0, 2.0]\n' +
    'b = [0.001, 0.001]\n' +
    'y ~ broadcast(Normal, mu = a, sigma = b)\n', 'y', N);
  assert.deepEqual(m.dims, [2]);
  assert.ok(Math.abs(colMean(m, 2, 0, N) - 1) < 0.02);
  assert.ok(Math.abs(colMean(m, 2, 1, N) - 2) < 0.02);
});

test('broadcast: held-constant scalar arg is not iterated', async () => {
  const N = 3000;
  const m = await materialise(
    'mm = [5.0, 50.0, 500.0]\n' +
    'w ~ broadcast(Normal, mm, 0.001)\n', 'w', N);
  assert.deepEqual(m.dims, [3]);
  assert.ok(Math.abs(colMean(m, 3, 2, N) - 500) < 0.05);
});

test('broadcast(Poisson, rates) — discrete kernel', async () => {
  const N = 5000;
  const m = await materialise(
    'rates = [1.0, 20.0]\n' +
    'c ~ broadcast(Poisson, rates)\n', 'c', N);
  assert.deepEqual(m.dims, [2]);
  assert.ok(Math.abs(colMean(m, 2, 0, N) - 1) < 0.15);
  assert.ok(Math.abs(colMean(m, 2, 1, N) - 20) < 0.6);
});

test('deterministic broadcast(f, …) is NOT classified as a kernel', async () => {
  // Value-broadcast must stay a plain value binding (array data),
  // not an array-valued measure.
  const m = await materialise(
    'A = [1.0, 2.0, 3.0]\n' +
    'B = broadcast(fn(_ * 2.0), A)\n', 'B', 100);
  assert.notEqual(m.shape, 'array');
  assert.deepEqual(Array.from(m.samples).slice(0, 3), [2, 4, 6]);
});

test('deterministic boolean-array broadcast is a value, not null elems', async () => {
  // Regression: a deterministic broadcast producing a boolean array
  // (elementwise comparison, incl. dotted `.<`) was mis-surfaced by
  // fixedValueToMeasure as a 3-element tuple of nulls because the
  // flat-vector check rejected booleans. Booleans coerce to 1/0,
  // exactly like a scalar `true` binding already did.
  const m = await materialise(
    'A = [1.0, 2.0, 3.0]\n' +
    'B = [3.0, 2.0, 9.0]\n' +
    'C = A .< B\n', 'C', 100);
  assert.notEqual(m.shape, 'array');
  assert.ok(m.samples, 'expected flat samples, not elems');
  assert.deepEqual(Array.from(m.samples).slice(0, 3), [1, 0, 1]);
  // explicit broadcast(fn) form must agree (dotted is pure sugar)
  const e = await materialise(
    'A = [1.0, 2.0, 3.0]\n' +
    'B = [3.0, 2.0, 9.0]\n' +
    'C = broadcast(fn(_ < _), A, B)\n', 'C', 100);
  assert.deepEqual(Array.from(e.samples).slice(0, 3), [1, 0, 1]);
});

test('broadcast(Normal,…) closed-form: per-element var + independence', async () => {
  // Exercises the MvNormal(mu, diag(sigma²)) specialization (diag
  // lower_cholesky=diag(σ), diag mulN). Each column j ~ N(mu_j,
  // sigma_j²) independently: sample variance ≈ sigma_j², and the
  // cross-column covariance ≈ 0 (diagonal covariance).
  const N = 20000, K = 3;
  const m = await materialise(
    'mu = [0.0, 0.0, 0.0]\n' +
    'sg = [1.0, 3.0, 0.5]\n' +
    'r ~ broadcast(Normal, mu, sg)\n', 'r', N);
  const mean = [0, 0, 0], col = [[], [], []];
  for (let i = 0; i < N; i++) for (let j = 0; j < K; j++) {
    const x = m.samples[i * K + j]; col[j].push(x); mean[j] += x / N;
  }
  const varj = [0, 0, 0];
  for (let j = 0; j < K; j++) {
    for (let i = 0; i < N; i++) varj[j] += (col[j][i] - mean[j]) ** 2 / N;
  }
  assert.ok(Math.abs(varj[0] - 1.0) < 0.1, 'var col0 ≈ 1');
  assert.ok(Math.abs(varj[1] - 9.0) < 0.6, 'var col1 ≈ 9');
  assert.ok(Math.abs(varj[2] - 0.25) < 0.05, 'var col2 ≈ 0.25');
  // cov(col0, col1) ≈ 0 (independent columns)
  let cov01 = 0;
  for (let i = 0; i < N; i++) cov01 += (col[0][i] - mean[0]) * (col[1][i] - mean[1]) / N;
  assert.ok(Math.abs(cov01) < 0.15, 'columns independent (cov≈0), got ' + cov01);
});

test('kernel-broadcast: distribution params are general deterministic exprs', async () => {
  // Regression: a kernel-broadcast whose distribution parameters are
  // themselves composite fixed-phase expressions (nested broadcasts,
  // arithmetic) — the pyhf-uncorrelated_background prior pattern
  // `gamma = draw(Gamma.(tau .+ 1.0, tau))` with
  // `tau = (bkg ./ dbkg) .^ 2`. Previously matKernelBroadcast resolved
  // param IRs through the minimal static resolveIRToValue, which threw
  // "unsupported op 'broadcast'". The resolver now delegates general
  // exprs to the real deterministic evaluator. Gamma(shape=t+1, rate=t)
  // has mean (t+1)/t = 1 + 1/t; tau = [50/3, 52/7]^2 = [277.78, 55.18].
  const N = 12000;
  const m = await materialise(
    'bkg  = [50.0, 52.0]\n' +
    'dbkg = [3.0, 7.0]\n' +
    'tau  = (bkg ./ dbkg) .^ 2\n' +
    'gamma ~ Gamma.(tau .+ 1.0, tau)\n', 'gamma', N);
  assert.equal(m.shape, 'array');
  assert.deepEqual(m.dims, [2]);
  assert.equal(m.logTotalmass, 0, 'product of probability measures');
  assert.ok(Math.abs(colMean(m, 2, 0, N) - (1 + 1 / (2500 / 9))) < 0.01);
  assert.ok(Math.abs(colMean(m, 2, 1, N) - (1 + 1 / (2704 / 49))) < 0.02);
});

test('broadcast: incompatible collection lengths is an error', async () => {
  await assert.rejects(
    materialise('p = [1.0, 2.0, 3.0]\n' +
                'q = [0.1, 0.2]\n' +
                'e ~ broadcast(Normal, p, q)\n', 'e', 100),
    /incompatible collection lengths/);
});
