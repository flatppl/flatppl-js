'use strict';

// Generic stochastic kernel-broadcast (spec §04): broadcast(Dist, …) →
// array-valued independent-product measure. v1 = distribution-
// constructor kernel, 1-D collection args + held-constant scalars,
// sampling (closed-form logdensity deferred, like matIid).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker');

function materialise(src, target, sampleCount) {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 4242 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m) => {
      const r = worker.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message)) : Promise.resolve(r);
    },
    sampleCount: sampleCount || 4000,
    rootSeed: 4242,
  };
  return ctx.getMeasure(target);
}

function colMean(m, K, j, N) {
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

test('broadcast: incompatible collection lengths is an error', async () => {
  await assert.rejects(
    materialise('p = [1.0, 2.0, 3.0]\n' +
                'q = [0.1, 0.2]\n' +
                'e ~ broadcast(Normal, p, q)\n', 'e', 100),
    /incompatible collection lengths/);
});
