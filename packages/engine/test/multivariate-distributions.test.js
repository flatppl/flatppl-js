'use strict';

// Tests for Dirichlet + Multinomial — the two M-sized multivariate
// distributions added in the §08 fill-in. Same makeCtx scaffolding as
// mvnormal.test.js so the materialiser pipeline (classifier → mat*
// handler → atom-major Value storage) is exercised end-to-end.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

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

function atomMean(samples, K) {
  const N = samples.length / K;
  const out = new Float64Array(K);
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < K; k++) out[k] += samples[i * K + k];
  }
  for (let k = 0; k < K; k++) out[k] /= N;
  return out;
}

// =====================================================================
// Dirichlet
// =====================================================================

test('Dirichlet: classifier recognises and produces shape=[N, K] atoms', async () => {
  const ctx = makeCtx(`
alpha = [1.0, 1.0, 1.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  assert.ok(m.value, 'matDirichlet must produce .value');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3]);
  assert.deepEqual(m.dims, [3]);
});

test('Dirichlet: every atom sums to 1 (l1-normalized)', async () => {
  const ctx = makeCtx(`
alpha = [2.0, 3.0, 5.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const N = m.samples.length / K;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += m.samples[i * K + k];
    assert.ok(Math.abs(s - 1) < 1e-10,
      `atom ${i} sum = ${s} (expected 1 within 1e-10)`);
  }
});

test('Dirichlet: every coordinate is non-negative', async () => {
  const ctx = makeCtx(`
alpha = [1.0, 1.0, 1.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    assert.ok(m.samples[i] >= 0, `samples[${i}] = ${m.samples[i]} must be >= 0`);
  }
});

test('Dirichlet: empirical mean ≈ alpha / sum(alpha)', async () => {
  // Asymmetric Dirichlet — easier diagnostic than the symmetric case.
  const ctx = makeCtx(`
alpha = [1.0, 2.0, 3.0, 4.0]
m = Dirichlet(alpha = alpha)
`);
  const m = await ctx.getMeasure('m');
  const K = 4;
  const alphaSum = 1 + 2 + 3 + 4; // 10
  const expected = [0.1, 0.2, 0.3, 0.4];
  const observed = atomMean(m.samples, K);
  // Dirichlet variance per coord: a_k (a0 - a_k) / (a0² (a0 + 1)). For
  // a_k=4, a0=10 → var ≈ 4*6 / (100*11) ≈ 0.022; sd ≈ 0.15; MC error
  // ≈ 0.15 / sqrt(4096) ≈ 0.0023. Tolerate 6× for the largest coord.
  for (let k = 0; k < K; k++) {
    assert.ok(Math.abs(observed[k] - expected[k]) < 0.015,
      `coord ${k}: observed=${observed[k]} expected=${expected[k]}`);
  }
});

test('Dirichlet: rejects non-positive alpha', async () => {
  const ctx = makeCtx(`
alpha = [1.0, 0.0, 1.0]
m = Dirichlet(alpha = alpha)
`);
  await assert.rejects(ctx.getMeasure('m'), /alpha\[1\] = 0 must be positive/);
});

test('Dirichlet: rejects scalar alpha', async () => {
  const ctx = makeCtx(`
m = Dirichlet(alpha = 1.0)
`);
  await assert.rejects(ctx.getMeasure('m'), /alpha must be a vector/);
});

test('Dirichlet: positional alpha argument also accepted', async () => {
  // Spec §08 doesn't mandate kwarg-only; the classifier picks up either
  // form. Test the positional version for completeness.
  const ctx = makeCtx(`
alpha = [1.0, 1.0, 1.0]
m = Dirichlet(alpha)
`);
  const m = await ctx.getMeasure('m');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3]);
});

// =====================================================================
// Multinomial
// =====================================================================

test('Multinomial: classifier recognises and produces shape=[N, K] atoms', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.3, 0.2]
m = Multinomial(n = 10, p = p)
`);
  const m = await ctx.getMeasure('m');
  assert.ok(m.value, 'matMultinomial must produce .value');
  assert.deepEqual(m.value.shape, [SAMPLE_COUNT, 3]);
  assert.deepEqual(m.dims, [3]);
});

test('Multinomial: every atom is a length-K vector summing to n', async () => {
  const N_TRIALS = 12;
  const ctx = makeCtx(`
p = [0.25, 0.5, 0.25]
m = Multinomial(n = 12, p = p)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const N = m.samples.length / K;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = 0; k < K; k++) {
      const c = m.samples[i * K + k];
      assert.ok(Number.isInteger(c), `samples[${i},${k}] = ${c} must be integer`);
      assert.ok(c >= 0, `samples[${i},${k}] = ${c} must be >= 0`);
      s += c;
    }
    assert.equal(s, N_TRIALS, `atom ${i} sum = ${s} expected ${N_TRIALS}`);
  }
});

test('Multinomial: empirical mean ≈ n * p per coord', async () => {
  const n = 20;
  const ctx = makeCtx(`
p = [0.1, 0.4, 0.5]
m = Multinomial(n = 20, p = p)
`);
  const m = await ctx.getMeasure('m');
  const K = 3;
  const expected = [n * 0.1, n * 0.4, n * 0.5];
  const observed = atomMean(m.samples, K);
  // Multinomial variance per coord: n*p_k*(1-p_k). For p=0.5, n=20:
  // var=5; sd≈2.24. MC error ≈ 2.24/sqrt(4096) ≈ 0.035. Tolerate 6×.
  for (let k = 0; k < K; k++) {
    assert.ok(Math.abs(observed[k] - expected[k]) < 0.2,
      `coord ${k}: observed=${observed[k]} expected=${expected[k]}`);
  }
});

test('Multinomial: n=0 produces zero-count atoms', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.5]
m = Multinomial(n = 0, p = p)
`);
  const m = await ctx.getMeasure('m');
  for (let i = 0; i < m.samples.length; i++) {
    assert.equal(m.samples[i], 0, `samples[${i}] = ${m.samples[i]} expected 0`);
  }
});

test('Multinomial: rejects negative n', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.5]
m = Multinomial(n = -1, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /n must be a non-negative integer/);
});

test('Multinomial: rejects non-integer n', async () => {
  const ctx = makeCtx(`
p = [0.5, 0.5]
m = Multinomial(n = 1.5, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /n must be a non-negative integer/);
});

test('Multinomial: rejects negative p coord', async () => {
  const ctx = makeCtx(`
p = [0.5, -0.1, 0.6]
m = Multinomial(n = 10, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /p\[1\] = -0\.1 must be non-negative/);
});

test('Multinomial: rejects all-zero p', async () => {
  const ctx = makeCtx(`
p = [0.0, 0.0, 0.0]
m = Multinomial(n = 10, p = p)
`);
  await assert.rejects(ctx.getMeasure('m'), /p must sum to > 0/);
});
